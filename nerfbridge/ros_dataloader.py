# Code adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/data/utils/dataloaders.py

"""
Defines the ROSDataloader object that subscribes to pose and images topics,
and populates an image tensor and Cameras object with values from these topics.
Image and pose pairs are added at a prescribed frequency and intermediary images
are discarded (could be used for evaluation down the line).
"""
import time
import threading
import warnings
from typing import Union

from rich.console import Console
import torch
from torch.utils.data.dataloader import DataLoader

import nerfbridge.pose_utils as pose_utils
from nerfbridge.ros_dataset import ROSDataset, ROSDepthDataset

import rclpy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from cv_bridge import CvBridge

import pdb

CONSOLE = Console(width=120)

# Suppress a warning from torch.tensorbuffer about copying that
# does not apply in this case.
warnings.filterwarnings("ignore", "The given buffer")


class ROSDataloader(DataLoader):
    """
    Creates batches of the dataset return type. In this case of nerfstudio this means
    that we are returning batches of full images, which then are sampled using a
    PixelSampler. For this class the image batches are progressively growing as
    more images are recieved from ROS, and stored in a pytorch tensor.

    Args:
        dataset: Dataset to sample from.
        publish_posearray: publish a PoseArray to a ROS topic that tracks the poses of the
            images that have been added to the training set.
        data_update_freq: Frequency (wall clock) that images are added to the training
            data tensors. If this value is less than the frequency of the topics to which
            this dataloader subscribes (pose and images) then this subsamples the ROS data.
            Otherwise, if the value is larger than the ROS topic rates then every pair of
            messages is added to the training bag.
        slam_method: string that determines which type of pose topic to subscribe to and
            what coordinate transforms to use when handling the poses. Currently, only
            "cuvslam" is supported.
        topic_sync: use "exact" when your slam algorithm matches pose and image time stamps,
            and use "approx" when it does not.
        topic_slop: if using approximate time synchronization, then this float determines
            the slop in seconds that is allowable between images and poses.
        device: Device to perform computation.
    """

    dataset: ROSDataset

    def __init__(
        self,
        dataset: ROSDataset,
        data_update_freq: float,
        slam_method: str,
        topic_sync: str,
        topic_slop: float,
        use_compressed_rgb: bool,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        # This is mostly a parameter placeholder, and manages the cameras
        self.dataset = dataset

        # Image meta data
        self.device = device
        self.num_images = len(self.dataset)
        self.H = self.dataset.image_height
        self.W = self.dataset.image_width
        self.n_channels = 3

        # Tracking ros updates
        self.current_idx = 0
        self.updated = True
        self.update_period = 1 / data_update_freq
        self.last_update_t = time.perf_counter()
        self.poselist = []

        # Keep it in the format so that it makes it look more like a
        # regular data loader.
        self.data_dict = {
            "image": self.dataset.image_tensor,
            "image_idx": self.dataset.image_indices,
        }

        # Flag for depth training, add depth tensor to data.
        self.listen_depth = False
        if isinstance(self.dataset, ROSDepthDataset):
            self.data_dict["depth_image"] = self.dataset.depth_tensor
            self.listen_depth = True

        super().__init__(dataset=dataset, **kwargs)

        self.bridge = CvBridge()
        self.slam_method = slam_method
        self.use_compressed_rgb = use_compressed_rgb

        # Initializing ROS2
        rclpy.init()
        self.node = rclpy.create_node("nerf_bridge_node")

        # Setting up ROS2 message_filter TimeSynchronzier
        self.subs = []
        if self.use_compressed_rgb:
            self.subs.append(
                Subscriber(
                    self.node,
                    CompressedImage,
                    self.dataset.image_topic_name,
                )
            )
        else:
            self.subs.append(
                Subscriber(
                    self.node,
                    Image,
                    self.dataset.image_topic_name,
                )
            )

        if slam_method == "cuvslam":
            self.subs.append(
                Subscriber(self.node, Odometry, self.dataset.pose_topic_name)
            )
        elif slam_method == "orbslam3":
            self.subs.append(
                Subscriber(self.node, PoseStamped, self.dataset.pose_topic_name)
            )
        elif slam_method == "mocap":
            self.subs.append(
                Subscriber(self.node, PoseStamped, self.dataset.pose_topic_name)
            )
        else:
            raise NameError(
                "Unsupported SLAM algorithm. Must be one of {cuvslam, orbslam3}"
            )

        if self.listen_depth:
            self.subs.append(
                Subscriber(self.node, Image, self.dataset.depth_topic_name)
            )

        if topic_sync == "approx":
            self.ts = ApproximateTimeSynchronizer(self.subs, 10, topic_slop)
        elif topic_sync == "exact":
            self.ts = TimeSynchronizer(self.subs, 10)
        else:
            raise NameError(
                "Unsupported topic sync method. Must be one of {approx, exact}."
            )

        self.ts.registerCallback(self.ts_callback)

        # Start a thread for processing the callbacks
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True
        )
        self.ros_thread.start()

    def msg_status(self, num_to_start):
        """
        Check if any image-pose pairs have been successfully streamed from
        ROS, and return True if so.
        """
        return self.current_idx >= (num_to_start - 1)

    def ts_callback(self, *args):
        now = time.perf_counter()
        if (
            now - self.last_update_t > self.update_period
            and self.current_idx < self.num_images
        ):
            # Process RGB and Pose
            self.image_callback(args[0])
            self.pose_callback(args[1])

            # Process Depth if using depth training
            if self.listen_depth:
                self.depth_callback(args[2])

            self.dataset.updated_indices.append(self.current_idx)
            self.updated = True
            self.current_idx += 1
            self.last_update_t = now

    def image_callback(self, image: Image | CompressedImage):
        """
        Callback for processing RGB Image Messages, and adding them to the
        dataset for training.
        """
        # Load the image message directly into the torch
        if self.use_compressed_rgb:
            im_cv = self.bridge.compressed_imgmsg_to_cv2(image)
        else:
            im_cv = self.bridge.imgmsg_to_cv2(image, image.encoding)

        im_tensor = torch.from_numpy(im_cv).to(dtype=torch.float32) / 255.0

        # COPY the image data into the data tensor
        self.dataset.image_tensor[self.current_idx] = im_tensor

    def pose_callback(self, pose: PoseStamped | Odometry):
        """
        Callback for Pose messages. Extracts pose, converts it to Nerfstudio coordinate
        convention, and inserts it into the Cameras object.
        """
        if self.slam_method == "cuvslam":
            # Odometry Message
            hom_pose = pose_utils.ros_pose_to_homogenous(pose.pose)
            c2w = pose_utils.cuvslam_to_nerfstudio(hom_pose)
        elif self.slam_method == "orbslam3":
            # PoseStamped Message
            hom_pose = pose_utils.ros_pose_to_homogenous(pose)
            c2w = pose_utils.orbslam3_to_nerfstudio(hom_pose)
        elif self.slam_method == "mocap":
            # PoseStamped Message
            hom_pose = pose_utils.ros_pose_to_homogenous(pose)
            c2w = pose_utils.mocap_to_nerfstudio(hom_pose)
        else:
            raise NameError("Unknown SLAM Method!")

        # Scale Pose to
        c2w[:3, 3] *= self.dataset.scale_factor

        # Insert in Cameras
        device = self.dataset.cameras.device
        self.dataset.cameras.camera_to_worlds[self.current_idx] = c2w.to(device)

    def depth_callback(self, depth: Image):
        """
        Callback for processing Depth Image messages. Similar to RGB image handling,
        but also rescales the depth to the appropriate value.
        """
        depth_cv = self.bridge.imgmsg_to_cv2(depth, depth.encoding)
        depth_tensor = torch.from_numpy(depth_cv.astype("float32")).to(
            dtype=torch.float32
        )

        aggregate_scale = self.dataset.scale_factor * self.dataset.depth_scale_factor

        self.dataset.depth_tensor[self.current_idx] = (
            depth_tensor.unsqueeze(-1) * aggregate_scale
        )

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_updated_batch(self):
        batch = {}
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[: self.current_idx, ...]
        return batch

    def __iter__(self):
        while True:
            if self.updated:
                self.batch = self._get_updated_batch()
                self.updated = False

            batch = self.batch
            yield batch
