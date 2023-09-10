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
from nerfbridge.ros_dataset import ROSDataset

import rclpy
from sensor_msgs.msg import Image
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

        super().__init__(dataset=dataset, **kwargs)

        self.bridge = CvBridge()
        self.slam_method = slam_method

        # Initializing ROS2
        rclpy.init()
        self.node = rclpy.create_node("nerf_bridge_node")

        # Setting up ROS2 message_filter TimeSynchronzier
        self.image_sub = Subscriber(
            self.node,
            Image,
            self.dataset.image_topic_name,
        )

        if slam_method == "cuvslam":
            self.pose_sub = Subscriber(
                self.node, Odometry, self.dataset.pose_topic_name
            )
        elif slam_method == "orbslam3":
            self.pose_sub = Subscriber(
                self.node, PoseStamped, self.dataset.pose_topic_name
            )
        else:
            raise NameError("Unsupported SLAM algorithm!")

        if topic_sync == "approx":
            self.ts = ApproximateTimeSynchronizer(
                [self.image_sub, self.pose_sub], 10, topic_slop
            )
        if topic_sync == "exact":
            self.ts = TimeSynchronizer([self.image_sub, self.pose_sub], 10)

        self.ts.registerCallback(self.ts_image_pose_callback)

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

    def ts_image_pose_callback(self, image: Image, pose: PoseStamped | Odometry):
        """
        The callback triggered when time synchronized image and pose messages
        are published on the topics specifed in the config JSON passed to
        the ROSDataParser.
        """
        now = time.perf_counter()
        if (
            now - self.last_update_t > self.update_period
            and self.current_idx < self.num_images
        ):
            # ----------------- Handling the IMAGE ----------------
            # Load the image message directly into the torch
            im_cv = self.bridge.imgmsg_to_cv2(image, image.encoding)
            im_tensor = torch.from_numpy(im_cv).to(dtype=torch.float32) / 255.0
            # Convert BGR -> RGB (this adds an extra copy, and might be able to
            # skip if we do something fancy with the reshape above)
            im_tensor = im_tensor.flip([-1])

            # COPY the image data into the data tensor
            self.dataset.image_tensor[self.current_idx] = im_tensor

            # ----------------- Handling the POSE ----------------
            if self.slam_method == "cuvslam":
                # Odometry Message
                hom_pose = pose_utils.ros_pose_to_homogenous(pose.pose)
                c2w = pose_utils.cuvslam_to_nerfstudio(hom_pose)
            elif self.slam_method == "orbslam3":
                # PoseStamped Message
                hom_pose = pose_utils.ros_pose_to_homogenous(pose)
                c2w = pose_utils.orbslam3_to_nerfstudio(hom_pose)
            else:
                raise NameError("Unknown SLAM Method!")

            device = self.dataset.cameras.device
            self.dataset.cameras.camera_to_worlds[self.current_idx] = c2w.to(device)

            self.dataset.updated_indices.append(self.current_idx)

            self.updated = True
            self.current_idx += 1
            self.last_update_t = now

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
