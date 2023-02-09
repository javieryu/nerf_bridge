# Code adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/data/utils/dataloaders.py

"""
Defines the ROSDataloader object that subscribes to pose and images topics,
and populates an image tensor and Cameras object with values from these topics.
Image and pose pairs are added at a prescribed frequency and intermediary images
are discarded (could be used for evaluation down the line).
"""
import time
import warnings
from typing import Union

import numpy as np
import scipy.spatial.transform as transform
from rich.console import Console
import torch
from torch.utils.data.dataloader import DataLoader

from nerfstudio.process_data.colmap_utils import qvec2rotmat
from nerfstudio.utils.poses import multiply

from nsros.ros_dataset import ROSDataset

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray
from message_filters import TimeSynchronizer, Subscriber


CONSOLE = Console(width=120)

# Suppress a warning from torch.tensorbuffer about copying that
# does not apply in this case.
warnings.filterwarnings("ignore", "The given buffer")


def ros_pose_to_nerfstudio(pose: PoseStamped, static_transform=None):
    """
    Takes a ROS Pose message and converts it to the
    3x4 transform format used by nerfstudio.
    """
    pose_msg = pose.pose
    quat = np.array(
        [
            pose_msg.orientation.w,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
        ],
    )
    posi = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    R = torch.tensor(qvec2rotmat(quat))
    T = torch.cat([R, posi.unsqueeze(-1)], dim=-1)
    T = T.to(dtype=torch.float32)
    if static_transform is not None:
        T = multiply(T, static_transform)
    return T.to(dtype=torch.float32)


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
        device: Device to perform computation.
    """

    dataset: ROSDataset

    def __init__(
        self,
        dataset: ROSDataset,
        publish_posearray: bool,
        data_update_freq: float,
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
        self.publish_posearray = publish_posearray
        self.poselist = []

        self.coord_st = torch.zeros(3, 4)
        R1 = transform.Rotation.from_euler("y", -90, degrees=True).as_matrix()
        R2 = transform.Rotation.from_euler("x", 90, degrees=True).as_matrix()
        R = torch.from_numpy(R2 @ R1)
        self.coord_st[:, :3] = R

        # Keep it in the format so that it makes it look more like a
        # regular data loader.
        self.data_dict = {
            "image": self.dataset.image_tensor,
            "image_idx": self.dataset.image_indices,
        }

        super().__init__(dataset=dataset, **kwargs)

        # All of the ROS CODE
        rospy.init_node("nsros_dataloader", anonymous=True)
        self.image_sub = Subscriber(self.dataset.image_topic_name, Image)
        self.pose_sub = Subscriber(self.dataset.pose_topic_name, PoseStamped)
        self.ts = TimeSynchronizer([self.image_sub, self.pose_sub], 10)
        self.ts.registerCallback(self.ts_image_pose_callback)
        self.posearray_pub = rospy.Publisher("training_poses", PoseArray, queue_size=1)

    def msg_status(self, num_to_start):
        """
        Check if any image-pose pairs have been successfully streamed from
        ROS, and return True if so.
        """
        return self.current_idx >= (num_to_start - 1)

    def ts_image_pose_callback(self, image: Image, pose: PoseStamped):
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
            im_tensor = torch.frombuffer(image.data, dtype=torch.uint8).reshape(
                self.H, self.W, -1
            )
            im_tensor = im_tensor.to(dtype=torch.float32) / 255.0
            # Convert BGR -> RGB (this adds an extra copy, and might be able to
            # skip if we do something fancy with the reshape above)
            im_tensor = im_tensor.flip([-1])

            # COPY the image data into the data tensor
            self.dataset.image_tensor[self.current_idx] = im_tensor

            # ----------------- Handling the POSE ----------------
            c2w = ros_pose_to_nerfstudio(pose, static_transform=self.coord_st)
            device = self.dataset.cameras.device
            c2w = c2w.to(device)
            self.dataset.cameras.camera_to_worlds[self.current_idx] = c2w

            if self.publish_posearray:
                self.poselist.append(pose.pose)
                pa = PoseArray(poses=self.poselist)
                pa.header.frame_id = "map"
                self.posearray_pub.publish(pa)

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
