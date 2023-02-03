# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Defines the ROSDataloader object that subscribes to pose and images topics,
and populates an image tensor and Cameras object with values from these topics.
Image and pose pairs are added at a prescribed frequency and intermediary images
are discarded (could be used for evaluation down the line).
"""
import time
from typing import Union
import sys
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
from geometry_msgs.msg import PoseStamped
from message_filters import TimeSynchronizer, Subscriber

import pdb


CONSOLE = Console(width=120)


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
    Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the dataset return type. In this case of nerfstudio this means that we are
    returning batches of full images, which then are sampled using a PixelSampler.

    Args:
        dataset: Dataset to sample from.
        num_images_to_start: How many images to sample rays for each batch.
        num_images_to_add: How often to collate new images.
        full_dataset: Ignore other settings and return the entire dataset.
        device: Device to perform computation.
    """

    dataset: ROSDataset

    def __init__(
        self,
        dataset: ROSDataset,
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
        self.update_period = 1 / self.dataset.update_freq
        self.last_update_t = time.perf_counter()

        self.coord_st = torch.zeros(3, 4)
        R1 = transform.Rotation.from_euler("y", -90, degrees=True).as_matrix()
        R2 = transform.Rotation.from_euler("x", 90, degrees=True).as_matrix()
        R = torch.from_numpy(R2 @ R1)
        self.coord_st[:, :3] = R

        # Keep it in the format so that it makes it look more like a
        # regular data loader.
        self.collated_data = {
            "image": self.dataset.image_tensor,
            "image_idx": self.dataset.image_indices,
        }

        super().__init__(dataset=dataset, **kwargs)

        rospy.init_node("nerfstudiolistener", anonymous=True)
        self.image_sub = Subscriber(self.dataset.image_topic_name, Image)
        self.pose_sub = Subscriber(self.dataset.pose_topic_name, PoseStamped)
        self.ts = TimeSynchronizer([self.image_sub, self.pose_sub], 10)
        self.ts.registerCallback(self.ts_image_pose_callback)

        # Subscribe to both topics and match messages if they have
        # appropriately close times in the header. Time "slop" is
        # controlled by data_dt_thresh (which is in seconds).
        # CONSOLE.print("here")

    def msg_status(self, num_to_start):
        """
        Check if any image-pose pairs have been successfully streamed from
        ROS, and return True if so.
        """
        return self.current_idx >= (num_to_start - 1)

    def ts_image_pose_callback(self, image: Image, pose: PoseStamped):
        """
        Check if its time to update the tensors
        Convert Image ROS message to Tensor.
        Convert Pose to appropriate orientation.
        Write image to self.images
        Write pose to dataset.cameras.camera_to_world
        increment current index
        change update status

        Increment and set update status at the end so that to avoid accidentally
        reading out of a section of images tensor that hasn't been written to yet.
        We'll see if this is an issue.
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

            self.dataset.updated_indices.append(self.current_idx)

            self.updated = True
            self.current_idx += 1
            self.last_update_t = now

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_collated_subset(self):
        collated_batch = {}
        for k, v in self.collated_data.items():
            if isinstance(v, torch.Tensor):
                collated_batch[k] = v[: self.current_idx, ...]
        return collated_batch

    def __iter__(self):
        while True:
            if self.updated:
                self.collated_batch = self._get_collated_subset()
                self.updated = False

            collated_batch = self.collated_batch
            yield collated_batch
