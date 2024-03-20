from typing import Union

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class ROSDataset(InputDataset):
    """
    This is a tensor dataset that keeps track of all of the data streamed by ROS.
    It's main purpose is to conform to the already defined workflow of nerfstudio:
        (dataparser -> inputdataset -> dataloader).

    In reality we could just store everything directly in ROSDataloader, but this
    would require rewritting more code than its worth.

    Images are tracked in self.image_tensor with uninitialized images set to
    all white (hence torch.ones).
    Poses are stored in self.cameras.camera_to_worlds as 3x4 transformation tensors.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "image_topic" in dataparser_outputs.metadata.keys()
            and "pose_topic" in dataparser_outputs.metadata.keys()
            and "num_images" in dataparser_outputs.metadata.keys()
        )
        self.image_topic_name = self.metadata["image_topic"]
        self.pose_topic_name = self.metadata["pose_topic"]
        self.num_images = self.metadata["num_images"]
        assert self.num_images > 0
        self.image_height = self.metadata["image_height"]
        self.image_width = self.metadata["image_width"]
        self.scene_scale_factor = dataparser_outputs.dataparser_scale 
        self.device = device

        self.cameras = self.cameras.to(device=self.device)

        self.image_tensor = torch.ones(
            self.num_images, self.image_height, self.image_width, 3, dtype=torch.float32
        )
        self.image_indices = torch.arange(self.num_images)

        self.updated_indices = []

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx: int):
        """
        This returns the data as a dictionary which is not actually how it is
        accessed in the dataloader, but we allow this as well so that we do not
        have to rewrite the several downstream functions.
        """
        data = {"image_idx": idx, "image": self.image_tensor[idx]}
        return data


class ROSDepthDataset(ROSDataset):
    """
    This is a tensor dataset that keeps track of all of the data streamed by ROS.
    It's main purpose is to conform to the already defined workflow of nerfstudio:
        (dataparser -> inputdataset -> dataloader).

    This is effectively the same as the regular ROSDataset, but includes additional
    tensors for depth.

    Images are tracked in self.image_tensor with uninitialized images set to
    all white (hence torch.ones). Similarly, we store depth data in self.depth_tensor.
    Poses are stored in self.cameras.camera_to_worlds as 3x4 transformation tensors.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__(dataparser_outputs, scale_factor, device)
        assert "depth_topic" in dataparser_outputs.metadata.keys()
        assert "depth_scale_factor" in dataparser_outputs.metadata.keys()

        self.depth_topic_name = self.metadata["depth_topic"]
        self.depth_scale_factor = self.metadata["depth_scale_factor"]
        self.depth_tensor = torch.ones(
            self.num_images,
            self.image_height,
            self.image_width,
            1,
            dtype=torch.float32,
        )

    def __getitem__(self, idx: int):
        """
        This returns the data as a dictionary which is not actually how it is
        accessed in the dataloader, but we allow this as well so that we do not
        have to rewrite the several downstream functions.
        """
        data = super().__getitem__(idx)
        data["depth"] = self.depth_tensor[idx]
        return data
