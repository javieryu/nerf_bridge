"""Data parser for loading ROS parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class ROSDataParserConfig(DataParserConfig):
    """ROS config file parser config."""

    _target: Type = field(default_factory=lambda: ROSDataParser)
    """target class to instantiate"""
    data: Path = Path("data/ros/nsros_config.json")
    """ Path to configuration JSON. """
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    aabb_scale: float = 2.0
    """ SceneBox aabb scale."""


@dataclass
class ROSDataParser(DataParser):
    """ROS DataParser"""

    config: ROSDataParserConfig

    def __init__(self, config: ROSDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.aabb = config.aabb_scale

    def get_dataparser_outputs(self, split="train", num_images: int = 500):
        dataparser_outputs = self._generate_dataparser_outputs(split, num_images)
        return dataparser_outputs

    def _generate_dataparser_outputs(self, split="train", num_images: int = 500):
        """
        This function generates a DataParserOutputs object. Typically in Nerfstudio
        this is used to populate the training and evaluation datasets, but since with
        NSROS Bridge our aim is to stream the data then we only have to worry about
        loading the proper camera parameters and ROS topic names.

        Args:
            split: Determines the data split (not used, but left in place for consistency
                with Nerfstudio)

            num_images: The size limit of the training image dataset. This is used to
                pre-allocate tensors for the Cameras object that tracks camera pose.
        """
        meta = load_from_json(self.data)

        image_height = meta["H"]
        image_width = meta["W"]
        fx = meta["fx"]
        fy = meta["fy"]
        cx = meta["cx"]
        cy = meta["cy"]

        k1 = meta["k1"] if "k1" in meta else 0.0
        k2 = meta["k2"] if "k2" in meta else 0.0
        k3 = meta["k3"] if "k3" in meta else 0.0
        k4 = meta["k4"] if "k4" in meta else 0.0
        p1 = meta["p1"] if "p1" in meta else 0.0
        p2 = meta["p2"] if "p2" in meta else 0.0
        distort = torch.tensor([k1, k2, k3, k4, p1, p2], dtype=torch.float32)

        camera_to_world = torch.stack(num_images * [torch.eye(4, dtype=torch.float32)])[
            :, :-1, :
        ]

        # in x,y,z order
        scene_size = self.aabb
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-scene_size, -scene_size, -scene_size],
                    [scene_size, scene_size, scene_size],
                ],
                dtype=torch.float32,
            )
        )

        # Create a dummy Cameras object with the appropriate number
        # of placeholders for poses.
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=image_height,
            width=image_width,
            distortion_params=distort,
            camera_type=CameraType.PERSPECTIVE,
        )

        image_filenames = []
        metadata = {
            "image_topic": meta["image_topic"],
            "pose_topic": meta["pose_topic"],
            "num_images": num_images,
            "image_height": image_height,
            "image_width": image_width,
        }

                # Only used if depth training is enabled
        if "depth_topic" in meta:
            metadata["depth_topic"] = meta["depth_topic"]
            metadata["depth_scale_factor"] = meta["depth_scale_factor"]


        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,  # This is empty
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs
