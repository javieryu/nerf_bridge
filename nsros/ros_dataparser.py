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
    """ROS dataset parser config"""

    _target: Type = field(default_factory=lambda: ROS)
    """target class to instantiate"""
    data: Path = Path("data/ros/home")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    max_keyframes: int = 300
    """ maximum number of nerf keyframes."""
    update_freq: float = 3.0
    """ Frequency in Hz that images are added to the dataset """
    aabb_scale: float = 2.0
    """ SceneBox aabb scale."""


@dataclass
class ROS(DataParser):
    """ROS DataParser"""

    config: ROSDataParserConfig

    def __init__(self, config: ROSDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.num_kfs: int = config.max_keyframes
        self.update_freq = config.update_freq
        self.aabb = config.aabb_scale

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.data / f"nerfstudio_config.json")

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

        camera_to_world = torch.stack(
            self.num_kfs * [torch.eye(4, dtype=torch.float32)]
        )[:, :-1, :]

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
            "num_kfs": self.num_kfs,
            "update_freq": self.update_freq,
            "image_height": image_height,
            "image_width": image_width
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,  # This is empty
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs
