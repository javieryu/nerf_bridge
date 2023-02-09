"""
Depth datamanager.
"""

from dataclasses import dataclass, field
from typing import Type, Dict, Tuple

from rich.console import Console

from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.rays import RayBundle

from nsros.ros_dataset import ROSDataset
from nsros.ros_dataloader import ROSDataloader
from nsros.ros_dataparser import ROSDataParserConfig

import pdb
import torch


CONSOLE = Console(width=120)


@dataclass
class ROSDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """A ROS datamanager that handles a streaming dataloader."""

    _target: Type = field(default_factory=lambda: ROSDataManager)
    dataparser: ROSDataParserConfig = ROSDataParserConfig()
    """ Must use only the ROSDataParser here """


class ROSDataManager(
    base_datamanager.VanillaDataManager
):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing depth data.
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ROSDataManagerConfig
    train_dataset: ROSDataset

    def create_train_dataset(self) -> ROSDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(
            split="train"
        )
        return ROSDataset(
            dataparser_outputs=self.train_dataparser_outputs, device=self.device
        )

    def setup_train(self):
        assert self.train_dataset is not None
        # Setup custom dataloader for ROS, and print rostopic statuses
        # Everything else should be the same.
        self.images_to_start = self.train_dataset.metadata["images_to_start"]

        self.train_image_dataloader = ROSDataloader(
            self.train_dataset,
            device=self.device,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(
            self.train_dataset, self.config.train_num_rays_per_batch
        )
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras,
            self.train_camera_optimizer,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """
        First, checks for updates to the ROSDataloader, and then returns the next
        batch of data from the train dataloader.
        """
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def setup_eval(self):
        """
        Evaluation data is not implemented!
        """
        pass

    def create_eval_dataset(self):
        """
        Evaluation data is not implemented!
        """
        pass

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        CONSOLE.print("Evaluation data is not setup!")
        raise NameError(
            "Evaluation funcationality not yet implemented with ROS Streaming."
        )
        # self.eval_count += 1
        # image_batch = next(self.iter_eval_image_dataloader)
        # assert self.eval_pixel_sampler is not None
        # batch = self.eval_pixel_sampler.sample(image_batch)
        # ray_indices = batch["indices"]
        # ray_bundle = self.eval_ray_generator(ray_indices)

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        CONSOLE.print("Evaluation data is not setup!")
        raise NameError(
            "Evaluation funcationality not yet implemented with ROS Streaming."
        )
        # for camera_ray_bundle, batch in self.eval_dataloader:
        #     assert camera_ray_bundle.camera_indices is not None
        #     image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
        #     return image_idx, camera_ray_bundle, batch
        # raise ValueError("No more eval images")
