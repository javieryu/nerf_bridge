"""
A datamanager for the NerfBridge.
"""

from dataclasses import dataclass, field
from typing import Type, Dict, Tuple, Generic, cast, get_origin, get_args
from typing_extensions import TypeVar
from functools import cached_property
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils import writer
from nerfstudio.utils.writer import EventType

from rich.console import Console

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.rays import RayBundle

from nerfbridge.ros_dataset import ROSDataset
from nerfbridge.ros_dataloader import ROSDataloader
from nerfbridge.ros_dataparser import ROSDataParserConfig

import pdb


CONSOLE = Console(width=120)


@dataclass
class ROSDataManagerConfig(VanillaDataManagerConfig):
    """A ROS datamanager that handles a streaming dataloader."""

    _target: Type = field(default_factory=lambda: ROSDataManager)
    dataparser: ROSDataParserConfig = ROSDataParserConfig()
    """ Must use only the ROSDataParser here """
    publish_training_posearray: bool = True
    """ Whether the dataloader should publish an pose array of the training image poses. """
    data_update_freq: float = 5.0
    """ Frequency, in Hz, that images are added to the training dataset tensor. """
    num_training_images: int = 500
    """ Number of images to train on (for dataset tensor pre-allocation). """
    slam_method: str = "cuvslam"
    """ Which slam method is being used. """
    topic_sync: str = "approx"
    """ Whether to use approximate or exact time synchronization for pose image pairs."""
    topic_slop: float = 0.05
    """ Slop in seconds for approximate time synchronization."""
    use_compressed_rgb: bool = False
    """ Whether to use compressed RGB image topic or not."""


TDataset = TypeVar("TDataset", bound=ROSDataset, default=ROSDataset)


class ROSDataManager(
    VanillaDataManager, Generic[TDataset]
):  # pylint: disable=abstract-method
    """Essentially the VannilaDataManager from Nerfstudio except that the
    typical dataloader for training images is replaced with one that streams
    image and pose data from ROS.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ROSDataManagerConfig
    train_dataset: ROSDataset

    def create_train_dataset(self) -> ROSDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(
            split="train", num_images=self.config.num_training_images
        )
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs, device=self.device
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """
        Returns the dataset type passed as the generic argument.

        NOTE: Hacked from the Vanilla DataManager implementation.
        """
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[ROSDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is ROSDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is ROSDataManager:
            return get_args(orig_class)[0]

    def setup_train(self):
        assert self.train_dataset is not None
        self.train_image_dataloader = ROSDataloader(
            self.train_dataset,
            self.config.data_update_freq,
            self.config.slam_method,
            self.config.topic_sync,
            self.config.topic_slop,
            self.config.use_compressed_rgb,
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
        Evaluation data is not implemented! This function is called by
        the parent class, but the results are never used.
        """
        pass

    def create_eval_dataset(self):
        """
        Evaluation data is not implemented! This function is called by
        the parent class, but the results are never used.
        """
        pass

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        CONSOLE.print("Evaluation data is not setup!")
        raise NameError(
            "Evaluation funcationality not yet implemented with ROS Streaming."
        )

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        CONSOLE.print("Evaluation data is not setup!")
        raise NameError(
            "Evaluation funcationality not yet implemented with ROS Streaming."
        )
