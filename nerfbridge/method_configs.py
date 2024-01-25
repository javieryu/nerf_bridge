# Code slightly adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/configs/method_configs.py

"""
NerfBridge Method Configs
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.model_components.losses import DepthLossType
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig


from nerfbridge.ros_datamanager import (
    ROSDataManagerConfig,
    ROSDataManager,
    ROSFullImageDataManagerConfig,
    ROSFullImageDataManager,
)
from nerfbridge.ros_dataparser import ROSDataParserConfig
from nerfbridge.ros_trainer import ROSTrainerConfig
from nerfbridge.ros_dataset import ROSDataset, ROSDepthDataset
from nerfbridge.ros_splatfacto import ROSSplatfactoModelConfig

RosNerfacto = MethodSpecification(
    config=ROSTrainerConfig(
        method_name="ros-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=30000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ROSDataManagerConfig(
                _target=ROSDataManager[ROSDataset],
                dataparser=ROSDataParserConfig(
                    aabb_scale=1.0,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=20000),
        vis="viewer",
    ),
    description="Run NerfBridge with the Nerfacto model, and train with streamed RGB images.",
)

RosDepthNerfacto = MethodSpecification(
    config=ROSTrainerConfig(
        method_name="ros-depth-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=30000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ROSDataManagerConfig(
                _target=ROSDataManager[ROSDepthDataset],
                pixel_sampler=PairPixelSamplerConfig(),
                dataparser=ROSDataParserConfig(
                    aabb_scale=1.0,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=DepthNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                depth_loss_type=DepthLossType.DS_NERF,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=20000),
        vis="viewer",
    ),
    description="Run NerfBridge with the DepthNerfacto model, and train with streamed RGB and depth images.",
)

RosDepthSplatfacto = MethodSpecification(
    config=ROSTrainerConfig(
        method_name="ros-depth-splatfacto",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100},
        pipeline=VanillaPipelineConfig(
            datamanager=ROSFullImageDataManagerConfig(
                _target=ROSFullImageDataManager[ROSDepthDataset],
                dataparser=ROSDataParserConfig(aabb_scale=1.0),
            ),
            model=ROSSplatfactoModelConfig(),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Run NerfBridge with the Splatfacto model, and train with streamed RGB and depth images.",
)
