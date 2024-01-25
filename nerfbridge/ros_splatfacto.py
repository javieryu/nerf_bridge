from dataclasses import dataclass, field
from typing import Type, List

import torch

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.cameras.cameras import Cameras


@dataclass
class ROSSplatfactoModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: ROSSplatfactoModel)


class ROSSplatfactoModel(SplatfactoModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def seed_cb(self, pipeline: Pipeline, step: int):
        pass

    def seed_from_rgbd(self, camera: Cameras, rgb: torch.Tensor, depth: torch.Tensor):
        pass

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs_base = super().get_training_callbacks(training_callback_attributes)

        cb_seed = TrainingCallback(
            [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            self.seed_cb,
            args=[training_callback_attributes.pipeline],
        )
        return [cb_seed] + cbs_base
