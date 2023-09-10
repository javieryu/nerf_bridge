from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Type
from typing_extensions import Literal

from rich.console import Console

from nerfstudio.utils.decorators import check_viewer_enabled
from nerfstudio.engine.trainer import Trainer, TrainerConfig

from nerfbridge.ros_dataset import ROSDataset

CONSOLE = Console(width=120)


@dataclass
class ROSTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: ROSTrainer)
    msg_timeout: float = 60.0
    """ How long to wait (seconds) for sufficient images to be received before training. """
    num_msgs_to_start: int = 30
    """ Number of images that must be recieved before training can start. """
    draw_training_images: bool = False
    """ Whether or not to draw the training images in the viewer. """


class ROSTrainer(Trainer):
    config: ROSTrainerConfig
    dataset: ROSDataset

    def __init__(
        self, config: ROSTrainerConfig, local_rank: int = 0, world_size: int = 0
    ):
        # We'll see if this throws and error (it expects a different config type)
        super().__init__(config, local_rank=local_rank, world_size=world_size)
        self.msg_timeout = self.config.msg_timeout
        self.cameras_drawn = []
        self.first_update = True
        self.num_msgs_to_start = config.num_msgs_to_start

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val"):
        """
        Runs the Trainer setup, and then waits until at least one image-pose
        pair is successfully streamed from ROS before allowing training to proceed.
        """
        # This gets called in the script that launches the training.
        # In this case ns_ros/ros_train.py
        super().setup(test_mode=test_mode)
        start = time.perf_counter()

        # Start Status check loop
        status = False
        CONSOLE.print(
            f"[bold green] (NerfBridge) Waiting for for image streaming to begin ...."
        )
        while time.perf_counter() - start < self.msg_timeout:
            if self.pipeline.datamanager.train_image_dataloader.msg_status(  # pyright: ignore
                self.num_msgs_to_start
            ):
                status = True
                break
            time.sleep(0.03)

        self.dataset = self.pipeline.datamanager.train_dataset  # pyright: ignore

        if not status:
            raise NameError(
                "ROSTrainer setup() timed out, check that topics are being published \
                 and that config.json correctly specifies their names."
            )
        else:
            CONSOLE.print(
                "[bold green] (NerfBridge) Dataloader is successfully streaming images!"
            )

    @check_viewer_enabled
    def _update_viewer_state(self, step: int):
        """
        Updates the viewer state by rendering out scene with current pipeline

        Args:
            step: current train step
        """
        super()._update_viewer_state(step)
        #
        # # Clear any old cameras!
        if self.config.draw_training_images:
            if self.first_update:
                self.viewer_state.vis["sceneState/cameras"].delete()
                self.first_update = False

            # Draw any new training images
            image_indices = self.dataset.updated_indices
            for idx in image_indices:
                if not idx in self.cameras_drawn:
                    # Do a copy here just to make sure we aren't
                    # changing the training data downstream.
                    # TODO: Verify if we need to do this
                    image = self.dataset[idx]["image"]
                    bgr = image[..., [2, 1, 0]]
                    camera_json = self.dataset.cameras.to_json(
                        camera_idx=idx, image=bgr, max_size=10
                    )

                    self.viewer_state.vis[f"sceneState/cameras/{idx:06d}"].write(
                        camera_json
                    )
                    self.cameras_drawn.append(idx)
