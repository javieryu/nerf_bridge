from __future__ import annotations

import time
import dataclasses
from dataclasses import dataclass, field
from typing import Type
from typing_extensions import Literal

from rich.console import Console

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.utils import profiler, writer

from nerfbridge.ros_dataset import ROSDataset
from nerfbridge.ros_viewer import ROSViewer

CONSOLE = Console(width=120)


@dataclass
class ROSTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: ROSTrainer)
    msg_timeout: float = 60.0
    """ How long to wait (seconds) for sufficient images to be received before training. """
    num_msgs_to_start: int = 3
    """ Number of images that must be recieved before training can start. """


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
        self.num_msgs_to_start = config.num_msgs_to_start

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            CONSOLE.print(
                "[bold red] (NerfBridge) Legacy Viewer is not supported by NerfBridge!"
            )
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ROSViewer(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info

        self._check_viewer_warnings()

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging,
            max_iter=self.config.max_num_iterations,
            banner_messages=banner_messages,
        )
        writer.put_config(
            name="config", config_dict=dataclasses.asdict(self.config), step=0
        )
        profiler.setup_profiler(self.config.logging, writer_log_path)

        # Start Status check loop
        start = time.perf_counter()
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
