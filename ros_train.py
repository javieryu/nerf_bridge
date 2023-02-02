#!/usr/bin/env python
"""Train a radiance field with nerfstudio using data streamed from ROS!
"""


import random
import traceback
from typing import Callable, Optional

import numpy as np
import torch
import tyro
from rich.console import Console

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils import profiler

from nsros.method_configs import AnnotatedBaseConfigUnion
from nsros.ros_trainer import ROSTrainerConfig

import pdb
import rospy

CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore

import signal
import sys

def sigint_handler(signal, frame):
    CONSOLE.print(traceback.format_exc())
    sys.exit(0)


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(config: ROSTrainerConfig):
    """Main training function that sets up and runs the trainer per process

    Args:
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed)
    trainer = config.setup(local_rank=0, world_size=1)
    trainer.setup()
    trainer.train()

def main(config: ROSTrainerConfig) -> None:
    """Main function."""

    config.set_timestamp()
    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.dataparser.data")
        config.pipeline.datamanager.dataparser.data = config.data

    # print and save config
    config.print_to_terminal()
    config.save_config()
    try:
        train_loop(config)
    except KeyboardInterrupt:
        # print the stack trace
        CONSOLE.print(traceback.format_exc())
    finally:
        profiler.flush_profiler(config.logging)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    entrypoint()
