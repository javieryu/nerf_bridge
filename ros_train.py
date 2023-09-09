#!/usr/bin/env python
# Code adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/scripts/train.py

"""
Train a radiance field with nerfstudio using data streamed from ROS!

This is a stripped back version of the nerfstudio train.py script without
all of the distributed training code that is not support by the NSROS Bridge.

All of the tyro help functionality should still work, but instead of a CLI
just call this script directly:
    python ros_train.py ros_nerfacto --data /path/to/config.json [OPTIONS]

Code adapted from Nerfstudio
https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/scripts/train.py
"""

import signal
import sys
import random
import traceback

import numpy as np
import torch
import tyro
from rich.console import Console

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils import profiler

from nerfbridge.method_configs import AnnotatedBaseConfigUnion
from nerfbridge.ros_trainer import ROSTrainerConfig

CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def sigint_handler(signal, frame):
    """Capture keyboard interrupts before they get caught by ROS."""
    CONSOLE.print(traceback.format_exc())
    sys.exit(0)


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config: ROSTrainerConfig) -> None:
    """Main function."""

    config.set_timestamp()
    if config.data:
        CONSOLE.log(
            "Using --data alias for --data.pipeline.datamanager.dataparser.data"
        )
        config.pipeline.datamanager.dataparser.data = config.data

    # print and save config
    config.print_to_terminal()
    config.save_config()
    _set_random_seed(config.machine.seed)
    trainer = config.setup(local_rank=0, world_size=1)
    trainer.setup()
    trainer.train()
    # print the stack trace
    CONSOLE.print(traceback.format_exc())
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
    # signal.signal(signal.SIGINT, sigint_handler)
    entrypoint()
