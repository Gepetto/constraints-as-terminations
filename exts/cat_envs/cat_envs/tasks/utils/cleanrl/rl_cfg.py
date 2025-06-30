# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class CleanRlPpoActorCriticCfg:
    seed: int = 42

    save_interval: int = MISSING

    learning_rate: float = MISSING
    num_steps: int = MISSING
    num_iterations: int = MISSING
    gamma: float = MISSING
    gae_lambda: float = MISSING
    updates_epochs: int = MISSING
    minibatch_size: int = MISSING
    clip_coef: float = MISSING
    ent_coef: float = MISSING
    vf_coef: float = MISSING
    max_grad_norm: float = MISSING
    norm_adv: bool = MISSING
    clip_vloss: bool = MISSING
    anneal_lr: bool = MISSING

    experiment_name: str = MISSING
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = MISSING

    load_run: str = MISSING
    load_checkpoint: str = MISSING
