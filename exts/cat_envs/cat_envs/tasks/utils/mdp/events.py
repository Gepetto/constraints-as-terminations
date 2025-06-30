# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING


import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    max_displacement: float,
    asset_cfg: SceneEntityCfg,
):
    """Randomize the CoM of the bodies by adding a random value sampled from the given range.

    .. tip::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()[:, body_ids, :3]

    # Randomize the com in range -max displacement to max displacement
    coms += torch.rand_like(coms) * 2 * max_displacement - max_displacement

    # Set the new coms
    new_coms = asset.root_physx_view.get_coms().clone()
    new_coms[:, asset_cfg.body_ids, 0:3] = coms
    asset.root_physx_view.set_coms(new_coms, env_ids)


def push_by_setting_velocity_with_random_envs(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """pushing function ported from isaacgym CaT"""

    p_push = env.physics_dt / (
        env.max_episode_length_s * 2
    )  # <- time step / duration of X seconds
    # There will be a probability of 0.63 of having at least one swap after X seconds have elapsed
    # (1 / p) policy steps for X seconds, and the probability of having no swap at all is (1 - p)**(1 / p) = 0.37
    # The mean number of swaps for (1 / p) steps with probability p is 1.
    push_idx = (
        torch.bernoulli(torch.full((env.num_envs,), p_push, device=env.device))
        .nonzero(as_tuple=False)
        .flatten()
    )

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[push_idx]

    # sample random velocities
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w[:] = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device
    )

    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=push_idx)
