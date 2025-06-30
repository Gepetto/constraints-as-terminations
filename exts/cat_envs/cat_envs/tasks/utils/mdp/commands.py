# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformVelocityCommandWithDeadzone(mdp.UniformVelocityCommand):
    """velocity command sampling class ported from isaacgym CaT"""

    cfg: "UniformVelocityCommandWithDeadzoneCfg"

    def __init__(
        self, cfg: "UniformVelocityCommandWithDeadzoneCfg", env: ManagerBasedEnv
    ):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)

        self.velocity_deadzone = cfg.velocity_deadzone
        self.dt = env.physics_dt
        self.max_episode_length_s = env.max_episode_length_s

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        # set small commands to zero
        self.vel_command_b *= (
            torch.any(
                torch.abs(self.vel_command_b[:, :3]) > self.velocity_deadzone, dim=1
            )
        ).unsqueeze(1)

        # Random velocity command resampling
        no_vel_command = (
            torch.norm(self.vel_command_b[:, :3], dim=1) < self.velocity_deadzone
        ).float()
        p_resample_command = 0.01 * no_vel_command + (
            self.dt / self.max_episode_length_s
        ) * (1 - no_vel_command)
        resample_command_idx = (
            torch.bernoulli(p_resample_command).nonzero(as_tuple=False).flatten()
        )
        if len(resample_command_idx) > 0:
            self._resample(resample_command_idx)

        # Random angular velocity inversion during the episode to avoid having the robot moving in circle
        p_ang_vel = (
            self.dt / self.max_episode_length_s
        )  # <- time step / duration of X seconds
        # There will be a probability of 0.63 of having at least one swap after X seconds have elapsed
        # (1 / p) policy steps for X seconds, and the probability of having no swap at all is (1 - p)**(1 / p) = 0.37
        # The mean number of swaps for (1 / p) steps with probability p is 1.
        self.vel_command_b[:, 2] *= (
            1
            - 2
            * torch.bernoulli(
                torch.full_like(self.vel_command_b[:, 2], p_ang_vel)
            ).float()
        )


@configclass
class UniformVelocityCommandWithDeadzoneCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = UniformVelocityCommandWithDeadzone
    velocity_deadzone: float = 0.1
