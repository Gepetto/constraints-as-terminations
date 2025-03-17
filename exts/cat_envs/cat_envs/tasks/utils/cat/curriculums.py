# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def modify_constraint_p(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
    init_max_p: float,
):
    if not hasattr(env.curriculum_manager, "constraints_curriculum"):
        env.curriculum_manager.constraints_curriculum = 0

    step_cur = 1.0 / num_steps

    env.curriculum_manager.constraints_curriculum = min(
        env.curriculum_manager.constraints_curriculum + step_cur, 1.0
    )

    # Linearly interpolate the expected time for episode end: soft_p is the maximum
    # termination probability so it is an image of the expected time of death.
    T_start = 20
    T_end = 1 / init_max_p
    init_max_p = 1 / (
        T_start + env.curriculum_manager.constraints_curriculum * (T_end - T_start)
    )

    # obtain term settings
    term_cfg = env.constraint_manager.get_term_cfg(term_name)
    # update term settings
    term_cfg.max_p = init_max_p
    env.constraint_manager.set_term_cfg(term_name, term_cfg)

    return init_max_p
