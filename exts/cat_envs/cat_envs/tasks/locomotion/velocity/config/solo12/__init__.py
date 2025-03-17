# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from cat_envs.tasks.utils.cat.cat_env import CaTEnv


##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-CaT-Flat-Solo12-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_flat_env_cfg:Solo12FlatEnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-CaT-Flat-Solo12-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_flat_env_cfg:Solo12FlatEnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)
