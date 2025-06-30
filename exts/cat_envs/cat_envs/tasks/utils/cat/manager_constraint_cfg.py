# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg


##
# Constraint manager.
##


@configclass
class ConstraintTermCfg(ManagerTermBaseCfg):
    func: Callable[..., torch.Tensor] = MISSING

    max_p: float = MISSING
