# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optimized constraint manager for computing prob signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Dict, List

from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
from .manager_constraint_cfg import ConstraintTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CaT:
    """Handle the computation of termination probabilities based on constraints violations."""

    def __init__(self, tau: float = 0.95, min_p: float = 0.0):
        self.running_maxes: Dict[str, torch.Tensor] = {}  # Polyak average of max constraint violation
        self.probs: Dict[str, torch.Tensor] = {}         # Termination probabilities
        self.max_p: Dict[str, torch.Tensor] = {}         # Maximum termination probabilities
        self.raw_constraints: Dict[str, torch.Tensor] = {} # Raw constraint values
        self.tau = tau                                  # Discount factor
        self.min_p = min_p                              # Minimum termination probability
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reset(self):
        """Reset the termination probabilities and constraints."""
        self.probs.clear()
        self.raw_constraints.clear()

    def add(self, name: str, constraint: torch.Tensor, max_p: float = 0.1):
        """Add and process a constraint violation."""
        # Ensure constraint is on the correct device and properly shaped
        if constraint.device != self._device:
            constraint = constraint.to(self._device)
            
        # Convert to float if needed and ensure proper shape
        if not torch.is_floating_point(constraint):
            constraint = constraint.float()
        if constraint.ndim == 1:
            constraint = constraint.unsqueeze(1)

        # Store raw constraints
        self.raw_constraints[name] = constraint

        # Compute constraint max with clamping for numerical stability
        constraint_max = constraint.max(dim=0, keepdim=True)[0].clamp(min=1e-6)

        # Update running max using in-place operations when possible
        if name in self.running_maxes:
            self.running_maxes[name].mul_(self.tau).add_((1.0 - self.tau) * constraint_max)
        else:
            self.running_maxes[name] = constraint_max

        # Pre-allocate probability tensor
        probs = torch.zeros_like(constraint)
        
        # Compute mask of violations only once
        mask = constraint > 0.0
        if mask.any():
            # Compute normalized violations with broadcasting
            normalized = constraint / self.running_maxes[name].expand_as(constraint)
            # Compute probabilities only for violating samples
            probs[mask] = self.min_p + torch.clamp(normalized[mask], 0.0, 1.0) * (max_p - self.min_p)

        self.probs[name] = probs
        self.max_p[name] = torch.full((constraint.shape[1],), max_p, 
                                    dtype=torch.float, device=self._device)

    def get_probs(self) -> torch.Tensor:
        """Returns the termination probabilities due to constraint violations."""
        if not self.probs:
            return torch.tensor([], device=self._device)
        return torch.cat(list(self.probs.values()), dim=1).max(1).values

    def get_raw_constraints(self) -> torch.Tensor:
        return torch.cat(list(self.raw_constraints.values()), dim=1) if self.raw_constraints else torch.tensor([], device=self._device)

    def get_running_maxes(self) -> torch.Tensor:
        return torch.cat(list(self.running_maxes.values()), dim=1) if self.running_maxes else torch.tensor([], device=self._device)

    def get_max_p(self) -> torch.Tensor:
        return torch.cat(list(self.max_p.values())) if self.max_p else torch.tensor([], device=self._device)

    def get_str(self, names: List[str] | None = None) -> str:
        """Get debug string with constraints names and average termination probabilities"""
        names = names or list(self.probs.keys())
        parts = []
        for name in names:
            prob = 100.0 * self.probs[name].max(1).values.gt(0.0).float().mean().item()
            parts.append(f"{name}: {prob:.1f}")
        return " ".join(parts)

    def log_all(self, episode_sums: Dict[str, torch.Tensor]):
        """Log terminations probabilities in episode_sums with cstr_NAME key."""
        for name, probs in self.probs.items():
            key = f"cstr_{name}"
            values = probs.max(1).values.gt(0.0).float()
            if key not in episode_sums:
                episode_sums[key] = torch.zeros_like(values)
            episode_sums[key].add_(values)

    def get_names(self) -> List[str]:
        return list(self.probs.keys())

    def get_vals(self) -> List[float]:
        return [100.0 * probs.max(1).values.gt(0.0).float().mean().item() 
               for probs in self.probs.values()]


class ConstraintManager(ManagerBase):
    _env: ManagerBasedRLEnv

    def __init__(
        self, cfg: object, env: ManagerBasedRLEnv, tau: float = 0.95, min_p: float = 0.0
    ):
        """Initialize the constraint manager."""
        # Initialize CaT with device from env
        self.cat = CaT(tau, min_p)
        self._device = env.device
        
        # Initialize buffers
        self._term_names: List[str] = []
        self._term_cfgs: List[ConstraintTermCfg] = []
        self._class_term_cfgs: List[ConstraintTermCfg] = []
        
        # Initialize base class and parse terms
        super().__init__(cfg, env)
        
        # Pre-allocate buffers
        self._episode_sums: Dict[str, torch.Tensor] = {}
        self._cstr_mean_values: Dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.num_envs, 
                                                      dtype=torch.float, 
                                                      device=self._device)
            self._cstr_mean_values[term_name] = torch.zeros(self.num_envs, 
                                                          dtype=torch.float, 
                                                          device=self._device)
        
        self._cstr_prob_buf = torch.zeros(self.num_envs, 
                                         dtype=torch.float, 
                                         device=self._device)

    def __str__(self) -> str:
        """Returns a string representation of the constraint manager."""
        msg = f"<ConstraintManager> contains {len(self._term_names)} active terms.\n"
        
        table = PrettyTable()
        table.title = "Active Constraint Terms"
        table.field_names = ["Index", "Name", "Limit", "Names", "Max p"]
        table.align["Name"] = "l"
        table.align["Limit"] = "r"
        table.align["Body"] = "r"
        table.align["Max p"] = "r"
        
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            limit_value = term_cfg.params.get("limit", "-")
            
            names_value = "-"
            if "asset_cfg" in term_cfg.params and term_cfg.params["asset_cfg"] is not None:
                asset_cfg = term_cfg.params["asset_cfg"]
                names_value = asset_cfg.body_names or asset_cfg.joint_names or "-"
            elif "names" in term_cfg.params:
                import warnings
                warnings.warn(
                    "Using 'names' parameter is deprecated. Use 'asset_cfg' instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                names_value = term_cfg.params["names"]
            
            table.add_row([index, name, limit_value, names_value, term_cfg.max_p])
        
        msg += table.get_string() + "\n"
        return msg

    @property
    def active_terms(self) -> List[str]:
        return self._term_names

    def reset(self, env_ids: Sequence[int] | None = None) -> Dict[str, torch.Tensor]:
        env_ids = slice(None) if env_ids is None else env_ids
        extras = {}
        
        for key in self._episode_sums:
            # Compute mean values using in-place operations where possible
            episode_lengths = self._env.episode_length_buf[env_ids]
            violation_mean = (self._episode_sums[key][env_ids] / episode_lengths).mean() * 100
            prob_mean = (self._cstr_mean_values[key][env_ids] / episode_lengths).mean()
            
            extras[f"Episode_Constraint_violation/{key}"] = violation_mean
            extras[f"Episode_Constraint_probability/{key}"] = prob_mean
            
            # Reset buffers
            self._episode_sums[key][env_ids] = 0.0
            self._cstr_mean_values[key][env_ids] = 0.0
        
        # Reset class terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
            
        return extras

    def compute(self) -> torch.Tensor:
        """Compute constraint probabilities with minimal overhead."""
        # Process all constraints
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            self.cat.add(name, term_cfg.func(self._env, **term_cfg.params), term_cfg.max_p)
        
        # Get combined probabilities
        cstr_prob = self.cat.get_probs()
        
        # Update statistics
        for name in self._term_names:
            probs = self.cat.probs[name]
            max_probs = probs.max(1).values
            self._episode_sums[name].add_(max_probs.gt(0.0).float())
            self._cstr_mean_values[name].add_(max_probs)
        
        return cstr_prob

    def set_term_cfg(self, term_name: str, cfg: ConstraintTermCfg):
        if term_name not in self._term_names:
            raise ValueError(f"Constraint term '{term_name}' not found.")
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> ConstraintTermCfg:
        if term_name not in self._term_names:
            raise ValueError(f"Constraint term '{term_name}' not found.")
        return self._term_cfgs[self._term_names.index(term_name)]

    def _prepare_terms(self):
        cfg_items = self.cfg.items() if isinstance(self.cfg, dict) else self.cfg.__dict__.items()
        
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
                
            if not isinstance(term_cfg, ConstraintTermCfg):
                raise TypeError(
                    f"Configuration for term '{term_name}' is not ConstraintTermCfg. "
                    f"Received: '{type(term_cfg)}'."
                )
                
            if not isinstance(term_cfg.max_p, (float, int)):
                raise TypeError(
                    f"Limit for term '{term_name}' must be float or int. "
                    f"Received: '{type(term_cfg.max_p)}'."
                )
                
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)