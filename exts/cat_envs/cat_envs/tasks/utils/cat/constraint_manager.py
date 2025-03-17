# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint manager for computing prob signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from omni.isaac.lab.managers.manager_base import ManagerBase, ManagerTermBase
from .manager_constraint_cfg import ConstraintTermCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class CaT:
    """Handle the computation of termination probabilities based on constraints
    violations (Constraints as Terminations).

    Args:
        tau (float): discount factor
        min_p (float): minimum termination probability
    """

    def __init__(self, tau=0.95, min_p=0.0):
        self.running_maxes = {}  # Polyak average of the maximum constraint violation
        self.running_mins = {}  # Polyak average of the minimum constraint violation
        self.probs = {}  # Termination probabilities for each constraint
        self.max_p = {}
        self.raw_constraints = {}
        self.tau = tau  # Discount factor
        self.min_p = min_p  # Minimum termination probability

    def reset(self):
        """Reset the termination probabilities of the constraint manager."""
        self.probs = {}
        self.raw_constraints = {}

    def add(self, name, constraint, max_p=0.1):
        """Add a constraint violation to the constraint manager and compute the
        associated termination probability.

        Args:
            name (string): name of the constraint
            constraint (float tensor): value of constraint violations for this constraint
            max_p (float): maximum termination probability
        """

        # First, put constraint in the form Torch.FloatTensor((num_envs, n_constraints))
        # Convert constraints violation to float if they are not
        if not torch.is_floating_point(constraint):
            constraint = constraint.float()

        # Ensure constraint is 2-dimensional even with a single element
        if len(constraint.size()) == 1:
            constraint = constraint.unsqueeze(1)

        # Get the maximum constraint violation for the current step
        constraint_max = constraint.max(dim=0, keepdim=True)[0].clamp(min=1e-6)

        # Compute polyak average of the maximum constraint violation for this constraint
        if name not in self.running_maxes:
            self.running_maxes[name] = constraint_max
        else:
            self.running_maxes[name] = (
                self.tau * self.running_maxes[name] + (1.0 - self.tau) * constraint_max
            )

        self.raw_constraints[name] = constraint

        # Get samples for which there is a constraint violation
        mask = constraint > 0.0

        # Compute the termination probability which scales between min_p and max_p with
        # increasing constraint violation. Remains at 0 when there is no violation.
        probs = torch.zeros_like(constraint)
        probs[mask] = self.min_p + torch.clamp(
            constraint[mask]
            / (self.running_maxes[name].expand(constraint.size())[mask]),
            min=0.0,
            max=1.0,
        ) * (max_p - self.min_p)
        self.probs[name] = probs
        self.max_p[name] = torch.tensor(max_p, device=constraint.device).repeat(
            constraint.shape[1]
        )

    def get_probs(self):
        """Returns the termination probabilities due to constraint violations."""
        probs = torch.cat(list(self.probs.values()), dim=1)
        probs = probs.max(1).values
        return probs

    def get_raw_constraints(self):
        return torch.cat(list(self.raw_constraints.values()), dim=1)

    def get_running_maxes(self):
        return torch.cat(list(self.running_maxes.values()), dim=1)

    def get_max_p(self):
        return torch.cat(list(self.max_p.values()))

    def get_str(self, names=None):
        """Get a debug string with constraints names and their average termination probabilities"""
        if names is None:
            names = list(self.probs.keys())
        txt = ""
        for name in names:
            txt += " {}: {}".format(
                name,
                str(
                    100.0 * self.probs[name].max(1).values.gt(0.0).float().mean().item()
                )[:4],
            )
            # txt += " {}: {}".format(name, str(100.0*self.probs[name].max(1).values.float().mean().item())[:4])

        return txt[1:]

    def log_all(self, episode_sums):
        """Log terminations probabilities in episode_sums with cstr_NAME key."""
        for name in list(self.probs.keys()):
            values = self.probs[name].max(1).values.gt(0.0).float()
            if "cstr_" + name not in episode_sums:
                episode_sums["cstr_" + name] = torch.zeros_like(values)
            episode_sums["cstr_" + name] += values

    def get_names(self):
        """Return a list of all constraint names."""
        return list(self.probs.keys())

    def get_vals(self):
        """Return a list of all constraint termination probabilities."""
        res = []
        for key in self.probs.keys():
            res += [100.0 * self.probs[key].max(1).values.gt(0.0).float().mean().item()]
        return res


class ConstraintManager(ManagerBase):
    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(
        self, cfg: object, env: ManagerBasedRLEnv, tau: float = 0.95, min_p: float = 0.0
    ):
        """Initialize the reward manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RewardTermCfg]``).
            env: The environment instance.
        """
        # CaT
        self.cat = CaT(tau, min_p)

        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._term_cfgs: list[ConstraintTermCfg] = list()
        self._class_term_cfgs: list[ConstraintTermCfg] = list()

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        # prepare extra info to store individual constraint term information
        self._episode_sums = dict()
        self._cstr_mean_values = dict()
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
            self._cstr_mean_values[term_name] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
        # create buffer for managing constraint prob per environment
        self._cstr_prob_buf = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<ConstraintManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Constraint Terms"
        table.field_names = ["Index", "Name", "Limit", "Names", "Max p"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Limit"] = "r"
        table.align["Body"] = "r"
        table.align["Max p"] = "r"
        # add info on each term
        for index, (name, term_cfg) in enumerate(
            zip(self._term_names, self._term_cfgs)
        ):
            if "limit" in term_cfg.params:
                if "names" in term_cfg.params:
                    table.add_row(
                        [
                            index,
                            name,
                            term_cfg.params["limit"],
                            term_cfg.params["names"],
                            term_cfg.max_p,
                        ]
                    )
                else:
                    table.add_row(
                        [index, name, term_cfg.params["limit"], "-", term_cfg.max_p]
                    )
            elif "names" in term_cfg.params:
                table.add_row(
                    [index, name, "-", term_cfg.params["names"], term_cfg.max_p]
                )
            else:
                table.add_row([index, name, "-", "-", term_cfg.max_p])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active reward terms."""
        return self._term_names

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual reward terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual reward terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for key in self._episode_sums.keys():
            # store information
            extras["Episode_Constraint_violation/" + key] = (
                torch.mean(
                    self._episode_sums[key][env_ids]
                    / self._env.episode_length_buf[env_ids],
                    dim=0,
                )
                * 100
            )
            extras["Episode_Constraint_probability/" + key] = torch.mean(
                self._cstr_mean_values[key][env_ids]
                / self._env.episode_length_buf[env_ids],
                dim=0,
            )
            # reset episodic sum
            self._episode_sums[key][env_ids] = 0.0
            self._cstr_mean_values[key][env_ids] = 0.0
        # reset all the constraints terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # iterate over all the reward terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            self.cat.add(
                name, term_cfg.func(self._env, **term_cfg.params), term_cfg.max_p
            )
        cstr_prob = self.cat.get_probs()

        for name in self._term_names:
            self._episode_sums[name] += (
                self.cat.probs[name].max(1).values.gt(0.0).float()
            )
            self._cstr_mean_values[name] += self.cat.probs[name].max(1).values

        return cstr_prob

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: ConstraintTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the reward term.
            cfg: The configuration for the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Constraint term '{term_name}' not found.")
        # set the configuration
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> ConstraintTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the reward term.

        Returns:
            The configuration of the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Constraint term '{term_name}' not found.")
        # return the configuration
        return self._term_cfgs[self._term_names.index(term_name)]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, ConstraintTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ConstraintTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # check for valid limit type
            if not isinstance(term_cfg.max_p, (float, int)):
                raise TypeError(
                    f"Limit for the term '{term_name}' is not of type float or int."
                    f" Received: '{type(term_cfg.max_p)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)
