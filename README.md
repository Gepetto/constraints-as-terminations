# Constraints As Terminations (CaT)

[Website](https://constraints-as-terminations.github.io) | [Technical Paper](https://arxiv.org/abs/2403.18765) | [Videos](https://www.youtube.com/watch?v=crWoYTb8QvU)

![](assets/teaser.png)

## About this repository

This repository contains the IsaacLab code associated with the article **CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning** by Elliot Chane-Sane\*, Pierre-Alexandre Leziart\*, Thomas Flayols, Olivier Stasse, Philippe Souères, and Nicolas Mansard.

This paper has been accepted for the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024).

This code relies on the [CleanRL](https://github.com/vwxyzjn/cleanrl) library and [IsaacLab](https://isaac-sim.github.io/IsaacLab/v1.4.1/index.html) (version 1.4.1).

Implementation of the constraints manager and modification of the environment can be found in the [CaT directory](exts/cat_envs/cat_envs/tasks/utils/cat/). The modified PPO implementation can be found in the [CleanRL directory](exts/cat_envs/cat_envs/tasks/utils/cleanrl/).

`ConstraintsManager` follows the manager-based Isaac Lab approach, allowing easy integration just like other managers. For a full example, check out [cat_flat_env_cfg.py](exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/solo12/cat_flat_env_cfg.py).

```python
@configclass
class ConstraintsCfg:
    # Safety Soft Constraints
    joint_torque = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={"limit": 3.0, "names": [".*_HAA", ".*_HFE", ".*_KFE"]},
    )
    # Safety Hard Constraints
    contact = ConstraintTerm(
        func=constraints.contact,
        max_p=1.0,
        params={"names": ["base_link", ".*_UPPER_LEG"]},
    )
```

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/v1.4.1/source/setup/installation/index.html).
- Clone the repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory).
- Using a Python interpreter that has Isaac Lab installed, install the library:

```bash
python -m pip install -e exts/cat_envs
```

## Running CaT

Navigate to the `/constraints-as-terminations` directory and launch a basic training setup on flat ground:

```bash
python scripts/clean_rl/train.py --task=Isaac-Velocity-CaT-Flat-Solo12-v0 --headless
```

If everything goes well, you will see monitoring statistics in the terminal as the training progresses. At the end, you can check the result with:

```bash
python scripts/clean_rl/play.py --task=Isaac-Velocity-CaT-Flat-Solo12-v0
```

## Citing

Please cite this work as:

```
@inproceedings{chane2024cat,
      title={CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning},
      author={Elliot Chane-Sane and Pierre-Alexandre Leziart and Thomas Flayols and Olivier Stasse and Philippe Sou{\`e}res and Nicolas Mansard},
      booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2024}
}
```