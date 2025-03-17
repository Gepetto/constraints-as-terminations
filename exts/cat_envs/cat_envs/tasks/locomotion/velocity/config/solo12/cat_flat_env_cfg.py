# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from cat_envs.tasks.utils.cat.manager_constraint_cfg import (
    ConstraintTermCfg as ConstraintTerm,
)
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import cat_envs.tasks.utils.cat.constraints as constraints
import cat_envs.tasks.utils.cat.curriculums as curriculums
import cat_envs.tasks.utils.mdp.observations as observations

import cat_envs.tasks.utils.mdp.terminations as terminations
import cat_envs.tasks.utils.mdp.events as events
import cat_envs.tasks.utils.mdp.commands as commands

##
# Pre-defined configs
##
from cat_envs.assets.odri import SOLO12_MINIMAL_CFG


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = SOLO12_MINIMAL_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands.UniformVelocityCommandWithDeadzoneCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        velocity_deadzone=0.1,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-0.78, 0.78)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FL_HAA",
            "FL_HFE",
            "FL_KFE",
            "FR_HAA",
            "FR_HFE",
            "FR_KFE",
            "HR_HAA",
            "HR_HFE",
            "HR_KFE",
            "HL_HAA",
            "HL_HFE",
            "HL_KFE",
        ],
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.001, n_max=0.001), scale=0.25
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), scale=0.1
        )
        joint_pos = ObsTerm(
            func=observations.joint_pos,
            params={
                "names": [
                    "FL_HAA",
                    "FL_HFE",
                    "FL_KFE",
                    "FR_HAA",
                    "FR_HFE",
                    "FR_KFE",
                    "HR_HAA",
                    "HR_HFE",
                    "HR_KFE",
                    "HL_HAA",
                    "HL_HFE",
                    "HL_KFE",
                ]
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=observations.joint_vel,
            params={
                "names": [
                    "FL_HAA",
                    "FL_HFE",
                    "FL_KFE",
                    "FR_HAA",
                    "FR_HFE",
                    "FR_KFE",
                    "HR_HAA",
                    "HR_HFE",
                    "HR_KFE",
                    "HL_HAA",
                    "HL_HFE",
                    "HL_KFE",
                ]
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 100,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "yaw": (-1.57, 1.57),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (-0.05, 0.05),
        },
    )

    # interval

    # set pushing every step, as only some of the environments are chosen
    # as in the isaacgym cat version
    push_robot = EventTerm(
        # Standard push_by_setting_velocity also works, but interestingly results
        # in a different gait
        func=events.push_by_setting_velocity_with_random_envs,
        mode="interval",
        is_global_time=True,
        interval_range_s=(0.0, 0.005),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )


@configclass
class ConstraintsCfg:
    # Safety Soft constraints
    joint_torque = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={"limit": 3.0, "names": [".*_HAA", ".*_HFE", ".*_KFE"]},
    )
    joint_velocity = ConstraintTerm(
        func=constraints.joint_velocity,
        max_p=0.25,
        params={"limit": 16.0, "names": [".*_HAA", ".*_HFE", ".*_KFE"]},
    )
    joint_acceleration = ConstraintTerm(
        func=constraints.joint_acceleration,
        max_p=0.25,
        params={"limit": 800.0, "names": [".*_HAA", ".*_HFE", ".*_KFE"]},
    )
    action_rate = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25,
        params={"limit": 80.0, "names": [".*_HAA", ".*_HFE", ".*_KFE"]},
    )

    # Safety Hard constraints
    # Knee and base
    contact = ConstraintTerm(
        func=constraints.contact,
        max_p=1.0,
        params={"names": ["base_link", ".*_UPPER_LEG"]},
    )
    foot_contact_force = ConstraintTerm(
        func=constraints.foot_contact_force,
        max_p=1.0,
        params={"limit": 50.0, "names": [".*_FOOT"]},
    )
    front_hfe_position = ConstraintTerm(
        func=constraints.joint_position,
        max_p=1.0,
        params={"limit": 1.3, "names": ["FL_HFE", "FR_HFE"]},
    )
    upsidedown = ConstraintTerm(
        func=constraints.upsidedown, max_p=1.0, params={"limit": 0.0}
    )

    # Style constraints
    hip_position = ConstraintTerm(
        func=constraints.joint_position_when_moving_forward,
        max_p=0.25,
        params={"limit": 0.2, "names": [".*_HAA"], "velocity_deadzone": 0.1},
    )
    base_orientation = ConstraintTerm(
        func=constraints.base_orientation, max_p=0.25, params={"limit": 0.1}
    )
    air_time = ConstraintTerm(
        func=constraints.air_time,
        max_p=0.25,
        params={"limit": 0.25, "names": [".*_FOOT"], "velocity_deadzone": 0.1},
    )
    no_move = ConstraintTerm(
        func=constraints.no_move,
        max_p=0.1,
        params={
            "names": [".*_HAA", ".*_HFE", ".*_KFE"],
            "velocity_deadzone": 0.1,
            "joint_vel_limit": 4.0,
        },
    )
    two_foot_contact = ConstraintTerm(
        func=constraints.n_foot_contact,
        max_p=0.25,
        params={
            "names": [".*_FOOT"],
            "number_of_desired_feet": 2,
            "min_command_value": 0.5,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base_link", ".*_UPPER_LEG"]
            ),
            "threshold": 1.0,
        },
    )
    upside_down = DoneTerm(
        func=terminations.upside_down,
        params={
            "limit": 0.1,
        },
    )


MAX_CURRICULUM_ITERATIONS = 1000


@configclass
class CurriculumCfg:
    # Safety Soft constraints
    joint_torque = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_torque",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_velocity = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_velocity",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_acceleration = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_acceleration",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    action_rate = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "action_rate",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )

    # Style constraints
    hip_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "hip_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    base_orientation = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "base_orientation",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    air_time = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "air_time",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    two_foot_contact = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "two_foot_contact",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )


##
# Environment configuration
##


@configclass
class Solo12FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0

        # simulation settings
        self.sim.solver_type = 0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.max_position_iteration_count = 4
        self.sim.max_velocity_iteration_count = 1
        self.sim.bounce_threshold_velocity = 0.2
        self.sim.gpu_max_rigid_contact_count = 33554432
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class Solo12FlatEnvCfg_PLAY(Solo12FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # set velocity command
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.7, 0.7)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.78, 0.78)
