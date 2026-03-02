"""UnitreeG1PlaceAppleInBowlFullBody-v1 — Custom ManiSkill environment.

Scene
-----
- Kitchen counter (via ``KitchenCounterSceneBuilder``)
- An **apple** (dynamic actor, sphere ~4 cm radius)
- A **bowl** (kinematic actor) placed on the counter
- Unitree G1 **full-body** robot (free-floating root, 37 DOF)

Task
----
The robot must **pick up the apple** from the counter and **place it in the
bowl**.  The full-body variant (37 DOF) requires the robot to maintain
balance while reaching, grasping, lifting, and placing.

Reward (dense, staged)
----------------------
Stage 0 — Reaching (0–1):
    ``w_reach * (1 - tanh(5 * dist(tcp, apple)))``
Stage 1 — Grasping (+1 when apple lifted):
    ``w_grasp * is_grasped``
Stage 2 — Placing (0–1 as apple approaches bowl centre):
    ``w_place * (1 - tanh(5 * dist(apple, bowl_centre)))``
Stage 3 — Success bonus (one-shot):
    ``+ SUCCESS_BONUS  when apple inside bowl``
Penalties:
    ``- w_time``  (per step)
    ``- w_jerk * ||a_t - a_{t-1}||^2``
    ``- w_act  * ||a_t||^2``

Observations (state mode)
-------------------------
Base robot state (auto from ManiSkill) + extras:
    - ``tcp_pose``       : Right TCP pose (7D)
    - ``apple_pos``      : Apple position (3D)
    - ``bowl_pos``       : Bowl centre (3D)
    - ``tcp_to_apple``   : Vector TCP -> apple (3D)
    - ``apple_to_bowl``  : Vector apple -> bowl (3D)
    - ``is_grasped``     : Binary, apple in hand (1D)

Termination
-----------
- ``success``: apple is inside the bowl (height & distance checks)
- Episode truncation at ``max_episode_steps`` (1000)
"""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill
from mani_skill.agents.robots.unitree_g1.g1 import UnitreeG1
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


# ---------------------------------------------------------------------------
# Custom agent: full-body G1 with **fixed root** so it can stand reliably.
#
# Why? The stock UnitreeG1 has fix_root_link=False (free-floating).  With the
# default low PD gains (kp=50) the robot collapses under gravity even before
# any action is taken — standing itself is an RL problem.  The official
# manipulation env (UnitreeG1PlaceAppleInBowl-v1) solves this by using the
# upper-body variant with fix_root_link=True.
#
# Our variant keeps **all 37 DOF** (legs + torso + arms + hands) controllable
# but fixes the pelvis in space so baselines are meaningful.  Gravity
# compensation (balance_passive_force=True) ensures zero-action = hold pose.
# ---------------------------------------------------------------------------
@register_agent()
class UnitreeG1FullBodyFixed(UnitreeG1):
    """Full-body G1 with fixed root for reliable standing."""
    uid = "unitree_g1_fullbody_fixed"
    fix_root_link = True                   # pelvis pinned in space
    body_stiffness = 1e3                   # match upper-body variant
    body_damping = 1e2                     # match upper-body variant
    body_force_limit = 100                 # gravity-compensated → less torque needed

    # Finger friction for better grasping (same as upper-body variant)
    urdf_config = dict(
        _materials=dict(
            finger=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            **{f"left_{k}_link": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1)
               for k in ["one", "two", "three", "four", "five", "six"]},
            **{f"right_{k}_link": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1)
               for k in ["one", "two", "three", "four", "five", "six"]},
            "left_palm_link": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_palm_link": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
        },
    )

    @property
    def _controller_configs(self):
        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=None, upper=None,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            normalize_action=False,
        )
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=-0.2, upper=0.2,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            use_delta=True,
        )
        # balance_passive_force=True → gravity compensation at joints
        # (safe for fix_root_link=True; the root is fixed so gravity
        #  only affects joint-level torques, not the base)
        return dict(
            pd_joint_pos=dict(body=body_pd_joint_pos, balance_passive_force=True),
            pd_joint_delta_pos=dict(body=body_pd_joint_delta_pos, balance_passive_force=True),
        )

# Path to built-in humanoid task assets (bowl mesh, apple mesh, etc.)
_HUMANOID_ASSETS_DIR = os.path.join(
    os.path.dirname(mani_skill.__file__),
    "envs", "tasks", "humanoid", "assets",
)

# ---------------------------------------------------------------------------
# Constants (exported for notebooks)
# ---------------------------------------------------------------------------
APPLE_INIT_POS = (0.0, 0.0, 0.7335)               # on counter (matches built-in)
BOWL_POS = (0.0, -0.4, 0.753)                      # bowl on counter side (matches built-in)
SUCCESS_DIST = 0.05                                # m — apple-bowl 3D dist for success

# Reward weights
W_REACH = 1.0
W_GRASP = 2.0
W_PLACE = 5.0
W_TIME = 0.01
W_JERK = 0.05
W_ACT = 0.005
SUCCESS_BONUS = 50.0

# Safety
GRASP_THRESHOLD = 0.01          # m — apple-TCP distance to count as grasped

# Scene scale
KITCHEN_SCENE_SCALE = 0.82


@register_env(
    "UnitreeG1PlaceAppleInBowlFullBody-v1",
    max_episode_steps=1000,
    asset_download_ids=[],
)
class UnitreeG1PlaceAppleInBowlFullBodyEnv(BaseEnv):
    """Full-body G1 (37 DOF) apple-in-bowl task."""

    SUPPORTED_ROBOTS = ["unitree_g1_fullbody_fixed"]
    agent: UnitreeG1FullBodyFixed

    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]

    def __init__(
        self,
        *args,
        robot_init_qpos_noise: float = 0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._prev_actions: torch.Tensor | None = None

        # Robot init pose — standing, offset from counter (matches built-in)
        self._robot_init_pose = copy.deepcopy(UnitreeG1FullBodyFixed.keyframes["standing"].pose)
        self._robot_init_pose.p = [-0.3, 0, 0.755]

        super().__init__(*args, robot_uids="unitree_g1_fullbody_fixed", **kwargs)

    # ------------------------------------------------------------------
    # Sim config
    # ------------------------------------------------------------------
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2 ** 22,
                max_rigid_patch_count=2 ** 21,
            ),
            scene_config=SceneConfig(contact_offset=0.01),
        )

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 1.2, 1.2], [-0.15, 0.0, 0.4])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _load_agent(self, options: dict):
        super()._load_agent(options, self._robot_init_pose)

    def _load_scene(self, options: dict):
        # Kitchen counter (includes ground plane with collision)
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=KITCHEN_SCENE_SCALE)

        scale = KITCHEN_SCENE_SCALE
        fix_rotation_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))

        # Bowl (kinematic — realistic mesh from built-in assets)
        builder = self.scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(
            filename=os.path.join(_HUMANOID_ASSETS_DIR, "frl_apartment_bowl_07.ply"),
            pose=fix_rotation_pose,
            scale=[scale] * 3,
        )
        builder.add_visual_from_file(
            filename=os.path.join(_HUMANOID_ASSETS_DIR, "frl_apartment_bowl_07.glb"),
            scale=[scale] * 3,
            pose=fix_rotation_pose,
        )
        builder.initial_pose = sapien.Pose(p=list(BOWL_POS))
        self.bowl = builder.build_kinematic(name="bowl")

        # Apple (dynamic — realistic mesh from built-in assets)
        builder = self.scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            filename=os.path.join(_HUMANOID_ASSETS_DIR, "apple_1.ply"),
            pose=fix_rotation_pose,
            scale=[scale * 0.8] * 3,
        )
        builder.add_visual_from_file(
            filename=os.path.join(_HUMANOID_ASSETS_DIR, "apple_1.glb"),
            scale=[scale * 0.8] * 3,
            pose=fix_rotation_pose,
        )
        builder.initial_pose = sapien.Pose(p=list(APPLE_INIT_POS))
        self.apple = builder.build(name="apple")

        # TCP link reference (right hand)
        self._tcp_link = sapien_utils.get_obj_by_name(
            self.agent.robot.get_links(), "right_palm_link"
        )

    # ------------------------------------------------------------------
    # Episode init
    # ------------------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        self.scene_builder.initialize(env_idx)

        # Robot init qpos with noise
        qpos_np = self.agent.keyframes["standing"].qpos
        qpos = torch.tensor(qpos_np, dtype=torch.float32, device=self.device)
        if self.robot_init_qpos_noise > 0:
            qpos = qpos + torch.randn_like(qpos) * self.robot_init_qpos_noise
        self.agent.robot.set_qpos(qpos)
        self.agent.robot.set_qvel(torch.zeros_like(qpos))   # zero velocity!
        self.agent.robot.set_pose(self._robot_init_pose)

        # Apple position — small random offset on counter
        apple_xyz = torch.zeros((b, 3), device=self.device)
        apple_xyz[:, :2] = (torch.rand(b, 2, device=self.device) - 0.5) * 0.05
        apple_xyz[:, 2] = APPLE_INIT_POS[2]
        self.apple.set_pose(sapien.Pose(p=apple_xyz[0].cpu().numpy()))

        # Bowl position — small random offset
        bowl_xyz = torch.zeros((b, 3), device=self.device)
        bowl_xyz[:, :2] = (torch.rand(b, 2, device=self.device) - 0.5) * 0.05
        bowl_xyz[:, 0] += BOWL_POS[0]
        bowl_xyz[:, 1] += BOWL_POS[1]
        bowl_xyz[:, 2] = BOWL_POS[2]
        self.bowl.set_pose(sapien.Pose(p=bowl_xyz[0].cpu().numpy()))

        # State tracking
        self._prev_actions = torch.zeros(
            b, self.agent.action_space.shape[0],
            dtype=torch.float32, device=self.device,
        )
        self._apple_init_z = apple_xyz[:, 2].clone()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if isinstance(obs, torch.Tensor):
            obs = torch.nan_to_num(obs, nan=0.0)
        return obs, reward, terminated, truncated, info

    def _get_obs_extra(self, info: dict) -> dict:
        tcp_pose = self._tcp_link.pose
        tcp_pos = torch.nan_to_num(tcp_pose.p, nan=0.0)
        tcp_q = torch.nan_to_num(tcp_pose.q, nan=0.0)
        apple_pos = torch.nan_to_num(self.apple.pose.p, nan=0.0)
        bowl_pos = torch.nan_to_num(self.bowl.pose.p, nan=0.0)

        tcp_to_apple = apple_pos - tcp_pos
        apple_to_bowl = bowl_pos - apple_pos

        # Check if grasped (apple close to TCP)
        dist_tcp_apple = torch.norm(tcp_to_apple, dim=-1, keepdim=True)
        is_grasped = (dist_tcp_apple < GRASP_THRESHOLD).float()

        return {
            "tcp_pose": torch.cat([tcp_pos, tcp_q], dim=-1),
            "apple_pos": apple_pos,
            "bowl_pos": bowl_pos.expand_as(apple_pos),
            "tcp_to_apple": tcp_to_apple,
            "apple_to_bowl": apple_to_bowl,
            "is_grasped": is_grasped,
        }

    # ------------------------------------------------------------------
    # Success / Failure evaluation
    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        apple_pos = self.apple.pose.p
        bowl_pos = self.bowl.pose.p

        # Guard against NaN (robot/apple flew off or simulation exploded)
        has_nan = torch.isnan(apple_pos).any(dim=-1) | torch.isnan(bowl_pos).any(dim=-1)
        apple_safe = torch.nan_to_num(apple_pos, nan=999.0)
        bowl_safe = torch.nan_to_num(bowl_pos, nan=0.0)

        dist_apple_bowl = torch.linalg.norm(apple_safe - bowl_safe, dim=-1)
        in_bowl = (dist_apple_bowl <= SUCCESS_DIST) & ~has_nan

        return {
            "success": in_bowl,
            "fail": has_nan,
            "dist_apple_bowl": dist_apple_bowl,
            "apple_height": apple_safe[:, 2],
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        tcp_pos = self._tcp_link.pose.p
        apple_pos = self.apple.pose.p
        bowl_pos = torch.nan_to_num(self.bowl.pose.p, nan=0.0)

        # Guard against NaN (simulation instability)
        has_nan = torch.isnan(tcp_pos).any(dim=-1) | torch.isnan(apple_pos).any(dim=-1)

        tcp_pos = torch.nan_to_num(tcp_pos, nan=0.0)
        apple_pos = torch.nan_to_num(apple_pos, nan=999.0)

        # Stage 0: Reaching
        dist_reach = torch.norm(tcp_pos - apple_pos, dim=-1)
        r_reach = W_REACH * (1.0 - torch.tanh(5.0 * dist_reach))

        # Stage 1: Grasp reward
        is_grasped = (dist_reach < GRASP_THRESHOLD).float()
        r_grasp = W_GRASP * is_grasped

        # Stage 2: Placing — target above bowl centre to encourage lifting
        goal_pos = bowl_pos + torch.tensor([0, 0, 0.15], device=self.device)
        dist_place = torch.norm(apple_pos - goal_pos, dim=-1)
        r_place = W_PLACE * is_grasped * (1.0 - torch.tanh(5.0 * dist_place))

        # Success bonus
        r_success = info["success"].float() * SUCCESS_BONUS

        # Penalties
        r_time = -W_TIME * torch.ones(action.shape[0], device=self.device)
        jerk = action - self._prev_actions
        r_jerk = -W_JERK * torch.sum(jerk ** 2, dim=-1)
        r_act = -W_ACT * torch.sum(action ** 2, dim=-1)

        self._prev_actions = action.clone()

        reward = r_reach + r_grasp + r_place + r_success + r_time + r_jerk + r_act
        # Return large penalty for NaN states instead of NaN
        reward = torch.where(has_nan, torch.tensor(-1.0, device=self.device), reward)
        return reward

    def compute_normalized_dense_reward(self, obs, action, info) -> torch.Tensor:
        max_rew = W_REACH + W_GRASP + W_PLACE + SUCCESS_BONUS
        return self.compute_dense_reward(obs, action, info) / max_rew
