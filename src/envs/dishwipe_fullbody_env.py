"""UnitreeG1DishWipeFullBody-v1 — Custom ManiSkill environment.

Scene
-----
- Kitchen counter with sink (via ``KitchenCounterSceneBuilder``)
- A flat rectangular **plate** placed **inside the kitchen sink** (kinematic actor)
- Unitree G1 **full-body** robot (free-floating root, 37 DOF)

Task
----
The robot must wipe the plate using its **left palm** by sweeping across
the surface.  A ``VirtualDirtGrid`` (10 × 10) tracks cleaning progress.
Cells are cleaned when *any* left-hand link (palm + fingers) contacts the
plate above them.

This is the **full-body variant** (37 DOF) of the original DishWipe env.
The robot must maintain balance while performing the wiping task.

Contact detection
-----------------
Multi-link: ``left_palm_link`` + ``left_two_link`` + ``left_four_link``
+ ``left_six_link``.

Reward (dense, staged)
----------------------
Same reward structure as the upper-body variant:
    Stage 0 — Reaching, Stage 1 — Contact, Stage 2 — Cleaning,
    Stage 3 — Sweep bonus.  Penalties for time, jerk, action norm, force.
    Success bonus when cleaned_ratio >= 0.95.

Observations (state mode)
-------------------------
Base robot state + extras:
    - ``tcp_pose``       : Left TCP pose (7D)
    - ``palm_pos``       : Left palm position (3D)
    - ``plate_pos``      : Plate centre (3D)
    - ``palm_to_plate``  : Vector palm -> plate (3D)
    - ``contact_force``  : Multi-link force magnitude (1D)
    - ``cleaned_ratio``  : Fraction cleaned (1D)
    - ``dirt_grid``      : Flat 10×10 grid (100D)
"""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

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
# Custom agent: full-body G1 with fixed root (see apple_fullbody_env.py)
# ---------------------------------------------------------------------------
@register_agent()
class UnitreeG1FullBodyFixedDW(UnitreeG1):
    """Full-body G1 with fixed root for reliable standing (DishWipe)."""
    uid = "unitree_g1_fullbody_fixed_dw"
    fix_root_link = True
    body_stiffness = 1e3
    body_damping = 1e2
    body_force_limit = 100

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
            self.body_joints, lower=None, upper=None,
            stiffness=self.body_stiffness, damping=self.body_damping,
            force_limit=self.body_force_limit, normalize_action=False,
        )
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joints, lower=-0.2, upper=0.2,
            stiffness=self.body_stiffness, damping=self.body_damping,
            force_limit=self.body_force_limit, use_delta=True,
        )
        return dict(
            pd_joint_pos=dict(body=body_pd_joint_pos, balance_passive_force=True),
            pd_joint_delta_pos=dict(body=body_pd_joint_delta_pos, balance_passive_force=True),
        )

from src.envs.dirt_grid import VirtualDirtGrid

# ---------------------------------------------------------------------------
# Constants (same as upper-body variant for consistency)
# ---------------------------------------------------------------------------
PLATE_RADIUS = 0.12                              # plate radius (disc shape)
PLATE_HALF_THICKNESS = 0.004                       # half-thickness (thin disc)
PLATE_HALF_SIZE = (PLATE_RADIUS, PLATE_RADIUS, PLATE_HALF_THICKNESS)  # for dirt-grid compat
PLATE_POS_IN_SINK = (0.15, 0.25, 0.59)            # plate inside kitchen sink basin

GRID_H, GRID_W = 10, 10
BRUSH_RADIUS = 1

# Reward weights
W_CLEAN   = 10.0
W_REACH   = 0.5
W_CONTACT = 1.0
W_SWEEP   = 0.3
W_TIME    = 0.01
W_JERK    = 0.05
W_ACT     = 0.005
W_FORCE   = 0.01
SUCCESS_BONUS = 50.0

# Safety / termination
FZ_SOFT   = 50.0
FZ_HARD   = 200.0
SUCCESS_CLEAN_RATIO = 0.95
CONTACT_THRESHOLD   = 0.5

KITCHEN_SCENE_SCALE = 0.82

_LEFT_CONTACT_LINKS = (
    "left_palm_link",
    "left_two_link",
    "left_four_link",
    "left_six_link",
)


@register_env(
    "UnitreeG1DishWipeFullBody-v1",
    max_episode_steps=1000,
    asset_download_ids=[],
)
class UnitreeG1DishWipeFullBodyEnv(BaseEnv):
    """Full-body G1 (37 DOF) dish-wiping task with VirtualDirtGrid."""

    SUPPORTED_ROBOTS = ["unitree_g1_fullbody_fixed_dw"]
    agent: UnitreeG1FullBodyFixedDW

    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]

    def __init__(
        self,
        *args,
        robot_init_qpos_noise: float = 0.02,
        grid_h: int = GRID_H,
        grid_w: int = GRID_W,
        brush_radius: int = BRUSH_RADIUS,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.brush_radius = brush_radius

        self._dirt_grids: list[VirtualDirtGrid] = []
        self._prev_actions: torch.Tensor | None = None
        self._prev_palm_pos: torch.Tensor | None = None
        self._success_bonus_given: torch.Tensor | None = None
        self._contact_links: list = []

        # Robot init pose — standing, offset from counter (matches built-in)
        self._robot_init_pose = copy.deepcopy(UnitreeG1FullBodyFixedDW.keyframes["standing"].pose)
        self._robot_init_pose.p = [-0.3, 0, 0.755]

        super().__init__(*args, robot_uids="unitree_g1_fullbody_fixed_dw", **kwargs)

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

        # Plate (kinematic cylinder disc — more realistic than flat box)
        builder = self.scene.create_actor_builder()
        flat_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))  # Y-axis → Z-axis
        builder.add_cylinder_collision(
            radius=PLATE_RADIUS, half_length=PLATE_HALF_THICKNESS,
            pose=flat_pose,
        )
        builder.add_cylinder_visual(
            radius=PLATE_RADIUS, half_length=PLATE_HALF_THICKNESS,
            material=sapien.render.RenderMaterial(
                base_color=[0.95, 0.93, 0.88, 1.0],  # off-white ceramic
            ),
            pose=flat_pose,
        )
        builder.initial_pose = sapien.Pose(p=list(PLATE_POS_IN_SINK))
        self.plate = builder.build_kinematic("plate")

        # Contact link references
        all_links = self.agent.robot.get_links()
        self._contact_links = [
            sapien_utils.get_obj_by_name(all_links, name)
            for name in _LEFT_CONTACT_LINKS
        ]
        self._palm_link = self._contact_links[0]

        # Dirt grids (one per parallel env)
        self._dirt_grids = [
            VirtualDirtGrid(
                H=self.grid_h, W=self.grid_w, brush_radius=self.brush_radius,
            )
            for _ in range(self.num_envs)
        ]

    # ------------------------------------------------------------------
    # Episode init
    # ------------------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        self.scene_builder.initialize(env_idx)

        # Plate position
        self.plate.set_pose(sapien.Pose(p=list(PLATE_POS_IN_SINK)))

        # Robot init qpos with noise
        qpos_np = self.agent.keyframes["standing"].qpos
        qpos = torch.tensor(qpos_np, dtype=torch.float32, device=self.device)
        if self.robot_init_qpos_noise > 0:
            qpos = qpos + torch.randn_like(qpos) * self.robot_init_qpos_noise
        self.agent.robot.set_qpos(qpos)
        self.agent.robot.set_qvel(torch.zeros_like(qpos))   # zero velocity!
        self.agent.robot.set_pose(self._robot_init_pose)

        # Reset dirt grids for these envs
        for idx in env_idx.cpu().tolist():
            self._dirt_grids[idx].reset()

        # State tracking
        act_dim = self.agent.action_space.shape[0]
        self._prev_actions = torch.zeros(b, act_dim, dtype=torch.float32, device=self.device)
        self._prev_palm_pos = self._palm_link.pose.p.clone()
        self._success_bonus_given = torch.zeros(b, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Contact force
    # ------------------------------------------------------------------
    def _get_contact_force_magnitude(self) -> torch.Tensor:
        """Total contact force magnitude from all left-hand links on plate."""
        total = torch.zeros(self.num_envs, device=self.device)
        contacts = self.scene.get_contacts()
        plate_entity = self.plate._objs[0] if hasattr(self.plate, '_objs') else self.plate
        for link in self._contact_links:
            link_entity = link._objs[0] if hasattr(link, '_objs') else link
            for c in contacts:
                if ({c.bodies[0], c.bodies[1]} == {link_entity, plate_entity}):
                    for p in c.points:
                        f = np.linalg.norm(p.force)
                        total[0] += f
        return total

    # ------------------------------------------------------------------
    # Dirt grid update
    # ------------------------------------------------------------------
    def _update_dirt_grid(self) -> torch.Tensor:
        """Update dirt grid based on palm-plate contact. Returns delta cleaned."""
        palm_pos = self._palm_link.pose.p
        plate_pos = torch.tensor(PLATE_POS_IN_SINK, device=self.device)
        plate_hs = torch.tensor(PLATE_HALF_SIZE[:2], device=self.device)

        delta = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            rel = palm_pos[i, :2] - (plate_pos[:2] - plate_hs)
            norm = rel / (2.0 * plate_hs)
            # Guard against NaN (e.g. robot has fallen, invalid pose)
            if torch.isnan(norm).any():
                continue
            gx = int(torch.clamp(norm[0] * self.grid_w, 0, self.grid_w - 1).item())
            gy = int(torch.clamp(norm[1] * self.grid_h, 0, self.grid_h - 1).item())
            prev_ratio = self._dirt_grids[i].get_cleaned_ratio()
            self._dirt_grids[i].mark_clean(gx, gy)
            delta[i] = self._dirt_grids[i].get_cleaned_ratio() - prev_ratio
        return delta

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if isinstance(obs, torch.Tensor):
            obs = torch.nan_to_num(obs, nan=0.0)
        return obs, reward, terminated, truncated, info

    def _get_obs_extra(self, info: dict) -> dict:
        palm_pos = torch.nan_to_num(self._palm_link.pose.p, nan=0.0)
        plate_pos = torch.tensor(PLATE_POS_IN_SINK, device=self.device).unsqueeze(0)
        palm_to_plate = plate_pos - palm_pos

        contact_force = self._get_contact_force_magnitude().unsqueeze(-1)
        cleaned = torch.tensor(
            [g.get_cleaned_ratio() for g in self._dirt_grids],
            device=self.device, dtype=torch.float32,
        ).unsqueeze(-1)
        dirt_flat = torch.tensor(
            np.stack([g.get_grid_flat() for g in self._dirt_grids]),
            device=self.device, dtype=torch.float32,
        )

        tcp_pose = self._palm_link.pose
        tcp_p = torch.nan_to_num(tcp_pose.p, nan=0.0)
        tcp_q = torch.nan_to_num(tcp_pose.q, nan=0.0)
        return {
            "tcp_pose": torch.cat([tcp_p, tcp_q], dim=-1),
            "palm_pos": palm_pos,
            "plate_pos": plate_pos.expand_as(palm_pos),
            "palm_to_plate": palm_to_plate,
            "contact_force": contact_force,
            "cleaned_ratio": cleaned,
            "dirt_grid": dirt_flat,
        }

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        ratios = torch.tensor(
            [g.get_cleaned_ratio() for g in self._dirt_grids],
            device=self.device, dtype=torch.float32,
        )
        force_mag = self._get_contact_force_magnitude()
        success = ratios >= SUCCESS_CLEAN_RATIO
        fail = force_mag > FZ_HARD

        # Detect NaN in palm position (simulation instability)
        palm_pos = self._palm_link.pose.p
        has_nan = torch.isnan(palm_pos).any(dim=-1)
        fail = fail | has_nan

        return {
            "success": success & ~has_nan,
            "fail": fail,
            "cleaned_ratio": ratios,
            "contact_force_mag": force_mag,
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        palm_pos = self._palm_link.pose.p
        plate_pos = torch.tensor(PLATE_POS_IN_SINK, device=self.device).unsqueeze(0)

        # Guard against NaN (simulation instability)
        has_nan = torch.isnan(palm_pos).any(dim=-1)
        palm_pos = torch.nan_to_num(palm_pos, nan=0.0)

        # Stage 0: Reaching
        dist_reach = torch.norm(palm_pos - plate_pos, dim=-1)
        r_reach = W_REACH * (1.0 - torch.tanh(5.0 * dist_reach))

        # Stage 1: Contact
        force_mag = self._get_contact_force_magnitude()
        is_contact = (force_mag > CONTACT_THRESHOLD).float()
        r_contact = W_CONTACT * is_contact

        # Stage 2: Cleaning
        delta_clean = self._update_dirt_grid()
        r_clean = W_CLEAN * delta_clean

        # Stage 3: Sweep bonus (lateral velocity while in contact)
        r_sweep = torch.zeros(self.num_envs, device=self.device)
        if self._prev_palm_pos is not None:
            prev_safe = torch.nan_to_num(self._prev_palm_pos, nan=0.0)
            lateral_vel = torch.norm(palm_pos[:, :2] - prev_safe[:, :2], dim=-1)
            r_sweep = W_SWEEP * is_contact * lateral_vel

        # Success bonus (one-shot)
        success = info.get("success", torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
        if self._success_bonus_given is not None:
            new_success = success & ~self._success_bonus_given
            r_success = new_success.float() * SUCCESS_BONUS
            self._success_bonus_given = self._success_bonus_given | success
        else:
            r_success = torch.zeros(self.num_envs, device=self.device)

        # Penalties
        r_time = -W_TIME * torch.ones(self.num_envs, device=self.device)
        jerk = action - self._prev_actions
        r_jerk = -W_JERK * torch.sum(jerk ** 2, dim=-1)
        r_act = -W_ACT * torch.sum(action ** 2, dim=-1)
        excess_force = torch.clamp(force_mag - FZ_SOFT, min=0)
        r_force = -W_FORCE * excess_force

        # Update state
        self._prev_actions = action.clone()
        self._prev_palm_pos = palm_pos.clone()

        reward = r_reach + r_contact + r_clean + r_sweep + r_success + r_time + r_jerk + r_act + r_force
        # Return large penalty for NaN states instead of NaN
        reward = torch.where(has_nan, torch.tensor(-1.0, device=self.device), reward)
        return reward

    def compute_normalized_dense_reward(self, obs, action, info) -> torch.Tensor:
        max_rew = W_REACH + W_CONTACT + W_CLEAN * self.grid_h * self.grid_w + SUCCESS_BONUS
        return self.compute_dense_reward(obs, action, info) / max_rew
