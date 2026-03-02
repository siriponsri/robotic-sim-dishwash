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

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.unitree_g1.g1 import UnitreeG1
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig

# ---------------------------------------------------------------------------
# Constants (exported for notebooks)
# ---------------------------------------------------------------------------
APPLE_RADIUS = 0.04                                # 4 cm
APPLE_INIT_POS = (0.25, 0.10, 0.82)               # on counter
BOWL_POS = (0.05, 0.30, 0.80)                      # bowl on counter
BOWL_RADIUS = 0.08
BOWL_HEIGHT = 0.06

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

    SUPPORTED_ROBOTS = ["unitree_g1"]
    agent: UnitreeG1

    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]

    def __init__(
        self,
        *args,
        robot_init_qpos_noise: float = 0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._prev_actions: torch.Tensor | None = None

        # Robot init pose — full-body G1 stands on ground (z=0)
        self._robot_init_pose = sapien.Pose(p=[0, 0, 0])

        super().__init__(*args, robot_uids="unitree_g1", **kwargs)

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
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        # Kitchen counter
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=KITCHEN_SCENE_SCALE)

        # Apple (dynamic sphere)
        builder = self.scene.create_actor_builder()
        builder.add_sphere_collision(radius=APPLE_RADIUS, density=500)
        builder.add_sphere_visual(radius=APPLE_RADIUS, material=sapien.render.RenderMaterial(
            base_color=[0.8, 0.1, 0.1, 1.0],  # red apple
        ))
        self.apple = builder.build("apple")

        # Bowl (kinematic)
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(radius=BOWL_RADIUS, half_length=BOWL_HEIGHT / 2)
        builder.add_cylinder_visual(
            radius=BOWL_RADIUS, half_length=BOWL_HEIGHT / 2,
            material=sapien.render.RenderMaterial(base_color=[0.6, 0.6, 0.3, 1.0]),
        )
        self.bowl = builder.build_kinematic("bowl")

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

        # Apple position
        apple_pos = torch.tensor(
            [APPLE_INIT_POS] * b, dtype=torch.float32, device=self.device
        )
        apple_pos[:, :2] += torch.randn(b, 2, device=self.device) * 0.02
        self.apple.set_pose(sapien.Pose(p=apple_pos[0].cpu().numpy()))

        # Bowl position (fixed)
        self.bowl.set_pose(sapien.Pose(p=list(BOWL_POS)))

        # Robot init qpos with noise
        qpos_np = self.agent.keyframes["standing"].qpos
        qpos = torch.tensor(qpos_np, dtype=torch.float32, device=self.device)
        if self.robot_init_qpos_noise > 0:
            qpos = qpos + torch.randn_like(qpos) * self.robot_init_qpos_noise
        self.agent.robot.set_qpos(qpos)
        self.agent.robot.set_pose(self._robot_init_pose)

        # State tracking
        self._prev_actions = torch.zeros(
            b, self.agent.action_space.shape[0],
            dtype=torch.float32, device=self.device,
        )
        self._apple_init_z = apple_pos[:, 2].clone()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_obs_extra(self, info: dict) -> dict:
        tcp_pose = self._tcp_link.pose
        tcp_pos = tcp_pose.p
        apple_pos = self.apple.pose.p
        bowl_pos = torch.tensor(BOWL_POS, device=self.device).unsqueeze(0)

        tcp_to_apple = apple_pos - tcp_pos
        apple_to_bowl = bowl_pos - apple_pos

        # Check if grasped (apple close to TCP)
        dist_tcp_apple = torch.norm(tcp_to_apple, dim=-1, keepdim=True)
        is_grasped = (dist_tcp_apple < GRASP_THRESHOLD).float()

        return {
            "tcp_pose": torch.cat([tcp_pos, tcp_pose.q], dim=-1),
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
        bowl_centre = torch.tensor(BOWL_POS, device=self.device).unsqueeze(0)

        dist_apple_bowl = torch.norm(apple_pos[:, :2] - bowl_centre[:, :2], dim=-1)
        height_ok = apple_pos[:, 2] < (bowl_centre[:, 2] + BOWL_HEIGHT)
        in_bowl = (dist_apple_bowl < BOWL_RADIUS * 0.8) & height_ok

        return {
            "success": in_bowl,
            "dist_apple_bowl": dist_apple_bowl,
            "apple_height": apple_pos[:, 2],
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        tcp_pos = self._tcp_link.pose.p
        apple_pos = self.apple.pose.p
        bowl_pos = torch.tensor(BOWL_POS, device=self.device).unsqueeze(0)

        # Stage 0: Reaching
        dist_reach = torch.norm(tcp_pos - apple_pos, dim=-1)
        r_reach = W_REACH * (1.0 - torch.tanh(5.0 * dist_reach))

        # Stage 1: Grasp reward
        is_grasped = (dist_reach < GRASP_THRESHOLD).float()
        r_grasp = W_GRASP * is_grasped

        # Stage 2: Placing
        dist_place = torch.norm(apple_pos[:, :2] - bowl_pos[:, :2], dim=-1)
        r_place = W_PLACE * is_grasped * (1.0 - torch.tanh(5.0 * dist_place))

        # Success bonus
        r_success = info["success"].float() * SUCCESS_BONUS

        # Penalties
        r_time = -W_TIME * torch.ones(action.shape[0], device=self.device)
        jerk = action - self._prev_actions
        r_jerk = -W_JERK * torch.sum(jerk ** 2, dim=-1)
        r_act = -W_ACT * torch.sum(action ** 2, dim=-1)

        self._prev_actions = action.clone()

        return r_reach + r_grasp + r_place + r_success + r_time + r_jerk + r_act

    def compute_normalized_dense_reward(self, obs, action, info) -> torch.Tensor:
        max_rew = W_REACH + W_GRASP + W_PLACE + SUCCESS_BONUS
        return self.compute_dense_reward(obs, action, info) / max_rew
