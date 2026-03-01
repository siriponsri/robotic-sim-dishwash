"""UnitreeG1DishWipe-v1 — Production-scale custom ManiSkill environment.

Scene
-----
- Kitchen counter with sink (via ``KitchenCounterSceneBuilder``)
- A flat rectangular **plate** placed **inside the kitchen sink** (kinematic actor)
- Unitree G1 upper-body robot (fixed root, 25 DOF)

Task
----
The robot must wipe the plate using its **left palm** by sweeping across
the surface.  A ``VirtualDirtGrid`` (10 x 10) tracks cleaning progress.
Cells are cleaned when *any* left-hand link (palm + fingers) contacts the
plate above them.

Contact detection
-----------------
Multi-link: ``left_palm_link`` + ``left_two_link`` + ``left_four_link``
+ ``left_six_link``.  The *centroid* of all contacting links is used for
dirt-grid mapping, ensuring the cleaned cell always matches the actual
contact location (fixes TCP-vs-palm offset problem).

Reward (dense, staged)
----------------------
Stage 0 -- Reaching (0-1):
    ``w_reach * (1 - tanh(5 * dist(palm, plate)))``
Stage 1 -- Contact (+1 when touching):
    ``w_contact * is_contacting``
Stage 2 -- Cleaning (0-+10 per new cell):
    ``w_clean * delta_clean``
Stage 3 -- Sweep bonus (encouraging lateral motion):
    ``w_sweep * lateral_vel`` (reward moving hand sideways while in contact)
Penalties:
    ``- w_time``
    ``- w_jerk * ||a_t - a_{t-1}||^2``
    ``- w_act  * ||a_t||^2``
    ``- w_force* max(0, F_contact - F_soft)``

.. note::
    ``F_contact`` is the **magnitude** (L2 norm) of multi-link contact
    forces, not a normal-axis component.  Because the plate is horizontal,
    magnitude ≈ Fz in practice but also captures lateral shear.
Success bonus:
    ``+ SUCCESS_BONUS`` (one-shot when ratio >= SUCCESS_CLEAN_RATIO)

Observations (state mode)
-------------------------
Base robot state (auto from ManiSkill) + extras:
    - ``tcp_pose``       : Left TCP pose (7D)
    - ``palm_pos``       : Left palm position (3D)
    - ``plate_pos``      : Plate centre (3D)
    - ``palm_to_plate``  : Vector palm -> plate (3D)
    - ``contact_force``  : Multi-link force magnitude (1D)
    - ``cleaned_ratio``  : Fraction cleaned (1D)
    - ``dirt_grid``      : Flat 10x10 grid (100D) — gives policy spatial info

Termination
-----------
- ``success``: cleaned_ratio >= SUCCESS_CLEAN_RATIO
- ``fail``: total contact force exceeds FZ_HARD
- Episode truncation at ``max_episode_steps`` (1000)
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.unitree_g1.g1_upper_body import UnitreeG1UpperBody
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig

from src.envs.dirt_grid import VirtualDirtGrid

# ---------------------------------------------------------------------------
# Constants (exported for notebooks)
# ---------------------------------------------------------------------------
PLATE_HALF_SIZE = (0.10, 0.10, 0.003)          # half-extents XYZ
PLATE_POS_IN_SINK = (0.10, 0.20, 0.58)         # plate inside kitchen sink basin
# Sink basin at scale=0.82: x∈[-0.01,0.32], y∈[0.04,0.49], z∈[0.56,0.79]
# Deprecated alias — kept for backward compat, will be removed
PLATE_POS_ON_COUNTER = PLATE_POS_IN_SINK

GRID_H, GRID_W = 10, 10
BRUSH_RADIUS = 1                                 # 3x3 cleaning footprint

# Reward weights
W_CLEAN   = 10.0      # per newly-cleaned cell
W_REACH   = 0.5       # shaped reaching -> plate (0-1)
W_CONTACT = 1.0       # bonus per step while contacting
W_SWEEP   = 0.3       # lateral velocity bonus during contact
W_TIME    = 0.01      # penalty per step
W_JERK    = 0.05      # ||a_t - a_{t-1}||^2
W_ACT     = 0.005     # ||a_t||^2
W_FORCE   = 0.01      # excess force above soft threshold
SUCCESS_BONUS = 50.0   # one-shot

# Safety / termination  (thresholds on force *magnitude*, not Fz component)
FZ_SOFT   = 50.0       # N - soft penalty begins  (magnitude ≈ Fz for flat plate)
FZ_HARD   = 200.0      # N - hard termination
SUCCESS_CLEAN_RATIO = 0.95
CONTACT_THRESHOLD   = 0.5   # N - minimum to count as "contact"

# Scene scale (matches PlaceAppleInBowl)
KITCHEN_SCENE_SCALE = 0.82

# Left-hand links used for multi-link contact sensing
_LEFT_CONTACT_LINKS = (
    "left_palm_link",
    "left_two_link",       # finger L1
    "left_four_link",      # finger R1
    "left_six_link",       # finger R2
)


class UnitreeG1DishWipeEnvBase(BaseEnv):
    """Base class -- kitchen scene + plate + multi-link contact + dirt grid."""

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

        # Populated during _load_scene / _initialize_episode
        self._dirt_grids: list[VirtualDirtGrid] = []
        self._prev_actions: torch.Tensor | None = None
        self._prev_palm_pos: torch.Tensor | None = None
        self._success_bonus_given: torch.Tensor | None = None

        # Populated in _load_scene (link references)
        self._contact_links: list = []

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Simulation config (production GPU budget)
    # ------------------------------------------------------------------
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2 ** 22,   # 4M contacts
                max_rigid_patch_count=2 ** 21,     # 2M patches
            ),
            scene_config=SceneConfig(
                contact_offset=0.01,
            ),
        )

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera", pose=pose, width=128, height=128, fov=np.pi / 2,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1,
        )

    # ------------------------------------------------------------------
    # Scene building
    # ------------------------------------------------------------------
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        # 1. Kitchen counter + ground
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=KITCHEN_SCENE_SCALE)

        # 2. Plate -- kinematic box on counter
        builder = self.scene.create_actor_builder()
        plate_half = PLATE_HALF_SIZE
        builder.add_box_collision(half_size=plate_half)
        mat = sapien.render.RenderMaterial()
        mat.base_color = [0.92, 0.92, 0.88, 1.0]     # off-white ceramic
        builder.add_box_visual(half_size=plate_half, material=mat)
        builder.initial_pose = sapien.Pose(p=list(PLATE_POS_IN_SINK))
        self.plate = builder.build_kinematic(name="plate")

        # 3. Cache references to left-hand contact links
        self._contact_links = []
        for link_name in _LEFT_CONTACT_LINKS:
            if link_name in self.agent.robot.links_map:
                self._contact_links.append(
                    self.agent.robot.links_map[link_name]
                )
        if not self._contact_links:
            raise RuntimeError(
                f"No contact links found! Available: "
                f"{sorted(self.agent.robot.links_map.keys())[:20]}"
            )

        # 4. Dirt grids (one per parallel env)
        self._dirt_grids = [
            VirtualDirtGrid(
                H=self.grid_h, W=self.grid_w, brush_radius=self.brush_radius,
            )
            for _ in range(self.num_envs)
        ]

    # ------------------------------------------------------------------
    # Episode initialisation
    # ------------------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.scene_builder.initialize(env_idx)
        with torch.device(self.device):
            b = len(env_idx)

            # Reset robot to standing pose with small qpos noise
            qpos_np = self.agent.keyframes["standing"].qpos
            qpos = torch.tensor(qpos_np, dtype=torch.float32, device=self.device)
            if self.robot_init_qpos_noise > 0:
                noise = torch.randn_like(qpos) * self.robot_init_qpos_noise
                qpos = qpos + noise
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(self._robot_init_pose)

            # Reset plate with +/- 2.5 cm XY randomisation
            from mani_skill.envs.utils import randomization
            from mani_skill.utils.structs.pose import Pose

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = randomization.uniform(
                low=-0.025, high=0.025, size=(b, 2),
            )
            xyz[:, 0] += PLATE_POS_IN_SINK[0]
            xyz[:, 1] += PLATE_POS_IN_SINK[1]
            xyz[:, 2] = PLATE_POS_IN_SINK[2]
            self.plate.set_pose(Pose.create_from_pq(xyz))

            # Reset dirt grids for these envs
            for idx in env_idx.cpu().tolist():
                self._dirt_grids[idx].reset()

            # Reset tracking tensors
            if (
                self._prev_actions is not None
                and self._prev_actions.shape[0] == self.num_envs
            ):
                self._prev_actions[env_idx] = 0.0
                self._success_bonus_given[env_idx] = False
                if self._prev_palm_pos is not None:
                    self._prev_palm_pos[env_idx] = 0.0
            else:
                self._prev_actions = None
                self._prev_palm_pos = None
                self._success_bonus_given = None

    # ------------------------------------------------------------------
    # Multi-link contact detection
    # ------------------------------------------------------------------
    def _get_contact_info(self) -> dict[str, torch.Tensor]:
        """Compute multi-link contact force and contact centroid position.

        Returns dict with:
            force     : (num_envs,) total contact force in Newtons
            centroid  : (num_envs, 3) weighted centroid of contacting links
            is_contact: (num_envs,) bool -- any link in contact
        """
        total_force = torch.zeros(self.num_envs, device=self.device)
        weighted_pos = torch.zeros(self.num_envs, 3, device=self.device)

        for link in self._contact_links:
            # (num_envs, 3) force vector per link
            fvec = self.scene.get_pairwise_contact_forces(link, self.plate)
            # We use L2 *magnitude*, not normal-axis projection.
            # For a horizontal plate the magnitude ≈ Fz; it also captures
            # any lateral shear, making the safety threshold more conservative.
            fmag = torch.linalg.norm(fvec, dim=1)  # (num_envs,)
            total_force += fmag
            # Weight position by force magnitude for centroid
            weighted_pos += fmag.unsqueeze(1) * link.pose.p

        # Centroid (fallback to palm_link pos when no contact)
        palm_pos = self._get_palm_pos()
        has_contact = total_force > CONTACT_THRESHOLD
        safe_total = total_force.clamp(min=1e-6)
        centroid = torch.where(
            has_contact.unsqueeze(1),
            weighted_pos / safe_total.unsqueeze(1),
            palm_pos,
        )

        return dict(
            force=total_force,
            centroid=centroid,
            is_contact=has_contact,
        )

    def _get_palm_pos(self) -> torch.Tensor:
        """Return (num_envs, 3) world position of left palm link."""
        return self.agent.robot.links_map["left_palm_link"].pose.p

    def _get_left_tcp_pos(self) -> torch.Tensor:
        """Return (num_envs, 3) world position of left TCP (kept for compat).

        Note: ``agent.left_tcp`` is defined by ``UnitreeG1UpperBody._after_init``
        as ``robot.links_map['left_tcp_link']``.  This helper is used only
        for observations; contact detection uses multi-link palm + fingers.
        """
        assert hasattr(self.agent, "left_tcp"), (
            "Agent has no 'left_tcp' attribute — check robot URDF or "
            "UnitreeG1UpperBody._after_init(). "
            f"Available attrs: {[a for a in dir(self.agent) if 'tcp' in a.lower()]}"
        )
        return self.agent.left_tcp.pose.p

    # ------------------------------------------------------------------
    # Dirt grid update
    # ------------------------------------------------------------------
    def _update_dirt_grids(self, contact_info: dict) -> torch.Tensor:
        """Update dirt grids using contact centroid.  Returns (num_envs,) delta."""
        centroid = contact_info["centroid"]   # (num_envs, 3)
        is_contact = contact_info["is_contact"]
        plate_pos = self.plate.pose.p         # (num_envs, 3)
        plate_half = np.array(PLATE_HALF_SIZE[:2])

        delta_cleans = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            if not is_contact[i]:
                continue
            pos_np = centroid[i].cpu().numpy()
            center_np = plate_pos[i].cpu().numpy()
            cell = self._dirt_grids[i].world_to_cell(pos_np, center_np, plate_half)
            delta = self._dirt_grids[i].mark_clean(cell[0], cell[1])
            delta_cleans[i] = float(delta)

        return delta_cleans

    # ------------------------------------------------------------------
    # Evaluate (success / fail / metrics)
    # ------------------------------------------------------------------
    def evaluate(self):
        # Guard: plate may not exist during first reconfigure
        if not hasattr(self, "plate") or self.plate is None:
            zeros_f = torch.zeros(self.num_envs, device=self.device)
            zeros_b = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device,
            )
            self._eval_cache = dict(
                contact_force=zeros_f,
                delta_clean=zeros_f.clone(),
                cleaned_ratio=zeros_f.clone(),
                is_contact=zeros_b,
                contact_centroid=torch.zeros(
                    self.num_envs, 3, device=self.device,
                ),
            )
            return {
                "success": zeros_b,
                "fail": zeros_b.clone(),
                "contact_force": zeros_f,
                "delta_clean": zeros_f.clone(),
                "cleaned_ratio": zeros_f.clone(),
            }

        # Multi-link contact
        contact_info = self._get_contact_info()
        contact_force = contact_info["force"]

        # Update dirt grids
        delta_clean = self._update_dirt_grids(contact_info)

        # Cleaned ratios
        cleaned_ratios = torch.tensor(
            [g.get_cleaned_ratio() for g in self._dirt_grids],
            device=self.device, dtype=torch.float32,
        )

        # Success / fail
        success = cleaned_ratios >= SUCCESS_CLEAN_RATIO
        fail = contact_force > FZ_HARD

        # Cache for reward
        self._eval_cache = dict(
            contact_force=contact_force,
            delta_clean=delta_clean,
            cleaned_ratio=cleaned_ratios,
            is_contact=contact_info["is_contact"],
            contact_centroid=contact_info["centroid"],
        )

        return {
            "success": success,
            "fail": fail,
            "contact_force": contact_force,
            "delta_clean": delta_clean,
            "cleaned_ratio": cleaned_ratios,
        }

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_obs_extra(self, info: dict):
        # Guard: left_tcp is set by UnitreeG1UpperBody._after_init()
        assert hasattr(self.agent, "left_tcp"), (
            f"Agent missing 'left_tcp'. Available: "
            f"{[a for a in dir(self.agent) if 'tcp' in a.lower()]}"
        )
        obs = dict(
            tcp_pose=self.agent.left_tcp.pose.raw_pose,  # (num_envs, 7)
        )
        if "state" in self.obs_mode:
            palm_p = self._get_palm_pos()
            plate_p = self.plate.pose.p
            obs.update(
                palm_pos=palm_p,                            # (num_envs, 3)
                plate_pos=plate_p,                          # (num_envs, 3)
                palm_to_plate=plate_p - palm_p,             # (num_envs, 3)
                contact_force=info.get(
                    "contact_force",
                    torch.zeros(self.num_envs, device=self.device),
                ).unsqueeze(-1),                            # (num_envs, 1)
                cleaned_ratio=info.get(
                    "cleaned_ratio",
                    torch.zeros(self.num_envs, device=self.device),
                ).unsqueeze(-1),                            # (num_envs, 1)
                dirt_grid=torch.tensor(
                    np.stack(
                        [g.get_grid_flat() for g in self._dirt_grids]
                    ),
                    device=self.device, dtype=torch.float32,
                ),                                          # (num_envs, H*W)
            )
        return obs

    # ------------------------------------------------------------------
    # Reward (staged, production-quality)
    # ------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        cache = self._eval_cache
        contact_force = cache["contact_force"]
        delta_clean   = cache["delta_clean"]
        cleaned_ratio = cache["cleaned_ratio"]
        is_contact    = cache["is_contact"]

        # Lazy-init tracking tensors
        if self._prev_actions is None and action is not None:
            self._prev_actions = torch.zeros_like(action)
            self._success_bonus_given = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device,
            )
            self._prev_palm_pos = self._get_palm_pos().clone()

        # -- Stage 0: Reaching (0->1) --
        palm_p = self._get_palm_pos()
        plate_p = self.plate.pose.p
        palm_to_plate_dist = torch.linalg.norm(plate_p - palm_p, dim=1)
        r_reach = W_REACH * (1.0 - torch.tanh(5.0 * palm_to_plate_dist))

        # -- Stage 1: Contact bonus --
        r_contact = W_CONTACT * is_contact.float()

        # -- Stage 2: Cleaning reward --
        r_clean = W_CLEAN * delta_clean

        # -- Stage 3: Sweep bonus (lateral velocity while in contact) --
        r_sweep = torch.zeros(self.num_envs, device=self.device)
        if self._prev_palm_pos is not None:
            palm_vel = palm_p - self._prev_palm_pos              # (num_envs, 3)
            lateral_speed = torch.linalg.norm(palm_vel[:, :2], dim=1)  # XY only
            r_sweep = W_SWEEP * lateral_speed * is_contact.float()

        # -- Penalties --
        r_time = -W_TIME * torch.ones(self.num_envs, device=self.device)

        if self._prev_actions is not None and action is not None:
            jerk_sq = torch.sum((action - self._prev_actions) ** 2, dim=1)
            r_jerk = -W_JERK * jerk_sq
        else:
            r_jerk = torch.zeros(self.num_envs, device=self.device)

        if action is not None:
            r_act = -W_ACT * torch.sum(action ** 2, dim=1)
        else:
            r_act = torch.zeros(self.num_envs, device=self.device)

        force_excess = torch.clamp(contact_force - FZ_SOFT, min=0.0)
        r_force = -W_FORCE * force_excess

        # -- Success bonus (one-shot) --
        r_success = torch.zeros(self.num_envs, device=self.device)
        if self._success_bonus_given is not None:
            new_success = (
                (cleaned_ratio >= SUCCESS_CLEAN_RATIO)
                & ~self._success_bonus_given
            )
            r_success[new_success] = SUCCESS_BONUS
            self._success_bonus_given = self._success_bonus_given | new_success

        # -- Total --
        reward = (
            r_reach + r_contact + r_clean + r_sweep
            + r_time + r_jerk + r_act + r_force + r_success
        )

        # Update trackers for next step
        if action is not None:
            self._prev_actions = action.clone()
        self._prev_palm_pos = palm_p.clone()

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict,
    ):
        # Approximate max per-step reward
        max_approx = W_CLEAN * 9 + W_REACH + W_CONTACT + W_SWEEP + SUCCESS_BONUS
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_approx


# ===========================================================================
# Registered environment
# ===========================================================================

@register_env("UnitreeG1DishWipe-v1", max_episode_steps=1000)
class UnitreeG1DishWipeEnv(UnitreeG1DishWipeEnvBase):
    """
    **Task Description:**
    Control the Unitree G1 humanoid (upper body, fixed legs) to wipe a plate
    placed **inside the kitchen sink** using the left hand.

    **Randomisations:**
    - Plate XY +/- 2.5 cm around nominal sink position
    - Robot qpos noise +/- 0.02 rad

    **Contact detection:**
    Multi-link (palm + 3 finger links) with force-weighted centroid for
    accurate dirt-grid mapping.

    **Success Conditions:**
    - VirtualDirtGrid cleaned_ratio >= 95 %

    **Failure Conditions:**
    - Total contact force magnitude exceeds ``FZ_HARD`` (200 N)

    **Observations (state mode):**
    Base robot state + extras:
    - ``tcp_pose``       (7D) Left TCP pose
    - ``palm_pos``       (3D) Left palm position
    - ``plate_pos``      (3D) Plate centre (inside sink)
    - ``palm_to_plate``  (3D) Vector palm -> plate
    - ``contact_force``  (1D) Multi-link force magnitude
    - ``cleaned_ratio``  (1D) Fraction cleaned
    - ``dirt_grid``      (100D) Flat 10x10 grid state
    """

    _sample_video_link = ""

    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body"]
    agent: UnitreeG1UpperBody

    def __init__(self, *args, **kwargs):
        self._robot_init_pose = copy.deepcopy(
            UnitreeG1UpperBody.keyframes["standing"].pose,
        )
        self._robot_init_pose.p = [-0.3, 0.0, 0.755]
        super().__init__(
            *args,
            robot_uids="unitree_g1_simplified_upper_body",
            **kwargs,
        )
