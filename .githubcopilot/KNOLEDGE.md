# KNOLEDGE.md — v4 for Copilot (Production-Scale Env Rewrite)
## ManiSkill • Unitree G1 • Custom DishWipe Env • Joint-control • PPO vs SAC vs Residual
---

## 1) Confirmed env_id to use (CUSTOM environment)
- env_id: `UnitreeG1DishWipe-v1`
- Source: `src/envs/dishwipe_env.py` (production rewrite v2)
- Robot: `unitree_g1_simplified_upper_body` (UnitreeG1UpperBody, 25 DOF)
- Scene: Kitchen counter + sink + plate (plate in sink basin)
- Registration: `@register_env("UnitreeG1DishWipe-v1", max_episode_steps=1000)`

**IMPORTANT**: The old `UnitreeG1Stand-v1` env (37 DOF, ground-only) is NO LONGER USED.
Import `from src.envs.dishwipe_env import UnitreeG1DishWipeEnv` before `gym.make()`.

---

## 2) Control Mode (joint-control)
- selected control_mode: `pd_joint_delta_pos`
- supported_control_modes: `['pd_joint_pos', 'pd_joint_delta_pos']`

**Rule**
- Use `pd_joint_delta_pos` for training (NB06–NB08).
- Provide fallback to `pd_joint_pos` if requested control_mode is not supported.

---

## 3) Observation / Action Specs (UPDATED v2)
- observation shape: `(1, 168)` (Box(-inf, inf, float32))
  - ManiSkill base state (qpos, qvel, tcp_pose, etc.) + extras:
    - `tcp_pose`       (7D) — left TCP raw pose
    - `palm_pos`       (3D) — left palm position (state mode only)
    - `plate_pos`      (3D) — plate centre
    - `palm_to_plate`  (3D) — vector palm → plate
    - `contact_force`  (1D) — multi-link force magnitude
    - `cleaned_ratio`  (1D) — fraction cleaned
    - `dirt_grid`      (100D) — flat 10×10 grid state (state mode only)
- action shape: `(25,)` (Box(-1.0, 1.0, float32))
- active joints count: `25` (source: robot.active_joints_map)
  - 1 torso + 11 upper body (shoulder/elbow) + 14 finger joints (7 per hand)
  - NO leg joints (legs are fixed)

**Implication**
- RL policy outputs a 25-dim joint delta action.
- Obs is flat tensor → `MlpPolicy` (NOT MultiInputPolicy).
- Dirt grid in obs gives policy spatial cleaning awareness.

---

## 4) Robot Introspection
- Active joints source: `robot.active_joints_map`
- TCP available: **True** (left_tcp, right_tcp)
- Palm links: left_palm_link, right_palm_link
- Finger links (contact): left_two_link, left_four_link, left_six_link
- Root link: **Fixed** (no balance needed)
- TCP-to-palm offset: ~7cm X, ~3.5cm Y (VERIFIED)

**Implication**
- Heuristic policies (NB05/NB08) may use TCP for guidance — that's fine.
- **Contact detection uses multi-link (palm + 3 fingers)** for accuracy.
- Dirt grid mapping uses **force-weighted centroid** of contacting links (not TCP).

---

## 5) Contact / Force API (MULTI-LINK — v2)
- contact API available: `True`
- Contact links (4 total):
  - `left_palm_link`
  - `left_two_link` (finger L1)
  - `left_four_link` (finger R1)
  - `left_six_link` (finger R2)
- Method: `scene.get_pairwise_contact_forces(link, plate)` per link → sum all
- Force-weighted centroid of contacting links used for dirt grid mapping
- Info dict keys: `contact_force` (total), `cleaned_ratio`, `delta_clean`

**Implication**
- Multi-link contact eliminates TCP-vs-palm offset bug (old version).
- Centroid accurately reflects where hand actually touches the plate.
- Thresholds unchanged:
  - `CONTACT_THRESHOLD = 0.5` N (minimum to count as contact)
  - `FZ_SOFT = 50.0` N (penalty begins, magnitude not Fz)
  - `FZ_HARD = 200.0` N (terminate, magnitude not Fz)

---

## 6) Dirt Grid (integrated in env)
- Module: `src/envs/dirt_grid.py` → `VirtualDirtGrid`
- Size: 10×10, brush_radius=1 (3×3 cleaning area)
- Grid is part of env state, reset on `env.reset()`
- Accessible: `env.unwrapped._dirt_grids[0]`
- Info dict keys: `cleaned_ratio`, `delta_clean`
- Success: cleaned_ratio ≥ 0.95
- **NEW**: Grid state (100D flat) included in observations for spatial awareness

---

## 7) Reward Function (dense, staged — v2)
Computed by `dishwipe_env.py :: compute_dense_reward()`:

| Component | Weight | Formula |
|-----------|--------|---------|
| Reaching | `W_REACH=0.5` | `1 - tanh(5 × palm_to_plate_dist)` |
| Contact | `W_CONTACT=1.0` | `is_contacting` (any link) |
| Cleaning | `W_CLEAN=10.0` | `delta_clean` (new cells cleaned) |
| **Sweep** | `W_SWEEP=0.3` | `lateral_vel × is_contact` (NEW) |
| Time | `-W_TIME=0.01` | per step penalty |
| Jerk | `-W_JERK=0.05` | `‖aₜ − aₜ₋₁‖²` |
| Action mag | `-W_ACT=0.005` | `‖aₜ‖²` |
| Force | `-W_FORCE=0.01` | `max(0, F_contact − 50)` (magnitude, not Fz) |
| Success | `+50.0` | one-shot when cleaned ≥ 95% |

**Key change v2**: Reaching reward uses **palm position** (not TCP) for consistency with contact mapping.
**NEW**: Sweep bonus encourages lateral hand movement during contact.

---

## 8) Smoke Test Status (NB01 - verified with v2 env)
- env_id: `UnitreeG1DishWipe-v1`
- obs: `(1, 168)`, act: `(25,)` — UPDATED
- TCP: both available
- Multi-link contact: works (palm + 3 fingers)
- Dirt grid: works (100D in obs)
- Environment stable for pipeline development.

---

## 9) Rendering status (known limitation)
- render success: `False` (on this CPU-only machine)
- Known error: Vulkan descriptor pool / `ErrorOutOfPoolMemory`

**Rule**
- Training notebooks (NB06–NB08) must not depend on render.
- NB01/NB09 must implement render as optional with try/except.

---

## 10) Scene Objects
- Kitchen counter: `KitchenCounterSceneBuilder` (glb model, scale=0.82)
- Plate: kinematic flat box, 20cm×20cm×6mm, **in sink basin** at ~(0.10, 0.20, 0.58), ±2.5cm XY random
- Sink: basin x∈[-0.01,0.32], y∈[0.04,0.49], z∈[0.56,0.79] at scale 0.82
- Ground: standard floor plane
- Robot: G1 upper body at (-0.3, 0, 0.755), fixed root, qpos noise ±0.02 rad

---

## 11) Project Structure
```
src/
  envs/
    __init__.py
    dishwipe_env.py    ← Custom ManiSkill env v2 (multi-link, sweep, grid obs)
    dirt_grid.py       ← VirtualDirtGrid module (unchanged)
notebooks/
  NB01-NB05            ← All verified, dry-run passed
  NB06_train_ppo.ipynb ← Ready for GPU training
  NB07_train_sac.ipynb ← Ready for GPU training
  NB08_residual_sac_ablation.ipynb ← Ready for GPU training
  NB09_evaluation.ipynb ← Ready for evaluation
```

---

## 12) Notebook generation rules for Copilot (must follow)
- Follow `RULE.md` strictly (sections order, config single-source-of-truth, artifacts contract).
- Use this KNOLEDGE.md as ground truth for all env specs.
- Always `sys.path.insert(0, PROJECT_ROOT)` before importing `src.envs`.
- Always import `from src.envs.dishwipe_env import UnitreeG1DishWipeEnv` before `gym.make()`.
- Use env_id `UnitreeG1DishWipe-v1`, NOT `UnitreeG1Stand-v1`.
- Action dim = 25, obs dim ~168, ManiSkill auto-flattens to (168,) for state mode.

---

## 10) TODO (optional upgrades)
If later needed, add:
- The exact list of active joint names (dump from active_joints_map keys) for documentation.
- The exact link name used for contact/pose extraction (so NB02 mapping uses that link pose explicitly).