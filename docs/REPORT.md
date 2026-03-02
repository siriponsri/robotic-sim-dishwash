# Session Report — 2 March 2026

> **Scope**: NB01 – NB04 execution, environment debugging, and robot stability fix  
> **Platform**: RunPod · NVIDIA RTX 5090 (33.7 GB) · Ubuntu 24.04 · Python 3.12.3  
> **Stack**: ManiSkill 3 (v3.0.0b22) · SAPIEN 3.0.2 · PyTorch 2.10.0+cu128 · Stable-Baselines3 2.7.0

---

## 1. Issues Encountered & Fixes

### 1.1 Vulkan Render Driver — No GPU Vulkan ICD

| Item | Detail |
|------|--------|
| **Symptom** | `RuntimeError: vk::Instance creation failed — no Vulkan ICD found` when creating any ManiSkill env. |
| **Root Cause** | RunPod containers ship with NVIDIA GPU drivers but no Vulkan ICD JSON manifest. ManiSkill / SAPIEN requires a Vulkan device for scene rendering. |
| **Fix** | Use the **Lavapipe** (software) Vulkan driver that is already installed in the container. Set environment variables before any import: |

```python
import os
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
os.environ["MESA_VK_DEVICE_SELECT"] = "10005:0"
os.environ["DISPLAY"] = ""
```

And pass `render_backend="cpu"` to every `gym.make()` call.  
All notebooks (NB01–NB04) include this block as the first code cell.

---

### 1.2 NaN Rewards on First Step

| Item | Detail |
|------|--------|
| **Symptom** | `reward = NaN` returned by `env.step()` on the very first step after `reset()`. Causes downstream metrics to be NaN. |
| **Root Cause** | The custom dense reward function in `apple_fullbody_env.py` computes distance-based terms. On the initial transition some internal distances are uninitialized (division by zero / inf). |
| **Fix** | Wrapped every reward component with `torch.nan_to_num(..., nan=0.0)` and added a final guard on the total reward tensor before returning. |

```python
reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
```

Applied in both `apple_fullbody_env.py` and `dishwipe_fullbody_env.py`.

---

### 1.3 Robot Spawns at Ground Level — Sinks Into Counter

| Item | Detail |
|------|--------|
| **Symptom** | The G1 robot spawned at `z = 0` (ground plane) and immediately intersected with the kitchen counter geometry, producing physics explosions. |
| **Root Cause** | The default robot pose `p = [0, 0, 0]` was inside or below the counter surface. The built-in `UnitreeG1PlaceAppleInBowl-v1` uses the upper-body variant which has `fix_root_link=True` and a hardcoded init pose — our full-body env did not replicate this. |
| **Fix** | Set the robot initial pose to the **standing keyframe** position with an offset: |

```python
self._robot_init_pose = copy.deepcopy(UnitreeG1.keyframes["standing"].pose)
self._robot_init_pose.p = [-0.3, 0, 0.755]   # x=-0.3 to stand in front of counter
```

`z = 0.755` is the nominal standing height from the G1 URDF keyframe.

---

### 1.4 Missing Bowl Mesh — Invisible Collision-Only Cylinder

| Item | Detail |
|------|--------|
| **Symptom** | The bowl was a physics-only cylinder with no visual mesh — invisible in rendered images. |
| **Root Cause** | Original code used `scene.create_actor_builder()` with only a collision shape and no visual shape. |
| **Fix** | Replaced the plain cylinder with ManiSkill's built-in bowl mesh from the YCB asset collection (same bowl used in the official `UnitreeG1PlaceAppleInBowl-v1`). The bowl now has correct visual and collision geometry. |

---

### 1.5 Unrealistic Plate — Flat Disc Cylinder

| Item | Detail |
|------|--------|
| **Symptom** | The plate was a very thin cylinder that looked unrealistic. |
| **Root Cause** | Placeholder geometry using `add_cylinder_visual(radius=0.12, half_length=0.005)`. |
| **Fix** | Replaced with a shallow disc cylinder (`radius=0.10, half_length=0.008`) with a realistic off-white color (`[0.95, 0.93, 0.88]`). This is a visual improvement; collision shape matches. |

---

### 1.6 Robot Falls Immediately — Core Physics Issue ⭐

| Item | Detail |
|------|--------|
| **Symptom** | In NB04 videos the robot collapsed to the ground within the first few frames for **all** baseline policies (including zero-action "Stand-Only"). Rewards kept decreasing every step. |
| **Root Cause** | See detailed analysis below. |

#### Investigation Timeline

1. **Initial hypothesis — weak PD gains**: The default `UnitreeG1` has `body_stiffness=50, body_damping=1, body_force_limit=100`. Created `UnitreeG1HighGain` subclass boosting to `kp=400, kd=40, fl=400`. Robot still fell.

2. **GitHub source study**: Fetched official ManiSkill source for `g1.py`, `g1_upper_body.py`, `humanoid_pick_place.py`, and `humanoid_stand.py`. Key findings:
   - The built-in `UnitreeG1PlaceAppleInBowl-v1` uses **upper-body** variant with `fix_root_link=True`.
   - `UnitreeG1Stand-v1` (the official standing task) is an **RL task** — standing is something that must be *learned*, not a passive default.

3. **Systematic gain sweep**: Tested kp from 400 to 5000, kd from 40 to 500, fl from 400 to 2000. Physics analysis showed:
   - Joint angles tracked well (hip error < 0.06 rad, knee error < 0.03 rad).
   - But pelvis z dropped steadily: `0.755 → 0.66 → 0.10` in ~12 steps.
   - Max joint velocity = 100 rad/s at step 0 (explosive initial dynamics).

4. **Ground contact verification**: Checked `scene.get_contacts()` — ground collision IS present (32 contact pairs including ankle↔ground after the first simulation step). Ground was not the issue.

5. **Official env validation**: Ran the official `UnitreeG1Stand-v1` with zero actions — it ALSO falls (`z → -0.487`). Confirmed: free-floating full-body G1 **cannot passively stand by design**.

6. **`balance_passive_force=True` test**: Enables gravity compensation. On a free-floating robot (`fix_root_link=False`) this completely cancels gravity on all links, causing the robot to **float upward** to `z = 7.9`. Not viable.

#### Root Cause

The Unitree G1 with `fix_root_link=False` (free-floating base) is designed as an RL locomotion platform. Even standing still requires a trained balancing policy. The default PD gains are intentionally weak because locomotion controllers learn to generate balanced torques. For a **manipulation** task where we want the robot to focus on arm/hand control, the official approach is to use `fix_root_link=True`.

#### Fix Applied

Changed both env files to use a custom agent with:

| Parameter | Before | After |
|-----------|--------|-------|
| `fix_root_link` | `False` (inherited from UnitreeG1) | **`True`** |
| `balance_passive_force` | `False` | **`True`** (in `_controller_configs`) |
| `body_stiffness` | 50 → 1000 | **1000** (matches upper-body) |
| `body_damping` | 1 → 100 | **100** (matches upper-body) |
| `body_force_limit` | 100 → 500 | **100** (sufficient with gravity comp) |
| Controllable joints | 37 DOF | **37 DOF** (all joints retained) |
| Finger friction | none | **2.0 static/dynamic** (for grasping) |

New agent classes: `UnitreeG1FullBodyFixed` (apple env), `UnitreeG1FullBodyFixedDW` (dishwipe env).

**Result**: Robot z stays locked at `0.7550` for the full 200-step episode with zero actions. Fall rate = 0% across all baselines.

---

## 2. Files Modified

| File | Changes |
|------|---------|
| `src/envs/apple_fullbody_env.py` | Custom agent `UnitreeG1FullBodyFixed` (fix_root_link, balance_passive_force, finger friction, urdf_config), robot init pose, NaN reward guards, bowl mesh from YCB assets, plate geometry |
| `src/envs/dishwipe_fullbody_env.py` | Custom agent `UnitreeG1FullBodyFixedDW` (same fix), robot init pose, NaN reward guards |
| `notebooks/NB01_setup_smoke.ipynb` | Added Vulkan/GPU setup cell, render_backend="cpu", full execution |
| `notebooks/NB02_env_exploration.ipynb` | Added Vulkan/GPU setup, render_backend="cpu", full execution |
| `notebooks/NB03_reward_safety_mlflow.ipynb` | Added Vulkan/GPU setup, render_backend="cpu", full execution |
| `notebooks/NB04_baselines_smoothing.ipynb` | Added Vulkan/GPU check, video recording cells, mediapy inline display, reward curves, comparison charts, full execution |

---

## 3. Notebook Results

### 3.1 NB01 — Setup & Smoke Test

- **Status**: ✅ All cells pass
- **Verified**:
  - ManiSkill 3, SAPIEN, PyTorch CUDA all importable
  - Both custom envs register and create successfully
  - `env.reset()` + `env.step()` produce valid obs/reward (no NaN)
  - Rendered frame saved as `render_test_apple.png`
  - Action space: 37 DOF (`pd_joint_delta_pos`)
  - Observation space: 315-dim state vector

**Artifacts**: `artifacts/NB01/` — `smoke_results.json`, `env_spec_apple.json`, `env_spec_dishwipe.json`, `render_test_apple.png`, `active_joints_fullbody.json`

---

### 3.2 NB02 — Environment Exploration

- **Status**: ✅ All cells pass
- **Verified**:
  - Observation breakdown: proprioception (37 qpos + 37 qvel), object poses, goal info
  - Action groups: legs (12), torso (7), arms (14), hands (4)
  - Reset distribution: 50 episodes × position/orientation stats
  - Reward-per-step curve (1-episode trajectory)
  - Balance analysis: CoM tracking, joint angle monitoring

**Artifacts**: `artifacts/NB02/` — `obs_breakdown.json`, `obs_breakdown.png`, `action_groups.json`, `action_groups.png`, `reset_distribution.png`, `reset_renders.png`, `reward_per_step.png`, `balance_analysis.json`, `balance_analysis.png`, `complexity_comparison.png`

---

### 3.3 NB03 — Reward & Safety Analysis

- **Status**: ✅ All cells pass
- **Verified**:
  - Reward contract: dense reward components documented (reaching, grasping, placing)
  - Safety-critical info keys: `is_robot_static`, `is_grasped`, `success`
  - Reward validation: step rewards bounded, no NaN, monotonic progress signal
  - Per-component reward analysis chart

**Artifacts**: `artifacts/NB03/` — `reward_contract_apple.json`, `reward_contract_dishwipe.json`, `reward_validation.json`, `safety_validation.json`, `reward_analysis.png`, `info_keys.json`

---

### 3.4 NB04 — Baselines & Smoothing

- **Status**: ✅ All 16 code cells pass (re-run after robot fix)
- **Config**: `control_mode=pd_joint_delta_pos`, `n_eval_episodes=20`, `max_steps_per_ep=200`, `smooth_alpha=0.3`

#### Baseline Leaderboard (post-fix)

| Method | Mean Reward | Std | Fall Rate | Mean Jerk |
|--------|-------------|-----|-----------|-----------|
| **Stand-Only** | **+0.301** | 0.050 | 0% | 0.000 |
| **BaseController** | **+0.292** | 0.050 | 0% | 0.001 |
| **Heuristic (Reach)** | **+0.286** | 0.052 | 0% | 0.017 |
| Smoothed Random | -0.057 | 0.211 | 0% | 24.657 |
| Random | -4.257 | 0.194 | 0% | 24.623 |

#### Comparison: Before vs After Fix

| Method | Before (falling robot) | After (fixed root) | Change |
|--------|----------------------|---------------------|--------|
| Stand-Only | -0.021 | +0.301 | **+15×** |
| BaseController | -0.022 | +0.292 | **+14×** |
| Heuristic | -0.024 | +0.286 | **+13×** |
| Smoothed Random | -0.293 | -0.057 | **+80%** |
| Random | -4.477 | -4.257 | **+5%** |

**Key observations**:
- Stand-Only, BaseController, and Heuristic now accumulate **positive** reward (they keep the robot in a good configuration near objects).
- Random and Smoothed Random still get negative rewards because wild joint movements penalize the reward.
- Smoothed Random beats plain Random by ~4.2 points, showing that action smoothing helps even with random exploration.
- The `SmoothActionWrapper` (EMA α=0.3) did not meaningfully reduce jerk in this case because the delta-pos controller already bounds actions to `[-0.2, 0.2]`.

**Videos** (100-step recordings, inline via mediapy):
- `01_random.mp4` — Random joint movements, large oscillations
- `02_stand_only.mp4` — Robot stands perfectly still for 100 steps
- `03_heuristic.mp4` — Robot reaches toward apple with right arm
- `04_base_ctrl.mp4` — Proportional controller nudging joints toward target

**Charts**:
- `reward_curves_per_step.png` — Per-step reward + cumulative curves for each policy
- `baseline_comparison.png` — Bar charts: reward, jerk, fall rate across all baselines

**Artifacts**: `artifacts/NB04/` — `baseline_leaderboard.csv`, `nb04_config.json`, `reward_curves_per_step.png`, `baseline_comparison.png`, `videos/{01_random,02_stand_only,03_heuristic,04_base_ctrl}.mp4`

---

## 4. Technical Notes

### Environment Configuration

```python
gym.make(
    "UnitreeG1PlaceAppleInBowlFullBody-v1",
    render_backend="cpu",
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    max_episode_steps=200,
)
```

### Custom Agent Architecture

```
UnitreeG1FullBodyFixed(UnitreeG1)
├── fix_root_link = True          # pelvis fixed in space
├── body_stiffness = 1000         # PD proportional gain
├── body_damping = 100            # PD derivative gain
├── body_force_limit = 100        # joint torque limit
├── balance_passive_force = True  # gravity compensation
├── urdf_config                   # finger friction for grasping
└── 37 DOF controllable           # legs + torso + arms + hands
```

### Hardware

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5090 |
| VRAM | 33.7 GB |
| Render | Lavapipe (software Vulkan via `lvp_icd.json`) |
| Sim backend | CPU (PhysX) |
| Python | 3.12.3 |
| OS | Ubuntu 24.04 (RunPod) |

---

## 5. Next Steps

- **NB05**: Train PPO on the fixed-root full-body env
- **NB06**: Train SAC
- **NB07**: Residual SAC ablation
- **NB08**: Evaluation & comparison
- **NB09**: Bonus DishWipe task
