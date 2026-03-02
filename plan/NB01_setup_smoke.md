# NB01 — Setup, Register Environments & Smoke Test — RTX 5090 Edition

> **Status: Not started**
> **Hardware: CPU (GPU detection + VRAM report)**
> **Task: Both (Apple Full Body + DishWipe Full Body)**

---

## Goal

Register both custom Full-Body environments, verify obs/act shapes, download
assets, and confirm that both environments create and step correctly.

---

## Required Input

| Item | Source | Description |
|------|--------|-------------|
| Virtual environment (`.env/`) | Local setup | Python 3.11+ with all deps installed |
| `src/envs/apple_fullbody_env.py` | Source code | Custom Full-Body Apple env |
| `src/envs/dishwipe_fullbody_env.py` | Source code | Custom Full-Body DishWipe env |
| `src/envs/dirt_grid.py` | Source code | VirtualDirtGrid module |
| `src/envs/__init__.py` | Source code | Register both envs |
| `.env.local` (MLflow credentials) | User-created | MLFLOW_TRACKING_URI, USERNAME, PASSWORD |

---

## Expected Output / Artifacts

| File | Path | Description |
|------|------|-------------|
| `env_spec_apple.json` | `artifacts/NB01/` | Apple env: obs shape, act shape, DOF, robot info |
| `env_spec_dishwipe.json` | `artifacts/NB01/` | DishWipe env: obs shape, act shape, DOF, robot info |
| `active_joints_fullbody.json` | `artifacts/NB01/` | All 37 active joint names grouped by body part |
| `requirements.txt` | `artifacts/NB01/` | pip freeze snapshot |
| `smoke_results.json` | `artifacts/NB01/` | Smoke test statistics (both envs) |
| `nb01_config.json` | `artifacts/NB01/` | Config used |

---

## Resources

| Resource | Requirement |
|----------|-------------|
| CPU | 2+ cores |
| RAM | 4 GB |
| GPU | Not required |
| Disk | ~2 GB (asset downloads) |
| Runtime | ~2-5 minutes |

---

## Steps

| Step | Purpose |
|------|---------|
| 1 | **Version check**: Python, NumPy, PyTorch, ManiSkill, SB3, SAPIEN |
| 2 | **Imports** + PROJECT_ROOT setup + register both custom envs |
| 3 | **Download assets**: `UnitreeG1PlaceAppleInBowl-v1` assets (apple, bowl GLB/PLY) |
| 4 | **Apple env smoke test**: create `UnitreeG1PlaceAppleInBowlFullBody-v1`, inspect obs/act |
| 5 | **DishWipe env smoke test**: create `UnitreeG1DishWipeFullBody-v1`, inspect obs/act |
| 6 | **Joint discovery**: list all 37 joints grouped (legs, torso, arms, hands) |
| 7 | **Balance check**: verify `is_standing()` / `is_fallen()` on full-body robot |
| 8 | **Render test**: try rgb_array render, handle CPU-only gracefully |
| 9 | **Quick smoke test**: 50 random steps on each env, collect basic stats |
| 10 | **Reproducibility**: multi-reset consistency check (seed 42, 123) |
| 11 | **Save artifacts** + MLflow logging |

---

## Main Code (Pseudocode)

```python
# ── Step 1: Version Check ──
import sys, platform
import numpy as np
import torch
import mani_skill
import stable_baselines3 as sb3

print(f"Python     : {sys.version}")
print(f"NumPy      : {np.__version__}")
print(f"PyTorch    : {torch.__version__}")
print(f"ManiSkill  : {mani_skill.__version__}")
print(f"SB3        : {sb3.__version__}")
print(f"CUDA       : {torch.cuda.is_available()}")

# ── Step 2: Imports + Register ──
import gymnasium as gym
import mani_skill.envs
from pathlib import Path
import json, os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# This triggers @register_env for both custom envs
from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv, UnitreeG1DishWipeFullBodyEnv

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB01"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 3: Download Assets ──
from mani_skill.utils.download_asset import download_asset
download_asset("UnitreeG1PlaceAppleInBowl-v1")

# ── Step 4: Apple Full-Body Smoke Test ──
CFG = {"seed": 42, "control_mode": "pd_joint_delta_pos"}

env_apple = gym.make(
    "UnitreeG1PlaceAppleInBowlFullBody-v1",
    num_envs=1,
    obs_mode="state",
    control_mode=CFG["control_mode"],
    render_mode="rgb_array",
)
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
env_apple = CPUGymWrapper(env_apple)
obs_a, info_a = env_apple.reset(seed=CFG["seed"])

apple_spec = {
    "env_id": "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "obs_shape": list(obs_a.shape),
    "act_shape": list(env_apple.action_space.shape),
    "act_low": env_apple.action_space.low[:3].tolist(),
    "act_high": env_apple.action_space.high[:3].tolist(),
    "robot_uid": "unitree_g1",
    "dof": env_apple.action_space.shape[0],
    "root_type": "free-floating",
}
print(f"Apple obs: {obs_a.shape}, act: {env_apple.action_space.shape}")

with open(ARTIFACTS_DIR / "env_spec_apple.json", "w") as f:
    json.dump(apple_spec, f, indent=2)

# ── Step 5: DishWipe Full-Body Smoke Test ──
env_dw = gym.make(
    "UnitreeG1DishWipeFullBody-v1",
    num_envs=1,
    obs_mode="state",
    control_mode=CFG["control_mode"],
    render_mode="rgb_array",
)
env_dw = CPUGymWrapper(env_dw)
obs_d, info_d = env_dw.reset(seed=CFG["seed"])

dishwipe_spec = {
    "env_id": "UnitreeG1DishWipeFullBody-v1",
    "obs_shape": list(obs_d.shape),
    "act_shape": list(env_dw.action_space.shape),
    "robot_uid": "unitree_g1",
    "dof": env_dw.action_space.shape[0],
    "root_type": "free-floating",
}
print(f"DishWipe obs: {obs_d.shape}, act: {env_dw.action_space.shape}")

with open(ARTIFACTS_DIR / "env_spec_dishwipe.json", "w") as f:
    json.dump(dishwipe_spec, f, indent=2)

# ── Step 6: Joint Discovery ──
agent = env_apple.unwrapped.agent
joint_info = {
    "total_dof": len(agent.robot.active_joints),
    "joint_groups": {
        "lower_body (legs)": [j.name for j in agent.robot.active_joints
                              if "left_hip" in j.name or "right_hip" in j.name
                              or "knee" in j.name or "ankle" in j.name],
        "torso": [j.name for j in agent.robot.active_joints
                  if "torso" in j.name],
        "upper_body (arms)": [j.name for j in agent.robot.active_joints
                              if "shoulder" in j.name or "elbow" in j.name
                              or "wrist" in j.name],
        "hands (fingers)": [j.name for j in agent.robot.active_joints
                            if any(k in j.name for k in
                                   ["zero", "one", "two", "three",
                                    "four", "five", "six"])],
    },
}
with open(ARTIFACTS_DIR / "active_joints_fullbody.json", "w") as f:
    json.dump(joint_info, f, indent=2)

# ── Step 7: Balance Check ──
agent_cls = type(agent)
has_balance = hasattr(agent, "is_standing") and hasattr(agent, "is_fallen")
print(f"Balance API available: {has_balance}")
if has_balance:
    print(f"  is_standing: {agent.is_standing()}")
    print(f"  is_fallen:   {agent.is_fallen()}")

# ── Step 8: Render Test ──
try:
    frame = env_apple.render()
    print(f"Render OK: {frame.shape if hasattr(frame, 'shape') else type(frame)}")
except Exception as e:
    print(f"Render not available (CPU-only): {e}")

# ── Step 9: Quick Smoke Test ──
smoke_results = {}
for env_name, env in [("apple", env_apple), ("dishwipe", env_dw)]:
    obs, info = env.reset(seed=42)
    rewards, contacts = [], []
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            obs, info = env.reset()
    smoke_results[env_name] = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
    }

with open(ARTIFACTS_DIR / "smoke_results.json", "w") as f:
    json.dump(smoke_results, f, indent=2)

# ── Step 10: Reproducibility ──
obs1, _ = env_apple.reset(seed=42)
obs2, _ = env_apple.reset(seed=42)
assert np.allclose(obs1, obs2), "Reproducibility check failed!"
print("Reproducibility check PASSED")

# ── Step 11: Cleanup + MLflow ──
env_apple.close()
env_dw.close()

# MLflow logging (optional — may fail without credentials)
try:
    from src.utils.mlflow_helpers import setup_mlflow, log_run
    setup_mlflow()
    log_run("NB01_setup_smoke", params=CFG, artifacts_dir=str(ARTIFACTS_DIR))
except Exception as e:
    print(f"MLflow logging skipped: {e}")

print("NB01 Smoke Test PASSED ✅")
```

---

## Key Assertions

- [ ] Apple env obs shape ≈ 100-120 dims
- [ ] Apple env act shape = 37 dims
- [ ] DishWipe env obs shape ≈ 192 dims
- [ ] DishWipe env act shape = 37 dims
- [ ] `is_standing()` / `is_fallen()` methods available
- [ ] Reproducibility: same seed → same obs
- [ ] Both envs step without error for 50 steps

---

## Notes

- Full-body G1 has **37 DOF** (12 legs + 1 torso + 10 arms + 14 fingers)
- Root is **free-floating** → robot CAN fall over
- `is_fallen()` checks if torso height drops below threshold
- Apple env needs `download_asset()` for apple/bowl meshes
- DishWipe env uses same `dirt_grid.py` as before (no change)
- Render may fail on CPU-only (Vulkan required) — this is expected

---

*Plan NB01 — Updated March 2026*
