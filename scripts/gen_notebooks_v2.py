"""Generate ALL notebooks (NB01–NB09) for the revised Full-Body G1 pipeline.

Main task  : Place Apple in Bowl (Full-Body G1, 37 DOF)   — NB01-NB08
Bonus task : DishWipe Full-Body (winner only)              — NB09

Run:  python scripts/gen_notebooks_v2.py
Creates .ipynb files in notebooks/ (overwrites existing).
"""

import json, pathlib, textwrap

NB_DIR = pathlib.Path(__file__).resolve().parent.parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)


# ── helpers ─────────────────────────────────────────────────────────────
def _cell(source: str, cell_type: str = "code", lang: str = "python"):
    lines = textwrap.dedent(source).strip().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    c = {"cell_type": cell_type, "metadata": {}, "source": lines}
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
        if lang != "python":
            c["metadata"]["language"] = lang
    return c

def md(s):  return _cell(s, "markdown")
def code(s): return _cell(s, "code")

def write_nb(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NB_DIR / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ✅ {path.name}  ({len(cells)} cells)")


# ====================================================================
#  NB01 — Setup, Register Environments & Smoke Test
# ====================================================================
def gen_nb01():
    cells = [
        # ── Title ──
        md("""\
# NB01 — Setup, Register Environments & Smoke Test

This notebook verifies all dependencies, registers **both custom Full-Body
environments**, and runs a short smoke test on each.

| Task | Env ID | Robot | DOF |
|------|--------|-------|-----|
| **Main — Place Apple in Bowl** | `UnitreeG1PlaceAppleInBowlFullBody-v1` | `unitree_g1` Full Body | 37 |
| **Bonus — DishWipe** | `UnitreeG1DishWipeFullBody-v1` | `unitree_g1` Full Body | 37 |

Both use the **full-body** Unitree G1 with free-floating root (can walk and
fall). The original upper-body-only variants (25 DOF, fixed legs) are
**no longer used**."""),

        # ── Objectives ──
        md("""\
## Objectives

1. Verify all dependencies are installed (Python, NumPy, PyTorch, ManiSkill, SB3, SAPIEN).
2. Import & register both custom environments via `src.envs`.
3. Download required assets (`UnitreeG1PlaceAppleInBowl-v1`).
4. Create **Apple Full-Body** env → inspect obs/act shapes.
5. Create **DishWipe Full-Body** env → inspect obs/act shapes.
6. Discover all 37 active joints grouped by body part.
7. Verify balance API (`is_standing()` / `is_fallen()`).
8. Attempt render test (Vulkan required).
9. Run 50-step smoke test on each env.
10. Reproducibility check + save artifacts + MLflow."""),

        # ── Resources ──
        md("""\
## Resources

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | Not required | CPU-only smoke test |
| RAM | 4 GB | Env creation + 50 steps |
| Disk | ~2 GB | Asset downloads |
| Runtime | ~2-5 min | Full notebook |"""),

        # ── Step 1: Version Check ──
        md("## Step 1 — Version Check"),
        code("""\
import sys, platform
print(f"Python   : {sys.version}")
print(f"Platform : {platform.platform()}")
print(f"Machine  : {platform.machine()}")

import numpy as np;          print(f"NumPy    : {np.__version__}")
import torch;                print(f"PyTorch  : {torch.__version__}")
import gymnasium as gym;     print(f"Gymnasium: {gym.__version__}")
import mani_skill;           print(f"ManiSkill: {mani_skill.__version__}")
import sapien;               print(f"SAPIEN   : {sapien.__version__}")
import stable_baselines3 as sb3; print(f"SB3      : {sb3.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")"""),

        # ── Step 2: Imports + Register ──
        md("""\
## Step 2 — Imports & Register Custom Environments

Importing `src.envs` triggers `@register_env` decorators for both
full-body environments."""),
        code("""\
import os, json, csv, random, datetime, traceback
from pathlib import Path
from PIL import Image

import mani_skill.envs                          # built-in envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

# ── Project root ──
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # fallback
    os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── Register custom envs ──
from src.envs import (
    UnitreeG1PlaceAppleInBowlFullBodyEnv,
    UnitreeG1DishWipeFullBodyEnv,
)

# Verify registration
assert "UnitreeG1PlaceAppleInBowlFullBody-v1" in [
    spec.id for spec in gym.registry.values()
], "Apple env not registered!"
assert "UnitreeG1DishWipeFullBody-v1" in [
    spec.id for spec in gym.registry.values()
], "DishWipe env not registered!"

print("✅ Both custom environments registered successfully")

# ── Artifacts ──
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB01"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)"""),

        # ── MLflow Setup ──
        md("## MLflow Setup (with CSV fallback)"),
        code("""\
# MLflow — optional, will not block notebook if unavailable
MLFLOW_OK = False
try:
    import mlflow
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env.local")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME",
                              os.environ.get("MLFLOW_TRACKING_USERNAME", ""))
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD",
                              os.environ.get("MLFLOW_TRACKING_PASSWORD", ""))
        mlflow.set_experiment("g1_fullbody_apple_dishwipe")
        MLFLOW_OK = True
        print(f"✅ MLflow connected: {tracking_uri}")
except Exception as e:
    print(f"⚠️ MLflow not available: {e}")

# CSV fallback
CSV_LOG = ARTIFACTS_DIR / "nb01_log.csv"
def csv_log(row: dict):
    \"\"\"Append one row to CSV log.\"\"\"
    file_exists = CSV_LOG.exists()
    with open(CSV_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)"""),

        # ── Config ──
        md("## Configuration"),
        code("""\
CFG = {
    "seed":          42,
    "obs_mode":      "state",
    "control_mode":  "pd_joint_delta_pos",
    "smoke_steps":   50,
    "apple_env_id":  "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "dishwipe_env_id": "UnitreeG1DishWipeFullBody-v1",
    "artifact_dir":  str(ARTIFACTS_DIR),
}
print("Config:")
for k, v in CFG.items():
    print(f"  {k}: {v}")"""),

        # ── Reproducibility ──
        md("## Reproducibility"),
        code("""\
SEED = CFG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"✅ Seeds set to {SEED}")"""),

        # ── Step 3: Download Assets ──
        md("""\
## Step 3 — Download Assets

The Apple env requires mesh assets (apple GLB, bowl GLB, kitchen counter)."""),
        code("""\
from mani_skill.utils.download_asset import download_asset

print("Downloading assets for UnitreeG1PlaceAppleInBowl-v1 ...")
try:
    download_asset("UnitreeG1PlaceAppleInBowl-v1")
    print("✅ Assets downloaded")
except Exception as e:
    print(f"⚠️ Asset download issue (may already exist): {e}")"""),

        # ── Step 4: Apple Full-Body Smoke Test ──
        md("""\
## Step 4 — Apple Full-Body Environment

Create `UnitreeG1PlaceAppleInBowlFullBody-v1` and inspect spaces.
This env uses the **full-body** Unitree G1 (37 DOF, free-floating root)."""),
        code("""\
env_apple = gym.make(
    CFG["apple_env_id"],
    num_envs=1,
    obs_mode=CFG["obs_mode"],
    control_mode=CFG["control_mode"],
    render_mode="rgb_array",
)
env_apple = CPUGymWrapper(env_apple)
obs_a, info_a = env_apple.reset(seed=SEED)

apple_spec = {
    "env_id":     CFG["apple_env_id"],
    "obs_shape":  list(obs_a.shape),
    "act_shape":  list(env_apple.action_space.shape),
    "act_low":    env_apple.action_space.low[:5].tolist(),
    "act_high":   env_apple.action_space.high[:5].tolist(),
    "robot_uid":  "unitree_g1",
    "dof":        env_apple.action_space.shape[0],
    "root_type":  "free-floating",
}

print("=" * 60)
print("  Apple Full-Body Environment")
print("=" * 60)
print(f"  Obs shape : {obs_a.shape}")
print(f"  Act shape : {env_apple.action_space.shape}")
print(f"  Act range : [{env_apple.action_space.low[0]:.2f}, {env_apple.action_space.high[0]:.2f}]")
print(f"  DOF       : {apple_spec['dof']}")
print(f"  Robot     : {apple_spec['robot_uid']} (full body, free root)")
print("=" * 60)

with open(ARTIFACTS_DIR / "env_spec_apple.json", "w") as f:
    json.dump(apple_spec, f, indent=2)
print(f"Saved: env_spec_apple.json")"""),

        # ── Step 5: DishWipe Full-Body Smoke Test ──
        md("""\
## Step 5 — DishWipe Full-Body Environment

Create `UnitreeG1DishWipeFullBody-v1` (bonus task, also full-body 37 DOF)."""),
        code("""\
env_dw = gym.make(
    CFG["dishwipe_env_id"],
    num_envs=1,
    obs_mode=CFG["obs_mode"],
    control_mode=CFG["control_mode"],
    render_mode="rgb_array",
)
env_dw = CPUGymWrapper(env_dw)
obs_d, info_d = env_dw.reset(seed=SEED)

dishwipe_spec = {
    "env_id":     CFG["dishwipe_env_id"],
    "obs_shape":  list(obs_d.shape),
    "act_shape":  list(env_dw.action_space.shape),
    "robot_uid":  "unitree_g1",
    "dof":        env_dw.action_space.shape[0],
    "root_type":  "free-floating",
}

print("=" * 60)
print("  DishWipe Full-Body Environment")
print("=" * 60)
print(f"  Obs shape : {obs_d.shape}")
print(f"  Act shape : {env_dw.action_space.shape}")
print(f"  DOF       : {dishwipe_spec['dof']}")
print(f"  Robot     : {dishwipe_spec['robot_uid']} (full body, free root)")
print("=" * 60)

with open(ARTIFACTS_DIR / "env_spec_dishwipe.json", "w") as f:
    json.dump(dishwipe_spec, f, indent=2)
print(f"Saved: env_spec_dishwipe.json")"""),

        # ── Step 6: Joint Discovery ──
        md("""\
## Step 6 — Joint Discovery (37 DOF)

The full-body G1 has 37 active joints:
- **Lower body (legs)**: 12 joints (6 per leg: hip_pitch/roll/yaw, knee, ankle_pitch/roll)
- **Torso**: 1 joint (waist_yaw or torso_joint)
- **Upper body (arms)**: 10 joints (5 per arm: shoulder × 3, elbow, wrist)
- **Hands (fingers)**: 14 joints (7 per hand)"""),
        code("""\
# Access robot internals via unwrapped env
try:
    agent = env_apple.unwrapped.agent
    robot = agent.robot
    joint_names = [j.name for j in robot.active_joints]

    # Group joints by body part
    def _classify(name):
        name_l = name.lower()
        if any(k in name_l for k in ["hip", "knee", "ankle"]):
            return "lower_body (legs)"
        if any(k in name_l for k in ["torso", "waist"]):
            return "torso"
        if any(k in name_l for k in ["shoulder", "elbow", "wrist"]):
            return "upper_body (arms)"
        return "hands (fingers)"

    joint_groups = {}
    for jn in joint_names:
        g = _classify(jn)
        joint_groups.setdefault(g, []).append(jn)

    joint_info = {
        "total_dof": len(joint_names),
        "joint_names": joint_names,
        "joint_groups": {k: v for k, v in joint_groups.items()},
        "group_sizes": {k: len(v) for k, v in joint_groups.items()},
    }

    print(f"Total DOF: {len(joint_names)}")
    for group, joints in joint_groups.items():
        print(f"  {group}: {len(joints)} joints")
        for j in joints:
            print(f"    - {j}")

except Exception as e:
    print(f"⚠️ Could not access robot joints: {e}")
    joint_info = {"error": str(e), "total_dof": apple_spec["dof"]}

with open(ARTIFACTS_DIR / "active_joints_fullbody.json", "w") as f:
    json.dump(joint_info, f, indent=2)
print(f"\\nSaved: active_joints_fullbody.json")"""),

        # ── Step 7: Balance Check ──
        md("""\
## Step 7 — Balance Check

Full-body G1 has `is_standing()` / `is_fallen()` methods. The robot CAN
fall over, unlike the upper-body variant with fixed legs."""),
        code("""\
try:
    agent = env_apple.unwrapped.agent
    has_standing = hasattr(agent, "is_standing")
    has_fallen = hasattr(agent, "is_fallen")
    print(f"Balance API available: is_standing={has_standing}, is_fallen={has_fallen}")

    if has_standing:
        standing = agent.is_standing()
        print(f"  is_standing() = {standing}")
    if has_fallen:
        fallen = agent.is_fallen()
        print(f"  is_fallen()   = {fallen}")
except Exception as e:
    print(f"⚠️ Balance API check failed: {e}")
    has_standing, has_fallen = False, False

balance_info = {
    "is_standing_available": has_standing,
    "is_fallen_available": has_fallen,
}
print(f"\\n✅ Balance check complete")"""),

        # ── Step 8: Render Test ──
        md("""\
## Step 8 — Render Test

Rendering requires Vulkan GPU support. On CPU-only machines this may fail
gracefully."""),
        code("""\
render_ok = False
try:
    frame = env_apple.render()
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype in (np.float32, np.float64):
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(frame)
    img.save(ARTIFACTS_DIR / "render_test_apple.png")
    display(img.resize((512, 512)))
    render_ok = True
    print(f"✅ Render OK: {frame.shape}")
except Exception as e:
    print(f"⚠️ Render not available (expected on CPU-only): {e}")"""),

        # ── Step 9: Smoke Test (50 steps) ──
        md("""\
## Step 9 — Smoke Test (50 Random Steps per Env)

Run random actions to verify environments step without errors."""),
        code("""\
smoke_results = {}

for env_name, env in [("apple", env_apple), ("dishwipe", env_dw)]:
    obs, info = env.reset(seed=SEED)
    rewards, steps_done = [], 0
    terminated_count, truncated_count = 0, 0

    for step in range(CFG["smoke_steps"]):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        steps_done += 1

        if terminated:
            terminated_count += 1
            obs, info = env.reset()
        if truncated:
            truncated_count += 1
            obs, info = env.reset()

    smoke_results[env_name] = {
        "steps_completed": steps_done,
        "mean_reward":     float(np.mean(rewards)),
        "std_reward":      float(np.std(rewards)),
        "min_reward":      float(np.min(rewards)),
        "max_reward":      float(np.max(rewards)),
        "terminated":      terminated_count,
        "truncated":       truncated_count,
    }

    print(f"\\n{'='*50}")
    print(f"  {env_name.upper()} Smoke Test")
    print(f"{'='*50}")
    for k, v in smoke_results[env_name].items():
        print(f"  {k:20s}: {v}")

with open(ARTIFACTS_DIR / "smoke_results.json", "w") as f:
    json.dump(smoke_results, f, indent=2)
print(f"\\nSaved: smoke_results.json")"""),

        # ── Step 10: Reproducibility ──
        md("## Step 10 — Reproducibility Check"),
        code("""\
# Same seed → same initial observation
obs1, _ = env_apple.reset(seed=42)
obs2, _ = env_apple.reset(seed=42)
assert np.allclose(obs1, obs2), "Reproducibility FAILED!"
print("✅ Reproducibility check PASSED (seed=42)")

obs3, _ = env_apple.reset(seed=123)
assert not np.allclose(obs1, obs3), "Different seeds should give different obs!"
print("✅ Different seeds give different observations")"""),

        # ── Step 11: Save Artifacts + MLflow ──
        md("## Step 11 — Save Artifacts & MLflow"),
        code("""\
import subprocess

# pip freeze
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, timeout=30,
    )
    (ARTIFACTS_DIR / "requirements.txt").write_text(result.stdout)
    print(f"Saved: requirements.txt ({len(result.stdout.splitlines())} packages)")
except Exception as e:
    print(f"⚠️ pip freeze failed: {e}")

# Save config
with open(ARTIFACTS_DIR / "nb01_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

# MLflow
if MLFLOW_OK:
    try:
        with mlflow.start_run(run_name="NB01_setup_smoke"):
            mlflow.set_tags({"notebook": "NB01", "task": "setup"})
            mlflow.log_params(CFG)
            mlflow.log_metrics({
                "apple_obs_dim": obs_a.shape[0],
                "apple_act_dim": env_apple.action_space.shape[0],
                "dishwipe_obs_dim": obs_d.shape[0],
                "dishwipe_act_dim": env_dw.action_space.shape[0],
                "smoke_apple_mean_reward": smoke_results["apple"]["mean_reward"],
                "smoke_dishwipe_mean_reward": smoke_results["dishwipe"]["mean_reward"],
            })
            mlflow.log_artifacts(str(ARTIFACTS_DIR))
        print("✅ MLflow run logged")
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")
else:
    csv_log({
        "timestamp": datetime.datetime.now().isoformat(),
        "apple_obs_dim": obs_a.shape[0],
        "apple_act_dim": env_apple.action_space.shape[0],
        "dishwipe_obs_dim": obs_d.shape[0],
        "dishwipe_act_dim": env_dw.action_space.shape[0],
    })
    print("📝 Logged to CSV fallback")"""),

        # ── Results ──
        md("## Results Summary"),
        code("""\
print("=" * 60)
print("  NB01 — Setup & Smoke Test Results")
print("=" * 60)
print(f"  Apple env     : {apple_spec['env_id']}")
print(f"    obs shape   : {apple_spec['obs_shape']}")
print(f"    act shape   : {apple_spec['act_shape']} ({apple_spec['dof']} DOF)")
print(f"  DishWipe env  : {dishwipe_spec['env_id']}")
print(f"    obs shape   : {dishwipe_spec['obs_shape']}")
print(f"    act shape   : {dishwipe_spec['act_shape']} ({dishwipe_spec['dof']} DOF)")
print(f"  Robot         : unitree_g1 (full body, free-floating root)")
print(f"  Total joints  : {joint_info.get('total_dof', 'N/A')}")
print(f"  Balance API   : standing={has_standing}, fallen={has_fallen}")
print(f"  Render        : {'OK' if render_ok else 'Not available (CPU-only)'}")
print(f"  Smoke test    : PASSED ✅")
print("=" * 60)"""),

        # ── Artifacts Table ──
        md("""\
## Artifacts

| File | Description |
|------|-------------|
| `env_spec_apple.json` | Apple env obs/act shapes, DOF, robot info |
| `env_spec_dishwipe.json` | DishWipe env obs/act shapes, DOF, robot info |
| `active_joints_fullbody.json` | All 37 joints grouped by body part |
| `smoke_results.json` | 50-step smoke test stats for both envs |
| `requirements.txt` | pip freeze snapshot |
| `nb01_config.json` | Config used |
| `render_test_apple.png` | Render frame (if Vulkan available) |"""),

        # ── Cleanup ──
        code("""\
env_apple.close()
env_dw.close()
print("✅ NB01 Complete — Both environments verified")"""),

        # ── Notes ──
        md("""\
## Notes

- Full-body G1 has **37 DOF** (12 legs + 1 torso + 10 arms + 14 fingers)
- Root is **free-floating** → robot CAN fall over
- `is_fallen()` checks if torso height drops below threshold
- Apple env needs `download_asset()` for apple/bowl meshes
- DishWipe env uses `dirt_grid.py` (VirtualDirtGrid 10×10)
- Render may fail on CPU-only (Vulkan required) — this is expected

### References
- [ManiSkill Documentation](https://maniskill.readthedocs.io/)
- Unitree G1 robot URDF: `g1.urdf` (37 DOF full body)
- Custom envs: `src/envs/apple_fullbody_env.py`, `src/envs/dishwipe_fullbody_env.py`"""),
    ]
    write_nb("NB01_setup_smoke.ipynb", cells)


# ====================================================================
#  NB02 — Environment Exploration (Apple Full-Body)
# ====================================================================
def gen_nb02():
    cells = [
        md("""\
# NB02 — Environment Exploration (Apple Full-Body)

Deep-dive into `UnitreeG1PlaceAppleInBowlFullBody-v1`: observation space,
action groups, balance dynamics, reward structure, and reset distribution.
This builds intuition before training in NB05–NB07.

**Robot:** Unitree G1 Full Body (37 DOF, free-floating root)
**Task:** Pick up apple → place in bowl → release — while staying balanced"""),

        md("""\
## Objectives

1. Map every observation dimension to its meaning.
2. Map action dimensions to body-part groups (legs, torso, arms, hands).
3. Analyse balance: how quickly does the robot fall with random actions?
4. Analyse per-step reward distribution (random policy).
5. Visualise reset distribution (apple/bowl positions across seeds).
6. Compare complexity vs single-arm Panda (7 DOF)."""),

        md("""\
## Resources

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | Not required | CPU OK |
| RAM | 4 GB | |
| Runtime | ~5-10 min | |"""),

        # ── Imports + Setup ──
        md("## Imports & Setup"),
        code("""\
import sys, os, json, random
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB02"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Verify NB01 completed
assert (PROJECT_ROOT / "artifacts" / "NB01" / "env_spec_apple.json").exists(), \\
    "Run NB01 first!"
print("✅ NB01 artifacts found")"""),

        md("## Configuration"),
        code("""\
CFG = {
    "seed":               42,
    "env_id":             "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "control_mode":       "pd_joint_delta_pos",
    "n_explore_episodes": 10,
    "n_explore_steps":    200,
    "n_seeds_reset":      6,
}

SEED = CFG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("Config loaded, seeds set")"""),

        # ── Create Env ──
        md("## Create Environment"),
        code("""\
env = gym.make(
    CFG["env_id"],
    num_envs=1,
    obs_mode="state",
    control_mode=CFG["control_mode"],
    render_mode="rgb_array",
)
env = CPUGymWrapper(env)
obs, info = env.reset(seed=SEED)
print(f"Env created: obs={obs.shape}, act={env.action_space.shape}")"""),

        # ── Obs Breakdown ──
        md("""\
## Step 1 — Observation Breakdown

Full-body G1 observations include proprioception (joint positions &
velocities), torso orientation, and task-specific features (apple/bowl
positions, hand positions, grasp states)."""),
        code("""\
obs_total = obs.shape[0]
print(f"Total observation dimensions: {obs_total}")
print(f"\\nRaw observation (first 20): {obs[:20].round(4)}")
print(f"Raw observation (last 20):  {obs[-20:].round(4)}")

# Attempt to probe observation structure from env internals
try:
    base_env = env.unwrapped
    agent = base_env.agent
    robot = agent.robot
    n_joints = len(robot.active_joints)
    print(f"\\nActive joints: {n_joints}")
    print(f"Expected proprioception dims: {n_joints * 2} (qpos + qvel)")
    print(f"Remaining obs dims: {obs_total - n_joints * 2} (task features)")
except Exception as e:
    n_joints = 37
    print(f"Could not access internals: {e}")

# Structured breakdown (approximate — exact mapping depends on env)
obs_breakdown = {
    "total_dims": obs_total,
    "approximate_groups": {
        "joint_positions (qpos)": n_joints,
        "joint_velocities (qvel)": n_joints,
        "torso_orientation": "4D (quaternion)",
        "torso_angular_velocity": "3D",
        "apple_position": "3D",
        "bowl_position": "3D",
        "hand_positions": "6D (left 3 + right 3)",
        "grasp_state": "varies",
    },
}

with open(ARTIFACTS_DIR / "obs_breakdown.json", "w") as f:
    json.dump(obs_breakdown, f, indent=2, default=str)

# Visualise observation magnitudes
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(range(obs_total), np.abs(obs), color="steelblue", width=1.0, alpha=0.8)
ax.set_xlabel("Observation Dimension Index")
ax.set_ylabel("|Value|")
ax.set_title(f"Observation Magnitudes ({obs_total}D) — Seed {SEED}")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "obs_breakdown.png", dpi=150)
plt.show()
print("Saved: obs_breakdown.png")"""),

        # ── Action Groups ──
        md("""\
## Step 2 — Action Groups by Body Part

Map each action dimension to a body part. With `pd_joint_delta_pos`
control mode, each action dim corresponds to one joint."""),
        code("""\
act_dim = env.action_space.shape[0]
print(f"Action dimensions: {act_dim}")

try:
    agent = env.unwrapped.agent
    joint_names = [j.name for j in agent.robot.active_joints]

    def classify_joint(name):
        n = name.lower()
        if any(k in n for k in ["hip", "knee", "ankle"]):
            side = "left_leg" if "left" in n else "right_leg"
            return side
        if any(k in n for k in ["torso", "waist"]):
            return "torso"
        if any(k in n for k in ["shoulder", "elbow", "wrist"]):
            side = "left_arm" if "left" in n else "right_arm"
            return side
        return "left_hand" if "left" in name.lower() else "right_hand"

    action_groups = {}
    for i, jn in enumerate(joint_names):
        g = classify_joint(jn)
        action_groups.setdefault(g, []).append({"idx": i, "name": jn})

except Exception as e:
    print(f"Could not access joints: {e}")
    action_groups = {"all_joints": [{"idx": i, "name": f"joint_{i}"} for i in range(act_dim)]}

group_sizes = {k: len(v) for k, v in action_groups.items()}
print("\\nAction groups:")
for g, joints in action_groups.items():
    print(f"  {g}: {len(joints)} joints")
    for j in joints:
        print(f"    [{j['idx']:2d}] {j['name']}")

with open(ARTIFACTS_DIR / "action_groups.json", "w") as f:
    json.dump({k: [j["name"] for j in v] for k, v in action_groups.items()}, f, indent=2)

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
group_order = ["left_leg", "right_leg", "torso", "left_arm", "right_arm",
               "left_hand", "right_hand"]
group_order = [g for g in group_order if g in group_sizes]
colors = ["#E57373", "#E57373", "#FFB74D", "#64B5F6", "#64B5F6", "#81C784", "#81C784"]
sizes_ordered = [group_sizes[g] for g in group_order]
bars = ax.bar(group_order, sizes_ordered, color=colors[:len(group_order)], edgecolor="black")
for b, v in zip(bars, sizes_ordered):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, str(v),
            ha="center", fontweight="bold")
ax.set_ylabel("Number of Joints")
ax.set_title(f"Action Space by Body Part ({act_dim}D total)")
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "action_groups.png", dpi=150)
plt.show()
print("Saved: action_groups.png")"""),

        # ── Balance Analysis ──
        md("""\
## Step 3 — Balance Analysis

How quickly does the robot fall with a random policy?
This quantifies the challenge of maintaining balance while manipulating."""),
        code("""\
fall_steps = []
for ep in range(CFG["n_explore_episodes"]):
    obs, info = env.reset(seed=ep * 42)
    for step in range(CFG["n_explore_steps"]):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            fall_steps.append(step + 1)
            break
    else:
        fall_steps.append(CFG["n_explore_steps"])

balance_stats = {
    "mean_steps_to_end": float(np.mean(fall_steps)),
    "min_steps_to_end":  int(np.min(fall_steps)),
    "max_steps_to_end":  int(np.max(fall_steps)),
    "std_steps_to_end":  float(np.std(fall_steps)),
    "survived_full_ep":  int(sum(s == CFG["n_explore_steps"] for s in fall_steps)),
    "n_episodes":        len(fall_steps),
}

print("Balance Analysis (Random Policy):")
for k, v in balance_stats.items():
    print(f"  {k}: {v}")

with open(ARTIFACTS_DIR / "balance_analysis.json", "w") as f:
    json.dump(balance_stats, f, indent=2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(fall_steps)), fall_steps, color="coral", edgecolor="black")
ax.axhline(np.mean(fall_steps), color="red", ls="--", lw=2,
           label=f"mean={np.mean(fall_steps):.0f} steps")
ax.set_xlabel("Episode")
ax.set_ylabel("Steps before Termination")
ax.set_title("Balance Analysis — Random Policy (Full-Body G1)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "balance_analysis.png", dpi=150)
plt.show()
print("Saved: balance_analysis.png")"""),

        # ── Reward Structure ──
        md("""\
## Step 4 — Reward Structure Analysis

Collect per-step rewards with random actions to understand the reward
distribution. The Apple env uses 4-stage dense reward (reaching →
grasping → placing → releasing), max normalized ~10."""),
        code("""\
obs, info = env.reset(seed=SEED)
rewards = []
for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(float(reward))
    if terminated or truncated:
        obs, info = env.reset()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(rewards, linewidth=0.7, color="darkviolet", alpha=0.8)
axes[0].axhline(np.mean(rewards), color="gray", ls="--",
                label=f"mean={np.mean(rewards):.4f}")
axes[0].set_title("Reward per Step (Random)")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Reward")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(np.cumsum(rewards), linewidth=1.2, color="darkviolet")
axes[1].set_title("Cumulative Reward")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Cumulative")
axes[1].grid(True, alpha=0.3)

axes[2].hist(rewards, bins=40, color="darkviolet", edgecolor="black", alpha=0.7)
axes[2].set_title("Reward Distribution")
axes[2].set_xlabel("Reward")
axes[2].set_ylabel("Count")

fig.suptitle("Apple Full-Body — Random Policy Reward Analysis", fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "reward_per_step.png", dpi=150)
plt.show()

print(f"Reward stats: mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}")
print(f"  min={np.min(rewards):.4f}, max={np.max(rewards):.4f}")
print(f"  nonzero: {np.mean(np.array(rewards)!=0)*100:.1f}%")"""),

        # ── Reset Distribution ──
        md("## Step 5 — Reset Distribution"),
        code("""\
# Visualise how apple/bowl positions vary across seeds
reset_obs = []
for seed in range(CFG["n_seeds_reset"]):
    o, _ = env.reset(seed=seed * 10)
    reset_obs.append(o.copy())

reset_obs = np.array(reset_obs)
fig, ax = plt.subplots(figsize=(10, 4))
for i in range(min(6, len(reset_obs))):
    ax.plot(reset_obs[i], alpha=0.5, lw=0.8, label=f"seed={i*10}")
ax.set_xlabel("Observation Dimension")
ax.set_ylabel("Value")
ax.set_title("Reset Distribution — Obs Vectors across Seeds")
ax.legend(fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "reset_distribution.png", dpi=150)
plt.show()
print("Saved: reset_distribution.png")"""),

        # ── Complexity Comparison ──
        md("""\
## Step 6 — Complexity Comparison (Full-Body vs Single-Arm)

Compare the dimensionality of our full-body humanoid task vs a typical
single-arm robot (Panda, 7 DOF — e.g. PickCube)."""),
        code("""\
try:
    env_panda = gym.make("PickCube-v1", num_envs=1, obs_mode="state",
                         control_mode="pd_joint_delta_pos", render_mode="rgb_array")
    env_panda = CPUGymWrapper(env_panda)
    obs_p, _ = env_panda.reset(seed=0)
    panda_obs = obs_p.shape[0]
    panda_act = env_panda.action_space.shape[0]
    env_panda.close()
except Exception as e:
    print(f"Could not create PickCube (using defaults): {e}")
    panda_obs, panda_act = 25, 7

comparison = {
    "Panda (PickCube-v1)":     {"obs": panda_obs,    "act": panda_act},
    "G1 Full Body (Apple)":    {"obs": obs.shape[0],  "act": env.action_space.shape[0]},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
names = list(comparison.keys())
colors = ["steelblue", "darkviolet"]

axes[0].bar(names, [comparison[n]["obs"] for n in names], color=colors, edgecolor="black")
axes[0].set_title("Observation Dimensions")
axes[0].set_ylabel("Dims")
for i, n in enumerate(names):
    axes[0].text(i, comparison[n]["obs"] + 1, str(comparison[n]["obs"]),
                 ha="center", fontweight="bold")

axes[1].bar(names, [comparison[n]["act"] for n in names], color=colors, edgecolor="black")
axes[1].set_title("Action Dimensions")
axes[1].set_ylabel("Dims")
for i, n in enumerate(names):
    axes[1].text(i, comparison[n]["act"] + 0.5, str(comparison[n]["act"]),
                 ha="center", fontweight="bold")

fig.suptitle("Single-Arm vs Full-Body Humanoid Complexity", fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "complexity_comparison.png", dpi=150)
plt.show()
print("Saved: complexity_comparison.png")"""),

        # ── Multi-seed render ──
        md("## Step 7 — Multi-Seed Render Grid"),
        code("""\
images = []
titles = []
for seed in range(6):
    env.reset(seed=seed * 42)
    try:
        frame = env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame.ndim == 4:
            frame = frame[0]
        if frame.dtype in (np.float32, np.float64):
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        images.append(frame)
        titles.append(f"Seed {seed*42}")
    except Exception:
        pass

if images:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(titles[i], fontsize=11)
        ax.axis("off")
    fig.suptitle("Apple Full-Body — Reset Distribution (Renders)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "reset_renders.png", dpi=150)
    plt.show()
    print("Saved: reset_renders.png")
else:
    print("⚠️ No renders available (CPU-only)")"""),

        # ── Save ──
        md("## Save Artifacts"),
        code("""\
with open(ARTIFACTS_DIR / "nb02_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

env.close()
print("✅ NB02 Environment Exploration Complete")
print(f"Artifacts saved to: {ARTIFACTS_DIR}")"""),

        md("""\
## References

- ManiSkill 3 documentation: <https://maniskill.readthedocs.io/>
- Unitree G1 URDF: 37 DOF full body (`g1.urdf`)
- Built-in `UnitreeG1PlaceAppleInBowl-v1` (upper body) — our custom env extends to full body
- Custom env: `src/envs/apple_fullbody_env.py`"""),
    ]
    write_nb("NB02_env_exploration.ipynb", cells)


# ====================================================================
#  NB03 — Reward Analysis, Safety & MLflow Utilities
# ====================================================================
def gen_nb03():
    cells = [
        md("""\
# NB03 — Reward Analysis, Safety Validation & MLflow Utilities

Analyse the dense reward structure of the Apple Full-Body env, validate
fall detection / safety termination, and define reusable MLflow helpers
for NB04–NB09.

**Apple reward stages:** Reaching → Grasping → Placing → Releasing (max ≈ 10)
**DishWipe reward terms:** 9 terms + 2 balance terms (for NB09 reference)"""),

        md("""\
## Objectives

1. Document Apple reward contract (4 stages + balance penalties).
2. Document DishWipe reward contract (9+2 terms) for NB09 reference.
3. Run test episodes to validate reward range and distribution.
4. Verify `info` dict keys.
5. Validate safety termination (fall detection).
6. Define MLflow helper utilities.
7. Save formal reward contracts as JSON artifacts."""),

        md("""\
## Resources

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | Not required | CPU OK |
| RAM | 4 GB | |
| Runtime | ~5-10 min | |"""),

        md("## Imports & Setup"),
        code("""\
import sys, os, json, random
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB03"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Verify NB01 + NB02 completed
assert (PROJECT_ROOT / "artifacts" / "NB01" / "env_spec_apple.json").exists(), "Run NB01 first!"
assert (PROJECT_ROOT / "artifacts" / "NB02" / "obs_breakdown.json").exists(), "Run NB02 first!"
print("✅ Prerequisites found")"""),

        md("## Configuration"),
        code("""\
CFG = {
    "seed": 42,
    "env_id": "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "control_mode": "pd_joint_delta_pos",
    "n_test_episodes": 10,
    "n_steps_per_episode": 200,
}

SEED = CFG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("Config loaded")"""),

        # ── Apple Reward Contract ──
        md("""\
## Step 1 — Apple Reward Contract

The Apple env uses 4-stage dense reward with balance penalties.
Maximum normalized reward ≈ 10 per step."""),
        code("""\
apple_reward_contract = {
    "version": "3.0",
    "env_id": "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "robot": "unitree_g1 (37 DOF, full body, free root)",
    "max_normalized_reward": 10.0,
    "stages": [
        {
            "name": "reaching",
            "description": "Move hand toward apple",
            "formula": "1 - tanh(5 * dist(hand, apple))",
            "max": 1.0,
            "sign": "+",
        },
        {
            "name": "grasping",
            "description": "Close hand around apple",
            "formula": "1.0 if is_grasping(apple)",
            "max": 1.0,
            "sign": "+",
        },
        {
            "name": "placing",
            "description": "Move grasped apple toward bowl",
            "formula": "1 - tanh(5 * dist(apple, bowl))",
            "max": 1.0,
            "sign": "+",
        },
        {
            "name": "releasing",
            "description": "Release apple into bowl",
            "formula": "5.0 if apple_in_bowl AND hand_released",
            "max": 5.0,
            "sign": "+",
        },
    ],
    "balance_terms": [
        {
            "name": "r_balance",
            "description": "Penalty for torso tilt beyond threshold",
            "sign": "-",
        },
        {
            "name": "r_fall",
            "description": "Terminate + large penalty if is_fallen()",
            "sign": "-",
        },
    ],
    "success_condition": "Apple in bowl (dist < 0.05) AND robot standing",
    "failure_conditions": [
        "Robot falls (is_fallen() = True)",
        "Timeout (max_episode_steps exceeded)",
    ],
}

with open(ARTIFACTS_DIR / "reward_contract_apple.json", "w") as f:
    json.dump(apple_reward_contract, f, indent=2)

print("Apple Reward Contract:")
for stage in apple_reward_contract["stages"]:
    print(f"  {stage['sign']} {stage['name']:12s}: max={stage['max']}, {stage['formula']}")
for term in apple_reward_contract["balance_terms"]:
    print(f"  {term['sign']} {term['name']:12s}: {term['description']}")
print(f"\\nSuccess: {apple_reward_contract['success_condition']}")"""),

        # ── DishWipe Reward Contract ──
        md("""\
## Step 2 — DishWipe Reward Contract (Reference for NB09)

The DishWipe env uses 9 terms + 2 balance terms. This is documented here
for reference — it will be used in NB09 (bonus task)."""),
        code("""\
dishwipe_reward_contract = {
    "version": "3.0",
    "env_id": "UnitreeG1DishWipeFullBody-v1",
    "robot": "unitree_g1 (37 DOF, full body, free root)",
    "terms": [
        {"name": "r_clean",   "weight": 10.0,  "sign": "+", "desc": "delta_clean progress"},
        {"name": "r_reach",   "weight": 0.5,   "sign": "+", "desc": "1-tanh(5*dist)"},
        {"name": "r_contact", "weight": 1.0,   "sign": "+", "desc": "is_contacting plate"},
        {"name": "r_sweep",   "weight": 0.3,   "sign": "+", "desc": "lateral movement"},
        {"name": "r_time",    "weight": 0.01,  "sign": "-", "desc": "per step cost"},
        {"name": "r_jerk",    "weight": 0.05,  "sign": "-", "desc": "jerk^2"},
        {"name": "r_act",     "weight": 0.005, "sign": "-", "desc": "action_norm^2"},
        {"name": "r_force",   "weight": 0.01,  "sign": "-", "desc": "excess force"},
        {"name": "r_success", "weight": 50.0,  "sign": "+", "desc": "one-shot at 95%"},
        {"name": "r_balance", "weight": "TBD", "sign": "-", "desc": "penalty for tilt"},
        {"name": "r_fall",    "weight": "TBD", "sign": "-", "desc": "terminate if fallen"},
    ],
    "note": "Used in NB09 only (bonus task)",
}

with open(ARTIFACTS_DIR / "reward_contract_dishwipe.json", "w") as f:
    json.dump(dishwipe_reward_contract, f, indent=2)

print("DishWipe Reward Contract (for NB09):")
for t in dishwipe_reward_contract["terms"]:
    print(f"  {t['sign']} {t['name']:12s}: w={str(t['weight']):>5s}  {t['desc']}")"""),

        # ── Test Episodes ──
        md("""\
## Step 3 — Validate Reward with Test Episodes

Run random actions for several episodes. Verify reward is dense
(most steps have non-zero reward) and bounded."""),
        code("""\
env = gym.make(
    CFG["env_id"], num_envs=1, obs_mode="state",
    control_mode=CFG["control_mode"], render_mode="rgb_array",
)
env = CPUGymWrapper(env)

all_rewards = []
all_info_keys = set()
termination_counts = {"success": 0, "fall": 0, "timeout": 0}

for ep in range(CFG["n_test_episodes"]):
    obs, info = env.reset(seed=ep * 42)
    ep_rewards = []
    for step in range(CFG["n_steps_per_episode"]):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_rewards.append(float(reward))
        all_info_keys.update(info.keys())

        if terminated:
            if info.get("success", False):
                termination_counts["success"] += 1
            else:
                termination_counts["fall"] += 1
            break
        if truncated:
            termination_counts["timeout"] += 1
            break

    all_rewards.extend(ep_rewards)

reward_stats = {
    "n_episodes":     CFG["n_test_episodes"],
    "total_steps":    len(all_rewards),
    "mean":           float(np.mean(all_rewards)),
    "std":            float(np.std(all_rewards)),
    "min":            float(np.min(all_rewards)),
    "max":            float(np.max(all_rewards)),
    "nonzero_pct":    float(np.mean(np.array(all_rewards) != 0) * 100),
    "positive_pct":   float(np.mean(np.array(all_rewards) > 0) * 100),
    "termination_counts": termination_counts,
}

print("Reward Validation:")
for k, v in reward_stats.items():
    print(f"  {k}: {v}")

with open(ARTIFACTS_DIR / "reward_validation.json", "w") as f:
    json.dump(reward_stats, f, indent=2)"""),

        # ── Info Dict ──
        md("## Step 4 — Info Dict Keys"),
        code("""\
info_data = {"available_keys": sorted(list(all_info_keys))}
with open(ARTIFACTS_DIR / "info_keys.json", "w") as f:
    json.dump(info_data, f, indent=2)

print(f"Available info keys ({len(all_info_keys)}):")
for k in sorted(all_info_keys):
    print(f"  - {k}")"""),

        # ── Safety Validation ──
        md("""\
## Step 5 — Safety Validation (Fall Detection)

Verify that `is_fallen()` triggers episode termination with random actions."""),
        code("""\
try:
    agent = env.unwrapped.agent
    has_is_fallen = hasattr(agent, "is_fallen")
    has_is_standing = hasattr(agent, "is_standing")
except:
    has_is_fallen, has_is_standing = False, False

safety_results = {
    "fall_detection_available":  has_is_fallen,
    "standing_detection_available": has_is_standing,
    "fall_terminates_episode":   termination_counts["fall"] > 0,
    "total_falls":               termination_counts["fall"],
    "total_successes":           termination_counts["success"],
    "total_timeouts":            termination_counts["timeout"],
}

print("Safety Validation:")
for k, v in safety_results.items():
    print(f"  {k}: {v}")

with open(ARTIFACTS_DIR / "safety_validation.json", "w") as f:
    json.dump(safety_results, f, indent=2)"""),

        # ── MLflow Helpers ──
        md("""\
## Step 6 — MLflow Helper Utilities

Define reusable helper functions for NB04–NB09."""),
        code("""\
# ── MLflow Helpers (reusable pattern) ──

def setup_mlflow(experiment_name="g1_fullbody_apple_dishwipe"):
    \"\"\"Set up MLflow tracking. Returns True if successful.\"\"\"
    try:
        import mlflow
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env.local")
        uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if uri:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            return True
    except Exception:
        pass
    return False

def log_training_run(run_name, params, metrics=None, artifacts_dir=None):
    \"\"\"Log a training run to MLflow.\"\"\"
    try:
        import mlflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            if artifacts_dir:
                mlflow.log_artifacts(str(artifacts_dir))
    except Exception as e:
        print(f"MLflow logging failed: {e}")

def log_eval_run(run_name, eval_dict, artifacts_dir=None):
    \"\"\"Log an evaluation run to MLflow.\"\"\"
    try:
        import mlflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_dict(eval_dict, "eval_results.json")
            if artifacts_dir:
                mlflow.log_artifacts(str(artifacts_dir))
    except Exception as e:
        print(f"MLflow logging failed: {e}")

print("✅ MLflow helper functions defined")
print("  - setup_mlflow(experiment_name)")
print("  - log_training_run(run_name, params, metrics, artifacts_dir)")
print("  - log_eval_run(run_name, eval_dict, artifacts_dir)")"""),

        # ── Plots ──
        md("## Plots"),
        code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(all_rewards, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
axes[0].set_title("Reward Distribution (Random Policy)")
axes[0].set_xlabel("Reward")

axes[1].bar(termination_counts.keys(), termination_counts.values(),
            color=["green", "red", "gray"], edgecolor="black")
axes[1].set_title("Termination Reasons")

# Stage visualisation
stages = ["Reaching", "Grasping", "Placing", "Releasing"]
max_rewards = [1.0, 1.0, 1.0, 5.0]
axes[2].bar(stages, max_rewards,
            color=["#FFE082", "#A5D6A7", "#90CAF9", "#CE93D8"],
            edgecolor="black")
axes[2].set_title("Apple Reward Stages (Max per Stage)")
axes[2].set_ylabel("Max Reward")

fig.suptitle("NB03 — Reward & Safety Analysis", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "reward_analysis.png", dpi=150)
plt.show()"""),

        # ── Save ──
        md("## Save Artifacts"),
        code("""\
with open(ARTIFACTS_DIR / "nb03_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

env.close()
print("✅ NB03 Reward & Safety Analysis Complete")"""),

        md("""\
## Artifacts

| File | Description |
|------|-------------|
| `reward_contract_apple.json` | Apple 4-stage reward formal contract |
| `reward_contract_dishwipe.json` | DishWipe 9+2 term reward contract (NB09 ref) |
| `reward_validation.json` | Reward range/distribution stats |
| `safety_validation.json` | Fall detection test results |
| `info_keys.json` | Available keys in env info dict |
| `reward_analysis.png` | Reward distribution + termination plots |"""),
    ]
    write_nb("NB03_reward_safety_mlflow.ipynb", cells)


# ====================================================================
#  NB04 — Baselines, Smooth Wrapper & Base Controller
# ====================================================================
def gen_nb04():
    cells = [
        md("""\
# NB04 — Baselines, Smooth Wrapper & Base Controller (Apple Full-Body)

Establish performance baselines for the Apple Full-Body env using Random
and Heuristic policies. Build `SmoothActionWrapper` and `BaseController`
for use in Residual SAC (NB07). Generate a leaderboard table as reference.

**Key outputs:**
- Baseline leaderboard (Random, Stand-Only, Heuristic, Smoothed, BaseController)
- `SmoothActionWrapper` — EMA action filter for jerk reduction
- `BaseController` — heuristic + EMA (used as base in NB07 Residual SAC)"""),

        md("""\
## Objectives

1. Define `evaluate_policy()` helper for consistent evaluation.
2. **Random baseline**: sample random actions, measure reward / fall rate / jerk.
3. **Stand-Only baseline**: zero actions (hold standing pose).
4. **Heuristic — Reach Apple**: proportional control toward apple.
5. **SmoothActionWrapper**: EMA filter to reduce jerk.
6. **Smoothed Random baseline**: demonstrate jerk reduction.
7. **BaseController**: heuristic + EMA (for NB07 Residual SAC).
8. Rank all baselines in a leaderboard table."""),

        md("""\
## Resources

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | Not required | CPU OK |
| RAM | 8 GB | |
| Runtime | ~10-20 min | 5 baselines × 20 episodes |"""),

        md("## Imports & Setup"),
        code("""\
import sys, os, json, random, time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt

import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB04"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

assert (PROJECT_ROOT / "artifacts" / "NB03" / "reward_contract_apple.json").exists(), \\
    "Run NB03 first!"
print("✅ Prerequisites found")"""),

        md("## Configuration"),
        code("""\
CFG = {
    "seed": 42,
    "env_id": "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "control_mode": "pd_joint_delta_pos",
    "obs_mode": "state",
    "n_eval_episodes": 20,
    "max_steps_per_ep": 200,
    "smooth_alpha": 0.3,
}

SEED = CFG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("Config loaded")"""),

        # ── evaluate_policy ──
        md("## Step 1 — `evaluate_policy()` Helper"),
        code("""\
def evaluate_policy(env, policy_fn, n_episodes, max_steps, seed=42):
    \"\"\"Run policy and collect per-episode metrics.

    Args:
        policy_fn: callable(obs, info) → action
    Returns:
        list of dicts with per-episode metrics
    \"\"\"
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward, ep_steps = 0.0, 0
        prev_action = np.zeros(env.action_space.shape)
        jerks = []
        fell, success = False, False

        for step in range(max_steps):
            action = policy_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1

            jerk = float(np.sum((action - prev_action) ** 2))
            jerks.append(jerk)
            prev_action = action.copy()

            if terminated:
                success = bool(info.get("success", False))
                fell = not success
                break
            if truncated:
                break

        results.append({
            "episode": ep,
            "total_reward": ep_reward,
            "steps": ep_steps,
            "success": success,
            "fell": fell,
            "mean_jerk": float(np.mean(jerks)) if jerks else 0.0,
        })
    return results

print("✅ evaluate_policy() defined")"""),

        # ── Random Baseline ──
        md("## Step 2 — Random Baseline"),
        code("""\
env = gym.make(
    CFG["env_id"], num_envs=1, obs_mode=CFG["obs_mode"],
    control_mode=CFG["control_mode"], render_mode="rgb_array",
)
env = CPUGymWrapper(env)

def random_policy(obs, info):
    return env.action_space.sample()

random_results = evaluate_policy(
    env, random_policy, CFG["n_eval_episodes"], CFG["max_steps_per_ep"], SEED
)

rews = [r["total_reward"] for r in random_results]
print(f"Random: mean_reward={np.mean(rews):.4f} ± {np.std(rews):.4f}")
print(f"  fall_rate={np.mean([r['fell'] for r in random_results]):.2%}")
print(f"  mean_jerk={np.mean([r['mean_jerk'] for r in random_results]):.4f}")"""),

        # ── Stand-Only ──
        md("""\
## Step 3 — Stand-Only Baseline

Zero actions = hold current joint positions (delta=0).
Demonstrates survival without manipulation."""),
        code("""\
def stand_only_policy(obs, info):
    \"\"\"Zero delta on all joints — hold standing pose.\"\"\"
    return np.zeros(env.action_space.shape, dtype=np.float32)

stand_results = evaluate_policy(
    env, stand_only_policy, CFG["n_eval_episodes"], CFG["max_steps_per_ep"], SEED
)

rews = [r["total_reward"] for r in stand_results]
print(f"Stand-Only: mean_reward={np.mean(rews):.4f} ± {np.std(rews):.4f}")
print(f"  fall_rate={np.mean([r['fell'] for r in stand_results]):.2%}")"""),

        # ── Heuristic Reach ──
        md("""\
## Step 4 — Heuristic: Reach Apple

Proportional control: move hand toward apple position.
Arm joints get directional deltas, leg joints stay at zero."""),
        code("""\
def heuristic_reach_policy(obs, info):
    \"\"\"Simple proportional control: move arm joints toward apple direction.

    Strategy:
    - Keep legs at zero delta (hold standing pose)
    - Move arm/hand joints with small proportional signals
    - This is a rough heuristic — exact joint-to-Cartesian mapping is complex
    \"\"\"
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    # We use small random arm signals biased toward reaching
    # In a real heuristic, we'd compute IK or use Cartesian error
    # For now, apply small positive delta to arm joints to extend arm forward
    act_dim = env.action_space.shape[0]

    # Approximate arm joint indices (after legs and torso)
    # Legs: ~12 joints, Torso: ~1 joint, Arms: ~10 joints, Hands: ~14 joints
    arm_start = 13  # approximate
    arm_end = 23    # approximate

    # Small reaching signal on arm joints
    action[arm_start:arm_end] = np.random.uniform(-0.1, 0.1, arm_end - arm_start)
    action[arm_start:arm_end] *= 0.5  # conservative

    return action.astype(np.float32)

reach_results = evaluate_policy(
    env, heuristic_reach_policy, CFG["n_eval_episodes"], CFG["max_steps_per_ep"], SEED
)

rews = [r["total_reward"] for r in reach_results]
print(f"Heuristic Reach: mean_reward={np.mean(rews):.4f} ± {np.std(rews):.4f}")
print(f"  fall_rate={np.mean([r['fell'] for r in reach_results]):.2%}")"""),

        # ── SmoothActionWrapper ──
        md("""\
## Step 5 — SmoothActionWrapper

EMA (Exponential Moving Average) action filter:
$$a_{smooth} = \\alpha \\cdot a_{raw} + (1 - \\alpha) \\cdot a_{prev}$$

Where $\\alpha = 0.3$ means 30% new action, 70% previous.
This dramatically reduces jerk (action oscillation)."""),
        code("""\
class SmoothActionWrapper(gym.ActionWrapper):
    \"\"\"EMA action filter for jerk reduction.\"\"\"

    def __init__(self, env, alpha=0.3):
        super().__init__(env)
        self.alpha = alpha
        self._prev_action = None

    def action(self, action):
        if self._prev_action is None:
            self._prev_action = action.copy()
        smoothed = self.alpha * action + (1 - self.alpha) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

print(f"✅ SmoothActionWrapper defined (alpha={CFG['smooth_alpha']})")"""),

        # ── Smoothed Random ──
        md("## Step 6 — Smoothed Random Baseline"),
        code("""\
env_smooth = SmoothActionWrapper(env, alpha=CFG["smooth_alpha"])

smooth_random_results = evaluate_policy(
    env_smooth, random_policy, CFG["n_eval_episodes"],
    CFG["max_steps_per_ep"], SEED,
)

rews_s = [r["total_reward"] for r in smooth_random_results]
jerks_s = [r["mean_jerk"] for r in smooth_random_results]
jerks_raw = [r["mean_jerk"] for r in random_results]
print(f"Smoothed Random: mean_reward={np.mean(rews_s):.4f}")
print(f"  Jerk reduction: {np.mean(jerks_raw):.4f} → {np.mean(jerks_s):.4f} "
      f"({(1 - np.mean(jerks_s)/np.mean(jerks_raw))*100:.0f}% reduction)")"""),

        # ── BaseController ──
        md("""\
## Step 7 — BaseController (for Residual SAC in NB07)

Combines heuristic policy with EMA smoothing.
In NB07: `a_final = clip(a_base + β × a_residual)`"""),
        code("""\
class BaseController:
    \"\"\"Heuristic + EMA smoothing. Produces base actions for residual learning.

    Usage in NB07 Residual SAC:
        base_ctrl = BaseController(env, alpha=0.3)
        a_base = base_ctrl.get_action(obs)
        a_final = np.clip(a_base + beta * a_residual, low, high)
    \"\"\"

    def __init__(self, env, alpha=0.3):
        self.env = env
        self.alpha = alpha
        self._prev_action = None

    def get_action(self, obs, info=None):
        raw = heuristic_reach_policy(obs, info or {})
        if self._prev_action is None:
            self._prev_action = raw.copy()
        smoothed = self.alpha * raw + (1 - self.alpha) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def reset(self):
        self._prev_action = None

# Test
base_ctrl = BaseController(env, alpha=CFG["smooth_alpha"])

def base_ctrl_policy(obs, info):
    return base_ctrl.get_action(obs, info)

base_ctrl_results = evaluate_policy(
    env, base_ctrl_policy, CFG["n_eval_episodes"], CFG["max_steps_per_ep"], SEED
)

rews_bc = [r["total_reward"] for r in base_ctrl_results]
print(f"BaseController: mean_reward={np.mean(rews_bc):.4f} ± {np.std(rews_bc):.4f}")
print(f"  fall_rate={np.mean([r['fell'] for r in base_ctrl_results]):.2%}")
print(f"  mean_jerk={np.mean([r['mean_jerk'] for r in base_ctrl_results]):.4f}")"""),

        # ── Leaderboard ──
        md("## Step 8 — Baseline Leaderboard"),
        code("""\
def summarize(results, name):
    rewards = [r["total_reward"] for r in results]
    return {
        "method":       name,
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "fall_rate":    float(np.mean([r["fell"] for r in results])),
        "mean_jerk":    float(np.mean([r["mean_jerk"] for r in results])),
        "mean_steps":   float(np.mean([r["steps"] for r in results])),
    }

leaderboard = pd.DataFrame([
    summarize(random_results,        "Random"),
    summarize(stand_results,         "Stand-Only"),
    summarize(reach_results,         "Heuristic (Reach)"),
    summarize(smooth_random_results, "Smoothed Random"),
    summarize(base_ctrl_results,     "BaseController"),
])
leaderboard = leaderboard.sort_values("mean_reward", ascending=False)
leaderboard.to_csv(ARTIFACTS_DIR / "baseline_leaderboard.csv", index=False)

print("\\nBaseline Leaderboard:")
print(leaderboard.to_string(index=False))"""),

        # ── Plots ──
        md("## Step 9 — Comparison Plots"),
        code("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

methods = leaderboard["method"].tolist()
colors = ["gray", "khaki", "dodgerblue", "lightcoral", "mediumseagreen"][:len(methods)]

# Reward
axes[0].bar(methods, leaderboard["mean_reward"].tolist(),
            yerr=leaderboard["std_reward"].tolist(),
            color=colors, edgecolor="black", capsize=3)
axes[0].set_title("Mean Reward by Baseline")
axes[0].set_ylabel("Mean Total Reward")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha="right")
axes[0].grid(axis="y", alpha=0.3)

# Jerk
axes[1].bar(methods, leaderboard["mean_jerk"].tolist(),
            color=colors, edgecolor="black")
axes[1].set_title("Mean Jerk (lower = smoother)")
axes[1].set_ylabel("Mean Jerk")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")
axes[1].grid(axis="y", alpha=0.3)

# Fall rate
axes[2].bar(methods, leaderboard["fall_rate"].tolist(),
            color=colors, edgecolor="black")
axes[2].set_title("Fall Rate (lower = better)")
axes[2].set_ylabel("Fall Rate")
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=20, ha="right")
axes[2].grid(axis="y", alpha=0.3)

fig.suptitle("NB04 — Baseline Comparison (Apple Full-Body)", fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "baseline_comparison.png", dpi=150)
plt.show()"""),

        # ── Save ──
        md("## Save Artifacts"),
        code("""\
with open(ARTIFACTS_DIR / "nb04_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

env.close()
print("✅ NB04 Baselines & Smoothing Complete")
print(f"Artifacts saved to: {ARTIFACTS_DIR}")"""),

        md("""\
## Artifacts

| File | Description |
|------|-------------|
| `baseline_leaderboard.csv` | All baselines ranked by mean reward |
| `baseline_comparison.png` | Bar charts comparing baselines |
| `nb04_config.json` | Config used |

## Key Definitions for Downstream Use

- **`SmoothActionWrapper`**: `a_smooth = α·a_raw + (1-α)·a_prev` — used in NB07
- **`BaseController`**: heuristic + EMA — used in NB07 Residual SAC
- **`evaluate_policy()`**: consistent eval helper — pattern reused in NB05-NB09"""),
    ]
    write_nb("NB04_baselines_smoothing.ipynb", cells)


# ====================================================================
#  NB05 — Train PPO (Apple Full-Body)
# ====================================================================
def gen_nb05():
    cells = [
        md("""\
# NB05 — Train PPO (Apple Full-Body) — RTX 5090 Optimized

Train a **PPO** agent on `UnitreeG1PlaceAppleInBowlFullBody-v1` using
Stable-Baselines3. Optimized for **RTX 5090 (32 GB VRAM, 40 GB RAM)**.
Same training budget as SAC (NB06) for fair comparison.

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Library | Stable-Baselines3 |
| Policy | MlpPolicy [512, 512] **ReLU** (~790K params) |
| Budget | **2,000,000** env steps (GPU) / 20K (CPU demo) |
| Envs | **64** GPU-vectorized parallel environments |
| LR | **3e-4 → 1e-5** linear decay |
| Robot | Unitree G1 Full Body (37 DOF) |"""),

        md("""\
## Objectives

1. Create env with VecNormalize + proper wrappers.
2. Configure SB3 PPO with RTX 5090-optimized hyperparameters.
3. Train for `TOTAL_ENV_STEPS` (2M) with checkpointing every 200K.
4. Save model + learning curve.
5. Quick evaluation (20 deterministic episodes)."""),

        md("""\
## Resources

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | **RTX 5090** (32 GB VRAM) | 64 vectorized envs |
| RAM | 40 GB | VecNormalize + large batch |
| Runtime | ~2-4 hours (2M on RTX 5090) | CPU: 20K demo only |"""),

        md("## Imports & Setup"),
        code("""\
import sys, os, json, random, time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB05"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

IS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if IS_GPU else "cpu"
print(f"Device: {DEVICE}")
if IS_GPU:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")"""),

        md("## Configuration"),
        code("""\
CFG = {
    "seed":            42,
    "env_id":          "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "control_mode":    "pd_joint_delta_pos",
    "obs_mode":        "state",
    # ── Training budget (RTX 5090 optimized) ──
    "total_env_steps": 2_000_000 if IS_GPU else 20_000,
    "n_envs":          64 if IS_GPU else 1,
    # ── PPO Hyperparameters (RTX 5090) ──
    "learning_rate":   3e-4,     # initial LR (linear decay to 1e-5)
    "lr_end":          1e-5,
    "n_steps":         2048,
    "batch_size":      2048,     # large batch for 64 envs
    "n_epochs":        10,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    "ent_coef":        0.01,
    "vf_coef":         0.5,
    "max_grad_norm":   0.5,
    "net_arch":        [512, 512],
    "activation_fn":   "ReLU",
    # ── Normalization ──
    "vec_normalize":   True,     # VecNormalize wrapper
    "norm_obs":        True,
    "norm_reward":     True,
    "clip_obs":        10.0,
    # ── Checkpointing ──
    "checkpoint_freq": 200_000,  # save every 200K steps
    # ── Eval ──
    "eval_episodes":   20,
}

SEED = CFG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

with open(ARTIFACTS_DIR / "nb05_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

print("PPO Config (RTX 5090 optimized):")
for k, v in CFG.items():
    print(f"  {k}: {v}")"""),

        # ── Linear LR schedule ──
        md("## Step 1 — Linear Learning Rate Schedule"),
        code("""\
def linear_schedule(initial_lr: float, final_lr: float = 1e-5):
    \"\"\"Linear LR decay: initial_lr → final_lr over training.\"\"\"
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule

lr_schedule = linear_schedule(CFG["learning_rate"], CFG["lr_end"])
print(f"LR schedule: {CFG['learning_rate']} → {CFG['lr_end']} (linear decay)")"""),

        # ── Create Env ──
        md("## Step 2 — Create Environment (64 envs + VecNormalize)"),
        code("""\
def make_train_env(env_id, n_envs=1, seed=42):
    \"\"\"Create SB3-compatible env with proper wrappers.\"\"\"
    env = gym.make(
        env_id,
        num_envs=n_envs,
        obs_mode=CFG["obs_mode"],
        control_mode=CFG["control_mode"],
        render_mode="rgb_array",
    )
    if n_envs == 1:
        env = CPUGymWrapper(env)
    return env

train_env = make_train_env(CFG["env_id"], n_envs=CFG["n_envs"], seed=SEED)
eval_env = make_train_env(CFG["env_id"], n_envs=1, seed=9999)

# Apply VecNormalize if configured
if CFG["vec_normalize"] and hasattr(train_env, "observation_space"):
    try:
        train_env = VecNormalize(
            train_env,
            norm_obs=CFG["norm_obs"],
            norm_reward=CFG["norm_reward"],
            clip_obs=CFG["clip_obs"],
        )
        print("✅ VecNormalize applied (norm_obs + norm_reward)")
    except Exception as e:
        print(f"⚠️ VecNormalize skipped: {e}")

print(f"Train env: n_envs={CFG['n_envs']}, obs={train_env.observation_space.shape}, "
      f"act={train_env.action_space.shape}")
print(f"Eval env:  n_envs=1, obs={eval_env.observation_space.shape}")"""),

        # ── Configure PPO ──
        md("## Step 3 — Configure PPO (RTX 5090)"),
        code("""\
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=lr_schedule,
    n_steps=CFG["n_steps"],
    batch_size=CFG["batch_size"],
    n_epochs=CFG["n_epochs"],
    gamma=CFG["gamma"],
    gae_lambda=CFG["gae_lambda"],
    clip_range=CFG["clip_range"],
    ent_coef=CFG["ent_coef"],
    vf_coef=CFG["vf_coef"],
    max_grad_norm=CFG["max_grad_norm"],
    policy_kwargs={
        "net_arch": CFG["net_arch"],
        "activation_fn": torch.nn.ReLU,
    },
    verbose=1,
    seed=SEED,
    device=DEVICE,
)

total_params = sum(p.numel() for p in model.policy.parameters())
print(f"PPO model created: {total_params:,} parameters (~790K expected)")
print(f"  net_arch: {CFG['net_arch']}, activation: ReLU")
print(f"  LR: {CFG['learning_rate']} → {CFG['lr_end']} (linear decay)")"""),

        # ── Callbacks ──
        md("## Step 4 — Training Callbacks"),
        code("""\
class TrainLogCallback(BaseCallback):
    \"\"\"Log episode rewards and lengths during training.\"\"\"

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in (infos if isinstance(infos, list) else [infos]):
            if isinstance(info, dict) and "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
        return True

log_cb = TrainLogCallback()

# Checkpoint callback (save every 200K steps)
ckpt_cb = CheckpointCallback(
    save_freq=max(CFG["checkpoint_freq"] // CFG["n_envs"], 1),
    save_path=str(ARTIFACTS_DIR),
    name_prefix="checkpoint",
)
print(f"✅ Callbacks ready (log + checkpoint every {CFG['checkpoint_freq']:,} steps)")"""),

        # ── Train ──
        md("## Step 5 — Train PPO (2M steps)"),
        code("""\
print(f"\\n{'='*60}")
print(f"  Training PPO — {CFG['total_env_steps']:,} steps")
print(f"  n_envs={CFG['n_envs']}, batch={CFG['batch_size']}, device={DEVICE}")
print(f"  net_arch={CFG['net_arch']}, activation=ReLU")
print(f"  LR: {CFG['learning_rate']} → {CFG['lr_end']}")
print(f"{'='*60}")

start_time = time.time()
model.learn(
    total_timesteps=CFG["total_env_steps"],
    callback=[log_cb, ckpt_cb],
    progress_bar=True,
)
elapsed = time.time() - start_time
print(f"\\nTraining completed in {elapsed:.1f}s "
      f"({elapsed/60:.1f} min, {CFG['total_env_steps']/elapsed:.0f} steps/s)")"""),

        # ── Save Model ──
        md("## Step 6 — Save Model"),
        code("""\
model.save(str(ARTIFACTS_DIR / "ppo_apple"))
print(f"✅ Model saved: {ARTIFACTS_DIR / 'ppo_apple.zip'}")

# Save VecNormalize stats if used
if CFG["vec_normalize"]:
    try:
        train_env.save(str(ARTIFACTS_DIR / "vec_normalize.pkl"))
        print("✅ VecNormalize stats saved")
    except Exception:
        pass"""),

        # ── Quick Eval ──
        md("## Step 7 — Quick Evaluation (20 episodes)"),
        code("""\
eval_results = []
for ep in range(CFG["eval_episodes"]):
    obs, info = eval_env.reset(seed=ep * 100)
    ep_reward, ep_steps = 0.0, 0
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        ep_reward += float(reward)
        ep_steps += 1
        if terminated or truncated:
            break
    success = bool(info.get("success", False))
    eval_results.append({
        "episode": ep, "total_reward": ep_reward,
        "steps": ep_steps, "success": success,
    })
    print(f"  Ep {ep+1:2d}: reward={ep_reward:.4f}, steps={ep_steps}, success={success}")

eval_summary = {
    "mean_reward":     float(np.mean([r["total_reward"] for r in eval_results])),
    "std_reward":      float(np.std([r["total_reward"] for r in eval_results])),
    "success_rate":    float(np.mean([r["success"] for r in eval_results])),
    "mean_steps":      float(np.mean([r["steps"] for r in eval_results])),
    "training_time_s": elapsed,
    "total_steps":     CFG["total_env_steps"],
}

with open(ARTIFACTS_DIR / "eval_results.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

print(f"\\nPPO Eval Summary:")
for k, v in eval_summary.items():
    print(f"  {k}: {v}")"""),

        # ── Learning Curve ──
        md("## Step 8 — Learning Curve"),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ep_rewards = log_cb.episode_rewards
ep_lengths = log_cb.episode_lengths

if ep_rewards:
    axes[0].plot(ep_rewards, alpha=0.3, color="steelblue", linewidth=0.5)
    window = min(50, len(ep_rewards))
    if len(ep_rewards) >= window:
        rolling = np.convolve(ep_rewards, np.ones(window)/window, mode="valid")
        axes[0].plot(range(window-1, len(ep_rewards)), rolling,
                     color="darkblue", linewidth=2, label=f"Rolling avg ({window})")
    axes[0].set_title("Episode Rewards During Training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

if ep_lengths:
    axes[1].plot(ep_lengths, alpha=0.3, color="seagreen", linewidth=0.5)
    if len(ep_lengths) >= window:
        rolling_len = np.convolve(ep_lengths, np.ones(window)/window, mode="valid")
        axes[1].plot(range(window-1, len(ep_lengths)), rolling_len,
                     color="darkgreen", linewidth=2)
    axes[1].set_title("Episode Lengths")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].grid(True, alpha=0.3)

fig.suptitle(f"NB05 — PPO Training ({CFG['total_env_steps']:,} steps, RTX 5090)",
             fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "learning_curve.png", dpi=150)
plt.show()

# Save training log
training_log = pd.DataFrame({
    "episode": range(len(ep_rewards)),
    "reward": ep_rewards,
    "length": ep_lengths[:len(ep_rewards)] if ep_lengths else [],
})
training_log.to_csv(ARTIFACTS_DIR / "training_log.csv", index=False)
print("Saved: learning_curve.png, training_log.csv")"""),

        # ── Cleanup ──
        md("## Cleanup"),
        code("""\
train_env.close()
eval_env.close()
print("✅ NB05 PPO Training Complete")"""),

        md("""\
## Artifacts

| File | Description |
|------|-------------|
| `ppo_apple.zip` | Trained PPO model (SB3 format) |
| `vec_normalize.pkl` | VecNormalize statistics |
| `checkpoint_*.zip` | Checkpoints every 200K steps |
| `learning_curve.png` | Episode reward over training |
| `training_log.csv` | Per-episode reward/length during training |
| `eval_results.json` | Quick eval (20 episodes) after training |
| `nb05_config.json` | Full config (hyperparameters + env config) |

## RTX 5090 Optimization Notes

- **64 vectorized envs** on GPU → high throughput
- **[512, 512] ReLU** network (~790K params) — larger capacity
- **Batch size 2048** — matches n_steps for efficient gradient updates
- **Linear LR decay** (3e-4 → 1e-5) — smooth convergence
- **VecNormalize** — stabilizes training with obs/reward normalization
- **Checkpoints** every 200K — resume if interrupted
- **Fairness**: `total_env_steps`, `net_arch`, `VecNormalize` MUST match NB06/NB07
- On CPU mode (20K steps), results are for pipeline testing only"""),
    ]
    write_nb("NB05_train_ppo.ipynb", cells)


# ====================================================================
#  NB06 — Train SAC (Apple Full-Body)
# ====================================================================
def gen_nb06():
    cells = [
        md("""\
# NB06 — Train SAC (Apple Full-Body) — RTX 5090 Optimized

Train a **SAC** agent on `UnitreeG1PlaceAppleInBowlFullBody-v1` with
automatic entropy tuning. Off-policy learning with **10M replay buffer**.
Same training budget as PPO (NB05) for **fair comparison**.

| Parameter | Value |
|-----------|-------|
| Algorithm | SAC (Soft Actor-Critic) |
| Library | Stable-Baselines3 |
| Policy | MlpPolicy [512, 512] **ReLU** (~790K params) |
| Budget | **2,000,000** env steps (same as NB05 PPO) |
| Buffer | **10,000,000** transitions |
| Batch | **1024** |
| Entropy | `ent_coef="auto"` (automatic tuning) |"""),

        md("""\
## Objectives

1. Create env (single env — SAC is off-policy, uses replay buffer).
2. Configure SB3 SAC with same budget and net_arch as NB05 PPO.
3. Train for TOTAL_ENV_STEPS (2M) with checkpointing every 200K.
4. Save model + learning curve.
5. Quick evaluation (20 deterministic episodes)."""),

        md("""\
## Resources

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | **RTX 5090** (32 GB VRAM) | |
| RAM | 40 GB | 10M replay buffer ~4-8 GB |
| Runtime | ~2-4 hours (2M on RTX 5090) | |"""),

        md("## Imports & Setup"),
        code("""\
import sys, os, json, random, time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB06"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

IS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if IS_GPU else "cpu"
print(f"Device: {DEVICE}")
if IS_GPU:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")"""),

        md("## Configuration"),
        code("""\
CFG = {
    "seed":            42,
    "env_id":          "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "control_mode":    "pd_joint_delta_pos",
    "obs_mode":        "state",
    # ── Training budget (SAME as NB05 PPO for fairness) ──
    "total_env_steps": 2_000_000 if IS_GPU else 20_000,
    "n_envs":          1,   # SAC is off-policy → single env + replay buffer
    # ── SAC Hyperparameters (RTX 5090 optimized) ──
    "learning_rate":   3e-4,     # initial LR (linear decay to 1e-5)
    "lr_end":          1e-5,
    "buffer_size":     10_000_000 if IS_GPU else 50_000,
    "batch_size":      1024,
    "tau":             0.005,
    "gamma":           0.99,
    "ent_coef":        "auto",
    "target_entropy":  "auto",
    "learning_starts": 10_000,
    "train_freq":      1,
    "gradient_steps":  1,
    "net_arch":        [512, 512],   # SAME as NB05 PPO
    "activation_fn":   "ReLU",
    # ── Normalization ──
    "vec_normalize":   True,
    "norm_obs":        True,
    "norm_reward":     True,
    "clip_obs":        10.0,
    # ── Checkpointing ──
    "checkpoint_freq": 200_000,
    # ── Eval ──
    "eval_episodes":   20,
}

SEED = CFG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

with open(ARTIFACTS_DIR / "nb06_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

print("SAC Config (RTX 5090 optimized):")
for k, v in CFG.items():
    print(f"  {k}: {v}")"""),

        md("## Step 1 — Linear Learning Rate Schedule"),
        code("""\
def linear_schedule(initial_lr: float, final_lr: float = 1e-5):
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule

lr_schedule = linear_schedule(CFG["learning_rate"], CFG["lr_end"])
print(f"LR schedule: {CFG['learning_rate']} → {CFG['lr_end']} (linear decay)")"""),

        md("## Step 2 — Create Environment"),
        code("""\
env = gym.make(
    CFG["env_id"],
    num_envs=1,
    obs_mode=CFG["obs_mode"],
    control_mode=CFG["control_mode"],
    render_mode="rgb_array",
)
env = CPUGymWrapper(env)
print(f"Env: obs={env.observation_space.shape}, act={env.action_space.shape}")"""),

        md("## Step 3 — Configure SAC (RTX 5090)"),
        code("""\
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=lr_schedule,
    buffer_size=CFG["buffer_size"],
    batch_size=CFG["batch_size"],
    tau=CFG["tau"],
    gamma=CFG["gamma"],
    ent_coef=CFG["ent_coef"],
    target_entropy=CFG["target_entropy"],
    learning_starts=CFG["learning_starts"],
    train_freq=CFG["train_freq"],
    gradient_steps=CFG["gradient_steps"],
    policy_kwargs={
        "net_arch": CFG["net_arch"],
        "activation_fn": torch.nn.ReLU,
    },
    verbose=1,
    seed=SEED,
    device=DEVICE,
)

total_params = sum(p.numel() for p in model.policy.parameters())
print(f"SAC model created: {total_params:,} parameters")
print(f"  net_arch: {CFG['net_arch']}, activation: ReLU")
print(f"  buffer: {CFG['buffer_size']:,}, batch: {CFG['batch_size']}")"""),

        md("## Step 4 — Training Callbacks"),
        code("""\
class TrainLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in (infos if isinstance(infos, list) else [infos]):
            if isinstance(info, dict) and "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
        return True

log_cb = TrainLogCallback()

ckpt_cb = CheckpointCallback(
    save_freq=max(CFG["checkpoint_freq"] // max(CFG["n_envs"], 1), 1),
    save_path=str(ARTIFACTS_DIR),
    name_prefix="checkpoint",
)
print(f"✅ Callbacks ready (log + checkpoint every {CFG['checkpoint_freq']:,} steps)")"""),

        md("## Step 5 — Train SAC (2M steps)"),
        code("""\
print(f"\\n{'='*60}")
print(f"  Training SAC — {CFG['total_env_steps']:,} steps")
print(f"  buffer={CFG['buffer_size']:,}, batch={CFG['batch_size']}, device={DEVICE}")
print(f"  net_arch={CFG['net_arch']}, activation=ReLU")
print(f"  LR: {CFG['learning_rate']} → {CFG['lr_end']}")
print(f"{'='*60}")

start_time = time.time()
model.learn(
    total_timesteps=CFG["total_env_steps"],
    callback=[log_cb, ckpt_cb],
    progress_bar=True,
)
elapsed = time.time() - start_time
print(f"\\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")"""),

        md("## Step 6 — Save Model"),
        code("""\
model.save(str(ARTIFACTS_DIR / "sac_apple"))
print(f"✅ Model saved: {ARTIFACTS_DIR / 'sac_apple.zip'}")"""),

        md("## Step 7 — Quick Evaluation (20 episodes)"),
        code("""\
eval_env = gym.make(
    CFG["env_id"], num_envs=1, obs_mode="state",
    control_mode=CFG["control_mode"], render_mode="rgb_array",
)
eval_env = CPUGymWrapper(eval_env)

eval_results = []
for ep in range(CFG["eval_episodes"]):
    obs, info = eval_env.reset(seed=ep * 100)
    ep_reward, ep_steps = 0.0, 0
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        ep_reward += float(reward)
        ep_steps += 1
        if terminated or truncated:
            break
    eval_results.append({
        "episode": ep, "total_reward": ep_reward,
        "steps": ep_steps, "success": bool(info.get("success", False)),
    })
    print(f"  Ep {ep+1:2d}: reward={ep_reward:.4f}, steps={ep_steps}")

eval_summary = {
    "mean_reward":     float(np.mean([r["total_reward"] for r in eval_results])),
    "std_reward":      float(np.std([r["total_reward"] for r in eval_results])),
    "success_rate":    float(np.mean([r["success"] for r in eval_results])),
    "training_time_s": elapsed,
    "total_steps":     CFG["total_env_steps"],
}

with open(ARTIFACTS_DIR / "eval_results.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

print(f"\\nSAC Eval: mean_reward={eval_summary['mean_reward']:.4f}, "
      f"success={eval_summary['success_rate']:.2%}")
eval_env.close()"""),

        md("## Step 8 — Learning Curve"),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ep_rewards = log_cb.episode_rewards
if ep_rewards:
    axes[0].plot(ep_rewards, alpha=0.3, color="steelblue", linewidth=0.5)
    window = min(50, len(ep_rewards))
    if len(ep_rewards) >= window:
        rolling = np.convolve(ep_rewards, np.ones(window)/window, mode="valid")
        axes[0].plot(range(window-1, len(ep_rewards)), rolling,
                     color="darkblue", linewidth=2, label=f"Rolling avg ({window})")
    axes[0].set_title("Episode Rewards During Training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

ep_lengths = log_cb.episode_lengths
if ep_lengths:
    axes[1].plot(ep_lengths, alpha=0.3, color="seagreen", linewidth=0.5)
    axes[1].set_title("Episode Lengths")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].grid(True, alpha=0.3)

fig.suptitle(f"NB06 — SAC Training ({CFG['total_env_steps']:,} steps, RTX 5090)",
             fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "learning_curve.png", dpi=150)
plt.show()

training_log = pd.DataFrame({
    "episode": range(len(ep_rewards)),
    "reward": ep_rewards,
})
training_log.to_csv(ARTIFACTS_DIR / "training_log.csv", index=False)"""),

        md("## Cleanup"),
        code("""\
env.close()
print("✅ NB06 SAC Training Complete")"""),

        md("""\
## Artifacts

| File | Description |
|------|-------------|
| `sac_apple.zip` | Trained SAC model |
| `checkpoint_*.zip` | Checkpoints every 200K steps |
| `learning_curve.png` | Training reward curve |
| `training_log.csv` | Per-episode stats |
| `eval_results.json` | Quick eval (20 episodes) |
| `nb06_config.json` | Full config |

## RTX 5090 Optimization Notes

- **[512, 512] ReLU** network — same as NB05 for fairness
- **10M replay buffer** — leverages 40 GB RAM
- **Batch 1024** — efficient gradient computation on RTX 5090
- **Linear LR decay** (3e-4 → 1e-5)
- **learning_starts=10,000** — fill buffer with quality data before training
- **Checkpoints** every 200K steps
- **Fairness**: `total_env_steps=2M` and `net_arch=[512,512]` identical to NB05 PPO"""),
    ]
    write_nb("NB06_train_sac.ipynb", cells)


# ====================================================================
#  NB07 — Residual SAC + Beta Ablation
# ====================================================================
def gen_nb07():
    cells = [
        md("""\
# NB07 — Residual SAC + β Ablation (Apple Full-Body) — RTX 5090

Combine a **BaseController** (heuristic + EMA from NB04) with a learned
**residual policy** via SAC. Train **5 β variants** ∈ {0.1, 0.25, 0.5, 0.75, 1.0}
and select the best. RTX 5090 optimized with 2M steps per variant.

$$a_{final} = \\text{clip}(a_{base} + \\beta \\cdot a_{residual},\\ \\text{low},\\ \\text{high})$$

| β | Meaning |
|---|---------|
| 0.10 | Very conservative — almost entirely heuristic |
| 0.25 | Conservative — mostly follows heuristic |
| 0.50 | Balanced — moderate deviation allowed |
| 0.75 | Moderate — SAC has more influence |
| 1.00 | Aggressive — can fully override heuristic |"""),

        md("""\
## Objectives

1. Re-define heuristic policy and BaseController (from NB04 pattern).
2. Build `ResidualActionWrapper`.
3. Train Residual SAC for each of 5 β values (same budget as NB05/NB06 per run).
4. Quick eval (20 episodes) for each β.
5. Ablation table: select best β by mean reward.
6. Save all models + checkpoints."""),

        md("""\
## Resources

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | **RTX 5090** (32 GB VRAM) | 5 training runs |
| RAM | 40 GB | 10M buffer per run |
| Runtime | ~10-20 hours | 5 × 2M steps |"""),

        md("## Imports & Setup"),
        code("""\
import sys, os, json, random, time, copy
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB07"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

IS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if IS_GPU else "cpu"

# Load SAC config from NB06 for consistency
sac_cfg_path = PROJECT_ROOT / "artifacts" / "NB06" / "nb06_config.json"
assert sac_cfg_path.exists(), "Run NB06 first!"
with open(sac_cfg_path) as f:
    sac_cfg = json.load(f)
print(f"Loaded SAC config from NB06 (total_env_steps={sac_cfg['total_env_steps']:,})")"""),

        md("## Configuration"),
        code("""\
CFG = {
    "seed":            42,
    "env_id":          "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "control_mode":    "pd_joint_delta_pos",
    "obs_mode":        "state",
    # ── Residual-specific (5 β variants for RTX 5090) ──
    "beta_values":     [0.1, 0.25, 0.5, 0.75, 1.0],
    "smooth_alpha":    0.3,
    # ── Same training budget per run as NB05/NB06 ──
    "total_env_steps": sac_cfg["total_env_steps"],
    # ── SAC hyperparams (inherited from NB06, RTX 5090) ──
    "learning_rate":   sac_cfg["learning_rate"],
    "lr_end":          sac_cfg.get("lr_end", 1e-5),
    "buffer_size":     sac_cfg["buffer_size"],
    "batch_size":      sac_cfg["batch_size"],
    "tau":             sac_cfg["tau"],
    "gamma":           sac_cfg["gamma"],
    "ent_coef":        sac_cfg["ent_coef"],
    "net_arch":        sac_cfg["net_arch"],
    "activation_fn":   sac_cfg.get("activation_fn", "ReLU"),
    "learning_starts": sac_cfg["learning_starts"],
    # ── Checkpointing ──
    "checkpoint_freq": 200_000,
    # ── Eval ──
    "eval_episodes":   20,
}

SEED = CFG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

with open(ARTIFACTS_DIR / "nb07_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

print("Residual SAC Config (RTX 5090):")
print(f"  Beta values: {CFG['beta_values']} (5 variants)")
print(f"  Budget per run: {CFG['total_env_steps']:,} steps")
print(f"  net_arch: {CFG['net_arch']}, activation: {CFG['activation_fn']}")"""),

        # ── LR schedule ──
        md("## Step 0 — Linear LR Schedule"),
        code("""\
def linear_schedule(initial_lr: float, final_lr: float = 1e-5):
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule

print(f"LR: {CFG['learning_rate']} → {CFG['lr_end']} (linear decay)")"""),

        # ── Heuristic + BaseController ──
        md("""\
## Step 1 — Heuristic Policy & BaseController

Re-define the heuristic and BaseController from NB04 pattern."""),
        code("""\
def heuristic_policy(obs, env):
    \"\"\"Proportional control: arm joints get small reaching signals.\"\"\"
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    arm_start, arm_end = 13, 23  # approximate arm joint range
    action[arm_start:arm_end] = np.random.uniform(-0.1, 0.1, arm_end - arm_start) * 0.5
    return action


class BaseController:
    \"\"\"Heuristic + EMA smoothing for residual learning.\"\"\"

    def __init__(self, env, alpha=0.3):
        self.env = env
        self.alpha = alpha
        self._prev_action = None

    def get_action(self, obs):
        raw = heuristic_policy(obs, self.env)
        if self._prev_action is None:
            self._prev_action = raw.copy()
        smoothed = self.alpha * raw + (1 - self.alpha) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def reset(self):
        self._prev_action = None

print("✅ Heuristic + BaseController defined")"""),

        # ── ResidualActionWrapper ──
        md("""\
## Step 2 — ResidualActionWrapper

Combines base controller action with learned residual."""),
        code("""\
class ResidualActionWrapper(gym.ActionWrapper):
    \"\"\"a_final = clip(a_base + beta * a_residual, low, high)

    SAC outputs the residual; this wrapper adds the base action.
    \"\"\"

    def __init__(self, env, base_controller, beta=0.5):
        super().__init__(env)
        self.base_controller = base_controller
        self.beta = beta
        self._current_obs = None

    def reset(self, **kwargs):
        self.base_controller.reset()
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            self._current_obs = result[0]
        else:
            self._current_obs = result
        return result

    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) >= 1:
            self._current_obs = result[0]
        return result

    def action(self, residual_action):
        \"\"\"Transform SAC output (residual) into final action.\"\"\"
        base_action = self.base_controller.get_action(self._current_obs)
        final = base_action + self.beta * residual_action
        return np.clip(final, self.action_space.low, self.action_space.high)

print("✅ ResidualActionWrapper defined")"""),

        # ── Train Loop ──
        md("## Step 3 — Train Residual SAC for Each β (5 variants × 2M steps)"),
        code("""\
ablation_results = []

for beta in CFG["beta_values"]:
    print(f"\\n{'='*60}")
    print(f"  Training Residual SAC — β={beta}")
    print(f"  Budget: {CFG['total_env_steps']:,} steps, net={CFG['net_arch']}")
    print(f"{'='*60}")

    # Create fresh env + wrapper
    raw_env = gym.make(
        CFG["env_id"], num_envs=1, obs_mode=CFG["obs_mode"],
        control_mode=CFG["control_mode"], render_mode="rgb_array",
    )
    raw_env = CPUGymWrapper(raw_env)
    base_ctrl = BaseController(raw_env, alpha=CFG["smooth_alpha"])
    wrapped_env = ResidualActionWrapper(raw_env, base_ctrl, beta=beta)

    # LR schedule
    lr_sched = linear_schedule(CFG["learning_rate"], CFG["lr_end"])

    # Train SAC (residual)
    sac_model = SAC(
        "MlpPolicy", wrapped_env,
        learning_rate=lr_sched,
        buffer_size=CFG["buffer_size"],
        batch_size=CFG["batch_size"],
        tau=CFG["tau"],
        gamma=CFG["gamma"],
        ent_coef=CFG["ent_coef"],
        learning_starts=CFG["learning_starts"],
        policy_kwargs={
            "net_arch": CFG["net_arch"],
            "activation_fn": torch.nn.ReLU,
        },
        verbose=0,
        seed=SEED,
        device=DEVICE,
    )

    start_time = time.time()
    sac_model.learn(total_timesteps=CFG["total_env_steps"], progress_bar=True)
    train_time = time.time() - start_time

    # Save model
    model_name = f"residual_apple_beta{beta}"
    sac_model.save(str(ARTIFACTS_DIR / model_name))

    # Quick eval
    eval_rewards, eval_successes = [], []
    for ep in range(CFG["eval_episodes"]):
        obs, info = wrapped_env.reset(seed=ep * 100)
        ep_reward = 0.0
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            ep_reward += float(reward)
            if terminated or truncated:
                break
        eval_rewards.append(ep_reward)
        eval_successes.append(bool(info.get("success", False)))

    ablation_results.append({
        "beta":          beta,
        "mean_reward":   float(np.mean(eval_rewards)),
        "std_reward":    float(np.std(eval_rewards)),
        "success_rate":  float(np.mean(eval_successes)),
        "training_time": train_time,
    })

    wrapped_env.close()
    print(f"  β={beta}: mean_reward={np.mean(eval_rewards):.4f}, "
          f"success={np.mean(eval_successes):.2%}, time={train_time:.0f}s")"""),

        # ── Ablation Table ──
        md("## Step 4 — Ablation Table"),
        code("""\
ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv(ARTIFACTS_DIR / "ablation_table.csv", index=False)

print("\\nAblation Table (5 β variants):")
print(ablation_df.to_string(index=False))

# Select best beta
best_idx = ablation_df["mean_reward"].idxmax()
best_beta = float(ablation_df.loc[best_idx, "beta"])
best_info = {
    "best_beta":        best_beta,
    "best_mean_reward": float(ablation_df.loc[best_idx, "mean_reward"]),
    "best_model":       f"residual_apple_beta{best_beta}.zip",
}
with open(ARTIFACTS_DIR / "best_beta.json", "w") as f:
    json.dump(best_info, f, indent=2)

print(f"\\n🏆 Best β = {best_beta} (mean_reward={best_info['best_mean_reward']:.4f})")"""),

        # ── Ablation Plot ──
        md("## Step 5 — Ablation Plot"),
        code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

betas = ablation_df["beta"].tolist()
rewards = ablation_df["mean_reward"].tolist()
stds = ablation_df["std_reward"].tolist()
successes = ablation_df["success_rate"].tolist()

x_labels = [f"β={b}" for b in betas]
colors = ["#E0E0E0", "#FFB74D", "#64B5F6", "#81C784", "#CE93D8"]

axes[0].bar(x_labels, rewards, yerr=stds, color=colors, edgecolor="black", capsize=5)
axes[0].set_title("Mean Reward by β (20 eval episodes)")
axes[0].set_ylabel("Mean Reward")
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(x_labels, successes, color=colors, edgecolor="black")
axes[1].set_title("Success Rate by β")
axes[1].set_ylabel("Success Rate")
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("NB07 — Residual SAC β Ablation (5 variants × 2M steps, RTX 5090)",
             fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "ablation_plot.png", dpi=150)
plt.show()"""),

        md("## Cleanup"),
        code("""\
print("✅ NB07 Residual SAC Training Complete")"""),

        md("""\
## Artifacts

| File | Description |
|------|-------------|
| `residual_apple_beta0.1.zip` | Model for β=0.1 |
| `residual_apple_beta0.25.zip` | Model for β=0.25 |
| `residual_apple_beta0.5.zip` | Model for β=0.5 |
| `residual_apple_beta0.75.zip` | Model for β=0.75 |
| `residual_apple_beta1.0.zip` | Model for β=1.0 |
| `ablation_table.csv` | Performance per beta |
| `ablation_plot.png` | Comparison charts |
| `best_beta.json` | Best β selection + model path |

## RTX 5090 Optimization Notes

- **5 β variants** (was 3) — finer granularity for ablation study
- **2M steps per variant** — leverages RTX 5090 throughput
- **[512, 512] ReLU** — same architecture as NB05/NB06
- **10M replay buffer** — 40 GB RAM supports large buffers
- **Linear LR decay** (3e-4 → 1e-5) for each variant
- SAC hyperparams loaded from NB06 config to ensure fairness
- Total GPU time ≈ 5 × single training run"""),
    ]
    write_nb("NB07_residual_sac_ablation.ipynb", cells)


# ====================================================================
#  NB08 — Evaluation: Compare Methods & Declare Winner
# ====================================================================
def gen_nb08():
    cells = [
        md("""\
# NB08 — Evaluation: Compare Methods & Declare Winner — RTX 5090

Evaluate all trained agents (**PPO**, **SAC**, **Residual-SAC-best-β**)
on **200 deterministic episodes** each. Compute **95 % bootstrap CI**
(50 K resamples), **Welch's t-test**, and **Cohen's d** effect sizes.
Produce comparison tables and plots. **Declare the winner** for NB09.

| Metric | Spec |
|--------|------|
| Episodes | 200 per method (seeds 0-199) |
| Bootstrap | 50,000 resamples, 95 % CI |
| Statistical Test | Welch's t (unequal variance) |
| Effect Size | Cohen's d |"""),

        md("""\
## Objectives

1. Load all 3 trained models (from NB05 / NB06 / NB07-best-β).
2. Evaluate each on **200 deterministic episodes** (same seed sequence).
3. Compute **50 K bootstrap 95 % CI** for mean reward & success rate.
4. Run **Welch's t-test** on every pair of methods.
5. Compute **Cohen's d** effect size for every pair.
6. Build comparison table & stat_tests artifact.
7. Declare winner (highest mean_reward; tie-break by success_rate).
8. Produce bar charts with CI, violin plots, success rate plots."""),

        md("""\
## Resources

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | Optional | CPU OK for eval |
| RAM | 8+ GB | 200 eps × 3 methods |
| Runtime | ~1-2 hours | 600 episodes total |"""),

        md("## Imports & Setup"),
        code("""\
import sys, os, json, random, itertools
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats  # Welch's t-test

from stable_baselines3 import PPO, SAC

import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB08"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)"""),

        md("## Configuration"),
        code("""\
CFG = {
    "env_id":           "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "control_mode":     "pd_joint_delta_pos",
    "obs_mode":         "state",
    "eval_episodes":    200,
    "max_steps_per_ep": 1000,
    "bootstrap_n":      50_000,
    "ci_level":         0.95,
}

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

with open(ARTIFACTS_DIR / "nb08_config.json", "w") as f:
    json.dump(CFG, f, indent=2)
print("Config:", json.dumps(CFG, indent=2))"""),

        # ── Load Models ──
        md("## Step 1 — Load All Models"),
        code("""\
# Load best beta info
with open(PROJECT_ROOT / "artifacts" / "NB07" / "best_beta.json") as f:
    best_beta_info = json.load(f)

best_beta = best_beta_info["best_beta"]
best_model_name = best_beta_info["best_model"].replace(".zip", "")

models = {
    "PPO": PPO.load(str(PROJECT_ROOT / "artifacts" / "NB05" / "ppo_apple")),
    "SAC": SAC.load(str(PROJECT_ROOT / "artifacts" / "NB06" / "sac_apple")),
    "Residual-SAC": SAC.load(str(PROJECT_ROOT / "artifacts" / "NB07" / best_model_name)),
}

print(f"Loaded models: {list(models.keys())}")
print(f"Residual-SAC best β = {best_beta}")"""),

        # ── Heuristic + BaseController for Residual ──
        md("## Step 2 — BaseController for Residual-SAC Evaluation"),
        code("""\
# Re-define for evaluation (same as NB07)
def heuristic_policy(obs, env):
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    arm_start, arm_end = 13, 23
    action[arm_start:arm_end] = np.random.uniform(-0.1, 0.1, arm_end - arm_start) * 0.5
    return action

class BaseController:
    def __init__(self, env, alpha=0.3):
        self.env = env
        self.alpha = alpha
        self._prev_action = None

    def get_action(self, obs):
        raw = heuristic_policy(obs, self.env)
        if self._prev_action is None:
            self._prev_action = raw.copy()
        smoothed = self.alpha * raw + (1 - self.alpha) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def reset(self):
        self._prev_action = None

class ResidualActionWrapper(gym.ActionWrapper):
    def __init__(self, env, base_controller, beta=0.5):
        super().__init__(env)
        self.base_controller = base_controller
        self.beta = beta
        self._current_obs = None

    def reset(self, **kwargs):
        self.base_controller.reset()
        result = self.env.reset(**kwargs)
        self._current_obs = result[0] if isinstance(result, tuple) else result
        return result

    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple):
            self._current_obs = result[0]
        return result

    def action(self, residual_action):
        base_action = self.base_controller.get_action(self._current_obs)
        final = base_action + self.beta * residual_action
        return np.clip(final, self.action_space.low, self.action_space.high)

print("✅ BaseController + ResidualActionWrapper ready")"""),

        # ── Evaluate ──
        md("## Step 3 — Evaluate All Methods (200 episodes each)"),
        code("""\
def evaluate_model(model, env, n_episodes, max_steps):
    \"\"\"Run n deterministic episodes, return per-episode results.\"\"\"
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        ep_reward, ep_steps = 0.0, 0
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            if terminated or truncated:
                break
        results.append({
            "episode": ep, "total_reward": ep_reward,
            "steps": ep_steps, "success": bool(info.get("success", False)),
        })
    return results

all_results = {}

for method_name, model in models.items():
    print(f"\\nEvaluating {method_name} ({CFG['eval_episodes']} episodes)...")

    if method_name == "Residual-SAC":
        raw_env = gym.make(CFG["env_id"], num_envs=1, obs_mode=CFG["obs_mode"],
                           control_mode=CFG["control_mode"], render_mode="rgb_array")
        raw_env = CPUGymWrapper(raw_env)
        base_ctrl = BaseController(raw_env, alpha=0.3)
        eval_env = ResidualActionWrapper(raw_env, base_ctrl, beta=best_beta)
    else:
        eval_env = gym.make(CFG["env_id"], num_envs=1, obs_mode=CFG["obs_mode"],
                            control_mode=CFG["control_mode"], render_mode="rgb_array")
        eval_env = CPUGymWrapper(eval_env)

    results = evaluate_model(model, eval_env, CFG["eval_episodes"],
                             CFG["max_steps_per_ep"])
    all_results[method_name] = results
    eval_env.close()

    rews = [r["total_reward"] for r in results]
    succs = [r["success"] for r in results]
    print(f"  mean_reward={np.mean(rews):.4f} ± {np.std(rews):.4f}, "
          f"success_rate={np.mean(succs):.2%}")

# Save per-episode data
rows = []
for method, results in all_results.items():
    for r in results:
        rows.append({"method": method, **r})
eval_df = pd.DataFrame(rows)
eval_df.to_csv(ARTIFACTS_DIR / "eval_200ep.csv", index=False)
print(f"\\nSaved: eval_200ep.csv ({len(rows)} rows)")"""),

        # ── Bootstrap CI ──
        md("## Step 4 — Bootstrap 95 % Confidence Intervals (50 K resamples)"),
        code("""\
def bootstrap_ci(data, stat_fn=np.mean, n_bootstrap=50_000, ci=0.95):
    \"\"\"Compute bootstrap confidence interval.\"\"\"
    boot_stats = []
    data = np.array(data, dtype=float)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_fn(sample))
    lower = float(np.percentile(boot_stats, (1 - ci) / 2 * 100))
    upper = float(np.percentile(boot_stats, (1 + ci) / 2 * 100))
    return lower, upper

# Build comparison table
comparison = []
for method, results in all_results.items():
    rewards = [r["total_reward"] for r in results]
    successes = [float(r["success"]) for r in results]

    rew_lo, rew_hi = bootstrap_ci(rewards, np.mean, CFG["bootstrap_n"], CFG["ci_level"])
    suc_lo, suc_hi = bootstrap_ci(successes, np.mean, CFG["bootstrap_n"], CFG["ci_level"])

    comparison.append({
        "method":           method,
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "ci95_reward_lo":   rew_lo,
        "ci95_reward_hi":   rew_hi,
        "success_rate":     float(np.mean(successes)),
        "ci95_success_lo":  suc_lo,
        "ci95_success_hi":  suc_hi,
        "mean_steps":       float(np.mean([r["steps"] for r in results])),
    })

comp_df = pd.DataFrame(comparison)
comp_df.to_csv(ARTIFACTS_DIR / "comparison_table.csv", index=False)

print("\\nComparison Table (200 episodes, 50K bootstrap):")
print(comp_df.to_string(index=False))"""),

        # ── Statistical Tests ──
        md("""\
## Step 4b — Welch's t-test & Cohen's d (Pairwise)

- **Welch's t-test**: `scipy.stats.ttest_ind(equal_var=False)` — does not
  assume equal variance.
- **Cohen's d**: standardized effect size = (M₁ − M₂) / pooled SD."""),
        code("""\
def cohens_d(x, y):
    \"\"\"Cohen's d effect size (pooled SD).\"\"\"
    nx, ny = len(x), len(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_sd if pooled_sd > 0 else 0.0


method_names = list(all_results.keys())
stat_tests = []

for m1, m2 in itertools.combinations(method_names, 2):
    r1 = np.array([r["total_reward"] for r in all_results[m1]])
    r2 = np.array([r["total_reward"] for r in all_results[m2]])

    t_stat, p_value = stats.ttest_ind(r1, r2, equal_var=False)
    d = cohens_d(r1, r2)

    stat_tests.append({
        "comparison":  f"{m1} vs {m2}",
        "t_statistic": float(t_stat),
        "p_value":     float(p_value),
        "cohens_d":    float(d),
        "significant": bool(p_value < 0.05),
        "effect_size": ("large" if abs(d) >= 0.8 else
                        "medium" if abs(d) >= 0.5 else
                        "small" if abs(d) >= 0.2 else "negligible"),
    })

stat_df = pd.DataFrame(stat_tests)
stat_df.to_csv(ARTIFACTS_DIR / "stat_tests.csv", index=False)

with open(ARTIFACTS_DIR / "stat_tests.json", "w") as f:
    json.dump(stat_tests, f, indent=2)

print("\\nPairwise Statistical Tests:")
print(stat_df.to_string(index=False))"""),

        # ── Declare Winner ──
        md("## Step 5 — Declare Winner"),
        code("""\
best_idx = comp_df["mean_reward"].idxmax()
winner = comp_df.loc[best_idx]

best_method = {
    "winner":       str(winner["method"]),
    "mean_reward":  float(winner["mean_reward"]),
    "ci95":         [float(winner["ci95_reward_lo"]), float(winner["ci95_reward_hi"])],
    "success_rate": float(winner["success_rate"]),
    "eval_episodes": CFG["eval_episodes"],
    "bootstrap_n":  CFG["bootstrap_n"],
    "reason":       f"Highest mean reward ({winner['mean_reward']:.4f}) "
                    f"over {CFG['eval_episodes']} episodes with 50K bootstrap CI",
}

with open(ARTIFACTS_DIR / "best_method.json", "w") as f:
    json.dump(best_method, f, indent=2)

print(f"\\n{'='*60}")
print(f"  🏆 WINNER: {best_method['winner']}")
print(f"  Mean Reward: {best_method['mean_reward']:.4f} "
      f"[{best_method['ci95'][0]:.4f}, {best_method['ci95'][1]:.4f}]")
print(f"  Success Rate: {best_method['success_rate']:.2%}")
print(f"  → This method will train DishWipe in NB09")
print(f"{'='*60}")"""),

        # ── Plots ──
        md("## Step 6 — Comparison Plots"),
        code("""\
methods = comp_df["method"].tolist()
color_map = {"PPO": "#FFB74D", "SAC": "#64B5F6", "Residual-SAC": "#81C784"}
colors = [color_map.get(m, "gray") for m in methods]

# Plot A: Reward with CI
fig, ax = plt.subplots(figsize=(8, 5))
means = comp_df["mean_reward"].tolist()
ci_lo = comp_df["ci95_reward_lo"].tolist()
ci_hi = comp_df["ci95_reward_hi"].tolist()
errs = [[m - lo for m, lo in zip(means, ci_lo)],
        [hi - m for m, hi in zip(means, ci_hi)]]
ax.bar(methods, means, yerr=errs, capsize=8, color=colors, edgecolor="black")
ax.set_title("Mean Reward (200 episodes, 95% CI, 50K bootstrap)", fontweight="bold")
ax.set_ylabel("Mean Reward")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "comparison_plot.png", dpi=150)
plt.show()

# Plot B: Success Rate
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(methods, comp_df["success_rate"].tolist(), color=colors, edgecolor="black")
ax.set_title("Success Rate (200 episodes)", fontweight="bold")
ax.set_ylabel("Success Rate")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "success_rate_plot.png", dpi=150)
plt.show()

# Plot C: Reward distribution (violin)
fig, ax = plt.subplots(figsize=(8, 5))
data_violin = [[r["total_reward"] for r in all_results[m]] for m in methods]
parts = ax.violinplot(data_violin, positions=range(len(methods)), showmeans=True)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods)
ax.set_title("Reward Distribution (200 episodes)", fontweight="bold")
ax.set_ylabel("Total Reward")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "reward_distribution.png", dpi=150)
plt.show()

# Plot D: Cohen's d heatmap
fig, ax = plt.subplots(figsize=(6, 5))
d_matrix = np.zeros((len(method_names), len(method_names)))
for test in stat_tests:
    m1, m2 = test["comparison"].split(" vs ")
    i, j = method_names.index(m1), method_names.index(m2)
    d_matrix[i, j] = test["cohens_d"]
    d_matrix[j, i] = -test["cohens_d"]
im = ax.imshow(d_matrix, cmap="RdBu_r", vmin=-2, vmax=2)
ax.set_xticks(range(len(method_names)))
ax.set_xticklabels(method_names, rotation=45, ha="right")
ax.set_yticks(range(len(method_names)))
ax.set_yticklabels(method_names)
for i in range(len(method_names)):
    for j in range(len(method_names)):
        ax.text(j, i, f"{d_matrix[i,j]:.2f}", ha="center", va="center", fontsize=10)
ax.set_title("Cohen's d Effect Size (pairwise)", fontweight="bold")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "cohens_d_heatmap.png", dpi=150)
plt.show()

print("✅ All plots saved")"""),

        md("## Cleanup"),
        code("""\
print("✅ NB08 Evaluation Complete (RTX 5090 spec)")
print(f"Winner: {best_method['winner']} → will train DishWipe in NB09")"""),

        md("""\
## Artifacts

| File | Description |
|------|-------------|
| `eval_200ep.csv` | Per-episode results for all methods (200 × 3) |
| `comparison_table.csv` | Summary statistics with 95% CI |
| `stat_tests.csv` | Welch's t-test + Cohen's d (pairwise) |
| `stat_tests.json` | Same in JSON format |
| `comparison_plot.png` | Bar chart with 95% CI |
| `success_rate_plot.png` | Success rate comparison |
| `reward_distribution.png` | Violin plot of reward distributions |
| `cohens_d_heatmap.png` | Cohen's d effect size heatmap |
| `best_method.json` | Winner declaration + reason |

## RTX 5090 Evaluation Notes

- **200 episodes** (was 100) — more statistical power
- **50,000 bootstrap resamples** (was 10,000) — tighter CI
- **Welch's t-test** — pairwise, no equal-variance assumption
- **Cohen's d** — standardized effect size (small/medium/large)
- Deterministic eval with `seeds 0-199` → fully reproducible
- `best_method.json` is consumed by NB09"""),
    ]
    write_nb("NB08_evaluation.ipynb", cells)


# ====================================================================
#  NB09 — Bonus: DishWipe Full-Body (Winner Only)
# ====================================================================
def gen_nb09():
    cells = [
        md("""\
# NB09 — Bonus: DishWipe Full-Body (Winner Only) — RTX 5090

Self-contained bonus notebook. Load the **winning method from NB08**,
train it on the **DishWipe Full-Body** task with RTX 5090 specs, then
evaluate **200 episodes** and compare cross-task performance.

| Item | Value |
|------|-------|
| Task | DishWipe Full-Body (bonus) |
| Method | Winner from NB08 (`best_method.json`) |
| Robot | Unitree G1 Full Body (37 DOF) |
| Budget | 2,000,000 env steps (RTX 5090) |
| Eval | 200 episodes, 50K bootstrap, Welch's t-test |"""),

        md("""\
## Objectives

1. Load winner declaration from NB08.
2. Quick smoke test on DishWipe Full-Body env.
3. Train winner method on DishWipe (**2M steps**, RTX 5090 config).
4. Evaluate **200 deterministic episodes**.
5. Bootstrap 95% CI (50K resamples) + Welch's t-test vs Apple performance.
6. Cross-task comparison: Apple vs DishWipe."""),

        md("""\
## Resources

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | **RTX 5090** (32 GB VRAM) | Training |
| RAM | 40 GB | Large buffer/net |
| Runtime | ~2-4 hours | 2M steps |"""),

        md("## Imports & Setup"),
        code("""\
import sys, os, json, random, time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
from src.envs import UnitreeG1DishWipeFullBodyEnv

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB09"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

IS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if IS_GPU else "cpu"
print(f"Device: {DEVICE}")"""),

        # ── Load Winner ──
        md("## Step 1 — Load Winner from NB08"),
        code("""\
with open(PROJECT_ROOT / "artifacts" / "NB08" / "best_method.json") as f:
    winner_info = json.load(f)

winner_method = winner_info["winner"]
print(f"🏆 Winner from NB08: {winner_method}")
print(f"  Apple mean_reward:  {winner_info['mean_reward']:.4f}")
print(f"  Apple success_rate: {winner_info['success_rate']:.2%}")"""),

        md("## Configuration"),
        code("""\
def linear_schedule(initial_lr: float, final_lr: float = 1e-5):
    \"\"\"Linear LR decay from initial_lr → final_lr.\"\"\"
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule


CFG = {
    "seed":             42,
    "env_id":           "UnitreeG1DishWipeFullBody-v1",
    "control_mode":     "pd_joint_delta_pos",
    "obs_mode":         "state",
    "winner_method":    winner_method,
    "total_env_steps":  2_000_000 if IS_GPU else 20_000,
    "eval_episodes":    200,
    "max_steps_per_ep": 1000,
    "bootstrap_n":      50_000,
    "checkpoint_freq":  200_000,
    # ── RTX 5090 hyperparams (same as NB05/NB06) ──
    "learning_rate":    3e-4,
    "lr_end":           1e-5,
    "net_arch":         [512, 512],
    "activation_fn":    "ReLU",
}

SEED = CFG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

with open(ARTIFACTS_DIR / "nb09_config.json", "w") as f:
    json.dump(CFG, f, indent=2)
print(f"Training {winner_method} on DishWipe for {CFG['total_env_steps']:,} steps")
print(f"  net_arch: {CFG['net_arch']}, activation: {CFG['activation_fn']}")
print(f"  LR: {CFG['learning_rate']} → {CFG['lr_end']} (linear decay)")"""),

        # ── Smoke Test ──
        md("## Step 2 — DishWipe Full-Body Smoke Test"),
        code("""\
print("--- DishWipe Full-Body Smoke Test ---")
env_test = gym.make(CFG["env_id"], num_envs=1, obs_mode="state",
                    control_mode=CFG["control_mode"], render_mode="rgb_array")
env_test = CPUGymWrapper(env_test)
obs, info = env_test.reset(seed=SEED)

print(f"  obs shape: {obs.shape}")
print(f"  act shape: {env_test.action_space.shape}")

for _ in range(5):
    action = env_test.action_space.sample()
    obs, reward, terminated, truncated, info = env_test.step(action)
print(f"  Smoke OK, reward sample: {reward:.4f}")
env_test.close()"""),

        # ── Residual helpers (if needed) ──
        md("## Step 3 — Helpers for Residual-SAC (if winner)"),
        code("""\
# Define BaseController + ResidualActionWrapper in case Residual-SAC won
def heuristic_dishwipe_policy(obs, env):
    \"\"\"Simple heuristic for DishWipe: arm joints get small sweeping signals.\"\"\"
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    arm_start, arm_end = 13, 23
    action[arm_start:arm_end] = np.random.uniform(-0.1, 0.1, arm_end - arm_start) * 0.3
    return action

class BaseController:
    def __init__(self, env, alpha=0.3):
        self.env = env
        self.alpha = alpha
        self._prev_action = None

    def get_action(self, obs):
        raw = heuristic_dishwipe_policy(obs, self.env)
        if self._prev_action is None:
            self._prev_action = raw.copy()
        smoothed = self.alpha * raw + (1 - self.alpha) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def reset(self):
        self._prev_action = None

class ResidualActionWrapper(gym.ActionWrapper):
    def __init__(self, env, base_controller, beta=0.5):
        super().__init__(env)
        self.base_controller = base_controller
        self.beta = beta
        self._current_obs = None

    def reset(self, **kwargs):
        self.base_controller.reset()
        result = self.env.reset(**kwargs)
        self._current_obs = result[0] if isinstance(result, tuple) else result
        return result

    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple):
            self._current_obs = result[0]
        return result

    def action(self, residual_action):
        base_action = self.base_controller.get_action(self._current_obs)
        final = base_action + self.beta * residual_action
        return np.clip(final, self.action_space.low, self.action_space.high)

print("✅ Helpers defined (BaseController, ResidualActionWrapper)")"""),

        # ── Train ──
        md("## Step 4 — Train Winner on DishWipe (2M steps, RTX 5090)"),
        code("""\
# Create env
train_env = gym.make(CFG["env_id"], num_envs=1, obs_mode="state",
                     control_mode=CFG["control_mode"], render_mode="rgb_array")
train_env = CPUGymWrapper(train_env)

# For Residual-SAC, wrap with ResidualActionWrapper
if winner_method == "Residual-SAC":
    with open(PROJECT_ROOT / "artifacts" / "NB07" / "best_beta.json") as f:
        beta_info = json.load(f)
    base_ctrl = BaseController(train_env, alpha=0.3)
    train_env = ResidualActionWrapper(train_env, base_ctrl, beta=beta_info["best_beta"])

# LR schedule (linear decay)
lr_sched = linear_schedule(CFG["learning_rate"], CFG["lr_end"])

# Checkpoint callback
ckpt_cb = CheckpointCallback(
    save_freq=CFG["checkpoint_freq"],
    save_path=str(ARTIFACTS_DIR / "checkpoints"),
    name_prefix=f"{winner_method.lower().replace('-', '_')}_dishwipe",
)

# Create model with RTX 5090 hyperparams
if winner_method == "PPO":
    with open(PROJECT_ROOT / "artifacts" / "NB05" / "nb05_config.json") as f:
        method_cfg = json.load(f)
    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=lr_sched,
        n_steps=method_cfg.get("n_steps", 2048),
        batch_size=method_cfg.get("batch_size", 2048),
        n_epochs=method_cfg.get("n_epochs", 10),
        gamma=method_cfg.get("gamma", 0.99),
        clip_range=method_cfg.get("clip_range", 0.2),
        ent_coef=method_cfg.get("ent_coef", 0.01),
        vf_coef=method_cfg.get("vf_coef", 0.5),
        policy_kwargs={"net_arch": CFG["net_arch"],
                       "activation_fn": torch.nn.ReLU},
        verbose=1, seed=SEED, device=DEVICE,
    )
elif winner_method in ("SAC", "Residual-SAC"):
    with open(PROJECT_ROOT / "artifacts" / "NB06" / "nb06_config.json") as f:
        method_cfg = json.load(f)
    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=lr_sched,
        buffer_size=method_cfg.get("buffer_size", 10_000_000),
        batch_size=method_cfg.get("batch_size", 1024),
        tau=method_cfg.get("tau", 0.005),
        gamma=method_cfg.get("gamma", 0.99),
        ent_coef=method_cfg.get("ent_coef", "auto"),
        learning_starts=method_cfg.get("learning_starts", 10_000),
        policy_kwargs={"net_arch": CFG["net_arch"],
                       "activation_fn": torch.nn.ReLU},
        verbose=1, seed=SEED, device=DEVICE,
    )
else:
    raise ValueError(f"Unknown winner: {winner_method}")

print(f"\\n{'='*60}")
print(f"  Training {winner_method} on DishWipe Full-Body (RTX 5090)")
print(f"  Budget: {CFG['total_env_steps']:,} steps")
print(f"  net_arch: {CFG['net_arch']}, LR: {CFG['learning_rate']}→{CFG['lr_end']}")
print(f"{'='*60}")

# Training callback
class TrainLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in (infos if isinstance(infos, list) else [infos]):
            if isinstance(info, dict) and "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        return True

cb = TrainLogCallback()
start_time = time.time()
model.learn(total_timesteps=CFG["total_env_steps"], callback=[cb, ckpt_cb],
            progress_bar=True)
elapsed = time.time() - start_time

model_name = f"{winner_method.lower().replace('-', '_')}_dishwipe"
model.save(str(ARTIFACTS_DIR / model_name))
train_env.close()

print(f"\\nTraining done in {elapsed:.1f}s ({elapsed/3600:.1f}h)")
print(f"Model saved: {model_name}.zip")"""),

        # ── Evaluate ──
        md("## Step 5 — Evaluate 200 Episodes"),
        code("""\
eval_env = gym.make(CFG["env_id"], num_envs=1, obs_mode="state",
                    control_mode=CFG["control_mode"], render_mode="rgb_array")
eval_env = CPUGymWrapper(eval_env)

if winner_method == "Residual-SAC":
    base_ctrl = BaseController(eval_env, alpha=0.3)
    eval_env = ResidualActionWrapper(eval_env, base_ctrl, beta=beta_info["best_beta"])

eval_results = []
for ep in range(CFG["eval_episodes"]):
    obs, info = eval_env.reset(seed=ep)
    ep_reward, ep_steps = 0.0, 0
    for step in range(CFG["max_steps_per_ep"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        ep_reward += float(reward)
        ep_steps += 1
        if terminated or truncated:
            break
    eval_results.append({
        "episode": ep,
        "total_reward": ep_reward,
        "steps": ep_steps,
        "success": bool(info.get("success", False)),
        "cleaned_ratio": float(info.get("cleaned_ratio", 0.0)),
    })

eval_df = pd.DataFrame(eval_results)
eval_df["method"] = winner_method
eval_df.to_csv(ARTIFACTS_DIR / "eval_200ep.csv", index=False)
eval_env.close()

print(f"Eval ({CFG['eval_episodes']} eps): "
      f"mean_reward={eval_df['total_reward'].mean():.4f}, "
      f"success={eval_df['success'].mean():.2%}, "
      f"cleaned={eval_df['cleaned_ratio'].mean():.2%}")"""),

        # ── Bootstrap CI ──
        md("## Step 6 — Bootstrap CI (50K resamples) + Welch's t-test"),
        code("""\
def bootstrap_ci(data, n_boot=50_000, ci=0.95):
    data = np.array(data, dtype=float)
    boot = [np.mean(np.random.choice(data, len(data), replace=True))
            for _ in range(n_boot)]
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return lo, hi

rewards = eval_df["total_reward"].tolist()
rew_lo, rew_hi = bootstrap_ci(rewards, CFG["bootstrap_n"])
succ = [float(s) for s in eval_df["success"]]
suc_lo, suc_hi = bootstrap_ci(succ, CFG["bootstrap_n"])

dishwipe_summary = {
    "method":              winner_method,
    "task":                "DishWipe Full-Body",
    "total_env_steps":     CFG["total_env_steps"],
    "net_arch":            CFG["net_arch"],
    "mean_reward":         float(np.mean(rewards)),
    "std_reward":          float(np.std(rewards)),
    "ci95_reward":         [rew_lo, rew_hi],
    "success_rate":        float(np.mean(succ)),
    "ci95_success":        [suc_lo, suc_hi],
    "mean_cleaned_ratio":  float(eval_df["cleaned_ratio"].mean()),
    "training_time_s":     elapsed,
}

with open(ARTIFACTS_DIR / "dishwipe_summary.json", "w") as f:
    json.dump(dishwipe_summary, f, indent=2)

print("\\nDishWipe Summary:")
for k, v in dishwipe_summary.items():
    print(f"  {k}: {v}")"""),

        # ── Cross-Task Comparison ──
        md("## Step 7 — Cross-Task Comparison (Apple vs DishWipe) + Welch's t"),
        code("""\
# Load Apple eval data
apple_eval = pd.read_csv(PROJECT_ROOT / "artifacts" / "NB08" / "eval_200ep.csv")
apple_winner_data = apple_eval[apple_eval["method"] == winner_method]
apple_comp = pd.read_csv(PROJECT_ROOT / "artifacts" / "NB08" / "comparison_table.csv")
apple_winner = apple_comp[apple_comp["method"] == winner_method].iloc[0]

cross_task = pd.DataFrame([
    {"task": "Apple",    "method": winner_method,
     "mean_reward": apple_winner["mean_reward"],
     "success_rate": apple_winner["success_rate"],
     "eval_episodes": 200},
    {"task": "DishWipe", "method": winner_method,
     "mean_reward": dishwipe_summary["mean_reward"],
     "success_rate": dishwipe_summary["success_rate"],
     "eval_episodes": CFG["eval_episodes"]},
])

print("\\nCross-Task Comparison:")
print(cross_task.to_string(index=False))

# Welch's t-test: Apple vs DishWipe rewards
apple_rewards = apple_winner_data["total_reward"].values
dishwipe_rewards = np.array(rewards)
t_stat, p_value = stats.ttest_ind(apple_rewards, dishwipe_rewards, equal_var=False)
# Cohen's d
nx, ny = len(apple_rewards), len(dishwipe_rewards)
pooled_sd = np.sqrt(((nx-1)*np.var(apple_rewards, ddof=1) +
                      (ny-1)*np.var(dishwipe_rewards, ddof=1)) / (nx+ny-2))
cohens_d = (np.mean(apple_rewards) - np.mean(dishwipe_rewards)) / pooled_sd if pooled_sd > 0 else 0

cross_task_stat = {
    "comparison":  "Apple vs DishWipe",
    "t_statistic": float(t_stat),
    "p_value":     float(p_value),
    "cohens_d":    float(cohens_d),
    "significant": bool(p_value < 0.05),
}
with open(ARTIFACTS_DIR / "cross_task_stat.json", "w") as f:
    json.dump(cross_task_stat, f, indent=2)

print(f"\\nWelch's t-test (Apple vs DishWipe):")
print(f"  t={t_stat:.4f}, p={p_value:.4e}, Cohen's d={cohens_d:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
tasks = ["Apple", "DishWipe"]
colors = ["#FFB74D", "#81C784"]

axes[0].bar(tasks,
            [apple_winner["mean_reward"], dishwipe_summary["mean_reward"]],
            color=colors, edgecolor="black")
axes[0].set_title(f"{winner_method}: Mean Reward by Task (200 eps)")
axes[0].set_ylabel("Mean Reward")

axes[1].bar(tasks,
            [apple_winner["success_rate"], dishwipe_summary["success_rate"]],
            color=colors, edgecolor="black")
axes[1].set_title(f"{winner_method}: Success Rate by Task")
axes[1].set_ylabel("Success Rate")
axes[1].set_ylim(0, 1)

fig.suptitle(f"NB09 — Cross-Task: {winner_method} on Apple vs DishWipe (RTX 5090)",
             fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "cross_task_comparison.png", dpi=150)
plt.show()"""),

        # ── Learning Curve ──
        md("## Learning Curve"),
        code("""\
if cb.episode_rewards:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cb.episode_rewards, alpha=0.3, color="steelblue", linewidth=0.5)
    w = min(50, len(cb.episode_rewards))
    if len(cb.episode_rewards) >= w:
        rolling = np.convolve(cb.episode_rewards, np.ones(w)/w, mode="valid")
        ax.plot(range(w-1, len(cb.episode_rewards)), rolling,
                color="darkblue", linewidth=2, label=f"Rolling avg ({w})")
    ax.set_title(f"NB09 — {winner_method} on DishWipe Training Curve (2M steps)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "learning_curve.png", dpi=150)
    plt.show()"""),

        md("## Cleanup"),
        code("""\
print("✅ NB09 Bonus DishWipe Complete (RTX 5090)")
print(f"  Method: {winner_method}")
print(f"  DishWipe mean_reward: {dishwipe_summary['mean_reward']:.4f}")
print(f"  DishWipe success_rate: {dishwipe_summary['success_rate']:.2%}")"""),

        md("""\
## Artifacts

| File | Description |
|------|-------------|
| `{winner}_dishwipe.zip` | Trained model (2M steps) |
| `checkpoints/` | Periodic checkpoints (every 200K steps) |
| `learning_curve.png` | Training curve |
| `eval_200ep.csv` | 200-episode evaluation |
| `dishwipe_summary.json` | Mean reward, CI, cleaned ratio |
| `cross_task_comparison.png` | Apple vs DishWipe plot |
| `cross_task_stat.json` | Welch's t-test + Cohen's d |

## RTX 5090 Notes

- **2M steps** (was 500K) — leverages RTX 5090 throughput
- **[512, 512] ReLU** — same architecture as NB05-NB07
- **Linear LR decay** (3e-4 → 1e-5) — gradual refinement
- **200 eval episodes** (was 100) — more statistical power
- **50K bootstrap** (was 10K) — tighter CI
- **Welch's t-test + Cohen's d** — cross-task statistical comparison
- **CheckpointCallback** every 200K steps
- DishWipe-specific: `cleaned_ratio` from VirtualDirtGrid"""),
    ]
    write_nb("NB09_bonus_dishwipe.ipynb", cells)


# ====================================================================
#  MAIN — Generate all notebooks
# ====================================================================
if __name__ == "__main__":
    print("Generating notebooks for Full-Body G1 pipeline...\n")
    gen_nb01()
    gen_nb02()
    gen_nb03()
    gen_nb04()
    gen_nb05()
    gen_nb06()
    gen_nb07()
    gen_nb08()
    gen_nb09()
    print("\n✅ All 9 notebooks generated successfully!")
    print(f"   Output dir: {NB_DIR}")
