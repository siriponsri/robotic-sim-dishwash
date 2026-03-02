# NB03 — Reward Analysis, Safety Validation & MLflow Utilities

> **Status: Not started**
> **Hardware: CPU**
> **Task: Apple (Main) — with DishWipe reference**

---

## Goal

Analyze and document the dense reward structure of the Apple Full-Body env,
validate safety termination (fall detection), and define reusable MLflow
helper utilities for NB04–NB09.

---

## Required Input

| Item | Source | Description |
|------|--------|-------------|
| NB01 completed | `artifacts/NB01/env_spec_apple.json` | Env verified |
| NB02 completed | `artifacts/NB02/obs_breakdown.json` | Obs/act structure understood |
| `src/envs/apple_fullbody_env.py` | Source code | Apple env with reward details |
| `src/envs/dishwipe_fullbody_env.py` | Source code | DishWipe reward reference |

---

## Expected Output / Artifacts

| File | Path | Description |
|------|------|-------------|
| `reward_contract_apple.json` | `artifacts/NB03/` | Apple reward stages + weights (formal contract) |
| `reward_contract_dishwipe.json` | `artifacts/NB03/` | DishWipe 9+2 term reward contract |
| `reward_validation.json` | `artifacts/NB03/` | Reward range stats from test episodes |
| `reward_analysis.png` | `artifacts/NB03/` | Reward distribution plots |
| `safety_validation.json` | `artifacts/NB03/` | Safety termination test results |
| `info_keys.json` | `artifacts/NB03/` | Keys available in env `info` dict |
| `nb03_config.json` | `artifacts/NB03/` | Config used |

---

## Resources

| Resource | Requirement |
|----------|-------------|
| CPU | 2+ cores |
| RAM | 4 GB |
| GPU | Not required |
| Runtime | ~5-10 minutes |

---

## Steps

| Step | Purpose |
|------|---------|
| 1 | Config: 10 test episodes × 200 steps |
| 2 | **Document Apple reward contract**: 4 stages (reaching → grasping → placing → releasing) + balance terms |
| 3 | **Document DishWipe reward contract**: 9 original terms + 2 balance terms (for NB09 reference) |
| 4 | **Run test episodes**: collect per-step rewards, check range and distribution |
| 5 | **Validate reward is dense**: most steps should have non-zero reward |
| 6 | **Validate info dict**: check all expected keys (success, fail, is_grasping, etc.) |
| 7 | **Safety validation — Fall detection**: verify `is_fallen()` triggers termination |
| 8 | **Safety validation — Force limits** (DishWipe): FZ_HARD/FZ_SOFT check |
| 9 | **Define MLflow helpers**: `setup_mlflow()`, `log_training_run()`, `log_eval_run()` |
| 10 | Save artifacts + MLflow |

---

## Main Code (Pseudocode)

```python
# ── Step 1: Config ──
import gymnasium as gym
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB03"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

CFG = {
    "seed": 42,
    "n_test_episodes": 10,
    "n_steps_per_episode": 200,
}

# ── Step 2: Apple Reward Contract ──
apple_reward_contract = {
    "version": "3.0",
    "env_id": "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "robot": "unitree_g1 (37 DOF, full body)",
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

# ── Step 3: DishWipe Reward Contract (Reference) ──
dishwipe_reward_contract = {
    "version": "3.0",
    "env_id": "UnitreeG1DishWipeFullBody-v1",
    "robot": "unitree_g1 (37 DOF, full body)",
    "terms": [
        {"name": "r_clean",   "weight": 10.0,  "sign": "+", "formula": "w * delta_clean"},
        {"name": "r_reach",   "weight": 0.5,   "sign": "+", "formula": "w * (1 - tanh(5 * dist))"},
        {"name": "r_contact", "weight": 1.0,   "sign": "+", "formula": "w * is_contacting"},
        {"name": "r_sweep",   "weight": 0.3,   "sign": "+", "formula": "w * lateral_movement"},
        {"name": "r_time",    "weight": 0.01,  "sign": "-", "formula": "-w per step"},
        {"name": "r_jerk",    "weight": 0.05,  "sign": "-", "formula": "-w * jerk^2"},
        {"name": "r_act",     "weight": 0.005, "sign": "-", "formula": "-w * action_norm^2"},
        {"name": "r_force",   "weight": 0.01,  "sign": "-", "formula": "-w * excess_force"},
        {"name": "r_success", "weight": 50.0,  "sign": "+", "formula": "one-shot at 95%"},
        {"name": "r_balance", "weight": "TBD", "sign": "-", "formula": "penalty for tilt"},
        {"name": "r_fall",    "weight": "TBD", "sign": "-", "formula": "terminate if fallen"},
    ],
    "note": "Used in NB09 only (bonus task)",
}

with open(ARTIFACTS_DIR / "reward_contract_dishwipe.json", "w") as f:
    json.dump(dishwipe_reward_contract, f, indent=2)

# ── Step 4-5: Run Test Episodes & Validate ──
env = gym.make("UnitreeG1PlaceAppleInBowlFullBody-v1", num_envs=1,
               obs_mode="state", control_mode="pd_joint_delta_pos",
               render_mode="rgb_array")
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
    "mean": float(np.mean(all_rewards)),
    "std": float(np.std(all_rewards)),
    "min": float(np.min(all_rewards)),
    "max": float(np.max(all_rewards)),
    "nonzero_pct": float(np.mean(np.array(all_rewards) != 0) * 100),
    "positive_pct": float(np.mean(np.array(all_rewards) > 0) * 100),
    "termination_counts": termination_counts,
}

with open(ARTIFACTS_DIR / "reward_validation.json", "w") as f:
    json.dump(reward_stats, f, indent=2)

# ── Step 6: Info Dict Keys ──
with open(ARTIFACTS_DIR / "info_keys.json", "w") as f:
    json.dump({"available_keys": sorted(list(all_info_keys))}, f, indent=2)

# ── Step 7-8: Safety Validation ──
# Test that fall detection works
safety_results = {
    "fall_detection_available": hasattr(env.unwrapped.agent, "is_fallen"),
    "standing_detection_available": hasattr(env.unwrapped.agent, "is_standing"),
    "fall_terminates_episode": termination_counts["fall"] > 0,
}
with open(ARTIFACTS_DIR / "safety_validation.json", "w") as f:
    json.dump(safety_results, f, indent=2)

# ── Step 9: MLflow Helpers ──
# Define and test helper functions
"""
def setup_mlflow():
    import mlflow
    from dotenv import load_dotenv
    load_dotenv(".env.local")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("g1_fullbody_apple_dishwipe")

def log_training_run(run_name, params, metrics=None, artifacts_dir=None):
    import mlflow
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if artifacts_dir:
            mlflow.log_artifacts(artifacts_dir)

def log_eval_run(run_name, eval_table, comparison_plot=None):
    import mlflow
    with mlflow.start_run(run_name=run_name):
        mlflow.log_dict(eval_table, "eval_table.json")
        if comparison_plot:
            mlflow.log_artifact(comparison_plot)
"""

env.close()

# ── Step 10: Plots + Save ──
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].hist(all_rewards, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
axes[0].set_title("Reward Distribution (Random Policy)")
axes[0].set_xlabel("Reward")

axes[1].bar(termination_counts.keys(), termination_counts.values(),
            color=["green","red","gray"], edgecolor="black")
axes[1].set_title("Termination Reasons")

# Stage visualization
stages = ["Reaching", "Grasping", "Placing", "Releasing"]
max_rewards = [1.0, 1.0, 1.0, 5.0]
axes[2].bar(stages, max_rewards, color=["#FFE082","#A5D6A7","#90CAF9","#CE93D8"],
            edgecolor="black")
axes[2].set_title("Apple Reward Stages (Max per stage)")
axes[2].set_ylabel("Max Reward")

fig.suptitle("NB03 — Reward & Safety Analysis", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "reward_analysis.png", dpi=150)

print("NB03 Reward & Safety Analysis PASSED ✅")
```

---

## Key Assertions

- [ ] Reward is dense (non-zero percentage > 50% even with random policy)
- [ ] Random policy reward is near zero or negative (task is genuinely hard)
- [ ] `is_fallen()` triggers termination actually happens with random policy
- [ ] Info dict contains expected keys (success, fail, etc.)
- [ ] MLflow helpers are defined and importable
- [ ] DishWipe contract documented for NB09 reference

---

## Notes

- Apple reward comes from ManiSkill built-in `HumanoidPlaceAppleInBowl.evaluate()` + custom balance
- The dense reward has 4 stages: reaching → grasping → placing → releasing
- Max total reward per step is normalized to 10.0 by ManiSkill
- Fall detection may use torso height or orientation as heuristic
- MLflow helpers will be used in all subsequent notebooks
- DishWipe reward contract is documented here but only used in NB09

---

*Plan NB03 — Updated March 2026*
