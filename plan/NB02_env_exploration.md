# NB02 — Environment Exploration (Apple Full-Body)

> **Status: Not started**
> **Hardware: CPU**
> **Task: Apple (Full Body)**

---

## Goal

Deep-dive into the `UnitreeG1PlaceAppleInBowlFullBody-v1` environment:
understand observation space, action groups, balance dynamics, reward
structure, and reset distribution. This builds intuition before training.

---

## Required Input

| Item | Source | Description |
|------|--------|-------------|
| NB01 completed | `artifacts/NB01/env_spec_apple.json` | Environment verified, obs/act shape known |
| `src/envs/apple_fullbody_env.py` | Source code | Custom Full-Body Apple env |
| Virtual environment with deps | `.env/` | ManiSkill + SAPIEN + matplotlib |

---

## Expected Output / Artifacts

| File | Path | Description |
|------|------|-------------|
| `obs_breakdown.json` | `artifacts/NB02/` | Observation dimensions mapped to meaning |
| `obs_breakdown.png` | `artifacts/NB02/` | Bar chart of observation groups |
| `action_groups.json` | `artifacts/NB02/` | Action dims mapped to body parts |
| `action_groups.png` | `artifacts/NB02/` | Action space visualization |
| `balance_analysis.json` | `artifacts/NB02/` | Time-to-fall stats with random policy |
| `balance_analysis.png` | `artifacts/NB02/` | Plot: steps until fall across seeds |
| `reward_per_step.png` | `artifacts/NB02/` | Reward distribution with random policy |
| `reset_distribution.png` | `artifacts/NB02/` | Apple/bowl positions across 6+ seeds |
| `complexity_comparison.png` | `artifacts/NB02/` | Full-body vs single-arm DOF comparison |
| `nb02_config.json` | `artifacts/NB02/` | Config used |

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
| 1 | Config & imports (seed=42, n_explore_episodes=10, n_seeds_reset=6) |
| 2 | Create Apple Full-Body env |
| 3 | **Observation breakdown**: map each obs dim to meaning (torso quat, angular vel, leg joints, arm joints, hand joints, qvel, apple pos, bowl pos, etc.) |
| 4 | **Action groups**: map action dims to body parts (legs 12D, torso 1D, arms 10D, fingers 14D) |
| 5 | **Balance analysis**: run random policy for N episodes, measure steps-until-fall |
| 6 | **Reward structure**: analyze per-step reward distribution with random actions |
| 7 | **Reset distribution**: visualize apple/bowl positions across multiple seeds |
| 8 | **Complexity comparison**: compare obs/act dims vs PushCube (Panda, 7 DOF) |
| 9 | **Multi-seed render**: show env from 6 different seeds as image grid |
| 10 | Save artifacts + MLflow |

---

## Main Code (Pseudocode)

```python
# ── Step 1: Config ──
import gymnasium as gym
import mani_skill.envs
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path(".").resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB02"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

CFG = {
    "seed": 42,
    "env_id": "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "n_explore_episodes": 10,
    "n_explore_steps": 200,
    "n_seeds_reset": 6,
}

# ── Step 2: Create Env ──
env = gym.make(CFG["env_id"], num_envs=1, obs_mode="state",
               control_mode="pd_joint_delta_pos", render_mode="rgb_array")
env = CPUGymWrapper(env)
obs, info = env.reset(seed=CFG["seed"])

# ── Step 3: Observation Breakdown ──
# Full-body G1 obs includes:
#   - torso orientation (quaternion, 4D)
#   - torso angular velocity (3D)
#   - joint positions — legs (12D) + torso (1D) + arms (10D) + hands (14D) = 37D
#   - joint velocities (37D)
#   - apple position (3D)
#   - apple rotation (quaternion, 4D?)
#   - bowl position (3D)
#   - hand positions left/right (6D)
#   - grasp state etc.
#
# Exact mapping determined by inspecting obs values vs known positions

obs_total = obs.shape[0]
obs_breakdown = {
    "total_dims": obs_total,
    "groups": {
        "proprioception (qpos+qvel)": "~74D (37 joints × 2)",
        "torso_orientation": "4D (quaternion)",
        "torso_angular_vel": "3D",
        "apple_position": "3D",
        "bowl_position": "3D",
        "hand_positions": "6D (left 3 + right 3)",
        "other_extras": f"remaining dims",
    }
}

# Visualize as horizontal bar chart
groups = list(obs_breakdown["groups"].keys())
# ... actual dim counts extracted from env ...

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(groups, sizes, color="steelblue", edgecolor="black")
ax.set_xlabel("Dimensions")
ax.set_title(f"Observation Space Breakdown ({obs_total}D total)")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "obs_breakdown.png", dpi=150)

# ── Step 4: Action Groups ──
act_dim = env.action_space.shape[0]  # Should be 37

# Access robot joint info
agent = env.unwrapped.agent
joint_names = [j.name for j in agent.robot.active_joints]

# Group by body part
action_groups = {
    "left_leg": [n for n in joint_names if "left" in n and any(k in n for k in ["hip", "knee", "ankle"])],
    "right_leg": [n for n in joint_names if "right" in n and any(k in n for k in ["hip", "knee", "ankle"])],
    "torso": [n for n in joint_names if "torso" in n],
    "left_arm": [n for n in joint_names if "left" in n and any(k in n for k in ["shoulder", "elbow"])],
    "right_arm": [n for n in joint_names if "right" in n and any(k in n for k in ["shoulder", "elbow"])],
    "left_hand": [n for n in joint_names if "left" in n and any(k in n for k in ["zero","one","two","three","four","five","six"])],
    "right_hand": [n for n in joint_names if "right" in n and any(k in n for k in ["zero","one","two","three","four","five","six"])],
}
group_sizes = {k: len(v) for k, v in action_groups.items()}

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(group_sizes.keys(), group_sizes.values(),
              color=["#E57373","#E57373","#FFB74D","#64B5F6","#64B5F6","#81C784","#81C784"])
ax.set_ylabel("Number of Joints")
ax.set_title(f"Action Space by Body Part ({act_dim}D total)")
for b, v in zip(bars, group_sizes.values()):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, str(v),
            ha="center", fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "action_groups.png", dpi=150)

# ── Step 5: Balance Analysis ──
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
        fall_steps.append(CFG["n_explore_steps"])  # survived full episode

balance_stats = {
    "mean_steps_to_fall": float(np.mean(fall_steps)),
    "min_steps_to_fall": int(np.min(fall_steps)),
    "max_steps_to_fall": int(np.max(fall_steps)),
    "std_steps_to_fall": float(np.std(fall_steps)),
    "survived_full_episode": int(sum(s == CFG["n_explore_steps"] for s in fall_steps)),
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(fall_steps)), fall_steps, color="coral", edgecolor="black")
ax.axhline(np.mean(fall_steps), color="red", ls="--", label=f"mean={np.mean(fall_steps):.0f}")
ax.set_xlabel("Episode")
ax.set_ylabel("Steps before Fall/Termination")
ax.set_title("Balance Analysis — Random Policy")
ax.legend()
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "balance_analysis.png", dpi=150)

# ── Step 6: Reward Structure ──
obs, info = env.reset(seed=42)
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
axes[0].legend()

axes[1].plot(np.cumsum(rewards), linewidth=1.2, color="darkviolet")
axes[1].set_title("Cumulative Reward")

axes[2].hist(rewards, bins=40, color="darkviolet", edgecolor="black", alpha=0.7)
axes[2].set_title("Reward Distribution")

fig.suptitle("Apple Full-Body — Random Policy Reward Analysis")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "reward_per_step.png", dpi=150)

# ── Step 7: Reset Distribution ──
positions = {"apple": [], "bowl": []}
for seed in range(CFG["n_seeds_reset"]):
    obs, info = env.reset(seed=seed * 10)
    # Extract apple/bowl positions from info or obs
    # positions["apple"].append(info.get("apple_pos", obs[some_slice]))
    # positions["bowl"].append(info.get("bowl_pos", obs[some_slice]))

fig, ax = plt.subplots(figsize=(8, 6))
# Scatter plot apple/bowl positions across seeds
ax.set_title("Reset Distribution — Apple & Bowl Positions")
fig.savefig(ARTIFACTS_DIR / "reset_distribution.png", dpi=150)

# ── Step 8: Complexity Comparison ──
env_panda = gym.make("PickCube-v1", num_envs=1, obs_mode="state",
                     control_mode="pd_joint_delta_pos", render_mode="rgb_array")
env_panda = CPUGymWrapper(env_panda)
obs_p, _ = env_panda.reset(seed=0)

comparison = {
    "Panda (PickCube)": {"obs": obs_p.shape[0],
                         "act": env_panda.action_space.shape[0]},
    "G1 Full Body (Apple)": {"obs": obs.shape[0],
                              "act": env.action_space.shape[0]},
}
env_panda.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
names = list(comparison.keys())
colors = ["steelblue", "darkviolet"]
obs_vals = [comparison[n]["obs"] for n in names]
act_vals = [comparison[n]["act"] for n in names]
axes[0].bar(names, obs_vals, color=colors, edgecolor="black")
axes[0].set_title("Observation Dimensions")
axes[1].bar(names, act_vals, color=colors, edgecolor="black")
axes[1].set_title("Action Dimensions")
fig.suptitle("Single-Arm vs Full-Body Humanoid Complexity")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "complexity_comparison.png", dpi=150)

# ── Step 9: Multi-seed Render Grid ──
images = []
for seed in range(6):
    env.reset(seed=seed * 42)
    try:
        frame = env.render()
        images.append(frame)
    except:
        pass

if images:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(f"Seed {i*42}")
        ax.axis("off")
    fig.suptitle("Apple Full-Body — Reset Distribution")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "reset_renders.png", dpi=150)

# ── Step 10: Save + MLflow ──
env.close()
with open(ARTIFACTS_DIR / "nb02_config.json", "w") as f:
    json.dump(CFG, f, indent=2)
with open(ARTIFACTS_DIR / "obs_breakdown.json", "w") as f:
    json.dump(obs_breakdown, f, indent=2)
with open(ARTIFACTS_DIR / "action_groups.json", "w") as f:
    json.dump(action_groups, f, indent=2, default=str)
with open(ARTIFACTS_DIR / "balance_analysis.json", "w") as f:
    json.dump(balance_stats, f, indent=2)
```

---

## Key Assertions

- [ ] Obs dims correctly mapped to physical meaning
- [ ] Action space groups total = 37
- [ ] Balance analysis shows robot falls quickly with random policy
- [ ] Reward near zero or negative with random policy (task is hard)
- [ ] Apple/bowl positions vary across seeds

---

## Notes

- Full-body G1 obs includes **leg joint positions/velocities** that upper body doesn't have
- Random policy will almost certainly cause immediate falling → very low reward
- This is **expected** — it demonstrates why RL is needed
- Exact obs mapping requires inspecting env source or stepping through dims experimentally
- The `info` dict may contain `is_grasping`, `apple_pos`, `bowl_pos` etc.
- Compare with reference code `Robotics Simulation.py` section 18 for baseline

---

*Plan NB02 — Updated March 2026*
