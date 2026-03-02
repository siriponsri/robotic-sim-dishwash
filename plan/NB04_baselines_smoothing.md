# NB04 — Baselines, Smooth Wrapper & Base Controller (Apple Full-Body) — RTX 5090 Edition

> **Status: Not started**
> **Hardware: CPU (GPU optional for speed)**
> **Task: Apple (Full Body)**

---

## Goal

Establish performance baselines for the Apple Full-Body env using Random and
Heuristic policies. Build SmoothActionWrapper and BaseController for use
in NB07 (Residual SAC). Generate leaderboard table as reference for training.

---

## Required Input

| Item | Source | Description |
|------|--------|-------------|
| NB01 completed | `artifacts/NB01/` | Envs verified |
| NB02 completed | `artifacts/NB02/action_groups.json` | Action groups known |
| NB03 completed | `artifacts/NB03/reward_contract_apple.json` | Reward structure known |
| `src/envs/apple_fullbody_env.py` | Source code | Apple Full-Body env |

---

## Expected Output / Artifacts

| File | Path | Description |
|------|------|-------------|
| `baseline_leaderboard.csv` | `artifacts/NB04/` | All baselines ranked by mean reward |
| `baseline_comparison.png` | `artifacts/NB04/` | Bar chart comparing baselines |
| `jerk_comparison.png` | `artifacts/NB04/` | Jerk (smoothness) comparison |
| `smooth_wrapper_demo.png` | `artifacts/NB04/` | Before/after EMA smoothing |
| `nb04_config.json` | `artifacts/NB04/` | Config used |

---

## Resources

| Resource | Requirement |
|----------|-------------|
| CPU | 4+ cores |
| RAM | 40 GB |
| GPU | Not required (RTX 5090 optional for speed) |
| Runtime | ~15-30 minutes (50 episodes × 5 baselines) |

---

## Steps

| Step | Purpose |
|------|---------|
| 1 | Config + imports (n_eval_episodes=**50**, max_steps=200) |
| 2 | **`evaluate_policy()` helper**: runs any policy, collects per-episode metrics |
| 3 | **Random baseline**: sample random actions, measure reward/success/fall/jerk |
| 4 | **Heuristic — Stand-Only**: just try to keep balance (zero arm actions) |
| 5 | **Heuristic — Reach Apple**: proportional control to move hand toward apple |
| 6 | **SmoothActionWrapper**: EMA filter (a_smooth = α×a_raw + (1-α)×a_prev) |
| 7 | **Smoothed Random baseline**: demonstrate jerk reduction |
| 8 | **BaseController**: heuristic + EMA smoothing (for Residual SAC in NB07) |
| 9 | **Leaderboard table**: rank all baselines |
| 10 | **Plots**: comparison bar charts + jerk analysis |
| 11 | Save artifacts + MLflow |

---

## Main Code (Pseudocode)

```python
# ── Step 1: Config ──
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path(".").resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB04"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

CFG = {
    "seed": 42,
    "env_id": "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "n_eval_episodes": 50,
    "max_steps_per_ep": 200,
    "smooth_alpha": 0.3,
}

# ── Step 2: evaluate_policy() ──
def evaluate_policy(env, policy_fn, n_episodes, max_steps, seed=42):
    """Run policy and collect per-episode metrics."""
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward, ep_steps = 0.0, 0
        prev_action = np.zeros(env.action_space.shape)
        jerks = []
        fell = False
        success = False

        for step in range(max_steps):
            action = policy_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1

            # Compute jerk (action difference)
            jerk = float(np.sum((action - prev_action) ** 2))
            jerks.append(jerk)
            prev_action = action.copy()

            if terminated:
                success = info.get("success", False)
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

# ── Step 3: Random Baseline ──
env = gym.make(CFG["env_id"], num_envs=1, obs_mode="state",
               control_mode="pd_joint_delta_pos", render_mode="rgb_array")
env = CPUGymWrapper(env)

def random_policy(obs, info):
    return env.action_space.sample()

random_results = evaluate_policy(env, random_policy,
                                  CFG["n_eval_episodes"], CFG["max_steps_per_ep"])

# ── Step 4: Heuristic — Stand-Only ──
def stand_only_policy(obs, info):
    """Zero actions on arms/hands, small corrections on legs to maintain balance."""
    action = np.zeros(env.action_space.shape)
    # Legs: try to return to standing keyframe (zero delta = hold position)
    # Arms: zero delta = hold position
    # This is the simplest "do nothing" policy
    return action

stand_results = evaluate_policy(env, stand_only_policy,
                                 CFG["n_eval_episodes"], CFG["max_steps_per_ep"])

# ── Step 5: Heuristic — Reach Apple ──
def reach_apple_policy(obs, info):
    """Proportional control: move left hand toward apple position."""
    action = np.zeros(env.action_space.shape)
    # Extract positions from obs/info
    # palm_pos = ... (from obs or env internals)
    # apple_pos = ... (from obs or info)
    # direction = apple_pos - palm_pos
    # Map direction to arm joint deltas via simple proportional control
    # Keep leg joints at zero (hold standing pose)
    return action

reach_results = evaluate_policy(env, reach_apple_policy,
                                 CFG["n_eval_episodes"], CFG["max_steps_per_ep"])

# ── Step 6: SmoothActionWrapper ──
class SmoothActionWrapper(gym.ActionWrapper):
    """EMA (Exponential Moving Average) action filter for jerk reduction."""
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

# ── Step 7: Smoothed Random ──
env_smooth = SmoothActionWrapper(env, alpha=CFG["smooth_alpha"])

smooth_random_results = evaluate_policy(
    env_smooth, random_policy,
    CFG["n_eval_episodes"], CFG["max_steps_per_ep"]
)

# ── Step 8: BaseController ──
class BaseController:
    """Heuristic + EMA for Residual SAC.
    Used as base_action in NB07: a_final = clip(a_base + β × a_residual)
    """
    def __init__(self, env, alpha=0.3):
        self.env = env
        self.alpha = alpha
        self._prev_action = None

    def get_action(self, obs, info):
        raw_action = reach_apple_policy(obs, info)
        if self._prev_action is None:
            self._prev_action = raw_action.copy()
        smoothed = self.alpha * raw_action + (1 - self.alpha) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def reset(self):
        self._prev_action = None

base_ctrl = BaseController(env, alpha=CFG["smooth_alpha"])

def base_ctrl_policy(obs, info):
    return base_ctrl.get_action(obs, info)

base_ctrl_results = evaluate_policy(env, base_ctrl_policy,
                                     CFG["n_eval_episodes"], CFG["max_steps_per_ep"])

# ── Step 9: Leaderboard ──
def summarize(results, name):
    rewards = [r["total_reward"] for r in results]
    jerks = [r["mean_jerk"] for r in results]
    return {
        "method": name,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "fall_rate": float(np.mean([r["fell"] for r in results])),
        "mean_jerk": float(np.mean(jerks)),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
    }

leaderboard = pd.DataFrame([
    summarize(random_results, "Random"),
    summarize(stand_results, "Stand-Only"),
    summarize(reach_results, "Heuristic (Reach)"),
    summarize(smooth_random_results, "Smoothed Random"),
    summarize(base_ctrl_results, "BaseController"),
])
leaderboard = leaderboard.sort_values("mean_reward", ascending=False)
leaderboard.to_csv(ARTIFACTS_DIR / "baseline_leaderboard.csv", index=False)
print(leaderboard.to_string(index=False))

# ── Step 10: Plots ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Reward comparison
methods = leaderboard["method"].tolist()
rewards = leaderboard["mean_reward"].tolist()
stds = leaderboard["std_reward"].tolist()
colors = ["gray", "khaki", "dodgerblue", "lightcoral", "mediumseagreen"]
axes[0].bar(methods, rewards, yerr=stds, color=colors, edgecolor="black", capsize=3)
axes[0].set_title("Mean Reward by Baseline")
axes[0].set_ylabel("Mean Total Reward")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha="right")

# Jerk comparison
jerks = leaderboard["mean_jerk"].tolist()
axes[1].bar(methods, jerks, color=colors, edgecolor="black")
axes[1].set_title("Mean Jerk (lower = smoother)")
axes[1].set_ylabel("Mean Jerk")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")

# Fall rate
fall_rates = leaderboard["fall_rate"].tolist()
axes[2].bar(methods, fall_rates, color=colors, edgecolor="black")
axes[2].set_title("Fall Rate (lower = better)")
axes[2].set_ylabel("Fall Rate")
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=20, ha="right")

fig.suptitle("NB04 — Baseline Comparison (Apple Full-Body)", fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "baseline_comparison.png", dpi=150)

# ── Step 11: Save + MLflow ──
env.close()
with open(ARTIFACTS_DIR / "nb04_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

print("NB04 Baselines & Smoothing PASSED ✅")
```

---

## Key Assertions

- [ ] Random baseline has very low reward and high fall rate
- [ ] Stand-only baseline survives longer but has zero manipulation progress
- [ ] Heuristic (Reach) may reach apple but unlikely to grasp successfully
- [ ] SmoothActionWrapper reduces jerk 50-80%
- [ ] BaseController ready for NB07 Residual SAC
- [ ] Leaderboard CSV saved with all baselines ranked

---

## Notes

- Full-body heuristic is much harder than upper-body — legs must maintain balance
- "Stand-Only" (zero actions) is a useful baseline showing pure balance difficulty
- BaseController combines reach heuristic + EMA smoothing
- The Residual SAC wrapper in NB07 will use: `a_final = clip(a_base + β × a_residual)`
- Expect all baselines to have near-zero success rate — this is normal for hard tasks
- SmoothActionWrapper wraps the env; BaseController is a policy-level wrapper

---

*Plan NB04 — Updated March 2026*
