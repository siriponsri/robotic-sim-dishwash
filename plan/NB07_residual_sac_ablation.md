# NB07 — Residual SAC + Beta Ablation (Apple Full-Body) — RTX 5090 Edition

> **Status: Not started**
> **Hardware: GPU (RTX 5090, 32 GB VRAM)**
> **Task: Apple (Full Body)**

---

## Goal

Combine the BaseController from NB04 (heuristic + EMA) with a learned
residual policy via SAC. Train **5 beta variants** (expanded from 3) and
select the best. Same budget per run as NB05/NB06. RTX 5090 allows
**10,000,000 total env steps** across all variants (5 × 2M).

---

## Required Input

| Item | Source | Description |
|------|--------|-------------|
| NB04 completed | `artifacts/NB04/` | BaseController defined and tested |
| NB06 completed | `artifacts/NB06/nb06_config.json` | SAC hyperparameters (reuse for fairness) |
| `src/envs/apple_fullbody_env.py` | Source code | Apple Full-Body env |
| GPU available | RunPod / Local | **RTX 5090 (32 GB VRAM)** |

---

## Expected Output / Artifacts

| File | Path | Description |
|------|------|-------------|
| `residual_apple_beta0.10.zip` | `artifacts/NB07/` | Model for β=0.10 |
| `residual_apple_beta0.25.zip` | `artifacts/NB07/` | Model for β=0.25 |
| `residual_apple_beta0.50.zip` | `artifacts/NB07/` | Model for β=0.50 |
| `residual_apple_beta0.75.zip` | `artifacts/NB07/` | Model for β=0.75 |
| `residual_apple_beta1.00.zip` | `artifacts/NB07/` | Model for β=1.00 |
| `ablation_table.csv` | `artifacts/NB07/` | Performance per beta |
| `ablation_plot.png` | `artifacts/NB07/` | Bar chart comparison |
| `ablation_learning_curves.png` | `artifacts/NB07/` | Learning curves overlay |
| `best_beta.json` | `artifacts/NB07/` | Best beta + reasoning |
| `nb07_config.json` | `artifacts/NB07/` | Config used |

---

## Resources (RTX 5090)

| Resource | Requirement |
|----------|-------------|
| CPU | 8+ cores |
| RAM | **40 GB** (10M buffer per variant, sequential) |
| GPU | **RTX 5090 (32 GB VRAM)** |
| Disk | 25+ GB (5 models + checkpoints) |
| Runtime | **~8-12 hours** (5 × 2M steps) |

---

## Steps

| Step | Purpose |
|------|---------|
| 1 | Config: **beta_values=[0.10, 0.25, 0.50, 0.75, 1.00]**, SAC from NB06 |
| 2 | Define `heuristic_policy()` — reach apple, maintain balance |
| 3 | Define `BaseController` — heuristic + EMA smoothing |
| 4 | Define `ResidualActionWrapper` — `a_final = clip(a_base + β × a_residual)` |
| 5 | Train Residual SAC for each β (2M steps each!) |
| 6 | Quick eval (20 eps) for each β |
| 7 | Ablation table (DataFrame) |
| 8 | Select best β by mean_reward |
| 9 | Overlaid learning curves + ablation plots |
| 10 | Save artifacts + MLflow |

---

## RTX 5090 Ablation Budget

| β Value | Env Steps | Est. Time | GPU Budget |
|---------|-----------|-----------|-----------|
| β = 0.10 | 2,000,000 | ~2 hours | Sequential |
| β = 0.25 | 2,000,000 | ~2 hours | Sequential |
| β = 0.50 | 2,000,000 | ~2 hours | Sequential |
| β = 0.75 | 2,000,000 | ~2 hours | Sequential |
| β = 1.00 | 2,000,000 | ~2 hours | Sequential |
| **Total** | **10,000,000** | **~10-12 hours** | **RTX 5090 all-in** |

---

## Main Code (Pseudocode)

```python
# ── Step 1: Config ──
import gymnasium as gym
import numpy as np
import torch
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path(".").resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB07"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

IS_GPU = torch.cuda.is_available()

# Load SAC config from NB06 for fairness
with open(PROJECT_ROOT / "artifacts" / "NB06" / "nb06_config.json") as f:
    sac_cfg = json.load(f)

CFG = {
    "seed": 42,
    "env_id": "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "control_mode": "pd_joint_delta_pos",
    # ── 5 Beta Variants (RTX 5090 expanded) ──
    "beta_values": [0.10, 0.25, 0.50, 0.75, 1.00],
    "smooth_alpha": 0.3,
    # ── SAC config (from NB06, fairness) ──
    "total_env_steps": sac_cfg["total_env_steps"],  # 2,000,000
    "learning_rate_start": sac_cfg["learning_rate_start"],
    "learning_rate_end": sac_cfg["learning_rate_end"],
    "buffer_size": sac_cfg["buffer_size"],  # 10,000,000
    "batch_size": sac_cfg["batch_size"],  # 1024
    "tau": sac_cfg["tau"],
    "gamma": sac_cfg["gamma"],
    "ent_coef": sac_cfg["ent_coef"],
    "net_arch": sac_cfg["net_arch"],  # [512, 512]
    "learning_starts": sac_cfg["learning_starts"],  # 10,000
    "gradient_steps": sac_cfg["gradient_steps"],  # 2
    "eval_episodes": 20,
}

with open(ARTIFACTS_DIR / "nb07_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

print(f"Beta variants: {CFG['beta_values']}")
print(f"Steps per variant: {CFG['total_env_steps']:,}")
print(f"Total GPU budget: {CFG['total_env_steps'] * len(CFG['beta_values']):,} steps")

# ── Step 2: Heuristic Policy ──
def heuristic_policy(obs, env):
    """Reach toward apple with arm, maintain balance with legs."""
    action = np.zeros(env.action_space.shape[0])
    return action

# ── Step 3: BaseController ──
class BaseController:
    """Heuristic + EMA smoothing for Residual SAC."""
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

# ── Step 4: ResidualActionWrapper ──
class ResidualActionWrapper(gym.ActionWrapper):
    """a_final = clip(a_base + β × a_residual)"""
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
        if isinstance(result, tuple) and len(result) == 5:
            self._current_obs = result[0]
        return result

    def action(self, residual_action):
        base_action = self.base_controller.get_action(self._current_obs)
        final = base_action + self.beta * residual_action
        return np.clip(final, self.action_space.low, self.action_space.high)

# ── Step 5: Train for each beta ──
def linear_schedule(initial_value, final_value=1e-5):
    def func(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return func

ablation_results = []
all_training_curves = {}

for i, beta in enumerate(CFG["beta_values"]):
    print(f"\n{'='*60}")
    print(f"  [{i+1}/{len(CFG['beta_values'])}] Training Residual SAC — β={beta}")
    print(f"  Budget: {CFG['total_env_steps']:,} steps")
    print(f"{'='*60}")

    raw_env = gym.make(CFG["env_id"], num_envs=1, obs_mode="state",
                       control_mode=CFG["control_mode"], render_mode="rgb_array")
    raw_env = CPUGymWrapper(raw_env)
    base_ctrl = BaseController(raw_env, alpha=CFG["smooth_alpha"])
    env = ResidualActionWrapper(raw_env, base_ctrl, beta=beta)

    lr_schedule = linear_schedule(CFG["learning_rate_start"], CFG["learning_rate_end"])

    model = SAC(
        "MlpPolicy", env,
        learning_rate=lr_schedule,
        buffer_size=CFG["buffer_size"],
        batch_size=CFG["batch_size"],
        tau=CFG["tau"],
        gamma=CFG["gamma"],
        ent_coef=CFG["ent_coef"],
        learning_starts=CFG["learning_starts"],
        gradient_steps=CFG["gradient_steps"],
        policy_kwargs={
            "net_arch": CFG["net_arch"],
            "activation_fn": torch.nn.ReLU,
        },
        verbose=0,
        seed=CFG["seed"],
        device="auto",
    )

    # Callback
    class BetaTrainCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
            return True

    cb = BetaTrainCallback()

    start_time = time.time()
    model.learn(total_timesteps=CFG["total_env_steps"], callback=cb, progress_bar=True)
    elapsed = time.time() - start_time

    # Save model
    beta_str = f"{beta:.2f}"
    model_name = f"residual_apple_beta{beta_str}"
    model.save(str(ARTIFACTS_DIR / model_name))

    # Store training curve
    all_training_curves[beta_str] = cb.episode_rewards.copy()

    # ── Step 6: Eval ──
    eval_rewards = []
    eval_successes = []
    for ep in range(CFG["eval_episodes"]):
        obs, info = env.reset(seed=ep * 100)
        ep_reward = 0.0
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            if terminated or truncated:
                break
        eval_rewards.append(ep_reward)
        eval_successes.append(info.get("success", False))

    ablation_results.append({
        "beta": beta,
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "success_rate": float(np.mean(eval_successes)),
        "training_time_s": elapsed,
        "steps_per_sec": CFG["total_env_steps"] / elapsed,
    })

    env.close()
    print(f"  β={beta}: mean_reward={np.mean(eval_rewards):.4f}, "
          f"success={np.mean(eval_successes):.2%}, "
          f"time={elapsed/60:.1f}min")

# ── Step 7: Ablation Table ──
ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv(ARTIFACTS_DIR / "ablation_table.csv", index=False)
print("\n" + "="*60)
print("Ablation Table:")
print(ablation_df.to_string(index=False))

# ── Step 8: Select Best Beta ──
best_idx = ablation_df["mean_reward"].idxmax()
best_beta = ablation_df.loc[best_idx, "beta"]
best_info = {
    "best_beta": float(best_beta),
    "best_mean_reward": float(ablation_df.loc[best_idx, "mean_reward"]),
    "best_success_rate": float(ablation_df.loc[best_idx, "success_rate"]),
    "best_model": f"residual_apple_beta{best_beta:.2f}.zip",
    "all_betas_tested": CFG["beta_values"],
    "total_gpu_budget": CFG["total_env_steps"] * len(CFG["beta_values"]),
}
with open(ARTIFACTS_DIR / "best_beta.json", "w") as f:
    json.dump(best_info, f, indent=2)
print(f"\n🏆 Best beta: {best_beta:.2f} "
      f"(mean_reward={best_info['best_mean_reward']:.4f})")

# ── Step 9: Ablation Plots ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

betas = ablation_df["beta"].tolist()
rewards = ablation_df["mean_reward"].tolist()
stds = ablation_df["std_reward"].tolist()
successes = ablation_df["success_rate"].tolist()
times = [t / 60 for t in ablation_df["training_time_s"].tolist()]

colors = ["#E57373", "#FFB74D", "#64B5F6", "#81C784", "#BA68C8"]
x_labels = [f"β={b:.2f}" for b in betas]

# Mean Reward
axes[0].bar(x_labels, rewards, yerr=stds, color=colors, edgecolor="black", capsize=5)
axes[0].set_title("Mean Reward by β")
axes[0].set_ylabel("Mean Reward")
axes[0].grid(axis="y", alpha=0.3)

# Success Rate
axes[1].bar(x_labels, successes, color=colors, edgecolor="black")
axes[1].set_title("Success Rate by β")
axes[1].set_ylabel("Success Rate")
axes[1].set_ylim(0, 1)
axes[1].grid(axis="y", alpha=0.3)

# Training Time
axes[2].bar(x_labels, times, color=colors, edgecolor="black")
axes[2].set_title("Training Time by β")
axes[2].set_ylabel("Time (minutes)")
axes[2].grid(axis="y", alpha=0.3)

fig.suptitle("NB07 — Residual SAC β Ablation (RTX 5090, 2M steps/variant)",
             fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "ablation_plot.png", dpi=150)

# Learning Curves Overlay
fig, ax = plt.subplots(figsize=(12, 6))
for i, (beta_str, curve) in enumerate(all_training_curves.items()):
    if curve:
        window = min(50, len(curve))
        if len(curve) >= window:
            rolling = np.convolve(curve, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(curve)), rolling,
                    color=colors[i], linewidth=2, label=f"β={beta_str}")
ax.set_title("Learning Curves — All β Variants (rolling avg)")
ax.set_xlabel("Episode")
ax.set_ylabel("Episode Reward")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "ablation_learning_curves.png", dpi=150)

# ── Step 10: MLflow ──
print(f"\nNB07 Residual SAC Ablation PASSED ✅")
print(f"Total GPU time: {sum(r['training_time_s'] for r in ablation_results)/60:.1f} min")
```

---

## Key Assertions

- [ ] 5 beta variants tested: {0.10, 0.25, 0.50, 0.75, 1.00}
- [ ] TOTAL_ENV_STEPS per run = 2,000,000 (same as NB05/NB06, fairness)
- [ ] SAC hyperparams loaded from NB06 config (fairness)
- [ ] All 5 models saved + best_beta.json
- [ ] Learning curves show performance differences across betas
- [ ] Total GPU budget: 10,000,000 steps (RTX 5090 can handle it)

---

## Notes

- **RTX 5090 Advantage**: Can afford 5 variants × 2M steps = 10M total budget
- **Residual Policy** (Silver et al., 2018): `a_final = a_base + β × a_residual`
- β=0.10: Very conservative (mostly heuristic)
- β=0.25: Conservative (small learned corrections)
- β=0.50: Balanced (equal heuristic + learned)
- β=0.75: Aggressive (mostly learned)
- β=1.00: Full residual (completely learned + heuristic offset)
- Each variant uses identical SAC config from NB06 for fairness
- All variants train sequentially (one at a time to avoid memory conflicts)
- ResidualActionWrapper is transparent to SAC — SAC sees normal action space
- Expected: β=0.25-0.50 may perform best (structured exploration advantage)

---

*Plan NB07 — RTX 5090 Edition — Updated March 2026*
