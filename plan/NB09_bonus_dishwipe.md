# NB09 — Bonus: DishWipe Full-Body (Winner Only) — RTX 5090 Edition

> **Status: Not started**
> **Hardware: GPU (RTX 5090, 32 GB VRAM)**
> **Task: DishWipe (Full Body) — Bonus**

---

## Goal

Self-contained bonus notebook. Register and verify the DishWipe Full-Body
env, then **train only the winning method from NB08** on the DishWipe task
with full RTX 5090 budget:
- **2,000,000 env steps** (same as Apple training)
- **200-episode evaluation** with CI + t-test
- **Video recording** (best/worst episodes)
- Cross-task comparison: Apple vs DishWipe

---

## Required Input

| Item | Source | Description |
|------|--------|-------------|
| Winner declaration | `artifacts/NB08/best_method.json` | Which method won Apple |
| Apple comparison | `artifacts/NB08/comparison_table.csv` | Cross-task reference |
| NB05/NB06 configs | `artifacts/NB05/nb05_config.json` etc. | Method hyperparameters |
| `src/envs/dishwipe_fullbody_env.py` | Source code | DishWipe Full-Body env |
| GPU available | RunPod / Local | **RTX 5090 (32 GB VRAM)** |

---

## Expected Output / Artifacts

| File | Path | Description |
|------|------|-------------|
| `{winner}_dishwipe.zip` | `artifacts/NB09/` | Trained model |
| `checkpoints/` | `artifacts/NB09/checkpoints/` | Checkpoints every 200K steps |
| `learning_curve.png` | `artifacts/NB09/` | Training curves |
| `eval_200ep.csv` | `artifacts/NB09/` | 200-episode evaluation |
| `dishwipe_summary.json` | `artifacts/NB09/` | Stats + CI |
| `cross_task_comparison.png` | `artifacts/NB09/` | Apple vs DishWipe |
| `cross_task_table.csv` | `artifacts/NB09/` | Cross-task metrics |
| `videos/` | `artifacts/NB09/videos/` | Best/worst episodes MP4 |
| `nb09_config.json` | `artifacts/NB09/` | Config |

---

## Resources (RTX 5090)

| Resource | Requirement |
|----------|-------------|
| CPU | 8+ cores |
| RAM | **40 GB** |
| GPU | **RTX 5090 (32 GB VRAM)** |
| Disk | 15+ GB |
| Runtime | **~2-4 hours** (2M training + 200-ep eval + video) |

---

## Steps

| Step | Purpose |
|------|---------|
| 1 | Load winner from `best_method.json` |
| 2 | Config: **2M training budget**, RTX 5090 hyperparameters |
| 3 | Quick DishWipe Full-Body smoke test (5 random steps) |
| 4 | Train winner on DishWipe (2M steps, checkpoints) |
| 5 | **200-episode deterministic evaluation** |
| 6 | Bootstrap 95% CI (50,000 resamples) |
| 7 | **Video recording** — best/worst episodes |
| 8 | **Cross-task comparison**: Apple vs DishWipe metrics |
| 9 | Plots (learning curve, cross-task, distribution) |
| 10 | Save artifacts + MLflow |

---

## Main Code (Pseudocode)

```python
# ── Step 1: Load Winner ──
import gymnasium as gym
import numpy as np
import torch
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path(".").resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB09"
VIDEO_DIR = ARTIFACTS_DIR / "videos"
CKPT_DIR = ARTIFACTS_DIR / "checkpoints"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

from src.envs import UnitreeG1DishWipeFullBodyEnv

with open(PROJECT_ROOT / "artifacts" / "NB08" / "best_method.json") as f:
    winner_info = json.load(f)

winner_method = winner_info["winner"]
print(f"Winner from NB08: {winner_method}")
print(f"Apple mean_reward: {winner_info['mean_reward']:.4f}")
print(f"Apple success_rate: {winner_info['success_rate']:.2%}")

IS_GPU = torch.cuda.is_available()

# ── Step 2: Config (RTX 5090) ──
CFG = {
    "seed": 42,
    "env_id": "UnitreeG1DishWipeFullBody-v1",
    "control_mode": "pd_joint_delta_pos",
    "obs_mode": "state",
    "winner_method": winner_method,
    # ── RTX 5090 Budget ──
    "total_env_steps": 2_000_000 if IS_GPU else 20_000,
    "eval_episodes": 200,
    "max_steps_per_ep": 1000,
    "bootstrap_n": 50_000,
    "ci_level": 0.95,
    "checkpoint_freq": 200_000,
}
with open(ARTIFACTS_DIR / "nb09_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

# ── Step 3: Smoke Test DishWipe ──
print("\n--- DishWipe Full-Body Smoke Test ---")
env = gym.make(CFG["env_id"], num_envs=1, obs_mode="state",
               control_mode=CFG["control_mode"])
env = CPUGymWrapper(env)
obs, info = env.reset()
print(f"obs shape: {obs.shape}")
print(f"act shape: {env.action_space.shape}")
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
print(f"Smoke OK, reward sample: {reward:.4f}")
env.close()

# ── Step 4: Train Winner (2M steps, RTX 5090) ──
print(f"\n{'='*60}")
print(f"  Training {winner_method} on DishWipe Full-Body")
print(f"  Budget: {CFG['total_env_steps']:,} steps (RTX 5090)")
print(f"{'='*60}")

# Load method-specific config from Apple training
if winner_method == "PPO":
    with open(PROJECT_ROOT / "artifacts" / "NB05" / "nb05_config.json") as f:
        method_cfg = json.load(f)
else:
    with open(PROJECT_ROOT / "artifacts" / "NB06" / "nb06_config.json") as f:
        method_cfg = json.load(f)

# Create env based on method
if winner_method == "PPO" and IS_GPU:
    n_envs = method_cfg.get("n_envs", 64)
    env = gym.make(CFG["env_id"], num_envs=n_envs, sim_backend="gpu",
                   obs_mode="state", control_mode=CFG["control_mode"],
                   render_mode="rgb_array")
else:
    env = gym.make(CFG["env_id"], num_envs=1, obs_mode="state",
                   control_mode=CFG["control_mode"], render_mode="rgb_array")
    env = CPUGymWrapper(env)

# For Residual-SAC, wrap env
if winner_method == "Residual-SAC":
    with open(PROJECT_ROOT / "artifacts" / "NB07" / "best_beta.json") as f:
        beta_info = json.load(f)
    base_ctrl = BaseController(env, alpha=0.3)
    env = ResidualActionWrapper(env, base_ctrl, beta=beta_info["best_beta"])

# LR schedule
def linear_schedule(initial_value, final_value=1e-5):
    def func(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return func

lr_start = method_cfg.get("learning_rate_start", 3e-4)
lr_end = method_cfg.get("learning_rate_end", 1e-5)
lr_schedule = linear_schedule(lr_start, lr_end)

# Create model
if winner_method == "PPO":
    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr_schedule,
        n_steps=method_cfg.get("n_steps", 512),
        batch_size=method_cfg.get("batch_size", 2048),
        n_epochs=method_cfg.get("n_epochs", 10),
        gamma=method_cfg.get("gamma", 0.99),
        clip_range=method_cfg.get("clip_range", 0.2),
        ent_coef=method_cfg.get("ent_coef", 0.005),
        max_grad_norm=method_cfg.get("max_grad_norm", 0.5),
        policy_kwargs={
            "net_arch": method_cfg.get("net_arch", [512, 512]),
            "activation_fn": torch.nn.ReLU,
        },
        verbose=1, seed=CFG["seed"], device="auto",
    )
elif winner_method in ("SAC", "Residual-SAC"):
    model = SAC(
        "MlpPolicy", env,
        learning_rate=lr_schedule,
        buffer_size=method_cfg.get("buffer_size", 10_000_000),
        batch_size=method_cfg.get("batch_size", 1024),
        tau=method_cfg.get("tau", 0.005),
        gamma=method_cfg.get("gamma", 0.99),
        ent_coef=method_cfg.get("ent_coef", "auto"),
        learning_starts=method_cfg.get("learning_starts", 10_000),
        gradient_steps=method_cfg.get("gradient_steps", 2),
        policy_kwargs={
            "net_arch": method_cfg.get("net_arch", [512, 512]),
            "activation_fn": torch.nn.ReLU,
        },
        verbose=1, seed=CFG["seed"], device="auto",
    )

# Callbacks
class TrainLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True

train_cb = TrainLogCallback()
ckpt_cb = CheckpointCallback(
    save_freq=CFG["checkpoint_freq"],
    save_path=str(CKPT_DIR),
    name_prefix=f"{winner_method.lower().replace('-','_')}_dishwipe",
)

start_time = time.time()
model.learn(total_timesteps=CFG["total_env_steps"],
            callback=[train_cb, ckpt_cb], progress_bar=True)
elapsed = time.time() - start_time

model_name = f"{winner_method.lower().replace('-','_')}_dishwipe"
model.save(str(ARTIFACTS_DIR / model_name))
env.close()
print(f"Training: {elapsed/60:.1f} min, {CFG['total_env_steps']/elapsed:.0f} steps/sec")

# ── Step 5: Evaluate 200 Episodes ──
eval_env = gym.make(CFG["env_id"], num_envs=1, obs_mode="state",
                    control_mode=CFG["control_mode"], render_mode="rgb_array")
eval_env = CPUGymWrapper(eval_env)

if winner_method == "Residual-SAC":
    base_ctrl = BaseController(eval_env, alpha=0.3)
    eval_env = ResidualActionWrapper(eval_env, base_ctrl,
                                      beta=beta_info["best_beta"])

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

# ── Step 6: Bootstrap CI ──
def bootstrap_ci(data, n_boot=50000, ci=0.95):
    data = np.array(data)
    rng = np.random.default_rng(42)
    boot = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boot, (1-ci)/2*100)), float(np.percentile(boot, (1+ci)/2*100))

rewards = eval_df["total_reward"].tolist()
rew_lo, rew_hi = bootstrap_ci(rewards)
succ = [float(s) for s in eval_df["success"]]
suc_lo, suc_hi = bootstrap_ci(succ)

dishwipe_summary = {
    "method": winner_method,
    "task": "DishWipe Full-Body",
    "mean_reward": float(np.mean(rewards)),
    "std_reward": float(np.std(rewards)),
    "ci95_reward": [rew_lo, rew_hi],
    "success_rate": float(np.mean(succ)),
    "ci95_success": [suc_lo, suc_hi],
    "mean_cleaned_ratio": float(eval_df["cleaned_ratio"].mean()),
    "training_time_s": elapsed,
    "total_env_steps": CFG["total_env_steps"],
    "eval_episodes": CFG["eval_episodes"],
    "bootstrap_n": CFG["bootstrap_n"],
}
with open(ARTIFACTS_DIR / "dishwipe_summary.json", "w") as f:
    json.dump(dishwipe_summary, f, indent=2)

print(f"DishWipe: mean_reward={np.mean(rewards):.4f}, success={np.mean(succ):.2%}")

# ── Step 7: Video Recording ──
best_ep = int(np.argmax(rewards))
worst_ep = int(np.argmin(rewards))

for tag, seed in [("best", best_ep), ("worst", worst_ep)]:
    vid_env = gym.make(CFG["env_id"], num_envs=1, obs_mode="state",
                       control_mode=CFG["control_mode"], render_mode="rgb_array")
    vid_env = CPUGymWrapper(vid_env)
    if winner_method == "Residual-SAC":
        base_ctrl = BaseController(vid_env, alpha=0.3)
        vid_env = ResidualActionWrapper(vid_env, base_ctrl, beta=beta_info["best_beta"])

    frames = []
    obs, info = vid_env.reset(seed=seed)
    for step in range(CFG["max_steps_per_ep"]):
        frame = vid_env.render()
        if frame is not None:
            frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = vid_env.step(action)
        if terminated or truncated:
            break

    if frames:
        imageio.mimsave(str(VIDEO_DIR / f"dishwipe_{tag}.mp4"), frames, fps=30)
    vid_env.close()

# ── Step 8: Cross-Task Comparison ──
apple_comp = pd.read_csv(PROJECT_ROOT / "artifacts" / "NB08" / "comparison_table.csv")
apple_winner = apple_comp[apple_comp["method"] == winner_method].iloc[0]

cross_task = pd.DataFrame([
    {"task": "Apple (Main)", "method": winner_method,
     "mean_reward": float(apple_winner["mean_reward"]),
     "std_reward": float(apple_winner["std_reward"]),
     "success_rate": float(apple_winner["success_rate"]),
     "eval_episodes": 200},
    {"task": "DishWipe (Bonus)", "method": winner_method,
     "mean_reward": dishwipe_summary["mean_reward"],
     "std_reward": dishwipe_summary["std_reward"],
     "success_rate": dishwipe_summary["success_rate"],
     "eval_episodes": 200},
])
cross_task.to_csv(ARTIFACTS_DIR / "cross_task_table.csv", index=False)
print("\nCross-Task Comparison:")
print(cross_task.to_string(index=False))

# ── Step 9: Plots ──
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Learning curve
ep_rewards = train_cb.episode_rewards
if ep_rewards:
    axes[0, 0].plot(ep_rewards, alpha=0.2, color="steelblue", linewidth=0.5)
    window = min(50, len(ep_rewards))
    if len(ep_rewards) >= window:
        rolling = np.convolve(ep_rewards, np.ones(window)/window, mode="valid")
        axes[0, 0].plot(range(window-1, len(ep_rewards)), rolling,
                        color="darkblue", linewidth=2)
    axes[0, 0].set_title("Training: Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].grid(True, alpha=0.3)

# Cross-task bar chart
tasks = ["Apple", "DishWipe"]
task_rewards = [float(apple_winner["mean_reward"]), dishwipe_summary["mean_reward"]]
task_success = [float(apple_winner["success_rate"]), dishwipe_summary["success_rate"]]
colors = ["#FFB74D", "#81C784"]

axes[0, 1].bar(tasks, task_rewards, color=colors, edgecolor="black")
axes[0, 1].set_title(f"{winner_method}: Mean Reward by Task")
axes[0, 1].set_ylabel("Mean Reward")

axes[1, 0].bar(tasks, task_success, color=colors, edgecolor="black")
axes[1, 0].set_title(f"{winner_method}: Success Rate by Task")
axes[1, 0].set_ylabel("Success Rate")
axes[1, 0].set_ylim(0, 1)

# Reward distribution
axes[1, 1].hist(rewards, bins=40, color="#64B5F6", edgecolor="black", alpha=0.8)
axes[1, 1].axvline(np.mean(rewards), color="red", linestyle="--", label="Mean")
axes[1, 1].set_title(f"DishWipe Reward Distribution ({CFG['eval_episodes']} eps)")
axes[1, 1].set_xlabel("Total Reward")
axes[1, 1].legend()

fig.suptitle(f"NB09 — Bonus DishWipe: {winner_method} (2M steps, RTX 5090)",
             fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "cross_task_comparison.png", dpi=150)

# ── Step 10: MLflow ──
print(f"\nNB09 Bonus DishWipe PASSED ✅")
print(f"  Training: {elapsed/60:.1f} min")
print(f"  DishWipe reward: {dishwipe_summary['mean_reward']:.4f}")
print(f"  DishWipe success: {dishwipe_summary['success_rate']:.2%}")
```

---

## Key Assertions

- [ ] Winner method correctly loaded from NB08
- [ ] **2,000,000 training steps** (same as Apple, RTX 5090 budget)
- [ ] DishWipe Full-Body env registers and runs correctly
- [ ] **200-episode eval** with bootstrap CI (50K resamples)
- [ ] Video recorded: best + worst episodes as MP4
- [ ] Cross-task comparison table produced (Apple vs DishWipe)
- [ ] Checkpoints saved every 200K steps

---

## Notes

- **RTX 5090 Budget**: Full 2M steps + 200-episode eval (same quality as Apple)
- DishWipe Full-Body env adapted from upper-body version:
  - `unitree_g1` (37 DOF, free-floating root)
  - Added balance penalty to reward
  - `cleaned_ratio` metric from VirtualDirtGrid
- If Residual-SAC wins, BaseController is adapted for DishWipe task
- Cross-task comparison demonstrates method generalization ability
- Self-contained: can run independently with just NB08 artifacts
- Both tasks now evaluated with **identical statistical rigor** (200 eps, CI, etc.)

---

*Plan NB09 — RTX 5090 Edition — Updated March 2026*
