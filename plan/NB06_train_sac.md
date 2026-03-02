# NB06 — Train SAC (Apple Full-Body) — RTX 5090 Edition

> **Status: Not started**
> **Hardware: GPU (RTX 5090, 32 GB VRAM)**
> **Task: Apple (Full Body)**

---

## Goal

Train a SAC agent on `UnitreeG1PlaceAppleInBowlFullBody-v1` with **RTX 5090-optimized hyperparameters**:
- **2,000,000 env steps** (same budget as PPO in NB05)
- **10,000,000 replay buffer** (10× previous, fits in 40 GB RAM)
- **[512, 512] networks** (4× capacity)
- **1024 batch size** (4× previous)
- **2 gradient steps** per env step
- **Linear LR decay** (3e-4 → 1e-5)
- **Checkpointing** every 200K steps
- Automatic entropy tuning (`ent_coef="auto"`)

Off-policy learning with massive replay buffer. Same budget as PPO for fair comparison.

---

## Required Input

| Item | Source | Description |
|------|--------|-------------|
| NB01 completed | `artifacts/NB01/` | Envs verified |
| NB04 completed | `artifacts/NB04/baseline_leaderboard.csv` | Baseline reference |
| `src/envs/apple_fullbody_env.py` | Source code | Apple Full-Body env |
| GPU available | RunPod / Local | **RTX 5090 (32 GB VRAM)** |

---

## Expected Output / Artifacts

| File | Path | Description |
|------|------|-------------|
| `sac_apple.zip` | `artifacts/NB06/` | Trained SAC model (final) |
| `checkpoints/sac_*.zip` | `artifacts/NB06/checkpoints/` | Checkpoints every 200K steps |
| `learning_curve.png` | `artifacts/NB06/` | Reward + entropy + Q-loss curves |
| `training_log.csv` | `artifacts/NB06/` | Per-episode stats |
| `eval_results.json` | `artifacts/NB06/` | Quick eval (20 episodes) |
| `gpu_utilization.json` | `artifacts/NB06/` | VRAM/GPU usage stats |
| `nb06_config.json` | `artifacts/NB06/` | Full config |

---

## Resources (RTX 5090)

| Resource | Requirement |
|----------|-------------|
| CPU | 8+ cores |
| RAM | **40 GB** (10M replay buffer ≈ 5-8 GB in RAM) |
| GPU | **RTX 5090 (32 GB VRAM)** |
| VRAM Usage | ~6-10 GB (policy + Q-networks + target networks) |
| Disk | 15+ GB (checkpoints) |
| Runtime | **~2-4 hours** (2M steps, single env + 2 gradient steps) |

---

## Steps

| Step | Purpose |
|------|---------|
| 1 | Config: RTX 5090 hyperparameters, GPU detection, VRAM check |
| 2 | Create env with CPUGymWrapper (SAC uses 1 env, off-policy) |
| 3 | Configure SB3 SAC: 10M buffer, 1024 batch, lr decay, [512,512] |
| 4 | Define callbacks: TrainLog + Checkpoint |
| 5 | Train SAC for 2,000,000 env steps |
| 6 | Save final model |
| 7 | Quick evaluation: 20 deterministic episodes |
| 8 | Plot learning curves (reward, entropy, Q-loss) |
| 9 | GPU utilization stats |
| 10 | Save artifacts + MLflow |

---

## RTX 5090 Hyperparameter Table

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `total_env_steps` | **2,000,000** | Same budget as PPO (fairness) |
| `n_envs` | 1 | Off-policy, single env sufficient |
| `buffer_size` | **10,000,000** | ~5-8 GB in RAM, fits in 40 GB |
| `batch_size` | **1024** | Large batch, GPU-friendly |
| `tau` | 0.005 | Soft target update |
| `gamma` | 0.99 | Standard discount |
| `ent_coef` | "auto" | Auto-tuned (SAC automatic entropy) |
| `target_entropy` | "auto" | -dim(A) = -37 |
| `learning_starts` | **10,000** | Fill buffer before training |
| `train_freq` | 1 | Update every step |
| `gradient_steps` | **2** | 2 gradient updates per env step |
| `learning_rate` | **3e-4 → 1e-5** | Linear decay |
| `net_arch` | **[512, 512]** | Same capacity as PPO (fairness) |
| `activation_fn` | **ReLU** | Better gradient flow |
| `checkpoint_freq` | **200,000** | Save every 200K steps |

---

## Main Code (Pseudocode)

```python
# ── Step 1: Config + GPU Check ──
import gymnasium as gym
import numpy as np
import torch
import json
import time
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path(".").resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB06"
CKPT_DIR = ARTIFACTS_DIR / "checkpoints"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

IS_GPU = torch.cuda.is_available()

if IS_GPU:
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")

CFG = {
    "seed": 42,
    "env_id": "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "control_mode": "pd_joint_delta_pos",
    "obs_mode": "state",
    # ── RTX 5090 Optimized ──
    "total_env_steps": 2_000_000 if IS_GPU else 20_000,
    "n_envs": 1,
    "learning_rate_start": 3e-4,
    "learning_rate_end": 1e-5,
    "buffer_size": 10_000_000 if IS_GPU else 50_000,
    "batch_size": 1024 if IS_GPU else 64,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto",
    "target_entropy": "auto",
    "learning_starts": 10_000 if IS_GPU else 500,
    "train_freq": 1,
    "gradient_steps": 2 if IS_GPU else 1,
    "net_arch": [512, 512],
    "activation_fn": "ReLU",
    "checkpoint_freq": 200_000,
    "eval_episodes": 20,
}

with open(ARTIFACTS_DIR / "nb06_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

# ── Step 2: Create Env ──
env = gym.make(
    CFG["env_id"],
    num_envs=1,
    obs_mode=CFG["obs_mode"],
    control_mode=CFG["control_mode"],
    render_mode="rgb_array",
)
env = CPUGymWrapper(env)
print(f"Env created: obs={env.observation_space.shape}, act={env.action_space.shape}")

# ── Step 3: Configure SAC with LR Decay ──
def linear_schedule(initial_value, final_value=1e-5):
    def func(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return func

lr_schedule = linear_schedule(CFG["learning_rate_start"], CFG["learning_rate_end"])

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
    seed=CFG["seed"],
    device="auto",
)

total_params = sum(p.numel() for p in model.policy.parameters())
print(f"Policy parameters: {total_params:,}")
print(f"Buffer capacity: {CFG['buffer_size']:,} transitions")
est_buffer_gb = CFG["buffer_size"] * 500 / 1e9  # rough estimate: 500 bytes per transition
print(f"Est. buffer RAM: ~{est_buffer_gb:.1f} GB")

# ── Step 4: Callbacks ──
class TrainLogCallback(BaseCallback):
    """Log per-episode stats + GPU memory + entropy."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.gpu_mem_log = []
        self.entropy_log = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.episode_successes.append(float(info.get("success", False)))
        # GPU memory
        if self.num_timesteps % 10_000 == 0 and torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            self.gpu_mem_log.append({"step": self.num_timesteps, "vram_gb": mem})
        # Log entropy coefficient
        if self.num_timesteps % 10_000 == 0 and hasattr(self.model, "ent_coef_tensor"):
            ent = float(self.model.ent_coef_tensor.exp().item())
            self.entropy_log.append({"step": self.num_timesteps, "ent_coef": ent})
        return True

train_cb = TrainLogCallback()

checkpoint_cb = CheckpointCallback(
    save_freq=CFG["checkpoint_freq"],
    save_path=str(CKPT_DIR),
    name_prefix="sac_apple",
)

# ── Step 5: Train ──
print(f"\n{'='*60}")
print(f"  Training SAC — {CFG['total_env_steps']:,} steps")
print(f"  buffer={CFG['buffer_size']:,}, batch={CFG['batch_size']}")
print(f"  gradient_steps={CFG['gradient_steps']}")
print(f"  net_arch={CFG['net_arch']}, activation=ReLU")
print(f"  LR: {CFG['learning_rate_start']} → {CFG['learning_rate_end']}")
print(f"  device={'RTX 5090 GPU' if IS_GPU else 'CPU'}")
print(f"{'='*60}")

start_time = time.time()
model.learn(
    total_timesteps=CFG["total_env_steps"],
    callback=[train_cb, checkpoint_cb],
    progress_bar=True,
)
elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Steps/sec: {CFG['total_env_steps'] / elapsed:.0f}")

# ── Step 6: Save ──
model.save(str(ARTIFACTS_DIR / "sac_apple"))
print(f"Model saved to {ARTIFACTS_DIR / 'sac_apple.zip'}")

# ── Step 7: Quick Eval ──
eval_env = gym.make(CFG["env_id"], num_envs=1, obs_mode="state",
                    control_mode=CFG["control_mode"], render_mode="rgb_array")
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
        "episode": ep,
        "total_reward": ep_reward,
        "steps": ep_steps,
        "success": info.get("success", False),
    })

eval_summary = {
    "mean_reward": float(np.mean([r["total_reward"] for r in eval_results])),
    "std_reward": float(np.std([r["total_reward"] for r in eval_results])),
    "success_rate": float(np.mean([r["success"] for r in eval_results])),
    "mean_steps": float(np.mean([r["steps"] for r in eval_results])),
    "training_time_s": elapsed,
    "steps_per_sec": CFG["total_env_steps"] / elapsed,
    "total_steps": CFG["total_env_steps"],
    "gpu": torch.cuda.get_device_name(0) if IS_GPU else "CPU",
}
with open(ARTIFACTS_DIR / "eval_results.json", "w") as f:
    json.dump(eval_summary, f, indent=2)
eval_env.close()

# ── Step 8: Learning Curves ──
import matplotlib.pyplot as plt
import pandas as pd

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ep_rewards = train_cb.episode_rewards
if ep_rewards:
    axes[0, 0].plot(ep_rewards, alpha=0.2, color="steelblue", linewidth=0.5)
    window = min(50, len(ep_rewards))
    if len(ep_rewards) >= window:
        rolling = np.convolve(ep_rewards, np.ones(window)/window, mode="valid")
        axes[0, 0].plot(range(window-1, len(ep_rewards)), rolling,
                        color="darkblue", linewidth=2, label=f"Rolling avg ({window})")
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

ep_lengths = train_cb.episode_lengths
if ep_lengths:
    axes[0, 1].plot(ep_lengths, alpha=0.2, color="seagreen", linewidth=0.5)
    if len(ep_lengths) >= window:
        rolling_len = np.convolve(ep_lengths, np.ones(window)/window, mode="valid")
        axes[0, 1].plot(range(window-1, len(ep_lengths)), rolling_len,
                        color="darkgreen", linewidth=2)
    axes[0, 1].set_title("Episode Lengths")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].grid(True, alpha=0.3)

# Entropy coefficient over time
if train_cb.entropy_log:
    ent_steps = [e["step"] for e in train_cb.entropy_log]
    ent_vals = [e["ent_coef"] for e in train_cb.entropy_log]
    axes[1, 0].plot(ent_steps, ent_vals, color="darkorange", linewidth=2)
    axes[1, 0].set_title("Entropy Coefficient (auto-tuned)")
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel("α (entropy coef)")
    axes[1, 0].grid(True, alpha=0.3)

# GPU Memory
if train_cb.gpu_mem_log:
    steps = [x["step"] for x in train_cb.gpu_mem_log]
    vrams = [x["vram_gb"] for x in train_cb.gpu_mem_log]
    axes[1, 1].plot(steps, vrams, color="purple", linewidth=2)
    axes[1, 1].set_title("GPU VRAM Usage")
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel("VRAM (GB)")
    axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(f"NB06 — SAC Training on Apple Full-Body\n"
             f"({CFG['total_env_steps']:,} steps, buffer={CFG['buffer_size']:,}, "
             f"net={CFG['net_arch']}, RTX 5090)",
             fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "learning_curve.png", dpi=150)

# ── Step 9: GPU Stats ──
gpu_stats = {
    "gpu_name": torch.cuda.get_device_name(0) if IS_GPU else "CPU",
    "vram_total_gb": float(torch.cuda.get_device_properties(0).total_mem / 1e9) if IS_GPU else 0,
    "vram_peak_gb": float(torch.cuda.max_memory_allocated() / 1e9) if IS_GPU else 0,
    "training_time_s": elapsed,
    "steps_per_sec": CFG["total_env_steps"] / elapsed,
    "buffer_size": CFG["buffer_size"],
    "gradient_steps": CFG["gradient_steps"],
}
with open(ARTIFACTS_DIR / "gpu_utilization.json", "w") as f:
    json.dump(gpu_stats, f, indent=2)

# ── Step 10: Training Log + MLflow ──
training_log = pd.DataFrame({
    "episode": range(len(ep_rewards)),
    "reward": ep_rewards,
    "length": ep_lengths[:len(ep_rewards)],
    "success": train_cb.episode_successes[:len(ep_rewards)],
})
training_log.to_csv(ARTIFACTS_DIR / "training_log.csv", index=False)

env.close()
print(f"\nNB06 SAC Training PASSED ✅")
print(f"  Total time: {elapsed/60:.1f} min")
print(f"  Steps/sec: {CFG['total_env_steps']/elapsed:.0f}")
print(f"  Eval mean_reward: {eval_summary['mean_reward']:.4f}")
```

---

## Key Assertions

- [ ] TOTAL_ENV_STEPS = 2,000,000 (identical to NB05 PPO, fairness)
- [ ] net_arch [512, 512] (identical to NB05 PPO, fairness)
- [ ] Buffer size 10M fits in 40 GB RAM
- [ ] 2 gradient steps per env step utilized
- [ ] Auto-entropy tuning converges (entropy coefficient logged)
- [ ] Checkpoints saved every 200K steps
- [ ] Model saved as `sac_apple.zip`

---

## Notes

- **RTX 5090 Advantage**: 10M buffer + 1024 batch + 2 gradient steps = much better sample efficiency
- **Fairness**: TOTAL_ENV_STEPS, net_arch, LR schedule identical to NB05 PPO
- SAC is off-policy → single env + replay buffer (no need for vectorized envs)
- `ent_coef="auto"` — SAC automatically tunes entropy temperature for optimal exploration
- `learning_starts=10000` — fill buffer with 10K random transitions before training begins
- `gradient_steps=2` — do 2 gradient updates per env step (more compute per sample)
- 10M buffer ≈ 5-8 GB RAM (depends on obs/act size) — well within 40 GB RAM
- SAC deterministic eval uses the **mean** of the policy distribution

---

*Plan NB06 — RTX 5090 Edition — Updated March 2026*
