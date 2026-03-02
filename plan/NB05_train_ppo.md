# NB05 — Train PPO (Apple Full-Body) — RTX 5090 Edition

> **Status: Not started**
> **Hardware: GPU (RTX 5090, 32 GB VRAM)**
> **Task: Apple (Full Body)**

---

## Goal

Train a PPO agent on `UnitreeG1PlaceAppleInBowlFullBody-v1` using
Stable-Baselines3 with **RTX 5090-optimized hyperparameters**:
- **2,000,000 env steps** (4× previous)
- **64 GPU-vectorized parallel environments** (ManiSkill GPU backend)
- **[512, 512] networks** (4× capacity)
- **Linear LR decay** (3e-4 → 1e-5)
- **Mixed precision** (torch.cuda.amp)
- **VecNormalize** observation normalization
- **Checkpointing** every 200K steps

Same budget as SAC (NB06) for fair comparison.

---

## Required Input

| Item | Source | Description |
|------|--------|-------------|
| NB01 completed | `artifacts/NB01/` | Envs verified, obs/act shapes known |
| NB04 completed | `artifacts/NB04/baseline_leaderboard.csv` | Baseline performance |
| `src/envs/apple_fullbody_env.py` | Source code | Apple Full-Body env |
| GPU available | RunPod / Local | **RTX 5090 (32 GB VRAM)** |

---

## Expected Output / Artifacts

| File | Path | Description |
|------|------|-------------|
| `ppo_apple.zip` | `artifacts/NB05/` | Trained PPO model (final) |
| `checkpoints/ppo_*.zip` | `artifacts/NB05/checkpoints/` | Checkpoints every 200K steps |
| `learning_curve.png` | `artifacts/NB05/` | Episode reward + entropy + value loss |
| `training_log.csv` | `artifacts/NB05/` | Per-episode reward/length/success |
| `eval_results.json` | `artifacts/NB05/` | Quick eval (20 episodes) after training |
| `gpu_utilization.json` | `artifacts/NB05/` | VRAM/GPU usage stats |
| `nb05_config.json` | `artifacts/NB05/` | Full config (hyperparameters + env config) |

---

## Resources (RTX 5090)

| Resource | Requirement |
|----------|-------------|
| CPU | 8+ cores |
| RAM | 40 GB (VecNormalize + multi-env obs) |
| GPU | **RTX 5090 (32 GB VRAM)** |
| VRAM Usage | ~8-12 GB (64 envs + policy network) |
| Disk | 15+ GB (checkpoints) |
| Runtime | **~1-2 hours** (2M steps, 64 envs) |

---

## Steps

| Step | Purpose |
|------|---------|
| 1 | Config: RTX 5090 hyperparameters, GPU detection, VRAM check |
| 2 | Create 64 GPU-vectorized envs (ManiSkill GPU backend) |
| 3 | Apply VecNormalize wrapper (obs normalization, clip=10) |
| 4 | Configure SB3 PPO: lr=3e-4→1e-5 decay, [512,512], batch=2048 |
| 5 | Define callbacks: TrainLog + Checkpoint + EarlyStopping |
| 6 | Train PPO for 2,000,000 env steps |
| 7 | Save final model + VecNormalize stats |
| 8 | Quick evaluation: 20 deterministic episodes |
| 9 | Plot learning curves (reward, entropy, value loss) |
| 10 | GPU utilization stats |
| 11 | Save artifacts + MLflow |

---

## RTX 5090 Hyperparameter Table

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `total_env_steps` | **2,000,000** | 4× previous, fills RTX 5090 budget |
| `n_envs` | **64** | GPU-vectorized ManiSkill backend |
| `n_steps` | **512** | Rollout per env (64×512 = 32,768 total) |
| `batch_size` | **2048** | Large batch for stable gradients |
| `n_epochs` | 10 | Standard PPO |
| `learning_rate` | **3e-4 → 1e-5** | Linear decay schedule |
| `gamma` | 0.99 | Standard discount |
| `gae_lambda` | 0.95 | Standard GAE |
| `clip_range` | 0.2 | Standard PPO clip |
| `ent_coef` | 0.005 | Exploration in high-dim (37 DOF) |
| `vf_coef` | 0.5 | Value function coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `net_arch` | **[512, 512]** | 4× capacity (~790K params) |
| `activation_fn` | **ReLU** | Better gradient flow than Tanh |
| `normalize_advantage` | True | Stabilizes training |
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path(".").resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB05"
CKPT_DIR = ARTIFACTS_DIR / "checkpoints"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

IS_GPU = torch.cuda.is_available()

# GPU info
if IS_GPU:
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")
else:
    print("WARNING: No GPU detected, using CPU fallback")

CFG = {
    "seed": 42,
    "env_id": "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "control_mode": "pd_joint_delta_pos",
    "obs_mode": "state",
    # ── RTX 5090 Optimized ──
    "total_env_steps": 2_000_000 if IS_GPU else 20_000,
    "n_envs": 64 if IS_GPU else 1,
    "sim_backend": "gpu" if IS_GPU else "cpu",
    "learning_rate_start": 3e-4,
    "learning_rate_end": 1e-5,
    "n_steps": 512,
    "batch_size": 2048 if IS_GPU else 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch": [512, 512],
    "activation_fn": "ReLU",
    "normalize_advantage": True,
    "use_vec_normalize": True,
    "clip_obs": 10.0,
    "clip_reward": 10.0,
    "checkpoint_freq": 200_000,
    "eval_episodes": 20,
}

with open(ARTIFACTS_DIR / "nb05_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

# ── Step 2: Create 64 GPU-Vectorized Envs ──
if IS_GPU and CFG["n_envs"] > 1:
    # ManiSkill GPU-vectorized backend
    env = gym.make(
        CFG["env_id"],
        num_envs=CFG["n_envs"],
        sim_backend=CFG["sim_backend"],
        obs_mode=CFG["obs_mode"],
        control_mode=CFG["control_mode"],
        render_mode="rgb_array",
    )
    # ManiSkillVectorEnv is already vectorized
else:
    env = gym.make(
        CFG["env_id"],
        num_envs=1,
        obs_mode=CFG["obs_mode"],
        control_mode=CFG["control_mode"],
        render_mode="rgb_array",
    )
    env = CPUGymWrapper(env)

print(f"Env created: {CFG['n_envs']} envs, obs={env.observation_space.shape}, "
      f"act={env.action_space.shape}")

# ── Step 3: VecNormalize ──
if CFG["use_vec_normalize"]:
    env = VecNormalize(env, norm_obs=True, norm_reward=True,
                       clip_obs=CFG["clip_obs"], clip_reward=CFG["clip_reward"])
    print("VecNormalize applied (obs + reward)")

# ── Step 4: Configure PPO with LR Decay ──
def linear_schedule(initial_value, final_value=1e-5):
    """Linear decay from initial_value to final_value."""
    def func(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return func

lr_schedule = linear_schedule(CFG["learning_rate_start"], CFG["learning_rate_end"])

model = PPO(
    "MlpPolicy",
    env,
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
    normalize_advantage=CFG["normalize_advantage"],
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

# ── Step 5: Callbacks ──
class TrainLogCallback(BaseCallback):
    """Log per-episode stats + GPU memory."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.gpu_mem_log = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.episode_successes.append(float(info.get("success", False)))
        # Log GPU memory every 10K steps
        if self.num_timesteps % 10_000 == 0 and torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            self.gpu_mem_log.append({"step": self.num_timesteps, "vram_gb": mem})
        return True

train_cb = TrainLogCallback()

checkpoint_cb = CheckpointCallback(
    save_freq=max(CFG["checkpoint_freq"] // CFG["n_envs"], 1),
    save_path=str(CKPT_DIR),
    name_prefix="ppo_apple",
    save_vecnormalize=True,
)

# ── Step 6: Train ──
print(f"\n{'='*60}")
print(f"  Training PPO — {CFG['total_env_steps']:,} steps")
print(f"  n_envs={CFG['n_envs']}, batch={CFG['batch_size']}")
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

# ── Step 7: Save Model + VecNormalize ──
model.save(str(ARTIFACTS_DIR / "ppo_apple"))
if CFG["use_vec_normalize"]:
    env.save(str(ARTIFACTS_DIR / "vec_normalize.pkl"))
print(f"Model saved to {ARTIFACTS_DIR / 'ppo_apple.zip'}")

# ── Step 8: Quick Eval ──
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

# ── Step 9: Learning Curves ──
import matplotlib.pyplot as plt

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

# Success rate over time (rolling window)
successes = train_cb.episode_successes
if successes and len(successes) >= window:
    rolling_succ = np.convolve(successes, np.ones(window)/window, mode="valid")
    axes[1, 0].plot(range(window-1, len(successes)), rolling_succ,
                    color="darkorange", linewidth=2)
    axes[1, 0].set_title(f"Success Rate (rolling {window})")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Success Rate")
    axes[1, 0].set_ylim(-0.05, 1.05)
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

fig.suptitle(f"NB05 — PPO Training on Apple Full-Body\n"
             f"({CFG['total_env_steps']:,} steps, {CFG['n_envs']} envs, "
             f"net={CFG['net_arch']}, RTX 5090)",
             fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "learning_curve.png", dpi=150)

# ── Step 10: GPU Stats ──
gpu_stats = {
    "gpu_name": torch.cuda.get_device_name(0) if IS_GPU else "CPU",
    "vram_total_gb": float(torch.cuda.get_device_properties(0).total_mem / 1e9) if IS_GPU else 0,
    "vram_peak_gb": float(torch.cuda.max_memory_allocated() / 1e9) if IS_GPU else 0,
    "training_time_s": elapsed,
    "steps_per_sec": CFG["total_env_steps"] / elapsed,
    "env_steps_total": CFG["total_env_steps"],
}
with open(ARTIFACTS_DIR / "gpu_utilization.json", "w") as f:
    json.dump(gpu_stats, f, indent=2)

# ── Step 11: Training Log + MLflow ──
import pandas as pd
training_log = pd.DataFrame({
    "episode": range(len(ep_rewards)),
    "reward": ep_rewards,
    "length": ep_lengths[:len(ep_rewards)],
    "success": successes[:len(ep_rewards)],
})
training_log.to_csv(ARTIFACTS_DIR / "training_log.csv", index=False)

env.close()
print(f"\nNB05 PPO Training PASSED ✅")
print(f"  Total time: {elapsed/60:.1f} min")
print(f"  Steps/sec: {CFG['total_env_steps']/elapsed:.0f}")
print(f"  Eval mean_reward: {eval_summary['mean_reward']:.4f}")
```

---

## Key Assertions

- [ ] Model trains without errors for 2,000,000 steps on RTX 5090
- [ ] 64 GPU-vectorized environments run simultaneously
- [ ] LR decays from 3e-4 → 1e-5 linearly
- [ ] VecNormalize stats saved alongside model
- [ ] Checkpoints saved every 200K steps (10 total)
- [ ] Learning curve shows 4 subplots: reward, length, success rate, VRAM
- [ ] Quick eval runs 20 episodes with deterministic actions
- [ ] Config matches NB06 (SAC) for fairness (net_arch, total_steps, etc.)

---

## Notes

- **RTX 5090 Advantage**: 64 parallel envs via ManiSkill GPU backend = ~16× faster data collection
- **Fairness**: TOTAL_ENV_STEPS, net_arch, activation_fn must be identical to NB06/NB07
- **VecNormalize**: Normalizes observations (mean=0, std=1) and clips rewards — critical for PPO stability
- **LR Decay**: Prevents overshooting near convergence; linear schedule is robust
- **Checkpointing**: Every 200K steps = 10 checkpoints for recovery/analysis
- PPO is on-policy → benefits massively from parallel envs (64 envs = 32K new samples per rollout!)
- On CPU mode (20K steps), results will be poor — just for testing pipeline
- Mixed precision (FP16) can be added via `torch.cuda.amp` if needed for even more speed

---

*Plan NB05 — RTX 5090 Edition — Updated March 2026*
