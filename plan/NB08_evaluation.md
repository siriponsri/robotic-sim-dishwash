# NB08 — Evaluation: Compare Methods & Declare Winner — RTX 5090 Edition

> **Status: Not started**
> **Hardware: CPU or GPU**
> **Task: Apple (Full Body)**

---

## Goal

Evaluate all trained agents (PPO, SAC, Residual-SAC-best-β) on **200
deterministic episodes** each. Compute:
- **95% bootstrap CI** (50,000 resamples)
- **Welch's t-test** (pairwise statistical significance)
- **Cohen's d** (effect size magnitude)
- **Action distribution analysis** (mean, std, entropy)
- **Video recording** (best/worst episodes as MP4)

Produce comprehensive comparison tables and plots. **Declare the winner**
for the DishWipe bonus round (NB09).

---

## Required Input

| Item | Source | Description |
|------|--------|-------------|
| PPO model | `artifacts/NB05/ppo_apple.zip` | Trained PPO |
| SAC model | `artifacts/NB06/sac_apple.zip` | Trained SAC |
| Residual-SAC best | `artifacts/NB07/best_beta.json` | Best-β model path |
| Baseline stats | `artifacts/NB04/baseline_leaderboard.csv` | References |
| Env source | `src/envs/apple_fullbody_env.py` | Apple env |

---

## Expected Output / Artifacts

| File | Path | Description |
|------|------|-------------|
| `eval_200ep.csv` | `artifacts/NB08/` | Per-episode results (all methods) |
| `comparison_table.csv` | `artifacts/NB08/` | Summary stats with CI + p-values |
| `comparison_plot.png` | `artifacts/NB08/` | Bar chart with 95% CI |
| `success_rate_plot.png` | `artifacts/NB08/` | Success rate comparison |
| `reward_distribution.png` | `artifacts/NB08/` | Violin/box plot |
| `pairwise_tests.csv` | `artifacts/NB08/` | Welch's t-test + Cohen's d |
| `action_analysis.png` | `artifacts/NB08/` | Action distribution per method |
| `videos/` | `artifacts/NB08/videos/` | MP4 best/worst per method |
| `best_method.json` | `artifacts/NB08/` | Winner declaration |
| `nb08_config.json` | `artifacts/NB08/` | Config |

---

## Resources

| Resource | Requirement |
|----------|-------------|
| CPU | 8+ cores |
| RAM | 16+ GB |
| GPU | Optional (can eval on CPU, GPU faster) |
| Disk | 10 GB (videos) |
| Runtime | ~1-2 hours (600 episodes + video) |

---

## Steps

| Step | Purpose |
|------|---------|
| 1 | Config: **eval_episodes=200**, bootstrap_n=50,000, methods list |
| 2 | Load all 3 trained models |
| 3 | Evaluate each: **200 deterministic episodes**, collect rich metrics |
| 4 | **Bootstrap 95% CI** (50K resamples) for reward and success rate |
| 5 | **Welch's t-test** — pairwise significance between methods |
| 6 | **Cohen's d** — effect size magnitude |
| 7 | **Action distribution analysis** |
| 8 | Comparison DataFrame + winner declaration |
| 9 | **Video recording** — best/worst episode per method |
| 10 | Rich plotting suite (6 plots) |
| 11 | Save artifacts + MLflow |

---

## Statistical Analysis Suite (RTX 5090 Edition)

| Analysis | Method | Value |
|----------|--------|-------|
| Confidence Interval | Bootstrap percentile | **50,000 resamples**, 95% CI |
| Significance Test | **Welch's t-test** | Unequal variance, pairwise |
| Effect Size | **Cohen's d** | Small/Medium/Large classification |
| Multiple Comparisons | Bonferroni correction | α' = 0.05/3 = 0.0167 |
| Episodes | Deterministic rollouts | **200** per method |
| Total Episodes | | **600** (3 methods × 200) |

---

## Main Code (Pseudocode)

```python
# ── Step 1: Config ──
import gymnasium as gym
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from stable_baselines3 import PPO, SAC
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

PROJECT_ROOT = Path(".").resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "NB08"
VIDEO_DIR = ARTIFACTS_DIR / "videos"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

from src.envs import UnitreeG1PlaceAppleInBowlFullBodyEnv

CFG = {
    "env_id": "UnitreeG1PlaceAppleInBowlFullBody-v1",
    "control_mode": "pd_joint_delta_pos",
    "obs_mode": "state",
    # ── RTX 5090 Extended Evaluation ──
    "eval_episodes": 200,
    "max_steps_per_ep": 1000,
    "bootstrap_n": 50_000,
    "ci_level": 0.95,
    "significance_alpha": 0.05,
    "bonferroni_comparisons": 3,
}
CFG["bonferroni_alpha"] = CFG["significance_alpha"] / CFG["bonferroni_comparisons"]

with open(ARTIFACTS_DIR / "nb08_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

print(f"Evaluation config:")
print(f"  Episodes per method: {CFG['eval_episodes']}")
print(f"  Bootstrap resamples: {CFG['bootstrap_n']:,}")
print(f"  Bonferroni α': {CFG['bonferroni_alpha']:.4f}")

# ── Step 2: Load Models ──
with open(PROJECT_ROOT / "artifacts" / "NB07" / "best_beta.json") as f:
    best_beta_info = json.load(f)

models = {
    "PPO": PPO.load(str(PROJECT_ROOT / "artifacts" / "NB05" / "ppo_apple")),
    "SAC": SAC.load(str(PROJECT_ROOT / "artifacts" / "NB06" / "sac_apple")),
    "Residual-SAC": SAC.load(str(PROJECT_ROOT / "artifacts" / "NB07" /
                                  best_beta_info["best_model"].replace(".zip", ""))),
}
print(f"Loaded: {list(models.keys())}")
print(f"Residual-SAC best β = {best_beta_info['best_beta']:.2f}")

# ── Step 3: Evaluate 200 Episodes Each ──
def evaluate_model(model, env, n_episodes, max_steps):
    """Rich evaluation: reward, success, steps, action stats."""
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        ep_reward, ep_steps = 0.0, 0
        all_actions = []
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            if terminated or truncated:
                break
        actions_arr = np.array(all_actions)
        results.append({
            "episode": ep,
            "total_reward": ep_reward,
            "steps": ep_steps,
            "success": bool(info.get("success", False)),
            "action_mean": float(np.mean(np.abs(actions_arr))),
            "action_std": float(np.std(actions_arr)),
            "action_max": float(np.max(np.abs(actions_arr))),
        })
    return results

all_results = {}
for method_name, model in models.items():
    print(f"\nEvaluating {method_name} ({CFG['eval_episodes']} episodes)...")

    if method_name == "Residual-SAC":
        raw_env = gym.make(CFG["env_id"], num_envs=1, obs_mode=CFG["obs_mode"],
                           control_mode=CFG["control_mode"], render_mode="rgb_array")
        raw_env = CPUGymWrapper(raw_env)
        base_ctrl = BaseController(raw_env, alpha=0.3)
        env = ResidualActionWrapper(raw_env, base_ctrl,
                                     beta=best_beta_info["best_beta"])
    else:
        env = gym.make(CFG["env_id"], num_envs=1, obs_mode=CFG["obs_mode"],
                       control_mode=CFG["control_mode"], render_mode="rgb_array")
        env = CPUGymWrapper(env)

    results = evaluate_model(model, env, CFG["eval_episodes"], CFG["max_steps_per_ep"])
    all_results[method_name] = results
    env.close()

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

# ── Step 4: Bootstrap CI ──
def bootstrap_ci(data, stat_fn=np.mean, n_bootstrap=50000, ci=0.95):
    boot_stats = []
    data = np.array(data)
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_fn(sample))
    lower = np.percentile(boot_stats, (1-ci)/2 * 100)
    upper = np.percentile(boot_stats, (1+ci)/2 * 100)
    return float(lower), float(upper)

# ── Step 5: Welch's t-test (pairwise) ──
def welch_ttest(data1, data2):
    """Two-sided Welch's t-test for unequal variance."""
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    return float(t_stat), float(p_value)

# ── Step 6: Cohen's d ──
def cohens_d(data1, data2):
    """Effect size (pooled std)."""
    n1, n2 = len(data1), len(data2)
    s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(data1) - np.mean(data2)) / pooled_std)

def d_magnitude(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

# Build comparison + pairwise tables
comparison = []
for method, results in all_results.items():
    rewards = [r["total_reward"] for r in results]
    successes = [float(r["success"]) for r in results]
    rew_lo, rew_hi = bootstrap_ci(rewards, np.mean, CFG["bootstrap_n"], CFG["ci_level"])
    suc_lo, suc_hi = bootstrap_ci(successes, np.mean, CFG["bootstrap_n"], CFG["ci_level"])
    comparison.append({
        "method": method,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "ci95_reward_lo": rew_lo,
        "ci95_reward_hi": rew_hi,
        "success_rate": float(np.mean(successes)),
        "ci95_success_lo": suc_lo,
        "ci95_success_hi": suc_hi,
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_action_std": float(np.mean([r["action_std"] for r in results])),
    })

comp_df = pd.DataFrame(comparison)
comp_df.to_csv(ARTIFACTS_DIR / "comparison_table.csv", index=False)

# Pairwise tests
method_names = list(all_results.keys())
pairwise_rows = []
for i in range(len(method_names)):
    for j in range(i+1, len(method_names)):
        m1, m2 = method_names[i], method_names[j]
        r1 = [r["total_reward"] for r in all_results[m1]]
        r2 = [r["total_reward"] for r in all_results[m2]]
        t_stat, p_val = welch_ttest(r1, r2)
        d = cohens_d(r1, r2)
        pairwise_rows.append({
            "method_1": m1,
            "method_2": m2,
            "t_statistic": t_stat,
            "p_value": p_val,
            "significant_bonferroni": p_val < CFG["bonferroni_alpha"],
            "cohens_d": d,
            "effect_magnitude": d_magnitude(d),
        })

pairwise_df = pd.DataFrame(pairwise_rows)
pairwise_df.to_csv(ARTIFACTS_DIR / "pairwise_tests.csv", index=False)

print("\nComparison Table:")
print(comp_df.to_string(index=False))
print("\nPairwise Statistical Tests:")
print(pairwise_df.to_string(index=False))

# ── Step 8: Declare Winner ──
best_idx = comp_df["mean_reward"].idxmax()
winner = comp_df.loc[best_idx]
best_method = {
    "winner": winner["method"],
    "mean_reward": float(winner["mean_reward"]),
    "ci95": [float(winner["ci95_reward_lo"]), float(winner["ci95_reward_hi"])],
    "success_rate": float(winner["success_rate"]),
    "eval_episodes": CFG["eval_episodes"],
    "bootstrap_n": CFG["bootstrap_n"],
    "reason": (f"Highest mean reward ({winner['mean_reward']:.4f}) "
               f"over {CFG['eval_episodes']} deterministic episodes "
               f"with 95% CI [{winner['ci95_reward_lo']:.4f}, {winner['ci95_reward_hi']:.4f}]"),
}
with open(ARTIFACTS_DIR / "best_method.json", "w") as f:
    json.dump(best_method, f, indent=2)
print(f"\n🏆 WINNER: {best_method['winner']} — reward={best_method['mean_reward']:.4f}")

# ── Step 9: Video Recording ──
# Record best and worst episode for each method as MP4
import imageio

for method_name, model in models.items():
    # Find best/worst episode indices
    method_results = all_results[method_name]
    rewards = [r["total_reward"] for r in method_results]
    best_ep = int(np.argmax(rewards))
    worst_ep = int(np.argmin(rewards))

    for tag, seed in [("best", best_ep), ("worst", worst_ep)]:
        env = gym.make(CFG["env_id"], num_envs=1, obs_mode=CFG["obs_mode"],
                       control_mode=CFG["control_mode"], render_mode="rgb_array")
        env = CPUGymWrapper(env)

        if method_name == "Residual-SAC":
            base_ctrl = BaseController(env, alpha=0.3)
            env = ResidualActionWrapper(env, base_ctrl,
                                         beta=best_beta_info["best_beta"])

        frames = []
        obs, info = env.reset(seed=seed)
        for step in range(CFG["max_steps_per_ep"]):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        if frames:
            video_path = VIDEO_DIR / f"{method_name.lower().replace('-','_')}_{tag}.mp4"
            imageio.mimsave(str(video_path), frames, fps=30)
            print(f"  Saved {video_path.name}")
        env.close()

# ── Step 10: Rich Plotting Suite ──
methods = comp_df["method"].tolist()
colors = {"PPO": "#FFB74D", "SAC": "#64B5F6", "Residual-SAC": "#81C784"}

# Plot 1: Mean Reward with 95% CI
fig, ax = plt.subplots(figsize=(10, 6))
means = comp_df["mean_reward"].tolist()
ci_lo = comp_df["ci95_reward_lo"].tolist()
ci_hi = comp_df["ci95_reward_hi"].tolist()
errs = [[m-lo for m,lo in zip(means, ci_lo)],
        [hi-m for m,hi in zip(means, ci_hi)]]
ax.bar(methods, means, yerr=errs, capsize=8,
       color=[colors[m] for m in methods], edgecolor="black", alpha=0.9)
ax.set_title(f"Mean Reward ({CFG['eval_episodes']} episodes, 95% CI, {CFG['bootstrap_n']:,} bootstrap)",
             fontweight="bold")
ax.set_ylabel("Mean Reward")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "comparison_plot.png", dpi=150)

# Plot 2: Success Rate
fig, ax = plt.subplots(figsize=(10, 6))
succ = comp_df["success_rate"].tolist()
succ_lo = comp_df["ci95_success_lo"].tolist()
succ_hi = comp_df["ci95_success_hi"].tolist()
errs_s = [[s-lo for s,lo in zip(succ, succ_lo)],
          [hi-s for s,hi in zip(succ, succ_hi)]]
ax.bar(methods, succ, yerr=errs_s, capsize=8,
       color=[colors[m] for m in methods], edgecolor="black", alpha=0.9)
ax.set_title(f"Success Rate ({CFG['eval_episodes']} episodes, 95% CI)", fontweight="bold")
ax.set_ylabel("Success Rate")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "success_rate_plot.png", dpi=150)

# Plot 3: Reward Distribution (Violin)
fig, ax = plt.subplots(figsize=(10, 6))
data_violin = [[r["total_reward"] for r in all_results[m]] for m in methods]
parts = ax.violinplot(data_violin, positions=range(len(methods)),
                       showmeans=True, showmedians=True)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods)
ax.set_title(f"Reward Distribution ({CFG['eval_episodes']} episodes)", fontweight="bold")
ax.set_ylabel("Total Reward")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "reward_distribution.png", dpi=150)

# Plot 4: Action Distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, method in enumerate(methods):
    act_stds = [r["action_std"] for r in all_results[method]]
    axes[i].hist(act_stds, bins=30, color=colors[method], edgecolor="black", alpha=0.8)
    axes[i].set_title(f"{method}: Action Std Distribution")
    axes[i].set_xlabel("Action Std")
fig.suptitle("Action Distribution Analysis", fontweight="bold")
fig.tight_layout()
fig.savefig(ARTIFACTS_DIR / "action_analysis.png", dpi=150)

# ── Step 11: MLflow ──
print(f"\nNB08 Evaluation PASSED ✅")
print(f"  Total episodes: {CFG['eval_episodes'] * len(models)}")
print(f"  Winner: {best_method['winner']}")
```

---

## Key Assertions

- [ ] All 3 methods evaluated on same 200 seeds (ep=0..199)
- [ ] `deterministic=True` for all predictions
- [ ] Bootstrap CI with **50,000 resamples** (highly precise)
- [ ] Welch's t-test with **Bonferroni correction** (α'=0.0167)
- [ ] Cohen's d computed for all pairwise comparisons
- [ ] Videos recorded: 6 total (best + worst × 3 methods)
- [ ] `best_method.json` consumed by NB09

---

## Notes

- **200 episodes** provides very high statistical power (error ∝ 1/√N)
- **Welch's t-test**: Robust to unequal variances (preferred over Student's t)
- **Cohen's d thresholds**: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)
- **Bonferroni correction**: With 3 pairwise comparisons, α' = 0.05/3 ≈ 0.0167
- If CI intervals overlap, check p-value; if p > α', difference is not significant
- **Video recording**: Best/worst episodes help explain agent behavior visually
- **Action analysis**: Reveals if methods explore differently (e.g., SAC may have higher action entropy)
- Winner used in NB09: retrain on DishWipe Full-Body with same config

---

*Plan NB08 — RTX 5090 Edition — Updated March 2026*
