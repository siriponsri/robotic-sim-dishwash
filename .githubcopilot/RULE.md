# RULE.md — Jupyter Notebook Standards (Unitree G1 • ManiSkill • RL)

## 0) Purpose
This document defines strict rules for creating Jupyter notebooks so that:
- Every notebook is reproducible on another machine
- Results are comparable across algorithms/seeds
- Debugging is fast (clear logs, clear artifacts)
- Nothing sensitive is leaked (tokens/passwords)

---

## 1) Naming & Structure

### 1.1 Notebook naming
Use this pattern:
- `NB01_setup_smoke.ipynb`
- `NB02_grid_mapping.ipynb`
- …
- `NB09_evaluation.ipynb`

### 1.2 Folder layout (recommended)
```

project_root/
notebooks/
NB01_setup_smoke.ipynb
...
src/
envs/
wrappers/
utils/
artifacts/
NB01/
NB02/
...
configs/
base.yaml
ppo.yaml
sac.yaml
reports/
.env
RULE.md


---

## 2) Required Compute/Resources Table (MANDATORY)

> Every notebook MUST include this table under `## Environment`.
> If a notebook can run on either CPU or GPU, state the recommended option.

| Notebook | Goal (1 line) | Required HW | Recommended HW | Min CPU | Min RAM | GPU VRAM | Notes |
|---|---|---|---|---:|---:|---:|---|
| NB01 | Setup + smoke test env | CPU | CPU | 2 cores | 4 GB | 0 GB | GPU optional for quick check |
| NB02 | Validate grid mapping | CPU | CPU | 2 cores | 4 GB | 0 GB | Pure math/plot |
| NB03 | Dirt engine + brush viz | CPU | CPU | 2 cores | 4 GB | 0 GB | Pure numpy/plot |
| NB04 | Reward/safety contract + logging | CPU | CPU | 2 cores | 4 GB | 0 GB | No training |
| NB05 | Baselines + smoothing | CPU/GPU | GPU (if heavy env) | 4 cores | 8 GB | 8 GB | If sim is slow, use GPU |
| NB06 | Train PPO | GPU (practical) | GPU | 8 cores | 16 GB | 12–24 GB | PPO on many envs benefits GPU |
| NB07 | Train SAC | GPU (practical) | GPU | 8 cores | 16 GB | 12–24 GB | Replay + nets compute heavy |
| NB08 | Residual SAC + ablation | GPU | GPU | 8 cores | 16 GB | 12–24 GB | Multiple runs β, needs VRAM |
| NB09 | Eval + CI + videos | CPU/GPU | GPU (for faster sim) | 4 cores | 8–16 GB | 8–16 GB | Rendering/video may be heavy |

**Rules for this table**
- **Required HW** = what is realistically needed to finish within class time.
- **Recommended HW** = best option for stable runs and faster iteration.
- If you use **vectorized env**, specify `N_ENVS` expectation in Notes.
- If you will run on cloud (RunPod/Colab), say so in Notes.

---

## 3) Mandatory Sections in Every Notebook (Order is fixed)

Every notebook must have these Markdown sections **in order**:
1. `# NBxx — <Title>`
2. `## Objective`
3. `## Environment`  ← must include the resource table row for this NB
4. `## Imports`
5. `## Config`
6. `## Reproducibility`
7. `## Implementation Steps`
8. `## Results`
9. `## Artifacts`
10. `## Notes / Troubleshooting`
11. `## References`

---

## 4) Cell Rules

### 4.1 Cell sizing
- One responsibility per cell
- Each step should be **1–3 code cells max**
- Avoid giant cells; break into small, testable chunks

### 4.2 No hidden state dependence
Notebook must run top-to-bottom without reordering.
- Don’t rely on variables created “somewhere else”
- If a cell depends on previous outputs, state it in Markdown

---

## 5) Environment & Versions (must be printed)

### 5.1 Version print (mandatory)
In `## Environment`, print:
- Python version
- OS
- CUDA availability + GPU name (if any)
- Key library versions: `gymnasium`, `mani_skill`, `stable-baselines3`, `torch`, `numpy`

### 5.2 Hardware assumption (must state)
Explicitly state:
- CPU/GPU
- `N_ENVS` (if vectorized)
- Approximate expected runtime (rough level: short/medium/long)

---

## 6) Config Policy (single source of truth)

### 6.1 Central config
Every notebook must define a single `config` dict (or load YAML).
No scattered hyperparams.

### 6.2 Config minimum fields
- `seed`
- `env_id`
- `robot` (UnitreeG1)
- `control_mode` (joint-control)
- `n_envs`
- `total_env_steps`
- `eval_episodes`
- reward weights: `w_clean, w_time, w_jerk, w_act, w_force, success_bonus`
- safety limits: `fz_soft, fz_hard, collision_limit, timeout_steps`
- algorithm params (ppo/sac)

### 6.3 No magic numbers
Any number affecting results must live in config.

---

## 7) Reproducibility Rules

### 7.1 Seeds (mandatory)
Set seeds for:
- Python `random`
- NumPy
- Torch
- Environment reset seed

### 7.2 Deterministic evaluation
- PPO: deterministic
- SAC: deterministic/mean action

### 7.3 Fixed budgets for comparison
When comparing algorithms:
- Same `TOTAL_ENV_STEPS`
- Same `EVAL_EPISODES`
- Same `SEEDS`

---

## 8) Logging Rules

### 8.1 Minimum console logs
Print:
- Config summary
- Start/end milestones
- End metrics summary

### 8.2 MLflow
- Never hardcode secrets
- Load from `.env`
- Log params/metrics/artifacts consistently

---

## 9) Artifact Contract (must be saved)

### 9.1 Artifact directory
Each notebook writes to:
`artifacts/NBxx/`

### 9.2 Required artifacts (by NB)
- NB01: `env_spec.json`, optional `smoke_video.mp4`
- NB02: `grid_trace.csv`, `grid_before.png`, `grid_after.png`
- NB03: `dirt_engine_test.png`, `brush_effect_demo.png`
- NB04: `reward_contract.md` (or `.json`)
- NB05: `baseline_leaderboard.csv`, `baseline_videos/`
- NB06/NB07/NB08: `model.zip`, `learning_curve.png`, `train_log.csv`
- NB09: `eval_table.csv`, `ci_table.csv`, `final_plots/`, `videos/`

---

## 10) Visualization Standards
- Label axes/units/title
- Use consistent metric names:
  - `cleaned_ratio`, `steps_to_95`, `mean_jerk`, `p95_jerk`,
  - `fz_mean`, `fz_p95`, `terminate_reason`

---

## 11) Safety Rules (robotics)
- Safety termination ON by default
- Don’t disable force/collision limits to inflate reward
- Report safety failure rates in NB09

---

## 12) Clean Code Rules
- Use small helper functions, avoid repeated blocks
- Avoid complex class hierarchies in notebooks
- Readable names (no heavy abbreviations)
- Assertions for shapes/ranges
- Handle expected failures gracefully (e.g., missing GPU)

---

## 13) References (mandatory)
End every notebook with `## References` linking:
- papers/docs/repos used
- algorithm docs (SB3 PPO/SAC)
- ManiSkill environment docs
- Residual policy learning resources if used

---

## 14) Quality Gate Checklist (must pass)
- [ ] Runs top-to-bottom
- [ ] Resource table present + accurate
- [ ] Config printed and centralized
- [ ] Seeds set
- [ ] Artifacts saved under `artifacts/NBxx/`
- [ ] Evaluation deterministic
- [ ] References present
```
