# NB06 - Train PPO (On-Policy)

> **Status: Ready for GPU (RunPod)**

## Goal

Train a PPO agent on `UnitreeG1DishWipe-v1` using Stable-Baselines3.
Uses CPUGymWrapper for SB3 compatibility. Designed for RunPod GPU (4090+).

## Steps (as implemented)

| Step | Purpose |
|------|---------|
| 1 | Create ManiSkill env with CPUGymWrapper (1 env) or ManiSkillVectorEnv (multi-env GPU) |
| 2 | Configure PPO: lr=3e-4, n_steps=2048, batch_size=256, net_arch=[256,256], Tanh |
| 3 | Train for TOTAL_ENV_STEPS (500K GPU / 20K CPU) with TrainLogCallback |
| 4 | Save model + evaluate (10 eps, deterministic) |
| 5 | Learning curve plot |
| 6 | Save artifacts + MLflow |

## Key Config
- `total_env_steps`: 500,000 (GPU) / 20,000 (CPU)
- `n_envs`: 4 (GPU) / 1 (CPU)
- Network: MLP [256, 256] with Tanh
- PPO: clip=0.2, ent_coef=0.01, gamma=0.99, gae_lambda=0.95

## Key Outputs
- `artifacts/NB06/ppo_model.zip`
- `artifacts/NB06/learning_curve.png`
- `artifacts/NB06/eval_results.json`

## Notes
- Uses `make_env()` helper that selects CPUGymWrapper vs ManiSkillVectorEnv
- Same TOTAL_ENV_STEPS as NB07 for fair comparison
- Obs ~168D, Act 25D (pd_joint_delta_pos)
