# NB07 - Train SAC (Off-Policy)

> **Status: Ready for GPU (RunPod)**

## Goal

Train a SAC agent on `UnitreeG1DishWipe-v1` with automatic entropy tuning.
Off-policy learning with replay buffer for sample efficiency.

## Steps (as implemented)

| Step | Purpose |
|------|---------|
| 1 | Create env with CPUGymWrapper (SAC uses 1 env, off-policy) |
| 2 | Configure SAC: lr=3e-4, buffer=1M, batch=256, ent_coef="auto" |
| 3 | Train for TOTAL_ENV_STEPS (same as PPO for fair comparison) |
| 4 | Save model + evaluate (10 eps, deterministic=True for mean action) |
| 5 | Learning curve plot |
| 6 | Save artifacts + MLflow |

## Key Config
- `total_env_steps`: 500,000 (GPU) / 20,000 (CPU)
- `n_envs`: 1 (off-policy, single env)
- `buffer_size`: 1,000,000 (GPU) / 50,000 (CPU)
- Network: MLP [256, 256]
- SAC: tau=0.005, gamma=0.99, ent_coef="auto", learning_starts=1000

## Key Outputs
- `artifacts/NB07/sac_model.zip`
- `artifacts/NB07/learning_curve.png`
- `artifacts/NB07/eval_results.json`

## Notes
- Same TOTAL_ENV_STEPS as NB06 PPO for fair comparison
- Automatic entropy tuning (target_entropy="auto")
- Deterministic eval uses mean of policy distribution
