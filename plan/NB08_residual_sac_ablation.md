# NB08 - Residual SAC + Beta Ablation

> **Status: Ready for GPU (RunPod)**

## Goal

Combine the **base controller** from NB05 (palm-guided heuristic + EMA smoothing)
with a learned **residual policy** via SAC. Train variants with different beta
scaling factors and select the best.

## Steps (as implemented)

| Step | Purpose |
|------|---------|
| 1 | Define `heuristic_policy()` (palm-guided) + `BaseController` (heuristic + EMA) |
| 2 | Define `ResidualActionWrapper`: a_final = clip(a_base + beta * a_residual) |
| 3 | Train Residual SAC for each beta in {0.25, 0.5, 1.0} |
| 4 | Ablation table (pandas DataFrame) |
| 5 | Visualise ablation (bar charts: reward, cleaned by beta) |
| 6 | MLflow logging |

## Key Config
- `beta_values`: [0.25, 0.5, 1.0]
- `smooth_alpha`: 0.3
- Same SAC hyperparameters as NB07
- Same TOTAL_ENV_STEPS

## Key Outputs
- `artifacts/NB08/residual_sac_beta*.zip` - models for each beta
- `artifacts/NB08/ablation_beta_table.csv`
- `artifacts/NB08/ablation_plot.png`

## Notes
- Heuristic uses `left_palm_link` position (aligned with env v2 reach reward)
- ResidualActionWrapper clips final action to env bounds
- Best beta selected by highest mean_reward for NB09 evaluation
- Residual Policy Learning: Silver et al. (2018)
