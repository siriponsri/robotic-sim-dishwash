# NB09 - Evaluation, Bootstrap Confidence Intervals, Videos & Summary

> **Status: Ready for CPU/GPU**

## Goal

Rigorous evaluation of all methods (Random, Heuristic, PPO, SAC, Residual SAC)
on `UnitreeG1DishWipe-v1`. Bootstrap CIs, comparison plots, video generation,
and final MLflow summary.

## Steps (as implemented)

| Step | Purpose |
|------|---------|
| 1 | Load all trained models (PPO from NB06, SAC from NB07, best Residual SAC from NB08) |
| 2 | Define `evaluate_model()` - handles SB3 models, function policies, and residual |
| 3 | Evaluate all methods: Random (20 eps), Heuristic (20 eps), trained models (100 eps) |
| 4 | Bootstrap 95% CIs (1000 bootstrap samples) for reward, cleaned, jerk, contact |
| 5 | Comparison bar plots (reward, cleaning, jerk across methods) |
| 6 | Render video of best method (optional, may fail on CPU-only) |
| 7 | MLflow final summary run |

## Key Config
- `eval_episodes`: 100 (trained models), 20 (baselines)
- `bootstrap_samples`: 1000
- `ci_level`: 0.95
- `max_steps_per_ep`: 1000

## Key Outputs
- `artifacts/NB09/eval_table.csv` - full evaluation with CIs
- `artifacts/NB09/eval_comparison.png` - bar chart comparison
- `artifacts/NB09/best_method_video.mp4` (optional)

## Notes
- Heuristic uses `left_palm_link` position (consistent with NB05/NB08)
- Residual eval: loads best beta from ablation table, applies inline BaseController
- Deterministic actions for all trained models (`predict(obs, deterministic=True)`)
- Video generation may fail on CPU-only (no Vulkan) - handled gracefully
