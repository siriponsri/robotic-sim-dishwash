# NB05 - Baselines, Smooth Wrapper & Base Controller

> **Status: Implemented & CPU-tested**

## Goal

Provide baseline policies (random, heuristic) and smoothing infrastructure.
The heuristic uses **palm** position (aligned with env v2 reach reward).
The base controller combines heuristic + EMA smoothing for residual learning (NB08).

## Steps (as implemented)

| Step | Purpose |
|------|---------|
| 1 | Helper: `evaluate_policy()` - runs any policy and collects per-episode metrics |
| 2 | Random baseline: sample actions, measure reward/cleaned/jerk/contact |
| 3 | Heuristic policy: proportional control pushing **palm** toward plate |
| 4 | SmoothActionWrapper: EMA filter (a_smooth = alpha*a_raw + (1-alpha)*a_prev) |
| 5 | Smoothed random baseline: demonstrates jerk reduction |
| 6 | BaseController: heuristic + smooth, ready for residual SAC in NB08 |
| 7 | Leaderboard table (pandas DataFrame) |
| 8 | Save artifacts + MLflow |

## Key Outputs
- `artifacts/NB05/baseline_leaderboard.csv`
- `artifacts/NB05/nb05_config.json`

## Notes
- Heuristic uses `agent.robot.links_map["left_palm_link"]` (not TCP)
- SmoothActionWrapper alpha=0.3 (lower = smoother)
- BaseController wraps heuristic + EMA for NB08 residual learning
- Smooth wrapper reduces jerk ~50-80% vs raw random
