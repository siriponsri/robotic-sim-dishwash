# NB04 - Reward Contract, Safety Validation & MLflow Utilities

> **Status: Implemented & CPU-tested**

## Goal

Document the full reward contract (9 terms including r_sweep), validate reward
values and safety termination with test episodes, and export reusable MLflow
helper utilities for NB05-NB09.

## Steps (as implemented)

| Step | Purpose |
|------|---------|
| 1 | Version check |
| 2 | Imports + all weight constants (W_CLEAN, W_REACH, W_CONTACT, W_SWEEP, W_TIME, W_JERK, W_ACT, W_FORCE, SUCCESS_BONUS) |
| 3 | Config (5 test episodes, 100 steps each) |
| 4 | Document reward contract v2.0 as JSON with all 9 terms |
| 5 | Run test episodes: validate reward is dense and bounded |
| 6 | Validate reward range (negative with random actions due to penalties) |
| 7 | Safety validation: verify info dict has all expected keys |
| 8 | Define MLflow helpers: `setup_mlflow()`, `log_training_run()` |
| 9 | Log to MLflow |

## Reward Contract v2.0 (9 terms)

| Term | Weight | Formula | Sign |
|------|--------|---------|------|
| r_clean | W_CLEAN=10.0 | w * delta_clean | + |
| r_reach | W_REACH=0.5 | w * (1 - tanh(5 * dist(palm, plate))) | + |
| r_contact | W_CONTACT=1.0 | w * is_contacting | + |
| r_sweep | W_SWEEP=0.3 | w * lateral_movement_while_contacting | + |
| r_time | W_TIME=0.01 | -w per step | - |
| r_jerk | W_JERK=0.05 | -w * jerk_squared | - |
| r_act | W_ACT=0.005 | -w * action_norm_squared | - |
| r_force | W_FORCE=0.01 | -w * excess_force | - |
| r_success | 50.0 | one-shot at 95% clean | + |

## Key Outputs
- `artifacts/NB04/reward_contract.json`

## Notes
- r_reach uses **palm** position (not TCP)
- r_sweep encourages lateral centroid movement while in contact
- Safety: FZ_HARD=200N terminates, FZ_SOFT=50N starts penalty
- Info keys validated: success, fail, contact_force, delta_clean, cleaned_ratio
