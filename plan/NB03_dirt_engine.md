# NB03 - Dirt Engine (Grid Update + Brush) and Visualisation

> **Status: Implemented & CPU-tested**

## Goal

Test the `VirtualDirtGrid` module within the live `UnitreeG1DishWipe-v1` env.
Verify brush radius, cleanup progress, reset behaviour, and coordinate mapping.

## Steps (as implemented)

| Step | Purpose |
|------|---------|
| 1 | Version check |
| 2 | Imports + VirtualDirtGrid + UnitreeG1DishWipeEnv |
| 3 | Config (seed, grid 10x10, brush_radius=1) |
| 4 | Reproducibility (seeds) |
| 5 | Standalone dirt grid tests (mark_clean, get_cleaned_ratio, reset) |
| 6 | Test brush radius effect (radius 0 vs 1 vs 2) |
| 7 | Test coordinate mapping with live env (world_to_uv, uv_to_cell) |
| 8 | Run env steps and track dirt grid progress |
| 9 | Cleaning progress curve plot |
| 10 | Save artifacts and config |
| 11 | MLflow logging |

## Key Outputs
- `artifacts/NB03/brush_radius_effect.json`
- `artifacts/NB03/cleaning_progress.csv`
- `artifacts/NB03/cleaning_progress.png`

## Notes
- `VirtualDirtGrid` lives in `src/envs/dirt_grid.py` (~145 lines)
- Integrated into env: `env.unwrapped._dirt_grids[0]`
- `info["cleaned_ratio"]` exposed at each step
- Brush radius=1: cleans 3x3 area per contact
