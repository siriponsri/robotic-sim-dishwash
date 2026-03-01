# NB02 - Grid Mapping Demo (10x10) with Custom Env

> **Status: Implemented & CPU-tested**

## Goal

Demonstrate the mapping between the robot's contact position and discrete cell
indices on the physical plate in `UnitreeG1DishWipe-v1`. The env v2 uses a
**force-weighted centroid** of multi-link contacts for grid mapping internally;
here we test the mapping functions independently.

## Steps (as implemented)

| Step | Purpose |
|------|---------|
| 1 | Create env, read plate position from physics |
| 2 | Test `VirtualDirtGrid.world_to_uv` and `uv_to_cell` with corners |
| 3 | Contact detection: multi-link `get_pairwise_contact_forces()` with threshold constants |
| 4 | Generate deterministic zig-zag path (100 waypoints) |
| 5 | Simulate zig-zag cleaning on grid, verify 100% coverage |
| 6 | Visualise grid before/after (matplotlib heatmaps) |
| 7 | Live env exploration: random actions, trace TCP + contact |
| 8 | Save artifacts (CSVs + PNGs + config JSON) |
| 9 | MLflow logging |

## Key Outputs
- `artifacts/NB02/grid_before.png`, `grid_after.png`
- `artifacts/NB02/grid_trace.csv` - cell-by-cell zig-zag trace
- `artifacts/NB02/env_exploration_trace.csv` - TCP/contact trace from live env

## Notes
- Env v2 uses 4 contact links (palm + 3 fingers) with force-weighted centroid
- Contact threshold: `CONTACT_THRESHOLD` from dishwipe_env.py
- Force limits: `FZ_SOFT` (penalty) and `FZ_HARD` (termination)
