# NB01 - Setup, Register Custom Environment & Smoke Test

> **Status: Implemented & CPU-tested**

## Goal

Register `UnitreeG1DishWipe-v1`, verify obs (~168D), act (25D), TCP/palm links,
multi-link contact forces, dirt grid integration, and log results to MLflow.

## Steps (as implemented)

| Step | Purpose |
|------|---------|
| 1 | Version check (Python, NumPy, PyTorch, ManiSkill, SB3) |
| 2 | Imports + PROJECT_ROOT setup + register custom env |
| 3 | Create env, inspect obs/act spaces, save `env_spec.json` |
| 4 | Discover robot joints (25 DOF upper body), save `active_joints.json` |
| 5 | Render test (rgb_array), handle CPU-only gracefully |
| 6 | Check TCP & palm links (left_tcp, left_palm_link) |
| 7 | Check multi-link contact force API (palm + 3 fingers vs plate) |
| 8 | Test dirt grid integration (VirtualDirtGrid 10x10) |
| 9 | Smoke test: 50 random steps, collect reward/contact/cleaned stats |
| 10 | Reproducibility: multi-reset consistency |
| 11 | MLflow logging |

## Key Outputs
- `artifacts/NB01/env_spec.json` - obs shape, act shape, control mode
- `artifacts/NB01/active_joints.json` - joint names and limits
- `artifacts/NB01/tcp_info.json` - TCP and palm positions
- `artifacts/NB01/contact_force_notes.json` - contact API info
- `artifacts/NB01/smoke_results.json` - 50-step smoke statistics

## Notes
- Obs ~168 dims: qpos(25) + qvel(25) + TCP(3) + palm(3) + plate(3) + palm_to_plate(3) + contact(1) + cleaned_ratio(1) + dirt_grid(100) + extras
- Contact: multi-link (left_palm_link + left_two_link + left_four_link + left_six_link)
- Reach reward uses **palm** position, not TCP
