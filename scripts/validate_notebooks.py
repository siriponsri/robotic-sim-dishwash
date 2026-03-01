"""Quick validation of NB02-NB05 core logic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gymnasium as gym

from src.envs.dishwipe_env import (
    UnitreeG1DishWipeEnv, PLATE_HALF_SIZE, CONTACT_THRESHOLD,
    W_CLEAN, W_REACH, W_CONTACT, W_TIME, W_JERK, W_ACT, W_FORCE,
    SUCCESS_BONUS, FZ_SOFT, FZ_HARD, SUCCESS_CLEAN_RATIO,
)
from src.envs.dirt_grid import VirtualDirtGrid

print("=" * 60)
print("Creating env (this loads meshes — may take 30-60s on CPU)...")
env = gym.make(
    "UnitreeG1DishWipe-v1",
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode=None,
    num_envs=1,
)
obs, info = env.reset(seed=42)
print(f"obs: {obs.shape}, act: {env.action_space.shape}")

# ── NB02: Grid mapping ──
print("\n" + "=" * 60)
print("NB02 — Grid Mapping")
plate_pos = env.unwrapped.plate.pose.p[0].cpu().numpy()
plate_half = np.array(PLATE_HALF_SIZE[:2])
tcp = env.unwrapped.agent.left_tcp.pose.p[0].cpu().numpy()
print(f"  Plate: {plate_pos}")
print(f"  TCP:   {tcp}")

grid = VirtualDirtGrid(H=10, W=10, brush_radius=1)
u, v = VirtualDirtGrid.world_to_uv(plate_pos, plate_pos, plate_half)
cell = grid.uv_to_cell(u, v)
print(f"  Center: uv=({u:.3f},{v:.3f}) -> cell={cell}")

grid.reset()
for row in range(10):
    v_n = (row + 0.5) / 10
    cols = range(10) if row % 2 == 0 else reversed(range(10))
    for col in cols:
        u_n = (col + 0.5) / 10
        ci, cj = grid.uv_to_cell(u_n, v_n)
        grid.mark_clean(ci, cj)
print(f"  Zigzag: {grid.get_cleaned_ratio()*100:.0f}%")
assert grid.get_cleaned_ratio() == 1.0
print("  ✅ NB02 PASSED")

# ── NB03: Dirt engine ──
print("\n" + "=" * 60)
print("NB03 — Dirt Engine")
g = VirtualDirtGrid(H=10, W=10, brush_radius=1)
g.reset()
assert g.get_cleaned_ratio() == 0.0
d1 = g.mark_clean(5, 5)
assert d1 == 9, f"Expected 9, got {d1}"
d2 = g.mark_clean(5, 5)
assert d2 == 0
d3 = g.mark_clean(0, 0)
assert d3 == 4, f"Expected 4 corner, got {d3}"
flat = g.get_grid_flat()
assert flat.shape == (100,) and flat.dtype == np.float32
for r in [0, 1, 2]:
    g2 = VirtualDirtGrid(H=10, W=10, brush_radius=r)
    g2.reset()
    d = g2.mark_clean(5, 5)
    expected = {0: 1, 1: 9, 2: 25}[r]
    assert d == expected, f"radius={r}: expected {expected}, got {d}"
    print(f"  radius={r}: delta={d} ✓")
print("  ✅ NB03 PASSED")

# ── NB04: Reward contract ──
print("\n" + "=" * 60)
print("NB04 — Reward Contract")
print(f"  W_CLEAN={W_CLEAN}, W_REACH={W_REACH}, W_CONTACT={W_CONTACT}")
print(f"  W_TIME={W_TIME}, W_JERK={W_JERK}, W_ACT={W_ACT}, W_FORCE={W_FORCE}")
print(f"  FZ_SOFT={FZ_SOFT}, FZ_HARD={FZ_HARD}")
obs, info = env.reset(seed=42)
obs, r, t, tr, info = env.step(env.action_space.sample())
for key in ["success", "fail", "contact_force", "delta_clean", "cleaned_ratio"]:
    assert key in info, f"Missing: {key}"
print(f"  reward={r.item():.4f}")
print(f"  All info keys present ✓")
print("  ✅ NB04 PASSED")

# ── NB05: Baselines smoke ──
print("\n" + "=" * 60)
print("NB05 — Baselines (smoke)")
obs, info = env.reset(seed=42)
# Random baseline - 10 steps
for i in range(10):
    obs, r, t, tr, info = env.step(env.action_space.sample())
    if t.any() or tr.any():
        obs, info = env.reset(seed=42+i)
print("  Random 10 steps OK")

# Heuristic - check TCP access
tcp = env.unwrapped.agent.left_tcp.pose.p[0].cpu().numpy()
plate = env.unwrapped.plate.pose.p[0].cpu().numpy()
delta = plate - tcp
print(f"  TCP-to-plate: {np.linalg.norm(delta):.4f} m")
print("  ✅ NB05 PASSED")

env.close()
print("\n" + "=" * 60)
print("ALL VALIDATIONS PASSED ✅")
print("=" * 60)
