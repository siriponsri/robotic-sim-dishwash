"""VirtualDirtGrid – tracks dirty/clean state of an H×W virtual plate grid.

This module is used by the custom DishWipe environment to track cleaning
progress.  The grid lives in CPU/numpy space; the environment converts
positions to grid cells when contact is detected.

Grid values:
    0 = dirty (default)
    1 = clean

Usage
-----
>>> grid = VirtualDirtGrid(H=10, W=10, brush_radius=1)
>>> grid.reset()
>>> delta = grid.mark_clean(4, 5)   # clean cell (4,5) + 3×3 neighbourhood
>>> print(grid.get_cleaned_ratio())  # fraction cleaned
"""

from __future__ import annotations

import numpy as np


class VirtualDirtGrid:
    """Tracks dirty/clean state of an H×W virtual plate grid.

    Parameters
    ----------
    H : int
        Number of rows.
    W : int
        Number of columns.
    brush_radius : int
        Radius of the cleaning brush.  radius=0 cleans only the target cell;
        radius=1 cleans a 3×3 neighbourhood; radius=2 cleans 5×5; etc.
    """

    def __init__(self, H: int = 10, W: int = 10, brush_radius: int = 1) -> None:
        assert H > 0 and W > 0, f"Grid dims must be positive, got H={H}, W={W}"
        assert brush_radius >= 0, f"brush_radius must be >= 0, got {brush_radius}"
        self.H = H
        self.W = W
        self.brush_radius = brush_radius
        self.grid = np.zeros((H, W), dtype=np.int8)
        self._total_cells = H * W

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the grid to all-dirty (0).  Returns a *copy* of the grid."""
        self.grid[:] = 0
        return self.grid.copy()

    def mark_clean(self, i: int, j: int) -> int:
        """Mark cell (i, j) and its brush neighbourhood as clean.

        Returns
        -------
        delta_clean : int
            Number of *newly* cleaned cells (0 if all were already clean).
        """
        assert 0 <= i < self.H, f"Row {i} out of bounds [0, {self.H})"
        assert 0 <= j < self.W, f"Col {j} out of bounds [0, {self.W})"
        r = self.brush_radius
        i_min = max(0, i - r)
        i_max = min(self.H, i + r + 1)
        j_min = max(0, j - r)
        j_max = min(self.W, j + r + 1)
        region = self.grid[i_min:i_max, j_min:j_max]
        newly_cleaned = int(np.sum(region == 0))
        self.grid[i_min:i_max, j_min:j_max] = 1
        return newly_cleaned

    def get_cleaned_ratio(self) -> float:
        """Return fraction of cells that are clean (0.0–1.0)."""
        return float(np.sum(self.grid == 1)) / self._total_cells

    def get_grid(self) -> np.ndarray:
        """Return a *copy* of the current grid."""
        return self.grid.copy()

    def get_grid_flat(self) -> np.ndarray:
        """Return a flat copy of the grid as float32 (for observations)."""
        return self.grid.astype(np.float32).flatten()

    # ------------------------------------------------------------------
    # Coordinate mapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def world_to_uv(
        xyz: np.ndarray,
        plate_center: np.ndarray,
        plate_half_size: np.ndarray,
    ) -> tuple[float, float]:
        """Convert world (x, y, z) → normalised plate coords (u, v) ∈ [0, 1].

        Parameters
        ----------
        xyz : (3,) array
            World position of the contact point / TCP.
        plate_center : (3,) array
            Center of the plate in world coordinates.
        plate_half_size : (2,) array
            Half extents of the plate surface [half_x, half_y].
        """
        x, y = float(xyz[0]), float(xyz[1])
        cx, cy = float(plate_center[0]), float(plate_center[1])
        hx, hy = float(plate_half_size[0]), float(plate_half_size[1])
        u = (x - (cx - hx)) / (2 * hx) if hx > 0 else 0.5
        v = (y - (cy - hy)) / (2 * hy) if hy > 0 else 0.5
        return (
            float(np.clip(u, 0.0, 1.0 - 1e-9)),
            float(np.clip(v, 0.0, 1.0 - 1e-9)),
        )

    def uv_to_cell(self, u: float, v: float) -> tuple[int, int]:
        """Convert (u, v) ∈ [0, 1)² → grid cell indices (i, j)."""
        i = int(np.clip(int(np.floor(v * self.H)), 0, self.H - 1))
        j = int(np.clip(int(np.floor(u * self.W)), 0, self.W - 1))
        return (i, j)

    def world_to_cell(
        self,
        xyz: np.ndarray,
        plate_center: np.ndarray,
        plate_half_size: np.ndarray,
    ) -> tuple[int, int]:
        """Convert world (x, y, z) directly → grid cell (i, j)."""
        u, v = self.world_to_uv(xyz, plate_center, plate_half_size)
        return self.uv_to_cell(u, v)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ratio = self.get_cleaned_ratio()
        return (
            f"VirtualDirtGrid(H={self.H}, W={self.W}, "
            f"brush_radius={self.brush_radius}, cleaned={ratio:.1%})"
        )
