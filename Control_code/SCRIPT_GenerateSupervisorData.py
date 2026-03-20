"""
SCRIPT_GenerateSupervisorData.py
=================================
Generates skeleton data for arena shapes.

Pipeline
--------
1. Load arena wall point cloud for each session.
2. Build a candidate position grid.
3. Filter: inside arena (sector test) + minimum wall clearance.
4. Keep only the largest connected component (removes unreachable zones such as
   positions inside enclosed obstacles).
5. Compute the desired-band mask and its morphological skeleton.

Outputs
-------
- SupervisorData/skeleton_sessionBXX.npy            (K, 2) float32: x, y skeleton points
- SupervisorData/plot_valid_space.png            diagnostic figure (one panel / session)
"""

import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion as _erode, binary_dilation as _dilate, gaussian_filter as _gaussian_filter
from skimage.morphology import skeletonize as _skeletonize

sys.path.insert(0, os.path.dirname(__file__))
from Library.DataProcessor import read_wall_mask, mask2coordinates
from Library.GeometryUtils import (
    inside_arena_mask,
    largest_component_mask,
    min_wall_distances,
)

DATA_DIR = "Data"


def load_arena(session_name: str) -> dict:
    """
    Fast arena loader — reads only meta.json + arena_annotated.png from the
    env directory, without loading any of the 500 dill sample files.

    Returns a plain dict with keys:
        walls         : (N, 2) float32 array in mm
        arena_min_x/y : float  (mm)
        arena_max_x/y : float  (mm)
    """
    env_dirs = sorted(glob.glob(os.path.join(DATA_DIR, session_name, "env_*")))
    if not env_dirs:
        raise FileNotFoundError(f"No env_* directory found for session '{session_name}'")
    env_dir = env_dirs[0]

    with open(os.path.join(env_dir, "meta.json")) as f:
        meta = json.load(f)

    wall_mask = read_wall_mask(os.path.join(env_dir, "arena_annotated.png"))
    wall_x, wall_y = mask2coordinates(wall_mask, meta)
    walls = np.column_stack([wall_x, wall_y]).astype(np.float32)

    bounds = meta["arena_bounds_mm"]
    return dict(
        walls=walls,
        arena_min_x=float(bounds["min_x"]),
        arena_max_x=float(bounds["max_x"]),
        arena_min_y=float(bounds["min_y"]),
        arena_max_y=float(bounds["max_y"]),
    )

# ── Configuration ──────────────────────────────────────────────────────────────

SESSIONS      = ["sessionB01", "sessionB02", "sessionB03", "sessionB04", "sessionB05"]
OUTPUT_DIR    = "SupervisorData"
GRID_STEP_MM  = 50    # position grid resolution (mm)
WALL_MARGIN_MM = 150  # min distance robot centre → any wall point (mm)

# Desired operating band for the supervisor (mm from nearest wall).
# Too close → collision risk; too far → walls out of sonar range.
DESIRED_MIN_MM = 300   # inner boundary of desired band
DESIRED_MAX_MM = 700   # outer boundary of desired band

# Morphological opening radius (grid cells): removes narrow dead-end passages.
# Any corridor thinner than 2 × OPEN_RADIUS_CELLS × GRID_STEP_MM gets pruned.
OPEN_RADIUS_CELLS  = 1

# Morphological closing radius (grid cells): fills small gaps and smooths
# concave corners left by the opening step.
CLOSE_RADIUS_CELLS = 2

# Gaussian smoothing sigma (grid cells) applied to the raw distance surface
# before band thresholding. Smooths noisy wall point clouds so the desired
# band and skeleton are less ragged. Set to 0 to disable.
DIST_SMOOTH_SIGMA_CELLS = 2

# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_valid_positions(arena: dict,
                            wall_margin_mm: float = WALL_MARGIN_MM,
                            grid_step_mm: float   = GRID_STEP_MM,
                            ) -> tuple:
    """
    Return grid data for all reachable positions.

    Returns
    -------
    xs_grid, ys_grid : 1-D coordinate axes of the grid
    dist_surface     : (n_rows, n_cols) float array, NaN where invalid
    n_valid          : number of valid positions
    """
    walls = arena["walls"]

    xs_grid = np.arange(arena["arena_min_x"], arena["arena_max_x"] + grid_step_mm, grid_step_mm)
    ys_grid = np.arange(arena["arena_min_y"], arena["arena_max_y"] + grid_step_mm, grid_step_mm)
    gx, gy  = np.meshgrid(xs_grid, ys_grid)
    shape   = gx.shape

    if len(walls) > 0:
        # Downsample walls for the O(M×N) sector test — every 10th point is
        # sufficient to cover all 8 angular sectors from any interior position.
        walls_sparse = walls[::10]
        inside   = inside_arena_mask(walls_sparse, gx.ravel(), gy.ravel(), n_sectors=8)
        min_d    = min_wall_distances(walls, gx.ravel(), gy.ravel())
        pos_mask = inside & (min_d >= wall_margin_mm)
        pos_mask = largest_component_mask(pos_mask, shape)
    else:
        min_d    = np.full(gx.size, np.inf)
        pos_mask = np.ones(gx.size, dtype=bool)

    dist_surface = np.where(pos_mask, min_d, np.nan).reshape(shape)

    # NaN-safe Gaussian smoothing via normalised convolution: blur numerator
    # (NaN→0) and denominator (valid mask) separately, then divide.
    if DIST_SMOOTH_SIGMA_CELLS > 0:
        valid = np.isfinite(dist_surface).astype(float)
        filled = np.where(valid, dist_surface, 0.0)
        blurred_num   = _gaussian_filter(filled, sigma=DIST_SMOOTH_SIGMA_CELLS)
        blurred_denom = _gaussian_filter(valid,  sigma=DIST_SMOOTH_SIGMA_CELLS)
        with np.errstate(invalid="ignore", divide="ignore"):
            dist_smooth = np.where(blurred_denom > 0, blurred_num / blurred_denom, np.nan)
    else:
        dist_smooth = dist_surface

    desired_raw = (
        np.isfinite(dist_smooth) &
        (dist_smooth >= DESIRED_MIN_MM) &
        (dist_smooth <= DESIRED_MAX_MM)
    )
    # Open (remove narrow passages) then close (fill small gaps, smooth concave corners)
    r_open = OPEN_RADIUS_CELLS
    cy, cx = np.ogrid[-r_open:r_open+1, -r_open:r_open+1]
    struct_open = (cx**2 + cy**2) <= r_open**2
    opened = _dilate(_erode(desired_raw, struct_open), struct_open)

    r_close = CLOSE_RADIUS_CELLS
    cy, cx = np.ogrid[-r_close:r_close+1, -r_close:r_close+1]
    struct_close = (cx**2 + cy**2) <= r_close**2
    desired_open = _erode(_dilate(opened, struct_close), struct_close)

    return xs_grid, ys_grid, dist_surface, dist_smooth, desired_open, int(pos_mask.sum())


def compute_skeleton(desired_open: np.ndarray,
                     xs_grid: np.ndarray,
                     ys_grid: np.ndarray) -> np.ndarray:
    """
    Morphological skeleton of the desired-band mask.
    Returns (K, 2) float array of (x, y) skeleton point coordinates in mm.
    """
    skel = _skeletonize(desired_open)          # bool array, same shape
    rows, cols = np.where(skel)
    xs = xs_grid[cols]
    ys = ys_grid[rows]
    return np.column_stack([xs, ys]).astype(np.float32)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_valid_space(session_results: dict, out_path: str):
    """
    session_results : {session: {"arena", "xs_grid", "ys_grid", "dist_surface", "desired_open", "skeleton", "n_valid"}}
    """
    n      = len(session_results)
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.5 * n_cols, 6.5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    fig.suptitle(
        f"Valid (reachable) floor space\n"
        f"wall margin={WALL_MARGIN_MM} mm  |  grid {GRID_STEP_MM} mm  |  "
        f"desired band {DESIRED_MIN_MM}–{DESIRED_MAX_MM} mm  |  "
        f"open={OPEN_RADIUS_CELLS} cells  |  close={CLOSE_RADIUS_CELLS} cells  |  "
        f"smooth σ={DIST_SMOOTH_SIGMA_CELLS} cells",
        fontsize=10,
    )

    for idx, (session, data) in enumerate(session_results.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        walls        = data["arena"]["walls"]
        xs_grid      = data["xs_grid"]
        ys_grid      = data["ys_grid"]
        dist_surface = data["dist_surface"]   # raw distances, NaN where invalid
        dist_smooth  = data["dist_smooth"]    # smoothed distances

        # pcolormesh needs cell-edge coordinates; shift by half a cell
        half = (xs_grid[1] - xs_grid[0]) / 2
        x_edges = np.append(xs_grid - half, xs_grid[-1] + half)
        y_edges = np.append(ys_grid - half, ys_grid[-1] + half)

        # Full reachable floor — greyed out
        ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(dist_surface),
                      cmap="Greys", vmin=0, vmax=3000,
                      shading="flat", zorder=1, alpha=0.3)

        # Desired band — coloured with smoothed distance values
        desired_open = data["desired_open"]
        desired_dist = np.where(desired_open, dist_smooth, np.nan)
        pm = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(desired_dist),
                           cmap="viridis", vmin=DESIRED_MIN_MM, vmax=DESIRED_MAX_MM,
                           shading="flat", zorder=2)
        cbar = fig.colorbar(pm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("dist to nearest wall (mm)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        skeleton = data["skeleton"]
        if len(skeleton) > 0:
            ax.scatter(skeleton[:, 0], skeleton[:, 1],
                       s=2, c="white", linewidths=0, zorder=5)

        n_desired = int(desired_open.sum())
        ax.set_title(
            f"{session}  |  {data['n_valid']} reachable  |  "
            f"{n_desired} in desired band [{DESIRED_MIN_MM}–{DESIRED_MAX_MM} mm]",
            fontsize=8)

        # Wall point cloud on top
        if len(walls) > 0:
            ax.scatter(walls[:, 0], walls[:, 1], s=1.5, c="#ff4400",
                       alpha=0.7, linewidths=0, zorder=2)

        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Y (mm)", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    session_results = {}
    for session in SESSIONS:
        print(f"Processing {session} ...", end=" ", flush=True)
        arena = load_arena(session)
        xs_grid, ys_grid, dist_surface, dist_smooth, desired_open, n_valid = compute_valid_positions(arena)
        skeleton = compute_skeleton(desired_open, xs_grid, ys_grid)
        np.save(os.path.join(OUTPUT_DIR, f"skeleton_{session}.npy"), skeleton)

        print(f"{n_valid} valid positions, {len(skeleton)} skeleton pts")
        session_results[session] = {
            "arena":        arena,
            "xs_grid":      xs_grid,
            "ys_grid":      ys_grid,
            "dist_surface": dist_surface,
            "dist_smooth":  dist_smooth,
            "desired_open": desired_open,
            "skeleton":     skeleton,
            "n_valid":      n_valid,
        }

    plot_valid_space(session_results,
                     os.path.join(OUTPUT_DIR, "plot_valid_space.png"))


if __name__ == "__main__":
    main()
