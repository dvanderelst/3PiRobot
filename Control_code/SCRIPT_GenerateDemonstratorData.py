"""
SCRIPT_GenerateDemonstratorData.py
=================================
Generates demonstrator data for arena shapes.

Pipeline
--------
1. Load arena wall point cloud for each session.
2. Build a candidate position grid.
3. Filter: inside arena (sector test) + minimum wall clearance.
4. Keep only the largest connected component (removes unreachable zones such as
   positions inside enclosed obstacles).
5. Compute the desired-band mask.

Outputs
-------
- DemonstratorData/plot_valid_space.png            diagnostic figure (one panel / session)
"""

import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion as _erode, binary_dilation as _dilate, gaussian_filter as _gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
from Library.DataProcessor import read_wall_mask, mask2coordinates
from Library.GeometryUtils import (
    inside_arena_mask,
    largest_component_mask,
    min_wall_distances,
)
from Library.Demonstrator import Demonstrator

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
OUTPUT_DIR    = "DemonstratorData"
GRID_STEP_MM  = 50    # position grid resolution (mm)
WALL_MARGIN_MM = 150  # min distance robot centre → any wall point (mm)

# Desired operating band for the demonstrator (mm from nearest wall).
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

# Gradient field shape: slope inside the desired band relative to outside.
# 1.0 = symmetric tent; <1.0 = gentler nudge once inside the band.
INNER_GRADIENT_SCALE = 0.1

# Quiver plot: show one arrow every N grid cells.
QUIVER_STRIDE = 3

# Demonstrator track simulation.
N_TRACKS     = 30              # number of random starting poses per session
N_STEPS      = 100              # steps to simulate per track
STEP_SIZE_MM = 100             # forward step size (mm)
MAX_TURN_RAD_MIN = np.deg2rad(30)   # max turn on the ridge (smooth cruising)
MAX_TURN_RAD_MAX = np.deg2rad(60)  # max turn near walls (aggressive correction)

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
    dist_smooth      : (n_rows, n_cols) float array, smoothed distance surface
    desired_open     : (n_rows, n_cols) bool array, desired band mask
    n_valid          : number of valid positions
    """
    walls = arena["walls"]

    xs_grid = np.arange(arena["arena_min_x"], arena["arena_max_x"] + grid_step_mm, grid_step_mm)
    ys_grid = np.arange(arena["arena_min_y"], arena["arena_max_y"] + grid_step_mm, grid_step_mm)
    gx, gy  = np.meshgrid(xs_grid, ys_grid)
    shape   = gx.shape

    if len(walls) > 0:
        walls_sparse = walls[::10]
        inside   = inside_arena_mask(walls_sparse, gx.ravel(), gy.ravel(), n_sectors=8)
        min_d    = min_wall_distances(walls, gx.ravel(), gy.ravel())
        pos_mask = inside & (min_d >= wall_margin_mm)
        pos_mask = largest_component_mask(pos_mask, shape)
    else:
        min_d    = np.full(gx.size, np.inf)
        pos_mask = np.ones(gx.size, dtype=bool)

    dist_surface = np.where(pos_mask, min_d, np.nan).reshape(shape)

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
    r_open = OPEN_RADIUS_CELLS
    cy, cx = np.ogrid[-r_open:r_open+1, -r_open:r_open+1]
    struct_open = (cx**2 + cy**2) <= r_open**2
    opened = _dilate(_erode(desired_raw, struct_open), struct_open)

    r_close = CLOSE_RADIUS_CELLS
    cy, cx = np.ogrid[-r_close:r_close+1, -r_close:r_close+1]
    struct_close = (cx**2 + cy**2) <= r_close**2
    desired_open = _erode(_dilate(opened, struct_close), struct_close)

    return xs_grid, ys_grid, dist_surface, dist_smooth, desired_open, int(pos_mask.sum())


def compute_potential(dist_smooth: np.ndarray) -> np.ndarray:
    """
    Tent-shaped potential peaking at band centre.

    Slope outside the band (d < DESIRED_MIN or d > DESIRED_MAX) is 1.
    Slope inside the band is INNER_GRADIENT_SCALE (gentler).
    The gradient of this field is the steering signal:
      - points toward band when outside it
      - gently nudges toward ridge when inside it
    """
    band_center = (DESIRED_MIN_MM + DESIRED_MAX_MM) / 2.0
    inner_half  = band_center - DESIRED_MIN_MM
    peak        = DESIRED_MIN_MM + inner_half * INNER_GRADIENT_SCALE

    d = dist_smooth
    V = np.where(
        d <= DESIRED_MIN_MM,
            d,
        np.where(
            d <= band_center,
                DESIRED_MIN_MM + (d - DESIRED_MIN_MM) * INNER_GRADIENT_SCALE,
            np.where(
                d <= DESIRED_MAX_MM,
                    peak - (d - band_center) * INNER_GRADIENT_SCALE,
                    peak - inner_half * INNER_GRADIENT_SCALE - (d - DESIRED_MAX_MM)
            )
        )
    )
    V_norm = (np.maximum(V, float(WALL_MARGIN_MM)) - WALL_MARGIN_MM) / (peak - WALL_MARGIN_MM)
    return np.where(np.isfinite(dist_smooth), V_norm, np.nan)


def compute_gradient(potential: np.ndarray,
                     xs_grid: np.ndarray,
                     ys_grid: np.ndarray) -> tuple:
    """
    Numerical gradient of the potential field.
    Returns (gx, gy): gradient components in x and y, NaN outside valid area.
    """
    filled = np.where(np.isfinite(potential), potential, 0.0)
    # np.gradient returns [d/drow, d/dcol] = [d/dy, d/dx]
    grad_y, grad_x = np.gradient(filled, ys_grid, xs_grid)
    valid = np.isfinite(potential)
    return (np.where(valid, grad_x, np.nan),
            np.where(valid, grad_y, np.nan))


def simulate_tracks(demonstrator: Demonstrator,
                    xs_grid: np.ndarray,
                    ys_grid: np.ndarray,
                    dist_surface: np.ndarray,
                    potential: np.ndarray,
                    n_tracks: int = N_TRACKS,
                    n_steps:  int = N_STEPS,
                    step_size_mm: float = STEP_SIZE_MM,
                    rng: np.random.Generator = None) -> list:
    """
    Simulate demonstrator tracks from random starting poses.

    Starting positions are sampled uniformly from all valid grid cells
    (inside and outside the desired band) with a random yaw.

    Returns a list of (xs, ys, yaw0) tuples, one per track.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Use dist_surface (not potential) for the valid mask — avoids gaussian bleed
    rows, cols = np.where(np.isfinite(dist_surface))
    if len(rows) == 0:
        return []

    chosen = rng.choice(len(rows), size=min(n_tracks, len(rows)), replace=False)
    tracks = []
    for i in chosen:
        x    = float(xs_grid[cols[i]])
        y    = float(ys_grid[rows[i]])
        yaw0 = rng.uniform(-np.pi, np.pi)
        yaw  = yaw0

        xs_track = [x]
        ys_track = [y]
        for _ in range(n_steps):
            delta = demonstrator.get_delta_angle(x, y, yaw)
            V = demonstrator.get_potential(x, y)
            V = 0.0 if np.isnan(V) else V
            max_turn = MAX_TURN_RAD_MIN + (1.0 - V) * (MAX_TURN_RAD_MAX - MAX_TURN_RAD_MIN)
            yaw  += np.clip(delta, -max_turn, max_turn)
            x    += step_size_mm * np.cos(yaw)
            y    += step_size_mm * np.sin(yaw)
            xs_track.append(x)
            ys_track.append(y)

        tracks.append((np.array(xs_track), np.array(ys_track), yaw0))

    return tracks


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_valid_space(session_results: dict, out_path: str):
    n = len(session_results)
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 6.5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    fig.suptitle(
        f"Valid (reachable) floor space\n"
        f"wall margin={WALL_MARGIN_MM} mm | grid {GRID_STEP_MM} mm | "
        f"desired band {DESIRED_MIN_MM}–{DESIRED_MAX_MM} mm | "
        f"open={OPEN_RADIUS_CELLS} cells | close={CLOSE_RADIUS_CELLS} cells | "
        f"smooth σ={DIST_SMOOTH_SIGMA_CELLS} cells",
        fontsize=10,
    )

    for idx, (session, data) in enumerate(session_results.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        walls = data["arena"]["walls"]
        xs_grid = data["xs_grid"]
        ys_grid = data["ys_grid"]
        dist_surface = data["dist_surface"]
        dist_smooth = data["dist_smooth"]

        half = (xs_grid[1] - xs_grid[0]) / 2
        x_edges = np.append(xs_grid - half, xs_grid[-1] + half)
        y_edges = np.append(ys_grid - half, ys_grid[-1] + half)

        ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(dist_surface),
                      cmap="Greys", vmin=0, vmax=3000,
                      shading="flat", zorder=1, alpha=0.3)

        desired_open = data["desired_open"]
        desired_dist = np.where(desired_open, dist_smooth, np.nan)
        pm = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(desired_dist),
                           cmap="viridis", vmin=DESIRED_MIN_MM, vmax=DESIRED_MAX_MM,
                           shading="flat", zorder=2)
        cbar = fig.colorbar(pm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("dist to nearest wall (mm)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        n_desired = int(desired_open.sum())
        ax.set_title(
            f"{session} | {data['n_valid']} reachable | "
            f"{n_desired} in desired band [{DESIRED_MIN_MM}–{DESIRED_MAX_MM} mm]",
            fontsize=8)

        if len(walls) > 0:
            ax.scatter(walls[:, 0], walls[:, 1], s=1.5, c="black",
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


def plot_gradient_field(session_results: dict, out_path: str):
    n = len(session_results)
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 6.5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    fig.suptitle(
        f"Potential field & gradient\n"
        f"inner slope={INNER_GRADIENT_SCALE} | band {DESIRED_MIN_MM}–{DESIRED_MAX_MM} mm",
        fontsize=10,
    )

    for idx, (session, data) in enumerate(session_results.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        xs_grid  = data["xs_grid"]
        ys_grid  = data["ys_grid"]
        potential = data["potential"]
        gx        = data["grad_x"]
        gy        = data["grad_y"]
        walls     = data["arena"]["walls"]

        half = (xs_grid[1] - xs_grid[0]) / 2
        x_edges = np.append(xs_grid - half, xs_grid[-1] + half)
        y_edges = np.append(ys_grid - half, ys_grid[-1] + half)

        pm = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(potential),
                           cmap="RdYlGn", shading="flat", zorder=1)
        cbar = fig.colorbar(pm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("potential", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        # Subsampled normalised quiver
        s = QUIVER_STRIDE
        gx_s = gx[::s, ::s]
        gy_s = gy[::s, ::s]
        xs_s = xs_grid[::s]
        ys_s = ys_grid[::s]
        mag = np.sqrt(gx_s**2 + gy_s**2)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(mag > 0, gx_s / mag, np.nan)
            v = np.where(mag > 0, gy_s / mag, np.nan)
        ax.quiver(xs_s, ys_s, u, v,
                  scale=30, scale_units="inches",
                  width=0.003, color="white", alpha=0.7, zorder=3)

        arrow_len = STEP_SIZE_MM * 1.5
        for xs_t, ys_t, yaw0 in data.get("tracks", []):
            ax.plot(xs_t, ys_t, "-", linewidth=1.2, alpha=0.8, zorder=5)
            ax.annotate("", zorder=6,
                        xy=(xs_t[0] + arrow_len * np.cos(yaw0),
                            ys_t[0] + arrow_len * np.sin(yaw0)),
                        xytext=(xs_t[0], ys_t[0]),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

        if len(walls) > 0:
            ax.scatter(walls[:, 0], walls[:, 1], s=1.5, c="black",
                       alpha=0.7, linewidths=0, zorder=4)

        ax.set_title(session, fontsize=8)
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

        potential = compute_potential(dist_smooth)
        grad_x, grad_y = compute_gradient(potential, xs_grid, ys_grid)

        demonstrator = Demonstrator(xs_grid, ys_grid, potential, grad_x, grad_y)
        tracks = simulate_tracks(demonstrator, xs_grid, ys_grid, dist_surface, potential)

        np.savez(
            os.path.join(OUTPUT_DIR, f"demonstrator_{session}.npz"),
            xs_grid=xs_grid, ys_grid=ys_grid,
            potential=potential, grad_x=grad_x, grad_y=grad_y,
            dist_surface=dist_surface,
        )
        print(f"{n_valid} valid positions, {int(desired_open.sum())} in desired band")
        session_results[session] = {
            "arena": arena,
            "xs_grid": xs_grid,
            "ys_grid": ys_grid,
            "dist_surface": dist_surface,
            "dist_smooth": dist_smooth,
            "desired_open": desired_open,
            "potential": potential,
            "grad_x": grad_x,
            "grad_y": grad_y,
            "tracks": tracks,
            "n_valid": n_valid,
        }

    plot_valid_space(session_results, os.path.join(OUTPUT_DIR, "plot_valid_space.png"))
    plot_gradient_field(session_results, os.path.join(OUTPUT_DIR, "plot_gradient_field.png"))


if __name__ == "__main__":
    main()
