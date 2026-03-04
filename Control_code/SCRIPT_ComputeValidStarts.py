"""
SCRIPT_ComputeValidStarts.py
============================
Computes valid robot starting positions and orientations for each arena session.

A position is valid if the robot centre is at least WALL_MARGIN_MM from every
wall point (i.e. it fits without touching anything).

A heading is valid at a position if there is no wall within
MIN_FORWARD_CLEARANCE_MM in a forward cone of ±CONE_HALF_WIDTH_DEG around
that heading.  This prevents spawning with the robot aimed directly at a wall.

Outputs
-------
- ValidStarts/<session>_valid_starts.json  : list of {x, y, yaw_deg}
- ValidStarts/plot_valid_starts.png        : diagnostic figure (one panel / session)

The core logic is also exposed as a reusable function:
    from SCRIPT_ComputeValidStarts import compute_valid_starts
    starts = compute_valid_starts(arena, ...)
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from Library.EnvironmentSimulator import ArenaLayout

# ── Configuration ─────────────────────────────────────────────────────────────

SESSIONS       = ["sessionB01", "sessionB02", "sessionB03", "sessionB04", "sessionB05"]
OUTPUT_DIR     = "ValidStarts"

GRID_STEP_MM          = 50     # position grid resolution (mm)
WALL_MARGIN_MM        = 150    # min distance robot centre → any wall point (mm)
                                #   150 mm > robot radius (85 mm) + comfortable margin
MIN_FORWARD_CLEARANCE_MM = 3 * WALL_MARGIN_MM
                                # heading blocked if wall within this distance ahead (mm)
                                #   set to 2× wall margin so orientation cutoff scales
                                #   automatically with the position clearance
CONE_HALF_WIDTH_DEG   = 15     # half-width of the forward-clearance cone (deg)
HEADING_STEP_DEG      = 10     # angular resolution for heading sweep (deg)

# How many example heading-fan arrows to draw per session in the plot
N_FAN_EXAMPLES = 12

# ── Core functions ─────────────────────────────────────────────────────────────

def _min_wall_distances(walls: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Vectorised: minimum distance from each (xs[i], ys[i]) to the wall point cloud.

    Parameters
    ----------
    walls : (N, 2) float array  — wall point cloud in mm
    xs, ys : (M,) float arrays  — query positions

    Returns
    -------
    (M,) float array of minimum distances in mm
    """
    pts = np.column_stack([xs, ys])          # (M, 2)
    # Compute pairwise distances efficiently using broadcasting
    # (M, 1, 2) - (1, N, 2)  →  (M, N)
    diff = pts[:, np.newaxis, :] - walls[np.newaxis, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))     # (M, N)
    return d.min(axis=1)                     # (M,)


def _inside_arena_mask(walls: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                       n_sectors: int = 8) -> np.ndarray:
    """
    Vectorised inside-arena test using angular sector coverage.

    A point is considered inside the arena if the wall point cloud contains at
    least one point in every angular sector around it.  A point outside the
    arena will have at least one empty sector (the one pointing away from the
    arena into open space), so this correctly rejects exterior positions even
    when they happen to be far from the nearest wall point.

    Parameters
    ----------
    walls    : (N, 2) wall point cloud in mm
    xs, ys   : (M,) candidate positions
    n_sectors: number of equal angular slices (8 → 45° each)

    Returns
    -------
    (M,) bool array — True where the point is enclosed by walls on all sides
    """
    if len(walls) == 0:
        return np.ones(len(xs), dtype=bool)

    dx = walls[:, 0][np.newaxis, :] - xs[:, np.newaxis]   # (M, N)
    dy = walls[:, 1][np.newaxis, :] - ys[:, np.newaxis]   # (M, N)
    angles = (np.degrees(np.arctan2(dy, dx)) % 360)        # (M, N) in [0, 360)

    sector_size = 360.0 / n_sectors
    inside = np.ones(len(xs), dtype=bool)
    for i in range(n_sectors):
        lo = i * sector_size
        hi = (i + 1) * sector_size
        has_wall = np.any((angles >= lo) & (angles < hi), axis=1)  # (M,)
        inside &= has_wall

    return inside


def _valid_headings(x: float, y: float, walls: np.ndarray,
                    heading_step: int = HEADING_STEP_DEG,
                    min_clearance: float = MIN_FORWARD_CLEARANCE_MM,
                    cone_half: float = CONE_HALF_WIDTH_DEG) -> np.ndarray:
    """
    Return array of valid heading angles (degrees, 0 = +X axis / East) at (x, y).

    A heading h is *invalid* if any wall point lies within `min_clearance` mm
    inside a cone of ±cone_half degrees around h.
    """
    headings = np.arange(0, 360, heading_step, dtype=float)
    if len(walls) == 0:
        return headings

    dx = walls[:, 0] - x
    dy = walls[:, 1] - y
    wall_angles = np.degrees(np.arctan2(dy, dx))   # (N,) in [-180, 180]
    wall_dists  = np.hypot(dx, dy)                 # (N,)

    valid = []
    for h in headings:
        dang = wall_angles - h
        dang = (dang + 180.0) % 360.0 - 180.0      # wrap to [-180, 180]
        in_cone = np.abs(dang) <= cone_half
        if not np.any(in_cone):
            # Nothing in this direction at all — definitely valid
            valid.append(h)
        elif wall_dists[in_cone].min() > min_clearance:
            valid.append(h)

    return np.array(valid, dtype=float)


def compute_valid_starts(
    arena: ArenaLayout,
    wall_margin_mm: float       = WALL_MARGIN_MM,
    min_forward_clearance_mm: float = None,  # defaults to 2 × wall_margin_mm
    grid_step_mm: float         = GRID_STEP_MM,
    heading_step_deg: int       = HEADING_STEP_DEG,
    cone_half_width_deg: float  = CONE_HALF_WIDTH_DEG,
) -> list:
    """
    Compute all valid (x, y, yaw_deg) starting configurations for a given arena.

    Parameters
    ----------
    arena                    : ArenaLayout instance (already loaded)
    wall_margin_mm           : minimum clearance from any wall point (mm)
    min_forward_clearance_mm : heading invalid if wall within this range ahead (mm)
    grid_step_mm             : spacing of the position candidate grid (mm)
    heading_step_deg         : angular resolution of heading sweep (degrees)
    cone_half_width_deg      : half-angle of forward-clearance cone (degrees)

    Returns
    -------
    list of dict  [{"x": float, "y": float, "yaw_deg": float}, ...]
    Each entry is one (position, heading) pair.  Multiple headings per position
    are returned as separate entries.
    """
    if min_forward_clearance_mm is None:
        min_forward_clearance_mm = 2.0 * wall_margin_mm

    walls = arena.walls  # (N, 2) mm

    # ── 1. Build candidate position grid ──────────────────────────────────────
    xs = np.arange(arena.arena_min_x, arena.arena_max_x + grid_step_mm, grid_step_mm)
    ys = np.arange(arena.arena_min_y, arena.arena_max_y + grid_step_mm, grid_step_mm)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel()
    gy = gy.ravel()

    # ── 2. Filter positions: inside arena AND correct wall clearance ──────────
    if len(walls) > 0:
        # Inside test: walls must exist in all 8 angular sectors — this rejects
        # points outside the arena boundary (e.g. rectangle corners outside an
        # octagonal arena) which would otherwise pass the distance test because
        # the wall point cloud has no points on the exterior side.
        inside = _inside_arena_mask(walls, gx, gy, n_sectors=8)
        min_d  = _min_wall_distances(walls, gx, gy)
        pos_mask = inside & (min_d >= wall_margin_mm)
    else:
        pos_mask = np.ones(len(gx), dtype=bool)

    valid_x = gx[pos_mask]
    valid_y = gy[pos_mask]

    # ── 3. For each valid position, compute valid headings ────────────────────
    starts = []
    for x, y in zip(valid_x, valid_y):
        headings = _valid_headings(
            x, y, walls,
            heading_step=heading_step_deg,
            min_clearance=min_forward_clearance_mm,
            cone_half=cone_half_width_deg,
        )
        for h in headings:
            starts.append({"x": float(x), "y": float(y), "yaw_deg": float(h)})

    return starts


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_session(ax, fig, arena: ArenaLayout, starts: list, session: str,
                  n_fan_examples: int = N_FAN_EXAMPLES,
                  arrow_len_mm: float = 250.0):
    """
    Draw one diagnostic panel for a session.

    Positions are coloured by the fraction of headings that are valid
    (0 = all blocked → dark, 1 = fully free → bright).  A rose of valid
    heading arrows is drawn at N_FAN_EXAMPLES evenly-spaced example positions
    so you can see the directional constraint near walls.
    """
    walls = arena.walls
    n_total_headings = 360 // HEADING_STEP_DEG  # e.g. 36

    # ── Wall point cloud ──────────────────────────────────────────────────────
    if len(walls) > 0:
        ax.scatter(walls[:, 0], walls[:, 1], s=0.8, c="#aaaaaa", alpha=0.5,
                   linewidths=0, zorder=1)

    if not starts:
        ax.set_title(f"{session}  |  no valid starts", fontsize=9)
        return

    # ── Build per-position dict ───────────────────────────────────────────────
    pos_to_heads: dict = {}
    for s in starts:
        key = (s["x"], s["y"])
        pos_to_heads.setdefault(key, []).append(s["yaw_deg"])

    positions   = np.array(list(pos_to_heads.keys()))   # (M, 2)
    frac_valid  = np.array(
        [len(pos_to_heads[k]) / n_total_headings for k in pos_to_heads]
    )  # (M,) in [0, 1]

    # ── Scatter: colour = heading freedom ────────────────────────────────────
    sc = ax.scatter(positions[:, 0], positions[:, 1],
                    c=frac_valid, cmap="YlGnBu", vmin=0.0, vmax=1.0,
                    s=12, linewidths=0, zorder=2, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("fraction of valid headings", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # ── Heading rose at N_FAN_EXAMPLES example positions ─────────────────────
    # Pick positions spread evenly by index — this samples across the spatial
    # layout rather than clustering at one corner.
    n_ex = min(n_fan_examples, len(positions))
    idx  = np.round(np.linspace(0, len(positions) - 1, n_ex)).astype(int)
    for i in idx:
        px, py = positions[i]
        for h in pos_to_heads[(px, py)]:
            rad = np.radians(h)
            ax.annotate("",
                xy=(px + arrow_len_mm * np.cos(rad),
                    py + arrow_len_mm * np.sin(rad)),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color="tomato",
                                lw=0.8, mutation_scale=5),
                zorder=3)
        # Black dot at the example position so it stands out
        ax.plot(px, py, "ko", ms=3, zorder=4)

    n_pos = len(positions)
    ax.set_title(
        f"{session}  |  {n_pos} valid positions  |  {len(starts)} (pos, yaw) pairs",
        fontsize=9,
    )
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def plot_valid_starts(session_data: dict, out_path: str):
    """
    session_data : {session_name: {"arena": ArenaLayout, "starts": list}}
    """
    n = len(session_data)
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.5 * n_cols, 6.5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    fig.suptitle(
        f"Valid start positions & headings\n"
        f"wall margin={WALL_MARGIN_MM} mm  |  "
        f"fwd clearance={MIN_FORWARD_CLEARANCE_MM} mm  |  "
        f"cone ±{CONE_HALF_WIDTH_DEG}°  |  "
        f"grid {GRID_STEP_MM} mm",
        fontsize=10,
    )

    for idx, (session, data) in enumerate(session_data.items()):
        row, col = divmod(idx, n_cols)
        _plot_session(axes[row, col], fig, data["arena"], data["starts"], session)

    # Hide unused panels
    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    session_data = {}

    for session in SESSIONS:
        print(f"Processing {session} ...", end=" ", flush=True)

        arena = ArenaLayout(session)

        starts = compute_valid_starts(
            arena,
            wall_margin_mm           = WALL_MARGIN_MM,
            min_forward_clearance_mm = MIN_FORWARD_CLEARANCE_MM,
            grid_step_mm             = GRID_STEP_MM,
            heading_step_deg         = HEADING_STEP_DEG,
            cone_half_width_deg      = CONE_HALF_WIDTH_DEG,
        )

        n_pos = len({(s["x"], s["y"]) for s in starts})
        print(f"{n_pos} positions, {len(starts)} (pos, heading) pairs")

        # Save JSON
        json_path = os.path.join(OUTPUT_DIR, f"{session}_valid_starts.json")
        with open(json_path, "w") as f:
            json.dump({
                "session":                   session,
                "wall_margin_mm":            WALL_MARGIN_MM,
                "min_forward_clearance_mm":  MIN_FORWARD_CLEARANCE_MM,
                "grid_step_mm":              GRID_STEP_MM,
                "heading_step_deg":          HEADING_STEP_DEG,
                "cone_half_width_deg":       CONE_HALF_WIDTH_DEG,
                "n_valid_positions":         n_pos,
                "n_valid_starts":            len(starts),
                "starts":                    starts,
            }, f, indent=2)
        print(f"  Saved: {json_path}")

        session_data[session] = {"arena": arena, "starts": starts}

    # Combined plot
    plot_path = os.path.join(OUTPUT_DIR, "plot_valid_starts.png")
    plot_valid_starts(session_data, plot_path)


if __name__ == "__main__":
    main()
