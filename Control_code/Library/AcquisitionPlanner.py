"""
Acquisition planner for vision-guided sonar data collection.

Iteratively grows a tour of feasible (x, y) waypoints in an annotated arena.
Each new waypoint must satisfy:
  - position clearance: at least `clearance_mm` from every wall point
  - min step length:    at least `min_step_mm` from the previous waypoint
  - segment clearance:  the straight line from previous to new stays at least
                        `clearance_mm` from every wall point

Constraint #3 is what lets `TrackerNav.go_to_pose` drive the leg open-loop
without obstacle planning — every segment is feasible by construction.

At each waypoint the robot will ping at `n_yaws` orientations, uniformly
spaced 360°/n_yaws apart starting from a per-position random offset. The
yaws are baked into the plan at build time so a saved plan is a complete
spec of the run.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import matplotlib
import os
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np


# ─── Geometry helpers ────────────────────────────────────────────────────────

def min_dist_point_to_walls(p, walls) -> float:
    """Min Euclidean distance (mm) from point `p` (2,) to any row of `walls` (N, 2)."""
    p = np.asarray(p, dtype=np.float64)
    walls = np.asarray(walls, dtype=np.float64)
    return float(np.linalg.norm(walls - p, axis=1).min())


def arena_polygon(walls) -> MplPath:
    """Convex hull of wall points, returned as a closed `matplotlib.path.Path`.

    The `arena_bounds_mm` from meta.json is the camera FOV bounding box, which
    is typically larger than the playable arena. Sampling uniformly inside the
    bounding box and rejecting on wall-clearance accepts points far outside
    the arena (they're far from any wall, so clearance trivially passes). The
    convex hull of the wall point cloud is a good approximation of "inside
    the arena" when the perimeter is roughly convex; an L-shaped or otherwise
    concave arena would need an alpha-shape instead — file an issue if that
    becomes a real case.
    """
    pts = np.asarray(walls, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    poly_closed = np.vstack([hull, hull[:1]])
    return MplPath(poly_closed)


def min_dist_segment_to_walls(p1, p2, walls) -> float:
    """Min distance (mm) from any wall point to the segment p1-p2.

    Vectorised: project each wall onto the segment, clamp to [0, 1] in the
    parametric coordinate, take the foot, return min point-to-foot distance.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    walls = np.asarray(walls, dtype=np.float64)
    v = p2 - p1
    L2 = float(v @ v)
    if L2 < 1e-9:
        return min_dist_point_to_walls(p1, walls)
    diff = walls - p1
    t = np.clip((diff @ v) / L2, 0.0, 1.0)
    foot = p1 + t[:, None] * v
    return float(np.linalg.norm(walls - foot, axis=1).min())


# ─── Arena loading ───────────────────────────────────────────────────────────

def load_arena(arena_dir):
    """Load wall point cloud and bounds from `<arena_dir>/env_*/`.

    Picks the most recent env_* by name (which sorts by ISO timestamp),
    so re-snapshotting the arena into a fresh env_* folder takes effect
    without code changes. Returns a dict with keys:
        walls   (N, 2) float32 array of wall points in world mm
        bounds  dict with min_x, max_x, min_y, max_y
        env_dir Path to the env folder used (for plot reference)
    """
    arena_dir = Path(arena_dir)
    env_dirs = sorted(p for p in arena_dir.iterdir()
                      if p.is_dir() and p.name.startswith("env_"))
    if not env_dirs:
        raise FileNotFoundError(f"No env_* subfolder under {arena_dir}")
    env_dir = env_dirs[-1]
    walls_path = env_dir / "arena_walls.npz"
    meta_path = env_dir / "meta.json"
    if not walls_path.exists():
        raise FileNotFoundError(
            f"{walls_path} not found — run SCRIPT_BuildArenaGeometry.py first")
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")
    d = np.load(walls_path)
    walls = np.column_stack([d["x_mm"], d["y_mm"]]).astype(np.float32)
    with open(meta_path) as f:
        meta = json.load(f)
    return {"walls": walls, "bounds": meta["arena_bounds_mm"], "env_dir": env_dir}


# ─── Plan dataclass and IO ───────────────────────────────────────────────────

@dataclass
class AcquisitionPlan:
    arena_name: str
    arena_dir: str
    positions: List[List[float]]            # [[x, y], ...] in tour order
    yaws_at_position: List[List[float]]     # [[yaw_0, ...], ...] one list per position
    n_yaws: int
    clearance_mm: float
    min_step_mm: float
    seed: int


def save_plan(plan: AcquisitionPlan, path):
    with open(path, "w") as f:
        json.dump(asdict(plan), f, indent=2)


def load_plan(path) -> AcquisitionPlan:
    with open(path) as f:
        d = json.load(f)
    return AcquisitionPlan(**d)


# ─── Sampling helpers ────────────────────────────────────────────────────────

def _sample_in_bounds(bounds, rng) -> np.ndarray:
    """Uniform random (x, y) in the arena bounding box."""
    return np.array([
        rng.uniform(bounds["min_x"], bounds["max_x"]),
        rng.uniform(bounds["min_y"], bounds["max_y"]),
    ], dtype=np.float64)


def yaw_set(n_yaws: int, rng) -> List[float]:
    """Uniform yaws spaced 360/n apart, starting at a random offset in [0, 360/n).

    Result wrapped to [-180, 180), matching the project's CCW yaw convention.
    """
    spacing = 360.0 / n_yaws
    offset = float(rng.uniform(0.0, spacing))
    yaws = []
    for i in range(n_yaws):
        y = offset + i * spacing
        y = ((y + 180.0) % 360.0) - 180.0
        yaws.append(y)
    return yaws


# ─── Plan construction ───────────────────────────────────────────────────────

def _find_feasible_start(walls, bounds, arena_path, clearance_mm, rng,
                         max_tries: int = 2000) -> np.ndarray:
    for _ in range(max_tries):
        cand = _sample_in_bounds(bounds, rng)
        if not arena_path.contains_point(cand):
            continue
        if min_dist_point_to_walls(cand, walls) >= clearance_mm:
            return cand
    raise RuntimeError(
        f"Could not find a start position with clearance {clearance_mm:.0f} mm "
        f"inside the arena polygon after {max_tries} tries; arena may be too "
        f"cluttered or the clearance too large."
    )


def build_plan(arena,
               target_k: int,
               clearance_mm: float,
               min_step_mm: float,
               max_attempts_per_step: int,
               n_yaws: int,
               arena_name: str,
               arena_dir,
               seed: int) -> AcquisitionPlan:
    """Iteratively grow a tour of feasible waypoints.

    Returns a plan with up to `target_k + 1` positions (start + target_k more),
    or fewer if the tour terminates early when no feasible next step is found
    within `max_attempts_per_step` candidates.
    """
    rng = np.random.default_rng(seed)
    walls = arena["walls"]
    bounds = arena["bounds"]
    arena_path = arena_polygon(walls)

    rejections = {"outside_arena": 0, "position_clearance": 0,
                  "min_step": 0, "segment_clearance": 0}

    start = _find_feasible_start(walls, bounds, arena_path, clearance_mm, rng)
    positions: List[np.ndarray] = [start]
    yaws: List[List[float]] = [yaw_set(n_yaws, rng)]

    for _ in range(target_k):
        added = False
        for _attempt in range(max_attempts_per_step):
            cand = _sample_in_bounds(bounds, rng)
            if not arena_path.contains_point(cand):
                rejections["outside_arena"] += 1
                continue
            if min_dist_point_to_walls(cand, walls) < clearance_mm:
                rejections["position_clearance"] += 1
                continue
            if np.linalg.norm(cand - positions[-1]) < min_step_mm:
                rejections["min_step"] += 1
                continue
            if min_dist_segment_to_walls(positions[-1], cand, walls) < clearance_mm:
                rejections["segment_clearance"] += 1
                continue
            positions.append(cand)
            yaws.append(yaw_set(n_yaws, rng))
            added = True
            break
        if not added:
            print(f"  plan terminated after {len(positions)} positions "
                  f"(no feasible next within {max_attempts_per_step} candidates)")
            break

    print(f"  rejections during build: "
          f"outside_arena={rejections['outside_arena']}, "
          f"position={rejections['position_clearance']}, "
          f"min_step={rejections['min_step']}, "
          f"segment={rejections['segment_clearance']}")

    return AcquisitionPlan(
        arena_name=arena_name,
        arena_dir=str(arena_dir),
        positions=[[float(p[0]), float(p[1])] for p in positions],
        yaws_at_position=yaws,
        n_yaws=n_yaws,
        clearance_mm=clearance_mm,
        min_step_mm=min_step_mm,
        seed=int(seed),
    )


# ─── Diagnostic plot ─────────────────────────────────────────────────────────

def plot_plan(plan: AcquisitionPlan, arena, out_path) -> None:
    """Save a diagnostic figure with arena background, walls, tour segments,
    and waypoints coloured by min-wall-distance so close-wall coverage is
    obvious at a glance."""
    walls = arena["walls"]
    bounds = arena["bounds"]
    env_dir = Path(arena["env_dir"])

    positions = np.array(plan.positions)
    min_dists = np.array([min_dist_point_to_walls(p, walls) for p in positions])

    fig, ax = plt.subplots(figsize=(11, 9))

    arena_img_path = env_dir / "arena.png"
    if arena_img_path.exists():
        try:
            import cv2
            img = cv2.imread(str(arena_img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(
                    img_rgb,
                    extent=(bounds["min_x"], bounds["max_x"],
                            bounds["min_y"], bounds["max_y"]),
                    origin="upper",
                    alpha=0.45,
                    zorder=0,
                )
        except ImportError:
            pass

    # Walls
    ax.scatter(walls[:, 0], walls[:, 1], s=0.6, c="black", alpha=0.35, zorder=1)

    # Arena polygon (convex hull of walls; the planner's "inside" region)
    hull_path = arena_polygon(walls)
    hull_pts = hull_path.vertices
    ax.plot(hull_pts[:, 0], hull_pts[:, 1], linestyle="--",
            color="#d62728", alpha=0.5, linewidth=1.0,
            label="planner feasible region", zorder=1.5)
    ax.legend(loc="lower left", fontsize=8)

    # Tour segments
    if len(positions) >= 2:
        ax.plot(positions[:, 0], positions[:, 1],
                color="#888", alpha=0.55, linewidth=0.8, zorder=2)

    # Waypoints coloured by min-wall-distance
    sc = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=min_dists, cmap="viridis_r",
        s=42, zorder=3, edgecolors="black", linewidths=0.4,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("min wall distance (mm)")

    # Annotate start, end, and every 10th
    for i, p in enumerate(positions):
        if i == 0:
            ax.annotate("START", (p[0], p[1]), fontsize=9, weight="bold",
                        color="red", xytext=(6, 6), textcoords="offset points",
                        zorder=4)
        elif i == len(positions) - 1:
            ax.annotate("END", (p[0], p[1]), fontsize=9, weight="bold",
                        color="red", xytext=(6, 6), textcoords="offset points",
                        zorder=4)
        elif i % 10 == 0:
            ax.annotate(str(i), (p[0], p[1]), fontsize=6, color="#444",
                        xytext=(3, 3), textcoords="offset points", zorder=4)

    ax.set_xlim(bounds["min_x"], bounds["max_x"])
    ax.set_ylim(bounds["min_y"], bounds["max_y"])
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    n_pos = len(plan.positions)
    n_pings = n_pos * plan.n_yaws
    ax.set_title(
        f"Acquisition plan: {plan.arena_name}\n"
        f"{n_pos} positions × {plan.n_yaws} yaws = {n_pings} pings  "
        f"(clearance {plan.clearance_mm:.0f} mm, "
        f"min step {plan.min_step_mm:.0f} mm, seed {plan.seed})"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
