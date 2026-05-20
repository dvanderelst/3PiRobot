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
    """Load feature point cloud + poles and bounds from `<arena_dir>/env_*/`.

    Picks the most recent env_* by name (which sorts by ISO timestamp),
    so re-snapshotting the arena into a fresh env_* folder takes effect
    without code changes. Returns a dict with keys:
        walls           (N, 2) float32 array of wall points in world mm
        poles           (M, 2) float32 array of pole centres in world mm
        pole_radius_mm  float, physical pole radius (0.0 if no poles)
        bounds          dict with min_x, max_x, min_y, max_y
        env_dir         Path to the env folder used (for plot reference)
    """
    arena_dir = Path(arena_dir)
    env_dirs = sorted(p for p in arena_dir.iterdir()
                      if p.is_dir() and p.name.startswith("env_"))
    if not env_dirs:
        raise FileNotFoundError(f"No env_* subfolder under {arena_dir}")
    env_dir = env_dirs[-1]
    features_path = env_dir / "arena_features.npz"
    meta_path = env_dir / "meta.json"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found — run SCRIPT_BuildArenaGeometry.py first")
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")
    d = np.load(features_path)
    x_all = np.asarray(d["x_mm"], dtype=np.float32)
    y_all = np.asarray(d["y_mm"], dtype=np.float32)
    kind = np.asarray(d["kind"], dtype=np.uint8)
    wall_mask = kind == 0
    pole_mask = kind == 1
    walls = np.column_stack([x_all[wall_mask], y_all[wall_mask]]).astype(np.float32)
    poles = np.column_stack([x_all[pole_mask], y_all[pole_mask]]).astype(np.float32)
    pole_radius_mm = float(d["pole_radius_mm"]) if "pole_radius_mm" in d.files else 0.0
    with open(meta_path) as f:
        meta = json.load(f)
    return {
        "walls": walls,
        "poles": poles,
        "pole_radius_mm": pole_radius_mm,
        "bounds": meta["arena_bounds_mm"],
        "env_dir": env_dir,
    }


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
    min_neighbor_mm: float                  # min distance to ANY prior waypoint
    seed: int


def save_plan(plan: AcquisitionPlan, path):
    with open(path, "w") as f:
        json.dump(asdict(plan), f, indent=2)


def load_plan(path) -> AcquisitionPlan:
    with open(path) as f:
        d = json.load(f)
    return AcquisitionPlan(**d)


# ─── Sampling helpers ────────────────────────────────────────────────────────

def _sample_in_polygon(poly_path: MplPath, rng) -> np.ndarray:
    """Uniform random `(x, y)` inside a convex polygon.

    Fan triangulation from vertex 0, pick a triangle weighted by area, then
    sample uniformly inside it via reflected-barycentric coordinates. Strictly
    faster than bbox + rejection when the polygon is much smaller than its
    bounding box (camera FOV vs. arena polygon ratio).
    """
    poly_pts = poly_path.vertices[:-1]  # drop the duplicated closing vertex
    n = len(poly_pts)
    v0 = poly_pts[0]
    tri_areas = np.array([
        0.5 * abs(np.cross(poly_pts[i] - v0, poly_pts[i + 1] - v0))
        for i in range(1, n - 1)
    ])
    cum = np.cumsum(tri_areas) / tri_areas.sum()
    tri_idx = int(np.searchsorted(cum, rng.random()))
    a = v0
    b = poly_pts[tri_idx + 1]
    c = poly_pts[tri_idx + 2]
    u1, u2 = rng.random(), rng.random()
    if u1 + u2 > 1.0:
        u1, u2 = 1.0 - u1, 1.0 - u2
    return a + u1 * (b - a) + u2 * (c - a)


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

def _find_feasible_start(walls, poles, pole_clearance_mm, arena_path,
                         clearance_mm, rng,
                         max_tries: int = 2000) -> np.ndarray:
    for _ in range(max_tries):
        cand = _sample_in_polygon(arena_path, rng)
        if min_dist_point_to_walls(cand, walls) < clearance_mm:
            continue
        if poles.size and min_dist_point_to_walls(cand, poles) < pole_clearance_mm:
            continue
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
               min_neighbor_mm: float,
               max_attempts_per_step: int,
               n_yaws: int,
               arena_name: str,
               arena_dir,
               seed: int) -> AcquisitionPlan:
    """Iteratively grow a tour of feasible waypoints.

    `min_step_mm` constrains *consecutive* waypoints; `min_neighbor_mm`
    constrains the new candidate against *every* prior waypoint and is what
    spreads samples uniformly. Set `min_neighbor_mm = 0` to disable the
    neighbor check (reverts to the old behaviour).

    Returns a plan with up to `target_k + 1` positions (start + target_k more),
    or fewer if the tour terminates early when no feasible next step is found
    within `max_attempts_per_step` candidates.
    """
    rng = np.random.default_rng(seed)
    walls = arena["walls"]
    poles = arena.get("poles", np.empty((0, 2), dtype=np.float32))
    pole_radius_mm = float(arena.get("pole_radius_mm", 0.0))
    pole_clearance_mm = clearance_mm + pole_radius_mm

    arena_path = arena_polygon(walls)

    rejections = {"position_clearance": 0, "pole_clearance": 0,
                  "min_step": 0, "min_neighbor": 0,
                  "segment_clearance": 0, "pole_segment": 0}

    start = _find_feasible_start(walls, poles, pole_clearance_mm,
                                 arena_path, clearance_mm, rng)
    positions: List[np.ndarray] = [start]
    yaws: List[List[float]] = [yaw_set(n_yaws, rng)]

    for _ in range(target_k):
        added = False
        pos_arr = np.asarray(positions)  # cached per outer iter; cheap
        for _attempt in range(max_attempts_per_step):
            cand = _sample_in_polygon(arena_path, rng)
            if min_dist_point_to_walls(cand, walls) < clearance_mm:
                rejections["position_clearance"] += 1
                continue
            if poles.size and min_dist_point_to_walls(cand, poles) < pole_clearance_mm:
                rejections["pole_clearance"] += 1
                continue
            if np.linalg.norm(cand - positions[-1]) < min_step_mm:
                rejections["min_step"] += 1
                continue
            if min_neighbor_mm > 0:
                if np.linalg.norm(pos_arr - cand, axis=1).min() < min_neighbor_mm:
                    rejections["min_neighbor"] += 1
                    continue
            if min_dist_segment_to_walls(positions[-1], cand, walls) < clearance_mm:
                rejections["segment_clearance"] += 1
                continue
            if poles.size and min_dist_segment_to_walls(
                    positions[-1], cand, poles) < pole_clearance_mm:
                rejections["pole_segment"] += 1
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
          f"position={rejections['position_clearance']}, "
          f"pole={rejections['pole_clearance']}, "
          f"min_step={rejections['min_step']}, "
          f"min_neighbor={rejections['min_neighbor']}, "
          f"segment={rejections['segment_clearance']}, "
          f"pole_segment={rejections['pole_segment']}")

    return AcquisitionPlan(
        arena_name=arena_name,
        arena_dir=str(arena_dir),
        positions=[[float(p[0]), float(p[1])] for p in positions],
        yaws_at_position=yaws,
        n_yaws=n_yaws,
        clearance_mm=clearance_mm,
        min_step_mm=min_step_mm,
        min_neighbor_mm=min_neighbor_mm,
        seed=int(seed),
    )


# ─── Diagnostic plot ─────────────────────────────────────────────────────────

def plot_plan(plan: AcquisitionPlan, arena, out_path) -> None:
    """Save a diagnostic figure with arena background, walls, tour segments,
    and waypoints coloured by min-wall-distance so close-wall coverage is
    obvious at a glance."""
    walls = arena["walls"]
    poles = arena.get("poles", np.empty((0, 2), dtype=np.float32))
    pole_radius_mm = float(arena.get("pole_radius_mm", 0.0))
    pole_clearance_mm = plan.clearance_mm + pole_radius_mm
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

    # Poles
    if poles.size:
        for px, py in poles:
            ax.add_patch(plt.Circle(
                (px, py), pole_radius_mm,
                facecolor="#984ea3", edgecolor="black", linewidth=0.6,
                alpha=0.85, zorder=2.2,
            ))

    # Arena polygon (convex hull of walls; the planner's "inside" region)
    hull_path = arena_polygon(walls)
    hull_pts = hull_path.vertices
    ax.plot(hull_pts[:, 0], hull_pts[:, 1], linestyle="--",
            color="#d62728", alpha=0.5, linewidth=1.0,
            label="planner feasible region", zorder=1.5)
    ax.legend(loc="lower left", fontsize=8)

    # Tour segments. Per-leg colour by feasibility: grey for feasible, red for
    # any leg whose segment-to-walls clearance is below the planner's threshold.
    # A red leg is a runner hazard — should never appear, but if it does the
    # plot makes it instantly visible (e.g. someone hand-edited a plan).
    if len(positions) >= 2:
        n_infeas = 0
        for i in range(len(positions) - 1):
            d_wall = min_dist_segment_to_walls(positions[i], positions[i + 1], walls)
            feasible = d_wall >= plan.clearance_mm
            if feasible and poles.size:
                d_pole = min_dist_segment_to_walls(positions[i], positions[i + 1], poles)
                feasible = d_pole >= pole_clearance_mm
            colour = "#888" if feasible else "#d62728"
            lw = 0.8 if feasible else 2.0
            alpha = 0.55 if feasible else 0.95
            ax.plot(positions[i:i + 2, 0], positions[i:i + 2, 1],
                    color=colour, alpha=alpha, linewidth=lw, zorder=2)
            if not feasible:
                n_infeas += 1
        if n_infeas > 0:
            ax.text(0.02, 0.98, f"WARNING: {n_infeas} infeasible leg(s)",
                    transform=ax.transAxes, fontsize=10, color="#d62728",
                    weight="bold", verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="#d62728"))

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


# ─── Tour reordering (post-hoc TSP heuristic) ────────────────────────────────

def total_path_length_mm(positions) -> float:
    """Sum of Euclidean leg lengths along the tour."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def reorder_tour(plan: AcquisitionPlan, arena) -> AcquisitionPlan:
    """Greedy nearest-neighbour reordering of `plan.positions[1:]` while keeping
    `positions[0]` as the start. Feasibility-aware: each candidate next-edge
    must have segment clearance ≥ `plan.clearance_mm` against the wall point
    cloud, otherwise the runner would try to drive through a wall.

    Sample-order plans are essentially random tours (≈ 5–10× optimal length);
    NN heuristic typically gives ~1.25× optimal, a substantial cut in robot
    drive time. If you want even tighter, layer 2-opt on top.

    Each candidate-edge segment-clearance check is cached to keep the worst
    case at O(n²) checks rather than per-step recomputation.
    """
    walls = arena["walls"]
    poles = arena.get("poles", np.empty((0, 2), dtype=np.float32))
    pole_clearance_mm = plan.clearance_mm + float(arena.get("pole_radius_mm", 0.0))
    positions = np.asarray(plan.positions, dtype=np.float64)
    n = positions.shape[0]
    if n <= 2:
        return plan

    feas_cache: dict = {}

    def is_feasible(i: int, j: int) -> bool:
        key = (i, j) if i < j else (j, i)
        if key not in feas_cache:
            d_wall = min_dist_segment_to_walls(positions[i], positions[j], walls)
            ok = d_wall >= plan.clearance_mm
            if ok and poles.size:
                d_pole = min_dist_segment_to_walls(positions[i], positions[j], poles)
                ok = d_pole >= pole_clearance_mm
            feas_cache[key] = ok
        return feas_cache[key]

    visited = [0]
    remaining = set(range(1, n))
    cur = 0

    while remaining:
        # Distances from cur to every remaining node, sorted ascending.
        rem_arr = np.array(sorted(remaining))
        d = np.linalg.norm(positions[rem_arr] - positions[cur], axis=1)
        order = np.argsort(d)

        next_idx = None
        for k in order:
            j = int(rem_arr[k])
            if is_feasible(cur, j):
                next_idx = j
                break

        if next_idx is None:
            # NN got stuck: every remaining waypoint is across a wall from
            # the current node. Truncate rather than accept a wall-crossing
            # edge. (The previous version accepted the nearest infeasible as
            # a "fallback", which let through a leg with 0 mm clearance and
            # the runner would try to drive through the obstacle.) Caller
            # can re-run with a different seed if the loss is significant.
            print(f"  reorder: truncated at {len(visited)}/{n} waypoints "
                  f"({len(remaining)} dropped — no feasible direct neighbour "
                  f"from current node)")
            break

        visited.append(next_idx)
        remaining.remove(next_idx)
        cur = next_idx

    new_positions = [plan.positions[i] for i in visited]
    new_yaws = [plan.yaws_at_position[i] for i in visited]
    return AcquisitionPlan(
        arena_name=plan.arena_name,
        arena_dir=plan.arena_dir,
        positions=new_positions,
        yaws_at_position=new_yaws,
        n_yaws=plan.n_yaws,
        clearance_mm=plan.clearance_mm,
        min_step_mm=plan.min_step_mm,
        min_neighbor_mm=plan.min_neighbor_mm,
        seed=plan.seed,
    )


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def plot_diagnostics(plan: AcquisitionPlan, arena, out_path) -> None:
    """Four-panel diagnostic figure.

    Panel 1 (polar): histogram of all yaws across the plan. With per-position
        random offset and uniform spacing, this should look close to uniform
        on the circle; a clear modal direction means the offset randomisation
        isn't doing what we expect.
    Panel 2: histogram of per-waypoint min-wall-distance. The clearance is
        marked. Tells us how close-wall-heavy the plan actually is.
    Panel 3: histogram of per-waypoint nearest-neighbor distance. Quantifies
        spatial uniformity — long left tail = clustering, long right tail =
        isolated points, tight = even spacing.
    Panel 4: 2D heatmap of waypoint density on the arena. Locates undersampled
        regions visually.
    """
    walls = arena["walls"]
    bounds = arena["bounds"]
    positions = np.array(plan.positions)
    all_yaws = np.array([y for ys in plan.yaws_at_position for y in ys])

    min_wall_dists = np.array([min_dist_point_to_walls(p, walls) for p in positions])

    if len(positions) > 1:
        diffs = positions[:, None, :] - positions[None, :, :]
        d_pair = np.sqrt(np.sum(diffs ** 2, axis=-1))
        np.fill_diagonal(d_pair, np.inf)
        nn_dists = d_pair.min(axis=1)
    else:
        nn_dists = np.array([])

    fig = plt.figure(figsize=(14, 11))

    # Panel 1: polar heading histogram
    ax1 = fig.add_subplot(2, 2, 1, projection="polar")
    n_bins = 36
    counts, edges = np.histogram(np.deg2rad(all_yaws), bins=n_bins,
                                 range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = 2 * np.pi / n_bins
    ax1.bar(centers, counts, width=width, alpha=0.7,
            edgecolor="black", linewidth=0.4, color="#4c72b0")
    ax1.set_theta_zero_location("E")
    ax1.set_theta_direction(1)  # CCW (matches yaw convention: +ccw from +X)
    ax1.set_title(f"Heading distribution\n({len(all_yaws)} yaws)")

    # Panel 2: min-wall-distance histogram per waypoint
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(min_wall_dists, bins=20, edgecolor="black", color="#4c72b0", alpha=0.8)
    ax2.axvline(plan.clearance_mm, color="red", linestyle="--",
                linewidth=1.2, label=f"clearance ({plan.clearance_mm:.0f} mm)")
    ax2.set_xlabel("min wall distance (mm)")
    ax2.set_ylabel("# waypoints")
    ax2.set_title(f"Distance to nearest wall (per waypoint)\n"
                  f"median={np.median(min_wall_dists):.0f} mm, "
                  f"min={min_wall_dists.min():.0f} mm, "
                  f"max={min_wall_dists.max():.0f} mm")
    ax2.legend()

    # Panel 3: nearest-neighbor distance histogram
    ax3 = fig.add_subplot(2, 2, 3)
    if nn_dists.size > 0:
        ax3.hist(nn_dists, bins=20, edgecolor="black", color="#4c72b0", alpha=0.8)
        if plan.min_neighbor_mm > 0:
            ax3.axvline(plan.min_neighbor_mm, color="red", linestyle="--",
                        linewidth=1.2,
                        label=f"min neighbor ({plan.min_neighbor_mm:.0f} mm)")
        ax3.axvline(plan.min_step_mm, color="orange", linestyle=":",
                    linewidth=1.0,
                    label=f"min step ({plan.min_step_mm:.0f} mm)")
        ax3.legend()
        title_extra = (f"\nmedian={np.median(nn_dists):.0f} mm, "
                       f"min={nn_dists.min():.0f} mm, "
                       f"max={nn_dists.max():.0f} mm")
    else:
        title_extra = ""
    ax3.set_xlabel("nearest-neighbor distance (mm)")
    ax3.set_ylabel("# waypoints")
    ax3.set_title("Inter-waypoint spacing" + title_extra)

    # Panel 4: 2D waypoint density heatmap on the arena
    ax4 = fig.add_subplot(2, 2, 4)
    n_grid = 20
    x_edges = np.linspace(bounds["min_x"], bounds["max_x"], n_grid + 1)
    y_edges = np.linspace(bounds["min_y"], bounds["max_y"], n_grid + 1)
    H, _, _ = np.histogram2d(positions[:, 0], positions[:, 1],
                             bins=[x_edges, y_edges])
    # Mask cells with zero count so they're visually distinct from the heatmap
    # max — `hot` maxes out at white, which is indistinguishable from masked
    # cells. Use `viridis` (max = yellow) and a contrasting background colour
    # for "no waypoints here".
    H_plot = np.ma.masked_where(H.T == 0, H.T)
    cmap_density = plt.get_cmap("viridis").copy()
    cmap_density.set_bad(color="#e8e8e8")  # light grey for empty cells
    im = ax4.imshow(
        H_plot,
        extent=(bounds["min_x"], bounds["max_x"],
                bounds["min_y"], bounds["max_y"]),
        origin="lower",
        cmap=cmap_density,
        aspect="equal",
        zorder=2,
    )
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("waypoints per cell")
    ax4.scatter(walls[::20, 0], walls[::20, 1],
                s=0.5, c="black", alpha=0.4, zorder=1)
    hull_path = arena_polygon(walls)
    hull_pts = hull_path.vertices
    ax4.plot(hull_pts[:, 0], hull_pts[:, 1], linestyle="--",
             color="#888", alpha=0.6, linewidth=0.8, zorder=3)
    ax4.set_xlim(bounds["min_x"], bounds["max_x"])
    ax4.set_ylim(bounds["min_y"], bounds["max_y"])
    ax4.set_xlabel("x (mm)")
    ax4.set_ylabel("y (mm)")
    ax4.set_title(f"Waypoint density ({n_grid}×{n_grid} grid, "
                  f"{len(positions)} waypoints)")

    fig.suptitle(f"Acquisition plan diagnostics: {plan.arena_name}  "
                 f"(seed {plan.seed})",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
