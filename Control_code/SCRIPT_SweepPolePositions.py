#!/usr/bin/env python3
"""
SCRIPT_SweepPolePositions.py

Search the arena for pole positions the direct-learning controller can actually
reach, by simulating the real reactive policy from the recorded start poses
against a grid of candidate pole placements.

Motivation
----------
Experiment 1 needs a handful of pole positions (3, at time of writing). Choosing
them by eye risks discovering only after a session of robot time that a position
is unreachable from half the starts, or trivially reachable from all of them.

The simulator can answer the geometric half of that question in under a minute.
So: annotate the arena's WALLS only, then sweep candidate poles through
simulation, and pick the three placements from the resulting map.

What this does and does not tell you
------------------------------------
The sim feeds the controller from `feature_from_geometry` — perfect, noise-free
perception, and kinematic motion without drive error. So a candidate that passes
here is *geometrically* reachable by the reactive rule; it says nothing about
whether the sonar inverse can support the same approach. A position that fails
here, though, will not be rescued by better perception: the controller cannot
get there even when told exactly where everything is. Read this as a filter that
removes bad placements, not a promise about good ones.

To keep the sim's information comparable to sonar's, the geometry front end runs
with HORIZON_MM (default 1000) rather than full sight, matching the range beyond
which the 3-class inverse abstains ("none"). Without that the sim sees the whole
arena and every position looks easy.

Why many seeds rather than a fine grid
--------------------------------------
The controller is stochastic: the curl sign and magnitude are re-rolled on every
wall contact, so the same pole is reached on one seed and missed on the next. A
single rollout per cell is a coin flip, not a verdict. Grid spacing beyond a few
hundred mm buys little — the reachability field is smooth — so the compute goes
into N_SEEDS repeats per (candidate, start) instead, and the score reported is
the fraction of rollouts that reached the pole.

Method
------
1. Load walls from the arena's `arena_features.npz` (poles ignored — the whole
   point is that candidates are synthetic) and the recorded start poses.
2. Build a grid over the arena interior, keeping candidates at least
   MIN_WALL_CLEAR_MM from any wall and MIN_START_DIST_MM from every start (a
   pole sitting on top of a start is reached before the controller does
   anything, which tells us nothing).
3. For each candidate, inject it as the single pole and roll the real
   `run_sim` out from every start, N_SEEDS times each.
4. Report per-candidate outcome counts and the reached fraction.

`run_sim` is called verbatim from SCRIPT_RunDirectPolicy, so the sweep exercises
the same ReactiveController, the same referee, and the same tuned constants the
robot will use. Trajectory plotting is disabled during the sweep (it costs ~40x
the rollout itself) and restored for the shortlist.

Prerequisites
-------------
- A walls-only `arena_features.npz` for ARENA_SOURCE (annotate green wall
  polylines, no blue dabs, then run SCRIPT_BuildArenaGeometry.py).
- `start_poses.json` from SCRIPT_DigitizeStartPoses.py.
No robot and no tracker: this is entirely offline.

Output (under OUT_DIR)
----------------------
    sweep_results.csv    one row per candidate: position, outcome counts,
                         reached fraction, median steps when reached
    sweep_map.png        reachability over the arena, on the arena image, with
                         the start poses drawn (diagnostic)
    pole_positions.png   just the chosen poles + starts -- the figure to carry
                         to the arena when marking tape
    pole_positions.json  chosen positions, their sim scores, range from each
                         start, and the criterion used

OUT_DIR defaults to the arena folder, so these land beside start_poses.json and
the whole Experiment 1 setup stays in one place.
"""

import contextlib
import csv
import io
import itertools
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Settings  ← change these before running
# ══════════════════════════════════════════════════════════════════════════════

# Arena supplying walls + the overlay backdrop. Accepts a snapshot folder, an
# env_* folder, or a parent containing env_* (newest used).
ARENA_SOURCE: str = "TempOutput/StartPositionDigitization"

# Recorded start poses (SCRIPT_DigitizeStartPoses.py).
START_POSES: str = "TempOutput/StartPositionDigitization/start_poses.json"

# Results land beside start_poses.json, at the arena-folder level: the sweep is
# about this room and these starts, and keeping them together means the poses,
# the map and the chosen positions travel as one record.
OUT_DIR: str = "TempOutput/StartPositionDigitization"

# Candidate grid spacing (mm). The reachability field is smooth, so this is
# about resolving the map, not about precision — 500 gives ~40 candidates in
# this arena. Compute scales linearly, so halving it quadruples the runtime.
GRID_MM: float = 250.0

# Candidate rejection. A pole hard against a wall is both hard to place and
# ambiguous for the "class of nearest object" selector; a pole on top of a
# start is reached trivially.
MIN_WALL_CLEAR_MM: float = 300.0
MIN_START_DIST_MM: float = 500.0

# Rollouts per (candidate, start). The controller is stochastic — see the
# module docstring. 10 gives a reachability fraction in 0.1 steps.
N_SEEDS: int = 10

# Vision/sim range horizon (mm) for the geometry front end. 1000 matches the
# range beyond which the 3-class sonar inverse abstains, so the sim's
# information is comparable to sonar's. None = full sight (over-optimistic).
HORIZON_MM: Optional[float] = 1000.0

# ─── Trial selection ─────────────────────────────────────────────────────────
# The design is a set of TRIALS, each one a (start, pole) pair run once per
# perceptual condition. Poles are physical placements: every extra pole costs a
# snapshot / annotate / build / recalibrate cycle, so few poles serving many
# starts is much cheaper than many poles serving few.

N_POLES: int = 2                  # physical placements
MIN_STARTS_PER_POLE: int = 4      # trials each placement must earn

# A pole must be reached on at least this fraction of rollouts. Kept below 1.0
# deliberately: the per-cell figure carries seed noise, and a hard cut at 0.98
# would flip positions in and out of eligibility on that noise alone.
PICK_MIN_FRAC: float = 0.95

# A trial must take at least this many simulated steps. Short approaches yield
# few perceptual samples and cannot discriminate sonar from vision.
MIN_TRIAL_STEPS: int = 40

# Minimum distance between chosen placements (mm). A hard constraint, not a
# tie-break: pole position is not a factor of interest, but each placement costs
# a full snapshot / annotate / build / recalibrate cycle, so two poles a few
# hundred mm apart spend that cycle on geometry you already have. Ranking with
# separation merely as a tie-break produced placements 250 mm apart.
MIN_POLE_SEPARATION_MM: float = 1500.0

# Sensing cone half-angle (deg), matching SCRIPT_RunDirectPolicy.CONE_HALF_DEG.
# Used with HORIZON_MM for the start-visibility test below.
CONE_HALF_DEG: float = 35.0

# A trial is rejected as trivial when the pole is ALREADY VISIBLE from the start
# pose — inside the cone AND within the horizon. Both conditions are required.
# Bearing alone is the wrong test and throws away good trials: a pole 3 deg
# off-axis at 2.5 m is dead ahead yet completely invisible, because the horizon
# is 1 m. Filtering on bearing alone was what made some starts look unpairable.

# ══════════════════════════════════════════════════════════════════════════════


def resolve_env_dir(source: str) -> Path:
    """Snapshot folder holding arena_features.npz / arena.png / meta.json."""
    p = Path(source)
    if not p.exists():
        raise SystemExit(f"ARENA_SOURCE does not exist: {p}")
    if (p / "meta.json").exists() or p.name.startswith("env_"):
        return p
    envs = sorted(d for d in p.iterdir() if d.is_dir() and d.name.startswith("env_"))
    if not envs:
        raise SystemExit(f"{p} holds no meta.json and no env_* folder.")
    return envs[-1]


def load_walls(env_dir: Path) -> Tuple[np.ndarray, float]:
    path = env_dir / "arena_features.npz"
    if not path.exists():
        raise SystemExit(
            f"No arena_features.npz in {env_dir}.\n"
            f"Annotate green wall polylines on arena_{{shark,tiger}}_annotated.png "
            f"and run SCRIPT_BuildArenaGeometry.py first.")
    d = np.load(str(path))
    kind = d["kind"]
    walls = np.column_stack([d["x_mm"][kind == 0],
                             d["y_mm"][kind == 0]]).astype(np.float32)
    if walls.shape[0] == 0:
        raise SystemExit(f"{path} contains no wall points.")
    radius = float(d["pole_radius_mm"]) if "pole_radius_mm" in d.files else 12.5
    n_poles = int((kind == 1).sum())
    if n_poles:
        print(f"  note: {n_poles} real pole(s) in the build are ignored — "
              f"candidates are synthetic.")
    return walls, radius


def build_candidates(walls: np.ndarray, starts: List[dict]) -> np.ndarray:
    """Grid over the arena interior, filtered by wall and start clearance."""
    xs = np.arange(walls[:, 0].min(), walls[:, 0].max() + GRID_MM, GRID_MM)
    ys = np.arange(walls[:, 1].min(), walls[:, 1].max() + GRID_MM, GRID_MM)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    n_grid = pts.shape[0]

    # Wall clearance. Also rejects everything outside the arena, since exterior
    # points are far from walls but the containment test below removes them.
    d_wall = np.hypot(pts[:, 0][:, None] - walls[:, 0][None, :],
                      pts[:, 1][:, None] - walls[:, 1][None, :]).min(axis=1)
    keep = d_wall >= MIN_WALL_CLEAR_MM

    # Containment: a point is inside if wall points surround it. Exterior points
    # see all walls within a limited angular span; interior points see them all
    # the way round. Cheap and robust for a closed point cloud.
    inside = np.zeros(pts.shape[0], dtype=bool)
    for i in np.where(keep)[0]:
        ang = np.degrees(np.arctan2(walls[:, 1] - pts[i, 1],
                                    walls[:, 0] - pts[i, 0]))
        hist, _ = np.histogram(ang, bins=72, range=(-180, 180))
        inside[i] = (hist > 0).all()
    keep &= inside

    if starts:
        sp = np.array([[s["x_mm"], s["y_mm"]] for s in starts], dtype=float)
        d_start = np.hypot(pts[:, 0][:, None] - sp[:, 0][None, :],
                           pts[:, 1][:, None] - sp[:, 1][None, :]).min(axis=1)
        keep &= d_start >= MIN_START_DIST_MM

    print(f"  grid {len(xs)}x{len(ys)} = {n_grid} points → {int(keep.sum())} "
          f"candidates after clearance filters")
    return pts[keep]


def sweep(candidates: np.ndarray, walls: np.ndarray, radius: float,
          starts: List[dict], tmpdir: str) -> List[dict]:
    """Roll the real controller out over every (candidate, start, seed)."""
    import SCRIPT_RunDirectPolicy as RDP

    RDP.GEOM_RANGE_HORIZON_MM = HORIZON_MM
    original_plot = RDP.save_trajectory_plot
    RDP.save_trajectory_plot = lambda *a, **k: None      # ~40x the rollout cost
    P = RDP.ReactiveParams()

    total = len(candidates) * len(starts) * N_SEEDS
    print(f"  {len(candidates)} candidates x {len(starts)} starts x {N_SEEDS} "
          f"seeds = {total} rollouts")

    rows: List[dict] = []
    try:
        for ci, (px, py) in enumerate(candidates):
            geom = {"walls": walls,
                    "poles": np.array([[px, py]], dtype=np.float32),
                    "pole_radius_mm": radius}
            counts: Dict[str, int] = {}
            steps_ok: List[int] = []
            steps_by_start: List[List[int]] = [[] for _ in starts]
            for si, s in enumerate(starts):
                for seed in range(N_SEEDS):
                    RDP.SIM_START_XY_MM = (s["x_mm"], s["y_mm"])
                    RDP.SIM_START_YAW_DEG = s["yaw_deg"]
                    # Deterministic per cell, so a rerun reproduces the map.
                    # Derived from COORDINATES, not the candidate index, so
                    # the same pole keeps its seeds when the grid changes.
                    RDP.SIM_SEED = (abs(int(px)) * 73856093
                                    ^ abs(int(py)) * 19349663
                                    ^ si * 83492791 ^ seed * 2654435761) % (2**31)
                    RDP.SESSION = "sweep"
                    with contextlib.redirect_stdout(io.StringIO()):
                        outcome = RDP.run_sim(geom, P, tmpdir)
                    counts[outcome] = counts.get(outcome, 0) + 1
                    if outcome == "reached_pole":
                        st = count_steps(tmpdir)
                        steps_ok.append(st)
                        steps_by_start[si].append(st)
            n = sum(counts.values())
            per_start = [int(np.median(v)) if v else -1 for v in steps_by_start]
            reached_any = [v for v in per_start if v >= 0]
            rows.append({
                "x_mm": round(float(px), 1), "y_mm": round(float(py), 1),
                "n": n,
                "reached": counts.get("reached_pole", 0),
                "collision": counts.get("collision", 0),
                "jammed": counts.get("jammed", 0),
                "max_steps": counts.get("max_steps", 0),
                "frac_reached": round(counts.get("reached_pole", 0) / n, 3),
                "median_steps": int(np.median(steps_ok)) if steps_ok else -1,
                # Easiest start is what decides whether a cell is trivial.
                "min_start_steps": min(reached_any) if reached_any else -1,
                "steps_by_start": ";".join(str(v) for v in per_start),
            })
            print(f"    [{ci + 1:3d}/{len(candidates)}] "
                  f"({px:7.0f},{py:8.0f})  reached "
                  f"{rows[-1]['frac_reached']:.2f}  "
                  f"median {rows[-1]['median_steps']:3d}  "
                  f"easiest start {rows[-1]['min_start_steps']:3d}  "
                  f"[{rows[-1]['steps_by_start']}]")
    finally:
        RDP.save_trajectory_plot = original_plot
    return rows


def count_steps(tmpdir: str) -> int:
    """Steps in the trajectory just written (header excluded)."""
    try:
        with open(os.path.join(tmpdir, "trajectory.tsv")) as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except OSError:
        return -1


def wrap_deg(d: float) -> float:
    return ((d + 180.0) % 360.0) - 180.0


def visible_from_start(rng_mm: float, bearing_deg: float) -> bool:
    """Would the robot see the pole from its start pose, before moving?

    Requires BOTH: inside the sensing cone, and within the range horizon. A
    pole failing either test is undetectable at step 0, so the controller has
    to search for it — which is what makes a trial worth running.
    """
    in_cone = abs(bearing_deg) <= CONE_HALF_DEG
    in_range = HORIZON_MM is None or rng_mm <= HORIZON_MM
    return in_cone and in_range


def trial_table(rows: List[dict], starts: List[dict]) -> Dict[tuple, dict]:
    """{(pole_x, pole_y): {start_index: {range, bearing, steps}}} for usable trials."""
    out: Dict[tuple, dict] = {}
    for i, s in enumerate(starts):
        for r in rows:
            if r["frac_reached"] < PICK_MIN_FRAC:
                continue
            steps = [int(v) for v in r["steps_by_start"].split(";")][i]
            if steps < MIN_TRIAL_STEPS:
                continue
            dx = r["x_mm"] - s["x_mm"]
            dy = r["y_mm"] - s["y_mm"]
            rng = math.hypot(dx, dy)
            brg = wrap_deg(math.degrees(math.atan2(dy, dx)) - s["yaw_deg"])
            if visible_from_start(rng, brg):
                continue
            out.setdefault((r["x_mm"], r["y_mm"]), {})[s["index"]] = {
                "range_mm": round(rng, 1),
                "bearing_deg": round(brg, 1),
                "sim_steps": steps,
                "hidden_by": ("beyond horizon" if HORIZON_MM is not None
                              and rng > HORIZON_MM else "outside cone"),
            }
    return out


def pick_poles(rows: List[dict], starts: List[dict]) -> Tuple[List[tuple], Dict[tuple, dict]]:
    """Choose N_POLES placements, each earning >= MIN_STARTS_PER_POLE trials.

    Ranked by starts covered overall, then total simulated steps, then mutual
    separation. Coverage first because an unused start is a wasted tape mark;
    separation last because pole position is not a factor of interest, so it is
    a tie-break rather than a goal.
    """
    table = trial_table(rows, starts)
    rich = {p: d for p, d in table.items() if len(d) >= MIN_STARTS_PER_POLE}
    if len(rich) < N_POLES:
        return [], table

    def score(combo):
        cov = len({si for p in combo for si in rich[p]})
        steps = sum(v["sim_steps"] for p in combo for v in rich[p].values())
        sep = (min(math.hypot(a[0] - b[0], a[1] - b[1])
                   for a, b in itertools.combinations(combo, 2))
               if len(combo) > 1 else 0.0)
        return (cov, steps, sep)

    combos = [c for c in itertools.combinations(sorted(rich), N_POLES)
              if len(c) < 2 or min(math.hypot(a[0] - b[0], a[1] - b[1])
                                   for a, b in itertools.combinations(c, 2))
              >= MIN_POLE_SEPARATION_MM]
    if not combos:
        return [], table
    best = max(combos, key=score)
    return sorted(best, key=lambda p: -p[1]), rich


def write_csv(rows: List[dict], out_dir: Path) -> None:
    path = out_dir / "sweep_results.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV  → {path}")


def draw_map(rows: List[dict], env_dir: Path, starts: List[dict],
             picked: List[tuple], out_dir: Path) -> None:
    import SCRIPT_DigitizeStartPoses as SD

    aff = SD.load_affine(env_dir)
    fig, ax = plt.subplots(figsize=(12, 12))
    img_path = env_dir / "arena.png"
    if img_path.exists():
        ax.imshow(plt.imread(str(img_path)))
    ax.set_axis_off()

    cols, rws, fr = [], [], []
    for r in rows:
        c, rr = SD.world_to_px(r["x_mm"], r["y_mm"], aff)
        cols.append(c); rws.append(rr); fr.append(r["frac_reached"])
    sc = ax.scatter(cols, rws, c=fr, cmap="RdYlGn", vmin=0, vmax=1,
                    s=260, edgecolors="black", linewidths=.6, zorder=4)
    cb = fig.colorbar(sc, ax=ax, fraction=.035, pad=.02)
    cb.set_label("fraction of rollouts reaching the pole", fontsize=10)

    for p in starts:
        SD.draw_pose(ax, p["x_mm"], p["y_mm"], p["yaw_deg"], p["index"], aff)
    for i, pole in enumerate(picked, 1):
        c, rr = SD.world_to_px(pole[0], pole[1], aff)
        ax.plot(c, rr, "*", ms=28, color="#0a84ff", mec="white", mew=1.4, zorder=6)
        ax.text(c + 18, rr + 18, f"Q{i}", color="#0a84ff", fontsize=15,
                fontweight="bold", zorder=7)
    ax.set_title(f"Pole reachability — {len(rows)} candidates, "
                 f"{len(starts)} starts x {N_SEEDS} seeds, horizon {HORIZON_MM} mm",
                 fontsize=12)
    path = out_dir / "sweep_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Map  → {path}")


POLE_COLORS = ["#c05cff", "#0a84ff", "#ff9f0a", "#30d158"]


def draw_picks(picked: List[tuple], rich: Dict[tuple, dict], env_dir: Path,
               starts: List[dict], walls: np.ndarray, out_dir: Path) -> None:
    """The design figure: chosen poles, and which starts run against each.

    Separate from the reachability map, which is a diagnostic dense with
    overlapping markers. This one is what you carry to the arena: it shows the
    placements to mark and, as a line per trial, exactly which runs to do.
    """
    import SCRIPT_DigitizeStartPoses as SD

    aff = SD.load_affine(env_dir)
    by_index = {s["index"]: s for s in starts}
    fig, ax = plt.subplots(figsize=(12, 12))
    img_path = env_dir / "arena.png"
    if img_path.exists():
        ax.imshow(plt.imread(str(img_path)))
    ax.set_axis_off()

    wc, wr = SD.world_to_px(walls[:, 0], walls[:, 1], aff)
    ax.plot(wc, wr, ".", ms=0.6, color="#00e5ff", alpha=.30)

    n_trials = 0
    for i, pole in enumerate(picked, 1):
        col = POLE_COLORS[(i - 1) % len(POLE_COLORS)]
        pc, pr = SD.world_to_px(pole[0], pole[1], aff)
        for si, d in sorted(rich[pole].items()):
            s = by_index[si]
            sc_, sr_ = SD.world_to_px(s["x_mm"], s["y_mm"], aff)
            ax.plot([sc_, pc], [sr_, pr], "-", lw=1.6, color=col, alpha=.55,
                    zorder=3)
            n_trials += 1
        ax.plot(pc, pr, "o", ms=22, mfc="none", mew=3.2, color=col, zorder=6)
        ax.plot(pc, pr, "+", ms=16, mew=2.2, color=col, zorder=6)
        ax.annotate(f"Q{i}  ({pole[0]:.0f}, {pole[1]:.0f})\n"
                    f"starts {sorted(rich[pole])}",
                    xy=(pc, pr), xytext=(pc + 34, pr - 34),
                    color=col, fontsize=12, fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col,
                              alpha=.92))
    for s in starts:
        SD.draw_pose(ax, s["x_mm"], s["y_mm"], s["yaw_deg"], s["index"], aff)

    ax.set_title(
        f"Experiment 1 design — {len(picked)} pole placements, {n_trials} trials\n"
        f"each line is one trial (1 sonar run + 1 vision run); "
        f"pole never visible from the start",
        fontsize=12)
    path = out_dir / "pole_positions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Design→ {path}")


def write_picks(picked: List[tuple], rich: Dict[tuple, dict],
                starts: List[dict], out_dir: Path) -> None:
    """The trial list, so the runner iterates a record rather than a copy-paste."""
    trials = []
    poles = []
    for i, pole in enumerate(picked, 1):
        label = f"Q{i}"
        poles.append({"label": label, "x_mm": pole[0], "y_mm": pole[1],
                      "starts": sorted(rich[pole])})
        for si, d in sorted(rich[pole].items()):
            trials.append({"pole": label, "pole_x_mm": pole[0],
                           "pole_y_mm": pole[1], "start": si, **d})
    payload = {
        "poles": poles,
        "trials": trials,
        "n_trials": len(trials),
        "runs_per_condition": len(trials),
        "criterion": {
            "min_frac_reached": PICK_MIN_FRAC,
            "min_trial_steps": MIN_TRIAL_STEPS,
            "min_starts_per_pole": MIN_STARTS_PER_POLE,
            "not_visible_from_start": (
                f"rejected when inside +-{CONE_HALF_DEG:.0f} deg cone AND "
                f"within {HORIZON_MM:.0f} mm horizon (both required)"),
            "min_pole_separation_mm": MIN_POLE_SEPARATION_MM,
            "chosen_by": ("starts covered, then total sim steps, then separation; "
                          "separation also a hard floor"),
        },
        "sim_settings": {"n_seeds": N_SEEDS, "horizon_mm": HORIZON_MM,
                         "grid_mm": GRID_MM,
                         "min_wall_clear_mm": MIN_WALL_CLEAR_MM,
                         "min_start_dist_mm": MIN_START_DIST_MM},
        "caveat": ("Simulated with perfect perception and no drive error. These "
                   "are geometrically reachable by the reactive rule; they are "
                   "not a prediction about sonar. The referee still needs each "
                   "pole's ACTUAL position, so snapshot, annotate and rebuild "
                   "the arena once each pole is physically placed."),
    }
    path = out_dir / "pole_positions.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"JSON  → {path}")


def main() -> None:
    import SCRIPT_DigitizeStartPoses as SD

    env_dir = resolve_env_dir(ARENA_SOURCE)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Arena       : {env_dir}")
    walls, radius = load_walls(env_dir)
    starts = SD.load_start_poses(START_POSES)
    print(f"Walls       : {walls.shape[0]} pts   pole radius {radius} mm")
    print(f"Starts      : {[s['index'] for s in starts]}")
    print(f"Horizon     : {HORIZON_MM} mm\n")

    candidates = build_candidates(walls, starts)
    if not len(candidates):
        raise SystemExit("No candidates survived the clearance filters — "
                         "loosen MIN_WALL_CLEAR_MM / MIN_START_DIST_MM.")

    tmpdir = tempfile.mkdtemp(prefix="polesweep_")
    try:
        rows = sweep(candidates, walls, radius, starts, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    rows.sort(key=lambda r: (-r["frac_reached"], -r["median_steps"]))
    write_csv(rows, out_dir)

    print(f"\nTop candidates by reached fraction:")
    print(f"{'x_mm':>9} {'y_mm':>10} {'frac':>6} {'steps':>6} "
          f"{'reach':>6} {'coll':>5} {'jam':>4} {'t/out':>6}")
    for r in rows[:12]:
        print(f"{r['x_mm']:>9.0f} {r['y_mm']:>10.0f} {r['frac_reached']:>6.2f} "
              f"{r['median_steps']:>6d} {r['reached']:>6d} {r['collision']:>5d} "
              f"{r['jammed']:>4d} {r['max_steps']:>6d}")

    picked, rich = pick_poles(rows, starts)
    if not picked:
        raise SystemExit(
            f"\nNo set of {N_POLES} poles each serving >= {MIN_STARTS_PER_POLE} "
            f"starts.\nLoosen MIN_TRIAL_STEPS ({MIN_TRIAL_STEPS}), lower "
            f"MIN_STARTS_PER_POLE ({MIN_STARTS_PER_POLE}), reduce "
            f"MIN_POLE_SEPARATION_MM ({MIN_POLE_SEPARATION_MM:.0f}), or allow "
            f"more poles (N_POLES={N_POLES}).")

    n_trials = sum(len(rich[p_]) for p_ in picked)
    cov = sorted({si for p_ in picked for si in rich[p_]})
    sep = (min(math.hypot(a_[0] - b_[0], a_[1] - b_[1])
               for a_, b_ in itertools.combinations(picked, 2))
           if len(picked) > 1 else 0.0)
    print(f"\n{len(picked)} pole placements, {n_trials} trials, "
          f"starts covered {cov}, separation {sep:.0f} mm")
    print("(a trial = one start x one pole, run once per perceptual condition)\n")

    total = 0
    for i, pole in enumerate(picked, 1):
        print(f"  Q{i} ({pole[0]:7.0f}, {pole[1]:8.0f})  "
              f"-> place pole, snapshot, annotate, build, recalibrate")
        for si, d in sorted(rich[pole].items()):
            total += d["sim_steps"]
            print(f"       S{si}: {d['range_mm']:6.0f} mm, "
                  f"{d['bearing_deg']:+5.0f} deg off heading, "
                  f"~{d['sim_steps']:3d} steps   ({d['hidden_by']})")
    print(f"\n  {n_trials} trials x 2 conditions = {n_trials * 2} robot runs")
    print(f"  ~{total} steps per condition")

    write_picks(picked, rich, starts, out_dir)
    draw_picks(picked, rich, env_dir, starts, walls, out_dir)
    draw_map(rows, env_dir, starts, picked, out_dir)

    print("\nReminder: the sim has perfect perception and no drive error. These "
          "positions are geometrically reachable by the reactive rule; sonar is "
          "the real test. Once the poles are placed, snapshot + annotate + "
          "rebuild so the referee has their ACTUAL positions.")


if __name__ == "__main__":
    main()
