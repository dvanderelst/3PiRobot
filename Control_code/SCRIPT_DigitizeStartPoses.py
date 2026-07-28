#!/usr/bin/env python3
"""
SCRIPT_DigitizeStartPoses.py

Record the robot's marked start poses (position + heading) by placing the robot
on each floor mark and reading the overhead tracker, then overlay the result on
an arena snapshot for visual confirmation.

Motivation
----------
Experiment 1 runs the direct-learning policy from a fixed set of start poses,
marked on the arena floor with tape. Two things need those poses in world mm:

  - the sim pre-flight, which rolls the reactive controller out from each start
    to check the pole is reachable in principle before spending robot time;
  - the paper, which should report where the runs started.

We read them from the tracker rather than digitizing them off an image, because
the tracker is the same pose source the real runs use. Whatever bias the marker
pipeline carries is then shared between these recorded starts and the live runs,
instead of being a fresh error term between two different measurement paths.

Scope: poses only
-----------------
This script records where the robot starts. It computes no distances to walls
or poles, and reads no arena geometry.

That is deliberate. The arena is re-laid between conditions: the pole moves by
design, and the wall layout has changed too (a snapshot from 2026-06-12 overlaid
on one from 2026-07-28 shows walls that are no longer there). Any clearance
measured against stale geometry would describe an arena that no longer exists.

Start poses, by contrast, are fixed by tape on the floor and outlive those
changes. So they are recorded once, here, and the geometry-dependent statistics
(distance to the pole, clearance to the nearest wall, whether the pole starts
inside the sonar cone) are computed later, per pole placement, against that
placement's own annotated arena_features.npz.

Procedure
---------
For each start, the script prompts you to place the robot on the mark, then
reads a settled pose (`wait_for_stable_pose`, the same settle used in the
control loop). It repeats the read N_READS times and reports the spread across
reads, so a mark sitting in a noisy corner of the tracker's coverage is visible
immediately rather than after the fact. You confirm, retry, or skip each start
before moving to the next.

The recorded pose is the median across reads. Position is the robot centre as
the tracker reports it; yaw follows the project convention (degrees, CCW-
positive, 0 deg = +x, wrapped to [-180, 180)) — the same convention `run_sim`
integrates with (x += drive*cos(yaw), y += drive*sin(yaw)).

Re-recording one mark
---------------------
Runs merge into whatever is already in OUT_PATH rather than replacing it, so a
single mark can be redone without walking the whole set again:

    REDO_INDICES = [4]      # prompt only for mark 4; 1,2,3,5 carry through

Skipping at the prompt keeps the existing value rather than dropping it, so
REDO_INDICES = None also works — just skip past the marks you are not redoing.
When a mark already has a pose, the prompt shows it, and a fresh read is
reported as a delta against it (how far the mark moved, how much it turned),
which is the quickest way to tell a genuine re-placement from tracker noise.
Each write first copies the previous file to a timestamped sibling
(start_poses.<YYYY-MM-DDTHHMMSS>.json), so successive runs cannot overwrite the
version you wanted to roll back to.

Prerequisites
-------------
The PyLorex tracking server must be running on its own machine and serving the
current calibration. The robot must be powered and its top-plate marker visible
to at least one camera. If every read comes back empty, check the server first.

Input
-----
A snapshot folder supplying the overlay backdrop and the world->pixel affine:

    arena.png     stitched colour top-down warp
    meta.json     arena_bounds_mm + map_mm_per_px

The snapshot may be out of date — it is there to show each pose against the
room's immovable features (walls, doorways, fixtures), which is enough to
recognise where a mark is. Nothing is measured from it.

Output
------
    start_poses.json    {"poses": [{index, x_mm, y_mm, yaw_deg, spread_*}, ...],
                         plus provenance: robot id, snapshot used, read settings}
    start_poses.png     overlay of the recorded poses on the snapshot

Both land beside the snapshot they were recorded against, so the poses and the
image they were checked on travel together.

Status of this output
---------------------
Planning artifact, deliberately untracked. It feeds the sim pre-flight and tells
you where to place the robot, but it is not the experimental record: each real
run logs its own true start pose as step 0 of trajectory.tsv, and that is the
pose to report, since the robot never lands exactly on the tape. Re-run this
script whenever the marks are re-laid.
"""

import json
import math
import os
import shutil
from datetime import datetime
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

ROBOT_ID: int = 1

# How many start poses to record. The skip/quit flow lets you stop early.
N_STARTS: int = 5

# Settled tracker reads per start. The median is recorded; the spread across
# reads is reported and stored. 1 is enough to get a pose; 3 tells you whether
# to trust it. Each read costs up to YAW_STABLE_TIMEOUT_S seconds.
N_READS: int = 3

# Which marks to (re)record this run. None = all of 1..N_STARTS. Set to a list
# to redo only those, e.g. [4] to re-do the 4th and leave the rest untouched:
# any pose already in OUT_PATH that is not listed here is carried through
# unchanged. Skipping at the prompt also keeps the existing value.
REDO_INDICES: Optional[List[int]] = [4]


# Snapshot supplying the overlay backdrop and the world<->pixel affine. Accepts
# a snapshot folder directly (one holding meta.json), an env_* folder, or a
# parent containing env_* folders (newest is used). May be out of date.
ENV_SOURCE: str = "TempOutput/StartPositionDigitization"

IMAGE_NAME: str = "arena.png"

# Where the result lands — beside the snapshot it was recorded against.
OUT_PATH: str = "TempOutput/StartPositionDigitization/start_poses.json"

# Pose-settle tolerances. Defaults match the control loop in
# SCRIPT_RunDirectPolicy so a start recorded here is a pose a run could
# actually settle on. See Library/TrackerNav.wait_for_stable_pose.
YAW_STABLE_TOL_DEG: float = 2.0
YAW_STABLE_POS_TOL_MM: float = 8.0
YAW_STABLE_N_CONSEC: int = 4
YAW_STABLE_POLL_S: float = 0.1
YAW_STABLE_TIMEOUT_S: float = 8.0

# Robot radius (mm) — drawn as the footprint circle on the overlay.
ROBOT_RADIUS_MM: float = 48.0

# ══════════════════════════════════════════════════════════════════════════════

ARROW_MM: float = 300.0          # drawn heading arrow length (display only)


# ─── Frame helpers ───────────────────────────────────────────────────────────

def resolve_env_dir(source: str) -> Path:
    """Return the snapshot folder supplying the overlay backdrop and affine.

    Accepts a snapshot folder directly (identified by holding meta.json — a
    bare capture dropped into TempOutput is not named env_*), an env_* folder,
    or a parent containing env_* folders, in which case the newest is used.
    """
    p = Path(source)
    if not p.exists():
        raise SystemExit(f"ENV_SOURCE does not exist: {p}")
    if (p / "meta.json").exists() or p.name.startswith("env_"):
        return p
    envs = sorted(d for d in p.iterdir() if d.is_dir() and d.name.startswith("env_"))
    if not envs:
        raise SystemExit(f"{p} holds no meta.json and no env_* folder.")
    return envs[-1]


def load_affine(env_dir: Path) -> Dict[str, float]:
    """Read the arena-grid affine (bounds + scale) from meta.json."""
    with open(env_dir / "meta.json") as fh:
        meta = json.load(fh)
    b = meta["arena_bounds_mm"]
    return {
        "min_x": float(b["min_x"]), "max_x": float(b["max_x"]),
        "min_y": float(b["min_y"]), "max_y": float(b["max_y"]),
        "mm_per_px": float(meta["map_mm_per_px"]),
    }


def world_to_px(x, y, aff: Dict[str, float]):
    """World mm at z = 0 → arena-image pixel. Drawing only.

    Inverse of the arena-grid step in SCRIPT_BuildArenaGeometry. The ray-plane
    correction that follows it there is a no-op on the floor plane, which is
    where both the tape marks and the robot's footprint sit.
    """
    s = aff["mm_per_px"]
    col = (x - aff["min_x"] - 0.5 * s) / s
    row = (aff["max_y"] + 0.5 * s - y) / s
    return col, row


def wrap_deg(deg: float) -> float:
    """Wrap to [-180, 180), the convention run_sim integrates yaw in."""
    return ((deg + 180.0) % 360.0) - 180.0


def circular_median_deg(angles: List[float]) -> float:
    """Median of angles, taken about their circular mean so the wrap point
    cannot split a tight cluster (e.g. 179 and -179 are 2 deg apart, but a
    plain median of [179.5, -179.5, 180] returns 179.5 instead of -180)."""
    a = np.radians(np.asarray(angles, dtype=float))
    centre = math.atan2(float(np.sin(a).mean()), float(np.cos(a).mean()))
    dev = np.angle(np.exp(1j * (a - centre)))
    return wrap_deg(math.degrees(centre + float(np.median(dev))))


def angular_spread_deg(angles: List[float]) -> float:
    """Peak-to-peak angular spread, wrap-safe."""
    if len(angles) < 2:
        return 0.0
    a = np.radians(np.asarray(angles, dtype=float))
    centre = math.atan2(float(np.sin(a).mean()), float(np.cos(a).mean()))
    dev = np.degrees(np.angle(np.exp(1j * (a - centre))))
    return float(dev.max() - dev.min())


# ─── Tracker capture ─────────────────────────────────────────────────────────

def read_settled(tracker, n_reads: int) -> List[Tuple[float, float, float]]:
    """Take up to `n_reads` settled poses of a stationary robot.

    No `prior_pose` is passed: the robot is placed by hand and is not expected
    to move between reads, so the motion-required gate would never be satisfied.
    """
    from Library.TrackerNav import wait_for_stable_pose

    reads: List[Tuple[float, float, float]] = []
    for i in range(n_reads):
        pose = wait_for_stable_pose(
            tracker, ROBOT_ID,
            yaw_tol_deg=YAW_STABLE_TOL_DEG, pos_tol_mm=YAW_STABLE_POS_TOL_MM,
            n_consec=YAW_STABLE_N_CONSEC, poll_s=YAW_STABLE_POLL_S,
            timeout_s=YAW_STABLE_TIMEOUT_S, strict_motion=False, verbose=False)
        if pose is None:
            print(f"    read {i + 1}/{n_reads}: no settled pose "
                  f"(marker not seen, or tracker not serving)")
            continue
        reads.append((float(pose[0]), float(pose[1]), float(pose[2])))
        print(f"    read {i + 1}/{n_reads}: x={pose[0]:8.1f}  y={pose[1]:9.1f}  "
              f"yaw={pose[2]:+7.2f}")
    return reads


def summarize(reads: List[Tuple[float, float, float]]) -> dict:
    """Median pose plus the peak-to-peak spread across reads."""
    xs = [r[0] for r in reads]
    ys = [r[1] for r in reads]
    yaws = [r[2] for r in reads]
    return {
        "x_mm": round(float(np.median(xs)), 1),
        "y_mm": round(float(np.median(ys)), 1),
        "yaw_deg": round(circular_median_deg(yaws), 2),
        "n_reads": len(reads),
        "spread_x_mm": round(float(np.ptp(xs)), 1),
        "spread_y_mm": round(float(np.ptp(ys)), 1),
        "spread_yaw_deg": round(angular_spread_deg(yaws), 2),
    }


def _fmt_pose(p: dict) -> str:
    return (f"x={p['x_mm']:.1f}  y={p['y_mm']:.1f}  yaw={p['yaw_deg']:+.2f}")


def capture_one(tracker, index: int, existing: Optional[dict]) -> Optional[dict]:
    """Prompt, read, report, confirm one mark.

    Returns the new pose, or None to leave `existing` in place. Skipping never
    discards: an already-recorded pose survives a skip, so a redo run can walk
    past the marks it is not re-doing.
    """
    held = f"  (currently {_fmt_pose(existing)})" if existing else ""
    keeps = "keeps current" if existing else "leaves unrecorded"
    while True:
        ans = input(f"\nStart {index}/{N_STARTS}{held}\n"
                    f"  place the robot on the mark, then [Enter] to read  "
                    f"([s]kip {keeps}, [q]uit): ").strip().lower()
        if ans == "q":
            raise KeyboardInterrupt
        if ans == "s":
            print(f"  skipped — {keeps}.")
            return None

        print(f"  reading {N_READS} settled pose(s)...")
        reads = read_settled(tracker, N_READS)
        if not reads:
            print("  No pose at all. Is the PyLorex server running and the "
                  "marker visible? [Enter] to retry, [s]kip, [q]uit.")
            continue

        pose = summarize(reads)
        print(f"\n  -> x={pose['x_mm']:8.1f}  y={pose['y_mm']:9.1f}  "
              f"yaw={pose['yaw_deg']:+7.2f}   (median of {pose['n_reads']})")
        print(f"     spread across reads: dx={pose['spread_x_mm']:.1f} mm  "
              f"dy={pose['spread_y_mm']:.1f} mm  dyaw={pose['spread_yaw_deg']:.2f} deg")
        if existing:
            d = math.hypot(pose["x_mm"] - existing["x_mm"],
                           pose["y_mm"] - existing["y_mm"])
            dyaw = wrap_deg(pose["yaw_deg"] - existing["yaw_deg"])
            print(f"     vs current: moved {d:.0f} mm, turned {dyaw:+.2f} deg")

        ans = input("  Accept? [Enter]=yes, [r]etry, [s]kip, [q]uit: ").strip().lower()
        if ans == "q":
            raise KeyboardInterrupt
        if ans == "s":
            print(f"  skipped — {keeps}.")
            return None
        if ans == "r":
            continue
        pose["index"] = index
        return pose


# ─── Reporting + IO ──────────────────────────────────────────────────────────

def draw_pose(ax, x: float, y: float, yaw: float, idx: int,
              aff: Dict[str, float]) -> None:
    """Draw one recorded pose: footprint circle, heading arrow, index label."""
    s = aff["mm_per_px"]
    col, row = world_to_px(x, y, aff)
    tcol, trow = world_to_px(x + ARROW_MM * math.cos(math.radians(yaw)),
                             y + ARROW_MM * math.sin(math.radians(yaw)), aff)
    ax.add_patch(plt.Circle((col, row), ROBOT_RADIUS_MM / s, fill=False,
                            color="#ff3b30", lw=1.6, zorder=5))
    ax.annotate("", xy=(tcol, trow), xytext=(col, row),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#ff3b30"),
                zorder=6)
    ax.text(col + 14, row - 14, str(idx), color="#ff3b30", fontsize=13,
            fontweight="bold", zorder=7)


def summary_table(poses: List[dict], updated: Optional[set] = None) -> None:
    updated = updated or set()
    print(f"\n{'#':>2}  {'x_mm':>8} {'y_mm':>9} {'yaw_deg':>8} "
          f"{'dx':>5} {'dy':>5} {'dyaw':>6}  status")
    for p in poses:
        status = "re-recorded" if p["index"] in updated else "kept"
        print(f"{p['index']:>2}  {p['x_mm']:>8.1f} {p['y_mm']:>9.1f} "
              f"{p['yaw_deg']:>8.2f} {p['spread_x_mm']:>5.1f} "
              f"{p['spread_y_mm']:>5.1f} {p['spread_yaw_deg']:>6.2f}  {status}")


def save(poses: List[dict], env_dir: Path, aff: Dict[str, float]) -> None:
    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # A redo run rewrites a file that already holds good poses, so keep the
    # previous version. The backup name is timestamped rather than a single
    # .bak slot: with one slot, two writes in a row overwrite the backup with
    # the very data you wanted to roll back past, which is exactly how the
    # first set of recorded poses was lost on 2026-07-28.
    if out_path.exists():
        stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        backup = out_path.with_name(f"{out_path.stem}.{stamp}.json")
        shutil.copy2(out_path, backup)
        print(f"Previous version backed up → {backup}")

    payload = {
        "poses": poses,
        "convention": ("yaw in degrees, CCW-positive, 0 = +x, wrapped to "
                       "[-180, 180); position is the robot centre in world mm, "
                       "as reported by the overhead tracker"),
        "source": "overhead tracker (wait_for_stable_pose)",
        "robot_id": ROBOT_ID,
        "n_reads_per_pose": N_READS,
        "settle": {
            "yaw_tol_deg": YAW_STABLE_TOL_DEG,
            "pos_tol_mm": YAW_STABLE_POS_TOL_MM,
            "n_consec": YAW_STABLE_N_CONSEC,
            "timeout_s": YAW_STABLE_TIMEOUT_S,
        },
        "overlay_snapshot": str(env_dir),
        "recorded": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {len(poses)} start poses → {out_path}")

    img_path = env_dir / IMAGE_NAME
    if not img_path.exists():
        print(f"No {IMAGE_NAME} in {env_dir} — skipping overlay.")
        return
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(plt.imread(str(img_path)))
    ax.set_axis_off()
    for p in poses:
        draw_pose(ax, p["x_mm"], p["y_mm"], p["yaw_deg"], p["index"], aff)
    ax.set_title(f"Tracker-recorded start poses ({len(poses)}) "
                 f"on {env_dir.name}", fontsize=11)
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Overlay → {png_path}")
    print("Check the overlay: each arrow should sit on its tape mark. If the "
          "arrows are offset, the tracker and the snapshot disagree on the frame.")


def load_start_poses(path: str = OUT_PATH) -> List[dict]:
    """Read back the recorded poses. Used by the sim pre-flight sweep."""
    with open(path) as fh:
        return json.load(fh)["poses"]


def load_existing(path: str = OUT_PATH) -> Dict[int, dict]:
    """Previously recorded poses keyed by index, or {} if none on disk."""
    if not Path(path).exists():
        return {}
    try:
        return {int(p["index"]): p for p in load_start_poses(path)}
    except Exception as e:
        print(f"Could not read existing poses from {path} ({e}) — starting fresh.")
        return {}


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    from Library import LorexTracker

    env_dir = resolve_env_dir(ENV_SOURCE)
    aff = load_affine(env_dir)

    existing = load_existing()
    targets = list(range(1, N_STARTS + 1)) if REDO_INDICES is None \
        else sorted(set(REDO_INDICES))
    stray = [i for i in targets if not 1 <= i <= N_STARTS]
    if stray:
        raise SystemExit(f"REDO_INDICES {stray} outside 1..{N_STARTS}.")

    print(f"Overlay backdrop : {env_dir / IMAGE_NAME}")
    print(f"Output           : {OUT_PATH}")
    print(f"Robot            : {ROBOT_ID}   reads per pose: {N_READS}")
    if existing:
        print(f"On disk          : {len(existing)} pose(s) "
              f"{sorted(existing)} — untouched unless re-recorded")
    print(f"Recording        : {targets}")
    print("\nThe PyLorex server must be running and serving current calibration.")

    tracker = LorexTracker.LorexTracker()

    merged = dict(existing)
    updated: set = set()
    try:
        for i in targets:
            pose = capture_one(tracker, i, merged.get(i))
            if pose is not None:
                merged[i] = pose
                updated.add(i)
    except KeyboardInterrupt:
        print("\nStopped early — poses already on disk are still kept.")

    if not merged:
        raise SystemExit("No start poses recorded — nothing written.")
    if not updated:
        print("\nNothing re-recorded; the file on disk is already current. "
              "Re-saving anyway to refresh the overlay.")

    poses = [merged[i] for i in sorted(merged)]
    missing = [i for i in range(1, N_STARTS + 1) if i not in merged]
    if missing:
        print(f"\nNote: marks {missing} have no pose yet — set "
              f"REDO_INDICES = {missing} to fill them in.")

    summary_table(poses, updated)
    save(poses, env_dir, aff)


if __name__ == "__main__":
    main()
