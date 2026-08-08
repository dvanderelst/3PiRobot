#!/usr/bin/env python3
"""
SCRIPT_AnalysePathRun.py

Two questions this answers, both of which cost a session to work out by hand
on 2026-08-07:

  1. **Is a path flyable?**  A path is only usable if its clearance exceeds the
     robot's tracking error. Path02 failed this and nobody noticed until the
     robot hit a block on lap 3: 155 mm minimum clearance against an 85 mm
     robot radius leaves 70 mm of margin, while measured tracking error is
     119 mm mean / 266 mm at the 90th percentile. Half the path was tighter
     than the robot's routine error, so collisions were structural.
     Run this on a candidate path BEFORE training against it.

  2. **Is the robot's motion calibrated?**  Per-step yaw residual (commanded
     rotation vs tracker-measured heading change) is the sensitive indicator.
     Drive-distance scale comes free from the same poses.

There is a trade-off the clearance number alone hides: informative perception
comes from being NEAR walls, so clearance and sonar usefulness pull against
each other. Path01 had 249 mm clearance and 26% informative steps; Path02 had
155 mm and 49%. Both are reported below so the compromise is explicit.

Usage:
    python3 SCRIPT_AnalysePathRun.py                 # uses the constants below
    python3 SCRIPT_AnalysePathRun.py Path02          # path geometry only
    python3 SCRIPT_AnalysePathRun.py Path02 default_Path02_run02
"""

import csv
import glob
import json
import os
import sys

import numpy as np

from Library import Settings as _settings
from Library.LocalFeature import true_local_feature

# ── Config ────────────────────────────────────────────────────────────────────
ARENA        = "Path02"                  # sub-folder under TargetArenas/
SESSION      = "default_Path02_run02"    # sub-folder under PolicyRuns/; "" = none
ROBOT_RADIUS_MM = 85.0
CONE_HALF_DEG   = 35.0
MAX_RANGE_MM    = 1000.0                 # the inverse's abstention horizon
STEP_MM         = 150.0                  # Config.fixed_drive_mm
DENSIFY         = 12                     # sub-samples per path segment

TARGET_ROOT = "TargetArenas"
RUNS_ROOT   = "PolicyRuns"


def load_arena(arena):
    """Walls+poles point cloud and the pole radius, from the newest env snapshot."""
    envs = sorted(glob.glob(os.path.join(TARGET_ROOT, arena, "env_*", "arena_features.npz")))
    if not envs:
        raise SystemExit(f"no arena_features.npz under {TARGET_ROOT}/{arena}/env_*")
    d = np.load(envs[-1])
    kind = d["kind"]
    walls = np.c_[d["x_mm"], d["y_mm"]][kind == 0]
    poles = np.c_[d["x_mm"], d["y_mm"]][kind == 1]
    radius = float(d["pole_radius_mm"]) if "pole_radius_mm" in d else 12.5
    return np.c_[d["x_mm"], d["y_mm"]], walls, poles, radius, envs[-1]


def load_path(arena):
    hits = glob.glob(os.path.join(TARGET_ROOT, arena, "**", "target_path.json"), recursive=True)
    if not hits:
        raise SystemExit(f"no target_path.json under {TARGET_ROOT}/{arena}")
    P = json.load(open(hits[0]))
    return np.array([[w["x_mm"], w["y_mm"]] for w in P["waypoints"]], dtype=float)


def densify(wp, n=DENSIFY):
    """Closed polyline -> dense sample points plus the heading along each."""
    pts, hdg = [], []
    for i in range(len(wp)):
        a, b = wp[i], wp[(i + 1) % len(wp)]
        h = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append(a + t * (b - a))
            hdg.append(h)
    return np.array(pts), np.array(hdg)


def main(arena=ARENA, session=SESSION):
    pts, walls, poles, pole_r, src = load_arena(arena)
    wp = load_path(arena)
    seg, hdg = densify(wp)

    length = sum(np.hypot(*(wp[(i + 1) % len(wp)] - wp[i]))
                 for i in range(len(wp)))
    print(f"\n=== PATH: {arena} ===")
    print(f"  geometry from {src}")
    print(f"  {len(wp)} waypoints, {length:.0f} mm, ~{length / STEP_MM:.0f} steps per lap "
          f"at {STEP_MM:.0f} mm")

    # ── clearance ────────────────────────────────────────────────────────────
    clr = np.array([np.min(np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])) for p in seg])
    margin = clr - ROBOT_RADIUS_MM
    print(f"\n  clearance (centreline -> nearest obstacle):")
    print(f"    min {clr.min():.0f}   5th {np.percentile(clr, 5):.0f}   "
          f"median {np.median(clr):.0f}   max {clr.max():.0f} mm")
    print(f"    usable margin after {ROBOT_RADIUS_MM:.0f} mm robot radius: "
          f"min {margin.min():.0f} mm")

    # ── informative perception ───────────────────────────────────────────────
    geom = {"walls": walls, "poles": poles, "pole_radius_mm": pole_r}
    cls = np.array([true_local_feature(p[0], p[1], h, geom, CONE_HALF_DEG,
                                       max_range_mm=MAX_RANGE_MM)[0]
                    for p, h in zip(seg, hdg)])
    n = len(cls)
    print(f"\n  perception along the path -- TRUE GEOMETRY, facing forward, cls != none:")
    print(f"    wall {100 * (cls == 0).mean():.1f}%   pole {100 * (cls == 1).mean():.1f}%   "
          f"none {100 * (cls == 2).mean():.1f}%   -> informative "
          f"{100 * (cls != 2).mean():.1f}%")
    print(f"    CAUTION: this is NOT the same metric as the 26% (Path01) / 49% (Path02)")
    print(f"    figures in handoff.md. Those came from a 2026-08-07 measurement whose")
    print(f"    definition was not recorded -- it reported 73.5% wall yet 49% informative,")
    print(f"    so 'informative' there was a SUBSET of wall hits, not simply 'not none'.")
    print(f"    Compare paths using THIS number consistently; don't mix the two.")
    print(f"    The trade-off still holds either way: informative perception comes from")
    print(f"    being near walls, which is exactly what costs clearance.")

    if not session:
        print("\n  no run session given -- skipping tracking/calibration analysis\n")
        return

    # ── the run ──────────────────────────────────────────────────────────────
    mpath = os.path.join(RUNS_ROOT, session, "step_metrics.tsv")
    if not os.path.exists(mpath):
        print(f"\n  {mpath} not found -- run crashed before the main loop?\n")
        return
    rows = list(csv.DictReader(open(mpath), delimiter="\t"))
    g = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan
                            for r in rows])
    x, y, yaw, rot = g("x_mm"), g("y_mm"), g("yaw_deg"), g("rot_deg")

    print(f"\n=== RUN: {session} ===")
    print(f"  {len(x)} steps  (~{len(x) / max(1, length / STEP_MM):.1f} laps)")

    dev = np.array([np.min(np.hypot(seg[:, 0] - a, seg[:, 1] - b)) for a, b in zip(x, y)])
    p90 = np.percentile(dev, 90)
    print(f"\n  tracking error (robot -> intended path):")
    print(f"    mean {dev.mean():.0f}   median {np.median(dev):.0f}   "
          f"90th {p90:.0f}   max {dev.max():.0f} mm")

    # The verdict this script exists for.
    unsafe = (margin < p90).mean()
    print(f"\n  >>> {100 * unsafe:.0f}% of the path has less margin than the robot's "
          f"90th-percentile\n      tracking error ({p90:.0f} mm). "
          f"{'COLLISIONS ARE STRUCTURAL.' if unsafe > 0.1 else 'Path has adequate room.'}")
    print(f"      For this run, a safe path needs min clearance >= "
          f"{p90 + ROBOT_RADIUS_MM:.0f} mm.")

    # ── motion calibration ───────────────────────────────────────────────────
    wrap = lambda a: (a + 180) % 360 - 180
    res = wrap(np.diff(yaw)) - rot[:-1]
    dist = np.hypot(np.diff(x), np.diff(y))
    ok = np.isfinite(res)
    res, dist = res[ok], dist[ok]
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    keep = np.abs(res - np.median(res)) <= 3 * mad

    print(f"\n  motion calibration (commanded rotation vs tracker heading change):")
    print(f"    yaw residual {res[keep].mean():+.2f} +/- "
          f"{res[keep].std(ddof=1) / np.sqrt(keep.sum()):.2f} deg/step  "
          f"(n={keep.sum()}, {(~keep).sum()} outliers dropped)")
    print(f"    drive scale  {dist.mean() / STEP_MM:.3f}  "
          f"({dist.mean():.1f} mm actual vs {STEP_MM:.0f} commanded)")

    # Adjacent equal-and-opposite residuals are ONE bad pose read, not two bad
    # steps: a wrong yaw[i] biases the step before and after equally and
    # oppositely. Worth naming, or you chase a control fault that isn't there.
    bad = np.where(~keep)[0]
    pairs = [(i, i + 1) for i in bad if i + 1 in bad and res[i] * res[i + 1] < 0
             and abs(res[i] + res[i + 1]) < 0.4 * abs(res[i])]
    if pairs:
        txt = ", ".join(f"steps {int(i)}/{int(j)}" for i, j in pairs)
        print(f"    NOTE {len(pairs)} adjacent equal-and-opposite outlier pair(s) "
              f"({txt}) -> single bad pose read, not a motion fault")

    cpath = os.path.join(RUNS_ROOT, session, "crashes.tsv")
    if os.path.exists(cpath):
        crashes = list(csv.DictReader(open(cpath), delimiter="\t"))
        print(f"\n  CRASHES: {len(crashes)} logged")
        for c in crashes:
            cx, cy = float(c["x"]), float(c["y"])
            i = int(np.argmin(np.hypot(seg[:, 0] - cx, seg[:, 1] - cy)))
            print(f"    step {c['step']:>4} at ({cx:+7.0f},{cy:+7.0f})  "
                  f"off-path {np.min(np.hypot(seg[:, 0] - cx, seg[:, 1] - cy)):.0f} mm, "
                  f"local path clearance {clr[i]:.0f} mm")
    print()


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else ARENA
    s = sys.argv[2] if len(sys.argv) > 2 else (SESSION if len(sys.argv) <= 1 else "")
    _settings.data_folder = TARGET_ROOT
    main(a, s)
