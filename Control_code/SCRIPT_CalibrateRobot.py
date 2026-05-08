#!/usr/bin/env python3
"""
SCRIPT_CalibrateRobot.py

Per-robot motion calibration via the Lorex tracker. Supersedes the old
SCRIPT_CalibrateRotation.py — this script measures *all three* corrections
that `Client.step` applies, in one pass:

  Phase 1 — rotation (in-place):
      Per-angle obtained-vs-desired table for `Client.step(angle=...)`'s
      rotation correction. Captures L/R asymmetric rotation bias so the
      policy's commanded rotations are achieved on-robot.
      Output: rotation_desired, rotation_obtained.

  Phase 2 — forward drive:
      N pure-forward drives at fixed `DRIVE_MM` extract two numbers
      simultaneously from each drive:
        • Δyaw / D_cmd  →  drive_yaw_curl_deg_per_mm  (signed; CCW per mm)
            yaw drift induced by the drive primitive (asymmetric wheel
            slip / diameter mismatch). `Client.step(distance=...)` adds an
            equal-and-opposite counter-rotation up front so the net heading
            change after the drive matches the caller's `angle`.
        • chord_xy / D_cmd  →  drive_distance_scale  (actual / commanded)
            multiplicative slip. `Client.step(distance=...)` divides the
            caller's requested distance by this scale before issuing the
            firmware command, so the actual distance driven matches.

      Both metrics come from the *same* settled-pose reads, so they cost
      no extra robot motion.

Pre-condition (sanity-checked at startup): all three calibration fields in
Settings.py must currently be at identity, otherwise Client.step would
pre-compensate during measurement and we'd be calibrating the residual on
top of the existing correction:
    rotation_desired == rotation_obtained
    drive_yaw_curl_deg_per_mm == 0.0
    drive_distance_scale       == 1.0

Workflow:
  1. Reset the three fields to identity in Library/Settings.py.
  2. Place the robot on a clear patch of floor inside tracker view, with
     ~360° clearance for spins (Phase 1) and a few metres in the forward
     direction (Phase 2).
  3. Run this script. Each phase pauses for an Enter press before motion.
  4. Saves Library/RobotCalibration/<robot_name>_calibration.json with the
     full per-rep raw samples; prints copy-paste lines for Settings.py.
"""

import json
import math
import os

import numpy as np

from Library import Client
from Library import LorexTracker
from Library import Utils
from Library import Settings as _settings
from Library.TrackerNav import wait_for_stable_pose


# ── Settings ──────────────────────────────────────────────────────────────────
ROBOT_ID            = 1

# Phase 1 — rotation
ANGLES              = [-40, -30, -20, -10, -5, 5, 10, 20, 30, 40]   # 0 added implicitly
ROTATION_REPEATS    = 3

# Phase 2 — forward drive (single distance; two metrics extracted per rep)
DRIVE_MM            = 200.0
DRIVE_REPEATS       = 16   # large enough that median is well-defined and
                           # robust against the occasional settle-artifact rep


def _read_pose_settled(tracker, robot_id, prior_pose=None):
    """Settled tracker read. With `prior_pose` given, requires post-motion
    pose change before declaring settled and returns None on timeout, so
    callers can discard contaminated samples instead of recording a
    pre-motion repeat as a zero-motion measurement."""
    return wait_for_stable_pose(
        tracker, robot_id,
        prior_pose=prior_pose,
        strict_motion=prior_pose is not None,
        verbose=True,
    )


def _check_identity(cfg) -> None:
    """Refuse to run if any calibration field is non-identity — otherwise
    Client.step would pre-compensate during measurement and we'd be
    calibrating residuals on top of the existing correction."""
    problems = []
    if list(cfg.rotation_desired) != list(cfg.rotation_obtained):
        problems.append(
            "  • rotation_desired ≠ rotation_obtained "
            "(reset to symmetric identity, e.g. "
            "[-40,-30,-20,-10,-5,0,5,10,20,30,40])"
        )
    if cfg.drive_yaw_curl_deg_per_mm != 0.0:
        problems.append(
            f"  • drive_yaw_curl_deg_per_mm = {cfg.drive_yaw_curl_deg_per_mm} "
            f"(reset to 0.0)"
        )
    if cfg.drive_distance_scale != 1.0:
        problems.append(
            f"  • drive_distance_scale = {cfg.drive_distance_scale} "
            f"(reset to 1.0)"
        )
    if problems:
        print("⚠️  Calibration fields in Settings.py must be at identity "
              "before measuring; otherwise Client.step pre-compensates and "
              "skews the result. Found:")
        for p in problems:
            print(p)
        raise SystemExit(1)


# ── Phase 1: rotation ─────────────────────────────────────────────────────────

def run_rotation_phase(client, tracker):
    print(f"\n{'='*72}\nPhase 1: rotation calibration  "
          f"(angles {ANGLES}, {ROTATION_REPEATS} reps each)")
    print("Robot needs ±360° clearance for in-place spins. Press Enter.")
    input()

    results = {a: [] for a in ANGLES}
    for rep in range(ROTATION_REPEATS):
        for a in ANGLES:
            pose_before = _read_pose_settled(tracker, ROBOT_ID)
            try:
                client.step(angle=a, rotation_speed=90)
            except RuntimeError as e:
                print(f"  rep {rep}, angle {a:+4d}: step aborted: {e}")
                continue
            pose_after = _read_pose_settled(tracker, ROBOT_ID,
                                            prior_pose=pose_before)
            if pose_before is None or pose_after is None:
                print(f"  rep {rep}, angle {a:+4d}: tracker missed pose; skipping")
                continue
            delta = float(Utils.wrap_angle(pose_after[2] - pose_before[2],
                                           mode="deg180"))
            results[a].append(delta)
            print(f"  rep {rep}, angle {a:+4d}: actual {delta:+7.2f}°")

    # Build the table, anchored at (0, 0). Median (not mean) so a single
    # contaminated rep — e.g., from a settle that fired on stale frames —
    # cannot pull the table value off; with 3 reps that means 1 outlier
    # gets ignored without changing the result.
    desired_out  = [0]
    obtained_out = [0.0]
    for a in ANGLES:
        vals = results[a]
        if not vals:
            print(f"  ⚠️  no successful measurements for {a:+}°; skipping")
            continue
        desired_out.append(int(a))
        obtained_out.append(round(float(np.median(vals)), 2))
    order        = np.argsort(desired_out)
    desired_out  = [desired_out[i]  for i in order]
    obtained_out = [obtained_out[i] for i in order]

    return desired_out, obtained_out, results


# ── Phase 2: forward drive (curl + distance scale) ────────────────────────────

def run_drive_phase(client, tracker):
    print(f"\n{'='*72}\nPhase 2: forward-drive calibration  "
          f"({DRIVE_REPEATS} reps × {DRIVE_MM:.0f} mm)")
    print("Robot needs roughly a 2–3 m clear corridor in its current "
          "forward direction (the path will curve as drift accumulates). "
          "Press Enter.")
    input()

    samples = []   # list of {delta_yaw_deg, chord_mm}
    for rep in range(DRIVE_REPEATS):
        pose_before = _read_pose_settled(tracker, ROBOT_ID)
        try:
            client.step(distance=DRIVE_MM / 1000.0)
        except RuntimeError as e:
            print(f"  rep {rep}: step aborted: {e}")
            continue
        pose_after = _read_pose_settled(tracker, ROBOT_ID,
                                        prior_pose=pose_before)
        if pose_before is None or pose_after is None:
            print(f"  rep {rep}: tracker missed pose; skipping")
            continue
        delta_yaw = float(Utils.wrap_angle(pose_after[2] - pose_before[2],
                                           mode="deg180"))
        chord_mm = math.hypot(pose_after[0] - pose_before[0],
                              pose_after[1] - pose_before[1])
        samples.append({"delta_yaw_deg": delta_yaw, "chord_mm": chord_mm})
        print(f"  rep {rep}: Δyaw={delta_yaw:+6.2f}°  chord={chord_mm:6.1f} mm  "
              f"(curl={delta_yaw/DRIVE_MM:+.4f}°/mm  scale={chord_mm/DRIVE_MM:.4f})")

    if not samples:
        print("  ⚠️  no successful drive measurements; aborting Phase 2.")
        return None, None, samples

    deltas  = np.array([s["delta_yaw_deg"] for s in samples], dtype=float)
    chords  = np.array([s["chord_mm"]      for s in samples], dtype=float)
    # Median over all reps. The drive phase tends to produce occasional
    # outliers (a settle that fires on stale frames mid-motion, a single
    # ping-pong drive that under-shoots, etc.) — see the lap-2 divergence
    # in default_Target02_h32_nosigma_run03/04 for context. Mean would
    # absorb those into the calibration; median ignores them and only
    # uses the central cluster of repeats.
    curl_rate      = round(float(np.median(deltas) / DRIVE_MM), 5)
    distance_scale = round(float(np.median(chords) / DRIVE_MM), 4)
    return curl_rate, distance_scale, samples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    client  = Client.Client(robot_number=ROBOT_ID)
    tracker = LorexTracker.LorexTracker()

    cfg        = client.configuration
    robot_name = cfg.robot_name

    _check_identity(cfg)

    # Quiet the radio while we calibrate.
    client.change_free_ping_period(0)

    print(f"Calibrating {robot_name} (aruco_id={cfg.aruco_id})")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    rot_desired, rot_obtained, rot_samples = run_rotation_phase(client, tracker)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    curl_rate, distance_scale, drive_samples = run_drive_phase(client, tracker)

    # ── Save combined JSON ───────────────────────────────────────────────────
    os.makedirs(_settings.calibration_folder, exist_ok=True)
    out_path = os.path.join(_settings.calibration_folder,
                            f"{robot_name}_calibration.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "rotation_desired":              rot_desired,
                "rotation_obtained":             rot_obtained,
                "rotation_n_repeats_per_angle":  ROTATION_REPEATS,
                "rotation_per_angle_samples":    {str(k): v for k, v in rot_samples.items()},
                "drive_commanded_mm":            DRIVE_MM,
                "drive_n_repeats":               DRIVE_REPEATS,
                "drive_yaw_curl_deg_per_mm":     curl_rate,
                "drive_distance_scale":          distance_scale,
                "drive_per_rep_samples":         drive_samples,
            },
            f, indent=2,
        )
    print(f"\nSaved: {out_path}")

    # ── Phase 1 summary + asymmetry ──────────────────────────────────────────
    print("\nRotation per-angle results (mean, std, n):")
    for d, o in zip(rot_desired, rot_obtained):
        if d == 0:
            print(f"  {d:+4d}°  →  {o:+7.2f}°  (anchor)")
            continue
        vals = np.asarray(rot_samples[d])
        print(f"  {d:+4d}°  →  {o:+7.2f}°  "
              f"(std {vals.std():.2f}°, n={len(vals)})")

    pairs = [(d, o) for d, o in zip(rot_desired, rot_obtained) if d != 0]
    pos = [(d, o)   for d, o in pairs if d > 0]
    neg = [(-d, -o) for d, o in pairs if d < 0]
    common = set(d for d, _ in pos) & set(d for d, _ in neg)
    if common:
        print("\nL/R asymmetry (|obtained| difference between -d and +d):")
        for d in sorted(common):
            op = next(o for dd, o in pos if dd == d)
            on = next(o for dd, o in neg if dd == d)
            print(f"  ±{d:>2}°:  +{op:.2f}  vs  -{on:.2f}  "
                  f"(diff {op-on:+.2f}°)")

    # ── Phase 2 summary ──────────────────────────────────────────────────────
    if curl_rate is not None and drive_samples:
        deltas = np.array([s["delta_yaw_deg"] for s in drive_samples])
        chords = np.array([s["chord_mm"]      for s in drive_samples])
        d_med, c_med = np.median(deltas), np.median(chords)
        # Flag reps far enough from the median that they suggest a
        # contaminated measurement rather than real wheel variance.
        d_outliers = [(i, v) for i, v in enumerate(deltas) if abs(v - d_med) > 3.0]
        c_outliers = [(i, v) for i, v in enumerate(chords)
                      if abs(v - c_med) / DRIVE_MM > 0.05]
        print(f"\nForward drive summary  ({DRIVE_REPEATS} reps × {DRIVE_MM:.0f} mm, "
              f"n={len(drive_samples)}):")
        print(f"  Δyaw per drive  : median {d_med:+6.2f}°  "
              f"(IQR {np.percentile(deltas,25):+.2f} … {np.percentile(deltas,75):+.2f}, "
              f"range {deltas.min():+.2f} … {deltas.max():+.2f})  "
              f"→ curl  {curl_rate:+.5f} °/mm")
        print(f"  chord per drive : median {c_med:6.1f} mm  "
              f"(IQR {np.percentile(chords,25):.1f} … {np.percentile(chords,75):.1f}, "
              f"range {chords.min():.1f} … {chords.max():.1f})  "
              f"→ scale {distance_scale:.4f}")
        if d_outliers:
            print(f"  ⚠️  {len(d_outliers)} Δyaw outlier(s) >3° from median — "
                  f"{['rep %d=%+.2f°' % (i, v) for i, v in d_outliers]}")
        if c_outliers:
            print(f"  ⚠️  {len(c_outliers)} chord outlier(s) >5% from median — "
                  f"{['rep %d=%.1fmm' % (i, v) for i, v in c_outliers]}")
        if d_outliers or c_outliers:
            print(f"      median is robust to these, but check the "
                  f"per-rep samples in the JSON if the table looks off.")
        if abs(curl_rate * 1000) > 1:
            sign = "right (CW)" if curl_rate < 0 else "left (CCW)"
            print(f"  → robot curls to the {sign} during forward drive; "
                  f"will be cancelled in Client.step.")
        if abs(distance_scale - 1.0) > 0.02:
            direction = "under-drives" if distance_scale < 1.0 else "over-drives"
            print(f"  → robot {direction} commanded distance by "
                  f"{(distance_scale - 1.0) * 100:+.1f}%; will be cancelled "
                  f"in Client.step.")

    # ── Paste-ready snippet ──────────────────────────────────────────────────
    print("\nPaste into the matching ClientConfig in Library/Settings.py:")
    print(f"    rotation_desired:           list  = field(default_factory=lambda: {rot_desired})")
    print(f"    rotation_obtained:          list  = field(default_factory=lambda: {rot_obtained})")
    if curl_rate is not None:
        print(f"    drive_yaw_curl_deg_per_mm:  float = {curl_rate}")
        print(f"    drive_distance_scale:       float = {distance_scale}")


if __name__ == "__main__":
    main()
