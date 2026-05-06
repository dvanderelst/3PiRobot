#!/usr/bin/env python3
"""
SCRIPT_CalibrateRotation.py

Measure per-robot rotation accuracy via the Lorex tracker and produce the
(rotation_desired, rotation_obtained) table that Client.step's
`get_correction` uses to compensate L/R-asymmetric rotation bias.

Why this is needed: in simulator training the policy sees rotations
applied with no asymmetric bias. On the real robot, a +X° command and a
-X° command can produce different magnitudes (motor mismatch, wheel
diameter mismatch, friction asymmetry). Over a long policy rollout the
heading drift accumulates and the policy ends up in poses it was never
trained on.

Workflow:
  1. Place the robot on a clear patch of floor inside the tracker view
     and away from walls (it will spin in place).
  2. The script disables free-pings, then for each desired angle reads
     yaw, commands the rotation, waits, reads yaw again, computes the
     wrapped delta. Repeats per angle for averaging.
  3. Saves Library/RobotCalibration/<robot_name>_rotation.json and
     prints copy-pasteable lines for Library/Settings.py so the next
     Client(...) instance picks up the new table.

Pre-condition: rotation_desired and rotation_obtained must currently be
the symmetric identity in Settings.py — otherwise Client.step would
pre-compensate during measurement and we'd be measuring residual error
on top of the existing table instead of raw firmware behaviour.
"""

import json
import os
import time

import numpy as np

from Library import Client
from Library import LorexTracker
from Library import Utils
from Library import Settings as _settings
from Library.TrackerNav import wait_for_stable_pose


# ── Settings ──────────────────────────────────────────────────────────────────
ROBOT_ID            = 1
ANGLES              = [-40, -30, -20, -10, -5, 5, 10, 20, 30, 40]   # 0 added implicitly as (0, 0)
REPEATS             = 3
POST_STEP_DELAY_S   = 1.0   # buffer after each step so the tracker has begun
                            # reflecting the new motion before settled-poll starts.
                            # Without this, identical lagged frames look "stable"
                            # and produce a y_after read that's still pre-rotation.


def _read_yaw_settled(tracker, robot_id):
    """Return a yaw read taken after the tracker has settled, or None."""
    pose = wait_for_stable_pose(tracker, robot_id, verbose=True)
    return pose[2] if pose is not None else None


def main():
    client  = Client.Client(robot_number=ROBOT_ID)
    tracker = LorexTracker.LorexTracker()

    cfg = client.configuration
    robot_name = cfg.robot_name

    # Sanity: must be running with identity calibration so we measure raw firmware behaviour.
    desired_now  = list(cfg.rotation_desired)
    obtained_now = list(cfg.rotation_obtained)
    if desired_now != obtained_now:
        print("⚠️  rotation_desired and rotation_obtained differ in Settings.py.")
        print("    Reset to the symmetric identity before measuring; otherwise")
        print("    Client.step.get_correction() will pre-compensate and skew")
        print("    the result.")
        raise SystemExit(1)

    # No free-pings while we're calibrating: keeps the radio quiet between
    # steps and avoids any guard-window interference.
    client.change_free_ping_period(0)

    print(f"Calibrating rotation for {robot_name} (aruco_id={cfg.aruco_id})")
    print(f"Angles: {ANGLES}    Repeats: {REPEATS}")
    print("Make sure the robot has open space for ±360° spins, then press Enter.")
    input()

    results = {a: [] for a in ANGLES}

    # Interleave repeats × angles so any monotonic drift (battery sag, slow
    # heating) is averaged across all angles instead of biasing one bin.
    for rep in range(REPEATS):
        for a in ANGLES:
            y_before = _read_yaw_settled(tracker, ROBOT_ID)
            try:
                client.step(angle=a, rotation_speed=90)
            except RuntimeError as e:
                print(f"  rep {rep}, angle {a:+4d}: step aborted: {e}")
                continue
            time.sleep(POST_STEP_DELAY_S)
            y_after = _read_yaw_settled(tracker, ROBOT_ID)
            if y_before is None or y_after is None:
                print(f"  rep {rep}, angle {a:+4d}: tracker missed pose; skipping")
                continue
            delta = float(Utils.wrap_angle(y_after - y_before, mode="deg180"))
            results[a].append(delta)
            print(f"  rep {rep}, angle {a:+4d}: actual {delta:+7.2f}°")

    # Aggregate per-angle stats and build the (desired, obtained) table.
    desired_out  = [0]
    obtained_out = [0.0]
    for a in ANGLES:
        vals = results[a]
        if not vals:
            print(f"  ⚠️  no successful measurements for {a:+}°; skipping")
            continue
        desired_out.append(int(a))
        obtained_out.append(round(float(np.mean(vals)), 2))

    # Sort by desired so np.interp inside get_correction sees monotonic input.
    order        = np.argsort(desired_out)
    desired_out  = [desired_out[i]  for i in order]
    obtained_out = [obtained_out[i] for i in order]

    # Save JSON
    os.makedirs(_settings.calibration_folder, exist_ok=True)
    out_path = os.path.join(_settings.calibration_folder,
                            f"{robot_name}_rotation.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "rotation_desired":     desired_out,
                "rotation_obtained":    obtained_out,
                "n_repeats_per_angle":  REPEATS,
                "per_angle_samples":    {str(k): v for k, v in results.items()},
            },
            f, indent=2,
        )
    print(f"\nSaved: {out_path}")

    # Per-angle summary
    print("\nPer-angle results (mean ± std, n):")
    for d, o in zip(desired_out, obtained_out):
        if d == 0:
            print(f"  {d:+4d}°  →  {o:+7.2f}°  (anchor)")
            continue
        vals = np.asarray(results[d])
        print(f"  {d:+4d}°  →  {o:+7.2f}°  "
              f"(std {vals.std():.2f}°, n={len(vals)})")

    # Asymmetry diagnostic
    pairs = [(d, o) for d, o in zip(desired_out, obtained_out) if d != 0]
    pos = [(d, o) for d, o in pairs if d > 0]
    neg = [(-d, -o) for d, o in pairs if d < 0]
    if pos and neg:
        # Compare each |d| pair across L/R
        common = set(d for d, _ in pos) & set(d for d, _ in neg)
        if common:
            print("\nL/R asymmetry (|obtained| difference between -d and +d):")
            for d in sorted(common):
                op = next(o for dd, o in pos if dd == d)
                on = next(o for dd, o in neg if dd == d)
                print(f"  ±{d:>2}°:  +{op:.2f}  vs  -{on:.2f}  "
                      f"(diff {op-on:+.2f}°)")

    # Paste-ready snippet for Settings.py
    print("\nPaste into the matching ClientConfig in Library/Settings.py:")
    print(f"    rotation_desired:  list = field(default_factory=lambda: {desired_out})")
    print(f"    rotation_obtained: list = field(default_factory=lambda: {obtained_out})")


if __name__ == "__main__":
    main()
