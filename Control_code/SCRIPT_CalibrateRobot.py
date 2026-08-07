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
import re
from pathlib import Path

import numpy as np

from Library import Client
from Library import LorexTracker
from Library import Utils
from Library import Settings as _settings
from Library.TrackerNav import wait_for_stable_pose


SETTINGS_PATH = Path(__file__).parent / "Library" / "Settings.py"
IDENTITY_ROTATION = [-40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40]


# ── Settings ──────────────────────────────────────────────────────────────────
ROBOT_ID            = 1

# Which phases to run. Flip to False to skip a phase (e.g., re-cal drive curl
# only after a wheel swap, without disturbing a known-good rotation table).
# Single-phase runs leave the other phase's Settings.py fields and JSON
# section untouched.
RUN_PHASE_1         = False   # rotation
RUN_PHASE_2         = True    # forward drive (curl + distance scale)

# Phase 1 — rotation
ANGLES              = [-40, -30, -20, -10, -5, 5, 10, 20, 30, 40]   # 0 added implicitly
ROTATION_REPEATS    = 5    # 3 was too few — a single contaminated rep can pull
                           # the median (e.g., 2-of-3 sign-flipped reps at ±5°
                           # in the 2026-05-10 cal). With 5 reps the median
                           # absorbs up to 2 contaminated reps cleanly.

# Phase 2 — forward drive (single distance; two metrics extracted per rep)
DRIVE_MM            = 150.0  # matches the deploy step (Config.fixed_drive_mm),
                             # so curl is measured at the operating point
DRIVE_REPEATS       = 10   # large enough that median is well-defined and
                           # robust against the occasional settle-artifact rep.
                           # 10 x 150mm needs ~1.5m of run-out; 16 x 200mm
                           # (the previous setting) needed 3.2m.


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


def _replace_field(content: str, field_name: str, new_rhs: str) -> str:
    """Find a `field_name: <type> = <rhs>` line in Settings.py and replace
    its RHS with `new_rhs`. Preserves the prefix (indent + name + type +
    `=` and surrounding whitespace) so the file's hand-formatted alignment
    is not disturbed."""
    pattern = re.compile(
        rf"^(\s*{re.escape(field_name)}\s*:\s*\S+\s*=\s*).*$",
        re.MULTILINE,
    )
    if not pattern.search(content):
        raise RuntimeError(
            f"Could not find field '{field_name}' in {SETTINGS_PATH}; "
            f"hand-edit may be required."
        )
    return pattern.sub(lambda m: m.group(1) + new_rhs, content)


def _write_settings(rotation_obtained=None,
                    curl_rate=None,
                    distance_scale=None) -> None:
    """Patch Settings.py in place. Each kwarg that is not None updates the
    corresponding field; unspecified fields are left untouched."""
    content = SETTINGS_PATH.read_text()
    if rotation_obtained is not None:
        content = _replace_field(
            content, "rotation_obtained",
            f"field(default_factory=lambda: {list(rotation_obtained)})",
        )
    if curl_rate is not None:
        content = _replace_field(
            content, "drive_yaw_curl_deg_per_mm", repr(float(curl_rate)),
        )
    if distance_scale is not None:
        content = _replace_field(
            content, "drive_distance_scale", repr(float(distance_scale)),
        )
    SETTINGS_PATH.write_text(content)


def _check_or_reset_identity(cfg, run_phase_1: bool, run_phase_2: bool) -> None:
    """Verify the calibration fields *relevant to the phases being run* are
    at identity (so Client.step won't pre-compensate during measurement).
    If not, summarise what's non-identity and offer to reset just those
    fields. Aborts if the user declines.

    Phase 1 needs `rotation_obtained == rotation_desired`. Phase 2 needs
    `drive_yaw_curl_deg_per_mm == 0.0` and `drive_distance_scale == 1.0`.
    Running only Phase 2 leaves the rotation table alone (since
    `step(distance=X, angle=0)` produces correction=0 at the table's
    anchor point), and vice versa."""
    rot_off = run_phase_1 and (
        list(cfg.rotation_desired) != list(cfg.rotation_obtained)
    )
    drv_off = run_phase_2 and (
        cfg.drive_yaw_curl_deg_per_mm != 0.0 or cfg.drive_distance_scale != 1.0
    )
    if not (rot_off or drv_off):
        return

    problems = []
    if rot_off:
        problems.append(
            f"  • rotation_obtained ≠ rotation_desired "
            f"(currently {list(cfg.rotation_obtained)})"
        )
    if drv_off and cfg.drive_yaw_curl_deg_per_mm != 0.0:
        problems.append(
            f"  • drive_yaw_curl_deg_per_mm = "
            f"{cfg.drive_yaw_curl_deg_per_mm} (need 0.0)"
        )
    if drv_off and cfg.drive_distance_scale != 1.0:
        problems.append(
            f"  • drive_distance_scale = {cfg.drive_distance_scale} "
            f"(need 1.0)"
        )

    print("⚠️  Calibration fields relevant to the requested phase(s) are "
          "non-identity. Measuring now would calibrate residuals on top of "
          "the existing correction (Client.step would pre-compensate). Found:")
    for p in problems:
        print(p)
    ans = input("Reset Settings.py and in-memory cfg to identity now? [y/N]: ")
    if ans.strip().lower() != "y":
        print("Aborted. Reset the fields manually before re-running.")
        raise SystemExit(0)

    # Reset only the fields relevant to the phases being run, so a Phase-2
    # only run doesn't blow away an existing rotation calibration.
    write_kwargs: dict = {}
    if rot_off:
        write_kwargs["rotation_obtained"] = IDENTITY_ROTATION
    if drv_off:
        write_kwargs["curl_rate"]      = 0.0
        write_kwargs["distance_scale"] = 1.0
    _write_settings(**write_kwargs)

    # Mirror the file change into the running cfg so this process measures
    # raw firmware behaviour without restarting.
    if rot_off:
        cfg.rotation_obtained = list(IDENTITY_ROTATION)
    if drv_off:
        cfg.drive_yaw_curl_deg_per_mm = 0.0
        cfg.drive_distance_scale = 1.0
    print(f"Reset relevant fields to identity in {SETTINGS_PATH} and in-memory cfg.")


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

def main(run_phase_1: bool = RUN_PHASE_1, run_phase_2: bool = RUN_PHASE_2):
    """Run the requested calibration phase(s).

    Defaults come from the `RUN_PHASE_1` / `RUN_PHASE_2` constants near the
    top of this file — flip those to skip a phase. Selecting one phase
    leaves the other's Settings.py fields and JSON section untouched, so
    re-running just one phase is safe (e.g., re-cal drive curl without
    disturbing a known-good rotation table). The phase-specific identity
    check ensures only the relevant fields need to be at identity for the
    chosen phase."""
    if not (run_phase_1 or run_phase_2):
        print("Both RUN_PHASE_1 and RUN_PHASE_2 are False — nothing to do.")
        raise SystemExit(0)

    client  = Client.Client(robot_number=ROBOT_ID)
    tracker = LorexTracker.LorexTracker()

    cfg        = client.configuration
    robot_name = cfg.robot_name

    _check_or_reset_identity(cfg, run_phase_1, run_phase_2)

    # Quiet the radio while we calibrate.
    client.change_free_ping_period(0)

    if run_phase_1 and run_phase_2:
        phase_label = "Phase 1 + Phase 2"
    elif run_phase_1:
        phase_label = "Phase 1 only (rotation)"
    else:
        phase_label = "Phase 2 only (forward drive)"
    print(f"Calibrating {robot_name} (aruco_id={cfg.aruco_id})  [{phase_label}]")

    # ── Run requested phase(s) ───────────────────────────────────────────────
    rot_desired = rot_obtained = rot_samples = None
    curl_rate = distance_scale = drive_samples = None

    if run_phase_1:
        rot_desired, rot_obtained, rot_samples = run_rotation_phase(client, tracker)
    if run_phase_2:
        curl_rate, distance_scale, drive_samples = run_drive_phase(client, tracker)

    # ── Save combined JSON, merging with any prior phase's data ──────────────
    # When running only one phase, preserve the not-run phase's fields so the
    # JSON stays a complete record of the latest calibration for each metric.
    os.makedirs(_settings.calibration_folder, exist_ok=True)
    out_path = os.path.join(_settings.calibration_folder,
                            f"{robot_name}_calibration.json")
    payload: dict = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            payload = {}
    if run_phase_1:
        payload["rotation_desired"]             = rot_desired
        payload["rotation_obtained"]            = rot_obtained
        payload["rotation_n_repeats_per_angle"] = ROTATION_REPEATS
        payload["rotation_per_angle_samples"]   = {str(k): v for k, v in rot_samples.items()}
    if run_phase_2:
        payload["drive_commanded_mm"]           = DRIVE_MM
        payload["drive_n_repeats"]              = DRIVE_REPEATS
        payload["drive_yaw_curl_deg_per_mm"]    = curl_rate
        payload["drive_distance_scale"]         = distance_scale
        payload["drive_per_rep_samples"]        = drive_samples
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Phase 1 summary + asymmetry + outliers ───────────────────────────────
    rotation_table_critical = False     # block the Settings.py write if True
    if run_phase_1:
        print("\nRotation per-angle results (median, std, n):")
        rot_outliers: list = []   # collected across angles for the warning block
        for d, o in zip(rot_desired, rot_obtained):
            if d == 0:
                print(f"  {d:+4d}°  →  {o:+7.2f}°  (anchor)")
                continue
            vals = np.asarray(rot_samples[d])
            # Flag reps far enough from the bin's median to look contaminated.
            # 3° from median is well above typical within-bin spread (<2°) but
            # below the magnitude of a true outlier (sign-flip, half-rotation,
            # stale-frame zero), so it cleanly catches the bad reps.
            bad = [(i, v) for i, v in enumerate(vals) if abs(v - o) > 3.0]
            flag = ""
            if bad:
                flag = (f"  ⚠️  {len(bad)} outlier(s): "
                        + ", ".join(f"rep {i}={v:+.2f}°" for i, v in bad))
                rot_outliers.append((d, bad, vals.tolist()))
            print(f"  {d:+4d}°  →  {o:+7.2f}°  "
                  f"(std {vals.std():.2f}°, n={len(vals)}){flag}")

        if rot_outliers:
            # Median absorbs up to ⌊n/2⌋ outliers; if we see more than that in
            # one bin, the saved value is suspect even after median filtering.
            critical = [(d, bad, vals) for d, bad, vals in rot_outliers
                        if len(bad) > len(vals) // 2]
            if critical:
                rotation_table_critical = True
                print("\n⚠️  CRITICAL: bins where outliers outnumber clean reps "
                      "(median is unreliable):")
                for d, bad, vals in critical:
                    print(f"  {d:+4d}°: {len(bad)}/{len(vals)} reps flagged. "
                          f"Reps={[round(v,2) for v in vals]}. The saved value "
                          f"may be wrong — re-run this bin or hand-fix in "
                          f"Settings.py before deploying.")

        # Sanity check: the table must be monotonic in `obtained` once sorted
        # by `desired` — otherwise np.interp inside Client.step.get_correction
        # will produce wildly wrong commands near non-monotonic regions.
        obtained_arr = np.asarray(rot_obtained)
        if not np.all(np.diff(obtained_arr) > 0):
            rotation_table_critical = True
            bad_pairs = [(rot_desired[i], rot_obtained[i],
                          rot_desired[i+1], rot_obtained[i+1])
                         for i in range(len(rot_obtained) - 1)
                         if rot_obtained[i+1] <= rot_obtained[i]]
            print("\n⚠️  CRITICAL: rotation table is non-monotonic — "
                  "Client.step's get_correction will misbehave near these bins:")
            for d1, o1, d2, o2 in bad_pairs:
                print(f"  desired {d1:+d}°→obtained {o1:+.2f}°  followed by  "
                      f"desired {d2:+d}°→obtained {o2:+.2f}°")
            print("  Fix the bad bin(s) before writing the table to Settings.py.")

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
    if run_phase_1:
        print(f"    rotation_desired:           list  = field(default_factory=lambda: {rot_desired})")
        print(f"    rotation_obtained:          list  = field(default_factory=lambda: {rot_obtained})")
    if run_phase_2 and curl_rate is not None:
        print(f"    drive_yaw_curl_deg_per_mm:  float = {curl_rate}")
        print(f"    drive_distance_scale:       float = {distance_scale}")

    # ── Offer to write directly ──────────────────────────────────────────────
    # When a CRITICAL flag fired in Phase 1 the rotation table is not safe to
    # deploy as-is, so we refuse the write rather than asking — the user has
    # to fix the bin or re-run before any update happens.
    if run_phase_1 and rotation_table_critical:
        print("\n⚠️  Skipping write: rotation table has CRITICAL issues "
              "(see warnings above). Fix the flagged bin(s) and re-run, or "
              "hand-edit Settings.py.")
        return
    ans = input(f"\nWrite these values directly to {SETTINGS_PATH}? [y/N]: ")
    if ans.strip().lower() == "y":
        write_kwargs: dict = {}
        if run_phase_1:
            write_kwargs["rotation_obtained"] = rot_obtained
        if run_phase_2 and curl_rate is not None:
            write_kwargs["curl_rate"]      = curl_rate
            write_kwargs["distance_scale"] = distance_scale
        _write_settings(**write_kwargs)
        print(f"Wrote calibration values to {SETTINGS_PATH}.")
    else:
        print("Skipped. Paste the lines above into Settings.py manually.")


if __name__ == "__main__":
    main()
