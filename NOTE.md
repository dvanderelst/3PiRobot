# Project state — 2026-05-08 evening

Picking up: re-run `SCRIPT_CalibrateRobot.py` with the improved (median +
16-rep) script, then re-deploy the policy and see whether the second-lap
divergence in `default_Target02_h32_nosigma_run04` is resolved.

## What we worked on today

The new policy `default_Target02_h32_nosigma` was trained overnight and
loads cleanly. Trajectories at epoch 1995 look comparable to teacher
rollouts. But on-robot deployment exposed a chain of issues that we worked
through in order:

**1. Tracker `wait_for_stable_pose` was returning pre-motion poses as
"settled."** Two failure modes, both fixed:

  - After a `client.step(...)` motion, the tracker would emit cached
    pre-motion frames for ~0.5–2 s. The spread-based stability check would
    trivially fire on identical lagged reads and declare the *pre-motion*
    pose as the post-motion read. Fix: motion-required mode with
    `prior_pose` — function refuses to count any read toward stability
    until at least one read differs from `prior_pose` by more than
    `2*tol`. `strict_motion=True` in calibration contexts so contaminated
    samples are skipped, not silently zero-rotated.
  - The client polls at 10 Hz, but the PyLorex server detection loop runs
    at ~7-8 Hz (`Simple_tcp.py:178-213`). When polling out-paces the
    server, the same buffered snapshot fills consecutive polls and
    trivially passes the spread test. Fix: bit-equality filter on
    consecutive reads — repeat tracker frames (same float `(x,y,yaw)`)
    don't advance the stability window. The clean fix is to surface
    `captured_at` on the server side; logged in `PyLorex/TODO.md`.

**2. Robot rotation calibration was contaminated by the above bug.** Three
of 30 repeats in the May-6 cal returned ~0° for ±20°/±30° commands because
their post-motion read was a stale pre-motion frame. Re-running with the
fixed `wait_for_stable_pose` gave clean numbers within ~1° of the original
table, confirming the rotation primitive is reproducible. So rotation
calibration was *not* the source of the −7°/step drift seen in
`run01`.

**3. Drive curl is the real problem.** Created
`SCRIPT_DiagnoseDriveCurl.py` (pure straight-drive + CW/CCW circles).
Pure-drive yaw drift was −0.05°/mm, consistent across all three test
phases. Confirmed wheel-mismatch / asymmetric drive primitive — the robot
curls right ~7° per 125 mm forward step. Marker-vs-trajectory check
confirmed this is real rotation during drive, not a marker mounting
offset.

**4. Calibrate-and-correct architecture for both axes.** Rather than
add tracker-based heading-hold (fragile when the tracker briefly loses
the marker), unified the rotation table with two new scalar corrections in
`SCRIPT_CalibrateRobot.py`:

  - `drive_yaw_curl_deg_per_mm` — yaw drift induced per mm forward.
    `Client.step(distance=...)` adds an opposite-sign counter-rotation up
    front so the net heading after rotate+drive matches the caller's
    `angle`.
  - `drive_distance_scale` — actual chord / commanded distance.
    `Client.step` divides the caller's distance by this scale before
    issuing the firmware command.

Both saved to `Library/RobotCalibration/<robot>_calibration.json` with
full per-rep raw samples; both retrievable from a single calibration
session (same drive sequence, two metrics extracted per rep).

`SCRIPT_CalibrateRotation.py` was retired (subsumed by
`SCRIPT_CalibrateRobot.py`).

## Where things stand

**Run01** (uncalibrated): robot spiraled inward, gave up at step 32 of
single lap. Per-step drift −7.0° ± 2.2°.

**Run03** (with calibration): completed one full figure-8 lap cleanly,
made it to step 103 on the second lap before crashing into the upper
interior obstacle. Per-step residual drift +2.2° ± 2.9° (over-correction
flipped from −7° to +2°).

**Run04** (after a re-cal): diverged earlier — the second-lap divergence
sets in before the figure-8 fully closes. Westward drift around step 73
puts the robot off-distribution; the policy didn't reach as far before
the same kind of misstep.

The diagnosis on the run03/run04 divergence: `motion_drive_gain_range_pct`
during training is 0.05 (±5%), but the calibration's chord measurements had
~15% spread on individual reps (range 171–242 mm for a commanded 200 mm)
because of residual settle-artifact contamination. So the calibrated
distance scale and curl rate carry noise comparable to the policy's
training tolerance. Lap 1 is fine; by lap 2 the cumulative errors put the
robot into state-action regions the policy didn't visit during training.

## What we did about it (last commit of the day)

Hardened `SCRIPT_CalibrateRobot.py`:

  - Aggregator switched from `np.mean` to `np.median` (both phases). One
    contaminated rep can no longer pull a table value off; with N reps and
    median, ⌊N/2⌋ outliers are silently absorbed.
  - `DRIVE_REPEATS` bumped 8 → 16 so the median has more samples and the
    SE on the curl and scale estimates tightens.
  - Drive summary now prints median, IQR, and range, and flags individual
    outliers (>3° Δyaw or >5% chord deviation from median) so they're
    visible at calibration time.

## What to do tomorrow

1. **Reset the three calibration fields in `Library/Settings.py` to
   identity:**
   ```
   rotation_desired:           [-40,-30,-20,-10,-5,0,5,10,20,30,40]
   rotation_obtained:          [-40,-30,-20,-10,-5,0,5,10,20,30,40]
   drive_yaw_curl_deg_per_mm:  0.0
   drive_distance_scale:       1.0
   ```
   The calibration script's identity-check will block the run otherwise.

2. **Run `SCRIPT_CalibrateRobot.py`.** Watch the printed Phase 2 summary
   — IQR/range/outlier-flag output will indicate whether the residual
   stale-frame contamination has been controlled. Paste the printed table
   lines into `Settings.py`.

3. **Re-run the policy** (`SCRIPT_RunPolicy.py` already retargeted at
   `default_Target02_h32_nosigma`, `Target02`). Bump `REPEAT` to next
   number. Compare second-lap divergence against run03/run04 — should be
   smaller if the new calibration estimate is tighter.

4. **If divergence is still present at lap 2:** add drive-curl and
   distance-scale as **per-episode random variables in training**
   (`SCRIPT_TrainPolicy.py`), parallel to the existing rot_gain /
   drive_gain biases. The policy currently sees per-step rotation+drive
   noise but doesn't learn to recover from cumulative curl drift across a
   long episode. Sampling `drive_curl ~ U(±0.05) °/mm` and
   `drive_scale ~ U(0.95, 1.05)` at episode reset and applying inside the
   simulator's drive primitive would force the policy to be robust to
   whatever residual remains after calibration. Multi-hour retrain.

## Open items

- **PyLorex `captured_at`** — properly fixing the "client gets back-to-
  back bit-identical snapshots" issue requires propagating
  `CameraSnapshot.captured_at` through `ServerClient.get_tracker(...)`.
  Tracked in `PyLorex/TODO.md`. Bit-equality filter in `TrackerNav.py` is
  a stopgap that works for now.
- **The +2°/step over-correction in run03** suggests the calibrated curl
  rate was a touch too aggressive vs the actual run-time curl. Could be
  surface/battery state difference between calibration and run time. The
  median-aggregator change should make calibration more reproducible run-
  to-run. After two clean calibrations on different days, we'll know if
  this is intrinsic variability or just noise in the old calibration.
- **Run04 trajectory** is at
  `Control_code/PolicyRuns/default_Target02_h32_nosigma_run04/trajectory.png`.
  Run03 at the corresponding `_run03/`. Worth re-comparing both against
  the next deployment.

## Commits today

3PiRobot:
1. Tracker settle: motion-required mode + repeat-frame filter
2. Policy run: use canonical wait_for_stable_pose; retarget to Target02
3. Add SCRIPT_DiagnoseDriveCurl: pure-drive + circle diagnostic
4. Calibration: unify rotation + drive-curl + distance into SCRIPT_CalibrateRobot
5. DefinePath: drop redundant linestyle; arenas list to Target02

PyLorex:
6. Add TODO: surface captured_at on get_tracker for client-side dedup

Pre-existing user WIP not yet committed (left for review):
- `Control_code/SCRIPT_TakeEnvSnapshot.py` (session name → "Target02")
- `Control_code/SCRIPT_TrainPolicy.py` (TARGET_ARENA → "Target02", CONDITION → "default")
- PyLorex: `Docs/calibration_process.md`, `LorexLib/Environment.py`,
  `LorexLib/Lorex.py`, `LorexLib/Settings.py`, `script_capture_environment.py`
