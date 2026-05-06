"""
TrackerNav: closed-loop "go to (x_target, y_target, yaw_target)" navigation
using the overhead Lorex tracker as feedback and Client.step as the
open-loop motion primitive.

Conventions (see Notion "Lorex Camera System"):
    World:    +X east, +Y north (mm).
    yaw_deg:  CCW from +X to robot forward, in [-180, +180).
    rotate:   client.step(angle=...) is CCW-positive degrees (left turn).
    drive:    client.step(distance=...) is forward metres.

Algorithm per iteration of go_to_pose:
    1. Read tracker pose. If marker missing for several retries -> abort.
    2. dx, dy = target - pose; dist = hypot(dx, dy).
       If dist <= pos_tol_mm -> proceed to final yaw alignment.
    3. bearing = atan2(dy, dx); heading_err = wrap(bearing - yaw).
    4. If |heading_err| > align_yaw_thresh_deg -> rotate by heading_err only.
       Else                                     -> drive min(dist, max_step) m.
    5. Settle, repeat.

After the position loop exits, if a yaw_target_deg was given a short
secondary loop rotates in place until |yaw - yaw_target| <= yaw_tol_deg.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from Library import Logging
from Library import Utils


def _yaw_spread(yaws) -> float:
    """Wrap-aware max pairwise difference (deg) across a sequence of yaws."""
    a = np.asarray(yaws, dtype=float)
    diffs = a.reshape(-1, 1) - a.reshape(1, -1)
    diffs = ((diffs + 180.0) % 360.0) - 180.0
    return float(np.abs(diffs).max())


def wait_for_stable_pose(tracker,
                         robot_id: int,
                         yaw_tol_deg: float = 0.5,
                         pos_tol_mm: float = 8.0,
                         n_consec: int = 4,
                         poll_s: float = 0.1,
                         timeout_s: float = 5.0,
                         verbose: bool = False) -> Optional[Tuple[float, float, float]]:
    """Poll `tracker` until both yaw AND position are stable across the
    last `n_consec` reads (yaw spread < `yaw_tol_deg`, position spread <
    `pos_tol_mm`), or `timeout_s` elapses.

    Returns `(x_mm, y_mm, yaw_deg)` or None if the tracker never produced a
    valid read in the timeout window. Used by TrackerNav and standalone
    callers (e.g., SCRIPT_CalibrateRotation) that need a post-motion read
    after the camera/processing pipeline has caught up. Yaw alone is not
    enough: integer-rounded yaw can stick on a stale frame while x/y are
    still drifting.

    ⚠ Caller responsibility — POST-STEP DELAY:
    If you call this immediately after `client.step(...)`, the tracker may
    still be emitting cached pre-motion frames for ~0.5-1.0 s. Those
    identical lagged reads trivially satisfy the stability criterion, and
    the function will return a "stable" pose that predates the motion you
    just commanded. Symptom: a measured rotation of ~0° for a non-trivial
    command (this bit `SCRIPT_CalibrateRotation` once already).

    The fix is at the call site, not here, because the right amount of
    pre-poll padding depends on what the caller just did (or didn't do):
      - `TrackerNav.go_to_pose` sleeps `post_step_delay_s` after each step.
      - `SCRIPT_CalibrateRotation` sleeps `POST_STEP_DELAY_S` after each step.
      - Idle re-reads (no recent motion) need no padding.
    Adding a delay inside this function would silently slow down callers
    that don't need it, so each caller declares its own buffer."""
    history_yaw: list = []
    history_x:   list = []
    history_y:   list = []
    last_pose: Optional[Tuple[float, float, float]] = None
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        pos = tracker.get_position(robot_id)
        if pos is not None and pos.get('x') is not None:
            x = float(pos['x']); y = float(pos['y']); yaw = float(pos['yaw_deg'])
            last_pose = (x, y, yaw)
            history_yaw.append(yaw); history_x.append(x); history_y.append(y)
            if len(history_yaw) >= n_consec:
                yaw_spread = _yaw_spread(history_yaw[-n_consec:])
                pos_spread = float(np.hypot(
                    np.ptp(history_x[-n_consec:]),
                    np.ptp(history_y[-n_consec:]),
                ))
                if yaw_spread < yaw_tol_deg and pos_spread < pos_tol_mm:
                    return last_pose
        time.sleep(poll_s)
    if last_pose is not None and verbose:
        Logging.print_message("tracker_settle",
                              f"pose did not stabilise within {timeout_s:.1f}s; "
                              f"using last pose", "WARNING")
    return last_pose


@dataclass
class GoToPoseResult:
    success: bool
    iterations: int
    final_pos_err_mm: float
    final_yaw_err_deg: float
    reason: str


class TrackerNav:
    def __init__(self, client, tracker, robot_number,
                 pos_tol_mm: float = 50.0,
                 yaw_tol_deg: float = 5.0,
                 align_yaw_thresh_deg: float = 15.0,
                 max_step_distance_m: float = 0.20,
                 max_iterations: int = 30,
                 max_yaw_iterations: int = 8,
                 yaw_stable_tol_deg: float = 0.5,
                 pos_stable_tol_mm: float = 8.0,
                 stable_n_consec: int = 4,
                 stable_poll_s: float = 0.1,
                 stable_timeout_s: float = 5.0,
                 post_step_delay_s: float = 0.5,
                 verbose: bool = True):
        self.client = client
        self.tracker = tracker
        self.robot_number = robot_number
        self.pos_tol_mm = pos_tol_mm
        self.yaw_tol_deg = yaw_tol_deg
        self.align_yaw_thresh_deg = align_yaw_thresh_deg
        self.max_step_distance_m = max_step_distance_m
        self.max_iterations = max_iterations
        self.max_yaw_iterations = max_yaw_iterations
        self.yaw_stable_tol_deg = yaw_stable_tol_deg
        self.pos_stable_tol_mm = pos_stable_tol_mm
        self.stable_n_consec = stable_n_consec
        self.stable_poll_s = stable_poll_s
        self.stable_timeout_s = stable_timeout_s
        self.post_step_delay_s = post_step_delay_s
        self.verbose = verbose

    def _log(self, message: str, category: str = "INFO") -> None:
        if self.verbose:
            Logging.print_message("TrackerNav", message, category)

    def read_pose(self) -> Optional[Tuple[float, float, float]]:
        """Settled tracker read. Thin wrapper around `wait_for_stable_pose`
        using this instance's stability parameters."""
        pose = wait_for_stable_pose(
            self.tracker, self.robot_number,
            yaw_tol_deg=self.yaw_stable_tol_deg,
            pos_tol_mm=self.pos_stable_tol_mm,
            n_consec=self.stable_n_consec,
            poll_s=self.stable_poll_s,
            timeout_s=self.stable_timeout_s,
            verbose=False,
        )
        if pose is None:
            return None
        # Mirror prior log behaviour when the timeout fired: we can't tell here
        # whether it timed out vs. converged, so the helper carries the warning
        # log internally (only printed when verbose=True). Stay quiet on the
        # nav-loop path; failures surface via "tracker lost robot" downstream.
        return pose

    def go_to_pose(self,
                   x_target_mm: float,
                   y_target_mm: float,
                   yaw_target_deg: Optional[float] = None) -> GoToPoseResult:
        """Drive to (x_target_mm, y_target_mm); optionally align to yaw_target_deg."""
        pos_iter = 0
        dist_mm = float('nan')
        arrived = False

        for pos_iter in range(self.max_iterations):
            pose = self.read_pose()
            if pose is None:
                return GoToPoseResult(False, pos_iter, float('nan'), float('nan'),
                                      "tracker lost robot")
            x, y, yaw = pose
            dx = x_target_mm - x
            dy = y_target_mm - y
            dist_mm = math.hypot(dx, dy)

            if dist_mm <= self.pos_tol_mm:
                self._log(f"arrived at position (err={dist_mm:.1f} mm) after {pos_iter} iters")
                arrived = True
                break

            bearing_deg = math.degrees(math.atan2(dy, dx))
            heading_err = Utils.wrap_angle(bearing_deg - yaw)

            if abs(heading_err) > self.align_yaw_thresh_deg:
                self._log(
                    f"iter {pos_iter}: pose=({x:.0f},{y:.0f},{yaw:+.1f}°) "
                    f"target=({x_target_mm:.0f},{y_target_mm:.0f}) "
                    f"dist={dist_mm:.0f}mm head_err={heading_err:+.1f}° -> rotate"
                )
                try:
                    self.client.step(angle=heading_err)
                except RuntimeError as e:
                    return GoToPoseResult(False, pos_iter, dist_mm, float('nan'),
                                          f"rotate aborted: {e}")
                time.sleep(self.post_step_delay_s)
            else:
                drive_m = min(dist_mm / 1000.0, self.max_step_distance_m)
                self._log(
                    f"iter {pos_iter}: pose=({x:.0f},{y:.0f},{yaw:+.1f}°) "
                    f"target=({x_target_mm:.0f},{y_target_mm:.0f}) "
                    f"dist={dist_mm:.0f}mm head_err={heading_err:+.1f}° -> drive {drive_m*1000:.0f}mm"
                )
                try:
                    self.client.step(distance=drive_m)
                except RuntimeError as e:
                    return GoToPoseResult(False, pos_iter, dist_mm, float('nan'),
                                          f"drive aborted: {e}")
                time.sleep(self.post_step_delay_s)

        if not arrived:
            return GoToPoseResult(False, self.max_iterations, dist_mm, float('nan'),
                                  "max_iterations reached without arriving")

        if yaw_target_deg is None:
            return GoToPoseResult(True, pos_iter, dist_mm, float('nan'),
                                  "arrived (yaw not requested)")

        for yaw_iter in range(self.max_yaw_iterations):
            pose = self.read_pose()
            if pose is None:
                return GoToPoseResult(False, pos_iter + yaw_iter, dist_mm, float('nan'),
                                      "tracker lost robot during yaw align")
            x, y, yaw = pose
            yaw_err = Utils.wrap_angle(yaw_target_deg - yaw)
            final_dist = math.hypot(x_target_mm - x, y_target_mm - y)

            if abs(yaw_err) <= self.yaw_tol_deg:
                self._log(f"yaw aligned (err={yaw_err:+.2f}°) after {yaw_iter} iters")
                return GoToPoseResult(True, pos_iter + yaw_iter, final_dist, yaw_err,
                                      "arrived")

            self._log(f"yaw iter {yaw_iter}: yaw={yaw:+.1f}°, err={yaw_err:+.1f}° -> rotate")
            try:
                self.client.step(angle=yaw_err)
            except RuntimeError as e:
                return GoToPoseResult(False, pos_iter + yaw_iter, final_dist, yaw_err,
                                      f"yaw rotate aborted: {e}")
            time.sleep(self.post_step_delay_s)

        pose = self.read_pose()
        if pose is None:
            return GoToPoseResult(False, pos_iter + self.max_yaw_iterations, dist_mm,
                                  float('nan'), "tracker lost robot at end of yaw align")
        x, y, yaw = pose
        return GoToPoseResult(False, pos_iter + self.max_yaw_iterations,
                              math.hypot(x_target_mm - x, y_target_mm - y),
                              Utils.wrap_angle(yaw_target_deg - yaw),
                              "yaw did not converge")

    def go_to_poses(self, poses, stop_on_failure: bool = True):
        """Run go_to_pose on a sequence of (x_mm, y_mm[, yaw_deg]) tuples."""
        results = []
        for i, pose in enumerate(poses):
            if len(pose) == 2:
                x, y = pose; yaw = None
            else:
                x, y, yaw = pose
            self._log(f"=== waypoint {i + 1}/{len(poses)}: ({x:.0f}, {y:.0f}, "
                      f"{'-' if yaw is None else f'{yaw:+.1f}°'}) ===")
            res = self.go_to_pose(x, y, yaw)
            self._log(f"  -> {res.reason} (pos_err={res.final_pos_err_mm:.1f}mm, "
                      f"yaw_err={res.final_yaw_err_deg:+.1f}°)")
            results.append(res)
            if not res.success and stop_on_failure:
                self._log(f"stopping waypoint sequence after failure", category="WARNING")
                break
        return results
