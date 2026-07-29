#!/usr/bin/env python3
"""
SCRIPT_RunDirectPolicy.py

Hand-coded reactive *direct-learning* demo (paper Par 9): steer the robot to a
single pole in the arena, avoiding walls, using a fixed reactive rule on the
local feature the cross-modal inverse produces. Stands in for the trial-by-trial
association a bat would acquire under reinforcement — no internal model, no map,
no learned policy.

The whole point of the demonstration is teacher/student parity: the *same*
reactive rule runs on a local feature that can come from either modality.

  SENSE_SOURCE = "sonar"   real robot; feature from a sonar ping → InverseModel
               = "vision"  real robot; feature from the overhead-tracker pose +
                           arena geometry (the vision "teacher" running the rule)
               = "sim"     no robot; feature from a simulated pose advanced
                           kinematically — offline tuning of gains/thresholds

Both producers emit the identical local feature by construction:
    class ∈ {wall, pole, empty};  wall  → (right, center, left) slice distances;
                                  pole  → signed azimuth (+ccw = LEFT);
                                  empty → nothing within range (the 3-class
                                          inverse's "none"; geometry's empty or
                                          over-horizon cone).
Sonar gets it from the two-headed inverse's heads; vision/sim get it from
geometry (`nearest_reflector_in_cone` + `compute_profile`), which is exactly
the supervision target the inverse was trained against.

The reactive rule (`ReactiveController`):
  - nearest object is a POLE → steer to null its azimuth (keep it in front),
    drive a fixed step → closes in.
  - nearest object is a WALL → fly a gentle arc (a gentle curl) until a wall
    comes within range, then reflect off it (billiard rebound estimated from the
    three slice depths). The curl sign and magnitude are re-randomised on each
    wall contact (bounce or escape) so open-space wandering doesn't trace a fixed
    circle.
  - cone empty / nothing within range ("none" from the sonar inverse) → wander
    forward with a gentle curl, crossing open space and sweeping the cone until a
    wall or pole enters range (or scan in place if EMPTY_WANDER is off).
Selector is "class of nearest object" (design constraint #2): if a wall is
nearer than the pole the robot wanders (arc + rebound) until the pole becomes
nearest. Perception is memoryless (close to constraint #3): the only carried
state is the baseline-curl sign — a motor bias, not perceptual memory, added to
remove the unnatural dead-straight sections of the pure-reactive rule.

Success / collision are judged by the overhead tracker (ground truth), not by
the sensing modality: the controller *steers* from the chosen modality, the
tracker is only the referee for when-the-pole-is-reached and for logging.

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ DEPLOY GATE: before any "sonar"/"vision" run on the real robot, run       │
  │ SCRIPT_CalibrateRobot.py. drive_yaw_curl / drive_distance_scale drift     │
  │ between sessions (battery, gear wear, tyre compression) and a stale       │
  │ drive model curls the trajectory. "sim" mode needs no robot and no        │
  │ calibration.                                                              │
  └─────────────────────────────────────────────────────────────────────────┘
"""

import csv
import json
import math
import os
import shutil
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Library.AcquisitionSessionLoader import (
    _load_features_for_session,
    compute_profile,
    nearest_reflector_in_cone,
)


# ══════════════════════════════════════════════════════════════════════════════
# Settings — edit these
# ══════════════════════════════════════════════════════════════════════════════

SENSE_SOURCE = "sonar"          # "sonar" | "vision" | "sim"

# ── Which trial is this? ──────────────────────────────────────────────────────
# Experiment 1 is a set of TRIALS, each one (pole placement x start mark), run
# once per perceptual condition. POLE names the placement physically in the
# arena right now; START names the tape mark the robot is standing on. The
# agreed trial list lives in TRIAL_LIST (written by SCRIPT_SweepPolePositions);
# both are checked against it at startup, so running a combination that is not
# part of the design, or with the arena built for a different placement, is
# caught before the robot moves rather than discovered in the analysis.
POLE   = "Q1"                   # placement label, must match TRIAL_LIST
START  = 1                      # start mark index, must match start_poses.json
ARENA  = "DirectQ1"             # TargetArenas/<ARENA>/ BUILT FOR THIS PLACEMENT
SUFFIX = ''                     # optional tag for a repeat or a re-run

TRIAL_LIST = "TempOutput/StartPositionDigitization/pole_positions.json"

# Session name carries pole, start and condition, so no two runs of the design
# can collide. The previous scheme keyed on start alone, which meant the same
# start against two placements wrote to one folder and the second silently
# overwrote the first.
SESSION = f"direct_{POLE}_S{START}_{SENSE_SOURCE}{SUFFIX}"

# How far the built arena's pole may sit from the intended placement before the
# run is refused (mm). Covers grid quantisation (positions come off a 250 mm
# grid) plus hand-placement error, but not pointing ARENA at the wrong build.
POLE_POSITION_TOL_MM = 400.0

ROBOT_ID     = 1
MAX_STEPS    = 200
INVERSE_FOLD = "deploy"       # spatial-holdout deployment model (sonar mode)
# ── Sensing cone ──────────────────────────────────────────────────────────────
# Forward ±half-angle the robot "sees". Defaults to the sonar inverse's cone so
# the two modalities share an aperture; settable independently for vision/sim.
CONE_HALF_DEG = 35.0

# Vision/sim range horizon (mm). The 3-class sonar inverse abstains ("none")
# when the nearest in-cone reflector is beyond its trained MAX_RANGE. Set this to
# that range to make the vision/sim geometry producer abstain identically — a
# controlled teacher/student comparison, and the right setting when using sim to
# tune gains for the sonar deployment. None = full sight (vision as the
# unrestricted teacher). Required only for parity at *deploy*; training always
# applies the horizon (it defines the 'none' class).
GEOM_RANGE_HORIZON_MM = None

# ── Reactive rule constants (curved-bounce wall rule) ─────────────────────────
DRIVE_MM           = 150.0   # nominal forward step per cycle (mm)
MAX_TURN_DEG       = 25.0   # cap on pole-steering / scan rotation per step
K_POLE             = 0.6    # pole steering gain: rotate = K_POLE * pole_azimuth
BASELINE_CURL_DEG  = 5.0    # constant arc applied while cruising past a far wall;
                            # sign random until the first rebound, then fixed to it
BOUNCE_TRIGGER_MM  = 750.0  # wall closer than this in the cone → reflect (bounce)
BOUNCE_MAX_TURN    = 110.0  # cap on a rebound turn (lets a head-on wall reverse)
WALL_JAM_MM        = 250.0  # below this, rebound with no drive (don't push into wall)
SCAN_TURN_DEG      = 20.0   # in-place rotation when the cone is empty (used only
                            # when EMPTY_WANDER is False)

# 'none'/empty-cone behaviour. True → wander: drive forward with a gentle curl
# (sign follows the baseline-curl bias) so the robot crosses open stretches and
# sweeps the cone until a wall or pole enters range. This is the right default
# now that the 3-class sonar inverse abstains ('none') beyond ~1 m — pure
# in-place scan would otherwise leave the robot spinning in open space. The
# forward cone is clear for >= the horizon, so a step is safe. False → original
# scan-in-place (rotate SCAN_TURN_DEG, no drive).
EMPTY_WANDER     = True
WANDER_CURL_DEG  = 5.0   # gentle forward curl (deg/step) applied while wanderingy

# Corner-jam escape. Memoryless billiard reflection traps in concave corners — it
# alternates ±BOUNCE_MAX_TURN without netting an escape and re-drives whenever
# clearance creeps above WALL_JAM. When the nearest cone slice stays below
# JAM_CLEAR_MM for JAM_PATIENCE consecutive steps, the controller stops reflecting
# and reverses out decisively (ESCAPE_TURN_DEG toward the more-open side + a full
# drive) — the way it came in is open by construction. A motor reflex on a small
# counter, like the curl-sign bias.
JAM_CLEAR_MM    = 500.0   # a cone slice below this counts as "near a wall"
JAM_PATIENCE    = 4       # consecutive near-wall steps before escaping
ESCAPE_TURN_DEG = 160.0   # escape turn magnitude (toward the more-open side)

# On each wall contact (bounce or corner escape), re-roll the curl bias so the
# robot doesn't deterministically arc the same way — re-randomise the curve/wander
# sign and the curve magnitude for varied exploration. Without this a constant
# curl makes open-space wandering trace a fixed circle. CURVE_CURL_{MIN,MAX}_DEG
# bound the new magnitude (deg/step).
RANDOMIZE_CURVE = True
CURVE_CURL_MIN_DEG     = 2.0
CURVE_CURL_MAX_DEG     = 12.0

# Controller RNG seed for the real-robot run (initial curl sign + escape re-rolls).
# Derived from (POLE, START) and deliberately NOT from SENSE_SOURCE, so the sonar
# and vision runs of one trial draw the identical curl sign, curl magnitude and
# escape re-rolls. That is what makes the two conditions a matched pair: any
# difference between them is attributable to the feature source rather than to
# the controller having wandered differently. crc32 rather than hash(), whose
# string seed is randomised per interpreter run.
# Set to None for fresh entropy (exploratory runs), or an int to reproduce one.
CONTROLLER_SEED = zlib.crc32(f"{POLE}|{START}".encode()) & 0x7FFFFFFF

# Referee-based jam abort (safety net if the escape can't free it): end the run
# when the ground-truth nearest-wall surface clearance stays below
# JAM_ABORT_CLEAR_MM for JAM_ABORT_STEPS consecutive steps.
JAM_ABORT_CLEAR_MM = 50.0
JAM_ABORT_STEPS    = 6

# Rotation execution (real robot). The robot's rotation-calibration table
# (Settings.rotation_obtained) tops out at ~±40°; Client.get_correction clamps
# any larger target, so a single big command (a ±110° bounce or ±160° escape)
# under-rotates to ~40°. We execute any turn beyond ROT_SUBSTEP_MAX_DEG as a
# sequence of equal in-place sub-rotations, each inside the calibrated range, so
# the commanded total is actually delivered. Fallback if the table is unreadable;
# the live cap is derived from the loaded client's table at 0.9× its max.
ROT_SUBSTEP_MAX_DEG = 35.0

# ── Referee (ground-truth tracker/geometry) ───────────────────────────────────
ROBOT_RADIUS_MM = 48.0    # 96 mm 3pi+ 2040 diameter
STOP_MARGIN_MM  = 40.0    # success when pole-surface distance < radius + margin
COLLISION_MM    = 20.0    # min wall clearance (surface) before we call a crash

# ── Sim-mode kinematics (SENSE_SOURCE == "sim") ───────────────────────────────
SIM_START_XY_MM   = None   # (x, y) mm; None → arena centroid
SIM_START_YAW_DEG = 0.0
SIM_SEED          = 0xC0FFEE
SIM_KIN_NOISE     = False  # add training-style rotate/drive noise to the rollout
SIM_ROT_NOISE_DEG = 3.0
SIM_DRIVE_NOISE_MM = 5.0

# ── Tracker settling (real modes) — mirror SCRIPT_RunPolicy defaults ──────────
YAW_STABLE_TOL_DEG    = 2.0
YAW_STABLE_POS_TOL_MM = 8.0
YAW_STABLE_N_CONSEC   = 3
YAW_STABLE_POLL_S     = 0.1
YAW_STABLE_TIMEOUT_S  = 8.0

# Dry-run flags (real modes): set False for tethered debugging without motion.
do_rotation    = True
do_translation = True

PLOT_EVERY = 1   # re-save trajectory.png every N steps during a real run so it
                 # can be watched live (0 = only at the end). Mirrors RunPolicy.

DATA_FOLDER = "PolicyRuns"
ARENAS_ROOT = "TargetArenas"


# ══════════════════════════════════════════════════════════════════════════════
# Local feature + reactive rule (modality-agnostic)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LocalFeature:
    """The one feature both modalities emit; the only input to `decide`."""
    cls: str                              # "pole" | "wall" | "empty"
    pole_az_deg: float = float("nan")     # signed bearing when cls == "pole"
    p_pole: float = float("nan")          # P(pole) from the sonar inverse (sonar only)
    p_none: float = float("nan")          # P(none) from the 3-class inverse (sonar only)
    pole_az_sigma_deg: float = float("nan")  # pole-bearing σ (sonar only)
    slices_mm: Dict[str, float] = field(  # right/center/left when cls == "wall"
        default_factory=lambda: {"right": float("nan"),
                                 "center": float("nan"),
                                 "left": float("nan")})


@dataclass
class ReactiveParams:
    drive_mm: float = DRIVE_MM
    max_turn_deg: float = MAX_TURN_DEG
    k_pole: float = K_POLE
    baseline_curl_deg: float = BASELINE_CURL_DEG
    bounce_trigger_mm: float = BOUNCE_TRIGGER_MM
    bounce_max_turn: float = BOUNCE_MAX_TURN
    wall_jam_mm: float = WALL_JAM_MM
    scan_turn_deg: float = SCAN_TURN_DEG
    empty_wander: bool = EMPTY_WANDER
    wander_curl_deg: float = WANDER_CURL_DEG
    jam_clear_mm: float = JAM_CLEAR_MM
    jam_patience: int = JAM_PATIENCE
    escape_turn_deg: float = ESCAPE_TURN_DEG
    randomize_curve: bool = RANDOMIZE_CURVE
    curve_curl_min_deg: float = CURVE_CURL_MIN_DEG
    curve_curl_max_deg: float = CURVE_CURL_MAX_DEG


# Bearings of the three wall slices within the ±cone (ascending azimuth):
# right ≈ -2/3·cone, center 0, left ≈ +2/3·cone. Used to place the slice depths
# as points in the robot frame for the bounce reflection.
_SLICE_BEARINGS_RAD = np.radians([-2.0 * CONE_HALF_DEG / 3.0, 0.0,
                                  2.0 * CONE_HALF_DEG / 3.0])


def _reflect_rotation_deg(r, c, l):
    """Rotation (deg, robot frame) that reflects the forward velocity off the
    wall, estimating the wall line from the finite slice points. Two+ slices →
    fit the line and mirror; a single slice → head-on reverse."""
    pts = [(d * math.cos(b), d * math.sin(b))
           for d, b in zip((r, c, l), _SLICE_BEARINGS_RAD) if np.isfinite(d)]
    if len(pts) >= 2:
        u = np.asarray(pts[-1]) - np.asarray(pts[0])      # along-wall direction
        nrm = float(np.hypot(*u))
        n = np.array([1.0, 0.0]) if nrm < 1e-6 else np.array([-u[1] / nrm, u[0] / nrm])
    else:
        n = np.array([1.0, 0.0])                          # one slice → head-on
    fwd = np.array([1.0, 0.0])
    v_ref = fwd - 2.0 * (fwd @ n) * n
    return math.degrees(math.atan2(v_ref[1], v_ref[0]))


class ReactiveController:
    """Curved-bounce reactive rule: the modality-agnostic policy of the direct-
    learning demo. Per step, maps the instantaneous local feature to
    (rotate_deg, drive_mm, tag). Angle convention: +deg = LEFT (CCW-positive).

      pole  → steer to null the azimuth (keep it in front), drive a fixed step.
      wall  → fly an arc (constant `baseline_curl_deg`) until a wall comes within
              `bounce_trigger_mm`, then reflect off it (billiard rebound). Below
              `wall_jam_mm` the rebound carries no drive so it can't push into
              the wall. If a slice stays below `jam_clear_mm` for `jam_patience`
              steps (a concave-corner trap), reverse out (`escape`) toward the
              more-open side — re-rolling the curve's sign + magnitude — instead
              of reflecting forever.
      empty → wander forward with a gentle curl (cross open space + sweep) until
              a wall/pole enters range; or scan in place if EMPTY_WANDER is off.

    Perception is memoryless — the only carried state is a small motor reflex:
    the curl sign and magnitude (re-randomised on each wall contact — bounce or
    escape) and a corner-jam counter that triggers the escape. These are motor
    priors, not perceptual memory; they remove the dead-straight cruise, the
    fixed-circle orbit, and the corner trap of the pure rule without consulting
    any map or past observation. `reset(seed)` re-seeds the RNG and curl bias and
    clears the jam counter.
    """

    def __init__(self, P: ReactiveParams, seed: int = 0):
        self.P = P
        self.reset(seed)

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)   # persistent: curl sign + wall-contact re-rolls
        self.curl_sign = 1.0 if self.rng.random() < 0.5 else -1.0
        self.curve_mag = self.P.baseline_curl_deg
        self.bounced = False
        self.jam_count = 0

    def _reroll_curl(self):
        """Re-randomise the curl bias (sign + curve magnitude). Called on each
        wall contact so exploration varies instead of tracing a fixed arc."""
        self.curl_sign = 1.0 if self.rng.random() < 0.5 else -1.0
        self.curve_mag = float(self.rng.uniform(self.P.curve_curl_min_deg,
                                                self.P.curve_curl_max_deg))

    def decide(self, feat: LocalFeature):
        P = self.P
        if feat.cls != "wall":
            self.jam_count = 0          # the jam counter only accrues against walls
        if feat.cls == "pole":
            rot = float(np.clip(P.k_pole * feat.pole_az_deg,
                                -P.max_turn_deg, P.max_turn_deg))
            return rot, P.drive_mm, "approach"

        if feat.cls == "wall":
            r = feat.slices_mm.get("right", float("nan"))
            c = feat.slices_mm.get("center", float("nan"))
            l = feat.slices_mm.get("left", float("nan"))
            finite = [v for v in (r, c, l) if np.isfinite(v)]
            c_min = min(finite) if finite else float("inf")

            # Corner-jam escape: billiard reflection traps in concave corners (it
            # alternates ±bounce_max_turn without netting an escape and re-drives
            # whenever clearance exceeds wall_jam). When the nearest slice stays
            # below jam_clear_mm for jam_patience steps, stop reflecting and
            # reverse out decisively toward the more-open side — the way we came
            # in is open by construction.
            self.jam_count = self.jam_count + 1 if c_min < P.jam_clear_mm else 0
            if self.jam_count >= P.jam_patience:
                turn = P.escape_turn_deg
                if np.isfinite(r) and np.isfinite(l) and r > l:
                    turn = -turn          # right side more open → turn right
                self.jam_count = 0
                if P.randomize_curve:           # re-roll so we don't arc back in
                    self._reroll_curl()
                return turn, P.drive_mm, "escape"

            # Far wall → arc forward with the current curl (no straights). The
            # magnitude is baseline_curl_deg until a wall contact re-randomises it.
            if c_min >= P.bounce_trigger_mm:
                return self.curl_sign * self.curve_mag, P.drive_mm, "curve"

            # Near wall → reflect. Re-roll the curl bias on each bounce so the
            # post-bounce wander/curve varies and open-space paths don't orbit;
            # without randomisation, the first rebound just fixes the curl sign.
            rot = float(np.clip(_reflect_rotation_deg(r, c, l),
                                -P.bounce_max_turn, P.bounce_max_turn))
            if P.randomize_curve:
                self._reroll_curl()
            elif not self.bounced and abs(rot) > 1e-6:
                self.curl_sign = math.copysign(1.0, rot)
                self.bounced = True
            drive = P.drive_mm if c_min > P.wall_jam_mm else 0.0
            return rot, drive, "bounce"

        # Empty cone / 'none' (nothing within range) → wander forward with a
        # gentle curl so the robot crosses open stretches and sweeps the cone,
        # rather than spinning in place (the 1 m sonar horizon makes 'none'
        # common). The forward cone is clear for >= the horizon, so the step is
        # safe; a wall or pole entering range immediately takes over. Sign
        # follows the baseline-curl bias. EMPTY_WANDER=False keeps in-place scan.
        if P.empty_wander:
            return self.curl_sign * P.wander_curl_deg, P.drive_mm, "wander"
        return P.scan_turn_deg, 0.0, "scan"


# ══════════════════════════════════════════════════════════════════════════════
# Feature producers
# ══════════════════════════════════════════════════════════════════════════════

def feature_from_geometry(x, y, yaw, geom, cone_half_deg,
                          max_range_mm=None) -> LocalFeature:
    """Vision/sim producer: local feature from pose + arena geometry. This is
    the exact supervision signal the inverse was trained against. When
    `max_range_mm` is set, abstain (→ "empty") if the nearest in-cone reflector
    is beyond it — mirroring the 3-class sonar inverse's 'none' class."""
    walls, poles = geom["walls"], geom["poles"]
    cls, pole_az, near = nearest_reflector_in_cone(
        walls, poles, geom["pole_radius_mm"], x, y, yaw, cone_half_deg)
    if not np.isfinite(cls):
        return LocalFeature(cls="empty")
    if max_range_mm is not None and np.isfinite(near) and near > max_range_mm:
        return LocalFeature(cls="empty")   # nothing within range → matches 'none'
    if cls == 1.0:
        return LocalFeature(cls="pole", pole_az_deg=pole_az)
    # Wall: geometric 3-slice (min distance per slice), matching the inverse's
    # wall head and the trainer's compute_slice_targets.
    prof = compute_profile(walls, x, y, yaw,
                           opening_angle=2.0 * cone_half_deg,
                           profile_steps=GEOM_PROFILE_STEPS,
                           profile_method="ray_center")
    slices = _slice_profile(prof, cone_half_deg)
    return LocalFeature(cls="wall", slices_mm=slices)


def feature_from_inverse(pred: Dict) -> LocalFeature:
    """Sonar producer: map InverseModel.predict_from_envelope output to the
    common feature. 3-class inverse: 0=wall, 1=pole, 2=none. The 'none' class
    (nothing within the trained range) maps to "empty" so the controller scans
    instead of chasing a phantom. p_pole / p_none / pole_az_sigma_deg are carried
    through for logging; they stay NaN on the geometry path."""
    p_pole = float(pred.get("p_pole", float("nan")))
    p_none = float(pred.get("p_none", float("nan")))
    cl = int(pred["class_label"])
    if cl == 2:                       # none → abstain (scan), don't chase
        return LocalFeature(cls="empty", p_pole=p_pole, p_none=p_none)
    if cl == 1:
        return LocalFeature(cls="pole", pole_az_deg=float(pred["pole_az_deg"]),
                            p_pole=p_pole, p_none=p_none,
                            pole_az_sigma_deg=float(
                                pred.get("pole_az_sigma_deg", float("nan"))))
    return LocalFeature(cls="wall", p_pole=p_pole, p_none=p_none, slices_mm={
        "right":  float(pred["distance_right_mm"]),
        "center": float(pred["distance_center_mm"]),
        "left":   float(pred["distance_left_mm"]),
    })


# Geometric profile resolution for vision/sim wall slices. Three slices span the
# ±cone, so the steps just need to resolve them; 30 bins over the cone is plenty.
GEOM_PROFILE_STEPS = 30


def _slice_profile(profile, cone_half_deg) -> Dict[str, float]:
    """Min distance in each of the three ±cone slices (right/center/left),
    ordered by ascending azimuth bin to match SLICE_NAMES."""
    edges = np.linspace(-cone_half_deg, cone_half_deg, len(profile) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    third = 2.0 * cone_half_deg / 3.0
    masks = {
        "right":  (centers >= -cone_half_deg)         & (centers < -cone_half_deg + third),
        "center": (centers >= -cone_half_deg + third) & (centers < -cone_half_deg + 2 * third),
        "left":   (centers >= -cone_half_deg + 2 * third) & (centers <= cone_half_deg),
    }
    out = {}
    for name, m in masks.items():
        sub = profile[m]
        sub = sub[np.isfinite(sub)]
        out[name] = float(sub.min()) if sub.size else float("nan")
    return out


def referee(x, y, yaw, geom, cone_half_deg):
    """Ground-truth state for success/collision/logging, independent of the
    sensing modality. Returns (true_cls, pole_near_dist_mm, min_wall_dist_mm)."""
    walls, poles = geom["walls"], geom["poles"]
    cls, _, near = nearest_reflector_in_cone(
        walls, poles, geom["pole_radius_mm"], x, y, yaw, cone_half_deg)
    # Nearest wall *surface* over all directions (not just the cone) for crash.
    min_wall = float("inf")
    if walls.size:
        d = np.hypot(walls[:, 0] - x, walls[:, 1] - y)
        min_wall = float(d.min()) - ROBOT_RADIUS_MM
    # Nearest pole surface over all directions for the success test.
    pole_near = float("inf")
    if poles.size:
        d = np.hypot(poles[:, 0] - x, poles[:, 1] - y) - geom["pole_radius_mm"]
        pole_near = float(d.min())
    return cls, pole_near, min_wall


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

# Per-step "what the robot sees" marker colours, keyed by the feature class the
# active modality (sonar inverse or vision geometry) emitted that step.
_SEES_COLORS = {"pole": "#ff7f0e", "wall": "#1f77b4", "empty": "#999999"}


def save_trajectory_plot(xs, ys, geom, out_path, title, feats=None):
    """Draw the path over the arena. When `feats` (a list of (x, y, cls) at the
    pose each perception was made) is given, colour the per-step markers by what
    the robot saw — pole / wall / empty — instead of the time gradient."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    if geom["walls"].size:
        ax.scatter(geom["walls"][:, 0], geom["walls"][:, 1],
                   color="green", s=2, alpha=0.3, label="walls")
    for px, py in geom["poles"]:
        ax.add_patch(plt.Circle((px, py), geom["pole_radius_mm"],
                                facecolor="#984ea3", edgecolor="black",
                                linewidth=0.6, alpha=0.7))
    if geom["poles"].size:
        ax.scatter([], [], s=40, c="#984ea3", edgecolor="black",
                   linewidth=0.6, label="pole")
    ax.plot(xs, ys, color="black", alpha=0.6, lw=1, label="trajectory")
    if feats:
        for cls, col in _SEES_COLORS.items():
            pts = [(fx, fy) for fx, fy, fc in feats if fc == cls]
            if pts:
                fxs, fys = zip(*pts)
                ax.scatter(fxs, fys, color=col, s=20, zorder=3,
                           label=f"sees {cls}")
    else:
        ax.scatter(xs, ys, c=range(len(xs)), cmap="viridis", s=18, zorder=3)
    if xs:
        ax.scatter([xs[0]], [ys[0]], marker="o", color="red", s=60, zorder=4, label="start")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(title); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Numeric trajectory log (for later re-plotting / analysis)
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v):
    return "" if v is None or not np.isfinite(v) else f"{float(v):.1f}"


def _fmt_p(v):
    return "" if v is None or not np.isfinite(v) else f"{float(v):.3f}"


def open_trajectory_log(out_dir):
    """Open <out_dir>/trajectory.tsv and write the header. Returns (file, writer);
    the caller logs one row per step and closes the file at the end."""
    f = open(os.path.join(out_dir, "trajectory.tsv"), "w", newline="")
    w = csv.writer(f, delimiter="\t")
    w.writerow([
        "step", "x_mm", "y_mm", "yaw_deg",
        "feat_cls", "pole_az_deg", "p_pole", "p_none", "pole_az_sigma_deg",  # steering feature
        "d_right_mm", "d_center_mm", "d_left_mm",        # wall slices (if wall)
        "true_cls", "pole_near_mm", "min_wall_mm",       # ground-truth referee
        "rot_deg", "drive_mm", "tag", "jam",             # action taken (+ jam streak)
    ])
    f.flush()
    return f, w


def log_trajectory_row(f, w, step, x, y, yaw, feat, true_cls, pole_near, min_wall,
                       rot, drive, tag, jam=0):
    """One per-step row, recorded at the pose where the feature was evaluated.
    `jam` is the controller's consecutive near-wall counter at decision time."""
    s = feat.slices_mm
    w.writerow([
        step, f"{x:.1f}", f"{y:.1f}", f"{yaw:.2f}",
        feat.cls, _fmt(feat.pole_az_deg), _fmt_p(feat.p_pole), _fmt_p(feat.p_none),
        _fmt(feat.pole_az_sigma_deg),
        _fmt(s.get("right")), _fmt(s.get("center")), _fmt(s.get("left")),
        "" if not np.isfinite(true_cls) else int(true_cls),
        _fmt(pole_near), _fmt(min_wall),
        f"{rot:.2f}", f"{drive:.1f}", tag, jam,
    ])
    f.flush()


# ══════════════════════════════════════════════════════════════════════════════
# Sim rollout (no robot)
# ══════════════════════════════════════════════════════════════════════════════

def run_sim(geom, P, out_dir):
    rng = np.random.default_rng(SIM_SEED)
    ctrl = ReactiveController(P, seed=SIM_SEED)
    if SIM_START_XY_MM is not None:
        x, y = float(SIM_START_XY_MM[0]), float(SIM_START_XY_MM[1])
    elif geom["walls"].size:
        x, y = float(geom["walls"][:, 0].mean()), float(geom["walls"][:, 1].mean())
    else:
        x, y = 0.0, 0.0
    yaw = float(SIM_START_YAW_DEG)

    xs, ys = [x], [y]
    sees = []                       # (x, y, cls) per decided step, for the plot
    outcome = "max_steps"
    jam_steps = 0                   # consecutive referee-near-wall steps (jam abort)
    f_log, w_log = open_trajectory_log(out_dir)
    for step in range(MAX_STEPS):
        feat = feature_from_geometry(x, y, yaw, geom, CONE_HALF_DEG,
                                     max_range_mm=GEOM_RANGE_HORIZON_MM)
        true_cls, pole_near, min_wall = referee(x, y, yaw, geom, CONE_HALF_DEG)
        jam_steps = jam_steps + 1 if min_wall < JAM_ABORT_CLEAR_MM else 0
        # Pre-emptive stop: the fixed drive step overshoots the stop window — a
        # single step can jump from outside it to inside the pole. Declare
        # success when the *next* approach step would carry us across the
        # threshold, and don't drive, rather than bumping the pole.
        if true_cls == 1.0 and pole_near - P.drive_mm < ROBOT_RADIUS_MM + STOP_MARGIN_MM:
            outcome = "reached_pole"; break
        if min_wall < COLLISION_MM:
            outcome = "collision"; break

        rot, drive, tag = ctrl.decide(feat)
        if SIM_KIN_NOISE:
            rot   += float(rng.normal(0.0, SIM_ROT_NOISE_DEG))
            drive += float(rng.normal(0.0, SIM_DRIVE_NOISE_MM))
            drive = max(0.0, drive)
        turn_lim = max(P.bounce_max_turn, P.escape_turn_deg)
        rot = float(np.clip(rot, -turn_lim, turn_lim))
        # Log at the pose the decision was made from (before the kinematic update).
        log_trajectory_row(f_log, w_log, step, x, y, yaw, feat,
                           true_cls, pole_near, min_wall, rot, drive, tag,
                           jam=ctrl.jam_count)
        sees.append((x, y, feat.cls))
        if jam_steps >= JAM_ABORT_STEPS:
            outcome = "jammed"
            print(f"step {step:3d}: jammed (min_wall<{JAM_ABORT_CLEAR_MM:.0f}mm "
                  f"x{JAM_ABORT_STEPS}) — aborting")
            break
        yaw = ((yaw + rot + 180.0) % 360.0) - 180.0
        rad = math.radians(yaw)
        x += drive * math.cos(rad)
        y += drive * math.sin(rad)
        xs.append(x); ys.append(y)
        print(f"step {step:3d}  {feat.cls:>5}/{tag:<8}  rot={rot:+6.1f}  "
              f"drive={drive:5.1f}  pose=({x:7.0f},{y:7.0f},{yaw:+6.0f})  "
              f"pole_near={pole_near:6.0f}  min_wall={min_wall:6.0f}")

    f_log.close()
    print(f"\nOutcome: {outcome} after {len(xs) - 1} steps.")
    out_path = os.path.join(out_dir, "trajectory.png")
    save_trajectory_plot(xs, ys, geom, out_path,
                         f"{SESSION} [sim] — {outcome} ({len(xs)-1} steps)",
                         feats=sees)
    print(f"Trajectory plot: {out_path}")
    return outcome


# ══════════════════════════════════════════════════════════════════════════════
# Real-robot run (sonar | vision)
# ══════════════════════════════════════════════════════════════════════════════

def rotation_substep_cap(client, fallback=ROT_SUBSTEP_MAX_DEG):
    """Largest faithful per-command rotation for this robot: 0.9× the max
    |rotation_obtained| in its calibration table (Client.get_correction clamps
    targets beyond that, so a bigger command under-rotates). Falls back if the
    table can't be read."""
    try:
        obt = client.configuration.rotation_obtained
        cap = 0.9 * max(abs(float(v)) for v in obt)
        return cap if cap > 1.0 else fallback
    except Exception:
        return fallback


def rotate_in_substeps(client, total_deg, cap_deg, settle_s=0.5):
    """Execute `total_deg` of in-place rotation as N equal sub-rotations, each
    within the calibrated range so it isn't clamped/under-rotated. Returns N."""
    n = max(1, int(math.ceil(abs(total_deg) / cap_deg)))
    per = total_deg / n
    for _ in range(n):
        client.step(angle=per)
        time.sleep(settle_s)
    return n


def copy_arena_source(arena_root, arena, out_dir, dest_name="arena_source"):
    """Copy the full source arena folder (<arena_root>/<arena>, including its
    env_*/arena_features.npz, arena.png, meta.json, …) into the run folder, so
    the results record exactly which arena geometry was used — not just the fresh
    env snapshot taken at deploy time. Returns the destination path, or None if
    the source folder doesn't exist."""
    src = os.path.join(arena_root, arena)
    if not os.path.isdir(src):
        return None
    dst = os.path.join(out_dir, dest_name, arena)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def run_robot(geom, P, out_dir, source, features_path=None):
    from Library import Client
    from Library import LorexTracker
    from Library.TrackerNav import wait_for_stable_pose

    print("\n" + "=" * 78)
    print("  DEPLOY GATE: did you run SCRIPT_CalibrateRobot.py this session?")
    print("  Drive constants drift; a stale model curls the trajectory.")
    print("=" * 78)
    if input("  Robot calibration current? proceed? [y/N]: ").strip().lower() != "y":
        print("Aborted before robot motion."); return None

    # Per-ping raw-data store, mirroring SCRIPT_RunPolicy: one dill per step with
    # the sonar package, pose, inverse prediction and action — so a sonar run can
    # be re-analysed offline. Sonar only (vision has no ping). DataWriter clears
    # <DATA_FOLDER>/<SESSION> on open, so it is created *before* the env/code
    # snapshot below writes into that same folder.
    writer = None
    if source == "sonar":
        from Library import DataStorage
        from Library import Settings as _settings
        _settings.data_folder = DATA_FOLDER
        writer = DataStorage.DataWriter(SESSION, autoclear=True, verbose=False)
        writer.add_file("SCRIPT_RunDirectPolicy.py")

    # Snapshot env + arena geometry + code into the run folder for reproducible
    # offline plotting, mirroring SCRIPT_RunPolicy. Non-fatal if it fails — the
    # run itself matters more than the snapshot.
    try:
        from Library import CodeLogger
        from LorexLib.Environment import capture_environment_layout
        capture_environment_layout(save_root=out_dir)
        if features_path is not None and os.path.exists(features_path):
            shutil.copy(features_path, os.path.join(out_dir, "arena_features.npz"))
        arena_dst = copy_arena_source(ARENAS_ROOT, ARENA, out_dir)
        if arena_dst:
            print(f"Copied source arena folder → {arena_dst}")
        CodeLogger.log_code(out_dir, [".", "Library"], label=SESSION)
        print(f"Snapshotted env + arena features + source arena + code into {out_dir}")
    except Exception as e:
        print(f"  (env/code snapshot skipped: {e})")

    inverse = None
    if source == "sonar":
        from Library.SonarModel import InverseModel
        inverse = InverseModel.load(model_dir="SonarModel", fold=INVERSE_FOLD,
                                    device="cpu")
        print(f"Loaded inverse: {inverse}")

    client  = Client.Client(robot_number=ROBOT_ID)
    tracker = LorexTracker.LorexTracker()
    ctrl    = ReactiveController(P, seed=CONTROLLER_SEED)

    rot_cap = rotation_substep_cap(client)
    print(f"Rotation sub-step cap: {rot_cap:.0f}° (turns beyond this are split so "
          f"the calibration table doesn't clamp them)")

    def settled_pose(prior=None):
        prior_t = None if prior is None else (prior["x"], prior["y"], prior["yaw_deg"])
        pose = wait_for_stable_pose(
            tracker, ROBOT_ID,
            yaw_tol_deg=YAW_STABLE_TOL_DEG, pos_tol_mm=YAW_STABLE_POS_TOL_MM,
            n_consec=YAW_STABLE_N_CONSEC, poll_s=YAW_STABLE_POLL_S,
            timeout_s=YAW_STABLE_TIMEOUT_S, prior_pose=prior_t,
            strict_motion=False, verbose=False)
        return None if pose is None else {"x": pose[0], "y": pose[1], "yaw_deg": pose[2]}

    if source == "sonar":
        for _ in range(5):              # warm up sonar
            client.acquire("ping"); time.sleep(0.5)

    xs, ys = [], []
    sees = []                       # (x, y, cls) per decided step, for the plot
    outcome = "max_steps"
    last = None
    jam_steps = 0                   # consecutive referee-near-wall steps (jam abort)
    f_log, w_log = open_trajectory_log(out_dir)
    for step in range(MAX_STEPS):
        # Sonar must be pinged at the current orientation *before* settling read.
        sonar_pkg = None
        if source == "sonar":
            sonar_pkg = client.read_and_process(do_ping=True, plot=False)
        pose = settled_pose(prior=last)
        if pose is None:
            print(f"step {step:3d}: no tracker pose — skipping"); continue
        x, y, yaw = pose["x"], pose["y"], pose["yaw_deg"]
        xs.append(x); ys.append(y)

        # Referee (ground truth) — success / collision / logging.
        true_cls, pole_near, min_wall = referee(x, y, yaw, geom, CONE_HALF_DEG)
        jam_steps = jam_steps + 1 if min_wall < JAM_ABORT_CLEAR_MM else 0
        # Pre-emptive stop: the fixed drive step overshoots the stop window — a
        # single step can jump from outside it to inside the pole. Declare
        # success when the *next* approach step would carry us across the
        # threshold, and don't drive, rather than bumping the pole.
        if true_cls == 1.0 and pole_near - P.drive_mm < ROBOT_RADIUS_MM + STOP_MARGIN_MM:
            outcome = "reached_pole"; print("  *** pole reached ***"); break

        # Feature from the chosen modality.
        if source == "sonar":
            if sonar_pkg is None:
                print(f"step {step:3d}: no sonar — skipping"); continue
            sd = np.asarray(sonar_pkg["sonar_data"], dtype=np.float32)
            pred = inverse.predict_from_envelope(sd[:, 1], sd[:, 2])
            feat = feature_from_inverse(pred)
        else:  # vision
            feat = feature_from_geometry(x, y, yaw, geom, CONE_HALF_DEG,
                                         max_range_mm=GEOM_RANGE_HORIZON_MM)

        rot, drive, tag = ctrl.decide(feat)
        log_trajectory_row(f_log, w_log, step, x, y, yaw, feat,
                           true_cls, pole_near, min_wall, rot, drive, tag,
                           jam=ctrl.jam_count)
        sees.append((x, y, feat.cls))
        pp = ""
        if np.isfinite(feat.p_pole):
            pp += f"  p_pole={feat.p_pole:.2f}"
        if np.isfinite(feat.p_none):
            pp += f"  p_none={feat.p_none:.2f}"
        print(f"step {step:3d}  {feat.cls:>5}/{tag:<8}  rot={rot:+6.1f}  "
              f"drive={drive:5.1f}  pose=({x:7.0f},{y:7.0f},{yaw:+6.0f})  "
              f"pole_near={pole_near:6.0f}  min_wall={min_wall:6.0f}{pp}")

        # Persist the raw ping + everything derived from it (sonar mode only).
        if writer is not None:
            writer.save_data(
                sonar_package=sonar_pkg,
                position={"x": x, "y": y, "yaw_deg": yaw},
                motion={"net_rotation": rot, "drive_mm": drive},
                inverse_prediction=pred,
                feature={"cls": feat.cls, "pole_az_deg": feat.pole_az_deg,
                         "p_pole": feat.p_pole, "p_none": feat.p_none,
                         "slices_mm": feat.slices_mm},
                referee={"true_cls": (int(true_cls) if np.isfinite(true_cls)
                                      else None),
                         "pole_near_mm": pole_near, "min_wall_mm": min_wall},
                step=step, tag=tag,
            )

        if jam_steps >= JAM_ABORT_STEPS:
            outcome = "jammed"
            print(f"  *** jammed (min_wall<{JAM_ABORT_CLEAR_MM:.0f}mm "
                  f"x{JAM_ABORT_STEPS}) — aborting ***")
            break

        if do_rotation and abs(rot) > 1e-6:
            n_sub = rotate_in_substeps(client, rot, rot_cap)
            if n_sub > 1:
                print(f"    (turn {rot:+.0f}° split into {n_sub}×{rot/n_sub:+.0f}° "
                      f"to stay within the ±{rot_cap:.0f}° calibrated range)")
        if do_translation and drive > 0.0:
            try:
                client.step(distance=drive / 1000.0); time.sleep(0.15)
            except RuntimeError as e:
                print(f"  *** drive aborted (likely against wall): {e} ***")
                outcome = "drive_blocked"; break
        last = pose

        # Live trajectory plot — watch the run build, like SCRIPT_RunPolicy.
        if PLOT_EVERY > 0 and step % PLOT_EVERY == 0:
            save_trajectory_plot(
                xs, ys, geom, os.path.join(out_dir, "trajectory.png"),
                f"{SESSION} [{source}] — step {step}", feats=sees)

    f_log.close()
    print(f"\nOutcome: {outcome} after {len(xs)} poses.")
    out_path = os.path.join(out_dir, "trajectory.png")
    save_trajectory_plot(xs, ys, geom, out_path,
                         f"{SESSION} [{source}] — {outcome} ({len(xs)} steps)",
                         feats=sees)
    print(f"Trajectory plot: {out_path}")
    print(f"Numeric trajectory: {os.path.join(out_dir, 'trajectory.tsv')}")
    if writer is not None:
        print(f"Sonar dills: {writer.get_file_count()} files in {writer.base_folder}")
    return outcome


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def load_arena_geometry(arena_root: str, arena: str):
    """Resolve arena_features.npz for an arena dir, tolerant of the env_* layout
    SCRIPT_BuildArenaGeometry writes: the features live under a timestamped
    env_*/ subdir, not the arena root. Newest env_* wins. Returns (geom, path)
    so the run can copy the exact features file into its output folder."""
    base = Path(arena_root) / arena
    direct = base / "arena_features.npz"
    if direct.exists():
        return _load_features_for_session(base), direct
    if base.exists():
        env_dirs = sorted(p for p in base.iterdir()
                          if p.is_dir() and p.name.startswith("env_"))
        for env in reversed(env_dirs):
            wp = env / "arena_features.npz"
            if wp.exists():
                print(f"Using arena geometry: {wp}")
                return _load_features_for_session(env), wp
    # Fall back to the loader's own resolution (session_meta.json), or its error.
    return _load_features_for_session(base), None


def resolve_trial():
    """Look this run's (POLE, START) up in the agreed trial list.

    Returns the trial record, or None when the list is missing or the pair is
    not in it. A missing list is not fatal -- exploratory runs are legitimate --
    but an unrecognised pair is worth shouting about, because the usual cause is
    a stale POLE or START left over from the previous run.
    """
    path = Path(TRIAL_LIST)
    if not path.exists():
        print(f"  (no trial list at {TRIAL_LIST} — running unchecked)")
        return None
    with open(path) as fh:
        trials = json.load(fh).get("trials", [])
    for t in trials:
        if t.get("pole") == POLE and int(t.get("start", -1)) == START:
            return t
    agreed = sorted({(t["pole"], t["start"]) for t in trials})
    print(f"\n  !! {POLE} x S{START} is NOT in the agreed trial list.")
    print(f"     Agreed trials: {agreed}")
    print(f"     Continuing anyway — but check POLE/START before using this run.")
    return None


def check_arena_matches_pole(geom, trial):
    """Refuse to run when the built arena is not the one for this placement.

    The referee scores against the pole in arena_features.npz. If ARENA still
    points at the previous placement's build, every run scores against a pole
    that is not physically there -- and nothing about the run looks wrong until
    the analysis. Cheap to check, expensive to miss.
    """
    if trial is None:
        return
    want = (float(trial["pole_x_mm"]), float(trial["pole_y_mm"]))
    poles = geom["poles"]
    d = np.hypot(poles[:, 0] - want[0], poles[:, 1] - want[1])
    i = int(np.argmin(d))
    if d[i] > POLE_POSITION_TOL_MM:
        raise SystemExit(
            f"\nArena/pole mismatch — refusing to run.\n"
            f"  {POLE} should be near ({want[0]:.0f}, {want[1]:.0f}).\n"
            f"  Nearest pole in TargetArenas/{ARENA} is "
            f"({poles[i, 0]:.0f}, {poles[i, 1]:.0f}), {d[i]:.0f} mm away "
            f"(tolerance {POLE_POSITION_TOL_MM:.0f} mm).\n"
            f"  Either ARENA points at the wrong build, or the pole was placed "
            f"somewhere other than {POLE}. Re-snapshot/annotate/build, or fix "
            f"ARENA.")
    print(f"  arena pole ({poles[i, 0]:.0f}, {poles[i, 1]:.0f}) is {d[i]:.0f} mm "
          f"from the intended {POLE} — OK")


def main():
    print(f"\nTrial: {POLE} x S{START}  [{SENSE_SOURCE}]  -> {SESSION}")
    trial = resolve_trial()
    if trial is not None:
        print(f"  expected: {trial['range_mm']:.0f} mm, "
              f"{trial['bearing_deg']:+.0f} deg off heading, "
              f"~{trial['sim_steps']} steps ({trial['hidden_by']})")
    if SENSE_SOURCE != "sim":
        print(f"  controller seed {CONTROLLER_SEED} "
              f"(same for sonar and vision — matched pair)")

    geom, features_path = load_arena_geometry(ARENAS_ROOT, ARENA)
    print(f"Arena '{ARENA}': {geom['walls'].shape[0]} wall pts, "
          f"{geom['poles'].shape[0]} poles (r={geom['pole_radius_mm']:.1f} mm)")
    if geom["poles"].shape[0] == 0:
        raise SystemExit("No poles in arena_features.npz — nothing to approach.")
    check_arena_matches_pole(geom, trial)

    out_dir = os.path.join(DATA_FOLDER, SESSION)
    os.makedirs(out_dir, exist_ok=True)
    P = ReactiveParams()

    if SENSE_SOURCE == "sim":
        run_sim(geom, P, out_dir)
    elif SENSE_SOURCE in ("sonar", "vision"):
        run_robot(geom, P, out_dir, SENSE_SOURCE, features_path)
    else:
        raise SystemExit(f"Unknown SENSE_SOURCE={SENSE_SOURCE!r}")


if __name__ == "__main__":
    main()
