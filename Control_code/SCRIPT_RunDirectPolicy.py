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
    class ∈ {wall, pole};  if wall → (right, center, left) slice distances;
                           if pole → signed azimuth (+ccw = LEFT).
Sonar gets it from the two-headed inverse's heads; vision/sim get it from
geometry (`nearest_reflector_in_cone` + `compute_profile`), which is exactly
the supervision target the inverse was trained against.

The reactive rule (`ReactiveController`):
  - nearest object is a POLE → steer to null its azimuth (keep it in front),
    drive a fixed step → closes in.
  - nearest object is a WALL → fly a gentle arc (a constant baseline curl) until
    a wall comes within range, then reflect off it (billiard rebound estimated
    from the three slice depths). The curl sign is random until the first
    rebound, then fixed to that rebound's direction.
  - cone empty → rotate in place to scan.
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
import math
import os
import shutil
import time
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

SENSE_SOURCE = "sim"          # "sonar" | "vision" | "sim"

ARENA        = "DirectTarget01"     # sub-folder under TargetArenas/ (arena_features.npz)
INVERSE_FOLD = "q0"           # which CV fold of the inverse to deploy (sonar mode)
SESSION      = "direct_pole_demo"

ROBOT_ID     = 1
MAX_STEPS    = 200

# ── Sensing cone ──────────────────────────────────────────────────────────────
# Forward ±half-angle the robot "sees". Defaults to the sonar inverse's cone so
# the two modalities share an aperture; settable independently for vision/sim.
CONE_HALF_DEG = 35.0

# ── Reactive rule constants (curved-bounce wall rule) ─────────────────────────
DRIVE_MM           = 50.0   # nominal forward step per cycle (mm)
MAX_TURN_DEG       = 25.0   # cap on pole-steering / scan rotation per step
K_POLE             = 0.6    # pole steering gain: rotate = K_POLE * pole_azimuth
BASELINE_CURL_DEG  = 5.0    # constant arc applied while cruising past a far wall;
                            # sign random until the first rebound, then fixed to it
BOUNCE_TRIGGER_MM  = 350.0  # wall closer than this in the cone → reflect (bounce)
BOUNCE_MAX_TURN    = 110.0  # cap on a rebound turn (lets a head-on wall reverse)
WALL_JAM_MM        = 180.0  # below this, rebound with no drive (don't push into wall)
SCAN_TURN_DEG      = 20.0   # in-place rotation when the cone is empty

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
              the wall.
      empty → rotate in place to scan.

    Perception is memoryless — the only carried state is a single motor bias:
    the baseline-curl sign, random until the first rebound and then fixed to that
    rebound's direction. This is a motor prior, not perceptual memory; it removes
    the unnatural dead-straight cruise of the pure rule without consulting any
    map or past observation. `reset(seed)` re-randomises the sign for a new run.
    """

    def __init__(self, P: ReactiveParams, seed: int = 0):
        self.P = P
        self.reset(seed)

    def reset(self, seed: int = 0):
        self.curl_sign = 1.0 if np.random.default_rng(seed).random() < 0.5 else -1.0
        self.bounced = False

    def decide(self, feat: LocalFeature):
        P = self.P
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

            # Far wall → arc forward with the baseline curl (no straights).
            if c_min >= P.bounce_trigger_mm:
                return self.curl_sign * P.baseline_curl_deg, P.drive_mm, "curve"

            # Near wall → reflect. The first rebound sets the curl sign.
            rot = float(np.clip(_reflect_rotation_deg(r, c, l),
                                -P.bounce_max_turn, P.bounce_max_turn))
            if not self.bounced and abs(rot) > 1e-6:
                self.curl_sign = math.copysign(1.0, rot)
                self.bounced = True
            drive = P.drive_mm if c_min > P.wall_jam_mm else 0.0
            return rot, drive, "bounce"

        # Empty cone → scan in place.
        return P.scan_turn_deg, 0.0, "scan"


# ══════════════════════════════════════════════════════════════════════════════
# Feature producers
# ══════════════════════════════════════════════════════════════════════════════

def feature_from_geometry(x, y, yaw, geom, cone_half_deg) -> LocalFeature:
    """Vision/sim producer: local feature from pose + arena geometry. This is
    the exact supervision signal the inverse was trained against."""
    walls, poles = geom["walls"], geom["poles"]
    cls, pole_az, _ = nearest_reflector_in_cone(
        walls, poles, geom["pole_radius_mm"], x, y, yaw, cone_half_deg)
    if not np.isfinite(cls):
        return LocalFeature(cls="empty")
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
    common feature."""
    if int(pred["class_label"]) == 1:
        return LocalFeature(cls="pole", pole_az_deg=float(pred["pole_az_deg"]))
    return LocalFeature(cls="wall", slices_mm={
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

def save_trajectory_plot(xs, ys, geom, out_path, title):
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


def open_trajectory_log(out_dir):
    """Open <out_dir>/trajectory.tsv and write the header. Returns (file, writer);
    the caller logs one row per step and closes the file at the end."""
    f = open(os.path.join(out_dir, "trajectory.tsv"), "w", newline="")
    w = csv.writer(f, delimiter="\t")
    w.writerow([
        "step", "x_mm", "y_mm", "yaw_deg",
        "feat_cls", "pole_az_deg",                       # steering feature
        "d_right_mm", "d_center_mm", "d_left_mm",        # wall slices (if wall)
        "true_cls", "pole_near_mm", "min_wall_mm",       # ground-truth referee
        "rot_deg", "drive_mm", "tag",                    # action taken
    ])
    f.flush()
    return f, w


def log_trajectory_row(f, w, step, x, y, yaw, feat, true_cls, pole_near, min_wall,
                       rot, drive, tag):
    """One per-step row, recorded at the pose where the feature was evaluated."""
    s = feat.slices_mm
    w.writerow([
        step, f"{x:.1f}", f"{y:.1f}", f"{yaw:.2f}",
        feat.cls, _fmt(feat.pole_az_deg),
        _fmt(s.get("right")), _fmt(s.get("center")), _fmt(s.get("left")),
        "" if not np.isfinite(true_cls) else int(true_cls),
        _fmt(pole_near), _fmt(min_wall),
        f"{rot:.2f}", f"{drive:.1f}", tag,
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
    outcome = "max_steps"
    f_log, w_log = open_trajectory_log(out_dir)
    for step in range(MAX_STEPS):
        feat = feature_from_geometry(x, y, yaw, geom, CONE_HALF_DEG)
        true_cls, pole_near, min_wall = referee(x, y, yaw, geom, CONE_HALF_DEG)
        if true_cls == 1.0 and pole_near < ROBOT_RADIUS_MM + STOP_MARGIN_MM:
            outcome = "reached_pole"; break
        if min_wall < COLLISION_MM:
            outcome = "collision"; break

        rot, drive, tag = ctrl.decide(feat)
        if SIM_KIN_NOISE:
            rot   += float(rng.normal(0.0, SIM_ROT_NOISE_DEG))
            drive += float(rng.normal(0.0, SIM_DRIVE_NOISE_MM))
            drive = max(0.0, drive)
        rot = float(np.clip(rot, -P.bounce_max_turn, P.bounce_max_turn))
        # Log at the pose the decision was made from (before the kinematic update).
        log_trajectory_row(f_log, w_log, step, x, y, yaw, feat,
                           true_cls, pole_near, min_wall, rot, drive, tag)
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
                         f"{SESSION} [sim] — {outcome} ({len(xs)-1} steps)")
    print(f"Trajectory plot: {out_path}")
    return outcome


# ══════════════════════════════════════════════════════════════════════════════
# Real-robot run (sonar | vision)
# ══════════════════════════════════════════════════════════════════════════════

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

    # Snapshot env + arena geometry + code into the run folder for reproducible
    # offline plotting, mirroring SCRIPT_RunPolicy. Non-fatal if it fails — the
    # run itself matters more than the snapshot.
    try:
        from Library import CodeLogger
        from LorexLib.Environment import capture_environment_layout
        capture_environment_layout(save_root=out_dir)
        if features_path is not None and os.path.exists(features_path):
            shutil.copy(features_path, os.path.join(out_dir, "arena_features.npz"))
        CodeLogger.log_code(out_dir, [".", "Library"], label=SESSION)
        print(f"Snapshotted env + arena features + code into {out_dir}")
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
    ctrl    = ReactiveController(P)

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
    outcome = "max_steps"
    last = None
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
        if true_cls == 1.0 and pole_near < ROBOT_RADIUS_MM + STOP_MARGIN_MM:
            outcome = "reached_pole"; print("  *** pole reached ***"); break

        # Feature from the chosen modality.
        if source == "sonar":
            if sonar_pkg is None:
                print(f"step {step:3d}: no sonar — skipping"); continue
            sd = np.asarray(sonar_pkg["sonar_data"], dtype=np.float32)
            pred = inverse.predict_from_envelope(sd[:, 1], sd[:, 2])
            feat = feature_from_inverse(pred)
        else:  # vision
            feat = feature_from_geometry(x, y, yaw, geom, CONE_HALF_DEG)

        rot, drive, tag = ctrl.decide(feat)
        log_trajectory_row(f_log, w_log, step, x, y, yaw, feat,
                           true_cls, pole_near, min_wall, rot, drive, tag)
        print(f"step {step:3d}  {feat.cls:>5}/{tag:<8}  rot={rot:+6.1f}  "
              f"drive={drive:5.1f}  pose=({x:7.0f},{y:7.0f},{yaw:+6.0f})  "
              f"pole_near={pole_near:6.0f}  min_wall={min_wall:6.0f}")

        if do_rotation and abs(rot) > 1e-6:
            client.step(angle=rot); time.sleep(0.5)
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
                f"{SESSION} [{source}] — step {step}")

    f_log.close()
    print(f"\nOutcome: {outcome} after {len(xs)} poses.")
    out_path = os.path.join(out_dir, "trajectory.png")
    save_trajectory_plot(xs, ys, geom, out_path,
                         f"{SESSION} [{source}] — {outcome} ({len(xs)} steps)")
    print(f"Trajectory plot: {out_path}")
    print(f"Numeric trajectory: {os.path.join(out_dir, 'trajectory.tsv')}")
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


def main():
    geom, features_path = load_arena_geometry(ARENAS_ROOT, ARENA)
    print(f"Arena '{ARENA}': {geom['walls'].shape[0]} wall pts, "
          f"{geom['poles'].shape[0]} poles (r={geom['pole_radius_mm']:.1f} mm)")
    if geom["poles"].shape[0] == 0:
        raise SystemExit("No poles in arena_features.npz — nothing to approach.")

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
