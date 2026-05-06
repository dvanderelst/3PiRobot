#!/usr/bin/env python3
"""
SCRIPT_RunPolicy.py

Deploy a vanilla-RNN policy trained by SCRIPT_TrainPolicy.py on the real robot.

Per-step sequence (matches training rollout exactly):
  1. Sonar ping → L/R envelopes → SonarModel.predict_from_envelope
                   → 6-key dict (3 distances + 3 σs).
  2. policy.encode_obs(meas, prev_rot) → obs vector (clamps + scales applied).
  3. policy.step(obs, hidden) → rotate_deg, new hidden state.
  4. Rotate body by rotate_deg.
  5. Drive forward policy.fixed_drive_mm.
  6. prev_rot = commanded rotation (NOT motor-noisy executed value).

State carried across steps:
  - hidden (numpy float32, shape (hidden_size,)), zero-initialised.
  - prev_rot (degrees), starts at 0.0.

The entire policy I/O — clamps, scales, channel order, forward pass — comes
from `best_policy.json` via Library.Policy. There is no parallel config in this
script; if the training-side encoding changes, deployment picks it up
automatically the next time the artifact is reloaded.
"""

import json
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np

from Library import Client
from Library import CodeLogger
from Library import DataStorage
from Library import Dialog
from Library import LorexTracker
from Library import PauseControl
from Library import PushOver
from Library import Settings as _settings
from Library.EnvironmentSimulator import EnvironmentSimulator
from Library.Policy import Policy
from Library.SonarModel import SonarModel
from LorexLib.Environment import capture_environment_layout


# ══════════════════════════════════════════════════════════════════════════════
# Settings — edit these
# ══════════════════════════════════════════════════════════════════════════════
POLICY    = "rnn_sup_loop2_h32_nosigma"   # sub-folder under PolicyTraining/
ARENA     = "loop2"                       # sub-folder under TargetArenas/
REPEAT    = "13"

MAX_STEPS = 500

ROBOT_ID    = 1
POLICY_FILE = "best_policy.json"
SESSION     = f"{POLICY}_run{REPEAT}"

# Dry-run flags (set False for tethered debugging without motion)
do_rotation    = True
do_translation = True

# Source of distances/σs fed to the policy. "live" is the SonarModel's prediction
# from the actual ping (the normal path). "sim" is the simulator's prediction at
# the tracker pose with σ_sim noise (training-distribution match). "sim_clean"
# is the noiseless geometric per-slice min from the arena map. Set to "sim" or
# "sim_clean" to bypass the SonarModel entirely and validate the rest of the
# control loop (rotation / drive calibration, policy, simulator geometry) as a
# sim-to-real sanity check.
POLICY_INPUT_SOURCE = "live"   # "live" | "sim" | "sim_clean"

# After a step (especially a sharp turn) the overhead tracker takes ~1–2 s to
# converge on the new pose; reading immediately gives a stale yaw and feeds the
# wrong geometric profile to the policy. Wait until the tracker yaw is stable
# (last N reads within YAW_STABLE_TOL_DEG) before using its value. Verbose flag
# prints each poll for early debugging.
YAW_STABLE_TOL_DEG    = 0.5    # max spread (deg) across the rolling window
YAW_STABLE_N_CONSEC   = 3      # how many consecutive in-tol reads required
YAW_STABLE_POLL_S     = 0.1    # seconds between polls
YAW_STABLE_TIMEOUT_S  = 3.0    # give up after this; use last available pose
YAW_STABLE_VERBOSE    = False  # print per-poll trace; flip on for debugging

PLOT_EVERY            = 1      # save trajectory plot every N steps (0 = disable)
wait_for_confirmation = False

# Pre-flight simulation preview. Before driving the real robot, read the
# physical start pose, jitter it with small noise, and roll the policy
# forward in the geometric simulator from each jittered start. Plot the
# ensemble against the arena walls so we can see whether the policy
# *should* succeed from this pose, and how sensitive the trajectory is
# to small starting-pose error. Set PREVIEW_N=0 to skip.
PREVIEW_N             = 8       # number of jittered rollouts (0 disables preview)
PREVIEW_STEPS         = 200     # max steps per rollout
PREVIEW_NOISE_XY_MM   = 50.0    # σ of Gaussian xy jitter on start pose (mm)
PREVIEW_NOISE_YAW_DEG = 5.0     # σ of Gaussian yaw jitter on start pose (deg)
PREVIEW_COLLISION_MM  = 50.0    # stop a rollout when min profile dist < this
PREVIEW_REQUIRE_OK    = True    # block before main loop until user confirms preview

# Training-matched kinematic noise applied to each non-canonical preview
# rollout — mirror of the per-episode gain bias and per-step Gaussian used
# during policy training in SCRIPT_TrainPolicy.py. The canonical (no-noise)
# rollout always runs first as a reference. Set PREVIEW_KIN_NOISE = False to
# show only the canonical no-noise trajectory.
PREVIEW_KIN_NOISE             = True
PREVIEW_ROT_GAIN_RANGE_PCT    = 0.15   # rot_gain ~ U(1-x, 1+x) per rollout
PREVIEW_DRIVE_GAIN_RANGE_PCT  = 0.05   # drive_gain ~ U(1-x, 1+x) per rollout
PREVIEW_ROTATE_NOISE_DEG      = 3.0    # σ of per-step Gaussian on rotation
PREVIEW_DRIVE_NOISE_MM        = 5.0    # σ of per-step Gaussian on drive

POLICY_DIR     = "PolicyTraining"
SONAR_MODEL_DIR = "SonarModel"
DATA_FOLDER    = "PolicyRuns"


# ══════════════════════════════════════════════════════════════════════════════
# Load policy + sonar model
# ══════════════════════════════════════════════════════════════════════════════

policy_path = os.path.join(POLICY_DIR, POLICY, POLICY_FILE)
policy = Policy.load(policy_path)
print(f"Loaded policy: {policy_path}")
print(f"  {policy}")
print(f"  obs_layout: {policy.obs_layout}")

sonar_model = SonarModel.load(model_dir=SONAR_MODEL_DIR, device="cpu")
print(f"Loaded sonar model: {sonar_model}")


# ══════════════════════════════════════════════════════════════════════════════
# Environment simulator (for sim-to-real diagnostic at tracker pose)
# ══════════════════════════════════════════════════════════════════════════════

_settings.data_folder = "TargetArenas"
sim = EnvironmentSimulator(ARENA, sonar_model_dir=SONAR_MODEL_DIR)
_settings.data_folder = DATA_FOLDER


# ══════════════════════════════════════════════════════════════════════════════
# Session bookkeeping
# ══════════════════════════════════════════════════════════════════════════════

session_folder = os.path.join(DATA_FOLDER, SESSION)
if os.path.exists(session_folder) and os.listdir(session_folder):
    response = input(f"Session folder '{session_folder}' already exists and is "
                     f"non-empty. Overwrite? [y/N]: ")
    if response.strip().lower() != "y":
        print("Aborted.")
        raise SystemExit(0)

def _install_training_walls(arena: str, deploy_env_dir: str) -> None:
    """Copy training-arena wall geometry into the deploy env folder and draw
    them over the deploy arena.png as a sanity-check overlay.

    Deployment assumes the robot runs in the same physical arena it was
    vicariously trained on (cameras are ceiling-mounted and stationary).
    The training walls become the ground truth for the trajectory plot;
    the overlay lets the user verify visually that no obstacle has moved.
    """
    src_root = os.path.join("TargetArenas", arena)
    env_subdirs = sorted(d for d in os.listdir(src_root)
                         if d.startswith("env_")
                         and os.path.isdir(os.path.join(src_root, d)))
    if not env_subdirs:
        raise FileNotFoundError(f"No env_* folder under {src_root}")
    src_env   = os.path.join(src_root, env_subdirs[-1])
    src_walls = os.path.join(src_env, "arena_walls.npz")
    if not os.path.exists(src_walls):
        raise FileNotFoundError(f"arena_walls.npz not found in {src_env}")

    dst_walls = os.path.join(deploy_env_dir, "arena_walls.npz")
    shutil.copy(src_walls, dst_walls)
    print(f"Copied training arena walls: {src_walls} → {dst_walls}")

    meta_path = os.path.join(deploy_env_dir, "meta.json")
    arena_png = os.path.join(deploy_env_dir, "arena.png")
    if not (os.path.exists(meta_path) and os.path.exists(arena_png)):
        print("  meta.json or arena.png missing — skipping overlay.")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    bounds    = meta["arena_bounds_mm"]
    mm_per_px = float(meta["map_mm_per_px"])
    min_x     = float(bounds["min_x"])
    max_y     = float(bounds["max_y"])

    walls = np.load(dst_walls)
    wx, wy = walls["x_mm"], walls["y_mm"]
    # Inverse of mask2coordinates pixel-centre convention (X = min_x + (c+0.5)·mm_per_px).
    col = (wx - min_x) / mm_per_px - 0.5
    row = (max_y - wy) / mm_per_px - 0.5

    img = plt.imread(arena_png)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.scatter(col, row, color="lime", s=2, alpha=0.6, label="training walls")
    ax.set_title(f"{arena}: training walls over current arena view "
                 f"(boxes should align with green points)")
    ax.legend(loc="best")
    ax.set_axis_off()
    fig.tight_layout()
    out_path = os.path.join(deploy_env_dir, "arena_walls_overlay.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote walls overlay: {out_path}")


control = PauseControl.PauseControl()
client  = Client.Client(robot_number=ROBOT_ID)
tracker = LorexTracker.LorexTracker()
writer  = DataStorage.DataWriter(SESSION, autoclear=True, verbose=False)
writer.add_file("SCRIPT_RunPolicy.py")
writer.add_file(policy_path)
snapshot = capture_environment_layout(save_root=f"{DATA_FOLDER}/{SESSION}")
_install_training_walls(ARENA, snapshot["rundir"])
CodeLogger.log_code(f"{DATA_FOLDER}/{SESSION}", [".", "Library"], label=SESSION)


# Arena walls for live trajectory plotting
_arena_walls_x = None
_arena_walls_y = None
_env_dir = snapshot.get("rundir")
if _env_dir:
    _walls_path = os.path.join(_env_dir, "arena_walls.npz")
    if os.path.exists(_walls_path):
        _w = np.load(_walls_path)
        _arena_walls_x = _w["x_mm"]
        _arena_walls_y = _w["y_mm"]
        print(f"Arena walls loaded: {len(_arena_walls_x)} points from {_walls_path}")
    else:
        print("No arena_walls.npz found — trajectory plot will show path only")

_traj_x:   list = []
_traj_y:   list = []
_traj_yaw: list = []
_traj_plot_path = f"{DATA_FOLDER}/{SESSION}/trajectory.png"

# ── Per-step metrics TSV — for offline analysis of model vs sim ──────────────
import csv
_metrics_path = f"{DATA_FOLDER}/{SESSION}/step_metrics.tsv"
_metrics_file = open(_metrics_path, "w", newline="")
_metrics_writer = csv.writer(_metrics_file, delimiter="\t")
_metrics_writer.writerow([
    "step",
    "x_mm", "y_mm", "yaw_deg",
    # Model output (envelope → 3-slice).
    "d_live_right_mm",  "d_live_center_mm",  "d_live_left_mm",
    "s_live_right_mm",  "s_live_center_mm",  "s_live_left_mm",
    # Simulator at the tracker pose: noiseless geometric truth.
    "d_clean_right_mm", "d_clean_center_mm", "d_clean_left_mm",
    # Simulator at the tracker pose: with σ_sim Gaussian noise (training match).
    "d_sim_right_mm",   "d_sim_center_mm",   "d_sim_left_mm",
    "s_sim_right_mm",   "s_sim_center_mm",   "s_sim_left_mm",
    # Policy action.
    "rot_deg",
    # Envelope summary for sanity-checking input distribution.
    "envL_min", "envL_max", "envR_min", "envR_max",
])
_metrics_file.flush()


def _write_metrics_row(step, position, meas_live, meas_clean, meas_sim,
                       rotate, L, R):
    def _g(d, k):
        return float(d[k]) if d is not None and k in d and d[k] is not None else float("nan")
    pos = position or {}
    _metrics_writer.writerow([
        step,
        pos.get("x", float("nan")),
        pos.get("y", float("nan")),
        pos.get("yaw_deg", float("nan")),
        _g(meas_live, "distance_right_mm"),
        _g(meas_live, "distance_center_mm"),
        _g(meas_live, "distance_left_mm"),
        _g(meas_live, "sigma_right_mm"),
        _g(meas_live, "sigma_center_mm"),
        _g(meas_live, "sigma_left_mm"),
        _g(meas_clean, "distance_right_mm"),
        _g(meas_clean, "distance_center_mm"),
        _g(meas_clean, "distance_left_mm"),
        _g(meas_sim,   "distance_right_mm"),
        _g(meas_sim,   "distance_center_mm"),
        _g(meas_sim,   "distance_left_mm"),
        _g(meas_sim,   "sigma_right_mm"),
        _g(meas_sim,   "sigma_center_mm"),
        _g(meas_sim,   "sigma_left_mm"),
        rotate,
        float(L.min()) if L is not None else float("nan"),
        float(L.max()) if L is not None else float("nan"),
        float(R.min()) if R is not None else float("nan"),
        float(R.max()) if R is not None else float("nan"),
    ])
    _metrics_file.flush()


def _yaw_spread(yaws):
    """Max wrapped pairwise difference (deg) across a sequence of yaws."""
    a = np.asarray(yaws, dtype=float)
    diffs = a.reshape(-1, 1) - a.reshape(1, -1)
    diffs = ((diffs + 180.0) % 360.0) - 180.0
    return float(np.abs(diffs).max())


def _wait_yaw_stable(robot_id,
                     tol_deg=YAW_STABLE_TOL_DEG,
                     n_consec=YAW_STABLE_N_CONSEC,
                     poll_s=YAW_STABLE_POLL_S,
                     timeout_s=YAW_STABLE_TIMEOUT_S,
                     verbose=YAW_STABLE_VERBOSE,
                     tag=""):
    """Poll the tracker until yaw stabilises (last `n_consec` reads have
    wrap-aware max-spread < `tol_deg`) or `timeout_s` elapses. Returns the
    last successfully-read position dict (or None if the tracker never
    answered). Used at the top of each policy iteration: after a step the
    overhead tracker takes ~1–2 s to converge on the new pose, and using
    a stale yaw feeds the simulator/policy a wrong geometric profile."""
    history_yaw = []
    last_pos    = None
    t0          = time.time()
    converged   = False
    while time.time() - t0 < timeout_s:
        pos = tracker.get_position(robot_id)
        elapsed_ms = (time.time() - t0) * 1000.0
        if pos is not None:
            last_pos = pos
            yaw = pos.get("yaw_deg")
            if yaw is not None:
                history_yaw.append(float(yaw))
                if verbose:
                    print(f"    [yaw_stable{tag}] t+{elapsed_ms:>4.0f}ms  "
                          f"yaw={yaw:+7.2f}°  n={len(history_yaw)}", end="")
                if len(history_yaw) >= n_consec:
                    spread = _yaw_spread(history_yaw[-n_consec:])
                    if verbose:
                        print(f"  spread(last {n_consec})={spread:.2f}°")
                    if spread < tol_deg:
                        converged = True
                        break
                elif verbose:
                    print()  # newline (no spread yet)
        elif verbose:
            print(f"    [yaw_stable{tag}] t+{elapsed_ms:>4.0f}ms  no pose")
        time.sleep(poll_s)
    if verbose:
        total_ms = (time.time() - t0) * 1000.0
        if converged:
            print(f"    [yaw_stable{tag}] converged after {total_ms:.0f}ms "
                  f"(n_polls={len(history_yaw)})")
        else:
            print(f"    [yaw_stable{tag}] TIMEOUT at {total_ms:.0f}ms; "
                  f"using last pose (n_polls={len(history_yaw)})")
    return last_pos


def _interp_gaps(arr):
    """Linear interpolation through NaN gaps; flat-extrapolation at edges."""
    out = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(out)
    if valid.all() or not valid.any():
        return out
    idx = np.arange(len(out))
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def _save_trajectory_plot() -> None:
    if not _traj_x:
        return
    xs_raw = np.asarray(_traj_x, dtype=np.float64)
    ys_raw = np.asarray(_traj_y, dtype=np.float64)
    if not (np.isfinite(xs_raw) & np.isfinite(ys_raw)).any():
        return
    xs = _interp_gaps(xs_raw)
    ys = _interp_gaps(ys_raw)
    fig, ax = plt.subplots(figsize=(10, 8))
    if _arena_walls_x is not None:
        ax.scatter(_arena_walls_x, _arena_walls_y,
                   color="green", s=2, alpha=0.3, label="Walls")
    ax.plot(xs, ys, color="black", alpha=0.5, linewidth=1, label="Trajectory")
    ax.scatter(xs, ys, color="blue", s=15, zorder=3)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i % max(1, PLOT_EVERY) == 0:
            ax.text(x, y, str(i), color="red", fontsize=7)
    ax.set_title(f"{SESSION}  —  step {len(_traj_x) - 1}")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(_traj_plot_path, dpi=100)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Sonar envelope extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_lr_envelope(sonar_package: dict):
    """Return (L, R) float32 envelopes from a Client.read_and_process result.

    Client returns sonar_data with columns [emitter, left, right]; DataProcessor
    strips the emitter column at training time, so training and deploy must
    use the same L, R indices into Client's column ordering — index 1 and 2.
    """
    sonar_data = np.asarray(sonar_package["sonar_data"], dtype=np.float32)
    return sonar_data[:, 1], sonar_data[:, 2]


# ══════════════════════════════════════════════════════════════════════════════
# Warm up sonar
# ══════════════════════════════════════════════════════════════════════════════

for _ in range(5):
    client.acquire("ping")
    time.sleep(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# RNN persistent state — zero-initialised (matches training)
# ══════════════════════════════════════════════════════════════════════════════

hidden   = policy.initial_hidden()
prev_rot = 0.0

crash_log_path = f"{DATA_FOLDER}/{SESSION}/crashes.tsv"
last_position  = None


def _log_crash(step_idx, position):
    pos = position or {}
    x, y, yaw = pos.get("x"), pos.get("y"), pos.get("yaw_deg")
    write_header = not os.path.exists(crash_log_path)
    with open(crash_log_path, "a") as f:
        if write_header:
            f.write("step\tx\ty\tyaw_deg\n")
        f.write(f"{step_idx}\t{x}\t{y}\t{yaw}\n")
    print(f"  *** Crash logged (step {step_idx}): x={x}, y={y}, yaw={yaw} ***")


# ══════════════════════════════════════════════════════════════════════════════
# Pre-flight: simulate policy from current physical pose
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_rollout(x0, y0, yaw0, n_steps,
                      rot_gain=1.0, drive_gain=1.0,
                      rotate_noise_deg=0.0, drive_noise_mm=0.0,
                      rng=None):
    """Roll policy forward through the geometric simulator from (x0, y0, yaw0).
    Uses sim_clean as the policy input (per-slice min of the geometric profile,
    σ=0).

    Kinematic noise injection (mirrors training in SCRIPT_TrainPolicy.py):
      - `rot_gain`, `drive_gain`: per-rollout multiplicative biases applied
        for the whole rollout. Defaults of 1.0 reproduce the original
        no-noise canonical trajectory.
      - `rotate_noise_deg`, `drive_noise_mm`: σ of zero-mean Gaussian noise
        added per step. Pass an `rng` for reproducibility; if None, a fresh
        default_rng() is used.
    `prev_rot` fed back to the policy is always the *commanded* rotation,
    matching deploy: the robot reports back what it asked the motors to do,
    not what they did.

    Returns (xs, ys, yaws, end_reason) where end_reason is 'ok', 'collision',
    or 'profile_fail'."""
    import math
    if rng is None:
        rng = np.random.default_rng()
    hidden   = policy.initial_hidden()
    prev_rot = 0.0
    x, y, yaw = float(x0), float(y0), float(yaw0)
    xs, ys, yaws = [x], [y], [yaw]
    end_reason = "ok"
    slice_masks  = sim.sonar_model.slice_masks
    slice_names  = sim.sonar_model.SLICE_NAMES
    drive_mm     = float(policy.fixed_drive_mm)
    max_rot_deg  = float(policy.max_rotate_deg)
    for _ in range(n_steps):
        try:
            profile = sim.get_profile_at_position(x, y, yaw)
        except Exception:
            end_reason = "profile_fail"
            break
        if not np.isfinite(profile).any():
            end_reason = "profile_fail"
            break
        meas = {
            f"distance_{name}_mm": float(profile[slice_masks[k]].min())
            for k, name in enumerate(slice_names)
        }
        for name in slice_names:
            meas[f"sigma_{name}_mm"] = 0.0
        if min(meas[f"distance_{n}_mm"] for n in slice_names) < PREVIEW_COLLISION_MM:
            end_reason = "collision"
            break
        obs = policy.encode_obs(meas, prev_rot)
        rotate, hidden = policy.step(obs, hidden)
        rotate_cmd = float(rotate)

        # Apply kinematic noise to the *executed* motion only; prev_rot is
        # still the commanded value (matches the deploy chain).
        rot_motor = rotate_cmd * rot_gain
        if rotate_noise_deg > 0.0:
            rot_motor += float(rng.normal(0.0, rotate_noise_deg))
        rot_motor = float(np.clip(rot_motor, -max_rot_deg, max_rot_deg))

        drive_motor = drive_mm * drive_gain
        if drive_noise_mm > 0.0:
            drive_motor += float(rng.normal(0.0, drive_noise_mm))
        drive_motor = max(0.0, drive_motor)

        yaw = float(((yaw + rot_motor + 180.0) % 360.0) - 180.0)
        prev_rot = rotate_cmd
        rad = math.radians(yaw)
        x += drive_motor * math.cos(rad)
        y += drive_motor * math.sin(rad)
        xs.append(x); ys.append(y); yaws.append(yaw)
    return xs, ys, yaws, end_reason


def _preview_rollouts(start_pose, n=PREVIEW_N, n_steps=PREVIEW_STEPS,
                      sigma_xy=PREVIEW_NOISE_XY_MM,
                      sigma_yaw=PREVIEW_NOISE_YAW_DEG):
    if n <= 0 or start_pose is None:
        print("Preview skipped (PREVIEW_N=0 or no start pose).")
        return None
    rng = np.random.default_rng(0xC0FFEE)
    fig, ax = plt.subplots(figsize=(10, 8))
    if _arena_walls_x is not None:
        ax.scatter(_arena_walls_x, _arena_walls_y, color="green", s=2, alpha=0.3,
                   label="walls")
    end_counts = {"ok": 0, "collision": 0, "profile_fail": 0}
    rot_gains:   List[float] = []
    drive_gains: List[float] = []

    for i in range(n + 1):  # i=0 is the no-noise canonical rollout
        if i == 0:
            x0, y0, yaw0 = start_pose["x"], start_pose["y"], start_pose["yaw_deg"]
            rot_g, drive_g, rot_n, drive_n = 1.0, 1.0, 0.0, 0.0
            lw, alpha, color = 2.0, 0.95, "black"
            label = "no-noise canonical"
        else:
            x0   = start_pose["x"]   + rng.normal(0.0, sigma_xy)
            y0   = start_pose["y"]   + rng.normal(0.0, sigma_xy)
            yaw0 = start_pose["yaw_deg"] + rng.normal(0.0, sigma_yaw)
            if PREVIEW_KIN_NOISE:
                rot_g   = float(rng.uniform(1.0 - PREVIEW_ROT_GAIN_RANGE_PCT,
                                            1.0 + PREVIEW_ROT_GAIN_RANGE_PCT))
                drive_g = float(rng.uniform(1.0 - PREVIEW_DRIVE_GAIN_RANGE_PCT,
                                            1.0 + PREVIEW_DRIVE_GAIN_RANGE_PCT))
                rot_n   = PREVIEW_ROTATE_NOISE_DEG
                drive_n = PREVIEW_DRIVE_NOISE_MM
                rot_gains.append(rot_g); drive_gains.append(drive_g)
            else:
                rot_g, drive_g, rot_n, drive_n = 1.0, 1.0, 0.0, 0.0
            lw, alpha, color = 1.0, 0.55, None
            label = (
                f"+ training noise (rot×{PREVIEW_ROT_GAIN_RANGE_PCT:.0%}, "
                f"drive×{PREVIEW_DRIVE_GAIN_RANGE_PCT:.0%})"
                if (PREVIEW_KIN_NOISE and i == 1) else None
            )
        xs, ys, _, end = _simulate_rollout(
            x0, y0, yaw0, n_steps,
            rot_gain=rot_g, drive_gain=drive_g,
            rotate_noise_deg=rot_n, drive_noise_mm=drive_n,
            rng=rng,
        )
        end_counts[end] = end_counts.get(end, 0) + 1
        ax.plot(xs, ys, lw=lw, alpha=alpha, color=color, label=label)
        ax.scatter([xs[0]], [ys[0]], color="red", s=18, zorder=3)
        marker = {"ok": "s", "collision": "x", "profile_fail": "?"}.get(end, "o")
        ax.scatter([xs[-1]], [ys[-1]], marker=marker,
                   color="black" if end == "ok" else "red",
                   s=40 if end != "ok" else 22, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    title_lines = [
        f"Pre-flight: 1 canonical + {n} noisy rollouts from current pose",
        f"σxy={sigma_xy:.0f}mm  σyaw={sigma_yaw:.0f}°"
        + (f"  rot_gain U(1±{PREVIEW_ROT_GAIN_RANGE_PCT:.0%}) "
           f" drive_gain U(1±{PREVIEW_DRIVE_GAIN_RANGE_PCT:.0%}) "
           f" rot_step σ={PREVIEW_ROTATE_NOISE_DEG:.0f}°  "
           f"drive_step σ={PREVIEW_DRIVE_NOISE_MM:.0f}mm"
           if PREVIEW_KIN_NOISE else "  (kinematic noise OFF)"),
        f"end → ok:{end_counts['ok']}  collision:{end_counts['collision']}  "
        f"profile_fail:{end_counts['profile_fail']}",
    ]
    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = f"{DATA_FOLDER}/{SESSION}/_preview_rollouts.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Pre-flight preview: {out_path}")
    print(f"  ends: {end_counts}")
    if PREVIEW_KIN_NOISE and rot_gains:
        print(f"  noisy-rollout gains: rot mean={np.mean(rot_gains):+.3f} "
              f"(min {min(rot_gains):+.3f}, max {max(rot_gains):+.3f})  "
              f"drive mean={np.mean(drive_gains):+.3f} "
              f"(min {min(drive_gains):+.3f}, max {max(drive_gains):+.3f})")
    return end_counts


print("\n=== Pre-flight: reading physical start pose for sim preview ===")
init_pose = _wait_yaw_stable(ROBOT_ID, tag="#init")
if init_pose is None:
    print("⚠️  could not read pose from tracker — preview skipped")
else:
    print(f"  start pose: x={init_pose['x']:.0f}  y={init_pose['y']:.0f}  "
          f"yaw={init_pose['yaw_deg']:+.1f}°")
    _preview_rollouts(init_pose)
    if PREVIEW_REQUIRE_OK:
        ans = input("Inspect _preview_rollouts.png. Proceed with real run? [y/N]: ")
        if ans.strip().lower() != "y":
            print("Aborted by user after preview.")
            raise SystemExit(0)


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

PushOver.send(f"Policy run started: {SESSION}")

for step in range(MAX_STEPS):
    if control.wait_if_paused():
        # User paused → robot bumped → log crash using last known position
        _log_crash(step, last_position)

    # ── Sonar ping at current orientation ─────────────────────────────────────
    sonar_package = client.read_and_process(do_ping=True, plot=True)
    # Wait for the tracker to converge on the post-step pose before reading.
    # After a sharp turn the tracker can lag the real yaw by ~1–2 s; reading
    # immediately would give a stale yaw and feed the simulator/policy a
    # geometric profile for the wrong heading.
    position      = _wait_yaw_stable(ROBOT_ID, tag=f"#step{step}")

    if sonar_package is None:
        # Skip the step entirely — do not advance hidden state or prev_rot.
        # The policy will pick up cleanly on the next successful ping.
        print(f"Step {step:4d}: no sonar data — skipping step")
        writer.save_data(
            sonar_package=None,
            position=position,
            motion={"rotate1": 0.0, "rotate2": 0.0,
                    "net_rotation": 0.0, "drive_mm": 0.0},
        )
        _write_metrics_row(step, position, None, None, None, 0.0, None, None)
        continue

    sonar_package["robot_number"] = ROBOT_ID

    # ── Live sonar prediction: envelope → 3-slice (distance, σ) ───────────────
    L, R = extract_lr_envelope(sonar_package)
    # Per-step envelope diagnostic — sanity-check input distribution across runs.
    print(f"  envL min={L.min():.0f} max={L.max():.0f}  | envR min={R.min():.0f} max={R.max():.0f}")
    # TEMP envelope dump — overlaid on training mean by SCRIPT_PlotEnvelopeOverlay.py
    if step == 0:
        np.savez(f"{DATA_FOLDER}/{SESSION}/_envelope_step0.npz", L=L, R=R)
    meas_live = sonar_model.predict_from_envelope(L, R)

    # ── Sim-to-real diagnostic: same prediction at the tracker's ground-truth
    #    pose, via the geometric simulator. Compare meas_live vs meas_sim
    #    offline to assess emulator quality on the real arena.
    #    meas_sim_clean = noiseless geometric truth (per-slice min of profile);
    #    meas_sim       = same with σ_sim Gaussian noise added (training match).
    meas_sim = None
    meas_sim_clean = None
    _pos = position or {}
    if None not in (_pos.get("x"), _pos.get("y"), _pos.get("yaw_deg")):
        meas_sim = sim.get_sonar_measurement(_pos["x"], _pos["y"], _pos["yaw_deg"])
        profile = sim.get_profile_at_position(_pos["x"], _pos["y"], _pos["yaw_deg"])
        meas_sim_clean = {
            f"distance_{name}_mm":
                float(profile[sim.sonar_model.slice_masks[i]].min())
            for i, name in enumerate(sim.sonar_model.SLICE_NAMES)
        }

    # ── Policy step ───────────────────────────────────────────────────────────
    if POLICY_INPUT_SOURCE == "sim" and meas_sim is not None:
        meas_for_policy = meas_sim
    elif POLICY_INPUT_SOURCE == "sim_clean" and meas_sim_clean is not None:
        # sim_clean has only distances. Borrow σs from meas_sim so use_sigma
        # policies still encode without KeyError; falls back silently otherwise.
        meas_for_policy = dict(meas_sim_clean)
        if meas_sim is not None:
            for k in ("sigma_right_mm", "sigma_center_mm", "sigma_left_mm"):
                meas_for_policy.setdefault(k, meas_sim[k])
    else:
        meas_for_policy = meas_live   # fallback when sim source requested but tracker missed
    obs       = policy.encode_obs(meas_for_policy, prev_rot)
    rotate, hidden_new = policy.step(obs, hidden)

    if do_rotation:
        client.step(angle=rotate)
        time.sleep(0.5)

    # ── Drive forward ─────────────────────────────────────────────────────────
    if do_translation:
        try:
            client.step(distance=policy.fixed_drive_mm / 1000.0)
            time.sleep(0.15)
        except RuntimeError as e:
            # Drive blocked → robot likely against wall. Log and resume after pause.
            print(f"  *** Drive aborted (step {step}): {e} ***")
            _log_crash(step, position)
            control.wait_if_paused()
            continue

    # ── Carry RNN state forward (only after a successful rotate+drive) ────────
    hidden   = hidden_new
    prev_rot = rotate

    # ── Console line ──────────────────────────────────────────────────────────
    rob_x       = position["x"]       if position else None
    rob_y       = position["y"]       if position else None
    rob_yaw_deg = position["yaw_deg"] if position else None
    pos_str = (f"({rob_x:.0f}, {rob_y:.0f}, {rob_yaw_deg:+.0f}°)"
               if None not in (rob_x, rob_y, rob_yaw_deg) else "N/A")
    sim_str = ""
    if meas_sim_clean is not None:
        sim_str = (f"  sim_geom=[{meas_sim_clean['distance_right_mm']:.0f},"
                   f"{meas_sim_clean['distance_center_mm']:.0f},"
                   f"{meas_sim_clean['distance_left_mm']:.0f}]")
    if meas_sim is not None:
        sim_str += (f"  sim_d+ε=[{meas_sim['distance_right_mm']:.0f},"
                    f"{meas_sim['distance_center_mm']:.0f},"
                    f"{meas_sim['distance_left_mm']:.0f}]")
    print(
        f"Step {step:4d}: "
        f"d=[{meas_live['distance_right_mm']:.0f},"
        f"{meas_live['distance_center_mm']:.0f},"
        f"{meas_live['distance_left_mm']:.0f}] mm  "
        f"σ=[{meas_live['sigma_right_mm']:.0f},"
        f"{meas_live['sigma_center_mm']:.0f},"
        f"{meas_live['sigma_left_mm']:.0f}]  "
        f"rot={rotate:+6.1f}°  pos={pos_str}{sim_str}"
    )

    writer.save_data(
        sonar_package=sonar_package,
        position=position,
        motion={
            "rotate1":      0.0,
            "rotate2":      rotate,
            "net_rotation": rotate,
            "drive_mm":     policy.fixed_drive_mm,
        },
        sonar_prediction=meas_live,
        sim_prediction=meas_sim,
        sim_prediction_clean=meas_sim_clean,
    )
    _write_metrics_row(step, position, meas_live, meas_sim_clean, meas_sim,
                       rotate, L, R)
    last_position = position

    # ── Live trajectory plot ──────────────────────────────────────────────────
    _traj_x.append(rob_x if rob_x is not None else np.nan)
    _traj_y.append(rob_y if rob_y is not None else np.nan)
    _traj_yaw.append(rob_yaw_deg if rob_yaw_deg is not None else np.nan)
    if PLOT_EVERY > 0 and step % PLOT_EVERY == 0:
        _save_trajectory_plot()

    if step % 100 == 0 and step > 0:
        PushOver.send(f"{SESSION}: {step}/{MAX_STEPS} steps")

    if wait_for_confirmation:
        response = Dialog.ask_yes_no("Continue", min_size=(400, 200))
        if response[0] == "No":
            break
    else:
        time.sleep(0.25)

_save_trajectory_plot()
PushOver.send(f"Policy run completed: {SESSION}, {step + 1} steps.")
