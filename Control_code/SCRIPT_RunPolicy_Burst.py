#!/usr/bin/env python3
"""
SCRIPT_RunPolicy_Burst.py

Deploy a burst-trained policy (SCRIPT_TrainPolicy_Burst) on the real robot.

Per step, the robot takes N_LOOKS sonar measurements at different head
directions from slightly different positions along the step-start heading
(heading_1), then executes a net body rotation r2 and drives inter_burst_drive_mm.

Physical execution (option-1 faithful-to-training sequence):
  cur_delta = 0  (body yaw relative to heading_1)
  for k = 0 .. N_LOOKS-1:
      rotate by (look_physical[k] - cur_delta)   # point sonar at heading_1+look[k]
      ping
      cur_delta = look_physical[k]
      if k < N_LOOKS-1 and intra_burst_drive > 0:
          rotate by (-cur_delta)                 # back to heading_1
          drive intra_burst_drive / (N_LOOKS-1)  # along heading_1
          cur_delta = 0
  rotate by (r2_physical - cur_delta)            # absorb final rotate-back into r2
  drive inter_burst_drive_mm                     # along heading_1 + r2
  new heading_1 = old heading_1 + r2_physical

IID symmetry uses two flips (matching training):
  flip_look  = sign of last step's final IID — canonicalises the N_LOOKS look angles
  flip_drive = sign of this step's final (last-look) IID — canonicalises r2
"""

import collections
import json
import os
import time

import numpy as np

from Library import Client
from Library import CodeLogger
from Library import DataStorage
from Library import Dialog
from Library import LorexTracker
from Library import PauseControl
from Library import PushOver
from matplotlib import pyplot as plt
from LorexLib.Environment import capture_environment_layout
from SCRIPT_TrainPolicy_Burst import Config, MLPPolicy, build_input, N_LOOKS


# ══════════════════════════════════════════════════════════════════════════════
# Settings — edit these
# ══════════════════════════════════════════════════════════════════════════════
POLICY    = 'new_burst_h01'     # sub-folder under PolicyTraining/
ARENA     = 'arena1'
REPEAT    = '01'
MAX_STEPS = 500

ROBOT_ID     = 1
SHORT_POLICY = POLICY.replace('policy', '')
POLICY_FILE  = "best_policy.json"
SESSION      = f"session_{SHORT_POLICY}_{ARENA}_{REPEAT}"

# Dry-run flags (set False to disable movement for debugging)
do_rotation    = True
do_translation = True

PLOT_EVERY = 5          # save trajectory plot every N steps (0 = disable)

wait_for_confirmation = False

POLICY_DIR   = "PolicyTraining"
DATA_FOLDER  = "PolicyRuns"


# ══════════════════════════════════════════════════════════════════════════════
# Load policy
# ══════════════════════════════════════════════════════════════════════════════

def load_policy(path: str):
    with open(path) as f:
        data = json.load(f)

    # Guard: module-level N_LOOKS in SCRIPT_TrainPolicy_Burst must match the
    # policy's training-time value. The MLP is sized off N_LOOKS, so a mismatch
    # means the saved genome doesn't fit the network.
    n_looks_saved = int(data.get("n_looks", N_LOOKS))
    if n_looks_saved != N_LOOKS:
        raise ValueError(
            f"N_LOOKS mismatch: policy was trained with n_looks={n_looks_saved} "
            f"but SCRIPT_TrainPolicy_Burst.N_LOOKS={N_LOOKS}. "
            f"Edit the module constant to match before running."
        )

    cfg = Config(
        history_len           = data["history_len"],
        hidden_sizes          = tuple(data["hidden_sizes"]),
        max_rotate1_deg       = data["max_rotate1_deg"],
        max_rotate2_deg       = data["max_rotate2_deg"],
        max_net_rotation_deg  = data["max_net_rotation_deg"],
        max_burst_spread_deg  = data["max_burst_spread_deg"],
        intra_burst_drive_mm  = data["intra_burst_drive_mm"],
        inter_burst_drive_mm  = data["inter_burst_drive_mm"],
        max_dist_mm           = data["max_dist_mm"],
        max_iid_db            = data["max_iid_db"],
    )
    policy = MLPPolicy(cfg)
    policy.set_genome(np.array(data["genome"], dtype=np.float32))
    return policy, cfg


# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════

policy_path = f"{POLICY_DIR}/{POLICY}/{POLICY_FILE}"
policy, cfg = load_policy(policy_path)
print(f"Loaded policy: {policy_path}")
print(f"  history_len={cfg.history_len}  hidden={cfg.hidden_sizes}  N_LOOKS={N_LOOKS}")
print(f"  max_rotate1={cfg.max_rotate1_deg}°  max_rotate2={cfg.max_rotate2_deg}°")
print(f"  max_net_rotation={cfg.max_net_rotation_deg}°  burst_spread={cfg.max_burst_spread_deg}°")
print(f"  intra_burst_drive={cfg.intra_burst_drive_mm} mm  inter_burst_drive={cfg.inter_burst_drive_mm} mm")

from Library import Settings as _settings
_settings.data_folder = DATA_FOLDER

session_folder = os.path.join(DATA_FOLDER, SESSION)
if os.path.exists(session_folder) and os.listdir(session_folder):
    response = input(f"Session folder '{session_folder}' already exists and is non-empty. Overwrite? [y/N]: ")
    if response.strip().lower() != "y":
        print("Aborted.")
        raise SystemExit(0)

control = PauseControl.PauseControl()
client  = Client.Client(robot_number=ROBOT_ID)
tracker = LorexTracker.LorexTracker()
writer  = DataStorage.DataWriter(SESSION, autoclear=True, verbose=False)
writer.add_file("SCRIPT_RunPolicy_Burst.py")
snapshot = capture_environment_layout(save_root=f"{DATA_FOLDER}/{SESSION}")
CodeLogger.log_code(f"{DATA_FOLDER}/{SESSION}", [".", "Library"], label=SESSION)

# Load arena walls from the env snapshot for live trajectory plotting
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

# Accumulate robot positions for the live plot (one per step — end-of-step)
_traj_x:   list = []
_traj_y:   list = []
_traj_yaw: list = []
_traj_plot_path = f"{DATA_FOLDER}/{SESSION}/trajectory.png"


def _interp_gaps(arr):
    """Linear interpolation through NaN gaps; flat-extrapolation at edges."""
    out = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(out)
    if valid.all() or not valid.any():
        return out
    idx = np.arange(len(out))
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def _interp_yaw_gaps(yaws_deg):
    """Unwrap valid yaw samples, interpolate through NaN gaps, re-wrap to [-180, 180]."""
    arr = np.asarray(yaws_deg, dtype=np.float64)
    valid = np.isfinite(arr)
    if not valid.any():
        return arr
    out = arr.copy()
    out[valid] = np.degrees(np.unwrap(np.radians(arr[valid])))
    out = _interp_gaps(out)
    return ((out + 180.0) % 360.0) - 180.0


def _save_trajectory_plot() -> None:
    """Render the current trajectory (and optional arena walls) and overwrite trajectory.png."""
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
        ax.scatter(_arena_walls_x, _arena_walls_y, color="green", s=2, alpha=0.3, label="Walls")
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


# Warm up sonar
for _ in range(5):
    client.acquire("ping")
    time.sleep(0.5)

# History buffer — zero-initialised, consistent with training.
# Each entry: (d1, i1, l1, d2, i2, l2, ..., dN, iN, lN, r2) — canonical values.
_history_entry_size = 3 * N_LOOKS + 1
history = collections.deque(
    [(0.0,) * _history_entry_size] * cfg.history_len, maxlen=cfg.history_len
)
last_physical_iid = 0.0   # sign used for flip_look on the first burst of next step

crash_log_path = f"{DATA_FOLDER}/{SESSION}/crashes.tsv"
last_position  = None


def _log_crash(step_idx: int, pos: dict) -> None:
    x, y, yaw = pos.get("x"), pos.get("y"), pos.get("yaw_deg")
    write_header = not os.path.exists(crash_log_path)
    with open(crash_log_path, "a") as f:
        if write_header:
            f.write("step\tx\ty\tyaw_deg\n")
        f.write(f"{step_idx}\t{x}\t{y}\t{yaw}\n")
    print(f"  *** Crash logged (step {step_idx}): x={x}, y={y}, yaw={yaw} ***")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

PushOver.send(f"Policy run started: {SESSION}")

step = 0
for step in range(MAX_STEPS):
    if control.wait_if_paused():
        _log_crash(step, last_position or {})

    # ── Plan phase: one forward pass with history only → all N_LOOKS look angles ──
    flip_look = last_physical_iid < 0.0
    inp = build_input(history, [], cfg)
    raw = policy.forward(inp)   # (N_LOOKS+1,) in [-1, 1]

    center_canonical = float(raw[0]) * cfg.max_rotate1_deg
    look_canonicals  = [center_canonical] + [
        center_canonical + float(raw[k]) * (cfg.max_burst_spread_deg / 2.0)
        for k in range(1, N_LOOKS)
    ]
    look_physicals = [-lc if flip_look else lc for lc in look_canonicals]

    # ── Burst phase: rotate → ping → (rotate-back + intra-drive) per look ──
    cur_delta      = 0.0   # body yaw relative to heading_1 (step-start heading)
    burst_samples  = []    # one dict per look
    last_meas_iid  = last_physical_iid   # fallback if all pings fail
    skip_step      = False
    step_drive     = cfg.intra_burst_drive_mm / (N_LOOKS - 1) if N_LOOKS > 1 else 0.0

    for k in range(N_LOOKS):
        # Rotate to heading_1 + look_physicals[k]
        rotation = look_physicals[k] - cur_delta
        if do_rotation and abs(rotation) > 1e-6:
            client.step(angle=rotation)
            time.sleep(0.3)
        cur_delta = look_physicals[k]

        # Ping — extra sleep afterwards matches RunPolicy.py's post-rotation
        # sleep(0.5) pattern and gives the ESP module time to flush the large
        # sonar payload before the next step command arrives.
        sonar_package = client.read_and_process(do_ping=True, plot=(k == N_LOOKS - 1))
        time.sleep(0.4)
        position      = tracker.get_position(ROBOT_ID)

        if sonar_package is None:
            print(f"Step {step:4d}: no sonar data at look {k} — skipping step")
            skip_step = True
            break

        sonar_package["robot_number"] = ROBOT_ID
        physical_iid = float(sonar_package["corrected_iid"])
        dist_mm      = min(float(sonar_package["corrected_distance"]) * 1000.0,
                           cfg.max_dist_mm)
        last_meas_iid = physical_iid

        burst_samples.append({
            "sonar":             sonar_package,
            "position":          position,
            "look_physical":     look_physicals[k],
            "look_canonical":    look_canonicals[k],
            "dist_mm":           dist_mm,
            "physical_iid":      physical_iid,
        })

        # Intra-burst transit: rotate back to heading_1 and drive forward.
        # Combined into one step command (turn-then-drive) to halve round-trips.
        if k < N_LOOKS - 1 and step_drive > 0.0:
            angle_back = -cur_delta if (do_rotation and abs(cur_delta) > 1e-6) else 0
            dist_fwd   = step_drive / 1000.0 if do_translation else 0
            if angle_back != 0 or dist_fwd != 0:
                try:
                    client.step(angle=angle_back, distance=dist_fwd)
                    time.sleep(0.15)
                except (RuntimeError, TimeoutError) as e:
                    print(f"  *** Intra-burst transit aborted (step {step}, look {k}): {e} ***")
                    _log_crash(step, position or {})
                    control.wait_if_paused()
                    skip_step = True
                    break
            cur_delta = 0.0

    if skip_step:
        # Best-effort recovery: return to heading_1 so the next step has a
        # clean reference.  Skip the history update (step did not complete).
        if do_rotation and abs(cur_delta) > 1e-6:
            try:
                client.step(angle=-cur_delta)
                time.sleep(0.3)
            except (RuntimeError, TimeoutError):
                pass
        last_physical_iid = 0.0   # reset: no reliable measurement this step
        writer.save_data(
            sonar_packages=None,
            position=tracker.get_position(ROBOT_ID),
            motion={"rotate2": 0.0, "net_rotation": 0.0,
                    "intra_burst_drive_mm": cfg.intra_burst_drive_mm,
                    "inter_burst_drive_mm": 0.0},
            burst_samples=[],
        )
        continue

    # ── Drive-decision phase: forward pass with all N_LOOKS measurements → r2 ──
    current_measurements = [
        (s["dist_mm"], abs(s["physical_iid"]), s["look_canonical"])
        for s in burst_samples
    ]
    flip_drive   = last_meas_iid < 0.0
    inp          = build_input(history, current_measurements, cfg)
    raw          = policy.forward(inp)
    r2_canonical = float(raw[N_LOOKS]) * cfg.max_rotate2_deg
    r2_physical  = -r2_canonical if flip_drive else r2_canonical
    # Clamp to ±max_net_rotation_deg and recompute canonical for history
    r2_physical  = float(np.clip(r2_physical,
                                 -cfg.max_net_rotation_deg, cfg.max_net_rotation_deg))
    r2_canonical = -r2_physical if flip_drive else r2_physical

    # Absorb the post-last-look rotate-back into r2 and combine with the
    # inter-burst drive — one round-trip instead of two.
    rotation  = r2_physical - cur_delta
    angle_r2  = rotation if (do_rotation and abs(rotation) > 1e-6) else 0
    dist_fwd  = cfg.inter_burst_drive_mm / 1000.0 if do_translation else 0
    if angle_r2 != 0 or dist_fwd != 0:
        try:
            client.step(angle=angle_r2, distance=dist_fwd)
            time.sleep(0.15)
        except (RuntimeError, TimeoutError) as e:
            print(f"  *** Inter-burst drive aborted (step {step}): {e} ***")
            _log_crash(step, tracker.get_position(ROBOT_ID) or {})
            control.wait_if_paused()
            continue
    cur_delta = r2_physical

    # ── Update history (canonical frame) ─────────────────────────────────
    hist_entry = tuple(
        v
        for s in burst_samples
        for v in (s["dist_mm"], abs(s["physical_iid"]), s["look_canonical"])
    ) + (r2_canonical,)
    history.append(hist_entry)
    last_physical_iid = last_meas_iid   # IID from final look of this step

    # ── Log ───────────────────────────────────────────────────────────────
    position     = tracker.get_position(ROBOT_ID)
    rob_x        = position["x"]
    rob_y        = position["y"]
    rob_yaw_deg  = position["yaw_deg"]
    pos_str = f"({rob_x:.3f}, {rob_y:.3f}, {rob_yaw_deg:.1f}°)" \
              if None not in (rob_x, rob_y, rob_yaw_deg) else "N/A"

    looks_str = "  ".join(
        f"L{k}: IID={s['physical_iid']:+6.2f}dB dist={s['dist_mm']:6.0f}mm look={s['look_physical']:+6.1f}°"
        for k, s in enumerate(burst_samples)
    )
    print(
        f"Step {step:4d}:  {looks_str}  r2={r2_physical:+6.1f}°  pos={pos_str}"
    )

    writer.save_data(
        sonar_packages=[s["sonar"] for s in burst_samples],
        position=position,
        motion={
            "rotate2":              r2_physical,
            "rotate2_canonical":    r2_canonical,
            "net_rotation":         r2_physical,
            "intra_burst_drive_mm": cfg.intra_burst_drive_mm,
            "inter_burst_drive_mm": cfg.inter_burst_drive_mm,
            "look_physicals":       look_physicals,
            "look_canonicals":      look_canonicals,
        },
        burst_samples=[
            {k: v for k, v in s.items() if k != "sonar"}   # sonar already in sonar_packages
            for s in burst_samples
        ],
    )
    last_position = position

    # ── Live trajectory plot (end-of-step position only) ──────────────────
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
