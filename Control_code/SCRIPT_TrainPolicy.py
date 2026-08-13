#!/usr/bin/env python3
"""
SCRIPT_TrainPolicy_RNN_Supervised.py — supervised RNN training for path following.

Same vanilla-RNN architecture as SCRIPT_TrainPolicy_RNN.py, but trained by
behavioural cloning of a privileged path-following teacher rather than by GA.

Teacher: pure pursuit. Given true (x, y) and the path, picks a target point a
fixed lookahead distance further along the path (in arc-length order) and
returns the rotation that points the robot at it.  The lookahead is the only
knob — small ≈ aggressive cross-track correction, large ≈ smooth tangent
following. Direction along the loop is fixed by arc-length order, so the
target field is a true 2D vector field with no yaw dependence.

Student: sees (dist, iid, prev_rot). Trained to regress the teacher's rotation
with MSE + BPTT through the RNN. Single-start pool around path[0] with Gaussian
noise (matches GA training), so the RNN can localise itself implicitly along
the loop by integrating sonar history from a known starting region.

Saves in the same JSON schema as the GA script.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")

import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from Library.EnvironmentSimulator import EnvironmentSimulator
from Library import Settings as _settings
from Library.Policy import (Policy, encode_obs, make_obs_layout,
                            make_policy_dict)
from Library.TargetPath import TargetPath, load_target_path

_settings.data_folder = "TargetArenas"


# ── Condition ────────────────────────────────────────────────────────────────
TARGET_ARENA = "Path02"
CONDITION    = "default"
BLIND        = False        # blind ablation: drop sonar, only prev_rot fed to
                            # the policy (in_dim=1). Output folder gets a
                            # `_blind` suffix. See Config.blind for details.


@dataclass
class Config:
    # Network
    hidden_size: int      = 32
    max_rotate_deg: float = 90.0
    # 150 mm matches Experiment 1's cruising step, so the argument made there
    # for the step size covers both experiments and need not be repeated. Note
    # Experiment 1 halves to 75 mm during the terminal approach, so the two
    # agree while cruising rather than throughout.
    fixed_drive_mm: float = 150.0

    # Input normalisation. Distances are clamped to [min_dist, max_dist] then
    # divided by max_dist; σ values are clamped to [0, max_sigma] then divided
    # by max_sigma. The clamps prevent occasional negative-from-noise values
    # or very large σs from dominating the input.
    # 1000, not 2000: the inverse abstains beyond 1 m, so a 2 m clamp let the
    # policy train on distances the sensor cannot produce. rationale.md warns
    # that train and deploy clamps drifting apart puts the policy out of
    # distribution on the real robot; this is that hazard.
    max_dist_mm:  float = 1000.0
    min_dist_mm: float = 300.0
    max_sigma_mm: float = 500.0    # σ_sim caps out around ~400-500 mm at the model's far edge

    # Sensor noise: σ_sim noise is now produced by the simulator (per-slice,
    # geometry-conditioned). No manual noise injection here.

    # Drop σ channels from the policy obs. With use_sigma=False the input is
    # 4-D (3 distances + prev_rot); with True it's 7-D (also 3 σs). σ at sim
    # is SonarModel.sigma_sim(d_true) — a deterministic interp through the
    # per-bin empirical σs from val data — and is mostly redundant with the
    # d slot (since both are functions of d_true). Set False to test whether
    # the policy actually uses σ.
    use_sigma: bool = False

    # Pole channels: the 3 class posteriors plus pole azimuth and range. On by
    # default now that the controller receives the full Experiment 1 local
    # feature. Kept switchable because the pole may carry little on a given
    # path -- on the first Path01 route the sensor saw roughly one true pole
    # per lap against five phantoms -- and the ablation is how that gets
    # established rather than assumed.
    use_poles: bool = True
    # Class-agnostic nearest-reflector range channel (+ its sigma when
    # use_sigma). The pole range channel saturates at ~750 mm by design -- it
    # is masked to 1 m as the terminal-stop signal -- so without this the
    # policy has no usable distance beyond a metre, which is exactly the
    # regime a path-following robot spends its time in. Off by default so
    # existing policies keep their recorded width; turn on for Path04.
    use_agn: bool = True

    # Blind ablation: when True the policy sees ONLY prev_rot (in_dim=1) —
    # all sonar channels are stripped. Used as a control: with motor noise
    # injection a blind policy cannot use sonar feedback to compensate, so
    # the expected outcome is poor downstream performance. That is the
    # evidence we want to publish: success of the sighted policy is sonar-
    # driven, not pure dead-reckoning. `blind` overrides `use_sigma`. Output
    # folder gets a `_blind` suffix automatically so the artifact does not
    # collide with the sighted run. Default comes from the top-level BLIND
    # constant under the Condition section so it lives next to TARGET_ARENA
    # / CONDITION rather than buried in the dataclass.
    blind: bool = BLIND

    # Teacher (pure pursuit)
    # Scaled with the step. Lookahead is implicitly a RATIO to the step: at
    # 125 mm a 200 mm lookahead sat 1.6 steps ahead; at 150 mm it would sit
    # 1.33, and pure pursuit becomes oscillatory as the lookahead approaches
    # the step. 240 mm holds the previous 1.6 ratio, so the teacher's tuning
    # carries over rather than needing to be refound.
    teacher_lookahead_mm: float = 240.0
    path_resample_mm: float     = 25.0

    # Teacher perturbation (action noise during rollout — drives the robot off
    # the teacher's path so the dataset contains recovery examples; the LABEL
    # is always the clean teacher rotation, only the COMMANDED rotation is noisy)
    teacher_perturb_prob:      float = 0.30
    teacher_perturb_sigma_deg: float = 30.0

    # Motor execution noise (sim-to-real). Real motors don't perfectly execute
    # commanded rotations or drive distances — wheel slip, encoder error, IMU
    # drift. This noise is added to the COMMANDED action before the simulator
    # step, but prev_rot fed back to the policy stays at the commanded value
    # (the robot knows what it asked its motors to do, not what they did).
    #
    # Two layers of perturbation:
    #   1. Per-step zero-mean Gaussian (`motion_rotate_noise_deg`,
    #      `motion_drive_noise_mm`). Averages out within ~10 steps; trains the
    #      policy to be robust to high-frequency execution noise.
    #   2. Per-episode multiplicative bias (`motion_rot_gain_range_pct`,
    #      `motion_drive_gain_range_pct`). One scalar sampled at episode
    #      reset, applied for the whole rollout. Forces the policy to use
    #      sonar feedback to detect and compensate for *sustained* execution
    #      error — the kind that doesn't average out and that the real robot
    #      exhibits (~10 % rotation gain mismatch, ~5 % drive gain mismatch).
    motion_rotate_noise_deg:     float = 3.0
    motion_drive_noise_mm:       float = 5.0
    motion_rot_gain_range_pct:   float = 0.15   # rot_gain ~ U(1-x, 1+x); 0 disables
    motion_drive_gain_range_pct: float = 0.05   # drive_gain ~ U(1-x, 1+x); 0 disables
    # Verbose flag for the kinematic-bias sampling. When True, print the
    # per-episode (rot_gain, drive_gain) pair at sample time and a one-line
    # summary of the first step's commanded vs perturbed action. Useful for
    # confirming the mechanism is wired correctly; flip off for full training.
    motion_noise_verbose:        bool  = False
    # If set, every per-episode bias sample is appended as one row to this TSV
    # (columns: episode_no, tag, rot_gain, drive_gain). Independent of the
    # verbose flag so you can keep a permanent record without console spam.
    # Resolved relative to the run's output_dir if not absolute.
    motion_noise_log_path:       Optional[str] = "motion_noise_log.tsv"

    # Episode rollout
    max_steps: int        = 150
    n_train_episodes: int = 1000
    n_val_episodes:   int = 100

    # Start pool. When the path JSON carries a start_box and start_arrow (which
    # SCRIPT_DefinePath.py writes), positions are drawn UNIFORMLY FROM THAT BOX
    # and `single_start_noise_xy_mm` is not consulted at all -- it is only the
    # fallback for paths defined without a box. Yaw noise applies either way, on
    # top of the drawn arrow direction.
    single_start_noise_xy_mm:   float = 50.0    # fallback only; see above
    # 20 deg, not 10: the robot is placed in the release box by hand, and the
    # box is a few hundred mm across, so its heading cannot be set to better
    # than about this. Training on a tighter spread than deployment actually
    # delivers would leave the policy out of distribution on the real robot at
    # step 0 -- the same train/deploy divergence the distance clamps have to
    # avoid.
    single_start_noise_yaw_deg: float = 20.0
    single_start_pool_size:     int   = 200

    # Optimisation
    n_epochs:        int   = 2000
    batch_size:      int   = 32
    learning_rate:   float = 1e-3
    weight_decay:    float = 0.0
    grad_clip_norm:  float = 1.0
    seed:            int   = 42

    # Arena / IO
    target_arena: str = TARGET_ARENA
    output_dir:   str = ""
    plot_trajectories_every_n: int = 5

    # Parallel rollout (data generation only — training stays single-process)
    parallel_eval: bool          = True
    num_workers:   Optional[int] = None


N_TRAJECTORY_EPISODES = 6


# ══════════════════════════════════════════════════════════════════════════════
# Teacher
# ══════════════════════════════════════════════════════════════════════════════

def _project_with_segment(path: TargetPath, x: float, y: float
                          ) -> Tuple[float, int, float]:
    """Like TargetPath.project but also returns (segment_index, t∈[0,1])."""
    pts = path.points
    a = pts[:-1]; b = pts[1:]
    ab = b - a
    seg_len_sq = np.einsum("ij,ij->i", ab, ab)
    pos = np.array([x, y], dtype=np.float64)
    ap = pos - a
    t = np.einsum("ij,ij->i", ap, ab) / np.maximum(seg_len_sq, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    foot = a + t[:, None] * ab
    diffs = foot - pos
    d2 = np.einsum("ij,ij->i", diffs, diffs)
    i = int(np.argmin(d2))
    return float(np.sqrt(d2[i])), int(i), float(t[i])


def teacher_target_unit(
    path: TargetPath, x: float, y: float, lookahead_mm: float,
) -> Tuple[float, float]:
    """Pure-pursuit target direction (unit vector) at (x, y).

    Project (x, y) onto the path, advance `lookahead_mm` along the path in
    arc-length order, return the unit vector from (x, y) to that target point.
    Yaw-independent: direction along the loop is fixed by arc-length order, so
    a robot starting "the wrong way" will be commanded to U-turn at first.
    """
    _, seg_i, t = _project_with_segment(path, x, y)
    foot_arc = float(path.cum_arc[seg_i] +
                     t * (path.cum_arc[seg_i + 1] - path.cum_arc[seg_i]))
    target_arc = (foot_arc + lookahead_mm) % path.total_length

    j = int(np.searchsorted(path.cum_arc, target_arc, side="right")) - 1
    j = max(0, min(j, path.points.shape[0] - 2))
    seg_len = max(float(path.cum_arc[j + 1] - path.cum_arc[j]), 1e-9)
    seg_t = (target_arc - float(path.cum_arc[j])) / seg_len
    target = path.points[j] + seg_t * (path.points[j + 1] - path.points[j])

    dx = float(target[0] - x)
    dy = float(target[1] - y)
    L = float(np.hypot(dx, dy))
    if L < 1e-9:
        return 0.0, 0.0
    return dx / L, dy / L


def teacher_rotation_deg(
    path: TargetPath,
    x: float, y: float, yaw_deg: float,
    lookahead_mm: float,
    max_rotate_deg: float,
) -> float:
    """Pure-pursuit teacher: rotation in degrees that points the robot at the
    lookahead target on the path."""
    tx, ty = teacher_target_unit(path, x, y, lookahead_mm)
    if tx == 0.0 and ty == 0.0:
        return 0.0
    target_yaw = float(np.arctan2(ty, tx))
    yaw = float(np.deg2rad(yaw_deg))
    delta = (target_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.clip(np.rad2deg(delta), -max_rotate_deg, max_rotate_deg))


# ══════════════════════════════════════════════════════════════════════════════
# Starts and sensor measurement
# ══════════════════════════════════════════════════════════════════════════════

def _make_box_aligned_starts(path: TargetPath, cfg: Config,
                             rng: np.random.Generator
                             ) -> List[Tuple[float, float, float]]:
    """Pool of starts uniformly in path.start_box, oriented along path.start_arrow
    with Gaussian yaw noise."""
    x_min, y_min, x_max, y_max = path.start_box
    bx, by, tx, ty = path.start_arrow
    arrow_yaw = float(np.degrees(np.arctan2(ty - by, tx - bx)))
    n = max(1, int(cfg.single_start_pool_size))
    starts: List[Tuple[float, float, float]] = []
    for _ in range(n):
        x    = float(rng.uniform(x_min, x_max))
        y    = float(rng.uniform(y_min, y_max))
        dyaw = float(rng.normal(0.0, cfg.single_start_noise_yaw_deg))
        starts.append((x, y, arrow_yaw + dyaw))
    return starts


def _make_path_aligned_starts(path: TargetPath, cfg: Config,
                              rng: np.random.Generator
                              ) -> List[Tuple[float, float, float]]:
    """Fallback: Gaussian pool around path[0] aligned with the initial tangent.
    Used when the path JSON has no start_box/start_arrow defined."""
    p0  = path.points[0]
    p1  = path.points[1]
    tangent_yaw = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    n = max(1, int(cfg.single_start_pool_size))
    starts: List[Tuple[float, float, float]] = []
    for _ in range(n):
        dx   = float(rng.normal(0.0, cfg.single_start_noise_xy_mm))
        dy   = float(rng.normal(0.0, cfg.single_start_noise_xy_mm))
        dyaw = float(rng.normal(0.0, cfg.single_start_noise_yaw_deg))
        starts.append((float(p0[0]) + dx, float(p0[1]) + dy, tangent_yaw + dyaw))
    return starts


def make_starts(path: TargetPath, cfg: Config,
                rng: np.random.Generator
                ) -> List[Tuple[float, float, float]]:
    """Sample starting poses. Uses path.start_box + path.start_arrow if both are
    defined (matches the real-robot release-box experimental setup), otherwise
    falls back to a Gaussian pool around path[0]."""
    if path.start_box is not None and path.start_arrow is not None:
        return _make_box_aligned_starts(path, cfg, rng)
    return _make_path_aligned_starts(path, cfg, rng)


def _obs_from_cfg(meas: Optional[Dict[str, float]], prev_rot: float, cfg: Config) -> np.ndarray:
    """Bind cfg's clamp + scale parameters to Library.Policy.encode_obs.
    Used during dataset generation, before a policy artifact has been saved.
    `meas` is unused (and may be None) when cfg.blind is True."""
    return encode_obs(
        meas, prev_rot,
        min_dist_mm=cfg.min_dist_mm, max_dist_mm=cfg.max_dist_mm,
        max_sigma_mm=cfg.max_sigma_mm, max_rotate_deg=cfg.max_rotate_deg,
        use_sigma=cfg.use_sigma, blind=cfg.blind, use_poles=cfg.use_poles,
        use_agn=cfg.use_agn,
    )


def _policy_from_net(net: "RNNNet", cfg: Config) -> Policy:
    """Build an in-memory Policy from a live torch RNN, so student rollouts
    exercise the same numpy forward pass that deployment will use."""
    return Policy(make_policy_dict(
        genome=net.to_genome(),
        hidden_size=net.hidden_size,
        in_dim=int(net.in_dim),
        out_dim=int(net.OUT_DIM),
        use_sigma=cfg.use_sigma,
        blind=cfg.blind,
        use_poles=cfg.use_poles,
        use_agn=cfg.use_agn,
        max_rotate_deg=net.max_rotate_deg,
        fixed_drive_mm=cfg.fixed_drive_mm,
        min_dist_mm=cfg.min_dist_mm,
        max_dist_mm=cfg.max_dist_mm,
        max_sigma_mm=cfg.max_sigma_mm,
    ))


_motion_noise_episode_counter: int = 0
_motion_noise_log_handle: Optional[Any] = None


def _open_motion_noise_log(cfg: "Config") -> None:
    """Open the motion-noise TSV (creating directory if needed) and write the
    header. Idempotent: a no-op if already open or if no path is configured.
    Resolves relative paths against the active run's output_dir when set,
    otherwise the current working directory."""
    global _motion_noise_log_handle
    if _motion_noise_log_handle is not None:
        return
    path = cfg.motion_noise_log_path
    if not path:
        return
    if not os.path.isabs(path):
        run_dir = cfg.output_dir if cfg.output_dir else "."
        path = os.path.join(run_dir, path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _motion_noise_log_handle = open(path, "w", newline="")
    _motion_noise_log_handle.write("episode\ttag\trot_gain\tdrive_gain\n")
    _motion_noise_log_handle.flush()


def _close_motion_noise_log() -> None:
    """Close the motion-noise TSV if open."""
    global _motion_noise_log_handle
    if _motion_noise_log_handle is not None:
        try:
            _motion_noise_log_handle.flush()
            _motion_noise_log_handle.close()
        except Exception:
            pass
        _motion_noise_log_handle = None


def _sample_motion_biases(cfg: "Config", rng: np.random.Generator,
                          tag: str = "") -> Tuple[float, float]:
    """Sample per-episode kinematic gain biases. Returns (rot_gain, drive_gain).
    Each is drawn once at episode reset and applied multiplicatively to every
    commanded action in the rollout. Set the corresponding *_range_pct to 0 to
    disable (returns 1.0)."""
    rg = float(cfg.motion_rot_gain_range_pct)
    dg = float(cfg.motion_drive_gain_range_pct)
    rot_gain   = float(rng.uniform(1.0 - rg, 1.0 + rg)) if rg > 0.0 else 1.0
    drive_gain = float(rng.uniform(1.0 - dg, 1.0 + dg)) if dg > 0.0 else 1.0

    global _motion_noise_episode_counter
    _motion_noise_episode_counter += 1
    ep = _motion_noise_episode_counter

    if cfg.motion_noise_verbose:
        prefix = f"[motion]{(' ' + tag) if tag else ''} ep#{ep:>4d}"
        print(f"{prefix}  rot_gain={rot_gain:+.4f}  drive_gain={drive_gain:+.4f}")

    if cfg.motion_noise_log_path:
        _open_motion_noise_log(cfg)
        if _motion_noise_log_handle is not None:
            _motion_noise_log_handle.write(
                f"{ep}\t{tag}\t{rot_gain:.6f}\t{drive_gain:.6f}\n"
            )
            _motion_noise_log_handle.flush()

    return rot_gain, drive_gain


def _apply_motion_noise(rot_exec: float,
                        cfg: "Config",
                        rng: np.random.Generator,
                        rot_gain: float,
                        drive_gain: float,
                        verbose_first_step: bool = False) -> Tuple[float, float]:
    """Apply per-episode gain × per-step Gaussian to a commanded (rot, drive).
    `rot_exec` is the policy/teacher's commanded rotation (deg); the fixed drive
    distance comes from cfg. Returns (rot_motor_deg, drive_motor_mm) ready to
    feed `simulator.simulate_robot_movement`. If `verbose_first_step` is True
    (only meant to be passed once per rollout, on step 0), print one line that
    shows the commanded vs perturbed action so the wiring is auditable."""
    rot_motor = rot_exec * rot_gain
    if cfg.motion_rotate_noise_deg > 0.0:
        rot_motor += float(rng.normal(0.0, cfg.motion_rotate_noise_deg))
    rot_motor = float(np.clip(rot_motor, -cfg.max_rotate_deg, cfg.max_rotate_deg))

    drive_motor = cfg.fixed_drive_mm * drive_gain
    if cfg.motion_drive_noise_mm > 0.0:
        drive_motor += float(rng.normal(0.0, cfg.motion_drive_noise_mm))
    drive_motor = max(0.0, drive_motor)

    if verbose_first_step and cfg.motion_noise_verbose:
        print(f"[motion]   step0  rot: cmd={rot_exec:+7.2f}°  → motor={rot_motor:+7.2f}°    "
              f"drive: cmd={cfg.fixed_drive_mm:7.1f}mm → motor={drive_motor:7.1f}mm")
    return rot_motor, drive_motor


# ══════════════════════════════════════════════════════════════════════════════
# Episode rollout under the teacher
# ══════════════════════════════════════════════════════════════════════════════

def rollout_with_teacher(
    simulator: EnvironmentSimulator,
    path: TargetPath,
    start: Tuple[float, float, float],
    cfg: Config,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]], bool]:
    """Drive the simulator with the teacher's rotation; record (sonar_in, target).

    Returns
    -------
    X         : (T, 7)  inputs = [d_L, d_C, d_R, σ_L, σ_C, σ_R, prev_rot]  (normalised)
    Y         : (T, 1)  targets = teacher rotation in deg
    positions : list of (x, y) including start; one entry per simulator step + 1
    collided  : whether episode ended in a wall collision
    """
    # Reseed the simulator's σ_sim noise generator from a derived integer so
    # the rollout is fully deterministic given the rng's incoming state.
    simulator.reseed(int(rng.integers(2**31 - 1)))

    # Per-episode kinematic gain biases — held constant across this rollout so
    # the policy has to use sonar feedback to compensate.
    rot_gain, drive_gain = _sample_motion_biases(cfg, rng, tag="teacher")

    x, y, yaw = start
    Xs: List[np.ndarray]  = []
    Ys: List[List[float]] = []
    positions: List[Tuple[float, float]] = [(float(x), float(y))]
    collided = False
    prev_rot = 0.0
    _verbose_step0 = True

    for _ in range(cfg.max_steps):
        meas = None if cfg.blind else simulator.get_sonar_measurement(x, y, yaw)
        Xs.append(_obs_from_cfg(meas, prev_rot, cfg))
        rot_clean = teacher_rotation_deg(
            path, x, y, yaw,
            cfg.teacher_lookahead_mm,
            cfg.max_rotate_deg,
        )
        Ys.append([rot_clean])

        if cfg.teacher_perturb_prob > 0.0 and rng.random() < cfg.teacher_perturb_prob:
            rot_exec = float(np.clip(
                rot_clean + rng.normal(0.0, cfg.teacher_perturb_sigma_deg),
                -cfg.max_rotate_deg, cfg.max_rotate_deg,
            ))
        else:
            rot_exec = rot_clean

        rot_motor, drive_motor = _apply_motion_noise(
            rot_exec, cfg, rng, rot_gain, drive_gain,
            verbose_first_step=_verbose_step0,
        )
        _verbose_step0 = False

        action = {"rotate1_deg": 0.0, "rotate2_deg": rot_motor, "drive_mm": drive_motor}
        result = simulator.simulate_robot_movement(x, y, yaw, [action], compute_sonar=False)[0]
        x   = float(result["position"]["x"])
        y   = float(result["position"]["y"])
        yaw = float(result["orientation"])
        positions.append((x, y))
        prev_rot = rot_exec
        if bool(result["collision"]["drive_blocked"]):
            collided = True
            break

    return (np.asarray(Xs, dtype=np.float32),
            np.asarray(Ys, dtype=np.float32),
            positions, collided)


# ── Parallel rollout workers ───────────────────────────────────────────────────

_WORKER_SIM:  Optional[EnvironmentSimulator] = None
_WORKER_PATH: Optional[TargetPath]           = None
_WORKER_CFG:  Optional[Config]               = None


def _init_rollout_worker(cfg_dict: dict) -> None:
    global _WORKER_SIM, _WORKER_PATH, _WORKER_CFG
    # Belt-and-braces: BLAS env vars at module top pin numpy/MKL/etc.; this
    # pins torch in case the simulator uses it under the hood.
    try:
        import torch as _t
        _t.set_num_threads(1)
    except ImportError:
        pass
    cfg = Config(**{k: v for k, v in cfg_dict.items()
                    if k in {f.name for f in dataclasses.fields(Config)}})
    _WORKER_CFG = cfg
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        _WORKER_SIM  = EnvironmentSimulator(cfg.target_arena)
        _WORKER_PATH = load_target_path(
            cfg.target_arena, _settings.data_folder, cfg.path_resample_mm,
        )


def _rollout_worker(args: Tuple[Tuple[float, float, float], int]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    start, seed = args
    rng = np.random.default_rng(seed)
    X, Y, _, _ = rollout_with_teacher(_WORKER_SIM, _WORKER_PATH, start, _WORKER_CFG, rng)
    return X, Y


def generate_dataset(
    simulator: EnvironmentSimulator,
    path: TargetPath,
    starts: List[Tuple[float, float, float]],
    n_episodes: int,
    cfg: Config,
    rng: np.random.Generator,
    label: str = "",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    t0 = time.time()

    # Pre-pick starts and per-rollout seeds on the main thread so every worker
    # is fully deterministic given the master seed.
    work: List[Tuple[Tuple[float, float, float], int]] = []
    for _ in range(n_episodes):
        idx  = int(rng.integers(len(starts)))
        seed = int(rng.integers(2**31 - 1))
        work.append((starts[idx], seed))

    def _record_progress(done: int) -> None:
        if done % 100 != 0 and done != n_episodes:
            return
        elapsed = time.time() - t0
        med = int(np.median([x.shape[0] for x in Xs])) if Xs else 0
        print(f"  {label}rollout {done:5d}/{n_episodes}  "
              f"median_len={med:3d}  ({elapsed:.1f}s)", flush=True)

    if cfg.parallel_eval:
        n_workers = cfg.num_workers or os.cpu_count()
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_rollout_worker,
            initargs=(asdict(cfg),),
        ) as ex:
            futs = [ex.submit(_rollout_worker, w) for w in work]
            done = 0
            for fut in as_completed(futs):
                X, Y = fut.result()
                if X.shape[0] >= 2:
                    Xs.append(X); Ys.append(Y)
                done += 1
                _record_progress(done)
    else:
        for i, (start, seed) in enumerate(work):
            local_rng = np.random.default_rng(seed)
            X, Y, _, _ = rollout_with_teacher(simulator, path, start, cfg, local_rng)
            if X.shape[0] >= 2:
                Xs.append(X); Ys.append(Y)
            _record_progress(i + 1)

    return Xs, Ys


# ══════════════════════════════════════════════════════════════════════════════
# Padded batching
# ══════════════════════════════════════════════════════════════════════════════

def make_batches(
    Xs: List[np.ndarray], Ys: List[np.ndarray],
    batch_size: int, rng: np.random.Generator,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    idx = rng.permutation(len(Xs))
    in_dim = Xs[0].shape[1]
    batches = []
    for k in range(0, len(idx), batch_size):
        chunk = idx[k:k + batch_size]
        T_max = max(Xs[i].shape[0] for i in chunk)
        B = len(chunk)
        Xb = np.zeros((B, T_max, in_dim),         dtype=np.float32)
        Yb = np.zeros((B, T_max, RNNNet.OUT_DIM), dtype=np.float32)
        Mb = np.zeros((B, T_max),                 dtype=np.float32)
        for j, i in enumerate(chunk):
            T = Xs[i].shape[0]
            Xb[j, :T] = Xs[i]
            Yb[j, :T] = Ys[i]
            Mb[j, :T] = 1.0
        batches.append((torch.from_numpy(Xb), torch.from_numpy(Yb), torch.from_numpy(Mb)))
    return batches


# ══════════════════════════════════════════════════════════════════════════════
# Network — same forward pass as numpy RNNPolicy in SCRIPT_TrainPolicy_RNN.py
# (weights map 1:1, so save format is identical)
# ══════════════════════════════════════════════════════════════════════════════

class RNNNet(nn.Module):
    # Input layout (canonical):
    #   [d_left, d_center, d_right,
    #    σ_left, σ_center, σ_right,    ← only when cfg.use_sigma
    #    prev_rot]
    # Distances normalised by max_dist_mm, σs by max_sigma_mm,
    # prev_rot by max_rotate_deg. in_dim comes from make_obs_layout: 4/7
    # wall-only, 9/14 with the class and pole channels, 1 when blind.
    # (just prev_rot — see Library/Policy.encode_obs).
    OUT_DIM = 1

    def __init__(self, hidden_size: int, max_rotate_deg: float, in_dim: int):
        super().__init__()
        h = hidden_size
        self.hidden_size = h
        self.in_dim = in_dim
        self.max_rotate_deg = max_rotate_deg
        self.W_xh = nn.Parameter(torch.randn(h, in_dim) * 0.1)
        Q, _ = torch.linalg.qr(torch.randn(h, h))
        self.W_hh = nn.Parameter(Q * 0.9)
        self.b_h  = nn.Parameter(torch.zeros(h))
        self.W_hy = nn.Parameter(torch.randn(self.OUT_DIM, h) * 0.1)
        self.b_y  = nn.Parameter(torch.zeros(self.OUT_DIM))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, T, IN) → returns (B, T, OUT)
        B, T, _ = X.shape
        h = X.new_zeros(B, self.hidden_size)
        outs = []
        for t in range(T):
            h = torch.tanh(X[:, t] @ self.W_xh.T + h @ self.W_hh.T + self.b_h)
            y = torch.tanh(h @ self.W_hy.T + self.b_y) * self.max_rotate_deg
            outs.append(y)
        return torch.stack(outs, dim=1)

    def to_genome(self) -> np.ndarray:
        parts = [self.W_xh.detach().numpy(),
                 self.W_hh.detach().numpy(),
                 self.b_h.detach().numpy(),
                 self.W_hy.detach().numpy(),
                 self.b_y.detach().numpy()]
        return np.concatenate([p.ravel() for p in parts]).astype(np.float32)


def genome_size(hidden_size: int, in_dim: int) -> int:
    h = hidden_size
    return h * in_dim + h * h + h + RNNNet.OUT_DIM * h + RNNNet.OUT_DIM


def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq = (pred - tgt) ** 2 * mask.unsqueeze(-1)
    return sq.sum() / mask.sum().clamp_min(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# IO  (schema-compatible with SCRIPT_TrainPolicy_RNN.save_policy)
# ══════════════════════════════════════════════════════════════════════════════

def save_policy(net: RNNNet, cfg: Config, val_loss: float, epoch: int, path: str) -> None:
    """Write a deploy-ready policy artifact. The deploy-relevant fields come
    from Library.Policy.make_policy_dict — the same dict layout used to build
    in-memory Policy objects during training-time student rollouts. Training
    metadata (val_loss, epoch, ...) is added on top."""
    data = make_policy_dict(
        genome=net.to_genome(),
        hidden_size=net.hidden_size,
        in_dim=int(net.in_dim),
        out_dim=int(net.OUT_DIM),
        use_sigma=cfg.use_sigma,
        blind=cfg.blind,
        use_poles=cfg.use_poles,
        use_agn=cfg.use_agn,
        max_rotate_deg=net.max_rotate_deg,
        fixed_drive_mm=cfg.fixed_drive_mm,
        min_dist_mm=cfg.min_dist_mm,
        max_dist_mm=cfg.max_dist_mm,
        max_sigma_mm=cfg.max_sigma_mm,
    )
    data.update({
        "genome_size":   genome_size(net.hidden_size, net.in_dim),
        "fitness":       0.0,
        "generation":    int(epoch),
        "training_kind": "supervised_teacher",
        "val_loss_mse":  float(val_loss),
    })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_teacher_field(
    path: TargetPath,
    walls: np.ndarray,
    arena,                                    # ArenaLayout via simulator.arena
    cfg: Config,
    output_dir: str,
    grid_step_mm: float = 100.0,
) -> None:
    """Quiver plot of the teacher's target heading. Pure pursuit's target is a
    function of (x, y) only, so this is a true 2D vector field — one arrow per
    grid cell points where the teacher would steer the robot from there."""
    xs = np.arange(arena.arena_min_x, arena.arena_max_x + grid_step_mm, grid_step_mm)
    ys = np.arange(arena.arena_min_y, arena.arena_max_y + grid_step_mm, grid_step_mm)
    gx, gy = np.meshgrid(xs, ys)

    U = np.zeros_like(gx, dtype=np.float64)
    V = np.zeros_like(gy, dtype=np.float64)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            tx, ty = teacher_target_unit(
                path, float(gx[i, j]), float(gy[i, j]), cfg.teacher_lookahead_mm,
            )
            U[i, j] = tx
            V[i, j] = ty

    fig, ax = plt.subplots(figsize=(9, 9))
    if walls is not None and len(walls) > 0:
        ax.scatter(walls[:, 0], walls[:, 1], s=0.5, c="#aaaaaa", linewidths=0, zorder=1)
    pts = path.points
    ax.plot(pts[:, 0], pts[:, 1], color="#d62728", linewidth=2.0, alpha=0.7,
            zorder=2, label="target path")
    ax.quiver(gx, gy, U, V, color="#333", scale=35, width=0.0025,
              headwidth=4, headlength=5, zorder=3)
    ax.set_title(
        f"Teacher target heading  (pure pursuit, lookahead = "
        f"{cfg.teacher_lookahead_mm:.0f} mm)",
        fontsize=11,
    )
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal", "box")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "teacher_field.png"), dpi=120)
    plt.close(fig)


def plot_teacher_rollouts(
    simulator: EnvironmentSimulator,
    path: TargetPath,
    starts: List[Tuple[float, float, float]],
    walls: np.ndarray,
    cfg: Config,
    output_dir: str,
    rng: np.random.Generator,
    n_rollouts: int = 6,
) -> None:
    """Plot a handful of teacher-driven trajectories. If these don't trace the
    loop cleanly, the student has no chance — fix the teacher before training."""
    fig, ax = plt.subplots(figsize=(8, 8))
    if walls is not None and len(walls) > 0:
        ax.scatter(walls[:, 0], walls[:, 1], s=0.5, c="#aaaaaa", linewidths=0, zorder=1)
    pts = path.points
    ax.plot(pts[:, 0], pts[:, 1], color="#d62728", linewidth=2.0, alpha=0.7,
            zorder=1.5, label="target path")
    cmap = plt.cm.tab10
    for i in range(n_rollouts):
        s = starts[int(rng.integers(len(starts)))]
        _, _, positions, collided = rollout_with_teacher(simulator, path, s, cfg, rng)
        positions = np.array(positions)
        c = cmap(i % 10)
        ls = "--" if collided else "-"
        ax.plot(positions[:, 0], positions[:, 1], color=c, linewidth=1.0, linestyle=ls,
                zorder=2, label=f"T{i+1}{' coll' if collided else ''}")
        ax.plot(positions[0, 0],  positions[0, 1],  "o", color=c, markersize=5, zorder=3)
        ax.plot(positions[-1, 0], positions[-1, 1], "x", color=c, markersize=6, zorder=3)
    ax.set_title("Teacher-driven trajectories (sanity check before training)", fontsize=11)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "teacher_rollouts.png"), dpi=120)
    plt.close(fig)


def plot_training_curve(history: Dict[str, list], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = range(len(history["train_loss"]))
    ax.plot(ep, history["train_loss"], label="train")
    ax.plot(ep, history["val_loss"],   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on rotation_deg")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curve.png"), dpi=120)
    plt.close(fig)


def rollout_student(
    net: RNNNet,
    simulator: EnvironmentSimulator,
    start: Tuple[float, float, float],
    cfg: Config,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Tuple[float, float]], bool]:
    if rng is None:
        rng = np.random.default_rng()
    simulator.reseed(int(rng.integers(2**31 - 1)))

    # Wrap the live torch net as the same numpy Policy that deployment will use,
    # so train-time visualisation rollouts exercise the deploy code path.
    policy = _policy_from_net(net, cfg)

    # Per-episode kinematic gain biases (same source of randomness as the teacher
    # rollouts the student is trained against).
    rot_gain, drive_gain = _sample_motion_biases(cfg, rng, tag="student")

    x, y, yaw = start
    positions: List[Tuple[float, float]] = [(x, y)]
    hidden = policy.initial_hidden()
    collided = False
    prev_rot = 0.0
    _verbose_step0 = True
    for _ in range(cfg.max_steps):
        meas    = None if cfg.blind else simulator.get_sonar_measurement(x, y, yaw)
        obs     = policy.encode_obs(meas, prev_rot)
        rot, hidden = policy.step(obs, hidden)

        rot_motor, drive_motor = _apply_motion_noise(
            float(rot), cfg, rng, rot_gain, drive_gain,
            verbose_first_step=_verbose_step0,
        )
        _verbose_step0 = False

        action = {"rotate1_deg": 0.0, "rotate2_deg": rot_motor, "drive_mm": drive_motor}
        r = simulator.simulate_robot_movement(x, y, yaw, [action], compute_sonar=False)[0]
        x   = float(r["position"]["x"])
        y   = float(r["position"]["y"])
        yaw = float(r["orientation"])
        positions.append((x, y))
        prev_rot = rot
        if bool(r["collision"]["drive_blocked"]):
            collided = True
            break
    return positions, collided


def plot_trajectories(
    net: RNNNet,
    simulator: EnvironmentSimulator,
    path: TargetPath,
    starts: List[Tuple[float, float, float]],
    walls: np.ndarray,
    epoch: int,
    output_dir: str,
    cfg: Config,
    rng: np.random.Generator,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    if walls is not None and len(walls) > 0:
        ax.scatter(walls[:, 0], walls[:, 1], s=0.5, c="#aaaaaa", linewidths=0, zorder=1)
    pts = path.points
    ax.plot(pts[:, 0], pts[:, 1], color="#d62728", linewidth=2.0, alpha=0.7,
            zorder=1.5, label="target path")
    cmap = plt.cm.tab10
    for i in range(N_TRAJECTORY_EPISODES):
        s = starts[int(rng.integers(len(starts)))]
        positions, collided = rollout_student(net, simulator, s, cfg, rng)
        positions = np.array(positions)
        c = cmap(i % 10)
        ls = "--" if collided else "-"
        ax.plot(positions[:, 0], positions[:, 1], color=c, linewidth=1.0, linestyle=ls,
                zorder=2, label=f"T{i+1}")
        ax.plot(positions[0, 0],  positions[0, 1],  "o", color=c, markersize=5, zorder=3)
        ax.plot(positions[-1, 0], positions[-1, 1], "x", color=c, markersize=6, zorder=3)
    ax.set_title(f"Epoch {epoch}", fontsize=11)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"trajectories_ep{epoch:04d}.png"), dpi=120)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = Config()
    # Derive from the canonical layout rather than hard-coding widths. The
    # literal 1/7/4 predated the class and pole channels and silently built a
    # 4-input network against 9-channel observations.
    in_dim = len(make_obs_layout(cfg.use_sigma, cfg.blind, cfg.use_poles,
                                 cfg.use_agn))
    cfg.output_dir = os.path.join(
        "PolicyTraining",
        f"{CONDITION}_{cfg.target_arena}{'_blind' if cfg.blind else ''}",
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    sim        = EnvironmentSimulator(cfg.target_arena)
    path       = load_target_path(cfg.target_arena, _settings.data_folder, cfg.path_resample_mm)
    starts_all = make_starts(path, cfg, rng)
    walls      = sim.arena.walls

    if path.start_box is not None and path.start_arrow is not None:
        x_min, y_min, x_max, y_max = path.start_box
        bx, by, tx, ty = path.start_arrow
        arrow_yaw = float(np.degrees(np.arctan2(ty - by, tx - bx)))
        start_desc = (f"box mode  "
                      f"x∈[{x_min:.0f},{x_max:.0f}]  y∈[{y_min:.0f},{y_max:.0f}]  "
                      f"yaw={arrow_yaw:.0f}°±{cfg.single_start_noise_yaw_deg:.0f}°")
    else:
        start_desc = (f"path-aligned (no box defined)  "
                      f"σ_xy={cfg.single_start_noise_xy_mm:.0f}mm  "
                      f"σ_yaw={cfg.single_start_noise_yaw_deg:.0f}°")
    print(f"Arena:           {cfg.target_arena}")
    print(f"Start pool:      {len(starts_all)}  ({start_desc})")
    print(f"Hidden size:     {cfg.hidden_size}")
    print(f"Train episodes:  {cfg.n_train_episodes}")
    print(f"Val episodes:    {cfg.n_val_episodes}")
    print(f"Output dir:      {cfg.output_dir}", flush=True)

    # Disjoint train/val start pools so val tests on unseen starts.
    perm       = rng.permutation(len(starts_all))
    n_val_pool = max(1, len(starts_all) // 5)
    val_starts   = [starts_all[int(i)] for i in perm[:n_val_pool]]
    train_starts = [starts_all[int(i)] for i in perm[n_val_pool:]]

    print("\n→ Teacher diagnostics")
    plot_teacher_field(path, walls, sim.arena, cfg, cfg.output_dir)
    plot_teacher_rollouts(sim, path, train_starts, walls, cfg, cfg.output_dir, rng)
    print(f"  saved teacher_field.png and teacher_rollouts.png to {cfg.output_dir}",
          flush=True)

    print("\n→ Generating training trajectories under teacher")
    Xtr, Ytr = generate_dataset(sim, path, train_starts,
                                cfg.n_train_episodes, cfg, rng, label="train ")
    print("\n→ Generating validation trajectories under teacher")
    Xva, Yva = generate_dataset(sim, path, val_starts,
                                cfg.n_val_episodes,   cfg, rng, label="val   ")

    net = RNNNet(cfg.hidden_size, cfg.max_rotate_deg, in_dim=in_dim)
    opt = torch.optim.Adam(net.parameters(),
                           lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(cfg.n_epochs):
        t0 = time.time()

        net.train()
        running = 0.0; n_b = 0
        for Xb, Yb, Mb in make_batches(Xtr, Ytr, cfg.batch_size, rng):
            pred = net(Xb)
            loss = masked_mse(pred, Yb, Mb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
            opt.step()
            running += float(loss.item()); n_b += 1
        train_loss = running / max(n_b, 1)

        net.eval()
        with torch.no_grad():
            running = 0.0; n_b = 0
            for Xb, Yb, Mb in make_batches(Xva, Yva, cfg.batch_size, rng):
                running += float(masked_mse(net(Xb), Yb, Mb).item())
                n_b += 1
            val_loss = running / max(n_b, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        plot_training_curve(history, cfg.output_dir)

        dt = time.time() - t0
        print(f"[ep {epoch:3d}/{cfg.n_epochs-1}] "
              f"train={train_loss:9.3f}  val={val_loss:9.3f}  ({dt:.1f}s)", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            save_policy(net, cfg, val_loss, epoch,
                        os.path.join(cfg.output_dir, "best_policy.json"))

        if cfg.plot_trajectories_every_n > 0 and epoch % cfg.plot_trajectories_every_n == 0:
            plot_trajectories(net, sim, path, val_starts, walls,
                              epoch, cfg.output_dir, cfg, rng)

    print(f"\nDone. Best val loss: {best_val:.3f}")


if __name__ == "__main__":
    main()
