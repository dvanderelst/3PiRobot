#!/usr/bin/env python3
"""
SCRIPT_TrainPolicy_Burst.py

GA-based policy training with burst scanning: the robot takes N_LOOKS sonar
measurements at each location before committing to a drive direction.

Motivation: bat echolocation calls are often grouped in bursts — a rapid volley
fired while the bat is moving.  The robot takes N_LOOKS sonar measurements per
step, moving straight (no rotation) by intra_burst_drive_mm between consecutive
calls.  Look directions are planned simultaneously before any measurement is
taken; the spread is bounded by max_burst_spread_deg (~30°, matching ~30 ms at
~400 °/s head-rotation speed).  After the burst the robot rotates by r2 and
drives inter_burst_drive_mm forward to begin the next step.

Step sequence per step t:
  1. Build input (history only, zeros for current measurements) → MLP →
       [centre, offset_1..offset_{N-1}, r2].  All N_LOOKS look directions are
       derived as [centre, centre+offset_1, ..., centre+offset_{N-1}].
       Spread is bounded by max_burst_spread_deg.  (flip_look applied)
  2. For k = 1..N_LOOKS:
       a. Measure at look_yaw_k = original_yaw + look_k_physical (at current call position).
       b. If k < N_LOOKS: robot moves straight intra_burst_drive_mm/(N_LOOKS-1)
          with no rotation; a wall collision terminates the episode.
  3. Build input (history + all N_LOOKS measurements) → MLP → r2  (flip_drive)
  4. Execute: body rotates by r2, drives forward inter_burst_drive_mm
       New heading = original_yaw + r2

Look angles in history are stored in network output order (raw[0]=centre,
raw[1..N-1]=offsets). Slot k corresponds to network output k.

IID symmetry uses two flips per step:
  flip_look  = sign of last step's final IID — used for look angle canonicalisation
  flip_drive = sign of this step's final measured IID — used for r2 canonicalisation

Input vector layout  (size = 3 * N_LOOKS * (history_len + 1) + history_len):
  For each look k = 1..N_LOOKS:
    [dist_k_{t-n}...dist_k_{t-1}, dist_k_t]    history_len+1 values
    [iid_k_{t-n}...iid_k_{t-1},  iid_k_t]     history_len+1 values
    [r1_k_{t-n}...r1_k_{t-1},    r1_k_t]      history_len+1 values
  [r2_{t-n}...r2_{t-1}]                        history_len   values

Current-step slots for look k and beyond are zero when deciding look k.

History entry per step: (d1,i1,l1, d2,i2,l2, ..., dN,iN,lN, r2)  — all canonical.
"""

import collections
import dataclasses
import glob
import json
import os

# Limit BLAS/OpenMP threads per worker process before numpy is imported.
# Workers are forked copies of this process, so these take effect in children too.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm

from Library.EnvironmentSimulator import EnvironmentSimulator
from Library import CodeLogger


# ── Burst size ────────────────────────────────────────────────────────────────
N_LOOKS = 2   # sonar measurements per step; spaced intra_burst_drive_mm/(N_LOOKS-1) apart along heading

# ── Condition ─────────────────────────────────────────────────────────────────
CONDITION = "new"
HISTORY_LENGTHS = [1,3]
IID_NOISE_DB = 1

# ── Pushover ──────────────────────────────────────────────────────────────────
try:
    from Library.PushOver import send as _pushover_send
    _PUSHOVER_AVAILABLE = True
except Exception:
    _PUSHOVER_AVAILABLE = False


def pushover_notify(msg: str, title: str = "3PiRobot") -> None:
    if not _PUSHOVER_AVAILABLE:
        return
    try:
        _pushover_send(f"[{title}] {msg}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Policy architecture
    history_len: int = 3          # overridden by main() from HISTORY_LENGTHS
    hidden_sizes: Tuple[int, int] = (32, 16)
    max_rotate1_deg: float = 90.0   # max virtual look rotation per look
    max_rotate2_deg: float = 90.0   # max drive rotation
    max_net_rotation_deg: float = 90.0  # hard cap on |r2| per step
    max_burst_spread_deg: float = 20.0  # max angular span across N_LOOKS look angles
    intra_burst_drive_mm: float = 50.0  # total straight distance driven during burst
    inter_burst_drive_mm: float = 100.0 # straight distance driven after burst

    # Input normalisation
    max_dist_mm: float = 2000.0
    min_dist_mm: float = 300.0
    max_iid_db: float = 12.0

    # Emulator noise
    iid_noise_db: float = 0.0

    # Sensor overrides (diagnostics)
    override_emulator_distance: bool = False
    override_half_angle_deg: float = 30.0
    override_emulator_iid: bool = False

    # Fitness
    angular_bin_deg: float = 10.0
    w_smooth: float = 0.05
    collision_discount: float = 0.1

    # GA
    population_size: int = 200
    generations: int = 50
    elitism_count: int = 5
    mutation_rate: float = 0.05
    mutation_sigma: float = 0.15
    crossover_prob: float = 0.5
    seed: int = 42

    # Evaluation
    episodes_per_policy: int = 250
    max_steps: int = 75
    max_crash_starts_per_session: int = 20
    crash_backtrack_steps: int = 15
    starts_dir: str = "ValidStarts"
    starts_suffixes: List[str] = field(
        default_factory=lambda: ["starts_headon", "starts_wall_left", "starts_wall_right"]
    )
    train_session_names: List[str] = field(
        default_factory=lambda: ["sessionB01", "sessionB02","sessionB03","sessionB04","sessionB05"]
    )
    validation_session_name: Optional[str] = None
    validation_episodes: int = 16

    # IO
    output_dir: str = ""
    pushover_every_n: int = 10
    plot_trajectories_every_n: int = 1
    head_arrow_every_n_steps: int = 10
    head_arrow_length_mm: float = 150.0
    quiet_setup: bool = True
    parallel_eval: bool = True
    num_workers: Optional[int] = None
    save_all_generation_policies: bool = False
    n_best_policies: int = 50


N_TRAJECTORY_EPISODES = 6
BLACK_BOX_STEPS = 10


# ══════════════════════════════════════════════════════════════════════════════
# Policy
# ══════════════════════════════════════════════════════════════════════════════

class MLPPolicy:
    """
    Single MLP with N_LOOKS+1 outputs, called twice per step:
      - Once with zeroed current-measurement slots (history only) to plan
        all N_LOOKS look angles simultaneously.
      - Once with current measurements filled in to produce the drive rotation r2.
    Outputs 0..N_LOOKS-1 are look angles (scaled by max_rotate1_deg).
    Output N_LOOKS is the drive rotation (scaled by max_rotate2_deg).

    Input size: 3 * N_LOOKS * (history_len + 1) + history_len
    Architecture: in_dim → h1 (tanh) → h2 (tanh) → N_LOOKS+1 (tanh).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.in_dim = 3 * N_LOOKS * (cfg.history_len + 1) + cfg.history_len
        h1, h2 = cfg.hidden_sizes
        n_out = N_LOOKS + 1   # look1..lookN + r2
        self.shapes: List[Tuple[int, ...]] = [
            (h1,    self.in_dim), (h1,),
            (h2,    h1),          (h2,),
            (n_out, h2),          (n_out,),
        ]
        self.params: List[np.ndarray] = [np.zeros(s, dtype=np.float32) for s in self.shapes]

    def genome_size(self) -> int:
        return int(sum(np.prod(s) for s in self.shapes))

    def set_genome(self, genome: np.ndarray) -> None:
        g = np.asarray(genome, dtype=np.float32).ravel()
        if g.size != self.genome_size():
            raise ValueError(f"Genome size mismatch: expected {self.genome_size()}, got {g.size}")
        off = 0
        self.params = []
        for s in self.shapes:
            n = int(np.prod(s))
            self.params.append(g[off:off + n].reshape(s))
            off += n

    def get_genome(self) -> np.ndarray:
        return np.concatenate([p.ravel() for p in self.params]).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass. Returns all N_LOOKS+1 outputs as a (N_LOOKS+1,) array in [-1, 1].
        Caller scales: outputs[0..N_LOOKS-1] by max_rotate1_deg, outputs[N_LOOKS] by max_rotate2_deg.
        """
        w1, b1, w2, b2, w3, b3 = self.params
        h = np.tanh(w1 @ x + b1)
        h = np.tanh(w2 @ h + b2)
        return np.tanh(w3 @ h + b3)


# ══════════════════════════════════════════════════════════════════════════════
# Input construction
# ══════════════════════════════════════════════════════════════════════════════

def build_input(
    history: collections.deque,
    current_measurements: List[Tuple[float, float, float]],
    cfg: Config,
) -> np.ndarray:
    """
    Build the flat input vector for one MLP call.

    history: deque of tuples (d1,i1,l1, d2,i2,l2, ..., dN,iN,lN, r2), len = history_len.
    current_measurements: list of (dist_mm, iid_canonical, r1_canonical) for completed
        looks this step.  Pass [] for the look-planning call; pass all N_LOOKS entries
        for the drive call.  Current-step slots for look k are zero when k >= len(current_measurements).

    Layout: for each look k, [dist_k history+current, iid_k history+current, r1_k history+current],
    then [r2 history only].
    """
    md  = cfg.max_dist_mm
    mi  = cfg.max_iid_db
    mr1 = cfg.max_rotate1_deg + cfg.max_burst_spread_deg / 2  # look angles span up to this
    mr2 = cfg.max_rotate2_deg

    parts: List[float] = []
    for k in range(N_LOOKS):
        d_hist = [h[3 * k]     / md  for h in history]
        i_hist = [h[3 * k + 1] / mi  for h in history]
        l_hist = [h[3 * k + 2] / mr1 for h in history]

        if k < len(current_measurements):
            d_curr = current_measurements[k][0] / md
            i_curr = current_measurements[k][1] / mi
            l_curr = current_measurements[k][2] / mr1
        else:
            d_curr = i_curr = l_curr = 0.0

        parts += d_hist + [d_curr]
        parts += i_hist + [i_curr]
        parts += l_hist + [l_curr]

    # r2 history only (no current slot — r2 is the output of the drive call)
    parts += [h[3 * N_LOOKS] / mr2 for h in history]

    return np.array(parts, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Fitness
# ══════════════════════════════════════════════════════════════════════════════

def compute_fitness(
    positions: List[Tuple[float, float]],
    net_turns: List[float],
    collided: bool,
    cfg: Config,
) -> float:
    """
    Angular-coverage fitness with survival and jitter penalty.
    net_turns: list of r2 values (physical body rotation per step).
    """
    steps_survived = len(positions) - 1
    if steps_survived < 1:
        return 0.0

    survival = steps_survived / max(cfg.max_steps, 1)

    xs = np.array([p[0] for p in positions], dtype=np.float64)
    ys = np.array([p[1] for p in positions], dtype=np.float64)
    x_c, y_c = xs.mean(), ys.mean()

    n_bins = max(1, round(360.0 / cfg.angular_bin_deg))
    sum_dists = np.zeros(n_bins, dtype=np.float64)
    counts    = np.zeros(n_bins, dtype=np.int64)

    dx = xs - x_c
    dy = ys - y_c
    dists  = np.hypot(dx, dy)
    angles = np.degrees(np.arctan2(dy, dx)) % 360.0
    bin_idx = (angles / cfg.angular_bin_deg).astype(int) % n_bins
    np.add.at(sum_dists, bin_idx, dists)
    np.add.at(counts,    bin_idx, 1)

    mean_dists = np.where(counts > 0, sum_dists / np.maximum(counts, 1), 0.0)
    coverage = float(np.mean(mean_dists))

    jitter_factor = 1.0
    if cfg.w_smooth > 0.0 and len(net_turns) >= 2:
        jerks = np.abs(np.diff(net_turns))
        max_jerk = 2.0 * cfg.max_net_rotation_deg
        mean_jerk_norm = float(np.mean(jerks)) / max(max_jerk, 1e-6)
        jitter_factor = max(0.0, 1.0 - cfg.w_smooth * mean_jerk_norm)

    discount = cfg.collision_discount if collided else 1.0
    return coverage * survival * jitter_factor * discount


# ══════════════════════════════════════════════════════════════════════════════
# Starts
# ══════════════════════════════════════════════════════════════════════════════

def load_starts(
    session_name: str,
    cfg: Config,
    quiet: bool = False,
) -> List[Tuple[float, float, float]]:
    all_starts: List[Tuple[float, float, float]] = []
    for suffix in cfg.starts_suffixes:
        path = os.path.join(cfg.starts_dir, f"{session_name}_{suffix}.json")
        if not os.path.isfile(path):
            if not quiet:
                print(f"  ⚠ Starts file not found: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        starts = [(float(s["x"]), float(s["y"]), float(s["yaw_deg"])) for s in data.get("starts", [])]
        all_starts.extend(starts)
    if not quiet:
        print(f"  Loaded {len(all_starts)} starts for {session_name} "
              f"(suffixes: {cfg.starts_suffixes})")
    return all_starts


# ══════════════════════════════════════════════════════════════════════════════
# Episode
# ══════════════════════════════════════════════════════════════════════════════

def _apply_r2_clamp(
    r2_physical: float,
    flip_drive: bool,
    cfg: Config,
) -> Tuple[float, float]:
    """Clip r2 to ±max_net_rotation_deg and compute canonical value."""
    r2_physical   = float(np.clip(r2_physical, -cfg.max_net_rotation_deg, cfg.max_net_rotation_deg))
    r2_canonical  = -r2_physical if flip_drive else r2_physical
    return r2_physical, r2_canonical


def _get_measurement(
    simulator: EnvironmentSimulator,
    x: float,
    y: float,
    look_yaw: float,
    cfg: Config,
) -> Tuple[float, float]:
    """Return (dist_mm, physical_iid) for the given position/look direction."""
    need_profile  = cfg.override_emulator_distance or cfg.override_emulator_iid
    need_emulator = (not cfg.override_emulator_distance) or (not cfg.override_emulator_iid)

    profile = simulator.get_profile_at_position(x, y, look_yaw) if need_profile else None
    meas    = simulator.emulator.predict_single(profile) if (need_profile and need_emulator) \
              else (simulator.get_sonar_measurement(x, y, look_yaw) if not need_profile else None)

    if cfg.override_emulator_distance:
        half_opening = simulator.opening_angle / 2
        edges   = np.linspace(-half_opening, half_opening, simulator.profile_steps + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        central = profile[np.abs(centers) <= cfg.override_half_angle_deg]
        valid   = central[~np.isnan(central)]
        geo_dist = float(np.min(valid)) if len(valid) > 0 else cfg.max_dist_mm
        dist_mm  = max(cfg.min_dist_mm, min(geo_dist, cfg.max_dist_mm))
    else:
        dist_mm = max(cfg.min_dist_mm, min(float(meas.get("distance_mm", cfg.max_dist_mm)), cfg.max_dist_mm))

    if cfg.override_emulator_iid:
        half_opening = simulator.opening_angle / 2
        edges   = np.linspace(-half_opening, half_opening, simulator.profile_steps + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        half    = cfg.override_half_angle_deg
        left    = profile[(centers > 0) & (np.abs(centers) <= half)]
        right   = profile[(centers < 0) & (np.abs(centers) <= half)]
        d_left  = float(np.nanmean(left))  if np.any(~np.isnan(left))  else cfg.max_dist_mm
        d_right = float(np.nanmean(right)) if np.any(~np.isnan(right)) else cfg.max_dist_mm
        physical_iid = 20.0 * float(np.log10(max(d_left, 1.0) / max(d_right, 1.0)))
    else:
        physical_iid = float(meas.get("iid_db", 0.0))

    return dist_mm, physical_iid


def _empty_history_entry() -> Tuple:
    return (0.0,) * (3 * N_LOOKS + 1)


def run_episode(
    policy: MLPPolicy,
    simulator: EnvironmentSimulator,
    starts: List[Tuple[float, float, float]],
    cfg: Config,
    rng: np.random.Generator,
) -> Tuple[float, bool]:
    """
    Run one episode.  Returns (fitness, collided).

    IID SYMMETRY: two flips are used per step.
      flip_look: determined from last_physical_iid (previous step's final look) —
                 used to canonicalise the N_LOOKS look angles.  Must be stale since
                 look directions are planned before any current measurement is taken.
      flip_drive: determined from last_meas_iid (final look of the current step) —
                  used to canonicalise r2, matching how the original script derives
                  flip2 from the live measurement.
    History stores canonical values throughout.
    """
    if not starts:
        return 0.0, False

    x, y, yaw = starts[int(rng.integers(len(starts)))]

    history: collections.deque = collections.deque(
        [_empty_history_entry()] * cfg.history_len, maxlen=cfg.history_len
    )
    last_physical_iid = 0.0

    positions: List[Tuple[float, float]] = [(float(x), float(y))]
    net_turns: List[float] = []
    collided = False

    for _ in range(cfg.max_steps):
        original_yaw = yaw
        current_measurements: List[Tuple[float, float, float]] = []

        # ── Look phase: plan all looks in one forward pass (history only) ─────
        flip_look = last_physical_iid < 0.0
        inp  = build_input(history, [], cfg)
        raw  = policy.forward(inp)   # (N_LOOKS+1,) in [-1, 1]
        # output[0] = burst centre; outputs[1..N_LOOKS-1] = offsets within burst
        center_canonical = float(raw[0]) * cfg.max_rotate1_deg
        look_canonicals  = (
            [center_canonical] + [
                center_canonical + float(raw[k]) * (cfg.max_burst_spread_deg / 2)
                for k in range(1, N_LOOKS)
            ]
        )
        look_physicals = [-lc if flip_look else lc for lc in look_canonicals]

        # ── Measure at all planned look directions (with intra-burst movement) ─
        step_drive     = cfg.intra_burst_drive_mm / (N_LOOKS - 1) if N_LOOKS > 1 else 0.0
        call_x, call_y = x, y
        last_meas_iid  = last_physical_iid
        burst_collided = False
        _no_overrides  = not cfg.override_emulator_distance and not cfg.override_emulator_iid

        if step_drive == 0.0 and _no_overrides:
            # Fast path: all looks from same position — one batched CNN forward pass.
            look_yaws = [original_yaw + look_physicals[k] for k in range(N_LOOKS)]
            batch = simulator.get_sonar_measurements_batch(
                [(call_x, call_y, ly) for ly in look_yaws]
            )
            for k in range(N_LOOKS):
                dist_mm     = max(cfg.min_dist_mm, min(float(batch[k]["distance_mm"]), cfg.max_dist_mm))
                physical_iid = float(batch[k]["iid_db"])
                if cfg.iid_noise_db > 0.0:
                    physical_iid += float(rng.normal(0.0, cfg.iid_noise_db))
                iid_canonical = abs(physical_iid)
                current_measurements.append((dist_mm, iid_canonical, look_canonicals[k]))
                last_meas_iid = physical_iid
        else:
            for k in range(N_LOOKS):
                look_yaw = original_yaw + look_physicals[k]
                dist_mm, physical_iid = _get_measurement(simulator, call_x, call_y, look_yaw, cfg)
                if cfg.iid_noise_db > 0.0:
                    physical_iid += float(rng.normal(0.0, cfg.iid_noise_db))
                iid_canonical = abs(physical_iid)
                current_measurements.append((dist_mm, iid_canonical, look_canonicals[k]))
                last_meas_iid = physical_iid

                if k < N_LOOKS - 1 and step_drive > 0.0:
                    burst_result = simulator.simulate_robot_movement(
                        call_x, call_y, original_yaw,
                        [{"rotate1_deg": 0.0, "rotate2_deg": 0.0, "drive_mm": step_drive}],
                        compute_sonar=False,
                    )[0]
                    call_x = float(burst_result["position"]["x"])
                    call_y = float(burst_result["position"]["y"])
                    if burst_result["collision"]["drive_blocked"]:
                        burst_collided = True
                        break

        if burst_collided:
            positions.append((call_x, call_y))
            collided = True
            break

        x, y = call_x, call_y

        # ── Drive phase: second forward pass with current measurements ────────
        flip_drive = last_meas_iid < 0.0
        inp          = build_input(history, current_measurements, cfg)
        raw          = policy.forward(inp)
        r2_canonical = float(raw[N_LOOKS]) * cfg.max_rotate2_deg
        r2_physical  = -r2_canonical if flip_drive else r2_canonical
        r2_physical, r2_canonical = _apply_r2_clamp(r2_physical, flip_drive, cfg)

        # ── Execute movement (body rotates by r2, drives inter_burst_drive_mm) ─
        action = {"rotate1_deg": 0.0, "rotate2_deg": r2_physical, "drive_mm": cfg.inter_burst_drive_mm}
        result = simulator.simulate_robot_movement(
            x, y, original_yaw, [action], compute_sonar=False
        )[0]

        x        = float(result["position"]["x"])
        y        = float(result["position"]["y"])
        yaw      = float(result["orientation"])
        blocked  = bool(result["collision"]["drive_blocked"])

        positions.append((x, y))
        net_turns.append(r2_physical)

        # Store canonical values in history
        hist_entry = tuple(v for d, i, l in current_measurements for v in (d, i, l)) + (r2_canonical,)
        history.append(hist_entry)
        last_physical_iid = last_meas_iid   # IID from final look of this step

        if blocked:
            collided = True
            break

    return compute_fitness(positions, net_turns, collided, cfg), collided


def evaluate_genome(
    genome: np.ndarray,
    simulators: List[EnvironmentSimulator],
    starts_by_session: List[List[Tuple[float, float, float]]],
    cfg: Config,
    rng: np.random.Generator,
    crash_starts_list: Optional[List[List[Tuple[float, float, float]]]] = None,
) -> Tuple[float, float]:
    policy = MLPPolicy(cfg)
    policy.set_genome(genome)

    eps_per_session = max(1, cfg.episodes_per_policy // len(simulators))
    fitnesses:  List[float] = []
    collisions: List[float] = []

    for i, (sim, starts) in enumerate(zip(simulators, starts_by_session)):
        crash_starts = crash_starts_list[i] if crash_starts_list else []
        n_guaranteed = min(len(crash_starts), eps_per_session)
        n_random     = eps_per_session - n_guaranteed

        for cs in crash_starts[:n_guaranteed]:
            fit, col = run_episode(policy, sim, [cs], cfg, rng)
            fitnesses.append(fit)
            collisions.append(float(col))

        for _ in range(n_random):
            fit, col = run_episode(policy, sim, starts, cfg, rng)
            fitnesses.append(fit)
            collisions.append(float(col))

    mean_fit  = float(np.mean(fitnesses))  if fitnesses  else 0.0
    coll_rate = float(np.mean(collisions)) if collisions else 0.0
    return mean_fit, coll_rate


# ══════════════════════════════════════════════════════════════════════════════
# Parallel evaluation
# ══════════════════════════════════════════════════════════════════════════════

_WORKER_SIMS:   Optional[List[EnvironmentSimulator]] = None
_WORKER_STARTS: Optional[List[List[Tuple[float, float, float]]]] = None
_WORKER_CFG:    Optional[Config] = None


def _init_worker(cfg_dict: dict) -> None:
    global _WORKER_SIMS, _WORKER_STARTS, _WORKER_CFG
    cfg = Config(**{k: v for k, v in cfg_dict.items()
                    if k in {f.name for f in dataclasses.fields(Config)}})
    _WORKER_CFG = cfg
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        _WORKER_SIMS   = [EnvironmentSimulator(sn) for sn in cfg.train_session_names]
        _WORKER_STARTS = [load_starts(sn, cfg, quiet=True) for sn in cfg.train_session_names]
    try:
        import torch as _t
        _t.set_num_threads(1)
    except ImportError:
        pass


def _eval_worker(
    args: Tuple[np.ndarray, Optional[List[List[Tuple[float, float, float]]]]]
) -> Tuple[float, float]:
    genome, crash_starts_list = args
    rng = np.random.default_rng()
    return evaluate_genome(genome, _WORKER_SIMS, _WORKER_STARTS, _WORKER_CFG, rng, crash_starts_list)


# ══════════════════════════════════════════════════════════════════════════════
# GA
# ══════════════════════════════════════════════════════════════════════════════

def next_generation(
    population: List[np.ndarray],
    fitnesses: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    sorted_idx = np.argsort(fitnesses)[::-1]
    elites   = [population[i] for i in sorted_idx[:cfg.elitism_count]]
    new_pop  = [e.copy() for e in elites]

    while len(new_pop) < cfg.population_size:
        if len(elites) >= 2 and rng.random() < cfg.crossover_prob:
            i, j  = rng.choice(len(elites), size=2, replace=False)
            mask  = rng.random(len(elites[0])) < 0.5
            child = np.where(mask, elites[i], elites[j]).copy()
        else:
            child = elites[int(rng.integers(len(elites)))].copy()

        mask = rng.random(len(child)) < cfg.mutation_rate
        child[mask] += rng.normal(0.0, cfg.mutation_sigma, int(mask.sum())).astype(np.float32)
        new_pop.append(child)

    return new_pop[:cfg.population_size]


# ══════════════════════════════════════════════════════════════════════════════
# IO
# ══════════════════════════════════════════════════════════════════════════════

def save_policy(policy: MLPPolicy, fitness: float, generation: int, path: str) -> None:
    data = {
        "n_looks":            N_LOOKS,
        "history_len":        policy.cfg.history_len,
        "hidden_sizes":       list(policy.cfg.hidden_sizes),
        "max_rotate1_deg":    policy.cfg.max_rotate1_deg,
        "max_rotate2_deg":    policy.cfg.max_rotate2_deg,
        "max_net_rotation_deg": policy.cfg.max_net_rotation_deg,
        "max_burst_spread_deg": policy.cfg.max_burst_spread_deg,
        "intra_burst_drive_mm": policy.cfg.intra_burst_drive_mm,
        "inter_burst_drive_mm": policy.cfg.inter_burst_drive_mm,
        "max_dist_mm":        policy.cfg.max_dist_mm,
        "max_iid_db":         policy.cfg.max_iid_db,
        "genome_size":        policy.genome_size(),
        "genome":             policy.get_genome().tolist(),
        "fitness":            float(fitness),
        "generation":         int(generation),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_plot(hist: Dict[str, list], output_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    gens = range(len(hist["best"]))

    axes[0].plot(gens, hist["best"], label="train best")
    axes[0].plot(gens, hist["mean"], label="train mean", alpha=0.6)
    if any(np.isfinite(v) for v in hist["val"]):
        axes[0].plot(gens, hist["val"], label="validation", linestyle="--")
    axes[0].set_ylabel("Fitness (mm)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(gens, hist["collision_rate"], color="red")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Collision rate (best genome)")
    axes[1].set_xlabel("Generation")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curve.png"), dpi=120)
    plt.close(fig)


def save_history(hist: Dict[str, list], output_dir: str) -> None:
    def _clean(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    clean = {k: [_clean(v) for v in vals] for k, vals in hist.items()}
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(clean, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Black box
# ══════════════════════════════════════════════════════════════════════════════

_BB_CSS = """
body  { font-family: monospace; font-size: 13px; margin: 24px; color: #222; }
h1    { font-size: 15px; margin-bottom: 4px; }
h2    { font-size: 13px; margin: 20px 0 4px; color: #444; border-top: 1px solid #ddd; padding-top: 8px; }
h3    { font-size: 13px; margin: 12px 0 4px; }
p.note { font-size: 11px; color: #888; margin: 2px 0 10px; }
img   { max-width: 100%; border: 1px solid #ddd; margin-bottom: 14px; display: block; }
table { border-collapse: collapse; margin-bottom: 16px; }
th, td { border: 1px solid #ccc; padding: 3px 8px; text-align: right; white-space: nowrap; }
th    { background: #f0f0f0; text-align: center; }
td.c  { text-align: center; }
tr.crash td { background: #ffe4e4; font-weight: bold; }
tr.look td  { background: #f8f8ff; }
a     { color: #197a4a; text-decoration: none; }
a:hover { text-decoration: underline; }
"""

_BB_INDEX_CSS = """
body  { font-family: monospace; font-size: 13px; margin: 24px; color: #222; }
h1    { font-size: 15px; }
p.note { font-size: 11px; color: #888; margin: 2px 0 12px; }
table { border-collapse: collapse; }
th, td { border: 1px solid #ccc; padding: 3px 12px; text-align: left; }
th    { background: #f0f0f0; }
td.num { text-align: right; }
a     { color: #197a4a; text-decoration: none; }
a:hover { text-decoration: underline; }
"""


def _bb_step_table(steps: List[Dict]) -> str:
    """Render step list as HTML table.  Each step has N_LOOKS look sub-rows + 1 drive row."""
    look_headers = "".join(
        f"<th>x_{k+1}(mm)</th><th>y_{k+1}(mm)</th>"
        f"<th>r1_{k+1}(&deg;)</th><th>dist_{k+1}(mm)</th><th>iid_{k+1}(dB)</th>"
        for k in range(N_LOOKS)
    )
    header = (
        "<tr>"
        "<th>step</th>"
        "<th>x(mm)</th><th>y(mm)</th><th>yaw(&deg;)</th>"
        + look_headers +
        "<th>r2(&deg;)</th><th>net(&deg;)</th>"
        "</tr>"
    )
    rows = []
    for i, s in enumerate(steps):
        cls = ' class="crash"' if i == len(steps) - 1 else ""
        look_cells = "".join(
            f'<td>{lk["call_x_mm"]:.0f}</td>'
            f'<td>{lk["call_y_mm"]:.0f}</td>'
            f'<td>{lk["r1_deg"]:+.1f}</td>'
            f'<td>{lk["emu_dist_mm"]:.1f}</td>'
            f'<td>{lk["emu_iid_db"]:+.2f}</td>'
            for lk in s["looks"]
        )
        rows.append(
            f'<tr{cls}>'
            f'<td class="c">{s["step"]}</td>'
            f'<td>{s["x_mm"]:.1f}</td><td>{s["y_mm"]:.1f}</td><td>{s["yaw_deg"]:+.1f}</td>'
            + look_cells +
            f'<td>{s["rotate2_deg"]:+.1f}</td>'
            f'<td>{s["net_rot_deg"]:+.1f}</td>'
            "</tr>"
        )
    return f'<table>{header}{"".join(rows)}</table>'


def _write_blackbox_html(
    crashes_by_session: Dict[str, List[Dict]],
    generation: int,
    blackbox_dir: str,
) -> None:
    img_src = f"../trajectories_gen{generation:04d}.png"
    sections = []
    for session_name, crashed_trials in crashes_by_session.items():
        trial_blocks = []
        for entry in crashed_trials:
            t     = entry["trial"]
            total = entry["total_steps"]
            steps = entry["last_steps"]
            shown = len(steps)
            trial_blocks.append(
                f'<h3>T{t} &#x2717; &mdash; crashed at step {total - 1} / '
                f'{total} &nbsp;(showing last {shown} steps)</h3>'
                + _bb_step_table(steps)
            )
        sections.append(f'<h2>{session_name}</h2>' + "".join(trial_blocks))

    html = (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<title>Black box — Gen {generation}</title>'
        f'<style>{_BB_CSS}</style></head><body>'
        f'<h1>Generation {generation} — crash log  (N_LOOKS={N_LOOKS})</h1>'
        f'<p class="note">IID &gt; 0 = wall on right &nbsp;|&nbsp; '
        f'IID &lt; 0 = wall on left &nbsp;|&nbsp; '
        f'last row (red) = crash step</p>'
        f'<img src="{img_src}" alt="trajectories gen {generation:04d}">'
        + "".join(sections)
        + "</body></html>"
    )
    path = os.path.join(blackbox_dir, f"blackbox_gen{generation:04d}.html")
    with open(path, "w") as f:
        f.write(html)


def _regenerate_blackbox_index(blackbox_dir: str) -> None:
    index_json = os.path.join(blackbox_dir, "_index.json")
    if not os.path.exists(index_json):
        return
    with open(index_json) as f:
        entries = json.load(f)
    entries.sort(key=lambda e: e["gen"])
    rows = []
    for e in entries:
        gen = e["gen"]
        sessions = e["sessions"]
        n_crashes = sum(len(ts) for ts in sessions.values())
        session_str = ", ".join(
            f"{sn} (T{', T'.join(str(t) for t in ts)})"
            for sn, ts in sessions.items()
        )
        rows.append(
            f'<tr>'
            f'<td class="num"><a href="blackbox_gen{gen:04d}.html">{gen}</a></td>'
            f'<td class="num">{n_crashes}</td>'
            f'<td>{session_str}</td>'
            f'</tr>'
        )
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>Black box index</title>'
        f'<style>{_BB_INDEX_CSS}</style></head><body>'
        '<h1>Black box — crash index</h1>'
        '<p class="note">Only generations with &ge;1 crash in the trajectory plot are listed.</p>'
        '<table><tr><th>gen</th><th>crashes</th><th>sessions / trials</th></tr>'
        + "".join(rows)
        + '</table></body></html>'
    )
    with open(os.path.join(blackbox_dir, "index.html"), "w") as f:
        f.write(html)


def save_blackbox(
    trajs_by_session: Dict[str, Tuple[List[Dict], EnvironmentSimulator]],
    generation: int,
    blackbox_dir: str,
    n_steps: int = BLACK_BOX_STEPS,
) -> bool:
    crashes_by_session: Dict[str, List[Dict]] = {}
    for session_name, (trajectories, _) in trajs_by_session.items():
        crashed = []
        for trial_idx, traj in enumerate(trajectories, 1):
            if not traj["collided"]:
                continue
            steps = traj.get("steps", [])
            crashed.append({
                "trial":       trial_idx,
                "total_steps": len(steps),
                "last_steps":  steps[-n_steps:],
            })
        if crashed:
            crashes_by_session[session_name] = crashed

    if not crashes_by_session:
        return False

    os.makedirs(blackbox_dir, exist_ok=True)
    _write_blackbox_html(crashes_by_session, generation, blackbox_dir)

    index_json = os.path.join(blackbox_dir, "_index.json")
    entries = []
    if os.path.exists(index_json):
        with open(index_json) as f:
            entries = json.load(f)
    entries = [e for e in entries if e["gen"] != generation]
    entries.append({
        "gen": generation,
        "sessions": {sn: [c["trial"] for c in cs] for sn, cs in crashes_by_session.items()},
    })
    with open(index_json, "w") as f:
        json.dump(entries, f)

    _regenerate_blackbox_index(blackbox_dir)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Hall of fame
# ══════════════════════════════════════════════════════════════════════════════

HofEntry = Tuple[float, int, np.ndarray]


def hof_try_insert(
    hof: List[HofEntry],
    fitness: float,
    generation: int,
    genome: np.ndarray,
    n_best: int,
) -> bool:
    if n_best <= 0:
        return False
    for _, _, g in hof:
        if np.array_equal(g, genome):
            return False
    if len(hof) < n_best or fitness > hof[-1][0]:
        hof.append((fitness, generation, genome.copy()))
        hof.sort(key=lambda e: e[0], reverse=True)
        if len(hof) > n_best:
            hof.pop()
        return True
    return False


def save_hof(hof: List[HofEntry], cfg: Config, hof_dir: str) -> None:
    for rank, (fitness, generation, genome) in enumerate(hof, 1):
        pol = MLPPolicy(cfg)
        pol.set_genome(genome)
        data = {
            "rank":               rank,
            "n_looks":            N_LOOKS,
            "history_len":        pol.cfg.history_len,
            "hidden_sizes":       list(pol.cfg.hidden_sizes),
            "max_rotate1_deg":    pol.cfg.max_rotate1_deg,
            "max_rotate2_deg":    pol.cfg.max_rotate2_deg,
            "max_net_rotation_deg": pol.cfg.max_net_rotation_deg,
            "max_burst_spread_deg": pol.cfg.max_burst_spread_deg,
            "intra_burst_drive_mm": pol.cfg.intra_burst_drive_mm,
            "inter_burst_drive_mm": pol.cfg.inter_burst_drive_mm,
            "max_dist_mm":        pol.cfg.max_dist_mm,
            "max_iid_db":         pol.cfg.max_iid_db,
            "genome_size":        pol.genome_size(),
            "genome":             pol.get_genome().tolist(),
            "fitness":            float(fitness),
            "generation":         int(generation),
        }
        with open(os.path.join(hof_dir, f"rank{rank:03d}.json"), "w") as f:
            json.dump(data, f, indent=2)


def _fresh_population(cfg: Config, rng: np.random.Generator) -> List[np.ndarray]:
    genome_size = MLPPolicy(cfg).genome_size()
    return [(rng.standard_normal(genome_size) * 0.1).astype(np.float32)
            for _ in range(cfg.population_size)]


def _load_hof_entries(hof_dir: str) -> List[HofEntry]:
    hof = []
    for path in sorted(glob.glob(os.path.join(hof_dir, "rank*.json"))):
        with open(path) as f:
            d = json.load(f)
        hof.append((float(d["fitness"]), int(d["generation"]),
                    np.array(d["genome"], dtype=np.float32)))
    hof.sort(key=lambda e: e[0], reverse=True)
    return hof


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory plotting
# ══════════════════════════════════════════════════════════════════════════════

def record_trajectories(
    policy: MLPPolicy,
    simulator: EnvironmentSimulator,
    starts: List[Tuple[float, float, float]],
    cfg: Config,
    n_episodes: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """Run n_episodes and return trajectory info including per-step look data."""
    geo_cfg = dataclasses.replace(cfg, override_emulator_distance=True, override_emulator_iid=True)

    trajectories = []
    for _ in range(n_episodes):
        if not starts:
            break
        x, y, yaw = starts[int(rng.integers(len(starts)))]
        start_pos  = (float(x), float(y), float(yaw))
        history: collections.deque = collections.deque(
            [_empty_history_entry()] * cfg.history_len, maxlen=cfg.history_len
        )
        last_physical_iid = 0.0
        positions  = [(float(x), float(y))]
        body_yaws  = [float(yaw)]
        # look_yaws_per_step: list of lists — one inner list per step, N_LOOKS entries each
        look_yaws_per_step: List[List[float]] = []
        steps_data: List[Dict] = []
        collided = False

        for step_idx in range(cfg.max_steps):
            original_yaw = yaw
            current_measurements: List[Tuple[float, float, float]] = []
            step_looks: List[Dict] = []
            step_look_yaws: List[float] = []

            # ── Look phase: plan all looks in one forward pass ────────────────
            flip_look = last_physical_iid < 0.0
            inp  = build_input(history, [], cfg)
            raw  = policy.forward(inp)
            center_canonical = float(raw[0]) * cfg.max_rotate1_deg
            look_canonicals  = (
                [center_canonical] + [
                    center_canonical + float(raw[k]) * (cfg.max_burst_spread_deg / 2)
                    for k in range(1, N_LOOKS)
                ]
            )
            look_physicals = [-lc if flip_look else lc for lc in look_canonicals]

            # ── Measure at all planned look directions (with intra-burst movement) ─
            step_drive     = cfg.intra_burst_drive_mm / (N_LOOKS - 1) if N_LOOKS > 1 else 0.0
            call_x, call_y = x, y
            last_meas_iid  = last_physical_iid
            burst_collided = False

            for k in range(N_LOOKS):
                look_yaw = original_yaw + look_physicals[k]
                step_look_yaws.append(look_yaw)

                dist_mm, physical_iid = _get_measurement(simulator, call_x, call_y, look_yaw, cfg)
                emu_dist = dist_mm
                emu_iid  = physical_iid
                if cfg.iid_noise_db > 0.0:
                    physical_iid += float(rng.normal(0.0, cfg.iid_noise_db))

                geo_dist, geo_iid = _get_measurement(simulator, call_x, call_y, look_yaw, geo_cfg)
                iid_canonical = abs(physical_iid)
                current_measurements.append((dist_mm, iid_canonical, look_canonicals[k]))
                last_meas_iid = physical_iid

                step_looks.append({
                    "call_x_mm":   round(float(call_x), 1),
                    "call_y_mm":   round(float(call_y), 1),
                    "r1_deg":      round(look_physicals[k], 1),
                    "emu_dist_mm": round(emu_dist, 1),
                    "emu_iid_db":  round(emu_iid, 2),
                    "geo_dist_mm": round(geo_dist, 1),
                    "geo_iid_db":  round(geo_iid, 2),
                })

                if k < N_LOOKS - 1 and step_drive > 0.0:
                    burst_result = simulator.simulate_robot_movement(
                        call_x, call_y, original_yaw,
                        [{"rotate1_deg": 0.0, "rotate2_deg": 0.0, "drive_mm": step_drive}],
                        compute_sonar=False,
                    )[0]
                    call_x = float(burst_result["position"]["x"])
                    call_y = float(burst_result["position"]["y"])
                    if burst_result["collision"]["drive_blocked"]:
                        burst_collided = True
                        break

            if burst_collided:
                positions.append((call_x, call_y))
                body_yaws.append(original_yaw)
                look_yaws_per_step.append(step_look_yaws)
                collided = True
                break

            x, y = call_x, call_y

            # ── Drive phase ───────────────────────────────────────────────────
            flip_drive = last_meas_iid < 0.0
            inp          = build_input(history, current_measurements, cfg)
            raw          = policy.forward(inp)
            r2_canonical = float(raw[N_LOOKS]) * cfg.max_rotate2_deg
            r2_physical  = -r2_canonical if flip_drive else r2_canonical
            r2_physical, r2_canonical = _apply_r2_clamp(r2_physical, flip_drive, cfg)

            action = {"rotate1_deg": 0.0, "rotate2_deg": r2_physical, "drive_mm": cfg.inter_burst_drive_mm}
            result = simulator.simulate_robot_movement(x, y, original_yaw, [action], compute_sonar=False)[0]
            x   = float(result["position"]["x"])
            y   = float(result["position"]["y"])
            yaw = float(result["orientation"])

            positions.append((x, y))
            body_yaws.append(float(yaw))
            look_yaws_per_step.append(step_look_yaws)

            hist_entry = tuple(v for d, i, l in current_measurements for v in (d, i, l)) + (r2_canonical,)
            history.append(hist_entry)
            last_physical_iid = last_meas_iid

            steps_data.append({
                "step":        step_idx,
                "x_mm":        round(float(positions[-2][0]), 1),
                "y_mm":        round(float(positions[-2][1]), 1),
                "yaw_deg":     round(original_yaw, 1),
                "looks":       step_looks,
                "rotate2_deg": round(r2_physical, 1),
                "net_rot_deg": round(r2_physical, 1),
            })

            if result["collision"]["drive_blocked"]:
                collided = True
                break

        trajectories.append({
            "positions":           positions,
            "body_yaws":           body_yaws,
            "look_yaws_per_step":  look_yaws_per_step,
            "collided":            collided,
            "steps":               steps_data,
            "start":               start_pos,
        })
    return trajectories


def plot_trajectories(
    trajs_by_session: Dict[str, Tuple[List[Dict], EnvironmentSimulator]],
    generation: int,
    fitness: float,
    cfg: Config,
    output_dir: str,
    crash_starts_by_session: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
) -> None:
    n = len(trajs_by_session)
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle(f"Gen {generation}  |  best fitness = {fitness:.1f}  |  N_LOOKS={N_LOOKS}", fontsize=11)

    colours    = plt.cm.tab10(np.linspace(0, 1, N_TRAJECTORY_EPISODES))
    arrow_len  = cfg.head_arrow_length_mm
    arrow_every = cfg.head_arrow_every_n_steps

    for idx, (session_name, (trajectories, simulator)) in enumerate(trajs_by_session.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        walls = simulator.arena.walls
        if len(walls) > 0:
            ax.scatter(walls[:, 0], walls[:, 1], s=0.5, c="#aaaaaa", linewidths=0, zorder=1)

        for traj, colour in zip(trajectories, colours):
            xs = [p[0] for p in traj["positions"]]
            ys = [p[1] for p in traj["positions"]]

            ax.plot(xs, ys, color=colour, linewidth=0.8,
                    linestyle="--" if traj["collided"] else "-", zorder=2)
            ax.plot(xs[0],  ys[0],  "o", color=colour, markersize=4, zorder=3)
            ax.plot(xs[-1], ys[-1], "x", color=colour, markersize=5, zorder=3)

            # Draw arrows for each look direction every N steps
            if arrow_every > 0:
                look_yaws_per_step = traj.get("look_yaws_per_step", [])
                for step, step_look_yaws in enumerate(look_yaws_per_step):
                    if step % arrow_every != 0:
                        continue
                    px, py = xs[step], ys[step]
                    for look_yaw in step_look_yaws:
                        rad = np.deg2rad(look_yaw)
                        ax.quiver(
                            px, py,
                            np.cos(rad) * arrow_len, np.sin(rad) * arrow_len,
                            angles="xy", scale_units="xy", scale=1,
                            color=colour, alpha=0.5, width=0.002,
                            headwidth=3, headlength=3, zorder=4,
                        )

        if crash_starts_by_session:
            pool = crash_starts_by_session.get(session_name, [])
            for (px, py, pyaw) in pool:
                ax.plot(px, py, "x", color="black", markersize=5,
                        markeredgewidth=1.2, zorder=5, alpha=0.7)
                rad = np.deg2rad(pyaw)
                ax.quiver(px, py,
                          np.cos(rad) * arrow_len * 0.7, np.sin(rad) * arrow_len * 0.7,
                          angles="xy", scale_units="xy", scale=1,
                          color="black", alpha=0.5, width=0.002,
                          headwidth=3, headlength=3, zorder=5)

        legend_handles = [
            Line2D([0], [0], color=c,
                   linestyle="--" if t["collided"] else "-",
                   linewidth=1.5,
                   label=f"T{i + 1}" + (" \u2717" if t["collided"] else ""))
            for i, (t, c) in enumerate(zip(trajectories, colours))
        ]
        ax.legend(handles=legend_handles, fontsize=6, loc="upper left",
                  framealpha=0.5, ncol=2, handlelength=1.5)

        n_coll = sum(1 for t in trajectories if t["collided"])
        ax.set_title(f"{session_name}  |  coll {n_coll}/{len(trajectories)}", fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Y (mm)", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"trajectories_gen{generation:04d}.png"), dpi=120)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def build_simulator(session_name: str, quiet: bool = True) -> EnvironmentSimulator:
    if quiet:
        with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
            return EnvironmentSimulator(session_name)
    return EnvironmentSimulator(session_name)


def train(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    hof_dir         = os.path.join(cfg.output_dir, "top_policies")
    checkpoint_path = os.path.join(cfg.output_dir, "checkpoint.npz")

    start_gen    = 0
    hof: List[HofEntry] = []
    hist: Dict[str, list] = {
        "best": [], "mean": [], "std": [], "min": [],
        "val": [], "val_std": [], "collision_rate": [], "mean_collision_rate": [],
    }
    best_genome, best_fitness = None, -np.inf
    crash_starts_by_session: Dict[str, List[Tuple[float, float, float]]] = {}
    crash_starts_path = os.path.join(cfg.output_dir, "crash_starts.json")

    if os.path.exists(checkpoint_path):
        ck        = np.load(checkpoint_path)
        saved_gen = int(ck["generation"])

        saved_cfg_path = os.path.join(cfg.output_dir, "config.json")
        if os.path.exists(saved_cfg_path):
            with open(saved_cfg_path) as f:
                saved = json.load(f)
            cur = asdict(cfg)
            def _eq(a, b):
                if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                    return list(a) == list(b)
                return a == b
            mismatches = [
                f"  {k}: saved={saved.get(k)!r}  current={cur[k]!r}"
                for k in cur
                if k != "output_dir" and not _eq(cur[k], saved.get(k))
            ]
            if mismatches:
                print(f"Config mismatch — cannot resume '{cfg.output_dir}':")
                for m in mismatches:
                    print(m)
                print("Fix the config or choose a different output_dir.")
                return

        response = input(
            f"Checkpoint found in '{cfg.output_dir}' (gen {saved_gen}). Resume? [Y/n]: "
        )
        if response.strip().lower() != "n":
            population = [ck["population"][i] for i in range(ck["population"].shape[0])]
            start_gen  = saved_gen + 1
            bp_path = os.path.join(cfg.output_dir, "best_policy.json")
            if os.path.exists(bp_path):
                with open(bp_path) as f:
                    bp = json.load(f)
                best_genome  = np.array(bp["genome"], dtype=np.float32)
                best_fitness = float(bp["fitness"])
            hist_path = os.path.join(cfg.output_dir, "training_history.json")
            if os.path.exists(hist_path):
                with open(hist_path) as f:
                    raw = json.load(f)
                hist = {k: [v if v is not None else float("nan") for v in vals]
                        for k, vals in raw.items()}
            if os.path.isdir(hof_dir):
                hof = _load_hof_entries(hof_dir)
            if os.path.exists(crash_starts_path):
                with open(crash_starts_path) as f:
                    raw = json.load(f)
                crash_starts_by_session = {
                    sn: [tuple(s) for s in starts]
                    for sn, starts in raw.items()
                }
            print(f"Resuming from generation {start_gen}  (best so far: {best_fitness:.1f})")
        else:
            population = _fresh_population(cfg, rng)
    elif os.path.exists(cfg.output_dir) and any(
        f.endswith(".json") for f in os.listdir(cfg.output_dir)
    ):
        response = input(f"Output dir '{cfg.output_dir}' already has data. Overwrite? [y/N]: ")
        if response.strip().lower() != "y":
            print(f"Skipping history_len={cfg.history_len}.")
            return
        population = _fresh_population(cfg, rng)
    else:
        population = _fresh_population(cfg, rng)

    if cfg.n_best_policies > 0:
        os.makedirs(hof_dir, exist_ok=True)

    if start_gen == 0:
        with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
            json.dump({**asdict(cfg), "n_looks": N_LOOKS}, f, indent=2)
        CodeLogger.log_code(cfg.output_dir, [".", "Library"], label="policy_burst")

    template = MLPPolicy(cfg)
    print(f"N_LOOKS:     {N_LOOKS}")
    print(f"Input dim:   {template.in_dim}")
    print(f"Genome size: {template.genome_size()}")
    print(f"Output dir:  {cfg.output_dir}")

    train_sims   = [build_simulator(sn, cfg.quiet_setup) for sn in cfg.train_session_names]
    train_starts = [load_starts(sn, cfg) for sn in cfg.train_session_names]

    val_sim    = build_simulator(cfg.validation_session_name, cfg.quiet_setup) \
                 if cfg.validation_session_name else None
    val_starts = load_starts(cfg.validation_session_name, cfg) \
                 if cfg.validation_session_name else []

    all_plot_sessions = list(cfg.train_session_names) + (
        [cfg.validation_session_name] if cfg.validation_session_name else []
    )
    plot_sims_by_session: Dict[str, Tuple[EnvironmentSimulator, List]] = {}
    for sn in all_plot_sessions:
        sim    = build_simulator(sn, quiet=True)
        starts = load_starts(sn, cfg, quiet=True)
        if starts:
            plot_sims_by_session[sn] = (sim, starts)

    executor = None
    if cfg.parallel_eval:
        n_workers = cfg.num_workers or max(1, (os.cpu_count() or 2) - 2)
        executor  = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(asdict(cfg),),
        )

    try:
        for gen in tqdm(range(start_gen, cfg.generations), desc="GA"):

            crash_starts_list = [
                crash_starts_by_session.get(sn, [])
                for sn in cfg.train_session_names
            ]
            if cfg.parallel_eval:
                results = list(executor.map(
                    _eval_worker, [(g, crash_starts_list) for g in population]
                ))
            else:
                results = [
                    evaluate_genome(g, train_sims, train_starts, cfg, rng, crash_starts_list)
                    for g in population
                ]

            fitnesses  = np.array([r[0] for r in results], dtype=np.float32)
            coll_rates = np.array([r[1] for r in results], dtype=np.float32)
            best_idx   = int(np.argmax(fitnesses))
            gen_best   = float(fitnesses[best_idx])
            gen_mean   = float(np.mean(fitnesses))
            gen_std    = float(np.std(fitnesses))
            gen_min    = float(np.min(fitnesses))
            gen_coll   = float(coll_rates[best_idx])
            gen_mean_coll = float(np.mean(coll_rates))

            if gen_best > best_fitness:
                best_fitness = gen_best
                best_genome  = population[best_idx].copy()
                pol = MLPPolicy(cfg)
                pol.set_genome(best_genome)
                save_policy(pol, best_fitness, gen,
                            os.path.join(cfg.output_dir, "best_policy.json"))

            if cfg.n_best_policies > 0:
                hof_changed = False
                for genome, fitness in sorted(
                    zip(population, fitnesses.tolist()),
                    key=lambda x: x[1], reverse=True,
                ):
                    if hof_try_insert(hof, fitness, gen, genome, cfg.n_best_policies):
                        hof_changed = True
                if hof_changed:
                    save_hof(hof, cfg, hof_dir)

            if cfg.save_all_generation_policies:
                pol = MLPPolicy(cfg)
                pol.set_genome(population[best_idx].copy())
                save_policy(pol, gen_best, gen,
                            os.path.join(cfg.output_dir, f"gen_{gen:04d}_policy.json"))

            val_fit = float("nan")
            val_std = float("nan")
            if val_sim and val_starts and best_genome is not None:
                val_pol = MLPPolicy(cfg)
                val_pol.set_genome(best_genome)
                val_results = [
                    run_episode(val_pol, val_sim, val_starts, cfg, rng)
                    for _ in range(cfg.validation_episodes)
                ]
                val_fits = [r[0] for r in val_results]
                val_fit  = float(np.mean(val_fits))
                val_std  = float(np.std(val_fits))

            if (cfg.plot_trajectories_every_n > 0
                    and gen % cfg.plot_trajectories_every_n == 0
                    and plot_sims_by_session and best_genome is not None):
                traj_pol = MLPPolicy(cfg)
                traj_pol.set_genome(best_genome)
                trajs_by_session = {
                    sn: (
                        record_trajectories(traj_pol, sim, starts, cfg, N_TRAJECTORY_EPISODES, rng),
                        sim,
                    )
                    for sn, (sim, starts) in plot_sims_by_session.items()
                }
                plot_trajectories(trajs_by_session, gen, best_fitness, cfg, cfg.output_dir,
                                  crash_starts_by_session=crash_starts_by_session)
                save_blackbox(trajs_by_session, gen, os.path.join(cfg.output_dir, "blackbox"))

                pool_changed = False
                for sn, (trajectories, _) in trajs_by_session.items():
                    for traj in trajectories:
                        if traj["collided"]:
                            positions  = traj["positions"]
                            body_yaws  = traj.get("body_yaws", [])
                            if not body_yaws:
                                continue
                            k   = min(cfg.crash_backtrack_steps, len(positions) - 1)
                            idx = -(k + 1)
                            bx, by = positions[idx]
                            byaw   = body_yaws[idx]
                            pool   = crash_starts_by_session.setdefault(sn, [])
                            pool.append((bx, by, byaw))
                            if len(pool) > cfg.max_crash_starts_per_session:
                                crash_starts_by_session[sn] = pool[-cfg.max_crash_starts_per_session:]
                            pool_changed = True

                if crash_starts_by_session and best_genome is not None:
                    retire_pol = MLPPolicy(cfg)
                    retire_pol.set_genome(best_genome)
                    for sn, sim in zip(cfg.train_session_names, train_sims):
                        pool = crash_starts_by_session.get(sn, [])
                        if not pool:
                            continue
                        survivors = []
                        for cs in pool:
                            _, col = run_episode(retire_pol, sim, [cs], cfg, rng)
                            if col:
                                survivors.append(cs)
                        if len(survivors) < len(pool):
                            crash_starts_by_session[sn] = survivors
                            pool_changed = True

                if pool_changed:
                    with open(crash_starts_path, "w") as f:
                        json.dump(crash_starts_by_session, f)

                n_traj_coll  = sum(t["collided"] for trajs, _ in trajs_by_session.values() for t in trajs)
                n_traj_total = sum(len(trajs)     for trajs, _ in trajs_by_session.values())
                traj_coll_rate = n_traj_coll / n_traj_total if n_traj_total > 0 else float("nan")
            else:
                traj_coll_rate = float("nan")

            hist["best"].append(gen_best)
            hist["mean"].append(gen_mean)
            hist["std"].append(gen_std)
            hist["min"].append(gen_min)
            hist["val"].append(val_fit)
            hist["val_std"].append(val_std)
            hist["collision_rate"].append(gen_coll)
            hist["mean_collision_rate"].append(gen_mean_coll)

            traj_str  = f"  traj_coll={traj_coll_rate:.2f}" if not np.isnan(traj_coll_rate) else ""
            n_crash_pool = sum(len(v) for v in crash_starts_by_session.values())
            crash_str = f"  crash_pool={n_crash_pool}" if n_crash_pool > 0 else ""
            tqdm.write(
                f"Gen {gen:4d} | best={gen_best:7.1f}  mean={gen_mean:7.1f}"
                f"  coll={gen_coll:.2f}{traj_str}{crash_str}"
            )

            population = next_generation(population, fitnesses, cfg, rng)

            np.savez(checkpoint_path,
                     population=np.stack(population),
                     generation=np.array(gen))

            if cfg.pushover_every_n > 0 and (gen + 1) % cfg.pushover_every_n == 0:
                pushover_notify(
                    f"burst h{cfg.history_len:02d} Gen {gen+1}/{cfg.generations}"
                    f"  best={gen_best:.0f}  coll={gen_coll:.2f}"
                )

            save_plot(hist, cfg.output_dir)
            save_history(hist, cfg.output_dir)

    finally:
        if executor:
            executor.shutdown(wait=False)

    print(f"\nDone. Best fitness: {best_fitness:.1f}")
    print(f"Results saved to:  {cfg.output_dir}")
    pushover_notify(f"Done burst h{cfg.history_len:02d}. Best fitness: {best_fitness:.1f}")


def main() -> None:
    for history_len in HISTORY_LENGTHS:
        cfg = Config(
            history_len=history_len,
            output_dir=f"PolicyTraining/{CONDITION}_burst_h{history_len:02d}",
            iid_noise_db=IID_NOISE_DB,
        )
        print(f"\n{'='*60}")
        print(f"Training burst policy  history_len={history_len}  →  {cfg.output_dir}")
        print(f"{'='*60}")
        train(cfg)


if __name__ == "__main__":
    main()
