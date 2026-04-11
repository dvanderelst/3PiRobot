#!/usr/bin/env python3
"""
SCRIPT_TrainPolicy2.py

GA-based policy training, implemented directly from rationale.txt.
Clean reimplementation — single MLP called twice per step, with IID bilateral symmetry wrapper.

Step sequence per step t:
  1. Build input with zero in current slot → MLP → rotate1
  2. look_yaw = current_yaw + rotate1
  3. Sonar measurement at look_yaw → distance_mm, iid_db
  4. Build input with actual measurement in current slot → MLP → rotate2
  5. Execute: robot rotates by rotate1 then rotate2, drives forward fixed_drive_mm
     New heading = current_yaw + rotate1 + rotate2

Input vector layout (size = 4 * history_len + 3):
  [dist_{t-n}...dist_{t-1}, dist_current]   n+1 values
  [iid_{t-n}...iid_{t-1},  iid_current]    n+1 values
  [r1_{t-n}...r1_{t-1},    r1_current]     n+1 values
  [r2_{t-n}...r2_{t-1}]                    n   values

Fitness (per episode):
  - coverage = mean over angular bins of mean distance from centroid (0 for empty bins)
  - survival = steps_survived / max_steps  (early termination on collision reduces this)
  - jitter_factor = 1 - w_smooth * mean(|turn_t - turn_{t-1}|) / max_possible_jerk
  - fitness = coverage * survival * jitter_factor
"""

import collections
import dataclasses
import glob
import json
import os
import sys
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


# ── Condition ────────────────────────────────────────────────────────────────────
CONDITION = "policy1"          # base name; output goes to PolicyTraining/<CONDITION>_hNN/
HISTORY_LENGTHS = [10, 5, 1, 0]  # train one run per history length, in order
IID_NOISE_DB = 1         # Gaussian noise std injected into emulator IID during training (dB); 0 = disabled

# ── Pushover ─────────────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Policy architecture
    history_len: int = 5  # overridden by main() from HISTORY_LENGTHS
    include_r1_in_input: bool = True  # if False, r1 slot removed from input vector (required when force_aligned=True)
    force_aligned: bool = False       # if True, rotate1 always 0 (head fixed to body) — baseline only
    hidden_sizes: Tuple[int, int] = (32, 16)
    max_rotate1_deg: float = 90.0
    max_rotate2_deg: float = 90.0
    max_net_rotation_deg: float = 90.0  # hard cap on |rotate1 + rotate2| per step
    fixed_drive_mm: float = 100.0

    # Input normalisation constants
    max_dist_mm: float = 2000.0     # distances divided by this before entering network
    min_dist_mm: float = 300.0      # sonar saturation floor (real robot cannot return below this)
    max_iid_db: float = 12.0        # IID divided by this before entering network

    # Emulator noise injection (applied during both fitness evaluation and trajectory plotting)
    iid_noise_db: float = 0.0       # std of Gaussian noise added to emulator IID output (dB); set via IID_NOISE_DB at top of script

    # Sensor overrides (for diagnostics — isolate emulator problems from GA/fitness problems)
    override_emulator_distance: bool = False        # replace emulator distance with geometric min over central cone
    override_half_angle_deg: float = 30.0  # half-width of cone used for both distance and IID overrides (degrees)
    override_emulator_iid: bool = False             # replace emulator IID with geometric 10·log10(d_left_min/d_right_min); implies distance override

    # Fitness
    angular_bin_deg: float = 10.0      # width of angular bins for coverage metric
    w_smooth: float = 0.05              # jitter penalty weight (0 = disabled, 1 = full)
    collision_discount: float = 0.1    # fitness multiplier on collision (< 1 penalises crashes)

    # GA
    population_size: int = 100
    generations: int = 50
    elitism_count: int = 5
    mutation_rate: float = 0.05          # fraction of weights perturbed per offspring
    mutation_sigma: float = 0.15
    crossover_prob: float = 0.5         # probability of crossover vs. single-parent mutation
    seed: int = 42

    # Evaluation
    episodes_per_policy: int = 240
    max_steps: int = 75
    max_crash_starts_per_session: int = 20  # cap on the per-session crash-start pool
    crash_backtrack_steps: int = 15         # how many steps before the crash to place the backtrack start
    starts_dir: str = "ValidStarts"
    starts_suffixes: List[str] = field(
        default_factory=lambda: ["starts_headon", "starts_wall_left", "starts_wall_right"] #"starts_wall_left", "starts_wall_right",
    )
    train_session_names: List[str] = field(
        default_factory=lambda: ["sessionB01", "sessionB02", "sessionB03", "sessionB04", "sessionB05"]
    )
    validation_session_name: Optional[str] = None
    validation_episodes: int = 16

    # IO
    output_dir: str = ""                # set by main() from CONDITION + history_len; do not set here
    pushover_every_n: int = 10          # 0 to disable
    plot_trajectories_every_n: int = 1  # 0 to disable; plots N_TRAJECTORY_EPISODES example paths
    head_arrow_every_n_steps: int = 10  # draw a head-direction arrow every N steps (0 to disable)
    head_arrow_length_mm: float = 150.0 # length of head-direction arrows in mm
    quiet_setup: bool = True
    parallel_eval: bool = True
    num_workers: Optional[int] = None
    save_all_generation_policies: bool = False
    n_best_policies: int = 50       # hall-of-fame size; 0 to disable

    def __post_init__(self):
        if self.force_aligned and self.include_r1_in_input:
            raise ValueError(
                "force_aligned=True requires include_r1_in_input=False "
                "(rotate1 is always 0, so the r1 slot must be removed from the input)"
            )


N_TRAJECTORY_EPISODES = 6   # number of example episodes to plot per trajectory snapshot
BLACK_BOX_STEPS = 10        # number of final steps to record for each crashed episode


# ══════════════════════════════════════════════════════════════════════════════════
# Policy
# ══════════════════════════════════════════════════════════════════════════════════

class MLPPolicy:
    """
    Single MLP called once (baseline, force_aligned=True) or twice (history policies) per step.
    When called twice: first to produce rotate1 (look direction), then rotate2 (body turn).
    When force_aligned: rotate1 is always 0 and only the rotate2 call is made.

    Input size: 4 * history_len + 3  (or +2 if include_r1_in_input=False)
    Architecture: in_dim → h1 (tanh) → h2 (tanh) → 1 (tanh), scaled to ±max_rotate_deg.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.in_dim = 4 * cfg.history_len + (3 if cfg.include_r1_in_input else 2)
        h1, h2 = cfg.hidden_sizes
        self.shapes: List[Tuple[int, ...]] = [
            (h1, self.in_dim), (h1,),   # layer 1
            (h2, h1),          (h2,),   # layer 2
            (1,  h2),          (1,),    # output
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

    def forward(self, x: np.ndarray, max_rotate_deg: float) -> float:
        """Forward pass. Returns rotation in degrees ∈ [-max_rotate_deg, +max_rotate_deg]."""
        v = x.reshape(-1, 1)
        w1, b1, w2, b2, w3, b3 = self.params
        h = np.tanh(w1 @ v + b1.reshape(-1, 1))
        h = np.tanh(w2 @ h + b2.reshape(-1, 1))
        return float(np.tanh(w3 @ h + b3.reshape(-1, 1))[0, 0]) * max_rotate_deg


# ══════════════════════════════════════════════════════════════════════════════════
# Input construction
# ══════════════════════════════════════════════════════════════════════════════════

def build_input(
    history: collections.deque,
    dist_current: float,
    iid_current: float,
    r1_current: float,
    cfg: Config,
) -> np.ndarray:
    """
    Build the flat input vector for one MLP call.

    history: deque of (dist_mm, iid_db, rotate1_deg, rotate2_deg), len = history_len.
    For the rotate1 call pass dist_current=iid_current=r1_current=0.
    For the rotate2 call pass the actual measured values.
    """
    md  = cfg.max_dist_mm
    mi  = cfg.max_iid_db
    mr1 = cfg.max_rotate1_deg
    mr2 = cfg.max_rotate2_deg

    dists = [h[0] / md  for h in history] + [dist_current / md]
    iids  = [h[1] / mi  for h in history] + [iid_current  / mi]
    r2s   = [h[3] / mr2 for h in history]

    if cfg.include_r1_in_input:
        r1s = [h[2] / mr1 for h in history] + [r1_current / mr1]
        return np.array(dists + iids + r1s + r2s, dtype=np.float32)
    else:
        return np.array(dists + iids + r2s, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════════
# Fitness
# ══════════════════════════════════════════════════════════════════════════════════

def compute_fitness(
    positions: List[Tuple[float, float]],
    net_turns: List[float],
    collided: bool,
    cfg: Config,
) -> float:
    """
    Angular-coverage fitness with survival and jitter penalty.

    coverage      = mean over angular bins of mean distance from centroid (0 for empty bins)
    survival      = steps_survived / max_steps  (early termination on collision reduces this)
    jitter_factor = 1 - w_smooth * mean(|turn_t - turn_{t-1}|) / max_possible_jerk
    fitness       = coverage * survival * jitter_factor
    """
    steps_survived = len(positions) - 1  # positions includes start
    if steps_survived < 1:
        return 0.0

    survival = steps_survived / max(cfg.max_steps, 1)

    # Angular coverage: spread of trajectory around its centroid.
    xs = np.array([p[0] for p in positions], dtype=np.float64)
    ys = np.array([p[1] for p in positions], dtype=np.float64)
    x_c, y_c = xs.mean(), ys.mean()

    n_bins = max(1, round(360.0 / cfg.angular_bin_deg))
    sum_dists = np.zeros(n_bins, dtype=np.float64)
    counts    = np.zeros(n_bins, dtype=np.int64)

    dx = xs - x_c
    dy = ys - y_c
    dists = np.hypot(dx, dy)
    angles = np.degrees(np.arctan2(dy, dx)) % 360.0
    bin_idx = (angles / cfg.angular_bin_deg).astype(int) % n_bins
    np.add.at(sum_dists, bin_idx, dists)
    np.add.at(counts,    bin_idx, 1)

    mean_dists = np.where(counts > 0, sum_dists / np.maximum(counts, 1), 0.0)
    coverage = float(np.mean(mean_dists))

    # Jitter penalty: penalise step-to-step reversals in net heading turn.
    jitter_factor = 1.0
    if cfg.w_smooth > 0.0 and len(net_turns) >= 2:
        jerks = np.abs(np.diff(net_turns))
        max_jerk = 2.0 * cfg.max_net_rotation_deg
        mean_jerk_norm = float(np.mean(jerks)) / max(max_jerk, 1e-6)
        jitter_factor = max(0.0, 1.0 - cfg.w_smooth * mean_jerk_norm)

    discount = cfg.collision_discount if collided else 1.0
    return coverage * survival * jitter_factor * discount


# ══════════════════════════════════════════════════════════════════════════════════
# Starts
# ══════════════════════════════════════════════════════════════════════════════════

def load_starts(
    session_name: str,
    cfg: Config,
    quiet: bool = False,
) -> List[Tuple[float, float, float]]:
    """Load and pool starts from all configured suffixes."""
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


# ══════════════════════════════════════════════════════════════════════════════════
# Episode
# ══════════════════════════════════════════════════════════════════════════════════

def _apply_net_rotation_clamp(
    rotate1: float,
    rotate2: float,
    flip2: bool,
    cfg: Config,
) -> Tuple[float, float]:
    """
    Clip rotate2 so that |rotate1 + rotate2| <= max_net_rotation_deg.
    Returns (rotate2_clipped, rotate2_canonical_clipped).
    """
    lo = -cfg.max_net_rotation_deg - rotate1
    hi =  cfg.max_net_rotation_deg - rotate1
    rotate2 = float(np.clip(rotate2, lo, hi))
    rotate2_canonical = -rotate2 if flip2 else rotate2
    return rotate2, rotate2_canonical


def _get_measurement(
    simulator: EnvironmentSimulator,
    x: float,
    y: float,
    look_yaw: float,
    cfg: Config,
) -> Tuple[float, float]:
    """
    Return (dist_mm, physical_iid) for the given position/look direction.

    The two flags are independent:
      override_emulator_distance — geometric distance (min over central cone), emulator IID
      override_emulator_iid      — geometric IID (10·log10(d_left/d_right)), emulator distance
      both True                  — both from geometry
      both False                 — both from emulator
    """
    need_profile = cfg.override_emulator_distance or cfg.override_emulator_iid
    need_emulator = (not cfg.override_emulator_distance) or (not cfg.override_emulator_iid)

    profile = simulator.get_profile_at_position(x, y, look_yaw) if need_profile else None
    meas    = simulator.emulator.predict_single(profile) if (need_profile and need_emulator) \
              else (simulator.get_sonar_measurement(x, y, look_yaw) if not need_profile else None)

    if cfg.override_emulator_distance:
        half_opening = simulator.opening_angle / 2
        edges = np.linspace(-half_opening, half_opening, simulator.profile_steps + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        central = profile[np.abs(centers) <= cfg.override_half_angle_deg]
        valid_central = central[~np.isnan(central)]
        geo_dist = float(np.min(valid_central)) if len(valid_central) > 0 else cfg.max_dist_mm
        dist_mm = max(cfg.min_dist_mm, min(geo_dist, cfg.max_dist_mm))
    else:
        dist_mm = max(cfg.min_dist_mm, min(float(meas.get("distance_mm", cfg.max_dist_mm)), cfg.max_dist_mm))

    if cfg.override_emulator_iid:
        half_opening = simulator.opening_angle / 2
        edges = np.linspace(-half_opening, half_opening, simulator.profile_steps + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        half = cfg.override_half_angle_deg
        # IID > 0 means wall closer on right (CCW-positive: centers>0 = left, centers<0 = right)
        left  = profile[(centers > 0) & (np.abs(centers) <= half)]
        right = profile[(centers < 0) & (np.abs(centers) <= half)]
        #d_left  = float(np.nanmin(left))  if np.any(~np.isnan(left))  else cfg.max_dist_mm
        #d_right = float(np.nanmin(right)) if np.any(~np.isnan(right)) else cfg.max_dist_mm
        d_left = float(np.nanmean(left)) if np.any(~np.isnan(left)) else cfg.max_dist_mm
        d_right = float(np.nanmean(right)) if np.any(~np.isnan(right)) else cfg.max_dist_mm
        physical_iid = 20.0 * float(np.log10(max(d_left, 1.0) / max(d_right, 1.0)))
    else:
        physical_iid = float(meas.get("iid_db", 0.0))

    return dist_mm, physical_iid


def run_episode(
    policy: MLPPolicy,
    simulator: EnvironmentSimulator,
    starts: List[Tuple[float, float, float]],
    cfg: Config,
    rng: np.random.Generator,
) -> Tuple[float, bool]:
    """
    Run one episode. Returns (fitness, collided).

    IID SYMMETRY: the network always operates in the canonical 'wall-on-right'
    frame (IID ≥ 0).  When the physical IID is negative (wall on left), we flip
    the IID sign fed into the network and negate the output rotation so the robot
    still turns the correct physical direction.  History stores canonical values.
    last_physical_iid drives the flip for rotate1 (decided before the measurement).
    """
    if not starts:
        return 0.0, False

    x, y, yaw = starts[int(rng.integers(len(starts)))]

    history: collections.deque = collections.deque(
        [(0.0, 0.0, 0.0, 0.0)] * cfg.history_len, maxlen=cfg.history_len
    )
    last_physical_iid = 0.0   # no prior measurement; no flip on first rotate1

    positions: List[Tuple[float, float]] = [(float(x), float(y))]
    net_turns: List[float] = []
    collided = False

    for _ in range(cfg.max_steps):
        original_yaw = yaw

        # ── Step 1: decide look direction (canonical frame) ───────────────────
        if cfg.force_aligned:
            rotate1_canonical = 0.0
            rotate1           = 0.0
        else:
            inp1 = build_input(history, 0.0, 0.0, 0.0, cfg)
            rotate1_canonical = policy.forward(inp1, cfg.max_rotate1_deg)
            flip1 = last_physical_iid < 0.0
            rotate1 = -rotate1_canonical if flip1 else rotate1_canonical
        look_yaw = original_yaw + rotate1

        # ── Step 2: sonar measurement at look direction ───────────────────────
        dist_mm, physical_iid = _get_measurement(simulator, x, y, look_yaw, cfg)
        if cfg.iid_noise_db > 0.0:
            physical_iid += float(rng.normal(0.0, cfg.iid_noise_db))

        # ── Step 3: decide body turn (canonical frame) ────────────────────────
        flip2         = physical_iid < 0.0
        canonical_iid = abs(physical_iid)
        inp2 = build_input(history, dist_mm, canonical_iid, rotate1_canonical, cfg)
        rotate2_canonical = policy.forward(inp2, cfg.max_rotate2_deg)
        rotate2 = -rotate2_canonical if flip2 else rotate2_canonical
        rotate2, rotate2_canonical = _apply_net_rotation_clamp(rotate1, rotate2, flip2, cfg)

        # ── Step 4: execute movement ──────────────────────────────────────────
        action = {"rotate1_deg": rotate1, "rotate2_deg": rotate2, "drive_mm": cfg.fixed_drive_mm}
        result = simulator.simulate_robot_movement(
            x, y, original_yaw, [action], compute_sonar=False
        )[0]

        x   = float(result["position"]["x"])
        y   = float(result["position"]["y"])
        yaw = float(result["orientation"])
        blocked = bool(result["collision"]["drive_blocked"])

        positions.append((x, y))
        net_turns.append(rotate1 + rotate2)
        # Store canonical values so history is always in the positive-IID frame.
        history.append((dist_mm, canonical_iid, rotate1_canonical, rotate2_canonical))
        last_physical_iid = physical_iid

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
    """
    Evaluate one genome across all training sessions.
    Returns (mean_fitness, collision_rate).

    crash_starts_list: per-session list of crash start positions (parallel to simulators).
    Each crash start is run exactly once (guaranteed); remaining eps_per_session slots
    are filled with randomly sampled starts from the normal pool.
    """
    policy = MLPPolicy(cfg)
    policy.set_genome(genome)

    eps_per_session = max(1, cfg.episodes_per_policy // len(simulators))
    fitnesses: List[float] = []
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


# ══════════════════════════════════════════════════════════════════════════════════
# Parallel evaluation
# ══════════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════════
# GA
# ══════════════════════════════════════════════════════════════════════════════════

def next_generation(
    population: List[np.ndarray],
    fitnesses: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    sorted_idx = np.argsort(fitnesses)[::-1]
    elites = [population[i] for i in sorted_idx[:cfg.elitism_count]]
    new_pop = [e.copy() for e in elites]

    while len(new_pop) < cfg.population_size:
        if len(elites) >= 2 and rng.random() < cfg.crossover_prob:
            i, j = rng.choice(len(elites), size=2, replace=False)
            mask  = rng.random(len(elites[0])) < 0.5
            child = np.where(mask, elites[i], elites[j]).copy()
        else:
            child = elites[int(rng.integers(len(elites)))].copy()

        # Gaussian mutation
        mask = rng.random(len(child)) < cfg.mutation_rate
        child[mask] += rng.normal(0.0, cfg.mutation_sigma, int(mask.sum())).astype(np.float32)
        new_pop.append(child)

    return new_pop[:cfg.population_size]


# ══════════════════════════════════════════════════════════════════════════════════
# IO
# ══════════════════════════════════════════════════════════════════════════════════

def save_policy(policy: MLPPolicy, fitness: float, generation: int, path: str) -> None:
    data = {
        "history_len":        policy.cfg.history_len,
        "hidden_sizes":       list(policy.cfg.hidden_sizes),
        "include_r1_in_input": policy.cfg.include_r1_in_input,
        "force_aligned":      policy.cfg.force_aligned,
        "max_rotate1_deg":    policy.cfg.max_rotate1_deg,
        "max_rotate2_deg":    policy.cfg.max_rotate2_deg,
        "max_net_rotation_deg": policy.cfg.max_net_rotation_deg,
        "fixed_drive_mm":     policy.cfg.fixed_drive_mm,
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
    """Save the full training history to JSON for later analysis and plotting."""
    # Replace nan/inf with None for JSON compatibility
    def _clean(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    clean = {k: [_clean(v) for v in vals] for k, vals in hist.items()}
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(clean, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════════
# Black box
# ══════════════════════════════════════════════════════════════════════════════════

_BB_CSS = """
body  { font-family: monospace; font-size: 13px; margin: 24px; color: #222; }
h1    { font-size: 15px; margin-bottom: 4px; }
h2    { font-size: 13px; margin: 20px 0 4px; color: #444; border-top: 1px solid #ddd; padding-top: 8px; }
h3    { font-size: 13px; margin: 12px 0 4px; }
p.note { font-size: 11px; color: #888; margin: 2px 0 10px; }
img   { max-width: 100%; border: 1px solid #ddd; margin-bottom: 14px; display: block; }
table { border-collapse: collapse; margin-bottom: 16px; }
th, td { border: 1px solid #ccc; padding: 3px 10px; text-align: right; white-space: nowrap; }
th    { background: #f0f0f0; text-align: center; }
td.c  { text-align: center; }
tr.crash td { background: #ffe4e4; font-weight: bold; }
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
    """Render a list of step dicts as an HTML table. Last row is the crash step."""
    header = (
        "<tr>"
        "<th>step</th>"
        "<th>x (mm)</th><th>y (mm)</th><th>yaw (&deg;)</th>"
        "<th>emu dist (mm)</th><th>emu IID (dB)</th>"
        "<th>geo dist (mm)</th><th>geo IID (dB)</th>"
        "<th>rot1 (&deg;)</th><th>rot2 (&deg;)</th><th>net (&deg;)</th>"
        "</tr>"
    )
    rows = []
    for i, s in enumerate(steps):
        cls = ' class="crash"' if i == len(steps) - 1 else ""
        rows.append(
            f'<tr{cls}>'
            f'<td class="c">{s["step"]}</td>'
            f'<td>{s["x_mm"]:.1f}</td><td>{s["y_mm"]:.1f}</td><td>{s["yaw_deg"]:+.1f}</td>'
            f'<td>{s["emu_dist_mm"]:.1f}</td><td>{s["emu_iid_db"]:+.2f}</td>'
            f'<td>{s["geo_dist_mm"]:.1f}</td><td>{s["geo_iid_db"]:+.2f}</td>'
            f'<td>{s["rotate1_deg"]:+.1f}</td><td>{s["rotate2_deg"]:+.1f}</td>'
            f'<td>{s["net_rot_deg"]:+.1f}</td>'
            "</tr>"
        )
    return f'<table>{header}{"".join(rows)}</table>'


def _write_blackbox_html(
    crashes_by_session: Dict[str, List[Dict]],
    generation: int,
    blackbox_dir: str,
) -> None:
    """Write blackbox/blackbox_gen{gen:04d}.html for one generation."""
    img_src = f"../trajectories_gen{generation:04d}.png"
    sections = []
    for session_name, crashed_trials in crashes_by_session.items():
        trial_blocks = []
        for entry in crashed_trials:
            t = entry["trial"]
            total = entry["total_steps"]
            steps = entry["last_steps"]
            shown = len(steps)
            trial_blocks.append(
                f'<h3>T{t} &#x2717; &mdash; crashed at step {total - 1} / '
                f'{total} &nbsp;(showing last {shown} steps)</h3>'
                + _bb_step_table(steps)
            )
        sections.append(
            f'<h2>{session_name}</h2>' + "".join(trial_blocks)
        )

    html = (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<title>Black box — Gen {generation}</title>'
        f'<style>{_BB_CSS}</style></head><body>'
        f'<h1>Generation {generation} — crash log</h1>'
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
    """
    Read _index.json (summary accumulated across gens) and rewrite index.html.
    """
    index_json = os.path.join(blackbox_dir, "_index.json")
    if not os.path.exists(index_json):
        return
    with open(index_json) as f:
        entries = json.load(f)   # list of {gen, sessions: {name: [trial_ids]}}

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
        '<table>'
        '<tr><th>gen</th><th>crashes</th><th>sessions / trials</th></tr>'
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
    """
    For every crashed episode in this generation's trajectory plot, write
    blackbox_dir/blackbox_gen{gen:04d}.html and update the index.
    Returns True if any crashes were found (and the file was written).
    """
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

    # Update the persistent index summary.
    index_json = os.path.join(blackbox_dir, "_index.json")
    entries = []
    if os.path.exists(index_json):
        with open(index_json) as f:
            entries = json.load(f)
    # Remove any existing entry for this generation (e.g. on resume).
    entries = [e for e in entries if e["gen"] != generation]
    entries.append({
        "gen": generation,
        "sessions": {
            sn: [c["trial"] for c in cs]
            for sn, cs in crashes_by_session.items()
        },
    })
    with open(index_json, "w") as f:
        json.dump(entries, f)

    _regenerate_blackbox_index(blackbox_dir)
    return True


# ══════════════════════════════════════════════════════════════════════════════════
# Hall of fame
# ══════════════════════════════════════════════════════════════════════════════════

# Each entry: (fitness, generation, genome)
HofEntry = Tuple[float, int, np.ndarray]


def hof_try_insert(
    hof: List[HofEntry],
    fitness: float,
    generation: int,
    genome: np.ndarray,
    n_best: int,
) -> bool:
    """
    Attempt to insert genome into the hall of fame.
    Skips exact duplicates (np.array_equal). Returns True if the HOF changed.
    """
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
    """Rewrite all HOF files (rank001.json … rankNNN.json) to hof_dir."""
    for rank, (fitness, generation, genome) in enumerate(hof, 1):
        pol = MLPPolicy(cfg)
        pol.set_genome(genome)
        data = {
            "rank":            rank,
            "history_len":     pol.cfg.history_len,
            "hidden_sizes":    list(pol.cfg.hidden_sizes),
            "max_rotate1_deg":     pol.cfg.max_rotate1_deg,
            "max_rotate2_deg":     pol.cfg.max_rotate2_deg,
            "max_net_rotation_deg": pol.cfg.max_net_rotation_deg,
            "fixed_drive_mm":      pol.cfg.fixed_drive_mm,
            "max_dist_mm":     pol.cfg.max_dist_mm,
            "max_iid_db":      pol.cfg.max_iid_db,
            "genome_size":     pol.genome_size(),
            "genome":          pol.get_genome().tolist(),
            "fitness":         float(fitness),
            "generation":      int(generation),
        }
        with open(os.path.join(hof_dir, f"rank{rank:03d}.json"), "w") as f:
            json.dump(data, f, indent=2)


def _fresh_population(cfg: Config, rng: np.random.Generator) -> List[np.ndarray]:
    genome_size = MLPPolicy(cfg).genome_size()
    return [(rng.standard_normal(genome_size) * 0.1).astype(np.float32)
            for _ in range(cfg.population_size)]


def _load_hof_entries(hof_dir: str) -> List[HofEntry]:
    """Reconstruct HOF list from saved rank*.json files."""
    hof = []
    for path in sorted(glob.glob(os.path.join(hof_dir, "rank*.json"))):
        with open(path) as f:
            d = json.load(f)
        hof.append((float(d["fitness"]), int(d["generation"]),
                    np.array(d["genome"], dtype=np.float32)))
    hof.sort(key=lambda e: e[0], reverse=True)
    return hof


# ══════════════════════════════════════════════════════════════════════════════════
# Trajectory plotting
# ══════════════════════════════════════════════════════════════════════════════════

def record_trajectories(
    policy: MLPPolicy,
    simulator: EnvironmentSimulator,
    starts: List[Tuple[float, float, float]],
    cfg: Config,
    n_episodes: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """Run n_episodes and return trajectory info (positions + collided flag + per-step black-box data)."""
    # Config for geometric reference measurements (both overrides on), used for black-box logging only.
    geo_cfg = dataclasses.replace(cfg, override_emulator_distance=True, override_emulator_iid=True)

    trajectories = []
    for _ in range(n_episodes):
        if not starts:
            break
        x, y, yaw = starts[int(rng.integers(len(starts)))]
        start_pos = (float(x), float(y), float(yaw))
        history: collections.deque = collections.deque(
            [(0.0, 0.0, 0.0, 0.0)] * cfg.history_len, maxlen=cfg.history_len
        )
        last_physical_iid = 0.0
        positions  = [(float(x), float(y))]
        body_yaws  = [float(yaw)]   # body heading after each step (index parallel to positions)
        look_yaws  = []             # head direction (yaw after rotate1) recorded per step
        steps_data: List[Dict] = []
        collided = False

        for step_idx in range(cfg.max_steps):
            original_yaw = yaw
            if cfg.force_aligned:
                rotate1_canonical = 0.0
                rotate1           = 0.0
            else:
                inp1 = build_input(history, 0.0, 0.0, 0.0, cfg)
                rotate1_canonical = policy.forward(inp1, cfg.max_rotate1_deg)
                flip1 = last_physical_iid < 0.0
                rotate1 = -rotate1_canonical if flip1 else rotate1_canonical
            look_yaw = original_yaw + rotate1
            look_yaws.append(look_yaw)

            dist_mm, physical_iid = _get_measurement(simulator, x, y, look_yaw, cfg)
            emu_dist = dist_mm          # emulator/override distance fed to the network
            emu_iid  = physical_iid     # emulator/override IID before noise
            if cfg.iid_noise_db > 0.0:
                physical_iid += float(rng.normal(0.0, cfg.iid_noise_db))

            # Geometric reference (always computed from geometry, regardless of cfg overrides).
            geo_dist, geo_iid = _get_measurement(simulator, x, y, look_yaw, geo_cfg)

            flip2         = physical_iid < 0.0
            canonical_iid = abs(physical_iid)
            inp2 = build_input(history, dist_mm, canonical_iid, rotate1_canonical, cfg)
            rotate2_canonical = policy.forward(inp2, cfg.max_rotate2_deg)
            rotate2 = -rotate2_canonical if flip2 else rotate2_canonical
            rotate2, rotate2_canonical = _apply_net_rotation_clamp(rotate1, rotate2, flip2, cfg)
            step_x, step_y = x, y   # position at start of step (where measurement was taken)
            action = {"rotate1_deg": rotate1, "rotate2_deg": rotate2, "drive_mm": cfg.fixed_drive_mm}
            result = simulator.simulate_robot_movement(x, y, original_yaw, [action], compute_sonar=False)[0]
            x   = float(result["position"]["x"])
            y   = float(result["position"]["y"])
            yaw = float(result["orientation"])
            positions.append((x, y))
            body_yaws.append(float(yaw))
            history.append((dist_mm, canonical_iid, rotate1_canonical, rotate2_canonical))
            last_physical_iid = physical_iid

            steps_data.append({
                "step":           step_idx,
                "x_mm":           round(step_x, 1),
                "y_mm":           round(step_y, 1),
                "yaw_deg":        round(original_yaw, 1),
                "emu_dist_mm":    round(emu_dist, 1),
                "emu_iid_db":     round(emu_iid, 2),
                "geo_dist_mm":    round(geo_dist, 1),
                "geo_iid_db":     round(geo_iid, 2),
                "rotate1_deg":    round(rotate1, 1),
                "rotate2_deg":    round(rotate2, 1),
                "net_rot_deg":    round(rotate1 + rotate2, 1),
            })

            if result["collision"]["drive_blocked"]:
                collided = True
                break

        trajectories.append({
            "positions":  positions,
            "body_yaws":  body_yaws,
            "look_yaws":  look_yaws,
            "collided":   collided,
            "steps":      steps_data,
            "start":      start_pos,
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
    """
    Multi-panel trajectory plot — one panel per session.
    trajs_by_session: {session_name: (trajectories, simulator)}
    Quiver arrows show the head direction (orientation after rotate1) every
    cfg.head_arrow_every_n_steps steps, coloured to match the path.
    """
    n = len(trajs_by_session)
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle(f"Gen {generation}  |  best fitness = {fitness:.1f}", fontsize=11)

    colours = plt.cm.tab10(np.linspace(0, 1, N_TRAJECTORY_EPISODES))
    arrow_len = cfg.head_arrow_length_mm
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

            # Path line
            ax.plot(xs, ys, color=colour, linewidth=0.8,
                    linestyle="--" if traj["collided"] else "-", zorder=2)
            ax.plot(xs[0],  ys[0],  "o", color=colour, markersize=4, zorder=3)
            ax.plot(xs[-1], ys[-1], "x", color=colour, markersize=5, zorder=3)

            # Head-direction arrows every N steps
            if arrow_every > 0:
                look_yaws = traj.get("look_yaws", [])
                for step, look_yaw in enumerate(look_yaws):
                    if step % arrow_every != 0:
                        continue
                    px, py = xs[step], ys[step]
                    rad = np.deg2rad(look_yaw)
                    ax.quiver(
                        px, py,
                        np.cos(rad) * arrow_len, np.sin(rad) * arrow_len,
                        angles="xy", scale_units="xy", scale=1,
                        color=colour, alpha=0.7, width=0.003,
                        headwidth=4, headlength=4, zorder=4,
                    )

        # Crash-pool start positions
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

        # Trial legend: colour + crash marker so the black-box file is easy to cross-reference.
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


# ══════════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════════

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

    # ── Resume or fresh start ─────────────────────────────────────────────────
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
                # treat lists and tuples as equivalent (JSON serialises tuples as lists)
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
            json.dump(asdict(cfg), f, indent=2)
        CodeLogger.log_code(cfg.output_dir, [".", "Library"], label="policy")

    template = MLPPolicy(cfg)
    print(f"Input dim:   {template.in_dim}")
    print(f"Genome size: {template.genome_size()}")
    print(f"Output dir:  {cfg.output_dir}")

    # Always build train_sims/train_starts — needed for crash-start retirement
    # even in parallel_eval mode.
    train_sims   = [build_simulator(sn, cfg.quiet_setup) for sn in cfg.train_session_names]
    train_starts = [load_starts(sn, cfg) for sn in cfg.train_session_names]

    # Validation simulator
    val_sim    = build_simulator(cfg.validation_session_name, cfg.quiet_setup) \
                 if cfg.validation_session_name else None
    val_starts = load_starts(cfg.validation_session_name, cfg) \
                 if cfg.validation_session_name else []

    # Simulators for trajectory plotting — all sessions (training + validation)
    all_plot_sessions = list(cfg.train_session_names) + (
        [cfg.validation_session_name] if cfg.validation_session_name else []
    )
    plot_sims_by_session: Dict[str, Tuple[EnvironmentSimulator, List]] = {}
    for sn in all_plot_sessions:
        sim = build_simulator(sn, quiet=True)
        starts = load_starts(sn, cfg, quiet=True)
        if starts:
            plot_sims_by_session[sn] = (sim, starts)

    executor = None
    if cfg.parallel_eval:
        n_workers = cfg.num_workers or os.cpu_count()
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(asdict(cfg),),
        )

    try:
        for gen in tqdm(range(start_gen, cfg.generations), desc="GA"):

            # ── Evaluate population ───────────────────────────────────────────
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
            gen_coll   = float(coll_rates[best_idx])   # collision rate of the best genome
            gen_mean_coll = float(np.mean(coll_rates)) # mean collision rate across population

            # ── Save overall best ─────────────────────────────────────────────
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_genome  = population[best_idx].copy()
                pol = MLPPolicy(cfg)
                pol.set_genome(best_genome)
                save_policy(pol, best_fitness, gen,
                            os.path.join(cfg.output_dir, "best_policy.json"))

            # ── Hall of fame ──────────────────────────────────────────────────
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

            # ── Optionally save every generation's best ───────────────────────
            if cfg.save_all_generation_policies:
                pol = MLPPolicy(cfg)
                pol.set_genome(population[best_idx].copy())
                save_policy(pol, gen_best, gen,
                            os.path.join(cfg.output_dir, f"gen_{gen:04d}_policy.json"))

            # ── Validation ────────────────────────────────────────────────────
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
                val_fit = float(np.mean(val_fits))
                val_std = float(np.std(val_fits))

            # ── Trajectory plot ───────────────────────────────────────────────
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

                # ── Update crash-start pool from this generation's crashes ────
                pool_changed = False
                for sn, (trajectories, _) in trajs_by_session.items():
                    for traj in trajectories:
                        if traj["collided"]:
                            positions  = traj["positions"]
                            body_yaws  = traj.get("body_yaws", [])
                            if not body_yaws:
                                continue
                            # Step back K positions from the crash (capped by trajectory length).
                            k   = min(cfg.crash_backtrack_steps, len(positions) - 1)
                            idx = -(k + 1)
                            bx, by = positions[idx]
                            byaw   = body_yaws[idx]
                            pool = crash_starts_by_session.setdefault(sn, [])
                            pool.append((bx, by, byaw))
                            # Keep only the most recent entries within the cap.
                            if len(pool) > cfg.max_crash_starts_per_session:
                                crash_starts_by_session[sn] = pool[-cfg.max_crash_starts_per_session:]
                            pool_changed = True
                # ── Retire pool entries the best genome now handles ───────────
                if crash_starts_by_session and best_genome is not None:
                    retire_pol = MLPPolicy(cfg)
                    retire_pol.set_genome(best_genome)
                    pool_changed_retire = False
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
                            pool_changed_retire = True

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

            traj_str = f"  traj_coll={traj_coll_rate:.2f}" if not np.isnan(traj_coll_rate) else ""
            n_crash_pool = sum(len(v) for v in crash_starts_by_session.values())
            crash_str = f"  crash_pool={n_crash_pool}" if n_crash_pool > 0 else ""
            tqdm.write(
                f"Gen {gen:4d} | best={gen_best:7.1f}  mean={gen_mean:7.1f}"
                f"  coll={gen_coll:.2f}{traj_str}{crash_str}"
            )

            # ── Evolve ────────────────────────────────────────────────────────
            population = next_generation(population, fitnesses, cfg, rng)

            # ── Checkpoint ────────────────────────────────────────────────────
            np.savez(checkpoint_path,
                     population=np.stack(population),
                     generation=np.array(gen))

            # ── Pushover ──────────────────────────────────────────────────────
            if cfg.pushover_every_n > 0 and (gen + 1) % cfg.pushover_every_n == 0:
                pushover_notify(
                    f"h{cfg.history_len:02d} Gen {gen+1}/{cfg.generations}  best={gen_best:.0f}  coll={gen_coll:.2f}"
                )

            save_plot(hist, cfg.output_dir)
            save_history(hist, cfg.output_dir)

    finally:
        if executor:
            executor.shutdown(wait=False)

    print(f"\nDone. Best fitness: {best_fitness:.1f}")
    print(f"Results saved to:  {cfg.output_dir}")
    pushover_notify(f"Done h{cfg.history_len:02d}. Best fitness: {best_fitness:.1f}")


def main() -> None:
    # history_len=0 → baseline (head fixed to body, responds only to current dist+iid)
    # history_len>0 → standard config with that history length
    for history_len in HISTORY_LENGTHS:
        if history_len == 0:
            cfg = Config(
                history_len=0,
                include_r1_in_input=False,
                force_aligned=True,
                output_dir=f"PolicyTraining/{CONDITION}_h00",
                iid_noise_db=IID_NOISE_DB,
            )
            label = "baseline"
        else:
            cfg = Config(
                history_len=history_len,
                output_dir=f"PolicyTraining/{CONDITION}_h{history_len:02d}",
                iid_noise_db=IID_NOISE_DB,
            )
            label = f"history_len={history_len}"
        print(f"\n{'='*60}")
        print(f"Training {label}  →  {cfg.output_dir}")
        print(f"{'='*60}")
        train(cfg)


if __name__ == "__main__":
    main()
