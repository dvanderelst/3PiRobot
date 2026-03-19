#!/usr/bin/env python3
"""
GA training script with history-based NN steering.

v2 — Symmetry via IID sign wrapper (replaces episode-level mirroring)
======================================================================
The previous approach (SCRIPT_TrainPolicy.py) negated IID and actions inside
each mirrored episode to simulate a reflected world.  That turned out to be
ineffective: because the robot is a wall-follower it settles near a wall in
both normal and mirrored conditions, so in steady state it always sees positive
stored IID.  The mirrored episodes contributed negative IID only briefly during
start-up transients, giving very uneven coverage.

v2 solution: the policy itself enforces bilateral symmetry via an IID sign wrapper
baked into decide_rotate1 / decide_rotate2.  See HistoryNNPolicy docstring for
details.  Episode-level mirroring is removed entirely.
"""
import collections
import json
import os
import random
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor
import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Library.EnvironmentSimulator import EnvironmentSimulator
from Library import CodeLogger


# ── Pushover helper ─────────────────────────────────────────────────────────────

try:
    from Library.PushOver import send as _pushover_send
    _PUSHOVER_AVAILABLE = True
except Exception:
    _PUSHOVER_AVAILABLE = False

def pushover_notify(message: str, title: str = "3PiRobot training") -> None:
    """Send a Pushover notification via Library.PushOver.  Silently skips on failure."""
    if not _PUSHOVER_AVAILABLE:
        return
    try:
        _pushover_send(f"[{title}] {message}")
    except Exception as e:
        print(f"[Pushover] notification failed: {e}")

# ── Experiment condition ────────────────────────────────────────────────────────
# Set CONDITION to a short label for this run; results go to Policy/<CONDITION>/.
# If that folder already exists and is non-empty, the script will ask via a
# dialog whether to overwrite it or abort.
CONDITION   = "memory10"   # subfolder under Policy
DESCRIPTION = "Progressive steps 15→100 over 20 gens; geometric distance emulator, IID-only NN, 90° profile."

# ── Pushover notifications ───────────────────────────────────────────────────────
PUSHOVER_EVERY_N  = 10    # send a notification every N generations (0 = disable mid-run)
# ─────────────────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    history_len: int = 5
    seed: int = 42
    session_name: str = "sessionB01"
    train_session_names: Optional[List[str]] = field(
        default_factory=lambda: ["sessionB01", "sessionB02", "sessionB03", "sessionB05"]
    )
    validation_session_name: Optional[str] = "sessionB04"

    # ── Fitness ──────────────────────────────────────────────────────────────────
    # fitness = mean(step_reward) * collision_discount
    #
    # step_reward = max(0, 1.0 - w_turn(clearance) * (sinuosity - 1.0))
    #
    # sinuosity   = path_length / straight_line_distance over a rolling window
    #               capped at 2.0 (1.0 = perfectly straight, 2.0 = very tortuous)
    #               window=3 is short enough that zigzag turns cannot cancel each other out
    #
    # w_turn(clearance) = w_turn_max * clip((clearance - free_turn_distance_mm) / (open_space_distance_mm - free_turn_distance_mm), 0, 1)
    #               → 0.0 at or below free_turn_distance_mm  (turns completely free — robot can make emergency turns)
    #               → w_turn_max at open_space_distance_mm   (orbiting/circling heavily penalised)
    #
    # collision_discount = collision_fitness_scale if collided else 1.0
    open_space_distance_mm:  float = 1000.0  # clearance at which sinuosity penalty reaches its maximum (w_turn_max)
    free_turn_distance_mm:   float = 300.0   # clearance below which turns are completely free (emergency turning zone)
    w_turn_max:              float = 1.25     # sinuosity penalty weight in open space
    sinuosity_window:        int   = 15      # short rolling window — prevents zigzag cancellation exploit
    collision_distance_mm:   float = 150.0  # hard collision threshold → episode ends
    collision_fitness_scale: float = 0.2    # fitness multiplier applied when episode ends in collision

    # Action limits
    max_rotate1_deg: float = 45.0
    max_rotate2_deg: float = 45.0
    fixed_drive_mm: float = 100.0
    hidden_sizes: Tuple[int, int] = (16, 16)

    # GA
    population_size: int = 75
    generations: int = 150
    elitism_count: int = 10   # keep top-n
    mutation_rate: float = 0.05   # ~40/890 weights perturbed per offspring; was 0.2
                                   # (160 changes destroyed parent behaviour)
    mutation_sigma: float = 0.2

    # Evaluation
    episodes_per_policy: int = 16
    max_steps: int = 200
    # Start positions are loaded from pre-computed JSON files produced by
    # SCRIPT_ComputeValidStarts.py.  Set starts_suffix to select which file:
    #   "starts_headon"    — robot faces a nearby wall (hardest)
    #   "starts_wall_left" / "starts_wall_right" — wall to one side
    #   "valid_starts"     — full arena, any heading
    # Set starts_suffix_secondary + starts_mix_ratio to blend two pools.
    # e.g. 50% headon + 50% valid_starts ensures urgent wall cases are always
    # represented while also exposing the robot to open-space situations.
    starts_dir: str                      = "ValidStarts"
    starts_suffix: str                   = "starts_headon"
    starts_suffix_secondary: Optional[str] = "valid_starts"  # None to disable mixing
    starts_mix_ratio: float              = 0.8   # fraction sampled from primary (starts_suffix)
    validation_episodes_per_generation: int = 16

    # Progressive difficulty
    use_progressive_steps: bool = False
    progressive_steps_start: int = 15
    progressive_steps_end: int = 250
    progressive_steps_generations: int = 30

    # IO
    output_dir: str = f"Policy/{CONDITION}"
    description: str = DESCRIPTION
    quiet_setup: bool = True
    save_generation_best_plots: bool = True
    parallel_eval: bool = True
    num_workers: Optional[int] = 12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(f):
        return default
    return f


def build_simulator(session_name: str, quiet_setup: bool = True) -> EnvironmentSimulator:
    if quiet_setup:
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                return EnvironmentSimulator(session_name)
            except Exception:
                return EnvironmentSimulator("_default_")
    try:
        return EnvironmentSimulator(session_name)
    except Exception:
        return EnvironmentSimulator("_default_")


class HistoryNNPolicy:
    """
    Two-head MLP with shared history encoder.

    BILATERAL SYMMETRY — CANONICAL FRAME DESIGN
    ============================================
    The network is trained exclusively in a canonical "wall-on-right" frame:
    IID values presented to the network are ALWAYS non-negative (wall on right
    or in deadband).  When the physical IID is negative (wall on left), both
    decide_rotate1 and decide_rotate2 flip the IID sign before running the
    network and negate the output rotation.  This gives correct physical
    behaviour on both wall sides without training on negative IID examples.

    IMPORTANT FOR DEPLOYMENT AND FUTURE READERS
    ============================================
    Do NOT pre-flip the IID before calling these methods.  Always pass the raw
    physical IID (can be negative).  The wrapper inside each method handles the
    flip transparently.  Removing or bypassing this wrapper will break bilateral
    symmetry.

    History features stored per step (always in canonical frame):
        [canonical_iid_norm, dist_norm, rot1_canonical_norm, rot2_canonical_norm,
         drive_norm, echo_present_prob]
    where canonical_iid = abs(physical_iid) and canonical_rotX = physical_rotX
    reflected back to the positive-IID frame.  echo_present_prob is symmetric
    (unchanged by flip) and passed directly.
    """

    def __init__(
        self,
        max_rotate1_deg: float,
        max_rotate2_deg: float,
        history_len: int,
        hidden_sizes: Tuple[int, int],
    ):
        self.max_rotate1_deg = float(max_rotate1_deg)
        self.max_rotate2_deg = float(max_rotate2_deg)
        self.history_len = int(history_len)
        self.hidden_sizes = tuple(int(v) for v in hidden_sizes)
        self.feature_dim = 6
        self.in_dim = self.history_len * self.feature_dim
        h1, h2 = self.hidden_sizes
        self.shapes = [
            (h1, self.in_dim), (h1,),   # shared encoder:       W1, b1
            (h2, h1),          (h2,),   # rot1 head hidden:     W2a, b2a
            (1,  h2),          (1,),    # rot1 head output:     W3a, b3a
            (h2, h1 + 3),      (h2,),   # rot2 head hidden:     W2b, b2b  (+3 = iid_n, dist_n, echo_n)
            (1,  h2),          (1,),    # rot2 head output:     W3b, b3b
        ]
        self.params: List[np.ndarray] = [np.zeros(s, dtype=np.float32) for s in self.shapes]

    def genome_size(self) -> int:
        return int(sum(int(np.prod(s)) for s in self.shapes))

    def set_genome(self, genome: np.ndarray) -> None:
        g = np.asarray(genome, dtype=np.float32).reshape(-1)
        if g.size != self.genome_size():
            raise ValueError(f"HistoryNNPolicy genome must have length {self.genome_size()}")
        out: List[np.ndarray] = []
        off = 0
        for s in self.shapes:
            n = int(np.prod(s))
            out.append(g[off:off + n].reshape(s))
            off += n
        self.params = out

    def get_genome(self) -> np.ndarray:
        return np.concatenate([p.reshape(-1) for p in self.params]).astype(np.float32)

    def _shared_h1(self, hist_vec: np.ndarray) -> np.ndarray:
        x = np.asarray(hist_vec, dtype=np.float32).reshape(self.in_dim, 1)
        w1, b1 = self.params[0], self.params[1]
        return np.tanh(w1 @ x + b1.reshape(-1, 1))

    def decide_rotate1(self, hist_vec: np.ndarray, last_iid_db: float) -> float:
        """Head 1: decide where to look (before measuring).

        SYMMETRY WRAPPER: last_iid_db is the raw physical IID from the previous
        step.  If it was negative (wall on left last step), we assume the wall is
        still on the left, run the network in the canonical positive-IID frame,
        and negate the output so the head turns toward the correct physical side.
        Pass the raw physical IID — do NOT pre-flip.
        """
        flip = safe_float(last_iid_db, 0.0) < 0.0    # wall was on left last step
        h1 = self._shared_h1(hist_vec)
        w2a, b2a, w3a, b3a = self.params[2], self.params[3], self.params[4], self.params[5]
        h2 = np.tanh(w2a @ h1 + b2a.reshape(-1, 1))
        y = np.tanh(w3a @ h2 + b3a.reshape(-1, 1))
        rotate1_canonical = float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate1_deg
        return -rotate1_canonical if flip else rotate1_canonical

    def decide_rotate2(self, hist_vec: np.ndarray, current_iid_db: float, current_dist_mm: float,
                       echo_present_prob: float = 1.0) -> float:
        """Head 2: decide body turn after looking (current measurement injected).

        SYMMETRY WRAPPER: current_iid_db is the raw physical IID just measured.
        If negative (wall on left), we reflect to canonical positive-IID frame,
        run the network, and negate the output so the body turns the correct
        physical direction.  Pass the raw physical IID — do NOT pre-flip.
        echo_present_prob is symmetric and injected directly without flip.
        """
        phys = safe_float(current_iid_db, 0.0)
        flip = phys < 0.0                             # wall is on left this step
        canonical_iid = abs(phys)
        h1 = self._shared_h1(hist_vec)
        # Network sees canonical (non-negative) IID — wall always appears on right.
        iid_n   = float(np.clip(canonical_iid / 12.0, 0.0, 2.0))
        dist_n  = float(np.clip(safe_float(current_dist_mm, 1800.0) / 2000.0, 0.0, 2.0))
        echo_n  = float(np.clip(safe_float(echo_present_prob, 1.0), 0.0, 1.0))
        h1_aug = np.concatenate([h1, np.array([[iid_n], [dist_n], [echo_n]], dtype=np.float32)], axis=0)
        w2b, b2b, w3b, b3b = self.params[6], self.params[7], self.params[8], self.params[9]
        h2 = np.tanh(w2b @ h1_aug + b2b.reshape(-1, 1))
        y = np.tanh(w3b @ h2 + b3b.reshape(-1, 1))
        rotate2_canonical = float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate2_deg
        return -rotate2_canonical if flip else rotate2_canonical

    def decide_rotate1_with_h1(self, h1: np.ndarray, last_iid_db: float) -> float:
        """Head 1 using pre-computed shared h1 (avoids recomputing for decide_rotate2)."""
        flip = safe_float(last_iid_db, 0.0) < 0.0
        w2a, b2a, w3a, b3a = self.params[2], self.params[3], self.params[4], self.params[5]
        h2 = np.tanh(w2a @ h1 + b2a.reshape(-1, 1))
        y  = np.tanh(w3a @ h2 + b3a.reshape(-1, 1))
        rotate1_canonical = float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate1_deg
        return -rotate1_canonical if flip else rotate1_canonical

    def decide_rotate2_with_h1(self, h1: np.ndarray, current_iid_db: float,
                                current_dist_mm: float, echo_present_prob: float = 1.0) -> float:
        """Head 2 using pre-computed shared h1 (avoids recomputing from hist_vec)."""
        phys = safe_float(current_iid_db, 0.0)
        flip = phys < 0.0
        canonical_iid = abs(phys)
        iid_n  = float(np.clip(canonical_iid / 12.0, 0.0, 2.0))
        dist_n = float(np.clip(safe_float(current_dist_mm, 1800.0) / 2000.0, 0.0, 2.0))
        echo_n = float(np.clip(safe_float(echo_present_prob, 1.0), 0.0, 1.0))
        h1_aug = np.concatenate([h1, np.array([[iid_n], [dist_n], [echo_n]], dtype=np.float32)], axis=0)
        w2b, b2b, w3b, b3b = self.params[6], self.params[7], self.params[8], self.params[9]
        h2 = np.tanh(w2b @ h1_aug + b2b.reshape(-1, 1))
        y  = np.tanh(w3b @ h2 + b3b.reshape(-1, 1))
        rotate2_canonical = float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate2_deg
        return -rotate2_canonical if flip else rotate2_canonical


def config_from_dict(d: Dict[str, Any]) -> Config:
    known = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in known})


class Evaluator:
    def __init__(self, simulator: EnvironmentSimulator, cfg: Config):
        self.sim = simulator
        self.cfg = cfg
        self.starts = self._load_starts()
        self.starts_secondary = self._load_secondary_starts()
        walls = getattr(self.sim.arena, "walls", np.array([], dtype=np.float32))
        self._wall_points = np.asarray(walls, dtype=np.float32) if walls is not None else np.array([], dtype=np.float32)

        # Precompute arena bounds once — avoids repeated meta dict lookups in the hot path.
        meta = getattr(self.sim.arena, "meta", {}) or {}
        b = meta.get("arena_bounds_mm", None)
        if isinstance(b, dict):
            self._bmin_x = safe_float(b.get("min_x"), 0.0)
            self._bmax_x = safe_float(b.get("max_x"), self.sim.arena.arena_width)
            self._bmin_y = safe_float(b.get("min_y"), 0.0)
            self._bmax_y = safe_float(b.get("max_y"), self.sim.arena.arena_height)
        else:
            self._bmin_x = 0.0
            self._bmax_x = float(self.sim.arena.arena_width)
            self._bmin_y = 0.0
            self._bmax_y = float(self.sim.arena.arena_height)

    def _load_starts(self) -> List[Tuple[float, float, float]]:
        """Load pre-computed (x, y, yaw) starts from ValidStarts JSON.

        Primary pool: cfg.starts_suffix.  Optional secondary pool: cfg.starts_suffix_secondary.
        sample_start() blends the two pools according to cfg.starts_mix_ratio.
        """
        session = getattr(self.sim.arena, "session_name", None)
        if not session:
            return []
        return self._load_starts_file(session, self.cfg.starts_suffix)

    def _load_starts_file(self, session: str, suffix: str) -> List[Tuple[float, float, float]]:
        jp = os.path.join(self.cfg.starts_dir, f"{session}_{suffix}.json")
        if not os.path.isfile(jp):
            print(f"  ⚠ Starts file not found: {jp}")
            return []
        try:
            with open(jp) as f:
                data = json.load(f)
            starts = [
                (float(s["x"]), float(s["y"]), float(s["yaw_deg"]))
                for s in data.get("starts", [])
            ]
            print(f"  Loaded {len(starts)} starts for {session} from {jp}")
            return starts
        except Exception as e:
            print(f"  ⚠ Could not load starts for {session} ({suffix}): {e}")
            return []

    def _load_secondary_starts(self) -> List[Tuple[float, float, float]]:
        if not self.cfg.starts_suffix_secondary:
            return []
        session = getattr(self.sim.arena, "session_name", None)
        if not session:
            return []
        return self._load_starts_file(session, self.cfg.starts_suffix_secondary)

    def sample_start(self, rng: random.Random) -> Tuple[float, float, float]:
        if not self.starts:
            raise RuntimeError(
                f"No starts loaded for session — run SCRIPT_ComputeValidStarts.py "
                f"and check starts_dir/starts_suffix in Config."
            )
        if self.starts_secondary and rng.random() > self.cfg.starts_mix_ratio:
            return self.starts_secondary[rng.randrange(len(self.starts_secondary))]
        return self.starts[rng.randrange(len(self.starts))]

    def _geometry_clearance_mm(self, x: float, y: float) -> float:
        boundary_clearance = max(0.0, float(min(
            x - self._bmin_x, self._bmax_x - x,
            y - self._bmin_y, self._bmax_y - y,
        )))
        if self._wall_points.size == 0:
            return boundary_clearance
        dx = self._wall_points[:, 0] - float(x)
        dy = self._wall_points[:, 1] - float(y)
        return min(boundary_clearance, float(np.min(np.hypot(dx, dy))))

    def episode(self, policy: HistoryNNPolicy, start: Tuple[float, float, float]) -> Dict[str, Any]:
        """Run one episode.

        The policy's IID sign wrapper (see HistoryNNPolicy docstring) handles
        bilateral symmetry transparently.  We pass raw physical IID to the policy
        at every step and store CANONICAL values in the history ring buffer so that
        the network always sees the positive-IID frame.

        Canonical frame convention
        --------------------------
        canonical_iid  = abs(physical_iid)        (always >= 0)
        canonical_rotX = physical_rotX if physical_iid >= 0 else -physical_rotX
        (i.e. the action the network would have produced without the flip)
        """
        x, y, yaw = start
        start_x, start_y = x, y
        collided = False
        end_reason = "max_steps_reached"
        total_drive = 0.0
        step_rewards: List[float] = []
        proximity_terms: List[float] = []
        aligned_terms: List[float] = []
        sign_match_terms: List[float] = []
        trajectory: List[Dict[str, Any]] = []
        position_history: collections.deque = collections.deque(maxlen=self.cfg.sinuosity_window)
        fdim = policy.feature_dim
        hl   = self.cfg.history_len
        # Ring buffer: avoids per-step deque→list conversion and np.concatenate.
        # hist_buf[row] = one feature vector; hist_flat is the flat view used as hist_vec.
        # Pre-fill with random plausible values so the network cannot use the
        # "empty history = episode start = near wall" cue.
        if hl > 0:
            hist_buf = np.empty((hl, fdim), dtype=np.float32)
            hist_buf[:, 0] = np.random.uniform(0.0,  1.0, size=hl)   # canonical_iid_norm
            hist_buf[:, 1] = np.random.uniform(0.2,  0.9, size=hl)   # dist_norm
            hist_buf[:, 2] = np.random.uniform(-0.5, 0.5, size=hl)   # canonical_rot1_norm
            hist_buf[:, 3] = np.random.uniform(-0.5, 0.5, size=hl)   # canonical_rot2_norm
            hist_buf[:, 4] = np.random.uniform(0.8,  1.0, size=hl)   # prev_drive_norm
            hist_buf[:, 5] = 1.0                                       # echo_present_prob
            hist_flat = hist_buf.reshape(-1)   # flat view into hist_buf; live-updated in place
        else:
            hist_buf = None
            hist_flat = np.zeros(0, dtype=np.float32)

        # last_physical_iid: raw measured IID from previous step.
        # Used by decide_rotate1 to determine flip direction before the current measurement.
        # Randomly signed to match the random pre-fill — avoids always starting with flip=False.
        last_physical_iid = float(np.random.uniform(0.0, 1.0)) * 12.0 * np.random.choice([-1.0, 1.0])
        prev_drive_norm   = 1.0

        for t in range(self.cfg.max_steps):
            # hist_flat is a live view of hist_buf — already the correct flattened vector.
            h1      = policy._shared_h1(hist_flat)
            rotate1 = policy.decide_rotate1_with_h1(h1, last_physical_iid)

            # --- Execute rotate1, then measure at the new look direction ---
            meas    = self.sim.get_sonar_measurement(x, y, yaw + rotate1)
            physical_iid      = safe_float(meas.get("iid_db"),           0.0)
            dist_mm           = safe_float(meas.get("distance_mm"),      1800.0)
            echo_present_prob = safe_float(meas.get("echo_present_prob"), 1.0)

            # --- Head 2: decide body turn (h1 reused from above — no second encoder pass) ---
            rotate2 = policy.decide_rotate2_with_h1(h1, physical_iid, dist_mm, echo_present_prob)

            # --- Compute canonical values for history storage ---
            # canonical frame: abs IID, actions reflected back to positive-IID frame.
            flip           = physical_iid < 0.0
            canonical_iid  = abs(physical_iid)
            canonical_rot1 = -rotate1 if flip else rotate1
            canonical_rot2 = -rotate2 if flip else rotate2
            canonical_iid_norm  = float(np.clip(canonical_iid  / 12.0,                    0.0, 2.0))
            dist_norm            = float(np.clip(dist_mm        / 2000.0,                  0.0, 2.0))
            canonical_rot1_norm  = float(np.clip(canonical_rot1 / self.cfg.max_rotate1_deg, -1.0, 1.0))
            canonical_rot2_norm  = float(np.clip(canonical_rot2 / self.cfg.max_rotate2_deg, -1.0, 1.0))

            if hl > 0:
                hist_buf[:-1] = hist_buf[1:]
                hist_buf[-1]  = (canonical_iid_norm, dist_norm, canonical_rot1_norm,
                                 canonical_rot2_norm, prev_drive_norm, echo_present_prob)
            last_physical_iid = physical_iid   # carry raw IID to next step for rotate1 flip

            # --- Execute rotate2 + drive ---
            action = {"rotate1_deg": rotate1, "rotate2_deg": rotate2, "drive_mm": self.cfg.fixed_drive_mm}
            step = self.sim.simulate_robot_movement(x, y, yaw, [action], compute_sonar=False)[0]
            nx       = safe_float(step["position"]["x"],  x)
            ny       = safe_float(step["position"]["y"],  y)
            nyaw     = safe_float(step["orientation"],    yaw)
            move     = step.get("movement", {})
            exec_drive = safe_float(move.get("executed_drive_mm"), np.hypot(nx - x, ny - y))
            total_drive += exec_drive

            net_turn_deg = rotate1 + rotate2
            prev_drive_norm = float(np.clip(exec_drive / max(self.cfg.fixed_drive_mm, 1e-6), 0.0, 1.5))
            clearance_mm    = self._geometry_clearance_mm(nx, ny)
            open_space_mm   = max(float(self.cfg.open_space_distance_mm), 1e-6)
            proximity_term  = float(np.clip((open_space_mm - clearance_mm) / open_space_mm, 0.0, 1.0))
            proximity_terms.append(proximity_term)

            # Diagnostic metrics (not part of fitness).
            iid_norm_phys = float(np.clip(physical_iid / 12.0, -2.0, 2.0))
            align_term = -iid_norm_phys * float(np.clip(
                net_turn_deg / max(self.cfg.max_rotate1_deg + self.cfg.max_rotate2_deg, 1e-6), -1.0, 1.0))
            aligned_terms.append(align_term)
            if abs(iid_norm_phys) > 0.15 and abs(net_turn_deg) > 2.0:
                sign_match_terms.append(1.0 if np.sign(net_turn_deg) == -np.sign(iid_norm_phys) else 0.0)

            # Per-step fitness: sinuosity penalised proportionally to wall clearance.
            # Below free_turn_distance_mm: w_turn = 0 (emergency turns completely free); ramps to w_turn_max at open_space_distance_mm.
            position_history.append((nx, ny))
            if len(position_history) >= 2:
                path_dist  = sum(
                    np.hypot(position_history[i][0] - position_history[i-1][0],
                             position_history[i][1] - position_history[i-1][1])
                    for i in range(1, len(position_history))
                )
                fx, fy     = position_history[0]
                lx, ly     = position_history[-1]
                straight   = float(np.hypot(lx - fx, ly - fy))
                sinuosity  = min(path_dist / straight if straight > 1e-6 else 2.0, 2.0)
            else:
                sinuosity  = 1.0
            free_mm     = float(self.cfg.free_turn_distance_mm)
            w_turn      = self.cfg.w_turn_max * float(np.clip((clearance_mm - free_mm) / max(open_space_mm - free_mm, 1e-6), 0.0, 1.0))
            step_reward = max(0.0, 1.0 - w_turn * (sinuosity - 1.0))
            step_rewards.append(step_reward)

            trajectory.append({
                "step": t,
                "x": nx, "y": ny, "yaw_deg": nyaw,
                "look_yaw_deg": yaw + rotate1,
                "iid_db": physical_iid,
                "distance_mm": dist_mm,
                "echo_present_prob": echo_present_prob,
                "rotate1_deg": rotate1,
                "rotate2_deg": rotate2,
                "executed_drive_mm": exec_drive,
                "clearance_mm": clearance_mm,
                "proximity_term": proximity_term,
                "sinuosity": sinuosity,
                "step_reward": step_reward,
            })

            x, y, yaw = nx, ny, nyaw
            if clearance_mm <= float(self.cfg.collision_distance_mm):
                end_reason = "collision_distance_reached"
                collided = True
                break

        net_displacement   = float(np.hypot(x - start_x, y - start_y))
        sign_match_rate    = float(np.mean(sign_match_terms)) if sign_match_terms else 0.5
        collision_discount = self.cfg.collision_fitness_scale if collided else 1.0
        normalized_fitness = float(np.mean(step_rewards)) * collision_discount if step_rewards else 0.0

        return {
            "fitness":                  normalized_fitness,
            "collision_discount":       collision_discount,
            "steps":                    len(trajectory),
            "total_executed_drive_mm":  float(total_drive),
            "net_displacement_mm":      net_displacement,
            "collided":                 collided,
            "end_reason":               end_reason,
            "proximity_mean":           float(np.mean(proximity_terms) if proximity_terms else 0.0),
            "alignment_mean":           float(np.mean(aligned_terms)   if aligned_terms   else 0.0),
            "sign_match_rate":          sign_match_rate,
            "trajectory":               trajectory,
        }

    def evaluate(self, policy: HistoryNNPolicy,
                 starts: List[Tuple[float, float, float]],
                 n_left: int = 0) -> Dict[str, Any]:
        """
        Run all episodes and return aggregated metrics.

        n_left > 0: the first n_left starts are wall-left episodes,
        the remainder are wall-right.  Fitness is a plain mean across all episodes.
        """
        eps       = [self.episode(policy, s) for s in starts]
        fit_array = np.array([e["fitness"] for e in eps], dtype=np.float32)

        if n_left > 0 and n_left < len(eps):
            left_fit  = float(np.mean(fit_array[:n_left]))
            right_fit = float(np.mean(fit_array[n_left:]))
        else:
            left_fit  = float("nan")
            right_fit = float("nan")
        fitness = float(np.mean(fit_array))

        return {
            "fitness":                  fitness,
            "fitness_std":              float(np.std(fit_array)),
            "left_fit":                 left_fit,
            "right_fit":                right_fit,
            "n_left":                   n_left,
            "collision_rate":           float(np.mean([1.0 if e["collided"] else 0.0 for e in eps])),
            "proximity_mean":           float(np.mean([e.get("proximity_mean", 0.0)  for e in eps])),
            "alignment_mean":           float(np.mean([e["alignment_mean"]            for e in eps])),
            "sign_match_rate":          float(np.mean([e["sign_match_rate"]           for e in eps])),
            "avg_drive_mm":             float(np.mean([e["total_executed_drive_mm"]   for e in eps])),
            "avg_net_displacement_mm":  float(np.mean([e["net_displacement_mm"]       for e in eps])),
            "episodes_raw":             eps,
        }


# ── Parallel worker ────────────────────────────────────────────────────────────

_WORKER_CFG: Optional[Config] = None
_WORKER_EVS: Optional[List[Evaluator]] = None


def _init_worker(cfg_dict: Dict[str, Any]) -> None:
    global _WORKER_CFG, _WORKER_EVS
    _WORKER_CFG = config_from_dict(cfg_dict)
    train_sessions = list(_WORKER_CFG.train_session_names) if _WORKER_CFG.train_session_names else [_WORKER_CFG.session_name]
    train_sessions = [s for s in train_sessions if isinstance(s, str) and s.strip()]
    if not train_sessions:
        train_sessions = [_WORKER_CFG.session_name]
    _WORKER_EVS = [Evaluator(build_simulator(sn, quiet_setup=_WORKER_CFG.quiet_setup), _WORKER_CFG)
                   for sn in train_sessions]
    import torch as _torch          # torch already loaded via Emulator above; set thread count last
    _torch.set_num_threads(1)


def _eval_genome_worker(genome: np.ndarray, starts_by_env: List[List[Tuple[float, float, float]]],
                        n_lefts_by_env: Optional[List[int]] = None,
                        max_steps: Optional[int] = None) -> Dict[str, Any]:
    if _WORKER_CFG is None or _WORKER_EVS is None:
        raise RuntimeError("Worker not initialized")
    if max_steps is not None:
        _WORKER_CFG.max_steps = max_steps
    pol = HistoryNNPolicy(
        _WORKER_CFG.max_rotate1_deg, _WORKER_CFG.max_rotate2_deg,
        _WORKER_CFG.history_len, _WORKER_CFG.hidden_sizes,
    )
    pol.set_genome(genome)
    n_lefts = n_lefts_by_env or [0] * len(_WORKER_EVS)
    per_env = [ev.evaluate(pol, starts, n_left)
               for ev, starts, n_left in zip(_WORKER_EVS, starts_by_env, n_lefts)]

    def mean_key(k: str) -> float:
        vals = [safe_float(r.get(k, float("nan")), float("nan")) for r in per_env]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    episode_fitness = mean_key("fitness")
    vis_idx = random.randrange(len(per_env)) if per_env else 0

    return {
        "fitness":                 episode_fitness,
        "fitness_std":             mean_key("fitness_std"),
        "left_fit":                mean_key("left_fit"),
        "right_fit":               mean_key("right_fit"),
        "n_left":                  per_env[0].get("n_left", 0) if per_env else 0,
        "collision_rate":          mean_key("collision_rate"),
        "proximity_mean":          mean_key("proximity_mean"),
        "alignment_mean":          mean_key("alignment_mean"),
        "sign_match_rate":         mean_key("sign_match_rate"),
        "avg_drive_mm":            mean_key("avg_drive_mm"),
        "avg_net_displacement_mm": mean_key("avg_net_displacement_mm"),
        "episodes_raw":            per_env[vis_idx].get("episodes_raw", []) if per_env else [],
        "vis_env_idx":             vis_idx,
        "per_env_fitness":         [safe_float(r.get("fitness", float("nan")), float("nan")) for r in per_env],
    }


# ── GA Trainer ─────────────────────────────────────────────────────────────────

class SimpleGATrainer:
    def __init__(self, evaluators_train: List[Evaluator], cfg: Config,
                 evaluator_validation: Optional[Evaluator] = None):
        if not evaluators_train:
            raise ValueError("Need at least one training evaluator")
        self.evs_train       = evaluators_train
        self.ev              = evaluators_train[0]
        self.ev_validation   = evaluator_validation
        self.cfg             = cfg
        self.sim             = self.ev.sim
        self.generation_plot_dir = os.path.join(self.cfg.output_dir, "generation_best")
        self.best_genome: Optional[np.ndarray] = None
        self.best_fitness = -float("inf")
        self.best_train_collision_rate: float = float("nan")
        self.best_val_fitness: float = float("nan")
        self.best_val_coll: float = float("nan")
        # Full-population history — one entry per completed generation.
        self.pop_genomes_history:  List[np.ndarray] = []   # (n_gen, pop_size, genome_size)
        self.pop_fitness_history:  List[np.ndarray] = []   # (n_gen, pop_size)
        self.history: Dict[str, List[float]] = {
            "best_fitness": [], "avg_fitness": [],
            "best_left_fit": [], "best_right_fit": [],
            "best_alignment_mean": [], "best_sign_match_rate": [],
            "best_collision_rate": [], "best_proximity_mean": [],
            "best_avg_net_displacement_mm": [],
            "val_best_fitness": [], "val_collision_rate": [],
            "val_avg_net_displacement_mm": [],
        }

    def _make_policy(self, genome: np.ndarray) -> HistoryNNPolicy:
        p = HistoryNNPolicy(
            self.cfg.max_rotate1_deg, self.cfg.max_rotate2_deg,
            self.cfg.history_len, self.cfg.hidden_sizes,
        )
        p.set_genome(genome)
        return p

    def init_population(self) -> List[np.ndarray]:
        gsize = HistoryNNPolicy(
            self.cfg.max_rotate1_deg, self.cfg.max_rotate2_deg,
            self.cfg.history_len, self.cfg.hidden_sizes,
        ).genome_size()
        start_policy_path = os.path.join(self.cfg.output_dir, "start_policy.json")
        if os.path.exists(start_policy_path):
            try:
                with open(start_policy_path) as f:
                    start_data = json.load(f)
                start_genome = np.array(start_data.get("genome", []), dtype=np.float32)
                if start_genome.size == gsize:
                    print(f"Initializing population from {start_policy_path}")
                    pop: List[np.ndarray] = [start_genome.copy()]
                    for _ in range(self.cfg.population_size - 1):
                        variation = np.random.normal(0.0, 0.1, size=gsize).astype(np.float32)
                        pop.append(np.clip(start_genome + variation, -3.0, 3.0))
                    return pop
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Could not load start_policy.json: {e}; using random init.")
        return [np.random.normal(0.0, 0.25, size=gsize).astype(np.float32)
                for _ in range(self.cfg.population_size)]

    def mutate(self, g: np.ndarray) -> np.ndarray:
        mask  = (np.random.rand(*g.shape) < self.cfg.mutation_rate).astype(np.float32)
        noise = np.random.normal(0.0, self.cfg.mutation_sigma, size=g.shape).astype(np.float32)
        return np.clip(g + mask * noise, -3.0, 3.0).astype(np.float32)

    def _generation_starts(
        self, gen: int
    ) -> Tuple[List[List[Tuple[float, float, float]]], List[int]]:
        """Return (starts_by_env, n_lefts_by_env).

        Left starts come first within each env's list so that evaluate() can
        split on n_left.  Do NOT shuffle — order must be preserved.
        """
        starts_all: List[List[Tuple[float, float, float]]] = []
        n_lefts_all: List[int] = []
        for env_i, ev in enumerate(self.evs_train):
            rng = random.Random(self.cfg.seed + 10000 * (gen + 1) + 1000 * env_i)
            n   = self.cfg.episodes_per_policy
            starts = [ev.sample_start(rng) for _ in range(n)]
            n_lefts_all.append(0)

            starts_all.append(starts)
        return starts_all, n_lefts_all

    def _evaluate_on_train_envs(self, genome: np.ndarray,
                                 starts_by_env: List[List[Tuple[float, float, float]]],
                                 n_lefts_by_env: Optional[List[int]] = None) -> Dict[str, Any]:
        pol     = self._make_policy(genome)
        n_lefts = n_lefts_by_env or [0] * len(self.evs_train)
        per_env = [ev.evaluate(pol, starts, n_left)
                   for ev, starts, n_left in zip(self.evs_train, starts_by_env, n_lefts)]

        def mean_key(k: str) -> float:
            vals = [safe_float(r.get(k, float("nan")), float("nan")) for r in per_env]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        episode_fitness = mean_key("fitness")
        vis_idx = random.randrange(len(per_env)) if per_env else 0

        return {
            "fitness":                 episode_fitness,
            "fitness_std":             mean_key("fitness_std"),
            "left_fit":                mean_key("left_fit"),
            "right_fit":               mean_key("right_fit"),
            "n_left":                  per_env[0].get("n_left", 0) if per_env else 0,
            "collision_rate":          mean_key("collision_rate"),
            "proximity_mean":          mean_key("proximity_mean"),
            "alignment_mean":          mean_key("alignment_mean"),
            "sign_match_rate":         mean_key("sign_match_rate"),
            "avg_drive_mm":            mean_key("avg_drive_mm"),
            "avg_net_displacement_mm": mean_key("avg_net_displacement_mm"),
            "episodes_raw":            per_env[vis_idx].get("episodes_raw", []),
            "vis_env_idx":             vis_idx,
            "per_env_fitness":         [safe_float(r.get("fitness", float("nan")), float("nan")) for r in per_env],
        }

    def train(self) -> np.ndarray:
        pop = self.init_population()

        # Pre-compute a fixed validation start set once so val_fitness is
        # comparable across generations (not re-sampled each time).
        val_starts_fixed: List[Tuple[float, float, float]] = []
        if self.ev_validation is not None:
            rng_val_init = random.Random(self.cfg.seed + 777)
            n_val = max(1, int(self.cfg.validation_episodes_per_generation))
            val_starts_fixed = [self.ev_validation.sample_start(rng_val_init)
                                 for _ in range(n_val)]

        # ── Create parallel pool once (reused across all generations) ───────────
        _pool: Optional[ProcessPoolExecutor] = None
        if self.cfg.parallel_eval and len(pop) > 1:
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            _workers  = self.cfg.num_workers or max(1, min(os.cpu_count() or 1, 8))
            _cfg_dict = asdict(self.cfg)
            try:
                _pool = ProcessPoolExecutor(
                    max_workers=_workers, initializer=_init_worker, initargs=(_cfg_dict,)
                )
                print(f"Parallel pool created with {_workers} workers.")
            except Exception as e:
                print(f"Could not create parallel pool ({type(e).__name__}: {e}); using serial.")

        try:
          for gen in range(self.cfg.generations):
            if self.cfg.use_progressive_steps and self.cfg.progressive_steps_generations > 0:
                progress = min(gen / self.cfg.progressive_steps_generations, 1.0)
                self.cfg.max_steps = int(
                    self.cfg.progressive_steps_start
                    + progress * (self.cfg.progressive_steps_end - self.cfg.progressive_steps_start)
                )
                print(f"Generation {gen+1}: max_steps = {self.cfg.max_steps} (progressive)")

            starts_by_env, n_lefts_by_env = self._generation_starts(gen)
            fitness: List[float]          = [0.0] * len(pop)
            details: List[Dict[str, Any]] = [None] * len(pop)  # type: ignore

            if _pool is not None:
                try:
                    cur_max_steps = self.cfg.max_steps
                    results = _pool.map(
                        _eval_genome_worker, pop,
                        [starts_by_env] * len(pop),
                        [n_lefts_by_env] * len(pop),
                        [cur_max_steps] * len(pop),
                    )
                    for i, res in enumerate(tqdm(results, total=len(pop),
                                                 desc=f"Gen {gen+1}/{self.cfg.generations}")):
                        fitness[i] = float(res["fitness"])
                        details[i] = res
                except Exception as e:
                    print(f"Parallel eval failed ({type(e).__name__}: {e}); using serial.")
                    _pool.shutdown(wait=False)
                    _pool = None

            if _pool is None:
                for i, g in enumerate(tqdm(pop, desc=f"Gen {gen+1}/{self.cfg.generations}")):
                    if details[i] is not None:
                        continue
                    res        = self._evaluate_on_train_envs(g, starts_by_env, n_lefts_by_env)
                    fitness[i] = float(res["fitness"])
                    details[i] = res

            f_np       = np.asarray(fitness, dtype=np.float32)
            order_asc  = np.argsort(f_np)
            best_idx   = int(order_asc[-1])
            median_idx = int(order_asc[len(order_asc) // 2])
            worst_idx  = int(order_asc[0])
            best_g     = pop[best_idx].copy()
            best_res   = details[best_idx]
            median_res = details[median_idx]
            worst_res  = details[worst_idx]

            self.history["best_fitness"].append(float(np.max(f_np)))
            self.history["avg_fitness"].append(float(np.mean(f_np)))
            self.history["best_left_fit"].append(safe_float(best_res.get("left_fit",  float("nan")), float("nan")))
            self.history["best_right_fit"].append(safe_float(best_res.get("right_fit", float("nan")), float("nan")))
            self.history["best_alignment_mean"].append(float(best_res["alignment_mean"]))
            self.history["best_sign_match_rate"].append(float(best_res["sign_match_rate"]))
            self.history["best_collision_rate"].append(float(best_res["collision_rate"]))
            self.history["best_proximity_mean"].append(float(best_res.get("proximity_mean", 0.0)))
            self.history["best_avg_net_displacement_mm"].append(float(best_res["avg_net_displacement_mm"]))

            eps = best_res.get("episodes_raw", [])
            if eps:
                ep_fits      = np.array([safe_float(ep.get("fitness"), float("-inf")) for ep in eps])
                best_ep_fit  = float(np.max(ep_fits))
                median_ep_fit = float(np.median(ep_fits))
                worst_ep_fit = float(np.min(ep_fits))
            else:
                best_ep_fit = median_ep_fit = worst_ep_fit = float("nan")

            val_fit = val_coll = val_net_disp = float("nan")
            val_res: Optional[Dict[str, Any]] = None
            if self.ev_validation is not None and val_starts_fixed:
                val_res = self.ev_validation.evaluate(self._make_policy(best_g), val_starts_fixed)
                val_fit   = float(val_res.get("fitness",                  float("nan")))
                val_coll  = float(val_res.get("collision_rate",           float("nan")))
                val_net_disp = float(val_res.get("avg_net_displacement_mm", float("nan")))
            self.history["val_best_fitness"].append(val_fit)
            self.history["val_collision_rate"].append(val_coll)
            self.history["val_avg_net_displacement_mm"].append(val_net_disp)

            if float(f_np[best_idx]) > self.best_fitness:
                self.best_fitness              = float(f_np[best_idx])
                self.best_genome               = best_g.copy()
                self.best_train_collision_rate = float(best_res.get("collision_rate", float("nan")))
                self.best_val_fitness          = val_fit
                self.best_val_coll             = val_coll

            best_left_fit  = safe_float(best_res.get("left_fit",  float("nan")), float("nan"))
            best_right_fit = safe_float(best_res.get("right_fit", float("nan")), float("nan"))
            side_str = (f"L={best_left_fit:.3f} R={best_right_fit:.3f}"
                        if np.isfinite(best_left_fit) else "no-split")
            print(
                f"Gen {gen+1}: best={self.history['best_fitness'][-1]:.3f} [{side_str}], "
                f"avg={self.history['avg_fitness'][-1]:.3f}, "
                f"best_ep={best_ep_fit:.3f}, med_ep={median_ep_fit:.3f}, worst_ep={worst_ep_fit:.3f}, "
                f"align={best_res['alignment_mean']:.3f}, "
                f"sign_match={best_res['sign_match_rate']:.3f}, "
                f"coll={best_res['collision_rate']:.3f}, "
                f"net_disp={best_res['avg_net_displacement_mm']:.1f}mm, "
                f"val_fit={val_fit:.3f}, val_coll={val_coll:.3f}"
            )
            self._save_population_history(
                np.stack([g.copy() for g in pop]),  # (pop_size, genome_size)
                f_np.copy(),                         # (pop_size,)
            )
            self._save_generation_best_plot(gen + 1, best_res, median_res, worst_res, val_res)
            self._save_generation_best_genome(gen + 1, best_g, best_res, val_fit, val_coll)
            self._save_live_policy_probe(best_g)
            self._save_live_best_policy(gen + 1)

            # ── Pushover: mid-run notification every PUSHOVER_EVERY_N generations ──
            is_last_gen = (gen == self.cfg.generations - 1)
            if PUSHOVER_EVERY_N > 0 and ((gen + 1) % PUSHOVER_EVERY_N == 0) and not is_last_gen:
                pushover_notify(
                    f"Gen {gen+1}/{self.cfg.generations} | {CONDITION}\n"
                    f"best={self.history['best_fitness'][-1]:.3f} [{side_str}]\n"
                    f"coll={best_res['collision_rate']:.3f}\n"
                    f"val_fit={val_fit:.3f}"
                )

            if gen < self.cfg.generations - 1:
                order   = np.argsort(f_np)[::-1]
                elite_n = min(self.cfg.elitism_count, len(pop))
                elites  = [pop[int(i)].copy() for i in order[:elite_n]]
                # Copy elites unchanged, then fill remaining slots by mutating
                # a rank-weighted randomly chosen elite.  Better elites are more
                # likely to be parents; avoids destructive crossover between
                # unrelated weight configurations (competing-conventions problem).
                nxt = elites[:]
                rank_weights = np.arange(elite_n, 0, -1, dtype=np.float64)
                rank_weights /= rank_weights.sum()
                while len(nxt) < self.cfg.population_size:
                    parent = elites[np.random.choice(elite_n, p=rank_weights)]
                    nxt.append(self.mutate(parent).astype(np.float32))
                pop = nxt

        finally:
            if _pool is not None:
                _pool.shutdown()

        if self.best_genome is None:
            raise RuntimeError("No best genome found")

        # ── Pushover: final notification ──────────────────────────────────────────
        pushover_notify(
            f"Training done! | {CONDITION}\n"
            f"{self.cfg.generations} gens, pop={self.cfg.population_size}\n"
            f"best_ever={self.best_fitness:.3f}\n"
            f"last gen: best={self.history['best_fitness'][-1]:.3f}  "
            f"coll={self.history['best_collision_rate'][-1]:.3f}",
            title=f"3Pi done: {CONDITION}",
        )

        return self.best_genome

    def _save_population_history(self, genomes: np.ndarray, fitnesses: np.ndarray) -> None:
        """Append this generation's full population and write atomically."""
        self.pop_genomes_history.append(genomes)
        self.pop_fitness_history.append(fitnesses)
        path     = os.path.join(self.cfg.output_dir, "population_history.npz")
        tmp_path = os.path.join(self.cfg.output_dir, "population_history.tmp.npz")
        np.savez(
            tmp_path,
            genomes=np.stack(self.pop_genomes_history),    # (n_gen, pop_size, genome_size)
            fitnesses=np.stack(self.pop_fitness_history),  # (n_gen, pop_size)
        )
        os.replace(tmp_path, path)

    def _save_generation_best_plot(self, generation_number: int, detail: Dict[str, Any],
                                    median_detail: Optional[Dict[str, Any]],
                                    worst_detail: Optional[Dict[str, Any]],
                                    validation_detail: Optional[Dict[str, Any]] = None) -> None:
        if not self.cfg.save_generation_best_plots:
            return

        def _pick_best(episodes):
            if not episodes:
                return None
            return int(np.argmax([safe_float(ep.get("fitness"), float("-inf")) for ep in episodes]))

        def _pick_crash_or_random(episodes):
            """Return (idx, had_crash). Picks worst-fitness crash; falls back to random."""
            if not episodes:
                return None, False
            crashes = [i for i, ep in enumerate(episodes) if ep.get("collided", False)]
            if crashes:
                idx = int(min(crashes, key=lambda i: safe_float(episodes[i].get("fitness"), float("inf"))))
                return idx, True
            return int(np.random.randint(len(episodes))), False

        def _draw(sim, episodes, idx, ax, label, show_legend=False):
            if idx is None or not episodes:
                ax.set_visible(False)
                return
            ep = episodes[idx]
            draw_episode_on_axis(sim, ep, ax,
                                 f"{label} | ep_fit={safe_float(ep.get('fitness'), float('nan')):.2f} | end={ep.get('end_reason', '?')}",
                                 show_legend=show_legend)

        episodes_best   = detail.get("episodes_raw", [])
        episodes_median = median_detail.get("episodes_raw", []) if median_detail else []
        episodes_worst  = worst_detail.get("episodes_raw",  []) if worst_detail  else []
        if not episodes_best:
            return

        best_idx             = _pick_best(episodes_best)
        crash_idx, had_crash = _pick_crash_or_random(episodes_best)
        median_ep_idx        = _pick_best(episodes_median)
        worst_ep_idx         = _pick_best(episodes_worst)

        os.makedirs(self.generation_plot_dir, exist_ok=True)
        train_fit = safe_float(detail.get("fitness"), float("nan"))
        has_val   = validation_detail is not None and self.ev_validation is not None

        n_rows = 2 if has_val else 1
        fig, axes = plt.subplots(n_rows, 4, figsize=(24, 6 * n_rows))
        axes = np.asarray(axes).reshape(n_rows, 4)

        def _env_sim(d: Optional[Dict[str, Any]]) -> Any:
            idx = d.get("vis_env_idx", 0) if d else 0
            idx = min(idx, len(self.evs_train) - 1)
            ev  = self.evs_train[idx]
            return ev.sim, getattr(ev.sim.arena, "session_name", f"env{idx}")

        sim_best,   sn_best   = _env_sim(detail)
        sim_median, sn_median = _env_sim(median_detail)
        sim_worst,  sn_worst  = _env_sim(worst_detail)

        _draw(sim_best,   episodes_best,   best_idx,      axes[0, 0], f"Best [{sn_best}]",                         show_legend=True)
        _draw(sim_best,   episodes_best,   crash_idx,     axes[0, 1], f"{'Crash' if had_crash else 'Random'} [{sn_best}]")
        _draw(sim_median, episodes_median, median_ep_idx, axes[0, 2], f"Median [{sn_median}]")
        _draw(sim_worst,  episodes_worst,  worst_ep_idx,  axes[0, 3], f"Worst [{sn_worst}]")

        if has_val:
            episodes_val             = validation_detail.get("episodes_raw", [])
            val_fit                  = safe_float(validation_detail.get("fitness"), float("nan"))
            val_best_idx             = _pick_best(episodes_val)
            val_crash_idx, val_crash = _pick_crash_or_random(episodes_val)
            val_fits       = [safe_float(ep.get("fitness"), float("-inf")) for ep in episodes_val]
            val_order      = np.argsort(val_fits)
            val_median_idx = int(val_order[len(val_order) // 2]) if episodes_val else None
            val_worst_idx  = int(val_order[0])                   if episodes_val else None
            _draw(self.ev_validation.sim, episodes_val, val_best_idx,   axes[1, 0], "Val Best",                            show_legend=True)
            _draw(self.ev_validation.sim, episodes_val, val_crash_idx,  axes[1, 1], "Val Crash" if val_crash else "Val Random")
            _draw(self.ev_validation.sim, episodes_val, val_median_idx, axes[1, 2], "Val Median ep")
            _draw(self.ev_validation.sim, episodes_val, val_worst_idx,  axes[1, 3], "Val Worst ep")
            fig.suptitle(f"Gen {generation_number} | train_fit={train_fit:.2f} | val_fit={val_fit:.2f}", fontsize=14)
        else:
            fig.suptitle(f"Gen {generation_number} | train_fit={train_fit:.2f}", fontsize=14)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        fig.savefig(os.path.join(self.generation_plot_dir, f"gen_{generation_number:03d}_triplet.png"), dpi=180)
        plt.close(fig)
        self._save_live_metrics_plot()

    def _save_live_metrics_plot(self) -> None:
        os.makedirs(self.generation_plot_dir, exist_ok=True)
        g = np.arange(1, len(self.history["best_fitness"]) + 1)
        if g.size == 0:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax0 = axes[0]

        # Box plot showing the full population fitness distribution per generation.
        if self.pop_fitness_history:
            bp = ax0.boxplot(
                [f.tolist() for f in self.pop_fitness_history],
                positions=g,
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor="#e0e0e0", alpha=0.6),
                medianprops=dict(color="#555555", lw=1.5),
                whiskerprops=dict(color="#888888", lw=1),
                capprops=dict(color="#888888", lw=1),
                flierprops=dict(marker=".", color="#aaaaaa", alpha=0.4, markersize=3),
                manage_ticks=False,
            )
            # Add a proxy artist so the box appears in the legend.
            from matplotlib.patches import Patch
            ax0.legend(handles=[
                bp["boxes"][0],
            ], labels=["Population distribution"], loc="lower right")

        ax0.plot(g, self.history["best_fitness"], label="Best fitness", color="black", lw=2, marker="o", ms=3)
        ax0.plot(g, self.history["avg_fitness"],  label="Avg fitness",  color="gray",  lw=1, linestyle="--", marker="o", ms=3)
        lf = np.asarray(self.history["best_left_fit"],  dtype=np.float64)
        rf = np.asarray(self.history["best_right_fit"], dtype=np.float64)
        if np.isfinite(lf).any():
            ax0.plot(g, lf, label="Best L-wall fit", color="#2e7d32", lw=1.5, linestyle="-.", marker="o", ms=3)
        if np.isfinite(rf).any():
            ax0.plot(g, rf, label="Best R-wall fit", color="#c62828", lw=1.5, linestyle="-.", marker="o", ms=3)
        if np.isfinite(np.asarray(self.history["val_best_fitness"], dtype=np.float64)).any():
            ax0.plot(g, self.history["val_best_fitness"], label="Val fitness", color="steelblue", lw=1.5, marker="o", ms=3)
        ax0.set_xlabel("Generation"); ax0.set_ylabel("Fitness")
        ax0.set_title("Fitness Progress (Live)"); ax0.grid(True, alpha=0.3); ax0.legend()
        ax1 = axes[1]
        ax1.plot(g, self.history["best_alignment_mean"],  label="Alignment")
        ax1.plot(g, self.history["best_sign_match_rate"], label="Sign match")
        ax1.plot(g, self.history["best_collision_rate"],  label="Best collision rate")
        ax1.plot(g, self.history["best_proximity_mean"],  label="Proximity")
        if np.isfinite(np.asarray(self.history["val_collision_rate"], dtype=np.float64)).any():
            ax1.plot(g, self.history["val_collision_rate"], label="Val collision")
        ax1.set_xlabel("Generation"); ax1.set_ylabel("Metric")
        ax1.set_title("Behavior Metrics (Live)"); ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3); ax1.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.generation_plot_dir, "live_metrics.png"), dpi=180)
        plt.close(fig)

    def _save_generation_best_genome(self, generation_number: int, genome: np.ndarray,
                                      detail: Dict[str, Any], val_fit: float, val_coll: float) -> None:
        os.makedirs(self.generation_plot_dir, exist_ok=True)
        path = os.path.join(self.generation_plot_dir, f"gen_{generation_number:03d}_best_policy.json")
        with open(path, "w") as f:
            json.dump({
                "generation":           generation_number,
                "policy_type":          "HistoryNNPolicy_v2",
                "symmetry":             "iid_sign_wrapper",
                "genome":               genome.tolist(),
                "genome_size":          len(genome),
                "history_len":          self.cfg.history_len,
                "hidden_sizes":         list(self.cfg.hidden_sizes),
                "max_rotate1_deg":      self.cfg.max_rotate1_deg,
                "max_rotate2_deg":      self.cfg.max_rotate2_deg,
                "train_fitness":        float(detail.get("fitness",        float("nan"))),
                "train_collision_rate": float(detail.get("collision_rate", float("nan"))),
                "train_sign_match_rate":float(detail.get("sign_match_rate",float("nan"))),
                "val_fitness":          val_fit,
                "val_collision_rate":   val_coll,
            }, f, indent=2)

    def _save_live_policy_probe(self, genome: np.ndarray) -> None:
        os.makedirs(self.generation_plot_dir, exist_ok=True)
        plot_policy_curve(self._make_policy(genome), self.cfg, self.generation_plot_dir,
                          filename="live_policy_probe_best.png")

    def _save_live_best_policy(self, generation_number: int) -> None:
        """Atomically write the all-time best policy to the output root after each generation."""
        if self.best_genome is None:
            return
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        path     = os.path.join(self.cfg.output_dir, "best_policy.json")
        tmp_path = os.path.join(self.cfg.output_dir, "best_policy.tmp.json")
        with open(tmp_path, "w") as f:
            json.dump({
                "policy_type":          "HistoryNNPolicy_v2",
                "symmetry":             "iid_sign_wrapper",
                "condition":            CONDITION,
                "description":          DESCRIPTION,
                "best_at_generation":   generation_number,
                "genome":               self.best_genome.tolist(),
                "genome_size":          len(self.best_genome),
                "history_len":          self.cfg.history_len,
                "hidden_sizes":         list(self.cfg.hidden_sizes),
                "max_rotate1_deg":      self.cfg.max_rotate1_deg,
                "max_rotate2_deg":      self.cfg.max_rotate2_deg,
                "train_fitness":        self.best_fitness,
                "train_collision_rate": self.best_train_collision_rate,
                "val_fitness":          self.best_val_fitness,
                "val_collision_rate":   self.best_val_coll,
            }, f, indent=2)
        os.replace(tmp_path, path)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_training(history: Dict[str, List[float]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    g = np.arange(1, len(history["best_fitness"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(g, history["best_fitness"], label="Best fitness")
    ax.plot(g, history["avg_fitness"],  label="Avg fitness")
    if np.isfinite(np.asarray(history.get("val_best_fitness", [float("nan")]), dtype=np.float64)).any():
        ax.plot(g, history["val_best_fitness"], label="Val fitness")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness")
    ax.set_title("History-NN GA Training (v2)"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot_fitness.png"), dpi=180)
    plt.close(fig)


def plot_policy_curve(policy: HistoryNNPolicy, cfg: Config, output_dir: str,
                      filename: str = "plot_policy_curve_best.png") -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for iid_sign in [-1.0, 1.0]:
        for dist in [500.0, 1000.0, 1500.0]:
            iid_val      = iid_sign * 6.0
            iid_norm_val = float(np.clip(abs(iid_val) / 12.0, 0.0, 2.0))  # canonical
            dist_norm_val = float(np.clip(dist / 2000.0, 0.0, 2.0))
            fdim = policy.feature_dim
            hist_vec = np.zeros(cfg.history_len * fdim, dtype=np.float32)
            for step in range(cfg.history_len):
                hist_vec[step*fdim + 0] = iid_norm_val   # canonical_iid_norm
                hist_vec[step*fdim + 1] = dist_norm_val  # dist_norm
                hist_vec[step*fdim + 4] = 1.0            # prev_drive_norm
                hist_vec[step*fdim + 5] = 1.0            # echo_present_prob
            rotate1 = policy.decide_rotate1(hist_vec, iid_val)
            rotate2 = policy.decide_rotate2(hist_vec, iid_val, dist, echo_present_prob=1.0)
            total   = rotate1 + rotate2
            marker  = "o" if iid_sign > 0 else "s"
            color   = "C0" if iid_sign > 0 else "C1"
            ax1.scatter(iid_sign, total,  marker=marker, color=color, alpha=0.8, s=100)
            ax2.scatter(total,   -rotate2, marker=marker, color=color, alpha=0.8, s=100)
    for ax in (ax1, ax2):
        ax.axhline(0, color="black", linewidth=1, alpha=0.3)
        ax.axvline(0, color="black", linewidth=1, alpha=0.3)
        ax.grid(True, alpha=0.3)
    ax1.set_xlabel("Wall Direction"); ax1.set_ylabel("Total Rotation (deg)")
    ax1.set_title("Total Rotation Response"); ax1.set_xlim(-1.5, 1.5)
    ax1.set_xticks([-1, 1]); ax1.set_xticklabels(["Left Wall", "Right Wall"])
    ax1.scatter([], [], marker="o", color="C0", label="Right wall"); ax1.scatter([], [], marker="s", color="C1", label="Left wall")
    ax1.legend()
    ax2.set_xlabel("Total Rotation (deg)"); ax2.set_ylabel("−rotate2 (deg)")
    ax2.set_title("Look-Drive Difference vs Total Rotation"); ax2.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_episode_on_axis(sim: EnvironmentSimulator, episode: Dict[str, Any],
                          ax: Any, title: str, show_legend: bool = True) -> None:
    traj = episode.get("trajectory", [])
    if not traj:
        return
    from matplotlib.collections import LineCollection
    xs = [s["x"] for s in traj]
    ys = [s["y"] for s in traj]
    iids = [safe_float(s.get("iid_db"), 0.0) for s in traj]
    walls = getattr(sim.arena, "walls", None)
    if walls is not None and len(walls) > 0:
        ax.scatter(walls[:, 0], walls[:, 1], s=1, color="#9e9e9e", alpha=0.25, label="Walls")
    if len(xs) >= 2:
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="RdBu_r", norm=plt.Normalize(-12, 12), linewidth=2, zorder=2)
        lc.set_array(np.array(iids[:-1]))
        ax.add_collection(lc)
        if show_legend:
            cb = plt.colorbar(lc, ax=ax, shrink=0.7)
            cb.set_label("IID (dB)\nred=wall right, blue=wall left")
    else:
        ax.plot(xs, ys, "-", color="#1565c0", linewidth=2)
    ax.scatter(xs[0],  ys[0],  s=60, color="#2e7d32", label="Start")
    ax.scatter(xs[-1], ys[-1], s=60, marker="x", color="#c62828", label="End")
    STRIDE = 5; ALEN = 150.0
    idxs = range(0, len(traj), STRIDE)
    ax.quiver(
        [traj[i]["x"] for i in idxs], [traj[i]["y"] for i in idxs],
        [ALEN * np.cos(np.deg2rad(traj[i]["look_yaw_deg"])) for i in idxs],
        [ALEN * np.sin(np.deg2rad(traj[i]["look_yaw_deg"])) for i in idxs],
        units="xy", angles="xy", scale_units="xy", scale=1,
        color="#ff9800", alpha=0.7, width=8.0, headwidth=4, headlength=5, label="Look dir",
    )
    all_x = list(xs) + (walls[:, 0].tolist() if walls is not None and len(walls) > 0 else [])
    all_y = list(ys) + (walls[:, 1].tolist() if walls is not None and len(walls) > 0 else [])
    pad_x = max(30.0, 0.05 * max(1.0, float(np.max(all_x)) - float(np.min(all_x))))
    pad_y = max(30.0, 0.05 * max(1.0, float(np.max(all_y)) - float(np.min(all_y))))
    ax.set_xlim(float(np.min(all_x)) - pad_x, float(np.max(all_x)) + pad_x)
    ax.set_ylim(float(np.min(all_y)) - pad_y, float(np.max(all_y)) + pad_y)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    if "end=" not in title:
        title = f"{title} | end={episode.get('end_reason', '?')}"
    ax.set_title(title); ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="upper left")


def plot_episode(sim: EnvironmentSimulator, episode: Dict[str, Any],
                 output_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_episode_on_axis(sim, episode, ax, title, show_legend=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_overview(cfg: Config, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    probe = HistoryNNPolicy(cfg.max_rotate1_deg, cfg.max_rotate2_deg,
                            cfg.history_len, cfg.hidden_sizes)
    lines = [
        "# History-NN GA (v2 — IID sign wrapper)",
        "",
        "Symmetry approach: IID sign wrapper inside HistoryNNPolicy.",
        "Network operates exclusively in canonical positive-IID frame.",
        "No episode-level mirroring.",
        "",
        f"**Description:** {cfg.description}" if cfg.description else "",
        "",
        f"- history_len: {cfg.history_len}",
        f"- hidden_sizes: {cfg.hidden_sizes}",
        f"- genome_size: {probe.genome_size()}",
        f"- train sessions: {cfg.train_session_names}",
        f"- validation session: {cfg.validation_session_name}",
        "",
        "```json",
        json.dumps(asdict(cfg), indent=2),
        "```",
    ]
    with open(os.path.join(output_dir, "run_overview.md"), "w") as f:
        f.write("\n".join(lines))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config()
    train_sessions = list(cfg.train_session_names) if cfg.train_session_names else [cfg.session_name]
    train_sessions = [s for s in train_sessions if isinstance(s, str) and s.strip()]
    if not train_sessions:
        train_sessions = [cfg.session_name]
    cfg.session_name = train_sessions[0]
    if cfg.validation_session_name in train_sessions:
        train_sessions = [s for s in train_sessions if s != cfg.validation_session_name]
        print(f"Removed '{cfg.validation_session_name}' from train sessions (held out for validation).")
    if not train_sessions:
        raise ValueError("No training sessions remain after removing validation session.")
    cfg.train_session_names = train_sessions

    set_seed(cfg.seed)

    # Check for existing output folder conflict.
    if os.path.isdir(cfg.output_dir) and os.listdir(cfg.output_dir):
        try:
            import easygui
            choice = easygui.buttonbox(
                msg=(
                    f"Output folder already exists and is non-empty:\n\n"
                    f"  {os.path.abspath(cfg.output_dir)}\n\n"
                    "Choose an action:"
                ),
                title="Folder conflict",
                choices=["Overwrite (clear folder)", "Abort"],
            )
        except ImportError:
            choice = input(
                f"\nFolder '{cfg.output_dir}' already exists and is non-empty.\n"
                "Type 'overwrite' to clear it, or press Enter to abort: "
            ).strip().lower()
            choice = "Overwrite (clear folder)" if choice == "overwrite" else "Abort"

        if choice != "Overwrite (clear folder)":
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(cfg.output_dir)
        print(f"Cleared '{cfg.output_dir}'.")

    os.makedirs(cfg.output_dir, exist_ok=True)
    CodeLogger.log_code(cfg.output_dir, ['.', 'Library'], label=CONDITION)
    write_overview(cfg, cfg.output_dir)

    print("History-NN GA training  (v2 — IID sign wrapper)")
    print("=" * 60)
    print(f"Train sessions:      {train_sessions}")
    print(f"Validation session:  {cfg.validation_session_name}")

    evs_train = [Evaluator(build_simulator(sn, quiet_setup=cfg.quiet_setup), cfg)
                 for sn in train_sessions]
    ev_validation = None
    if cfg.validation_session_name and cfg.validation_session_name not in train_sessions:
        ev_validation = Evaluator(build_simulator(cfg.validation_session_name, quiet_setup=cfg.quiet_setup), cfg)

    trainer    = SimpleGATrainer(evs_train, cfg, evaluator_validation=ev_validation)
    best_genome = trainer.train()

    best_policy = HistoryNNPolicy(cfg.max_rotate1_deg, cfg.max_rotate2_deg,
                                   cfg.history_len, cfg.hidden_sizes)
    best_policy.set_genome(best_genome)

    rng          = random.Random(cfg.seed + 999_999)
    starts       = [evs_train[0].sample_start(rng) for _ in range(cfg.episodes_per_policy)]
    final        = evs_train[0].evaluate(best_policy, starts)

    rng_ex       = random.Random(cfg.seed + 424242)
    example_ep   = evs_train[0].episode(best_policy, evs_train[0].sample_start(rng_ex))
    plot_episode(evs_train[0].sim, example_ep,
                 os.path.join(cfg.output_dir, "plot_example_path.png"),
                 "History-NN v2 Example Path (training env)")

    if ev_validation is not None:
        rng_ex_val = random.Random(cfg.seed + 525252)
        example_ep_val = ev_validation.episode(best_policy, ev_validation.sample_start(rng_ex_val))
        plot_episode(ev_validation.sim, example_ep_val,
                     os.path.join(cfg.output_dir, "plot_example_path_validation.png"),
                     "History-NN v2 Example Path (validation env)")

    # Save best policy JSON.
    best_path = os.path.join(cfg.output_dir, "best_policy.json")
    with open(best_path, "w") as f:
        json.dump({
            "policy_type":          "HistoryNNPolicy_v2",
            "symmetry":             "iid_sign_wrapper",
            "condition":            CONDITION,
            "description":          DESCRIPTION,
            "genome":               best_genome.tolist(),
            "genome_size":          len(best_genome),
            "history_len":          cfg.history_len,
            "hidden_sizes":         list(cfg.hidden_sizes),
            "max_rotate1_deg":      cfg.max_rotate1_deg,
            "max_rotate2_deg":      cfg.max_rotate2_deg,
            "train_fitness":        float(final.get("fitness",        float("nan"))),
            "train_collision_rate": float(final.get("collision_rate", float("nan"))),
            "val_fitness":          float("nan"),
            "val_collision_rate":   float("nan"),
        }, f, indent=2)
    print(f"Best policy saved to {best_path}")

    plot_training(trainer.history, cfg.output_dir)
    plot_policy_curve(best_policy, cfg, cfg.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
