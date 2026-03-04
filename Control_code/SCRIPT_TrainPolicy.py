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

import csv
import json
import os
import random
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor
import collections
import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Library.EnvironmentSimulator import EnvironmentSimulator


@dataclass
class Config:
    seed: int = 42
    session_name: str = "sessionB01"
    train_session_names: Optional[List[str]] = field(
        default_factory=lambda: ["sessionB02", "sessionB03", "sessionB04", "sessionB05"]
    )
    validation_session_name: Optional[str] = "sessionB01"

    # Action limits
    max_rotate1_deg: float = 45.0
    max_rotate2_deg: float = 45.0
    fixed_drive_mm: float = 100.0
    iid_deadband_db: float = 0.25
    history_len: int = 5
    hidden_sizes: Tuple[int, int] = (16, 8)

    # GA
    population_size: int = 80
    generations: int = 120
    elitism_count: int = 8
    tournament_size: int = 5
    crossover_rate: float = 0.85
    mutation_rate: float = 0.2
    mutation_sigma: float = 0.2

    # Evaluation
    episodes_per_policy: int = 8
    max_steps: int = 50
    spawn_margin_mm: float = 150.0
    use_empirical_starts: bool = True
    randomize_empirical_yaw: bool = True
    empirical_position_fraction: float = 1.0
    # When a ValidStarts JSON exists for the session it is used instead of the
    # raw empirical positions.  Valid-starts already carry pre-filtered yaws so
    # randomize_empirical_yaw is ignored for them.
    valid_starts_dir: str = "ValidStarts"
    validation_episodes_per_generation: int = 8

    # Progressive difficulty
    use_progressive_steps: bool = True
    progressive_steps_start: int = 50
    progressive_steps_end: int = 250
    progressive_steps_generations: int = 50

    # Fitness
    w_turn_penalty: float = 2.0
    sinuosity_window: int = 15
    warning_distance_mm: float = 300.0
    collision_distance_mm: float = 150.0

    # IO
    output_dir: str = "Policy"
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
         drive_norm, blocked]
    where canonical_iid = abs(physical_iid) and canonical_rotX = physical_rotX
    reflected back to the positive-IID frame.
    """

    def __init__(
        self,
        max_rotate1_deg: float,
        max_rotate2_deg: float,
        deadband_db: float,
        history_len: int,
        hidden_sizes: Tuple[int, int],
    ):
        self.max_rotate1_deg = float(max_rotate1_deg)
        self.max_rotate2_deg = float(max_rotate2_deg)
        self.deadband_db = float(deadband_db)
        self.history_len = int(history_len)
        self.hidden_sizes = tuple(int(v) for v in hidden_sizes)
        self.feature_dim = 6
        self.in_dim = self.history_len * self.feature_dim
        h1, h2 = self.hidden_sizes
        self.shapes = [
            (h1, self.in_dim), (h1,),   # shared encoder:       W1, b1
            (h2, h1),          (h2,),   # rot1 head hidden:     W2a, b2a
            (1,  h2),          (1,),    # rot1 head output:     W3a, b3a
            (h2, h1 + 2),      (h2,),   # rot2 head hidden:     W2b, b2b  (+2 = iid_n, dist_n)
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
        phys_last = safe_float(last_iid_db, 0.0)
        flip = phys_last < 0.0                        # wall was on left last step
        canonical_last = abs(phys_last)
        if canonical_last < self.deadband_db:
            return 0.0
        h1 = self._shared_h1(hist_vec)
        w2a, b2a, w3a, b3a = self.params[2], self.params[3], self.params[4], self.params[5]
        h2 = np.tanh(w2a @ h1 + b2a.reshape(-1, 1))
        y = np.tanh(w3a @ h2 + b3a.reshape(-1, 1))
        rotate1_canonical = float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate1_deg
        return -rotate1_canonical if flip else rotate1_canonical

    def decide_rotate2(self, hist_vec: np.ndarray, current_iid_db: float, current_dist_mm: float) -> float:
        """Head 2: decide body turn after looking (current measurement injected).

        SYMMETRY WRAPPER: current_iid_db is the raw physical IID just measured.
        If negative (wall on left), we reflect to canonical positive-IID frame,
        run the network, and negate the output so the body turns the correct
        physical direction.  Pass the raw physical IID — do NOT pre-flip.
        """
        phys = safe_float(current_iid_db, 0.0)
        flip = phys < 0.0                             # wall is on left this step
        canonical_iid = abs(phys)
        if canonical_iid < self.deadband_db:
            return 0.0
        h1 = self._shared_h1(hist_vec)
        # Network sees canonical (non-negative) IID — wall always appears on right.
        iid_n  = float(np.clip(canonical_iid / 12.0, 0.0, 2.0))
        dist_n = float(np.clip(safe_float(current_dist_mm, 1800.0) / 2000.0, 0.0, 2.0))
        h1_aug = np.concatenate([h1, np.array([[iid_n], [dist_n]], dtype=np.float32)], axis=0)
        w2b, b2b, w3b, b3b = self.params[6], self.params[7], self.params[8], self.params[9]
        h2 = np.tanh(w2b @ h1_aug + b2b.reshape(-1, 1))
        y = np.tanh(w3b @ h2 + b3b.reshape(-1, 1))
        rotate2_canonical = float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate2_deg
        return -rotate2_canonical if flip else rotate2_canonical


def config_from_dict(d: Dict[str, Any]) -> Config:
    known = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in known})


class Evaluator:
    def __init__(self, simulator: EnvironmentSimulator, cfg: Config):
        self.sim = simulator
        self.cfg = cfg
        self.valid_starts   = self._load_valid_starts()   # pre-filtered (x,y,yaw) or []
        self.empirical_starts = self._build_empirical_starts()
        walls = getattr(self.sim.arena, "walls", np.array([], dtype=np.float32))
        self._wall_points = np.asarray(walls, dtype=np.float32) if walls is not None else np.array([], dtype=np.float32)

    def _load_valid_starts(self) -> List[Tuple[float, float, float]]:
        """
        Load pre-computed valid (x, y, yaw) starts from ValidStarts JSON if it
        exists for this session.  These were generated by SCRIPT_ComputeValidStarts
        and already have position AND heading filtered for wall clearance, so they
        cover the full arena rather than only the empirical robot track.
        Returns [] if the file is not found (falls back to empirical starts).
        """
        session = getattr(self.sim.arena, "session_name", None)
        if not session:
            return []
        json_path = os.path.join(self.cfg.valid_starts_dir, f"{session}_valid_starts.json")
        if not os.path.isfile(json_path):
            return []
        try:
            with open(json_path) as f:
                data = json.load(f)
            starts = [
                (float(s["x"]), float(s["y"]), float(s["yaw_deg"]))
                for s in data.get("starts", [])
            ]
            print(f"  Loaded {len(starts)} valid starts for {session} from {json_path}")
            return starts
        except Exception as e:
            print(f"  ⚠ Could not load valid starts for {session}: {e}")
            return []

    def _build_empirical_starts(self) -> List[Tuple[float, float, float]]:
        dc = getattr(self.sim.arena, "dc", None)
        if dc is None or not hasattr(dc, "processors") or len(dc.processors) == 0:
            return []
        p = dc.processors[0]
        x   = np.asarray(getattr(p, "rob_x",       []), dtype=np.float64)
        y   = np.asarray(getattr(p, "rob_y",       []), dtype=np.float64)
        yaw = np.asarray(getattr(p, "rob_yaw_deg", []), dtype=np.float64)
        n = min(x.size, y.size, yaw.size)
        starts: List[Tuple[float, float, float]] = []
        for i in range(n):
            xi  = safe_float(x[i],   default=np.nan)
            yi  = safe_float(y[i],   default=np.nan)
            yiw = safe_float(yaw[i], default=np.nan)
            if not (np.isfinite(xi) and np.isfinite(yi) and np.isfinite(yiw)):
                continue
            if not self._is_valid_spawn(xi, yi):
                continue
            starts.append((xi, yi, yiw % 360.0))
        return starts

    def _is_valid_spawn(self, x: float, y: float) -> bool:
        if not self.sim._is_in_bounds(x, y):
            return False
        if self.sim._segment_collides_with_walls(x, y, x, y, self.sim.robot_radius_mm * 1.1):
            return False
        return True

    def _random_spawn_bounds(self) -> Tuple[float, float, float, float]:
        m = float(self.cfg.spawn_margin_mm)
        meta = getattr(self.sim.arena, "meta", {}) or {}
        b = meta.get("arena_bounds_mm", None)
        if isinstance(b, dict):
            return (
                safe_float(b.get("min_x"), 0.0) + m,
                safe_float(b.get("max_x"), self.sim.arena.arena_width) - m,
                safe_float(b.get("min_y"), 0.0) + m,
                safe_float(b.get("max_y"), self.sim.arena.arena_height) - m,
            )
        return (m, float(self.sim.arena.arena_width) - m,
                m, float(self.sim.arena.arena_height) - m)

    def sample_start(self, rng: random.Random) -> Tuple[float, float, float]:
        # Prefer pre-computed valid starts (full arena coverage, filtered yaws).
        # Fall back to raw empirical positions, then to a random spawn.
        if self.valid_starts:
            x, y, yaw = self.valid_starts[rng.randrange(len(self.valid_starts))]
            return x, y, yaw
        use_empirical = (
            self.cfg.use_empirical_starts
            and len(self.empirical_starts) > 0
            and rng.random() < float(np.clip(self.cfg.empirical_position_fraction, 0.0, 1.0))
        )
        if use_empirical:
            x, y, yaw = self.empirical_starts[rng.randrange(len(self.empirical_starts))]
            if self.cfg.randomize_empirical_yaw:
                yaw = rng.uniform(0.0, 360.0)
            return x, y, yaw
        x_lo, x_hi, y_lo, y_hi = self._random_spawn_bounds()
        for _ in range(400):
            x = rng.uniform(x_lo, x_hi)
            y = rng.uniform(y_lo, y_hi)
            if self._is_valid_spawn(x, y):
                return x, y, rng.uniform(0.0, 360.0)
        return 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), rng.uniform(0.0, 360.0)

    def _geometry_clearance_mm(self, x: float, y: float) -> float:
        meta = getattr(self.sim.arena, "meta", {}) or {}
        b = meta.get("arena_bounds_mm", None)
        if isinstance(b, dict):
            min_x = safe_float(b.get("min_x"), 0.0)
            max_x = safe_float(b.get("max_x"), self.sim.arena.arena_width)
            min_y = safe_float(b.get("min_y"), 0.0)
            max_y = safe_float(b.get("max_y"), self.sim.arena.arena_height)
        else:
            min_x, max_x = 0.0, float(self.sim.arena.arena_width)
            min_y, max_y = 0.0, float(self.sim.arena.arena_height)
        boundary_clearance = float(min(x - min_x, max_x - x, y - min_y, max_y - y))
        boundary_clearance = max(0.0, boundary_clearance)
        if self._wall_points.size == 0:
            return boundary_clearance
        dx = self._wall_points[:, 0] - float(x)
        dy = self._wall_points[:, 1] - float(y)
        wall_clearance = float(np.min(np.hypot(dx, dy)))
        return min(boundary_clearance, wall_clearance)

    def episode(self, policy: HistoryNNPolicy, start: Tuple[float, float, float]) -> Dict[str, Any]:
        """Run one episode.

        The policy's IID sign wrapper (see HistoryNNPolicy docstring) handles
        bilateral symmetry transparently.  We pass raw physical IID to the policy
        at every step and store CANONICAL values in the history deque so that the
        network always sees the positive-IID frame.

        Canonical frame convention
        --------------------------
        canonical_iid  = abs(physical_iid)        (always >= 0)
        canonical_rotX = physical_rotX if physical_iid >= 0 else -physical_rotX
        (i.e. the action the network would have produced without the flip)
        """
        x, y, yaw = start
        start_x, start_y = x, y
        blocked_any = False
        end_reason = "max_steps_reached"
        total_reward = 0.0
        total_drive = 0.0
        proximity_terms: List[float] = []
        aligned_terms: List[float] = []
        sign_match_terms: List[float] = []
        trajectory: List[Dict[str, Any]] = []
        hist: collections.deque = collections.deque(maxlen=self.cfg.history_len)

        # last_physical_iid: raw measured IID from previous step.
        # Used by decide_rotate1 to determine flip direction before the current measurement.
        last_physical_iid = 0.0
        prev_drive_norm   = 0.0
        prev_blocked      = 0.0
        position_history: collections.deque = collections.deque(maxlen=self.cfg.sinuosity_window)

        for t in range(self.cfg.max_steps):
            # Build history vector from canonical history entries.
            pad_n = self.cfg.history_len - len(hist)
            if pad_n > 0:
                hist_vec = np.concatenate(
                    [np.zeros((pad_n * 6,), dtype=np.float32)] + list(hist), axis=0,
                ).astype(np.float32)
            else:
                hist_vec = np.concatenate(list(hist), axis=0).astype(np.float32)

            # --- Head 1: decide where to look (pass raw physical IID for flip detection) ---
            rotate1 = policy.decide_rotate1(hist_vec, last_physical_iid)

            # --- Execute rotate1, then measure at the new look direction ---
            meas    = self.sim.get_sonar_measurement(x, y, yaw + rotate1)
            physical_iid = safe_float(meas.get("iid_db"),      0.0)
            dist_mm      = safe_float(meas.get("distance_mm"), 1800.0)

            # --- Head 2: decide body turn (pass raw physical IID for flip detection) ---
            rotate2 = policy.decide_rotate2(hist_vec, physical_iid, dist_mm)

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

            hist.append(np.array(
                [canonical_iid_norm, dist_norm, canonical_rot1_norm, canonical_rot2_norm,
                 prev_drive_norm, prev_blocked],
                dtype=np.float32,
            ))
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

            coll        = step.get("collision", {})
            blocked     = bool(coll.get("drive_blocked", False))
            blocked_any = blocked_any or blocked
            net_turn_deg = rotate1 + rotate2
            prev_drive_norm = float(np.clip(exec_drive / max(self.cfg.fixed_drive_mm, 1e-6), 0.0, 1.5))
            prev_blocked    = 1.0 if blocked else 0.0
            clearance_mm    = self._geometry_clearance_mm(nx, ny)
            warn            = max(float(self.cfg.warning_distance_mm), 1e-6)
            proximity_term  = float(np.clip((warn - clearance_mm) / warn, 0.0, 1.0))
            proximity_terms.append(proximity_term)

            # Diagnostic metrics (not part of fitness).
            iid_norm_phys = float(np.clip(physical_iid / 12.0, -2.0, 2.0))
            align_term = -iid_norm_phys * float(np.clip(
                net_turn_deg / max(self.cfg.max_rotate1_deg + self.cfg.max_rotate2_deg, 1e-6), -1.0, 1.0))
            aligned_terms.append(align_term)
            if abs(iid_norm_phys) > 0.15 and abs(net_turn_deg) > 2.0:
                sign_match_terms.append(1.0 if np.sign(net_turn_deg) == -np.sign(iid_norm_phys) else 0.0)

            # Sinuosity fitness.
            position_history.append((nx, ny))
            if len(position_history) >= 2:
                path_dist = sum(
                    np.hypot(position_history[i][0] - position_history[i-1][0],
                             position_history[i][1] - position_history[i-1][1])
                    for i in range(1, len(position_history))
                )
                fx, fy = position_history[0]
                lx, ly = position_history[-1]
                straight = np.hypot(lx - fx, ly - fy)
                sinuosity = min(path_dist / straight if straight > 1e-6 else 1.0, 2.0)
                reward = 1.0 - self.cfg.w_turn_penalty * (sinuosity - 1.0)
            else:
                reward = 1.0
            total_reward += reward

            trajectory.append({
                "step": t,
                "x": nx, "y": ny, "yaw_deg": nyaw,
                "look_yaw_deg": yaw + rotate1,
                "iid_db": physical_iid,
                "distance_mm": dist_mm,
                "rotate1_deg": rotate1,
                "rotate2_deg": rotate2,
                "executed_drive_mm": exec_drive,
                "blocked": blocked,
                "clearance_mm": clearance_mm,
                "proximity_term": proximity_term,
                "reward": float(reward),
            })

            x, y, yaw = nx, ny, nyaw
            if clearance_mm <= float(self.cfg.collision_distance_mm):
                end_reason = "collision_distance_reached"
                blocked_any = True
                break
            if blocked:
                end_reason = "collision_blocked"
                break

        net_displacement    = float(np.hypot(x - start_x, y - start_y))
        sign_match_rate     = float(np.mean(sign_match_terms)) if sign_match_terms else 0.5
        normalized_fitness  = float(total_reward) / float(self.cfg.max_steps)

        return {
            "fitness":                  normalized_fitness,
            "fitness_raw":              float(total_reward),
            "steps":                    len(trajectory),
            "total_executed_drive_mm":  float(total_drive),
            "net_displacement_mm":      net_displacement,
            "collided":                 bool(blocked_any),
            "end_reason":               end_reason,
            "proximity_mean":           float(np.mean(proximity_terms) if proximity_terms else 0.0),
            "alignment_mean":           float(np.mean(aligned_terms)   if aligned_terms   else 0.0),
            "sign_match_rate":          sign_match_rate,
            "trajectory":               trajectory,
        }

    def evaluate(self, policy: HistoryNNPolicy, starts: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        eps = [self.episode(policy, s) for s in starts]
        fit_array = np.array([e["fitness"] for e in eps], dtype=np.float32)
        fitness   = float(np.mean(fit_array))
        return {
            "fitness":                  fitness,
            "fitness_std":              float(np.std(fit_array)),
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


def _eval_genome_worker(genome: np.ndarray, starts_by_env: List[List[Tuple[float, float, float]]]) -> Dict[str, Any]:
    if _WORKER_CFG is None or _WORKER_EVS is None:
        raise RuntimeError("Worker not initialized")
    pol = HistoryNNPolicy(
        _WORKER_CFG.max_rotate1_deg, _WORKER_CFG.max_rotate2_deg,
        _WORKER_CFG.iid_deadband_db, _WORKER_CFG.history_len, _WORKER_CFG.hidden_sizes,
    )
    pol.set_genome(genome)
    per_env = [ev.evaluate(pol, starts) for ev, starts in zip(_WORKER_EVS, starts_by_env)]

    def mean_key(k: str) -> float:
        vals = [safe_float(r.get(k, float("nan")), float("nan")) for r in per_env]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "fitness":                 mean_key("fitness"),
        "fitness_std":             mean_key("fitness_std"),
        "collision_rate":          mean_key("collision_rate"),
        "proximity_mean":          mean_key("proximity_mean"),
        "alignment_mean":          mean_key("alignment_mean"),
        "sign_match_rate":         mean_key("sign_match_rate"),
        "avg_drive_mm":            mean_key("avg_drive_mm"),
        "avg_net_displacement_mm": mean_key("avg_net_displacement_mm"),
        "episodes_raw":            per_env[0].get("episodes_raw", []) if per_env else [],
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
        self.history: Dict[str, List[float]] = {
            "best_fitness": [], "avg_fitness": [],
            "best_alignment_mean": [], "best_sign_match_rate": [],
            "best_collision_rate": [], "best_proximity_mean": [],
            "best_avg_net_displacement_mm": [],
            "val_best_fitness": [], "val_collision_rate": [],
            "val_avg_net_displacement_mm": [],
        }

    def _make_policy(self, genome: np.ndarray) -> HistoryNNPolicy:
        p = HistoryNNPolicy(
            self.cfg.max_rotate1_deg, self.cfg.max_rotate2_deg,
            self.cfg.iid_deadband_db, self.cfg.history_len, self.cfg.hidden_sizes,
        )
        p.set_genome(genome)
        return p

    def init_population(self) -> List[np.ndarray]:
        gsize = HistoryNNPolicy(
            self.cfg.max_rotate1_deg, self.cfg.max_rotate2_deg,
            self.cfg.iid_deadband_db, self.cfg.history_len, self.cfg.hidden_sizes,
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

    def select(self, pop: List[np.ndarray], fitness: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        k = min(self.cfg.tournament_size, len(pop))
        idx1 = random.sample(range(len(pop)), k)
        idx2 = random.sample(range(len(pop)), k)
        return (pop[max(idx1, key=lambda i: fitness[i])].copy(),
                pop[max(idx2, key=lambda i: fitness[i])].copy())

    def crossover(self, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        if random.random() > self.cfg.crossover_rate:
            return g1 if random.random() < 0.5 else g2
        alpha = np.random.rand(*g1.shape).astype(np.float32)
        return np.clip(alpha * g1 + (1.0 - alpha) * g2, -3.0, 3.0).astype(np.float32)

    def mutate(self, g: np.ndarray) -> np.ndarray:
        mask  = (np.random.rand(*g.shape) < self.cfg.mutation_rate).astype(np.float32)
        noise = np.random.normal(0.0, self.cfg.mutation_sigma, size=g.shape).astype(np.float32)
        return np.clip(g + mask * noise, -3.0, 3.0).astype(np.float32)

    def _generation_starts(self, gen: int) -> List[List[Tuple[float, float, float]]]:
        starts_all = []
        for env_i, ev in enumerate(self.evs_train):
            rng = random.Random(self.cfg.seed + 10000 * (gen + 1) + 1000 * env_i)
            starts_all.append([ev.sample_start(rng) for _ in range(self.cfg.episodes_per_policy)])
        return starts_all

    def _evaluate_on_train_envs(self, genome: np.ndarray,
                                 starts_by_env: List[List[Tuple[float, float, float]]]) -> Dict[str, Any]:
        pol     = self._make_policy(genome)
        per_env = [ev.evaluate(pol, starts) for ev, starts in zip(self.evs_train, starts_by_env)]

        def mean_key(k: str) -> float:
            vals = [safe_float(r.get(k, float("nan")), float("nan")) for r in per_env]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        return {
            "fitness":                 mean_key("fitness"),
            "fitness_std":             mean_key("fitness_std"),
            "collision_rate":          mean_key("collision_rate"),
            "proximity_mean":          mean_key("proximity_mean"),
            "alignment_mean":          mean_key("alignment_mean"),
            "sign_match_rate":         mean_key("sign_match_rate"),
            "avg_drive_mm":            mean_key("avg_drive_mm"),
            "avg_net_displacement_mm": mean_key("avg_net_displacement_mm"),
            "episodes_raw":            per_env[0].get("episodes_raw", []),
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

        for gen in range(self.cfg.generations):
            if self.cfg.use_progressive_steps and self.cfg.progressive_steps_generations > 0:
                progress = min(gen / self.cfg.progressive_steps_generations, 1.0)
                self.cfg.max_steps = int(
                    self.cfg.progressive_steps_start
                    + progress * (self.cfg.progressive_steps_end - self.cfg.progressive_steps_start)
                )
                print(f"Generation {gen+1}: max_steps = {self.cfg.max_steps} (progressive)")

            starts_by_env = self._generation_starts(gen)
            fitness: List[float]          = [0.0] * len(pop)
            details: List[Dict[str, Any]] = [None] * len(pop)  # type: ignore

            use_parallel = self.cfg.parallel_eval and len(pop) > 1
            if use_parallel:
                workers  = self.cfg.num_workers or max(1, min(os.cpu_count() or 1, 8))
                cfg_dict = asdict(self.cfg)
                try:
                    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                             initargs=(cfg_dict,)) as ex:
                        results = ex.map(_eval_genome_worker, pop, [starts_by_env] * len(pop))
                        for i, res in enumerate(tqdm(results, total=len(pop),
                                                     desc=f"Gen {gen+1}/{self.cfg.generations}")):
                            fitness[i] = float(res["fitness"])
                            details[i] = res
                except Exception as e:
                    print(f"Parallel eval failed ({type(e).__name__}: {e}); using serial.")
                    use_parallel = False

            if not use_parallel:
                for i, g in enumerate(tqdm(pop, desc=f"Gen {gen+1}/{self.cfg.generations}")):
                    if details[i] is not None:
                        continue
                    res        = self._evaluate_on_train_envs(g, starts_by_env)
                    fitness[i] = float(res["fitness"])
                    details[i] = res

            f_np     = np.asarray(fitness, dtype=np.float32)
            best_idx = int(np.argmax(f_np))
            best_g   = pop[best_idx].copy()
            best_res = details[best_idx]

            if float(f_np[best_idx]) > self.best_fitness:
                self.best_fitness = float(f_np[best_idx])
                self.best_genome  = best_g.copy()

            self.history["best_fitness"].append(float(np.max(f_np)))
            self.history["avg_fitness"].append(float(np.mean(f_np)))
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

            print(
                f"Gen {gen+1}: best={self.history['best_fitness'][-1]:.3f}, "
                f"avg={self.history['avg_fitness'][-1]:.3f}, "
                f"best_ep={best_ep_fit:.3f}, med_ep={median_ep_fit:.3f}, worst_ep={worst_ep_fit:.3f}, "
                f"align={best_res['alignment_mean']:.3f}, "
                f"sign_match={best_res['sign_match_rate']:.3f}, "
                f"coll={best_res['collision_rate']:.3f}, "
                f"prox={best_res.get('proximity_mean', 0.0):.3f}, "
                f"net_disp={best_res['avg_net_displacement_mm']:.1f}mm, "
                f"val_fit={val_fit:.3f}, val_coll={val_coll:.3f}"
            )
            self._save_generation_best_plot(gen + 1, best_res, val_res)
            self._save_generation_best_genome(gen + 1, best_g, best_res, val_fit, val_coll)
            self._save_live_policy_probe(best_g)

            if gen < self.cfg.generations - 1:
                order  = np.argsort(f_np)[::-1]
                elite_n = min(self.cfg.elitism_count, len(pop))
                nxt    = [pop[int(i)].copy() for i in order[:elite_n]]
                while len(nxt) < self.cfg.population_size:
                    p1, p2 = self.select(pop, fitness)
                    nxt.append(self.mutate(self.crossover(p1, p2)).astype(np.float32))
                pop = nxt

        if self.best_genome is None:
            raise RuntimeError("No best genome found")
        return self.best_genome

    def _save_generation_best_plot(self, generation_number: int, detail: Dict[str, Any],
                                    validation_detail: Optional[Dict[str, Any]] = None) -> None:
        if not self.cfg.save_generation_best_plots:
            return

        def _selection(episodes):
            if not episodes:
                return None
            fitnesses = np.array([safe_float(ep.get("fitness"), float("-inf")) for ep in episodes])
            order = np.argsort(fitnesses)
            return [("Best", int(order[-1])), ("Median", int(order[len(order) // 2])), ("Worst", int(order[0]))]

        episodes_train = detail.get("episodes_raw", [])
        sel_train      = _selection(episodes_train)
        episodes_val   = validation_detail.get("episodes_raw", []) if validation_detail else []
        sel_val        = _selection(episodes_val)
        if sel_train is None:
            return
        os.makedirs(self.generation_plot_dir, exist_ok=True)

        has_val = sel_val is not None and self.ev_validation is not None
        train_fit = safe_float(detail.get("fitness"), float("nan"))
        if has_val:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = np.asarray(axes)
            val_fit = safe_float(validation_detail.get("fitness"), float("nan")) if validation_detail else float("nan")
            for j, (label, idx) in enumerate(sel_train):
                ep = episodes_train[idx]
                draw_episode_on_axis(self.sim, ep, axes[0, j],
                                     f"Train {label} | ep_fit={safe_float(ep.get('fitness'), float('nan')):.2f} | end={ep.get('end_reason', '?')}",
                                     show_legend=(j == 0))
            for j, (label, idx) in enumerate(sel_val):
                ep = episodes_val[idx]
                draw_episode_on_axis(self.ev_validation.sim, ep, axes[1, j],
                                     f"Val {label} | ep_fit={safe_float(ep.get('fitness'), float('nan')):.2f} | end={ep.get('end_reason', '?')}",
                                     show_legend=(j == 0))
            fig.suptitle(f"Gen {generation_number} | train_fit={train_fit:.2f} | val_fit={val_fit:.2f}", fontsize=14)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for ax, (label, idx) in zip(axes, sel_train):
                ep = episodes_train[idx]
                draw_episode_on_axis(self.sim, ep, ax,
                                     f"Train {label} | ep_fit={safe_float(ep.get('fitness'), float('nan')):.2f} | end={ep.get('end_reason', '?')}",
                                     show_legend=(label == "Best"))
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
        ax0.plot(g, self.history["best_fitness"], label="Best fitness")
        ax0.plot(g, self.history["avg_fitness"],  label="Avg fitness")
        if np.isfinite(np.asarray(self.history["val_best_fitness"], dtype=np.float64)).any():
            ax0.plot(g, self.history["val_best_fitness"], label="Val fitness")
        ax0.set_xlabel("Generation"); ax0.set_ylabel("Fitness")
        ax0.set_title("Fitness Progress (Live)"); ax0.grid(True, alpha=0.3); ax0.legend()
        ax1 = axes[1]
        ax1.plot(g, self.history["best_alignment_mean"],  label="Alignment")
        ax1.plot(g, self.history["best_sign_match_rate"], label="Sign match")
        ax1.plot(g, self.history["best_collision_rate"],  label="Collision rate")
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
                "iid_deadband_db":      self.cfg.iid_deadband_db,
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
            hist_vec = np.zeros(cfg.history_len * 6, dtype=np.float32)
            for step in range(cfg.history_len):
                hist_vec[step*6 + 0] = iid_norm_val
                hist_vec[step*6 + 1] = dist_norm_val
                hist_vec[step*6 + 4] = 1.0
            rotate1 = policy.decide_rotate1(hist_vec, iid_val)
            rotate2 = policy.decide_rotate2(hist_vec, iid_val, dist)
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
    xs = [s["x"] for s in traj]
    ys = [s["y"] for s in traj]
    walls = getattr(sim.arena, "walls", None)
    if walls is not None and len(walls) > 0:
        ax.scatter(walls[:, 0], walls[:, 1], s=1, color="#9e9e9e", alpha=0.25, label="Walls")
    ax.plot(xs, ys, "-", color="#1565c0", linewidth=2, label="Trajectory")
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
        ax.legend(loc="best")


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
                            cfg.iid_deadband_db, cfg.history_len, cfg.hidden_sizes)
    lines = [
        "# History-NN GA (v2 — IID sign wrapper)",
        "",
        "Symmetry approach: IID sign wrapper inside HistoryNNPolicy.",
        "Network operates exclusively in canonical positive-IID frame.",
        "No episode-level mirroring.",
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
    os.makedirs(cfg.output_dir, exist_ok=True)
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
                                   cfg.iid_deadband_db, cfg.history_len, cfg.hidden_sizes)
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
            "genome":               best_genome.tolist(),
            "genome_size":          len(best_genome),
            "history_len":          cfg.history_len,
            "hidden_sizes":         list(cfg.hidden_sizes),
            "iid_deadband_db":      cfg.iid_deadband_db,
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
