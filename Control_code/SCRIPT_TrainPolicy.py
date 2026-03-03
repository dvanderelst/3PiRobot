#!/usr/bin/env python3
"""
GA training script with history-based NN steering.

Design:
- policy controls both rotate1 and rotate2 (each scaled to ±90°)
- drive distance is fixed to 100 mm
- inputs: short history of IID, distance, previous rotate1, previous rotate2,
  previous executed drive, previous blocked flag
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
    session_name: str = "sessionB01" # <-fallback
    train_session_names: Optional[List[str]] = field(
        default_factory=lambda: ["sessionB02", "sessionB03", "sessionB04", "sessionB05"]
    )
    validation_session_name: Optional[str] = "sessionB01"

    # Action limits/components
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
    validation_episodes_per_generation: int = 8
    # Directional bias prevention: episodes are split 50/50 forced-normal / forced-mirrored;
    # genome fitness = min(mean_normal_fitness, mean_mirrored_fitness) so neither direction can be ignored.
    use_random_mirroring: bool = True
    
    # Progressive difficulty: increase max_steps over generations to prevent early stopping
    use_progressive_steps: bool = True  # Enable progressive step increase
    progressive_steps_start: int = 50     # Starting max_steps (short episodes for initial learning)
    progressive_steps_end: int = 250     # Final max_steps (full-length episodes)
    progressive_steps_generations: int = 20  # Number of generations to reach final step count

    # Simplified fitness: reward survival and straight paths
    # (removed complex reward weights - now using simple 1 - w_turn * turn_penalty)
    w_turn_penalty: float = 2.0  # Weight for turn penalty (2.0 means full turn = -1.0 reward)
    sinuosity_window: int = 15    # Steps over which sinuosity is measured (longer = penalises slower oscillations)
    # Proximity shaping: penalize being close to geometry (walls/bounds).
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
    See policy_architecture.md for full design rationale.

    Shared encoder processes the history, then two separate heads decide:
      - Head 1 (rotate1): where to point the sonar head — decided before measuring.
      - Head 2 (rotate2): how to turn the body — decided after measuring at the look direction,
        with the current [iid_norm, dist_norm] injected alongside the shared encoding.

    History features per step: [iid_norm, dist_norm, rot1_norm, rot2_norm, drive_norm, blocked]
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
            (h2, h1 + 2),      (h2,),   # rot2 head hidden:     W2b, b2b  (+2 = iid_norm, dist_norm)
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
        """Head 1: decide where to look. Deadband uses last step's measured IID."""
        if abs(safe_float(last_iid_db, 0.0)) < self.deadband_db:
            return 0.0
        h1 = self._shared_h1(hist_vec)
        w2a, b2a, w3a, b3a = self.params[2], self.params[3], self.params[4], self.params[5]
        h2 = np.tanh(w2a @ h1 + b2a.reshape(-1, 1))
        y = np.tanh(w3a @ h2 + b3a.reshape(-1, 1))
        return float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate1_deg

    def decide_rotate2(self, hist_vec: np.ndarray, current_iid_db: float, current_dist_mm: float) -> float:
        """Head 2: decide body turn after looking. Current measurement injected alongside shared encoding."""
        if abs(safe_float(current_iid_db, 0.0)) < self.deadband_db:
            return 0.0
        h1 = self._shared_h1(hist_vec)
        iid_n = float(np.clip(safe_float(current_iid_db, 0.0) / 12.0, -2.0, 2.0))
        dist_n = float(np.clip(safe_float(current_dist_mm, 1800.0) / 2000.0, 0.0, 2.0))
        h1_aug = np.concatenate([h1, np.array([[iid_n], [dist_n]], dtype=np.float32)], axis=0)
        w2b, b2b, w3b, b3b = self.params[6], self.params[7], self.params[8], self.params[9]
        h2 = np.tanh(w2b @ h1_aug + b2b.reshape(-1, 1))
        y = np.tanh(w3b @ h2 + b3b.reshape(-1, 1))
        return float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate2_deg


def config_from_dict(d: Dict[str, Any]) -> Config:
    known = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in known})


class Evaluator:
    def __init__(self, simulator: EnvironmentSimulator, cfg: Config):
        self.sim = simulator
        self.cfg = cfg
        self.empirical_starts = self._build_empirical_starts()
        walls = getattr(self.sim.arena, "walls", np.array([], dtype=np.float32))
        self._wall_points = np.asarray(walls, dtype=np.float32) if walls is not None else np.array([], dtype=np.float32)

    def _build_empirical_starts(self) -> List[Tuple[float, float, float]]:
        dc = getattr(self.sim.arena, "dc", None)
        if dc is None or not hasattr(dc, "processors") or len(dc.processors) == 0:
            return []
        p = dc.processors[0]
        x = np.asarray(getattr(p, "rob_x", []), dtype=np.float64)
        y = np.asarray(getattr(p, "rob_y", []), dtype=np.float64)
        yaw = np.asarray(getattr(p, "rob_yaw_deg", []), dtype=np.float64)
        n = min(x.size, y.size, yaw.size)
        starts: List[Tuple[float, float, float]] = []
        for i in range(n):
            xi = safe_float(x[i], default=np.nan)
            yi = safe_float(y[i], default=np.nan)
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
        return (
            m,
            float(self.sim.arena.arena_width) - m,
            m,
            float(self.sim.arena.arena_height) - m,
        )

    def sample_start(self, rng: random.Random) -> Tuple[float, float, float]:
        use_empirical_pos = (
            self.cfg.use_empirical_starts
            and len(self.empirical_starts) > 0
            and rng.random() < float(np.clip(self.cfg.empirical_position_fraction, 0.0, 1.0))
        )
        if use_empirical_pos:
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

    def episode(self, policy: HistoryNNPolicy, start: Tuple[float, float, float], mirrored: bool = False) -> Dict[str, Any]:
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

        is_mirrored = mirrored

        last_iid = 0.0  # IID from previous step's look; used for deadband on first step
        prev_drive_norm = 0.0
        prev_blocked = 0.0
        # For sinuosity calculation: track recent positions
        position_history: collections.deque = collections.deque(maxlen=self.cfg.sinuosity_window)

        for t in range(self.cfg.max_steps):
            # --- Build history vector (rotate1 decided here; rotate2 decided after measuring) ---
            pad_n = self.cfg.history_len - len(hist)
            if pad_n > 0:
                hist_vec = np.concatenate(
                    [np.zeros((pad_n * 6,), dtype=np.float32)] + list(hist),
                    axis=0,
                ).astype(np.float32)
            else:
                hist_vec = np.concatenate(list(hist), axis=0).astype(np.float32)
            # --- Head 1: decide where to look (before measuring) ---
            rotate1 = policy.decide_rotate1(hist_vec, last_iid)
            rot1_norm = float(np.clip(rotate1 / self.cfg.max_rotate1_deg, -1.0, 1.0))
            if is_mirrored:
                rotate1 = -rotate1
                rot1_norm = -rot1_norm

            # --- Execute rotate1 then measure at the new look direction ---
            meas = self.sim.get_sonar_measurement(x, y, yaw + rotate1)
            iid = safe_float(meas.get("iid_db"), 0.0)
            dist_mm = safe_float(meas.get("distance_mm"), 1800.0)
            if is_mirrored:
                iid = -iid
            last_iid = iid
            # Full IID representation for smoother path generation
            iid_norm = float(np.clip(iid / 12.0, -2.0, 2.0))  # Normalized IID with full range
            dist_norm = float(np.clip(dist_mm / 2000.0, 0.0, 2.0))

            # --- Head 2: decide body turn using current measurement ---
            rotate2 = policy.decide_rotate2(hist_vec, iid, dist_mm)
            rot2_norm = float(np.clip(rotate2 / self.cfg.max_rotate2_deg, -1.0, 1.0))
            if is_mirrored:
                rotate2 = -rotate2
                rot2_norm = -rot2_norm

            # History entry: what was seen after this step's look + what was decided
            current_feat = np.array(
                [iid_norm, dist_norm, rot1_norm, rot2_norm, prev_drive_norm, prev_blocked],
                dtype=np.float32,
            )
            hist.append(current_feat)

            # --- Execute rotate2 + drive ---
            action = {
                "rotate1_deg": rotate1,
                "rotate2_deg": rotate2,
                "drive_mm": self.cfg.fixed_drive_mm,
            }

            step = self.sim.simulate_robot_movement(x, y, yaw, [action], compute_sonar=False)[0]
            nx = safe_float(step["position"]["x"], x)
            ny = safe_float(step["position"]["y"], y)
            nyaw = safe_float(step["orientation"], yaw)

            move = step.get("movement", {})
            exec_drive = safe_float(move.get("executed_drive_mm"), np.hypot(nx - x, ny - y))
            total_drive += exec_drive

            coll = step.get("collision", {})
            blocked = bool(coll.get("drive_blocked", False))
            blocked_any = blocked_any or blocked
            net_turn_deg = rotate1 + rotate2
            prev_drive_norm = float(np.clip(exec_drive / max(self.cfg.fixed_drive_mm, 1e-6), 0.0, 1.5))
            prev_blocked = 1.0 if blocked else 0.0
            clearance_mm = self._geometry_clearance_mm(nx, ny)
            warn = max(float(self.cfg.warning_distance_mm), 1e-6)
            proximity_term = float(np.clip((warn - clearance_mm) / warn, 0.0, 1.0))
            proximity_terms.append(proximity_term)

            # Diagnostic only (not part of fitness): positive when rotate2 opposes iid sign/magnitude.
            align_term = -iid_norm * float(np.clip(net_turn_deg / max(self.cfg.max_rotate1_deg + self.cfg.max_rotate2_deg, 1e-6), -1.0, 1.0))
            aligned_terms.append(align_term)

            if abs(iid_norm) > 0.15 and abs(net_turn_deg) > 2.0:
                sign_match_terms.append(1.0 if np.sign(net_turn_deg) == -np.sign(iid_norm) else 0.0)

            # Add current position to history for sinuosity calculation
            position_history.append((nx, ny))
            if len(position_history) >= 2:
                path_distance = sum(
                    np.hypot(position_history[i][0] - position_history[i-1][0],
                             position_history[i][1] - position_history[i-1][1])
                    for i in range(1, len(position_history))
                )
                first_x, first_y = position_history[0]
                last_x, last_y = position_history[-1]
                straight_distance = np.hypot(last_x - first_x, last_y - first_y)
                sinuosity = path_distance / straight_distance if straight_distance > 1e-6 else 1.0
                sinuosity = min(sinuosity, 2.0)
                reward = 1.0 - self.cfg.w_turn_penalty * (sinuosity - 1.0)
            else:
                reward = 1.0  # window not yet full
            total_reward += reward

            trajectory.append(
                {
                    "step": t,
                    "x": nx,
                    "y": ny,
                    "yaw_deg": nyaw,
                    "look_yaw_deg": yaw + rotate1,  # body yaw at step start + head turn = actual measurement direction
                    "iid_db": iid,
                    "distance_mm": dist_mm,
                    "rotate1_deg": rotate1,
                    "rotate2_deg": rotate2,
                    "executed_drive_mm": exec_drive,
                    "blocked": blocked,
                    "clearance_mm": clearance_mm,
                    "proximity_term": proximity_term,
                    "reward": float(reward),
                }
            )

            x, y, yaw = nx, ny, nyaw
            # Stop on unsafe geometric clearance or blocked drive.
            if clearance_mm <= float(self.cfg.collision_distance_mm):
                end_reason = "collision_distance_reached"
                blocked_any = True
                break
            if blocked:
                end_reason = "collision_blocked"
                break

        net_displacement = float(np.hypot(x - start_x, y - start_y))

        sign_match_rate = float(np.mean(sign_match_terms)) if sign_match_terms else 0.5
        
        # Normalize fitness by episode length for fair comparison across different max_steps
        normalized_fitness = float(total_reward) / float(self.cfg.max_steps)
        
        return {
            "fitness": normalized_fitness,  # Length-normalized fitness
            "fitness_raw": float(total_reward),  # Original unnormalized fitness for reference
            "steps": len(trajectory),
            "total_executed_drive_mm": float(total_drive),
            "net_displacement_mm": net_displacement,
            "collided": bool(blocked_any),
            "end_reason": end_reason,
            "proximity_mean": float(np.mean(proximity_terms) if proximity_terms else 0.0),
            "alignment_mean": float(np.mean(aligned_terms) if aligned_terms else 0.0),
            "sign_match_rate": sign_match_rate,
            "trajectory": trajectory,
            "was_mirrored": is_mirrored,  # Add mirroring status to metadata
        }

    def evaluate(self, policy: HistoryNNPolicy, starts: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        if self.cfg.use_random_mirroring and len(starts) >= 2:
            n_mirrored = len(starts) // 2
            n_normal = len(starts) - n_mirrored
            eps_normal = [self.episode(policy, s, mirrored=False) for s in starts[:n_normal]]
            eps_mirrored = [self.episode(policy, s, mirrored=True) for s in starts[n_normal:]]
            eps = eps_normal + eps_mirrored
            fit_normal = float(np.mean([e["fitness"] for e in eps_normal]))
            fit_mirrored = float(np.mean([e["fitness"] for e in eps_mirrored]))
            fitness = min(fit_normal, fit_mirrored)
        else:
            eps = [self.episode(policy, s, mirrored=False) for s in starts]
            fit_normal = float(np.mean([e["fitness"] for e in eps]))
            fit_mirrored = fit_normal
            fitness = fit_normal

        fit = np.array([e["fitness"] for e in eps], dtype=np.float32)
        return {
            "fitness": fitness,
            "fitness_std": float(np.std(fit)),
            "fitness_normal": fit_normal,
            "fitness_mirrored": fit_mirrored,
            "collision_rate": float(np.mean([1.0 if e["collided"] else 0.0 for e in eps])),
            "proximity_mean": float(np.mean([e.get("proximity_mean", 0.0) for e in eps])),
            "alignment_mean": float(np.mean([e["alignment_mean"] for e in eps])),
            "sign_match_rate": float(np.mean([e["sign_match_rate"] for e in eps])),
            "avg_drive_mm": float(np.mean([e["total_executed_drive_mm"] for e in eps])),
            "avg_net_displacement_mm": float(np.mean([e["net_displacement_mm"] for e in eps])),
            "episodes_raw": eps,
        }


_WORKER_CFG: Optional[Config] = None
_WORKER_EVS: Optional[List[Evaluator]] = None


def _init_worker(cfg_dict: Dict[str, Any]) -> None:
    global _WORKER_CFG, _WORKER_EVS
    _WORKER_CFG = config_from_dict(cfg_dict)
    train_sessions = list(_WORKER_CFG.train_session_names) if _WORKER_CFG.train_session_names else [_WORKER_CFG.session_name]
    train_sessions = [s for s in train_sessions if isinstance(s, str) and s.strip()]
    if not train_sessions:
        train_sessions = [_WORKER_CFG.session_name]
    evs: List[Evaluator] = []
    for sname in train_sessions:
        sim = build_simulator(sname, quiet_setup=_WORKER_CFG.quiet_setup)
        evs.append(Evaluator(sim, _WORKER_CFG))
    _WORKER_EVS = evs


def _eval_genome_worker(genome: np.ndarray, starts_by_env: List[List[Tuple[float, float, float]]]) -> Dict[str, Any]:
    if _WORKER_CFG is None or _WORKER_EVS is None:
        raise RuntimeError("Worker not initialized")
    pol = HistoryNNPolicy(
        _WORKER_CFG.max_rotate1_deg,
        _WORKER_CFG.max_rotate2_deg,
        _WORKER_CFG.iid_deadband_db,
        _WORKER_CFG.history_len,
        _WORKER_CFG.hidden_sizes,
    )
    pol.set_genome(genome)
    per_env: List[Dict[str, Any]] = []
    for ev, starts in zip(_WORKER_EVS, starts_by_env):
        per_env.append(ev.evaluate(pol, starts))

    def mean_key(k: str) -> float:
        vals = [safe_float(r.get(k, float("nan")), float("nan")) for r in per_env]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "fitness": mean_key("fitness"),
        "fitness_std": mean_key("fitness_std"),
        "fitness_normal": mean_key("fitness_normal"),
        "fitness_mirrored": mean_key("fitness_mirrored"),
        "collision_rate": mean_key("collision_rate"),
        "proximity_mean": mean_key("proximity_mean"),
        "alignment_mean": mean_key("alignment_mean"),
        "sign_match_rate": mean_key("sign_match_rate"),
        "avg_drive_mm": mean_key("avg_drive_mm"),
        "avg_net_displacement_mm": mean_key("avg_net_displacement_mm"),
        "episodes_raw": per_env[0].get("episodes_raw", []) if per_env else [],
        "per_env_fitness": [safe_float(r.get("fitness", float("nan")), float("nan")) for r in per_env],
    }


class SimpleGATrainer:
    def __init__(self, evaluators_train: List[Evaluator], cfg: Config, evaluator_validation: Optional[Evaluator] = None):
        if not evaluators_train:
            raise ValueError("Need at least one training evaluator")
        self.evs_train = evaluators_train
        self.ev = evaluators_train[0]
        self.ev_validation = evaluator_validation
        self.cfg = cfg
        self.sim = self.ev.sim
        self.generation_plot_dir = os.path.join(self.cfg.output_dir, "generation_best")
        self.best_genome: Optional[np.ndarray] = None
        self.best_fitness = -float("inf")
        self.history: Dict[str, List[float]] = {
            "best_fitness": [],
            "avg_fitness": [],
            "best_alignment_mean": [],
            "best_sign_match_rate": [],
            "best_collision_rate": [],
            "best_proximity_mean": [],
            "best_avg_net_displacement_mm": [],
            "val_best_fitness": [],
            "val_collision_rate": [],
            "val_avg_net_displacement_mm": [],
        }

    def _make_policy(self, genome: np.ndarray) -> HistoryNNPolicy:
        p = HistoryNNPolicy(
            self.cfg.max_rotate1_deg,
            self.cfg.max_rotate2_deg,
            self.cfg.iid_deadband_db,
            self.cfg.history_len,
            self.cfg.hidden_sizes,
        )
        p.set_genome(genome)
        return p

    def init_population(self) -> List[np.ndarray]:
        probe = HistoryNNPolicy(
            self.cfg.max_rotate1_deg,
            self.cfg.max_rotate2_deg,
            self.cfg.iid_deadband_db,
            self.cfg.history_len,
            self.cfg.hidden_sizes,
        )
        gsize = probe.genome_size()
        
        # Check for start_policy.json to initialize population
        start_policy_path = os.path.join(self.cfg.output_dir, "start_policy.json")
        if os.path.exists(start_policy_path):
            try:
                with open(start_policy_path, 'r') as f:
                    start_data = json.load(f)
                    start_genome = np.array(start_data.get("genome", []), dtype=np.float32)
                    if start_genome.size == gsize:
                        print(f"Initializing population from {start_policy_path}")
                        # Create population around the start policy with some variation
                        pop: List[np.ndarray] = []
                        for i in range(self.cfg.population_size):
                            if i == 0:
                                # First individual is the exact start policy
                                pop.append(start_genome.copy())
                            else:
                                # Others are variations around the start policy
                                variation = np.random.normal(0.0, 0.1, size=gsize).astype(np.float32)
                                varied_genome = np.clip(start_genome + variation, -3.0, 3.0)
                                pop.append(varied_genome)
                        return pop
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Could not load start_policy.json: {e}")
                print("Initializing random population instead")
        
        # Default random initialization
        pop: List[np.ndarray] = []
        for _ in range(self.cfg.population_size):
            pop.append(np.random.normal(0.0, 0.25, size=gsize).astype(np.float32))
        return pop

    def select(self, pop: List[np.ndarray], fitness: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        k = min(self.cfg.tournament_size, len(pop))
        idx1 = random.sample(range(len(pop)), k)
        idx2 = random.sample(range(len(pop)), k)
        p1 = pop[max(idx1, key=lambda i: fitness[i])]
        p2 = pop[max(idx2, key=lambda i: fitness[i])]
        return p1.copy(), p2.copy()

    def crossover(self, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        if random.random() > self.cfg.crossover_rate:
            return g1 if random.random() < 0.5 else g2
        alpha = np.random.rand(*g1.shape).astype(np.float32)
        child = alpha * g1 + (1.0 - alpha) * g2
        return np.clip(child, -3.0, 3.0).astype(np.float32)

    def mutate(self, g: np.ndarray) -> np.ndarray:
        mask = (np.random.rand(*g.shape) < self.cfg.mutation_rate).astype(np.float32)
        noise = np.random.normal(0.0, self.cfg.mutation_sigma, size=g.shape).astype(np.float32)
        out = g + mask * noise
        return np.clip(out, -3.0, 3.0).astype(np.float32)

    def _generation_starts(self, gen: int) -> List[List[Tuple[float, float, float]]]:
        starts_all: List[List[Tuple[float, float, float]]] = []
        for env_i, ev in enumerate(self.evs_train):
            rng = random.Random(self.cfg.seed + 10000 * (gen + 1) + 1000 * env_i)
            starts = [ev.sample_start(rng) for _ in range(self.cfg.episodes_per_policy)]
            starts_all.append(starts)
        return starts_all

    def _evaluate_on_train_envs(
        self,
        genome: np.ndarray,
        starts_by_env: List[List[Tuple[float, float, float]]],
    ) -> Dict[str, Any]:
        per_env: List[Dict[str, Any]] = []
        pol = self._make_policy(genome)
        for ev, starts in zip(self.evs_train, starts_by_env):
            per_env.append(ev.evaluate(pol, starts))

        def mean_key(k: str) -> float:
            vals = [safe_float(r.get(k, float("nan")), float("nan")) for r in per_env]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        out: Dict[str, Any] = {
            "fitness": mean_key("fitness"),
            "fitness_std": mean_key("fitness_std"),
            "fitness_normal": mean_key("fitness_normal"),
            "fitness_mirrored": mean_key("fitness_mirrored"),
            "collision_rate": mean_key("collision_rate"),
            "proximity_mean": mean_key("proximity_mean"),
            "alignment_mean": mean_key("alignment_mean"),
            "sign_match_rate": mean_key("sign_match_rate"),
            "avg_drive_mm": mean_key("avg_drive_mm"),
            "avg_net_displacement_mm": mean_key("avg_net_displacement_mm"),
            # Keep primary env episodes for plotting triplets.
            "episodes_raw": per_env[0].get("episodes_raw", []),
            "per_env_fitness": [safe_float(r.get("fitness", float("nan")), float("nan")) for r in per_env],
        }
        return out

    def train(self) -> np.ndarray:
        pop = self.init_population()
        for gen in range(self.cfg.generations):
            # Apply progressive step increase if enabled
            if self.cfg.use_progressive_steps and self.cfg.progressive_steps_generations > 0:
                # Calculate current max_steps based on generation
                progress = min(gen / self.cfg.progressive_steps_generations, 1.0)
                current_max_steps = int(
                    self.cfg.progressive_steps_start + 
                    progress * (self.cfg.progressive_steps_end - self.cfg.progressive_steps_start)
                )
                # Update the config for this generation
                self.cfg.max_steps = current_max_steps
                print(f"Generation {gen+1}: max_steps = {current_max_steps} (progressive)")
            
            starts_by_env = self._generation_starts(gen)
            fitness: List[float] = [0.0] * len(pop)
            details: List[Dict[str, Any]] = [None] * len(pop)  # type: ignore

            use_parallel = self.cfg.parallel_eval and len(pop) > 1
            if use_parallel:
                workers = self.cfg.num_workers or max(1, min(os.cpu_count() or 1, 8))
                cfg_dict = asdict(self.cfg)
                try:
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=_init_worker,
                        initargs=(cfg_dict,),
                    ) as ex:
                        results = ex.map(_eval_genome_worker, pop, [starts_by_env] * len(pop))
                        for i, res in enumerate(tqdm(results, total=len(pop), desc=f"Gen {gen+1}/{self.cfg.generations}")):
                            fitness[i] = float(res["fitness"])
                            details[i] = res
                except Exception as e:
                    print(f"Parallel eval failed ({type(e).__name__}: {e}); using serial.")
                    use_parallel = False

            if not use_parallel:
                for i, g in enumerate(tqdm(pop, desc=f"Gen {gen+1}/{self.cfg.generations}")):
                    if details[i] is not None:   # already evaluated in parallel
                        continue
                    res = self._evaluate_on_train_envs(g, starts_by_env)
                    fitness[i] = float(res["fitness"])
                    details[i] = res

            f_np = np.asarray(fitness, dtype=np.float32)
            best_idx = int(np.argmax(f_np))
            best_g = pop[best_idx].copy()
            best_res = details[best_idx]

            if float(f_np[best_idx]) > self.best_fitness:
                self.best_fitness = float(f_np[best_idx])
                self.best_genome = best_g.copy()

            self.history["best_fitness"].append(float(np.max(f_np)))
            self.history["avg_fitness"].append(float(np.mean(f_np)))
            self.history["best_alignment_mean"].append(float(best_res["alignment_mean"]))
            self.history["best_sign_match_rate"].append(float(best_res["sign_match_rate"]))
            self.history["best_collision_rate"].append(float(best_res["collision_rate"]))
            self.history["best_proximity_mean"].append(float(best_res.get("proximity_mean", 0.0)))
            self.history["best_avg_net_displacement_mm"].append(float(best_res["avg_net_displacement_mm"]))
            eps = best_res.get("episodes_raw", [])
            if eps:
                ep_fits = np.array([safe_float(ep.get("fitness"), float("-inf")) for ep in eps], dtype=np.float64)
                best_ep_fit = float(np.max(ep_fits))
                median_ep_fit = float(np.median(ep_fits))
                worst_ep_fit = float(np.min(ep_fits))
            else:
                best_ep_fit = float("nan")
                median_ep_fit = float("nan")
                worst_ep_fit = float("nan")

            val_fit = float("nan")
            val_coll = float("nan")
            val_net_disp = float("nan")
            val_res: Optional[Dict[str, Any]] = None
            if self.ev_validation is not None:
                rng_val = random.Random(self.cfg.seed + 20000 * (gen + 1) + 777)
                n_val = max(1, int(self.cfg.validation_episodes_per_generation))
                starts_val = [self.ev_validation.sample_start(rng_val) for _ in range(n_val)]
                val_res = self.ev_validation.evaluate(self._make_policy(best_g), starts_val)
                val_fit = float(val_res.get("fitness", float("nan")))
                val_coll = float(val_res.get("collision_rate", float("nan")))
                val_net_disp = float(val_res.get("avg_net_displacement_mm", float("nan")))
            self.history["val_best_fitness"].append(val_fit)
            self.history["val_collision_rate"].append(val_coll)
            self.history["val_avg_net_displacement_mm"].append(val_net_disp)

            print(
                f"Gen {gen+1}: best={self.history['best_fitness'][-1]:.3f}, "
                f"avg={self.history['avg_fitness'][-1]:.3f}, "
                f"fit_norm={best_res.get('fitness_normal', float('nan')):.3f}, "
                f"fit_mirr={best_res.get('fitness_mirrored', float('nan')):.3f}, "
                f"best_ep={best_ep_fit:.3f}, med_ep={median_ep_fit:.3f}, worst_ep={worst_ep_fit:.3f}, "
                f"align={best_res['alignment_mean']:.3f}, "
                f"sign_match={best_res['sign_match_rate']:.3f}, "
                f"coll_rate={best_res['collision_rate']:.3f}, "
                f"prox={best_res.get('proximity_mean', 0.0):.3f}, "
                f"net_disp={best_res['avg_net_displacement_mm']:.1f}mm, "
                f"val_fit={val_fit:.3f}, val_coll={val_coll:.3f}"
            )
            self._save_generation_best_plot(gen + 1, best_res, val_res)
            self._save_generation_best_genome(gen + 1, best_g, best_res, val_fit, val_coll)
            self._save_live_policy_probe(best_g)

            if gen < self.cfg.generations - 1:
                order = np.argsort(f_np)[::-1]
                elite_n = min(self.cfg.elitism_count, len(pop))
                nxt = [pop[int(i)].copy() for i in order[:elite_n]]
                while len(nxt) < self.cfg.population_size:
                    p1, p2 = self.select(pop, fitness)
                    c = self.crossover(p1, p2)
                    c = self.mutate(c)
                    nxt.append(c.astype(np.float32))
                pop = nxt

        if self.best_genome is None:
            raise RuntimeError("No best genome found")
        return self.best_genome

    def _save_generation_best_plot(
        self,
        generation_number: int,
        detail: Dict[str, Any],
        validation_detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.cfg.save_generation_best_plots:
            return

        def _selection(episodes: List[Dict[str, Any]]) -> Optional[List[Tuple[str, int]]]:
            if not episodes:
                return None
            fitnesses = np.array([safe_float(ep.get("fitness"), float("-inf")) for ep in episodes], dtype=np.float64)
            order = np.argsort(fitnesses)
            return [("Best", int(order[-1])), ("Median", int(order[len(order) // 2])), ("Worst", int(order[0]))]

        episodes_train = detail.get("episodes_raw", [])
        sel_train = _selection(episodes_train)
        episodes_val = validation_detail.get("episodes_raw", []) if validation_detail else []
        sel_val = _selection(episodes_val)
        if sel_train is None:
            return
        os.makedirs(self.generation_plot_dir, exist_ok=True)

        has_val = sel_val is not None and self.ev_validation is not None
        if has_val:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = np.asarray(axes)
            train_fit = safe_float(detail.get("fitness"), float("nan"))
            val_fit = safe_float(validation_detail.get("fitness"), float("nan")) if validation_detail else float("nan")
            for j, (label, idx) in enumerate(sel_train):
                ep = episodes_train[idx]
                ep_fit = safe_float(ep.get("fitness"), float("nan"))
                end_reason = str(ep.get("end_reason", "unknown"))
                mirrored_str = " (Mirrored)" if ep.get("was_mirrored", False) else ""
                panel_title = f"Train {label}{mirrored_str} | ep_fit={ep_fit:.2f} | end={end_reason}"
                draw_episode_on_axis(self.sim, ep, axes[0, j], panel_title, show_legend=(j == 0))
            for j, (label, idx) in enumerate(sel_val):
                ep = episodes_val[idx]
                ep_fit = safe_float(ep.get("fitness"), float("nan"))
                end_reason = str(ep.get("end_reason", "unknown"))
                mirrored_str = " (Mirrored)" if ep.get("was_mirrored", False) else ""
                panel_title = f"Val {label}{mirrored_str} | ep_fit={ep_fit:.2f} | end={end_reason}"
                draw_episode_on_axis(self.ev_validation.sim, ep, axes[1, j], panel_title, show_legend=(j == 0))
            fig.suptitle(
                f"Gen {generation_number} | train_fit={train_fit:.2f} | val_fit={val_fit:.2f}",
                fontsize=14,
            )
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            train_fit = safe_float(detail.get("fitness"), float("nan"))
            for ax, (label, idx) in zip(axes, sel_train):
                ep = episodes_train[idx]
                ep_fit = safe_float(ep.get("fitness"), float("nan"))
                end_reason = str(ep.get("end_reason", "unknown"))
                panel_title = f"Train {label} | ep_fit={ep_fit:.2f} | end={end_reason}"
                draw_episode_on_axis(self.sim, ep, ax, panel_title, show_legend=(label == "Best"))
            fig.suptitle(f"Gen {generation_number} | train_fit={train_fit:.2f}", fontsize=14)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        output_path = os.path.join(self.generation_plot_dir, f"gen_{generation_number:03d}_triplet.png")
        fig.savefig(output_path, dpi=180)
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
        ax0.plot(g, self.history["avg_fitness"], label="Average fitness")
        if "val_best_fitness" in self.history and np.isfinite(np.asarray(self.history["val_best_fitness"], dtype=np.float64)).any():
            ax0.plot(g, self.history["val_best_fitness"], label="Validation best fitness")
        ax0.set_xlabel("Generation")
        ax0.set_ylabel("Fitness")
        ax0.set_title("Fitness Progress (Live)")
        ax0.grid(True, alpha=0.3)
        ax0.legend()

        ax1 = axes[1]
        ax1.plot(g, self.history["best_alignment_mean"], label="Alignment")
        ax1.plot(g, self.history["best_sign_match_rate"], label="Sign match")
        ax1.plot(g, self.history["best_collision_rate"], label="Collision rate")
        if "best_proximity_mean" in self.history:
            ax1.plot(g, self.history["best_proximity_mean"], label="Proximity")
        if "val_collision_rate" in self.history and np.isfinite(np.asarray(self.history["val_collision_rate"], dtype=np.float64)).any():
            ax1.plot(g, self.history["val_collision_rate"], label="Val collision rate")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Metric")
        ax1.set_title("Behavior Metrics (Live)")
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        fig.tight_layout()
        out = os.path.join(self.generation_plot_dir, "live_metrics.png")
        fig.savefig(out, dpi=180)
        plt.close(fig)

    def _save_generation_best_genome(
        self,
        generation_number: int,
        genome: np.ndarray,
        detail: Dict[str, Any],
        val_fit: float,
        val_coll: float,
    ) -> None:
        os.makedirs(self.generation_plot_dir, exist_ok=True)
        path = os.path.join(self.generation_plot_dir, f"gen_{generation_number:03d}_best_policy.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "generation": generation_number,
                    "policy_type": "HistoryNNPolicy",
                    "genome": genome.tolist(),
                    "genome_size": len(genome),
                    "history_len": self.cfg.history_len,
                    "hidden_sizes": list(self.cfg.hidden_sizes),
                    "iid_deadband_db": self.cfg.iid_deadband_db,
                    "max_rotate1_deg": self.cfg.max_rotate1_deg,
                    "max_rotate2_deg": self.cfg.max_rotate2_deg,
                    "train_fitness": float(detail.get("fitness", float("nan"))),
                    "train_collision_rate": float(detail.get("collision_rate", float("nan"))),
                    "train_sign_match_rate": float(detail.get("sign_match_rate", float("nan"))),
                    "val_fitness": val_fit,
                    "val_collision_rate": val_coll,
                },
                f,
                indent=2,
            )

    def _save_live_policy_probe(self, genome: np.ndarray) -> None:
        os.makedirs(self.generation_plot_dir, exist_ok=True)
        pol = self._make_policy(genome)
        plot_policy_curve(
            pol,
            self.cfg,
            self.generation_plot_dir,
            filename="live_policy_probe_best.png",
        )


def plot_training(history: Dict[str, List[float]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    g = np.arange(1, len(history["best_fitness"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(g, history["best_fitness"], label="Best fitness")
    ax.plot(g, history["avg_fitness"], label="Average fitness")
    if "val_best_fitness" in history and np.isfinite(np.asarray(history["val_best_fitness"], dtype=np.float64)).any():
        ax.plot(g, history["val_best_fitness"], label="Validation best fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("History-NN GA Training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot_fitness.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(g, history["best_alignment_mean"], label="Best alignment mean")
    ax.plot(g, history["best_sign_match_rate"], label="Best sign match rate")
    ax.plot(g, history["best_collision_rate"], label="Best collision rate")
    if "best_proximity_mean" in history:
        ax.plot(g, history["best_proximity_mean"], label="Best proximity mean")
    if "best_avg_net_displacement_mm" in history:
        ax.plot(
            g,
            np.asarray(history["best_avg_net_displacement_mm"]) / 1000.0,
            label="Best avg net displacement (m)",
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Metric")
    ax.set_title("Alignment and Safety Metrics")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot_alignment_metrics.png"), dpi=180)
    plt.close(fig)


def plot_policy_curve(
    policy: HistoryNNPolicy,
    cfg: Config,
    output_dir: str,
    filename: str = "plot_policy_curve_best.png",
) -> None:
    """
    Simple visualization of the two-head policy response to wall directions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple 1x2 figure layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Test different IID signs and distances
    iid_signs = [-1.0, 1.0]  # Left wall, Right wall
    distances = [500.0, 1000.0, 1500.0]  # Different distances
    
    # Store results for both panels
    look_results = []  # For panel 1: rotate1 values
    strategy_results = []  # For panel 2: (total_rotation, correction) pairs
    
    for iid_sign in iid_signs:
        for dist in distances:
            iid_val = iid_sign * 6.0
            iid_norm_val = float(np.clip(iid_val / 12.0, -2.0, 2.0))
            dist_norm_val = float(np.clip(dist / 2000.0, 0.0, 2.0))

            # Create history feature vector with realistic steady-state values
            hist_vec = np.zeros(cfg.history_len * 6, dtype=np.float32)
            for step in range(cfg.history_len):
                hist_vec[step*6 + 0] = iid_norm_val   # iid_norm
                hist_vec[step*6 + 1] = dist_norm_val   # dist_norm
                hist_vec[step*6 + 4] = 1.0             # drive_norm (robot was driving)
            
            # Get policy decisions
            rotate1 = policy.decide_rotate1(hist_vec, iid_val)
            rotate2 = policy.decide_rotate2(hist_vec, iid_val, dist)
            
            # Calculate metrics
            total_rotation = rotate1 + rotate2
            look_vs_drive_diff = -rotate2  # How much look direction differed from drive direction
            
            look_results.append((iid_sign, rotate1))
            strategy_results.append((total_rotation, look_vs_drive_diff))
            
            # Simple scatter plot - no complex annotations
            marker = 'o' if iid_sign > 0 else 's'
            color = 'C0' if iid_sign > 0 else 'C1'
            
            # Panel 1: Total rotation response by wall type
            ax1.scatter(iid_sign, total_rotation, marker=marker, color=color, alpha=0.8, s=100)
            
            # Panel 2: Total rotation vs look-drive difference
            ax2.scatter(total_rotation, look_vs_drive_diff, marker=marker, color=color, alpha=0.8, s=100)
    
    # Format panel 1: Total rotation response
    ax1.axhline(0, color='black', linewidth=1, alpha=0.3)
    ax1.axvline(0, color='black', linewidth=1, alpha=0.3)
    ax1.set_xlabel('Wall Direction', fontsize=11)
    ax1.set_ylabel('Total Rotation Response (deg)', fontsize=11)
    ax1.set_title('Total Rotation Response to Walls', fontsize=13, fontweight='bold')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_xticks([-1, 1])
    ax1.set_xticklabels(['Left Wall', 'Right Wall'])
    ax1.set_ylim(-100, 100)
    ax1.grid(True, alpha=0.3)
    
    # Simple legend with explanation
    ax1.scatter([], [], marker='o', color='C0', label='Right wall (3 distances)', alpha=0.8, s=80)
    ax1.scatter([], [], marker='s', color='C1', label='Left wall (3 distances)', alpha=0.8, s=80)
    ax1.legend(fontsize=10, loc='upper right', title='Wall Response (500/1000/1500mm)')
    
    # Format panel 2: Rotation strategy
    ax2.axhline(0, color='black', linewidth=1, alpha=0.3)
    ax2.axvline(0, color='black', linewidth=1, alpha=0.3)
    lim = cfg.max_rotate1_deg + cfg.max_rotate2_deg
    ax2.plot([-lim, lim], [0, 0], '--', color='gray', alpha=0.5)  # Reference line at y=0
    ax2.set_xlabel('Total Rotation Response (deg)', fontsize=11)
    ax2.set_ylabel('Look-Drive Direction Difference (deg)', fontsize=11)
    ax2.set_title('Rotation Strategy: Total vs Look-Drive Difference', fontsize=13, fontweight='bold')
    ax2.set_xlim(-lim*1.1, lim*1.1)
    ax2.set_ylim(-lim*1.1, lim*1.1)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=180, bbox_inches='tight')
    plt.close(fig)


def draw_episode_on_axis(
    sim: EnvironmentSimulator,
    episode: Dict[str, Any],
    ax: Any,
    title: str,
    show_legend: bool = True,
) -> None:
    traj = episode.get("trajectory", [])
    if not traj:
        return
    xs = [s["x"] for s in traj]
    ys = [s["y"] for s in traj]

    walls = getattr(sim.arena, "walls", None)
    if walls is not None and len(walls) > 0:
        ax.scatter(walls[:, 0], walls[:, 1], s=1, color="#9e9e9e", alpha=0.25, label="Walls")
    ax.plot(xs, ys, "-", color="#1565c0", linewidth=2, label="Trajectory")
    ax.scatter(xs[0], ys[0], s=60, color="#2e7d32", label="Start")
    ax.scatter(xs[-1], ys[-1], s=60, marker="x", color="#c62828", label="End")

    # Look-direction arrows every 5 steps
    ARROW_STRIDE = 5
    ARROW_LEN_MM = 150.0
    indices = range(0, len(traj), ARROW_STRIDE)
    qx = [traj[i]["x"] for i in indices]
    qy = [traj[i]["y"] for i in indices]
    look_rad = [np.deg2rad(traj[i]["look_yaw_deg"]) for i in indices]
    qu = [ARROW_LEN_MM * np.cos(r) for r in look_rad]
    qv = [ARROW_LEN_MM * np.sin(r) for r in look_rad]
    ax.quiver(
        qx, qy, qu, qv,
        units="xy", angles="xy", scale_units="xy", scale=1,
        color="#ff9800", alpha=0.7, width=8.0,
        headwidth=4, headlength=5,
        label="Look dir",
    )

    all_x = list(xs)
    all_y = list(ys)
    if walls is not None and len(walls) > 0:
        all_x.extend(walls[:, 0].tolist())
        all_y.extend(walls[:, 1].tolist())
    min_x, max_x = float(np.min(all_x)), float(np.max(all_x))
    min_y, max_y = float(np.min(all_y)), float(np.max(all_y))
    pad_x = max(30.0, 0.05 * max(1.0, max_x - min_x))
    pad_y = max(30.0, 0.05 * max(1.0, max_y - min_y))
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    end_reason = str(episode.get("end_reason", "unknown"))
    if "end=" not in title:
        title = f"{title} | end={end_reason}"
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="best")


def plot_episode(sim: EnvironmentSimulator, episode: Dict[str, Any], output_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_episode_on_axis(sim, episode, ax, title, show_legend=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_overview(cfg: Config, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    lines = [
        "# History-NN GA Test",
        "",
        "Controller:",
        "- policy outputs both rotate1 and rotate2",
        "- drive fixed to 100 mm",
        "- policy is a small MLP that outputs [rotate1, rotate2] from history features",
        "- per-step features: [iid, distance, prev_rotate1, prev_rotate2, prev_exec_drive, prev_blocked]",
        f"- history length: {cfg.history_len}",
        f"- hidden sizes: {cfg.hidden_sizes}",
        f"- genome size: {HistoryNNPolicy(cfg.max_rotate1_deg, cfg.max_rotate2_deg, cfg.iid_deadband_db, cfg.history_len, cfg.hidden_sizes).genome_size()}",
        "- fitness: 1 - w_turn * (sinuosity - 1) (rewards survival, penalizes path tortuosity over short history)",
        "- initialization: if 'start_policy.json' exists in output_dir, population is initialized around it",
        "- directional bias prevention: behavioral symmetry training (random IID sign flipping + rotation output negation)",
        "- run ends if geometric clearance <= collision_distance_mm",
        "",
        f"- population: {cfg.population_size}",
        f"- generations: {cfg.generations}",
        f"- episodes per policy: {cfg.episodes_per_policy}",
        f"- max steps per episode: {cfg.max_steps}",
        f"- session (legacy): {cfg.session_name}",
        f"- train sessions: {cfg.train_session_names}",
        f"- validation session: {cfg.validation_session_name}",
        f"- validation episodes/gen: {cfg.validation_episodes_per_generation}",
        f"- empirical position fraction: {cfg.empirical_position_fraction}",
        f"- randomize empirical yaw: {cfg.randomize_empirical_yaw}",
        "",
        "```json",
        json.dumps(asdict(cfg), indent=2),
        "```",
        "",
    ]
    with open(os.path.join(output_dir, "run_overview.md"), "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    cfg = Config()
    train_sessions = list(cfg.train_session_names) if cfg.train_session_names else [cfg.session_name]
    train_sessions = [s for s in train_sessions if isinstance(s, str) and s.strip()]
    if not train_sessions:
        train_sessions = [cfg.session_name]
    # Backward compatibility: keep primary session field as first train env.
    cfg.session_name = train_sessions[0]
    if cfg.validation_session_name in train_sessions:
        train_sessions = [s for s in train_sessions if s != cfg.validation_session_name]
        print(
            f"Removed validation session '{cfg.validation_session_name}' from train sessions to keep it held out."
        )
    if not train_sessions:
        raise ValueError("No training sessions remain after removing held-out validation session.")
    cfg.train_session_names = train_sessions

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    write_overview(cfg, cfg.output_dir)

    print("History-NN GA training")
    print("=" * 60)
    print("Policy: drive=100mm, rotate1+rotate2 from history NN.")
    print(f"Train sessions: {train_sessions}")
    if cfg.validation_session_name:
        print(
            f"Validation session: {cfg.validation_session_name} "
            f"({cfg.validation_episodes_per_generation} eps/gen)"
        )

    evs_train: List[Evaluator] = []
    for sname in train_sessions:
        sim = build_simulator(sname, quiet_setup=cfg.quiet_setup)
        evs_train.append(Evaluator(sim, cfg))
    ev_validation: Optional[Evaluator] = None
    if cfg.validation_session_name and cfg.validation_session_name not in train_sessions:
        sim_val = build_simulator(cfg.validation_session_name, quiet_setup=cfg.quiet_setup)
        ev_validation = Evaluator(sim_val, cfg)
    trainer = SimpleGATrainer(evs_train, cfg, evaluator_validation=ev_validation)

    best_genome = trainer.train()
    best_policy = HistoryNNPolicy(
        cfg.max_rotate1_deg,
        cfg.max_rotate2_deg,
        cfg.iid_deadband_db,
        cfg.history_len,
        cfg.hidden_sizes,
    )
    best_policy.set_genome(best_genome)

    # Final deterministic evaluation
    rng = random.Random(cfg.seed + 999_999)
    starts = [evs_train[0].sample_start(rng) for _ in range(cfg.episodes_per_policy)]
    final = evs_train[0].evaluate(best_policy, starts)

    # Save one deterministic example trajectory in training environment.
    rng_example = random.Random(cfg.seed + 424242)
    example_start = evs_train[0].sample_start(rng_example)
    example_episode = evs_train[0].episode(best_policy, example_start)
    plot_episode(
        evs_train[0].sim,
        example_episode,
        os.path.join(cfg.output_dir, "plot_example_path.png"),
        "History-NN Example Path (training env)",
    )

    # Optional example trajectory in validation environment.
    if ev_validation is not None:
        rng_example_val = random.Random(cfg.seed + 525252)
        example_start_val = ev_validation.sample_start(rng_example_val)
        example_episode_val = ev_validation.episode(best_policy, example_start_val)
        plot_episode(
            ev_validation.sim,
            example_episode_val,
            os.path.join(cfg.output_dir, "plot_example_path_validation.png"),
            "History-NN Example Path (validation env)",
        )

    # Save artifacts
    with open(os.path.join(cfg.output_dir, "best_policy.json"), "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "policy_type": "HistoryNNPolicy",
                "genome": best_genome.tolist(),
                "genome_size": len(best_genome),
                "history_len": cfg.history_len,
                "hidden_sizes": list(cfg.hidden_sizes),
                "iid_deadband_db": cfg.iid_deadband_db,
                "max_rotate1_deg": cfg.max_rotate1_deg,
                "max_rotate2_deg": cfg.max_rotate2_deg,
                "final_eval": {
                    "fitness": final["fitness"],
                    "fitness_std": final["fitness_std"],
                    "sign_match_rate": final["sign_match_rate"],
                    "alignment_mean": final["alignment_mean"],
                    "collision_rate": final["collision_rate"],
                    "proximity_mean": final.get("proximity_mean", 0.0),
                    "avg_drive_mm": final["avg_drive_mm"],
                    "avg_net_displacement_mm": final["avg_net_displacement_mm"],
                },
            },
            f,
            indent=2,
        )

    with open(os.path.join(cfg.output_dir, "training_metrics.json"), "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": asdict(cfg),
                "best_fitness": trainer.best_fitness,
                "history": trainer.history,
                "final_eval": final,
            },
            f,
            indent=2,
        )

    plot_training(trainer.history, cfg.output_dir)
    plot_policy_curve(best_policy, cfg, cfg.output_dir)

    eps = final.get("episodes_raw", [])
    final_rows: List[Dict[str, Any]] = []
    for i, ep in enumerate(eps):
        traj = ep.get("trajectory", [])
        if traj:
            s0 = traj[0]
            eN = traj[-1]
            start_x, start_y, start_yaw = safe_float(s0.get("x")), safe_float(s0.get("y")), safe_float(s0.get("yaw_deg"))
            end_x, end_y, end_yaw = safe_float(eN.get("x")), safe_float(eN.get("y")), safe_float(eN.get("yaw_deg"))
        else:
            start_x = start_y = start_yaw = end_x = end_y = end_yaw = float("nan")

        final_rows.append(
            {
                "episode_index": i + 1,
                "fitness": safe_float(ep.get("fitness"), float("nan")),
                "steps": int(ep.get("steps", 0)),
                "collided": bool(ep.get("collided", False)),
                "end_reason": str(ep.get("end_reason", "unknown")),
                "total_executed_drive_mm": safe_float(ep.get("total_executed_drive_mm"), float("nan")),
                "net_displacement_mm": safe_float(ep.get("net_displacement_mm"), float("nan")),
                "proximity_mean": safe_float(ep.get("proximity_mean"), float("nan")),
                "alignment_mean": safe_float(ep.get("alignment_mean"), float("nan")),
                "sign_match_rate": safe_float(ep.get("sign_match_rate"), float("nan")),
                "start_x_mm": start_x,
                "start_y_mm": start_y,
                "start_yaw_deg": start_yaw,
                "end_x_mm": end_x,
                "end_y_mm": end_y,
                "end_yaw_deg": end_yaw,
            }
        )

        plot_episode(
            evs_train[0].sim,
            ep,
            os.path.join(cfg.output_dir, f"plot_eval_episode_{i+1:02d}.png"),
            f"History-NN Eval Episode {i+1} | fitness={ep['fitness']:.2f}",
        )
    if eps:
        fitnesses = np.array([safe_float(ep.get("fitness"), float("-inf")) for ep in eps], dtype=np.float64)
        best_idx = int(np.argmax(fitnesses))
        median_idx = int(np.argsort(fitnesses)[len(fitnesses) // 2])
        worst_idx = int(np.argmin(fitnesses))

        plot_episode(
            evs_train[0].sim,
            eps[best_idx],
            os.path.join(cfg.output_dir, "plot_eval_best_episode.png"),
            f"History-NN Best Episode #{best_idx+1}",
        )
        plot_episode(
            evs_train[0].sim,
            eps[median_idx],
            os.path.join(cfg.output_dir, "plot_eval_median_episode.png"),
            f"History-NN Median Episode #{median_idx+1}",
        )
        plot_episode(
            evs_train[0].sim,
            eps[worst_idx],
            os.path.join(cfg.output_dir, "plot_eval_worst_episode.png"),
            f"History-NN Worst Episode #{worst_idx+1}",
        )

    if final_rows:
        csv_path = os.path.join(cfg.output_dir, "final_eval_episodes.csv")
        fieldnames = list(final_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in final_rows:
                w.writerow(row)

    print("\nFinal evaluation:")
    print(f"  fitness: {final['fitness']:.3f} ± {final['fitness_std']:.3f}")
    print(f"  sign_match_rate: {final['sign_match_rate']:.3f}")
    print(f"  alignment_mean: {final['alignment_mean']:.3f}")
    print(f"  collision_rate: {final['collision_rate']:.3f}")
    print(f"  proximity_mean: {final.get('proximity_mean', 0.0):.3f}")
    print(f"  avg_drive_mm: {final['avg_drive_mm']:.1f}")
    print(f"  avg_net_displacement_mm: {final['avg_net_displacement_mm']:.1f}")
    print(f"\nOutputs in: {cfg.output_dir}")


if __name__ == "__main__":
    main()
