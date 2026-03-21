#!/usr/bin/env python3
"""
SCRIPT_TrainDemonstratorPolicy.py
==================================
GA training via imitation of the Demonstrator.

Fitness
-------
At each step the robot produces a net turn (rotate1 + rotate2).
The Demonstrator provides the target delta_angle for the robot's current
(x, y, yaw).  The per-step reward is:

    step_reward = cos(k * wrap(net_turn_rad - demonstrator_delta_rad)) + survival_bonus
    # wrap error to [-π, π]; k = imitation_sharpness; survival_bonus ensures surviving > crashing

This is 1.0 for a perfect match and -1.0 for the opposite direction.
Episode fitness = sum(step_rewards)  (early death penalised naturally by fewer survival_bonus additions).

Episode loop
------------
1. rotate1  = head1(history, last_iid)
2. measure  = sim.get_sonar_measurement(x, y, yaw_deg + rotate1)
3. rotate2  = head2(history, iid, dist, echo)
4. update history (canonical frame)
5. move robot
"""

import dataclasses
import json
import os
import random
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from Library.EnvironmentSimulator import EnvironmentSimulator
from Library.Demonstrator import Demonstrator


# ── Pushover helper ────────────────────────────────────────────────────────────

try:
    from Library.PushOver import send as _pushover_send
    _PUSHOVER_AVAILABLE = True
except Exception:
    _PUSHOVER_AVAILABLE = False

def pushover_notify(message: str, title: str = "3PiRobot training") -> None:
    if not _PUSHOVER_AVAILABLE:
        return
    try:
        _pushover_send(f"[{title}] {message}")
    except Exception as e:
        print(f"[Pushover] notification failed: {e}")


# ── Config ─────────────────────────────────────────────────────────────────────

CONDITION   = "imitation_v1"
DESCRIPTION = "Imitation of Demonstrator potential-field policy."

@dataclass
class Config:
    seed:        int = 42
    train_session_names: List[str] = field(
        default_factory=lambda: ["sessionB01", "sessionB02", "sessionB03", "sessionB04"]
    )
    validation_session_name: Optional[str] = "sessionB05"

    # NN
    history_len:  int            = 7
    hidden_sizes: Tuple[int,int] = (16, 8)

    # Action limits (no hard cap — tanh × 180° gives full ±180° range)
    max_rotate1_deg: float = 180.0
    max_rotate2_deg: float = 180.0
    fixed_drive_mm:  float = 100.0

    # Fitness
    collision_distance_mm:   float = 150.0
    imitation_sharpness:     float = 1.0   # k in cos(k*error), error wrapped to [-π, π]
    survival_bonus:          float = 0.2   # added to each step reward so surviving > crashing

    # GA
    population_size: int   = 75
    generations:     int   = 150
    elitism_count:   int   = 10
    mutation_rate:   float = 0.05
    mutation_sigma:  float = 0.2
    crossover_prob:  float = 0.5

    # Evaluation
    episodes_per_policy:                int = 16
    max_steps:                          int = 150
    validation_episodes_per_generation: int = 16

    # Demonstrator data
    demonstrator_dir: str = "DemonstratorData"

    # IO
    output_dir:       str            = f"Policy/{CONDITION}"
    description:      str            = DESCRIPTION
    quiet_setup:      bool           = True
    parallel_eval:    bool           = True
    num_workers:      Optional[int]  = 12
    pushover_every_n: int            = 10
    plot_every_n:     int            = 1   # save trajectory plot every N generations


# ── Utilities ──────────────────────────────────────────────────────────────────

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(f) else f


# ── Policy ─────────────────────────────────────────────────────────────────────

class HistoryNNPolicy:
    """
    Two-head MLP with shared history encoder.  Bilateral symmetry via IID sign
    wrapper — always pass raw physical IID, never pre-flip.
    """

    def __init__(self, max_rotate1_deg: float, max_rotate2_deg: float,
                 history_len: int, hidden_sizes: Tuple[int, int]):
        self.max_rotate1_deg = float(max_rotate1_deg)
        self.max_rotate2_deg = float(max_rotate2_deg)
        self.history_len     = int(history_len)
        self.hidden_sizes    = tuple(int(v) for v in hidden_sizes)
        self.feature_dim     = 5
        self.in_dim          = self.history_len * self.feature_dim
        h1, h2 = self.hidden_sizes
        self.shapes = [
            (h1, self.in_dim), (h1,),
            (h2, h1),          (h2,),
            (1,  h2),          (1,),
            (h2, h1 + 3),      (h2,),
            (1,  h2),          (1,),
        ]
        self.params: List[np.ndarray] = [np.zeros(s, dtype=np.float32) for s in self.shapes]

    def genome_size(self) -> int:
        return int(sum(int(np.prod(s)) for s in self.shapes))

    def set_genome(self, genome: np.ndarray) -> None:
        g, out, off = np.asarray(genome, dtype=np.float32).reshape(-1), [], 0
        for s in self.shapes:
            n = int(np.prod(s))
            out.append(g[off:off + n].reshape(s))
            off += n
        self.params = out

    def get_genome(self) -> np.ndarray:
        return np.concatenate([p.reshape(-1) for p in self.params]).astype(np.float32)

    def _shared_h1(self, hist_vec: np.ndarray) -> np.ndarray:
        x = np.asarray(hist_vec, dtype=np.float32).reshape(self.in_dim, 1)
        return np.tanh(self.params[0] @ x + self.params[1].reshape(-1, 1))

    def decide_rotate1_with_h1(self, h1: np.ndarray, last_iid_db: float) -> float:
        flip = safe_float(last_iid_db, 0.0) < 0.0
        h2   = np.tanh(self.params[2] @ h1 + self.params[3].reshape(-1, 1))
        y    = np.tanh(self.params[4] @ h2 + self.params[5].reshape(-1, 1))
        r    = float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate1_deg
        return -r if flip else r

    def decide_rotate2_with_h1(self, h1: np.ndarray, current_iid_db: float,
                                current_dist_mm: float, echo_present_prob: float = 1.0) -> float:
        phys          = safe_float(current_iid_db, 0.0)
        flip          = phys < 0.0
        iid_n  = float(np.clip(abs(phys) / 12.0,                          0.0, 2.0))
        dist_n = float(np.clip(safe_float(current_dist_mm, 1800.0)/2000.0, 0.0, 2.0))
        echo_n = float(np.clip(safe_float(echo_present_prob, 1.0),         0.0, 1.0))
        h1_aug = np.concatenate([h1, np.array([[iid_n],[dist_n],[echo_n]], dtype=np.float32)], axis=0)
        h2     = np.tanh(self.params[6] @ h1_aug + self.params[7].reshape(-1, 1))
        y      = np.tanh(self.params[8] @ h2     + self.params[9].reshape(-1, 1))
        r      = float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate2_deg
        return -r if flip else r


# ── Evaluator ──────────────────────────────────────────────────────────────────

def _build_evaluator(session_name: str, cfg: "Config") -> "Evaluator":
    return Evaluator(session_name, cfg)


class Evaluator:
    def __init__(self, session_name: str, cfg: "Config"):
        self.cfg = cfg
        if cfg.quiet_setup:
            with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
                self.sim = EnvironmentSimulator(session_name)
        else:
            self.sim = EnvironmentSimulator(session_name)

        path = os.path.join(cfg.demonstrator_dir, f"demonstrator_{session_name}.npz")
        d    = np.load(path)
        self.demo         = Demonstrator(d["xs_grid"], d["ys_grid"], d["potential"], d["grad_x"], d["grad_y"])
        self.xs_grid      = d["xs_grid"]
        self.ys_grid      = d["ys_grid"]
        self.dist_surface = d["dist_surface"]

        walls = getattr(self.sim.arena, "walls", np.array([], dtype=np.float32))
        self._walls = np.asarray(walls, dtype=np.float32) if walls is not None else np.array([], dtype=np.float32)

    def _clearance_mm(self, x: float, y: float) -> float:
        if self._walls.size == 0:
            return np.inf
        return float(np.min(np.hypot(self._walls[:, 0] - x, self._walls[:, 1] - y)))

    def _sample_start(self, rng: random.Random) -> Tuple[float, float, float]:
        rows, cols = np.where(self.dist_surface > self.cfg.collision_distance_mm)
        i = rng.randrange(len(rows))
        return float(self.xs_grid[cols[i]]), float(self.ys_grid[rows[i]]), rng.uniform(-180.0, 180.0)

    def _run_episode(self, policy: HistoryNNPolicy, start: Tuple[float, float, float],
                     rng_np: np.random.Generator, record: bool) -> Any:
        x, y, yaw_deg = start
        cfg           = self.cfg
        fdim, hl      = policy.feature_dim, cfg.history_len

        if hl > 0:
            hist_buf       = np.empty((hl, fdim), dtype=np.float32)
            hist_buf[:, 0] = rng_np.uniform(0.0,  1.0, size=hl)
            hist_buf[:, 1] = rng_np.uniform(0.2,  0.9, size=hl)
            hist_buf[:, 2] = rng_np.uniform(-0.5, 0.5, size=hl)
            hist_buf[:, 3] = rng_np.uniform(-0.5, 0.5, size=hl)
            hist_buf[:, 4] = 1.0
        else:
            hist_buf = None

        last_iid = float(rng_np.uniform(0.0, 1.0)) * 12.0 * float(rng_np.choice([-1.0, 1.0]))
        rewards:    List[float] = []
        trajectory: List[dict]  = []

        for _ in range(cfg.max_steps):
            h1      = policy._shared_h1(hist_buf.reshape(-1) if hist_buf is not None else np.zeros(0, dtype=np.float32))
            rotate1 = policy.decide_rotate1_with_h1(h1, last_iid)

            meas      = self.sim.get_sonar_measurement(x, y, yaw_deg + rotate1)
            iid       = safe_float(meas.get("iid_db"),            0.0)
            dist_mm   = safe_float(meas.get("distance_mm"),       1800.0)
            echo_prob = safe_float(meas.get("echo_present_prob"),  1.0)
            rotate2   = policy.decide_rotate2_with_h1(h1, iid, dist_mm, echo_prob)

            yaw_rad      = np.deg2rad(yaw_deg)
            demo_delta   = self.demo.get_delta_angle(x, y, yaw_rad)
            net_turn_rad = np.deg2rad(rotate1 + rotate2)
            error_rad    = (net_turn_rad - demo_delta + np.pi) % (2 * np.pi) - np.pi
            reward       = float(np.cos(cfg.imitation_sharpness * error_rad)) + cfg.survival_bonus
            rewards.append(reward)

            if record:
                trajectory.append({
                    "x": x, "y": y,
                    "yaw_deg": yaw_deg,
                    "iid_db": iid,
                    "demo_heading_deg": float(np.rad2deg(yaw_rad + demo_delta)),
                    "imitation_reward": reward,
                })

            flip1 = last_iid < 0.0
            flip2 = iid     < 0.0
            if hl > 0:
                hist_buf[:-1] = hist_buf[1:]
                hist_buf[-1]  = (
                    float(np.clip(abs(iid)              / 12.0,               0.0, 2.0)),
                    float(np.clip(dist_mm               / 2000.0,             0.0, 2.0)),
                    float(np.clip((-rotate1 if flip1 else rotate1) / cfg.max_rotate1_deg, -1.0, 1.0)),
                    float(np.clip((-rotate2 if flip2 else rotate2) / cfg.max_rotate2_deg, -1.0, 1.0)),
                    echo_prob,
                )
            last_iid = iid

            step       = self.sim.simulate_robot_movement(x, y, yaw_deg,
                             [{"rotate1_deg": rotate1, "rotate2_deg": rotate2, "drive_mm": cfg.fixed_drive_mm}],
                             compute_sonar=False)[0]
            x       = safe_float(step["position"]["x"], x)
            y       = safe_float(step["position"]["y"], y)
            yaw_deg = safe_float(step["orientation"],   yaw_deg)

            if self._clearance_mm(x, y) < cfg.collision_distance_mm:
                fitness = float(np.sum(rewards)) if rewards else -float(cfg.max_steps)
                return (fitness, trajectory) if record else fitness

        fitness = float(np.sum(rewards)) if rewards else -float(cfg.max_steps)
        return (fitness, trajectory) if record else fitness

    def episode(self, policy: HistoryNNPolicy, start: Tuple[float, float, float],
                rng_np: Optional[np.random.Generator] = None) -> float:
        if rng_np is None:
            rng_np = np.random.default_rng()
        return self._run_episode(policy, start, rng_np, record=False)

    def evaluate(self, policy: HistoryNNPolicy, rng: random.Random, n_episodes: int) -> float:
        rng_np = np.random.default_rng(rng.randint(0, 2**31))
        return float(np.mean([self.episode(policy, self._sample_start(rng), rng_np)
                               for _ in range(n_episodes)]))

    def episode_with_trajectory(self, policy: HistoryNNPolicy,
                                 start: Tuple[float, float, float],
                                 rng_np: Optional[np.random.Generator] = None) -> Tuple[float, List[dict]]:
        """Like episode() but also records per-step data for plotting."""
        if rng_np is None:
            rng_np = np.random.default_rng()
        return self._run_episode(policy, start, rng_np, record=True)


# ── Parallel worker ────────────────────────────────────────────────────────────
# Module-level state: each worker process initialises its own evaluators once.

_worker_evaluators: Optional[List[Evaluator]] = None
_worker_cfg: Optional[Config] = None

def _worker_init(cfg_dict: dict, session_names: List[str]) -> None:
    global _worker_evaluators, _worker_cfg
    import os as _os
    import torch as _torch
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    _torch.set_num_threads(1)
    pid = _os.getpid()
    print(f"  [worker {pid}] starting ({len(session_names)} sessions) ...", flush=True)
    cfg = Config(**{k: v for k, v in cfg_dict.items()
                    if k in {f.name for f in dataclasses.fields(Config)}})
    cfg.quiet_setup   = True
    cfg.hidden_sizes  = tuple(cfg.hidden_sizes)
    _worker_cfg       = cfg
    _worker_evaluators = [Evaluator(s, cfg) for s in session_names]
    print(f"  [worker {pid}] ready.", flush=True)


def _worker_warmup(_: int) -> bool:
    return True


def _eval_genome(args: Tuple[np.ndarray, int, int]) -> float:
    genome, n_eps_per_eval, seed = args
    rng    = random.Random(seed)
    policy = HistoryNNPolicy(
        _worker_cfg.max_rotate1_deg, _worker_cfg.max_rotate2_deg,
        _worker_cfg.history_len,     _worker_cfg.hidden_sizes,
    )
    policy.set_genome(genome)
    return float(np.mean([ev.evaluate(policy, rng, n_eps_per_eval) for ev in _worker_evaluators]))


# ── GA helpers ─────────────────────────────────────────────────────────────────

def _mutate(genome: np.ndarray, rate: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(genome.shape) < rate
    out  = genome.copy()
    out[mask] += rng.normal(0, sigma, mask.sum()).astype(np.float32)
    return np.clip(out, -3.0, 3.0).astype(np.float32)

def _crossover(g1: np.ndarray, g2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.where(rng.random(g1.shape) < 0.5, g1, g2)

def _make_offspring(elites: List[np.ndarray], cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if len(elites) >= 2 and rng.random() < cfg.crossover_prob:
        i, j = rng.choice(len(elites), size=2, replace=False)
        g    = _crossover(elites[i], elites[j], rng)
    else:
        g = elites[rng.integers(len(elites))].copy()
    return _mutate(g, cfg.mutation_rate, cfg.mutation_sigma, rng)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_progress(history: List[dict], out_path: str) -> None:
    import matplotlib.pyplot as plt
    gens      = [h["generation"]   for h in history]
    best      = [h["best_fitness"] for h in history]
    mean      = [h["mean_fitness"] for h in history]
    val       = [h["val_fitness"]  for h in history]
    pop_fits  = [h.get("population_fitnesses") for h in history]
    max_steps      = history[0].get("max_steps", 150) if history else 150
    survival_bonus = history[0].get("survival_bonus", 0.0) if history else 0.0
    max_fitness    = max_steps * (1.0 + survival_bonus)

    fig, (ax_box, ax_line) = plt.subplots(1, 2, figsize=(14, 4))

    # ── Box plot ──────────────────────────────────────────────────────────────
    if any(p is not None for p in pop_fits):
        data     = [p for p in pop_fits if p is not None]
        positions = [h["generation"] for h, p in zip(history, pop_fits) if p is not None]
        ax_box.boxplot(data, positions=positions, widths=0.7,
                       patch_artist=True,
                       boxprops=dict(facecolor="#90caf9", alpha=0.6),
                       medianprops=dict(color="#1565c0", linewidth=1.5),
                       flierprops=dict(marker=".", markersize=2, alpha=0.3),
                       whiskerprops=dict(linewidth=0.8),
                       capprops=dict(linewidth=0.8))
    ax_box.axhline(0,           color="grey", linewidth=0.5, linestyle=":")
    ax_box.axhline(max_fitness, color="grey", linewidth=0.5, linestyle=":")
    ax_box.set_xlabel("Generation")
    ax_box.set_ylabel("Fitness (Σ (cos + bonus) per step)")
    ax_box.set_title("Population fitness distribution")
    ax_box.set_ylim(-max_steps * 1.05, max_fitness * 1.05)

    # ── Line plot ─────────────────────────────────────────────────────────────
    ax_line.plot(gens, best, label="best (train)",  linewidth=1.5)
    ax_line.plot(gens, mean, label="mean (train)",  linewidth=1.0, alpha=0.7)
    if any(v is not None for v in val):
        val_clean = [v if v is not None else np.nan for v in val]
        ax_line.plot(gens, val_clean, label="best (val)", linewidth=1.5, linestyle="--")
    ax_line.axhline(0,           color="grey", linewidth=0.5, linestyle=":")
    ax_line.axhline(max_fitness, color="grey", linewidth=0.5, linestyle=":")
    ax_line.set_xlabel("Generation")
    ax_line.set_ylabel("Fitness (Σ (cos + bonus) per step)")
    ax_line.set_title(f"Imitation fitness — {CONDITION}")
    ax_line.legend(fontsize=8)
    ax_line.set_ylim(-max_steps * 1.05, max_fitness * 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_generation_best(policy: HistoryNNPolicy,
                         evaluators: List[Evaluator],
                         val_evaluator: Optional[Evaluator],
                         gen: int, best_genome_gen: int, out_path: str) -> None:
    import matplotlib.pyplot as plt
    """
    One panel per session (train + val).  Each panel shows:
      - Arena walls (grey)
      - Robot path coloured by imitation reward (green=good, red=bad)
      - Orange arrows: demonstrator target heading every N steps
      - Green dot = start, red X = end
    """
    from matplotlib.collections import LineCollection

    all_evals  = list(evaluators) + ([val_evaluator] if val_evaluator else [])
    labels     = [f"train: {e.sim.arena.session_name}" for e in evaluators]
    if val_evaluator:
        labels.append(f"val: {val_evaluator.sim.arena.session_name}")

    n      = len(all_evals)
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 6.0 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)
    fig.suptitle(f"Gen {gen} — best-so-far policy (from gen {best_genome_gen}) trajectories", fontsize=11)

    rng    = random.Random(gen)
    rng_np = np.random.default_rng(gen)
    STRIDE = 5
    ALEN   = 120.0   # arrow length in mm

    for idx, (ev, label) in enumerate(zip(all_evals, labels)):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        _, traj = ev.episode_with_trajectory(policy, ev._sample_start(rng), rng_np)

        walls = ev._walls
        if walls.size > 0:
            ax.scatter(walls[:, 0], walls[:, 1], s=1, color="#9e9e9e", alpha=0.3, zorder=1)

        if len(traj) >= 2:
            xs  = [s["x"] for s in traj]
            ys  = [s["y"] for s in traj]
            rew = [s["imitation_reward"] for s in traj]

            pts  = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc   = LineCollection(segs, cmap="RdYlGn", norm=plt.Normalize(-1, 1),
                                  linewidth=2, zorder=2)
            lc.set_array(np.array(rew[:-1]))
            ax.add_collection(lc)
            cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("imitation reward", fontsize=7)
            cb.ax.tick_params(labelsize=6)

            # Demonstrator heading arrows
            idxs = range(0, len(traj), STRIDE)
            ax.quiver(
                [traj[i]["x"] for i in idxs],
                [traj[i]["y"] for i in idxs],
                [ALEN * np.cos(np.deg2rad(traj[i]["demo_heading_deg"])) for i in idxs],
                [ALEN * np.sin(np.deg2rad(traj[i]["demo_heading_deg"])) for i in idxs],
                units="xy", angles="xy", scale_units="xy", scale=1,
                color="#ff9800", alpha=0.8, width=8, headwidth=4, zorder=4,
            )

            ax.scatter(xs[0],  ys[0],  s=60, color="#2e7d32", zorder=5, label="start")
            ax.scatter(xs[-1], ys[-1], s=60, color="#c62828", marker="x", zorder=5, label="end")
            fitness = float(np.sum(rew))
            ax.set_title(f"{label}  |  fitness={fitness:.3f}", fontsize=8)
        else:
            ax.set_title(f"{label}  (no trajectory)", fontsize=8)

        ax.set_xlabel("X (mm)", fontsize=7)
        ax.set_ylabel("Y (mm)", fontsize=7)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng_np = np.random.default_rng(cfg.seed)
    rng_ga = random.Random(cfg.seed)

    if os.path.exists(cfg.output_dir):
        shutil.rmtree(cfg.output_dir)
    os.makedirs(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # ── Setup ──────────────────────────────────────────────────────────────────
    print("Loading simulators and demonstrators ...")
    # Build evaluators in the main process for trajectory plotting
    plot_evaluators = [Evaluator(s, cfg) for s in cfg.train_session_names]
    val_evaluator   = Evaluator(cfg.validation_session_name, cfg) if cfg.validation_session_name else None
    print(f"  {len(plot_evaluators)} training sessions, validation: {cfg.validation_session_name}")

    template = HistoryNNPolicy(cfg.max_rotate1_deg, cfg.max_rotate2_deg,
                               cfg.history_len, cfg.hidden_sizes)
    gsize    = template.genome_size()
    print(f"  Genome size: {gsize}")
    print(f"  Parallel: {cfg.parallel_eval}  workers: {cfg.num_workers}")

    n_eps_per_eval = max(1, cfg.episodes_per_policy // len(cfg.train_session_names))
    cfg_dict       = asdict(cfg)

    # ── Population ─────────────────────────────────────────────────────────────
    population   = [np.clip(rng_np.standard_normal(gsize) * 0.25, -3.0, 3.0).astype(np.float32)
                    for _ in range(cfg.population_size)]
    best_genome     = None
    best_genome_gen = -1
    best_fitness    = -np.inf
    history      = []

    # Create the executor once — workers initialise their simulators only on startup,
    # not on every generation.
    use_parallel = cfg.parallel_eval and cfg.num_workers and cfg.num_workers > 1
    executor = None
    if use_parallel:
        print(f"  Starting {cfg.num_workers} worker processes ...")
        executor = ProcessPoolExecutor(
            max_workers=cfg.num_workers,
            initializer=_worker_init,
            initargs=(cfg_dict, cfg.train_session_names),
            mp_context=get_context("spawn"),
        )
        print(f"  Warming up {cfg.num_workers} workers ...")
        futs = [executor.submit(_worker_warmup, i) for i in range(cfg.num_workers)]
        for fut in futs:
            fut.result()
        print("  All workers ready.")
    else:
        _worker_init(cfg_dict, cfg.train_session_names)

    gen_bar = tqdm(range(cfg.generations), desc="Generations", unit="gen")

    try:
        for gen in gen_bar:
            seeds = [rng_ga.randint(0, 2**31) for _ in population]
            args  = [(g, n_eps_per_eval, s) for g, s in zip(population, seeds)]

            if use_parallel:
                fitnesses = []
                for i, f in enumerate(executor.map(_eval_genome, args), 1):
                    fitnesses.append(f)
                    gen_bar.set_postfix_str(f"eval {i}/{len(args)}", refresh=True)
            else:
                fitnesses = []
                for i, a in enumerate(args, 1):
                    fitnesses.append(_eval_genome(a))
                    gen_bar.set_postfix_str(f"eval {i}/{len(args)}", refresh=True)

            fitnesses = np.array(fitnesses)
            ranked    = np.argsort(fitnesses)[::-1]
            gen_best  = float(fitnesses[ranked[0]])
            gen_mean  = float(fitnesses.mean())

            if gen_best > best_fitness:
                best_fitness    = gen_best
                best_genome     = population[ranked[0]].copy()
                best_genome_gen = gen

            # Validation — scores this generation's best genome (not the all-time best)
            val_fitness = None
            if val_evaluator is not None:
                vp = HistoryNNPolicy(cfg.max_rotate1_deg, cfg.max_rotate2_deg,
                                     cfg.history_len, cfg.hidden_sizes)
                vp.set_genome(population[ranked[0]])
                val_fitness = val_evaluator.evaluate(vp, rng_ga, cfg.validation_episodes_per_generation)

            history.append({"generation": gen, "best_fitness": gen_best,
                             "mean_fitness": gen_mean, "val_fitness": val_fitness,
                             "max_steps": cfg.max_steps,
                             "survival_bonus": cfg.survival_bonus,
                             "population_fitnesses": fitnesses.tolist()})

            val_str = f"  val={val_fitness:.3f}" if val_fitness is not None else ""
            gen_bar.set_postfix_str(f"best={gen_best:.3f}  mean={gen_mean:.3f}{val_str}")
            tqdm.write(f"Gen {gen:4d} | best={gen_best:.3f}  mean={gen_mean:.3f}{val_str}")

            # Save
            np.save(os.path.join(cfg.output_dir, "best_genome.npy"), best_genome)
            with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
            plot_progress(history, os.path.join(cfg.output_dir, "fitness_curve.png"))

            if cfg.plot_every_n > 0 and (gen % cfg.plot_every_n == 0 or gen == cfg.generations - 1):
                bp = HistoryNNPolicy(cfg.max_rotate1_deg, cfg.max_rotate2_deg,
                                     cfg.history_len, cfg.hidden_sizes)
                bp.set_genome(best_genome)
                plot_generation_best(
                    bp, plot_evaluators, val_evaluator, gen, best_genome_gen,
                    os.path.join(cfg.output_dir, f"trajectories_gen{gen:04d}.png"),
                )

            if cfg.pushover_every_n > 0 and (gen + 1) % cfg.pushover_every_n == 0:
                pushover_notify(f"Gen {gen+1}/{cfg.generations} | best={best_fitness:.3f}{val_str}")

            # Breed
            elites          = [population[i] for i in ranked[:cfg.elitism_count]]
            next_population = list(elites)
            while len(next_population) < cfg.population_size:
                next_population.append(_make_offspring(elites, cfg, rng_np))
            population = next_population

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print(f"\nDone. Best fitness: {best_fitness:.3f}")
    print(f"Results saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
