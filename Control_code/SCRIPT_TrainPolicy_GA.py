#!/usr/bin/env python3
"""
Genetic Algorithm Policy Learning for active sensing navigation.

This version evolves a controller that outputs:
- rotate1_deg (pre-measurement sensing turn)
- rotate2_deg (post-measurement movement turn)
- drive_mm     (forward drive command)

Fitness is based on executed simulator behavior (not requested commands),
including collision/block penalties and smoothness costs.
"""

import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Library.EnvironmentSimulator import EnvironmentSimulator, create_test_simulator

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class TrainConfig:
    seed: int = 42

    # Policy
    history_length: int = 3
    hidden_sizes: Tuple[int, int] = (24, 16)

    # Action ranges
    max_rotate_deg: float = 180.0
    max_drive_mm: float = 350.0

    # GA
    population_size: int = 250
    generations: int = 250
    elitism_count: int = 25
    tournament_size: int = 5
    crossover_rate: float = 0.85
    mutation_rate: float = 0.06
    mutation_sigma: float = 0.08

    # Evaluation
    max_steps: int = 100
    episodes_per_policy: int = 8
    parallel_eval: bool = True
    num_workers: Optional[int] = 8
    quiet_worker_setup: bool = True
    quiet_main_setup: bool = False
    save_generation_best_plots: bool = True

    # Fitness weights (per-step accumulation)
    w_progress: float = 1.0  # Reward normalized executed forward movement each step.
    w_turn_cost: float = 1  # Penalize large sensing/movement rotations to reduce zig-zag behavior.
    w_action_delta: float = 0.5 # Penalize abrupt action changes to encourage smooth control.
    w_proximity: float = 0.8  # Penalize driving too close to obstacles even without direct collision.
    blocked_penalty: float = 4.0  # Extra penalty when commanded drive is blocked/truncated by geometry.
    collision_penalty: float = 30.0  # Large penalty for collision events to strongly discourage unsafe policies.

    # Safety distances in mm
    warning_distance_mm: float = 650.0  # Distance threshold below which proximity penalty begins.
    collision_distance_mm: float = 280.0  # Distance threshold treated as collision (terminate + heavy penalty).

    # Start sampling margin from outer arena boundaries
    spawn_margin_mm: float = 150.0
    use_empirical_starts: bool = True

    # IO
    output_dir: str = "policy_training_results"


# -----------------------------
# Utilities
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(f):
        return default
    return f


def write_run_overview_md(cfg: TrainConfig, output_dir: str) -> str:
    """
    Persist genome/controller + GA budget summary to markdown.
    """
    os.makedirs(output_dir, exist_ok=True)

    probe_policy = ActiveSensingPolicy(
        history_length=cfg.history_length,
        hidden_sizes=cfg.hidden_sizes,
        max_rotate_deg=cfg.max_rotate_deg,
        max_drive_mm=cfg.max_drive_mm,
    )
    n_params = int(sum(p.numel() for p in probe_policy.parameters()))
    input_dim = int(probe_policy.history_length * probe_policy.FEATURES_PER_STEP)
    genome_evals = int(cfg.population_size * cfg.generations)
    total_episodes = int(genome_evals * cfg.episodes_per_policy)
    total_step_ceiling = int(total_episodes * cfg.max_steps)

    lines = [
        "# Run Overview",
        "",
        "## Controller (Genome)",
        f"- Features per step: `{probe_policy.FEATURES_PER_STEP}`",
        f"- History length: `{probe_policy.history_length}`",
        f"- Input dimension: `{input_dim}`",
        f"- Hidden sizes: `{cfg.hidden_sizes}`",
        f"- Genome length (NN parameters): `{n_params}`",
        "",
        "## GA Budget",
        f"- Population size: `{cfg.population_size}`",
        f"- Generations: `{cfg.generations}`",
        f"- Episodes per policy: `{cfg.episodes_per_policy}`",
        f"- Genome evaluations: `{genome_evals}`",
        f"- Episode rollouts: `{total_episodes}`",
        f"- Max steps per episode: `{cfg.max_steps}`",
        f"- Max simulated steps (ceiling): `{total_step_ceiling}`",
        "",
        "## Key Settings",
        f"- Seed: `{cfg.seed}`",
        f"- Empirical starts: `{cfg.use_empirical_starts}`",
        f"- Action ranges: `rotate in [-{cfg.max_rotate_deg}, {cfg.max_rotate_deg}]`, `drive in [0, {cfg.max_drive_mm}]`",
        "",
        "## Config Snapshot",
        "```json",
        json.dumps(asdict(cfg), indent=2),
        "```",
        "",
    ]

    out_path = os.path.join(output_dir, "run_overview.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


def train_config_from_dict(d: Dict[str, Any]) -> TrainConfig:
    data = dict(d)
    hs = data.get("hidden_sizes", (24, 16))
    data["hidden_sizes"] = tuple(hs)
    return TrainConfig(**data)


# -----------------------------
# Policy
# -----------------------------
class ActiveSensingPolicy(nn.Module):
    """
    Feed-forward policy over fixed-length history.

    Per-history-step features:
    - iid_norm
    - distance_norm
    - prev_rotate1_norm
    - prev_rotate2_norm
    - prev_drive_norm
    """

    FEATURES_PER_STEP = 5

    def __init__(
        self,
        history_length: int,
        hidden_sizes: Tuple[int, int],
        max_rotate_deg: float,
        max_drive_mm: float,
    ):
        super().__init__()
        self.history_length = int(history_length)
        self.max_rotate_deg = float(max_rotate_deg)
        self.max_drive_mm = float(max_drive_mm)

        input_dim = self.history_length * self.FEATURES_PER_STEP
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, 3),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x shape: (batch, input_dim)
        y = self.net(x)
        rotate1 = torch.tanh(y[:, 0]) * self.max_rotate_deg
        rotate2 = torch.tanh(y[:, 1]) * self.max_rotate_deg
        drive = torch.sigmoid(y[:, 2]) * self.max_drive_mm
        return {
            "rotate1_deg": rotate1,
            "rotate2_deg": rotate2,
            "drive_mm": drive,
        }

    def get_genome(self) -> np.ndarray:
        flat = [p.detach().cpu().numpy().ravel() for p in self.parameters()]
        return np.concatenate(flat).astype(np.float32)

    def set_genome(self, genome: np.ndarray) -> None:
        genome = np.asarray(genome, dtype=np.float32)
        idx = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                chunk = genome[idx:idx + n]
                if chunk.size != n:
                    raise ValueError("Genome size mismatch while setting parameters")
                p.copy_(torch.from_numpy(chunk.reshape(p.shape)).to(dtype=p.dtype))
                idx += n
        if idx != genome.size:
            raise ValueError("Genome has extra values beyond model parameter count")


# -----------------------------
# Evaluation
# -----------------------------
class PolicyEvaluator:
    def __init__(self, simulator: EnvironmentSimulator, cfg: TrainConfig):
        self.simulator = simulator
        self.cfg = cfg
        self.arena = simulator.get_arena_info()
        self.empirical_starts: List[Tuple[float, float, float]] = self._build_empirical_start_pool()

    def _build_empirical_start_pool(self) -> List[Tuple[float, float, float]]:
        """
        Build candidate start states from recorded robot trajectory.
        """
        dc = getattr(self.simulator.arena, "dc", None)
        if dc is None or not hasattr(dc, "processors") or len(dc.processors) == 0:
            return []

        proc = dc.processors[0]
        rob_x = np.asarray(getattr(proc, "rob_x", []), dtype=np.float64)
        rob_y = np.asarray(getattr(proc, "rob_y", []), dtype=np.float64)
        rob_yaw = np.asarray(getattr(proc, "rob_yaw_deg", []), dtype=np.float64)
        n = min(rob_x.size, rob_y.size, rob_yaw.size)
        if n == 0:
            return []

        starts: List[Tuple[float, float, float]] = []
        for i in range(n):
            x = safe_float(rob_x[i], default=np.nan)
            y = safe_float(rob_y[i], default=np.nan)
            yaw = safe_float(rob_yaw[i], default=np.nan)
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(yaw)):
                continue
            if not self._is_valid_spawn(x, y):
                continue
            starts.append((x, y, yaw % 360.0))

        return starts

    def _arena_sampling_bounds(self) -> Tuple[float, float, float, float]:
        """
        Resolve world-coordinate bounds for sampling and plotting.

        Preference order:
        1. arena metadata bounds (if available)
        2. wall point cloud extent
        3. [0, width] x [0, height] fallback
        """
        margin = float(self.cfg.spawn_margin_mm)

        meta = getattr(self.simulator.arena, "meta", {}) or {}
        arena_bounds = meta.get("arena_bounds_mm", None)
        if isinstance(arena_bounds, dict):
            min_x = safe_float(arena_bounds.get("min_x"), 0.0)
            max_x = safe_float(arena_bounds.get("max_x"), self.arena["width_mm"])
            min_y = safe_float(arena_bounds.get("min_y"), 0.0)
            max_y = safe_float(arena_bounds.get("max_y"), self.arena["height_mm"])
            return min_x + margin, max_x - margin, min_y + margin, max_y - margin

        walls = getattr(self.simulator.arena, "walls", None)
        if walls is not None and len(walls) > 0:
            min_x = float(np.min(walls[:, 0]))
            max_x = float(np.max(walls[:, 0]))
            min_y = float(np.min(walls[:, 1]))
            max_y = float(np.max(walls[:, 1]))
            return min_x + margin, max_x - margin, min_y + margin, max_y - margin

        return (
            0.0 + margin,
            float(self.arena["width_mm"]) - margin,
            0.0 + margin,
            float(self.arena["height_mm"]) - margin,
        )

    def _is_valid_spawn(self, x: float, y: float) -> bool:
        """
        Spawn validity check:
        - must satisfy simulator boundary margins
        - must not already overlap wall clearance region
        """
        if not self.simulator._is_in_bounds(x, y):
            return False
        if self.simulator._segment_collides_with_walls(
            x, y, x, y, self.simulator.robot_radius_mm * 1.1
        ):
            return False
        return True

    def _sample_start_state(self, rng: random.Random) -> Tuple[float, float, float]:
        if self.cfg.use_empirical_starts and self.empirical_starts:
            idx = rng.randrange(len(self.empirical_starts))
            return self.empirical_starts[idx]

        x_lo, x_hi, y_lo, y_hi = self._arena_sampling_bounds()

        # Guard degenerate bounds
        if x_hi <= x_lo:
            x_lo, x_hi = x_lo, x_lo + 1.0
        if y_hi <= y_lo:
            y_lo, y_hi = y_lo, y_lo + 1.0

        for _ in range(400):
            x = rng.uniform(x_lo, x_hi)
            y = rng.uniform(y_lo, y_hi)
            if self._is_valid_spawn(x, y):
                yaw = rng.uniform(0.0, 360.0)
                return x, y, yaw

        # Fallback: deterministic search on coarse grid.
        gx = np.linspace(x_lo, x_hi, 20)
        gy = np.linspace(y_lo, y_hi, 20)
        for x in gx:
            for y in gy:
                if self._is_valid_spawn(float(x), float(y)):
                    yaw = rng.uniform(0.0, 360.0)
                    return float(x), float(y), yaw

        # Last resort: center of configured arena rectangle.
        yaw = rng.uniform(0.0, 360.0)
        return (
            0.5 * (x_lo + x_hi),
            0.5 * (y_lo + y_hi),
            yaw,
        )

    def _normalize_obs(self, dist_mm: float, iid_db: float, prev_action: Dict[str, float]) -> np.ndarray:
        # Keep scales bounded for stable evolution.
        distance_norm = np.clip(dist_mm / 2000.0, 0.0, 2.0)
        iid_norm = np.clip(iid_db / 12.0, -2.0, 2.0)

        r1 = np.clip(prev_action["rotate1_deg"] / self.cfg.max_rotate_deg, -1.0, 1.0)
        r2 = np.clip(prev_action["rotate2_deg"] / self.cfg.max_rotate_deg, -1.0, 1.0)
        drv = np.clip(prev_action["drive_mm"] / self.cfg.max_drive_mm, 0.0, 1.0)

        return np.array(
            [iid_norm, distance_norm, r1, r2, drv],
            dtype=np.float32,
        )

    def _history_to_input(self, history: List[np.ndarray], history_length: int) -> np.ndarray:
        if len(history) < history_length:
            pad = [np.zeros(ActiveSensingPolicy.FEATURES_PER_STEP, dtype=np.float32)] * (history_length - len(history))
            history = pad + history
        else:
            history = history[-history_length:]
        return np.concatenate(history, axis=0).astype(np.float32)

    def _step_reward(
        self,
        action: Dict[str, float],
        prev_action: Dict[str, float],
        executed_drive_mm: float,
        dist_after_drive_mm: float,
        drive_blocked: bool,
        collided: bool,
    ) -> float:
        c = self.cfg

        progress = executed_drive_mm / c.max_drive_mm

        turn_cost = (abs(action["rotate1_deg"]) + abs(action["rotate2_deg"])) / (2.0 * c.max_rotate_deg)

        action_delta = (
            abs(action["rotate1_deg"] - prev_action["rotate1_deg"]) / c.max_rotate_deg
            + abs(action["rotate2_deg"] - prev_action["rotate2_deg"]) / c.max_rotate_deg
            + abs(action["drive_mm"] - prev_action["drive_mm"]) / c.max_drive_mm
        ) / 3.0

        proximity = 0.0
        if dist_after_drive_mm < c.warning_distance_mm:
            gap = (c.warning_distance_mm - max(0.0, dist_after_drive_mm)) / c.warning_distance_mm
            proximity = gap * gap

        reward = (
            c.w_progress * progress
            - c.w_turn_cost * turn_cost
            - c.w_action_delta * action_delta
            - c.w_proximity * proximity
        )

        if drive_blocked:
            reward -= c.blocked_penalty
        if collided:
            reward -= c.collision_penalty

        return float(reward)

    def run_episode(
        self,
        policy: ActiveSensingPolicy,
        start_state: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        x, y, yaw = start_state
        prev_action = {"rotate1_deg": 0.0, "rotate2_deg": 0.0, "drive_mm": 0.0}
        obs_history: List[np.ndarray] = []

        episode_reward = 0.0
        total_executed_drive = 0.0
        total_turn = 0.0
        blocked_count = 0
        min_obstacle = float("inf")
        collided = False
        trajectory: List[Dict[str, Any]] = []

        for step_idx in range(self.cfg.max_steps):
            meas = self.simulator.get_sonar_measurement(x, y, yaw)
            dist_mm = safe_float(meas.get("distance_mm"), default=5000.0)
            iid_db = safe_float(meas.get("iid_db"), default=0.0)
            min_obstacle = min(min_obstacle, dist_mm)

            obs = self._normalize_obs(dist_mm, iid_db, prev_action)
            obs_history.append(obs)
            policy_in = self._history_to_input(obs_history, policy.history_length)

            with torch.no_grad():
                out = policy(torch.from_numpy(policy_in).unsqueeze(0))

            action = {
                "rotate1_deg": safe_float(out["rotate1_deg"].item()),
                "rotate2_deg": safe_float(out["rotate2_deg"].item()),
                "drive_mm": safe_float(out["drive_mm"].item()),
            }

            sim_step = self.simulator.simulate_robot_movement(x, y, yaw, [action])[0]

            next_x = safe_float(sim_step["position"]["x"], default=x)
            next_y = safe_float(sim_step["position"]["y"], default=y)
            next_yaw = safe_float(sim_step["orientation"], default=yaw)

            movement = sim_step.get("movement", {})
            executed_drive = safe_float(movement.get("executed_drive_mm"), default=np.hypot(next_x - x, next_y - y))
            total_executed_drive += executed_drive
            total_turn += abs(action["rotate1_deg"]) + abs(action["rotate2_deg"])

            collision_info = sim_step.get("collision", {})
            drive_blocked = bool(collision_info.get("drive_blocked", False))
            if drive_blocked:
                blocked_count += 1
            else:
                blocked_count = 0

            meas_after = sim_step.get("measurements", {}).get("after_drive", {})
            dist_after = safe_float(meas_after.get("distance_mm"), default=dist_mm)
            min_obstacle = min(min_obstacle, dist_after)

            collided = dist_after < self.cfg.collision_distance_mm

            r = self._step_reward(
                action=action,
                prev_action=prev_action,
                executed_drive_mm=executed_drive,
                dist_after_drive_mm=dist_after,
                drive_blocked=drive_blocked,
                collided=collided,
            )
            episode_reward += r

            trajectory.append(
                {
                    "step": step_idx,
                    "x": next_x,
                    "y": next_y,
                    "yaw_deg": next_yaw,
                    "action": action,
                    "executed_drive_mm": executed_drive,
                    "drive_blocked": drive_blocked,
                    "distance_after_drive_mm": dist_after,
                    "reward": r,
                }
            )

            x, y, yaw = next_x, next_y, next_yaw
            prev_action = action

            # Terminal conditions
            if collided:
                break
            if blocked_count >= 3:
                break

        steps_taken = len(trajectory)
        return {
            "fitness": float(episode_reward),
            "steps_taken": steps_taken,
            "total_executed_drive_mm": float(total_executed_drive),
            "total_rotation_deg": float(total_turn),
            "min_obstacle_distance_mm": float(min_obstacle),
            "collided": bool(collided),
            "trajectory": trajectory,
        }

    def evaluate_policy(
        self,
        policy: ActiveSensingPolicy,
        starts: List[Tuple[float, float, float]],
    ) -> Dict[str, Any]:
        runs = [self.run_episode(policy, s) for s in starts]
        fitnesses = np.array([r["fitness"] for r in runs], dtype=np.float32)

        # Aggregate for robust policy ranking.
        result = {
            "fitness": float(np.mean(fitnesses)),
            "fitness_std": float(np.std(fitnesses)),
            "episodes": len(runs),
            "avg_steps": float(np.mean([r["steps_taken"] for r in runs])),
            "avg_executed_drive_mm": float(np.mean([r["total_executed_drive_mm"] for r in runs])),
            "avg_rotation_deg": float(np.mean([r["total_rotation_deg"] for r in runs])),
            "avg_min_obstacle_distance_mm": float(np.mean([r["min_obstacle_distance_mm"] for r in runs])),
            "collision_rate": float(np.mean([1.0 if r["collided"] else 0.0 for r in runs])),
            "episodes_raw": runs,
        }
        return result


# -----------------------------
# Parallel worker state
# -----------------------------
_WORKER_EVALUATOR: Optional[PolicyEvaluator] = None
_WORKER_CFG: Optional[TrainConfig] = None


def _init_eval_worker(cfg_dict: Dict[str, Any]) -> None:
    global _WORKER_EVALUATOR, _WORKER_CFG
    _WORKER_CFG = train_config_from_dict(cfg_dict)
    sim = build_simulator(quiet_setup=_WORKER_CFG.quiet_worker_setup)
    _WORKER_EVALUATOR = PolicyEvaluator(sim, _WORKER_CFG)


def _evaluate_genome_worker(genome: np.ndarray, starts: List[Tuple[float, float, float]]) -> Dict[str, Any]:
    if _WORKER_EVALUATOR is None or _WORKER_CFG is None:
        raise RuntimeError("Worker evaluator is not initialized")
    policy = ActiveSensingPolicy(
        history_length=_WORKER_CFG.history_length,
        hidden_sizes=_WORKER_CFG.hidden_sizes,
        max_rotate_deg=_WORKER_CFG.max_rotate_deg,
        max_drive_mm=_WORKER_CFG.max_drive_mm,
    )
    policy.set_genome(genome)
    return _WORKER_EVALUATOR.evaluate_policy(policy, starts)


# -----------------------------
# GA Trainer
# -----------------------------
class GeneticAlgorithmTrainer:
    def __init__(self, simulator: EnvironmentSimulator, cfg: TrainConfig):
        self.simulator = simulator
        self.cfg = cfg
        self.evaluator = PolicyEvaluator(simulator, cfg)
        self.generation_plot_dir = os.path.join(self.cfg.output_dir, "generation_best")

        self.best_policy: Optional[ActiveSensingPolicy] = None
        self.best_fitness = -float("inf")
        self.history: Dict[str, List[float]] = {
            "best_fitness": [],
            "avg_fitness": [],
            "std_fitness": [],
            "best_collision_rate": [],
            "best_avg_executed_drive_mm": [],
        }

    def _new_policy(self) -> ActiveSensingPolicy:
        return ActiveSensingPolicy(
            history_length=self.cfg.history_length,
            hidden_sizes=self.cfg.hidden_sizes,
            max_rotate_deg=self.cfg.max_rotate_deg,
            max_drive_mm=self.cfg.max_drive_mm,
        )

    def create_initial_population(self) -> List[ActiveSensingPolicy]:
        return [self._new_policy() for _ in range(self.cfg.population_size)]

    def _episode_starts_for_generation(self, generation_idx: int) -> List[Tuple[float, float, float]]:
        # Same starts for all policies in generation -> fairer selection.
        rng = random.Random(self.cfg.seed + 10_000 * (generation_idx + 1))
        return [self.evaluator._sample_start_state(rng) for _ in range(self.cfg.episodes_per_policy)]

    def evaluate_population(
        self, population: List[ActiveSensingPolicy], generation_idx: int
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        starts = self._episode_starts_for_generation(generation_idx)
        scores: List[float] = [0.0] * len(population)
        details: List[Dict[str, Any]] = [None] * len(population)  # type: ignore

        use_parallel = self.cfg.parallel_eval and len(population) > 1
        if use_parallel:
            workers = self.cfg.num_workers or max(1, min(os.cpu_count() or 1, 8))
            genomes = [p.get_genome() for p in population]
            cfg_dict = asdict(self.cfg)
            try:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_eval_worker,
                    initargs=(cfg_dict,),
                ) as ex:
                    results_iter = ex.map(_evaluate_genome_worker, genomes, [starts] * len(genomes))
                    for i, result in enumerate(tqdm(results_iter, total=len(population), desc="Evaluating population")):
                        score = float(result["fitness"])
                        scores[i] = score
                        details[i] = result
                        if score > self.best_fitness:
                            self.best_fitness = score
                            self.best_policy = self._new_policy()
                            self.best_policy.load_state_dict(population[i].state_dict())
            except Exception as e:
                print(f"Parallel evaluation failed ({type(e).__name__}: {e}); falling back to serial.")
                use_parallel = False
        if not use_parallel:
            for i, policy in enumerate(tqdm(population, desc="Evaluating population")):
                result = self.evaluator.evaluate_policy(policy, starts)
                score = float(result["fitness"])
                scores[i] = score
                details[i] = result
                if score > self.best_fitness:
                    self.best_fitness = score
                    self.best_policy = self._new_policy()
                    self.best_policy.load_state_dict(policy.state_dict())

        return scores, details

    def _save_generation_best_plot(self, generation_number: int, detail: Dict[str, Any]) -> None:
        if not self.cfg.save_generation_best_plots:
            return
        episodes = detail.get("episodes_raw", [])
        if not episodes:
            return
        os.makedirs(self.generation_plot_dir, exist_ok=True)
        best_idx = int(np.argmax([ep["fitness"] for ep in episodes]))
        ep = episodes[best_idx]
        output_path = os.path.join(self.generation_plot_dir, f"gen_{generation_number:03d}_best.png")
        title = (
            f"Generation {generation_number} Best | "
            f"fitness={detail.get('fitness', float('nan')):.2f}, "
            f"collision_rate={detail.get('collision_rate', float('nan')):.2f}"
        )
        plot_episode_trajectory(self.simulator, ep, output_path, title)

    def select_parents(
        self, population: List[ActiveSensingPolicy], fitness_scores: List[float]
    ) -> Tuple[ActiveSensingPolicy, ActiveSensingPolicy]:
        k = min(self.cfg.tournament_size, len(population))

        idxs1 = random.sample(range(len(population)), k)
        idxs2 = random.sample(range(len(population)), k)

        p1 = max(idxs1, key=lambda i: fitness_scores[i])
        p2 = max(idxs2, key=lambda i: fitness_scores[i])
        return population[p1], population[p2]

    def crossover(self, p1: ActiveSensingPolicy, p2: ActiveSensingPolicy) -> ActiveSensingPolicy:
        child = self._new_policy()
        if random.random() > self.cfg.crossover_rate:
            child.load_state_dict(p1.state_dict() if random.random() < 0.5 else p2.state_dict())
            return child

        g1 = p1.get_genome()
        g2 = p2.get_genome()
        alpha = np.random.rand(g1.size).astype(np.float32)
        mask = (np.random.rand(g1.size) < 0.5).astype(np.float32)

        # Mix with both uniform inheritance and small blend.
        inherited = np.where(mask > 0.5, g1, g2)
        blended = alpha * g1 + (1.0 - alpha) * g2
        child_genome = 0.7 * inherited + 0.3 * blended

        child.set_genome(child_genome)
        return child

    def mutate(self, policy: ActiveSensingPolicy) -> ActiveSensingPolicy:
        out = self._new_policy()
        out.load_state_dict(policy.state_dict())

        g = out.get_genome()
        m = (np.random.rand(g.size) < self.cfg.mutation_rate)
        noise = np.random.normal(0.0, self.cfg.mutation_sigma, size=g.size).astype(np.float32)
        g = g + m.astype(np.float32) * noise

        out.set_genome(g)
        return out

    def create_next_generation(
        self, population: List[ActiveSensingPolicy], fitness_scores: List[float]
    ) -> List[ActiveSensingPolicy]:
        order = np.argsort(np.asarray(fitness_scores))[::-1]

        next_population: List[ActiveSensingPolicy] = []
        elite_n = min(self.cfg.elitism_count, len(population))
        for i in order[:elite_n]:
            elite = self._new_policy()
            elite.load_state_dict(population[int(i)].state_dict())
            next_population.append(elite)

        while len(next_population) < self.cfg.population_size:
            p1, p2 = self.select_parents(population, fitness_scores)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            next_population.append(child)

        return next_population

    def train(self) -> ActiveSensingPolicy:
        population = self.create_initial_population()

        for gen in range(self.cfg.generations):
            print(f"\n=== Generation {gen + 1}/{self.cfg.generations} ===")
            scores, details = self.evaluate_population(population, generation_idx=gen)

            scores_np = np.asarray(scores, dtype=np.float32)
            best_idx = int(np.argmax(scores_np))
            best_detail = details[best_idx]

            self.history["best_fitness"].append(float(np.max(scores_np)))
            self.history["avg_fitness"].append(float(np.mean(scores_np)))
            self.history["std_fitness"].append(float(np.std(scores_np)))
            self.history["best_collision_rate"].append(float(best_detail["collision_rate"]))
            self.history["best_avg_executed_drive_mm"].append(float(best_detail["avg_executed_drive_mm"]))

            print(
                f"Fitness: best={self.history['best_fitness'][-1]:.3f}, "
                f"avg={self.history['avg_fitness'][-1]:.3f}, "
                f"std={self.history['std_fitness'][-1]:.3f}"
            )
            print(
                f"Best policy: collision_rate={best_detail['collision_rate']:.3f}, "
                f"avg_executed_drive={best_detail['avg_executed_drive_mm']:.1f}mm, "
                f"avg_steps={best_detail['avg_steps']:.1f}"
            )
            self._save_generation_best_plot(gen + 1, best_detail)

            if gen < self.cfg.generations - 1:
                population = self.create_next_generation(population, scores)

        if self.best_policy is None:
            raise RuntimeError("Training finished without a best policy")
        return self.best_policy

    def save_results(self) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        if self.best_policy is None:
            raise RuntimeError("No best policy to save")

        policy_path = os.path.join(self.cfg.output_dir, "best_policy.pth")
        torch.save(
            {
                "state_dict": self.best_policy.state_dict(),
                "history_length": self.cfg.history_length,
                "hidden_sizes": list(self.cfg.hidden_sizes),
                "max_rotate_deg": self.cfg.max_rotate_deg,
                "max_drive_mm": self.cfg.max_drive_mm,
            },
            policy_path,
        )

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.cfg),
            "best_fitness": self.best_fitness,
            "history": self.history,
        }
        metrics_path = os.path.join(self.cfg.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        self._save_training_plots()

        print(f"Saved best policy to {policy_path}")
        print(f"Saved training metrics to {metrics_path}")

    def _save_training_plots(self) -> None:
        generations = np.arange(1, len(self.history["best_fitness"]) + 1)
        if generations.size == 0:
            return

        # Fitness progression
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(generations, self.history["best_fitness"], label="Best fitness", linewidth=2)
        ax.plot(generations, self.history["avg_fitness"], label="Average fitness", linewidth=2)
        ax.fill_between(
            generations,
            np.array(self.history["avg_fitness"]) - np.array(self.history["std_fitness"]),
            np.array(self.history["avg_fitness"]) + np.array(self.history["std_fitness"]),
            alpha=0.2,
            label="Avg ± std",
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("GA Training Progress")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.cfg.output_dir, "plot_fitness_progress.png"), dpi=180)
        plt.close(fig)

        # Safety/behavior metrics
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(
            generations,
            self.history["best_collision_rate"],
            color="#c62828",
            linewidth=2,
            label="Best policy collision rate",
        )
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Collision rate", color="#c62828")
        ax1.tick_params(axis="y", labelcolor="#c62828")
        ax1.set_ylim(-0.02, 1.02)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(
            generations,
            self.history["best_avg_executed_drive_mm"],
            color="#1565c0",
            linewidth=2,
            label="Best policy avg executed drive (mm)",
        )
        ax2.set_ylabel("Executed drive per episode (mm)", color="#1565c0")
        ax2.tick_params(axis="y", labelcolor="#1565c0")

        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")

        fig.tight_layout()
        fig.savefig(os.path.join(self.cfg.output_dir, "plot_safety_progress.png"), dpi=180)
        plt.close(fig)


def plot_episode_trajectory(
    simulator: EnvironmentSimulator,
    episode: Dict[str, Any],
    output_path: str,
    title: str,
) -> None:
    traj = episode.get("trajectory", [])
    if not traj:
        return

    xs = [s["x"] for s in traj]
    ys = [s["y"] for s in traj]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw arena walls if available
    walls = getattr(simulator.arena, "walls", None)
    if walls is not None and len(walls) > 0:
        ax.scatter(walls[:, 0], walls[:, 1], s=1, alpha=0.25, color="#9e9e9e", label="Walls")

    # Trajectory
    ax.plot(xs, ys, "-", linewidth=2, color="#1565c0", label="Trajectory")
    ax.scatter(xs[0], ys[0], color="#2e7d32", s=60, marker="o", label="Start")
    ax.scatter(xs[-1], ys[-1], color="#c62828", s=60, marker="x", label="End")

    # Auto-fit to actual coordinate extents (supports negative arena coordinates).
    all_x = list(xs)
    all_y = list(ys)
    if walls is not None and len(walls) > 0:
        all_x.extend(walls[:, 0].tolist())
        all_y.extend(walls[:, 1].tolist())

    if all_x and all_y:
        min_x = float(np.min(all_x))
        max_x = float(np.max(all_x))
        min_y = float(np.min(all_y))
        max_y = float(np.max(all_y))
        pad_x = max(30.0, 0.05 * max(1.0, max_x - min_x))
        pad_y = max(30.0, 0.05 * max(1.0, max_y - min_y))
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# -----------------------------
# Entrypoint
# -----------------------------
def build_simulator(quiet_setup: bool = False) -> EnvironmentSimulator:
    if quiet_setup:
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                return create_test_simulator()
            except Exception:
                return EnvironmentSimulator("_default_")

    print("Creating environment simulator...")
    try:
        sim = create_test_simulator()
        print("Simulator created successfully")
        return sim
    except Exception as e:
        print(f"Primary simulator creation failed: {e}")
        print("Falling back to default arena simulator...")
        return EnvironmentSimulator("_default_")


def main() -> None:
    cfg = TrainConfig()
    set_global_seed(cfg.seed)

    print("Active Sensing Policy Learning - Genetic Algorithm")
    print("=" * 60)
    overview_path = write_run_overview_md(cfg, cfg.output_dir)
    print(f"Run overview written to {overview_path}")

    simulator = build_simulator(quiet_setup=cfg.quiet_main_setup)
    trainer = GeneticAlgorithmTrainer(simulator, cfg)

    t0 = time.time()
    best_policy = trainer.train()
    elapsed = time.time() - t0

    print(f"\nTraining completed in {elapsed:.1f} seconds")
    trainer.save_results()

    # Quick post-train evaluation
    evaluator = PolicyEvaluator(simulator, cfg)
    rng = random.Random(cfg.seed + 999_999)
    starts = [evaluator._sample_start_state(rng) for _ in range(cfg.episodes_per_policy)]
    final = evaluator.evaluate_policy(best_policy, starts)

    print("\nBest policy evaluation:")
    print(f"  Fitness: {final['fitness']:.3f} ± {final['fitness_std']:.3f}")
    print(f"  Avg executed drive: {final['avg_executed_drive_mm']:.1f} mm")
    print(f"  Avg rotation: {final['avg_rotation_deg']:.1f} deg")
    print(f"  Collision rate: {final['collision_rate']:.3f}")
    print(f"  Avg steps: {final['avg_steps']:.1f}")

    # Save evaluation trajectory plots
    os.makedirs(cfg.output_dir, exist_ok=True)
    episodes = final.get("episodes_raw", [])
    for i, ep in enumerate(episodes):
        plot_episode_trajectory(
            simulator=simulator,
            episode=ep,
            output_path=os.path.join(cfg.output_dir, f"plot_eval_episode_{i+1:02d}.png"),
            title=f"Best Policy Evaluation Episode {i+1} | fitness={ep['fitness']:.2f}",
        )

    if episodes:
        best_idx = int(np.argmax([ep["fitness"] for ep in episodes]))
        plot_episode_trajectory(
            simulator=simulator,
            episode=episodes[best_idx],
            output_path=os.path.join(cfg.output_dir, "plot_eval_best_episode.png"),
            title=f"Best Policy - Best Episode (#{best_idx + 1})",
        )


if __name__ == "__main__":
    main()
