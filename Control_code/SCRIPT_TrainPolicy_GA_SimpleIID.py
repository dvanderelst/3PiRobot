#!/usr/bin/env python3
"""
Minimal GA sanity test for IID steering policy.

Design constraints for this test:
- rotate1 is fixed to 0 deg
- history length is 1 (current measurement only)
- drive distance is fixed to 100 mm
- policy controls rotate2 from current IID sign + distance-based magnitude

Goal:
Learn the hand-coded behavior: if IID indicates left is louder, turn right,
and vice versa, with stronger turns at short distance.
"""

import csv
import json
import os
import random
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
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
    session_name: str = "sessionB03"

    # Fixed action components
    fixed_rotate1_deg: float = 0.0
    fixed_drive_mm: float = 100.0
    max_rotate2_deg: float = 90.0
    iid_deadband_db: float = 0.25

    # GA
    population_size: int = 80
    generations: int = 80
    elitism_count: int = 8
    tournament_size: int = 5
    crossover_rate: float = 0.85
    mutation_rate: float = 0.2
    mutation_sigma: float = 0.2

    # Evaluation
    episodes_per_policy: int = 8
    max_steps: int = 80
    spawn_margin_mm: float = 150.0
    use_empirical_starts: bool = True

    # Reward weights (minimal objective)
    step_reward: float = 1.0
    w_rotate2_cost: float = 1
    # Proximity shaping: penalize being close to geometry (walls/bounds).
    warning_distance_mm: float = 500.0
    collision_distance_mm: float = 250.0
    w_proximity_cost: float = 0.5

    # IO
    output_dir: str = "policy_training_results_simple_iid"
    quiet_setup: bool = True
    save_generation_best_plots: bool = True
    parallel_eval: bool = True
    num_workers: Optional[int] = 8


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


class IIDSignDistancePolicy:
    """
    Tiny policy:
        turn direction from IID sign only
        turn magnitude from distance only (piecewise-linear between far/near)
        if |iid_db| < deadband: rotate2 = 0
    Genome = [near_turn_deg, far_turn_deg], both >= 0
    """

    def __init__(self, max_rotate2_deg: float, deadband_db: float = 0.25):
        self.max_rotate2_deg = float(max_rotate2_deg)
        self.deadband_db = float(deadband_db)
        self.near_turn_deg = 45.0
        self.far_turn_deg = 8.0
        self.distance_near_mm = 280.0
        self.distance_far_mm = 1800.0

    def set_genome(self, genome: np.ndarray) -> None:
        g = np.asarray(genome, dtype=np.float32).reshape(-1)
        if g.size != 2:
            raise ValueError("IIDSignDistancePolicy genome must have length 2")
        self.near_turn_deg = max(0.0, float(g[0]))
        self.far_turn_deg = max(0.0, float(g[1]))

    def get_genome(self) -> np.ndarray:
        return np.array([self.near_turn_deg, self.far_turn_deg], dtype=np.float32)

    def _turn_magnitude_deg(self, distance_mm: float) -> float:
        d = safe_float(distance_mm, self.distance_far_mm)
        if self.distance_far_mm <= self.distance_near_mm:
            return float(np.clip(self.near_turn_deg, 0.0, self.max_rotate2_deg))
        # t=1 at near distance, t=0 at far distance.
        t = (self.distance_far_mm - d) / (self.distance_far_mm - self.distance_near_mm)
        t = float(np.clip(t, 0.0, 1.0))
        mag = self.far_turn_deg + t * (self.near_turn_deg - self.far_turn_deg)
        return float(np.clip(mag, 0.0, self.max_rotate2_deg))

    def rotate2_deg(self, iid_db: float, distance_mm: float) -> float:
        iid = safe_float(iid_db, 0.0)
        if abs(iid) < self.deadband_db:
            return 0.0
        mag = self._turn_magnitude_deg(distance_mm)
        rot = -np.sign(iid) * mag
        return float(np.clip(rot, -self.max_rotate2_deg, self.max_rotate2_deg))


def config_from_dict(d: Dict[str, Any]) -> Config:
    return Config(**dict(d))


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
        if self.cfg.use_empirical_starts and self.empirical_starts:
            return self.empirical_starts[rng.randrange(len(self.empirical_starts))]

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

    def episode(self, policy: IIDSignDistancePolicy, start: Tuple[float, float, float]) -> Dict[str, Any]:
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

        for t in range(self.cfg.max_steps):
            meas = self.sim.get_sonar_measurement(x, y, yaw)
            iid = safe_float(meas.get("iid_db"), 0.0)

            dist_mm = safe_float(meas.get("distance_mm"), 1800.0)
            rotate2 = policy.rotate2_deg(iid, dist_mm)
            action = {
                "rotate1_deg": self.cfg.fixed_rotate1_deg,
                "rotate2_deg": rotate2,
                "drive_mm": self.cfg.fixed_drive_mm,
            }

            step = self.sim.simulate_robot_movement(x, y, yaw, [action])[0]
            nx = safe_float(step["position"]["x"], x)
            ny = safe_float(step["position"]["y"], y)
            nyaw = safe_float(step["orientation"], yaw)

            move = step.get("movement", {})
            exec_drive = safe_float(move.get("executed_drive_mm"), np.hypot(nx - x, ny - y))
            total_drive += exec_drive

            coll = step.get("collision", {})
            blocked = bool(coll.get("drive_blocked", False))
            blocked_any = blocked_any or blocked
            iid_norm = float(np.clip(iid / 12.0, -2.0, 2.0))
            rot_norm = float(np.clip(rotate2 / self.cfg.max_rotate2_deg, -1.0, 1.0))
            clearance_mm = self._geometry_clearance_mm(nx, ny)
            warn = max(float(self.cfg.warning_distance_mm), 1e-6)
            proximity_term = float(np.clip((warn - clearance_mm) / warn, 0.0, 1.0))
            proximity_terms.append(proximity_term)

            # Diagnostic only (not part of fitness): positive when rotate2 opposes iid sign/magnitude.
            align_term = -iid_norm * rot_norm
            aligned_terms.append(align_term)

            if abs(iid_norm) > 0.15 and abs(rot_norm) > 0.05:
                sign_match_terms.append(1.0 if np.sign(rot_norm) == -np.sign(iid_norm) else 0.0)

            reward = (
                (self.cfg.step_reward if not blocked else 0.0)
                - self.cfg.w_rotate2_cost * abs(rot_norm)
                - self.cfg.w_proximity_cost * proximity_term
            )
            total_reward += reward

            trajectory.append(
                {
                    "step": t,
                    "x": nx,
                    "y": ny,
                    "yaw_deg": nyaw,
                    "iid_db": iid,
                    "distance_mm": dist_mm,
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
        return {
            "fitness": float(total_reward),
            "steps": len(trajectory),
            "total_executed_drive_mm": float(total_drive),
            "net_displacement_mm": net_displacement,
            "collided": bool(blocked_any),
            "end_reason": end_reason,
            "proximity_mean": float(np.mean(proximity_terms) if proximity_terms else 0.0),
            "alignment_mean": float(np.mean(aligned_terms) if aligned_terms else 0.0),
            "sign_match_rate": sign_match_rate,
            "trajectory": trajectory,
        }

    def evaluate(self, policy: IIDSignDistancePolicy, starts: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        eps = [self.episode(policy, s) for s in starts]
        fit = np.array([e["fitness"] for e in eps], dtype=np.float32)
        return {
            "fitness": float(np.mean(fit)),
            "fitness_std": float(np.std(fit)),
            "collision_rate": float(np.mean([1.0 if e["collided"] else 0.0 for e in eps])),
            "proximity_mean": float(np.mean([e.get("proximity_mean", 0.0) for e in eps])),
            "alignment_mean": float(np.mean([e["alignment_mean"] for e in eps])),
            "sign_match_rate": float(np.mean([e["sign_match_rate"] for e in eps])),
            "avg_drive_mm": float(np.mean([e["total_executed_drive_mm"] for e in eps])),
            "avg_net_displacement_mm": float(np.mean([e["net_displacement_mm"] for e in eps])),
            "episodes_raw": eps,
        }


_WORKER_CFG: Optional[Config] = None
_WORKER_EV: Optional[Evaluator] = None


def _init_worker(cfg_dict: Dict[str, Any]) -> None:
    global _WORKER_CFG, _WORKER_EV
    _WORKER_CFG = config_from_dict(cfg_dict)
    sim = build_simulator(_WORKER_CFG.session_name, quiet_setup=_WORKER_CFG.quiet_setup)
    _WORKER_EV = Evaluator(sim, _WORKER_CFG)


def _eval_genome_worker(genome: np.ndarray, starts: List[Tuple[float, float, float]]) -> Dict[str, Any]:
    if _WORKER_CFG is None or _WORKER_EV is None:
        raise RuntimeError("Worker not initialized")
    pol = IIDSignDistancePolicy(_WORKER_CFG.max_rotate2_deg, _WORKER_CFG.iid_deadband_db)
    pol.set_genome(genome)
    return _WORKER_EV.evaluate(pol, starts)


class SimpleGATrainer:
    def __init__(self, evaluator: Evaluator, cfg: Config):
        self.ev = evaluator
        self.cfg = cfg
        self.sim = evaluator.sim
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
        }

    def _make_policy(self, genome: np.ndarray) -> IIDSignDistancePolicy:
        p = IIDSignDistancePolicy(self.cfg.max_rotate2_deg, self.cfg.iid_deadband_db)
        p.set_genome(genome)
        return p

    def init_population(self) -> List[np.ndarray]:
        # [near_turn_deg, far_turn_deg], both in degrees.
        pop: List[np.ndarray] = []
        for _ in range(self.cfg.population_size):
            near = np.random.uniform(20.0, self.cfg.max_rotate2_deg)
            far = np.random.uniform(0.0, 25.0)
            pop.append(np.array([near, far], dtype=np.float32))
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
        return np.clip(child, 0.0, self.cfg.max_rotate2_deg).astype(np.float32)

    def mutate(self, g: np.ndarray) -> np.ndarray:
        mask = (np.random.rand(*g.shape) < self.cfg.mutation_rate).astype(np.float32)
        noise = np.random.normal(0.0, self.cfg.mutation_sigma, size=g.shape).astype(np.float32)
        out = g + mask * noise
        return np.clip(out, 0.0, self.cfg.max_rotate2_deg).astype(np.float32)

    def _generation_starts(self, gen: int) -> List[Tuple[float, float, float]]:
        rng = random.Random(self.cfg.seed + 10000 * (gen + 1))
        return [self.ev.sample_start(rng) for _ in range(self.cfg.episodes_per_policy)]

    def train(self) -> np.ndarray:
        pop = self.init_population()
        for gen in range(self.cfg.generations):
            starts = self._generation_starts(gen)
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
                        results = ex.map(_eval_genome_worker, pop, [starts] * len(pop))
                        for i, res in enumerate(tqdm(results, total=len(pop), desc=f"Gen {gen+1}/{self.cfg.generations}")):
                            fitness[i] = float(res["fitness"])
                            details[i] = res
                except Exception as e:
                    print(f"Parallel eval failed ({type(e).__name__}: {e}); using serial.")
                    use_parallel = False

            if not use_parallel:
                for i, g in enumerate(tqdm(pop, desc=f"Gen {gen+1}/{self.cfg.generations}")):
                    res = self.ev.evaluate(self._make_policy(g), starts)
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

            print(
                f"Gen {gen+1}: best={self.history['best_fitness'][-1]:.3f}, "
                f"avg={self.history['avg_fitness'][-1]:.3f}, "
                f"best_ep={best_ep_fit:.3f}, med_ep={median_ep_fit:.3f}, worst_ep={worst_ep_fit:.3f}, "
                f"align={best_res['alignment_mean']:.3f}, "
                f"sign_match={best_res['sign_match_rate']:.3f}, "
                f"coll_rate={best_res['collision_rate']:.3f}, "
                f"prox={best_res.get('proximity_mean', 0.0):.3f}, "
                f"net_disp={best_res['avg_net_displacement_mm']:.1f}mm"
            )
            self._save_generation_best_plot(gen + 1, best_res)

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

    def _save_generation_best_plot(self, generation_number: int, detail: Dict[str, Any]) -> None:
        if not self.cfg.save_generation_best_plots:
            return
        episodes = detail.get("episodes_raw", [])
        if not episodes:
            return
        os.makedirs(self.generation_plot_dir, exist_ok=True)
        fitnesses = np.array([safe_float(ep.get("fitness"), float("-inf")) for ep in episodes], dtype=np.float64)
        order = np.argsort(fitnesses)
        best_idx = int(order[-1])
        median_idx = int(order[len(order) // 2])
        worst_idx = int(order[0])

        selections = [("Best", best_idx), ("Median", median_idx), ("Worst", worst_idx)]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        policy_fit = safe_float(detail.get("fitness"), float("nan"))
        for ax, (label, idx) in zip(axes, selections):
            ep = episodes[idx]
            ep_fit = safe_float(ep.get("fitness"), float("nan"))
            end_reason = str(ep.get("end_reason", "unknown"))
            panel_title = f"{label} | ep_fit={ep_fit:.2f} | end={end_reason}"
            draw_episode_on_axis(self.sim, ep, ax, panel_title, show_legend=(label == "Best"))
        fig.suptitle(f"Gen {generation_number} | policy_fit={policy_fit:.2f}", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        output_path = os.path.join(self.generation_plot_dir, f"gen_{generation_number:03d}_triplet.png")
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


def plot_training(history: Dict[str, List[float]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    g = np.arange(1, len(history["best_fitness"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(g, history["best_fitness"], label="Best fitness")
    ax.plot(g, history["avg_fitness"], label="Average fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Simple IID GA Training")
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


def plot_policy_curve(policy: IIDSignDistancePolicy, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    iid = np.linspace(-12, 12, 300)
    dists = [300.0, 700.0, 1200.0, 1800.0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in dists:
        rot = np.array([policy.rotate2_deg(v, d) for v in iid], dtype=np.float32)
        ax.plot(iid, rot, linewidth=2, label=f"distance={int(d)}mm")
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("IID (dB)")
    ax.set_ylabel("rotate2 (deg)")
    ax.set_title("Learned (IID sign + distance magnitude) -> rotate2 mapping")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot_policy_curve.png"), dpi=180)
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
        "# Simple IID GA Test",
        "",
        "Controller:",
        "- rotate1 fixed to 0 deg",
        "- drive fixed to 100 mm",
        "- policy uses IID sign for turn direction and distance for turn magnitude",
        "- rotate2 = -sign(iid_db) * magnitude(distance_mm), with IID deadband",
        "- magnitude(distance) linearly interpolates between far_turn_deg and near_turn_deg",
        "- genome size: 2 parameters (`near_turn_deg`, `far_turn_deg`), both >= 0",
        "- fitness: +step_reward per non-blocked step, minus rotate2 cost and proximity cost",
        "- run ends if geometric clearance <= collision_distance_mm",
        "",
        f"- population: {cfg.population_size}",
        f"- generations: {cfg.generations}",
        f"- episodes per policy: {cfg.episodes_per_policy}",
        f"- max steps per episode: {cfg.max_steps}",
        f"- session: {cfg.session_name}",
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
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    write_overview(cfg, cfg.output_dir)

    print("Simple IID GA sanity test")
    print("=" * 60)
    print("Policy: rotate1=0, drive=100mm, direction from IID sign, magnitude from distance.")
    print(f"Session: {cfg.session_name}")

    sim = build_simulator(cfg.session_name, quiet_setup=cfg.quiet_setup)
    ev = Evaluator(sim, cfg)
    trainer = SimpleGATrainer(ev, cfg)

    best_genome = trainer.train()
    best_policy = IIDSignDistancePolicy(cfg.max_rotate2_deg, cfg.iid_deadband_db)
    best_policy.set_genome(best_genome)

    # Final deterministic evaluation
    rng = random.Random(cfg.seed + 999_999)
    starts = [ev.sample_start(rng) for _ in range(cfg.episodes_per_policy)]
    final = ev.evaluate(best_policy, starts)

    # Save artifacts
    with open(os.path.join(cfg.output_dir, "best_policy.json"), "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "policy_type": "IIDSignDistancePolicy",
                "genome": best_genome.tolist(),
                "near_turn_deg": float(best_genome[0]),
                "far_turn_deg": float(best_genome[1]),
                "iid_deadband_db": cfg.iid_deadband_db,
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
    plot_policy_curve(best_policy, cfg.output_dir)

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
            sim,
            ep,
            os.path.join(cfg.output_dir, f"plot_eval_episode_{i+1:02d}.png"),
            f"Simple IID Eval Episode {i+1} | fitness={ep['fitness']:.2f}",
        )
    if eps:
        fitnesses = np.array([safe_float(ep.get("fitness"), float("-inf")) for ep in eps], dtype=np.float64)
        best_idx = int(np.argmax(fitnesses))
        median_idx = int(np.argsort(fitnesses)[len(fitnesses) // 2])
        worst_idx = int(np.argmin(fitnesses))

        plot_episode(
            sim,
            eps[best_idx],
            os.path.join(cfg.output_dir, "plot_eval_best_episode.png"),
            f"Simple IID Best Episode #{best_idx+1}",
        )
        plot_episode(
            sim,
            eps[median_idx],
            os.path.join(cfg.output_dir, "plot_eval_median_episode.png"),
            f"Simple IID Median Episode #{median_idx+1}",
        )
        plot_episode(
            sim,
            eps[worst_idx],
            os.path.join(cfg.output_dir, "plot_eval_worst_episode.png"),
            f"Simple IID Worst Episode #{worst_idx+1}",
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
