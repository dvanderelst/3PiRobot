#!/usr/bin/env python3
"""
SCRIPT_AssessCollisions.py

Runs a large number of episodes with the best policy from a single training
run and analyses where crashes occur in each arena.

The episode runner mirrors run_episode() in SCRIPT_TrainPolicy.py exactly
(same _get_measurement, distance clamping, IID symmetry wrapper, net-rotation
clamp, force_aligned handling).

Episodes are dispatched in parallel across all sessions using
ProcessPoolExecutor (same initializer pattern as SCRIPT_TrainPolicy).

Outputs (to CollisionAssessment/<run_name>/):
  collisions.csv              — one row per episode (crash position if applicable)
  crash_map_<session>.png     — crash locations overlaid on each arena
"""

import collections
import csv
import dataclasses
import json
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, List, Optional, Tuple

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Library.EnvironmentSimulator import EnvironmentSimulator
from SCRIPT_TrainPolicy import (
    Config, MLPPolicy, build_input, load_starts,
    _get_measurement, _apply_net_rotation_clamp,
)


# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════

RUN_DIR                = "PolicyTraining/policy5_h10"
N_EPISODES_PER_SESSION = 500
NUM_WORKERS            = None   # None = os.cpu_count()
SEED                   = 42
N_LAST_STEPS           = 15    # steps to record per episode for emulator accuracy analysis
MIN_CRASH_STEPS        = 11     # exclude crashes that occur within this many steps (memory not yet useful)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_cfg(run_dir: str) -> Config:
    with open(os.path.join(run_dir, "config.json")) as f:
        d = json.load(f)
    valid = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in valid})


def load_best_policy(run_dir: str, cfg: Config) -> Tuple[MLPPolicy, float, int]:
    path = os.path.join(run_dir, "best_policy.json")
    with open(path) as f:
        d = json.load(f)
    pol = MLPPolicy(cfg)
    pol.set_genome(np.array(d["genome"], dtype=np.float32))
    return pol, float(d["fitness"]), int(d.get("generation", -1))


def build_simulator(session_name: str) -> EnvironmentSimulator:
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        return EnvironmentSimulator(session_name)


# ══════════════════════════════════════════════════════════════════════════════
# Parallel worker
# ══════════════════════════════════════════════════════════════════════════════

_WORKER_SIMS:    Optional[Dict[str, EnvironmentSimulator]] = None
_WORKER_STARTS:  Optional[Dict[str, List[Tuple[float, float, float]]]] = None
_WORKER_CFG:     Optional[Config] = None
_WORKER_GEO_CFG: Optional[Config] = None
_WORKER_GENOME:  Optional[np.ndarray] = None


def _init_worker(cfg_dict: dict, genome: np.ndarray, session_names: List[str]) -> None:
    global _WORKER_SIMS, _WORKER_STARTS, _WORKER_CFG, _WORKER_GEO_CFG, _WORKER_GENOME
    cfg = Config(**{k: v for k, v in cfg_dict.items()
                    if k in {f.name for f in dataclasses.fields(Config)}})
    _WORKER_CFG     = cfg
    _WORKER_GEO_CFG = dataclasses.replace(cfg, override_emulator_distance=True,
                                           override_emulator_iid=True)
    _WORKER_GENOME  = genome
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        _WORKER_SIMS   = {sn: EnvironmentSimulator(sn) for sn in session_names}
        _WORKER_STARTS = {sn: load_starts(sn, cfg, quiet=True) for sn in session_names}
    try:
        import torch as _t
        _t.set_num_threads(1)
    except ImportError:
        pass


def _episode_worker(args: Tuple[str, int, int]) -> Tuple[str, int, dict]:
    """Run one episode. Returns (session_name, episode_idx, result_dict)."""
    session_name, episode_idx, seed = args
    rng    = np.random.default_rng(seed)
    policy = MLPPolicy(_WORKER_CFG)
    policy.set_genome(_WORKER_GENOME)
    starts = _WORKER_STARTS[session_name]
    sim    = _WORKER_SIMS[session_name]

    x, y, yaw = starts[int(rng.integers(len(starts)))]
    start_x, start_y, start_yaw = x, y, yaw

    history: collections.deque = collections.deque(
        [(0.0, 0.0, 0.0, 0.0)] * _WORKER_CFG.history_len,
        maxlen=_WORKER_CFG.history_len,
    )
    last_physical_iid = 0.0
    collided = False
    n_steps  = 0
    cfg      = _WORKER_CFG
    geo_cfg  = _WORKER_GEO_CFG
    steps_deque: collections.deque = collections.deque(maxlen=N_LAST_STEPS)

    for _ in range(cfg.max_steps):
        original_yaw = yaw

        if cfg.force_aligned:
            rotate1_canonical = 0.0
            rotate1           = 0.0
        else:
            inp1 = build_input(history, 0.0, 0.0, 0.0, cfg)
            rotate1_canonical = policy.forward(inp1, cfg.max_rotate1_deg)
            flip1   = last_physical_iid < 0.0
            rotate1 = -rotate1_canonical if flip1 else rotate1_canonical
        look_yaw = original_yaw + rotate1

        dist_mm, physical_iid = _get_measurement(sim, x, y, look_yaw, cfg)
        geo_dist_mm, geo_iid  = _get_measurement(sim, x, y, look_yaw, geo_cfg)
        if cfg.iid_noise_db > 0.0:
            physical_iid += float(rng.normal(0.0, cfg.iid_noise_db))

        flip2         = physical_iid < 0.0
        canonical_iid = abs(physical_iid)
        inp2 = build_input(history, dist_mm, canonical_iid, rotate1_canonical, cfg)
        rotate2_canonical = policy.forward(inp2, cfg.max_rotate2_deg)
        rotate2 = -rotate2_canonical if flip2 else rotate2_canonical
        rotate2, rotate2_canonical = _apply_net_rotation_clamp(rotate1, rotate2, flip2, cfg)

        action = {"rotate1_deg": rotate1, "rotate2_deg": rotate2,
                  "drive_mm": cfg.fixed_drive_mm}
        result = sim.simulate_robot_movement(
            x, y, original_yaw, [action], compute_sonar=False
        )[0]

        steps_deque.append({
            "step":         n_steps,
            "emu_dist_mm":  round(dist_mm, 1),
            "emu_iid_db":   round(physical_iid, 3),
            "geo_dist_mm":  round(geo_dist_mm, 1),
            "geo_iid_db":   round(geo_iid, 3),
        })

        x   = float(result["position"]["x"])
        y   = float(result["position"]["y"])
        yaw = float(result["orientation"])
        history.append((dist_mm, canonical_iid, rotate1_canonical, rotate2_canonical))
        last_physical_iid = physical_iid
        n_steps += 1

        if bool(result["collision"]["drive_blocked"]):
            collided = True
            break

    rec = {
        "episode":   episode_idx,
        "n_steps":   n_steps,
        "collided":  int(collided),
        "start_x":   round(start_x, 1),
        "start_y":   round(start_y, 1),
        "start_yaw": round(start_yaw, 1),
    }
    if collided:
        rec["crash_x"]   = round(x, 1)
        rec["crash_y"]   = round(y, 1)
        rec["crash_yaw"] = round(yaw, 1)
    rec["steps"] = list(steps_deque)

    return session_name, episode_idx, rec


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_crash_map(
    results: List[dict],
    simulator: EnvironmentSimulator,
    session_name: str,
    out_path: str,
    policy_fitness: float,
    policy_generation: int,
) -> None:
    crashes  = [r for r in results if r["collided"] and r.get("n_steps", 0) > MIN_CRASH_STEPS]
    n_total  = len(results)
    n_crash  = len(crashes)
    coll_pct = 100.0 * n_crash / n_total if n_total > 0 else float("nan")

    def _corr(r: dict, emu_key: str, geo_key: str) -> float:
        steps = r.get("steps", [])
        if len(steps) < 2:
            return float("nan")
        emu = np.array([s[emu_key] for s in steps])
        geo = np.array([s[geo_key] for s in steps])
        if np.std(emu) == 0 or np.std(geo) == 0:
            return float("nan")
        return float(np.corrcoef(emu, geo)[0, 1])

    def _iid_sign_change_rate(r: dict) -> float:
        """Fraction of consecutive step pairs where emu IID sign flips (0=stable, 1=every step)."""
        steps = r.get("steps", [])
        if len(steps) < 2:
            return float("nan")
        signs = np.sign([s["emu_iid_db"] for s in steps])
        flips = np.sum(signs[1:] != signs[:-1])
        return float(flips / (len(signs) - 1))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        f"{session_name}  —  {n_crash}/{n_total} crashed  ({coll_pct:.1f}%)"
        f"  [after step {MIN_CRASH_STEPS}]  |  "
        f"gen {policy_generation}  fitness {policy_fitness:.1f}",
        fontsize=9,
    )

    panels = [
        ("IID corr (emu vs geo)",      "emu_iid_db",  "geo_iid_db",  "jet", -1, 1),
        ("Distance corr (emu vs geo)", "emu_dist_mm", "geo_dist_mm", "jet", -1, 1),
        ("IID sign-change rate",       None,           None,          "jet",  0, 1),
    ]

    for ax, (label, emu_key, geo_key, cmap, vmin, vmax) in zip(axes, panels):
        walls = simulator.arena.walls
        if len(walls) > 0:
            ax.scatter(walls[:, 0], walls[:, 1], s=0.5, c="#bbbbbb",
                       linewidths=0, zorder=1)

        if crashes:
            cx = [r["crash_x"] for r in crashes]
            cy = [r["crash_y"] for r in crashes]
            if emu_key is None:
                values = [_iid_sign_change_rate(r) for r in crashes]
            else:
                values = [_corr(r, emu_key, geo_key) for r in crashes]
            sc = ax.scatter(cx, cy, s=30, c=values, cmap=cmap,
                            vmin=vmin, vmax=vmax, alpha=0.8, linewidths=0.3,
                            edgecolors="grey", zorder=3)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label(f"(last {N_LAST_STEPS} steps)", fontsize=8)

        ax.set_title(label, fontsize=9)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def print_summary(
    results_by_session: Dict[str, List[dict]],
    policy_fitness: float,
    policy_generation: int,
) -> None:
    col_w = 16
    print(f"\n{'Session':<{col_w}}  {'Episodes':>9}  {'Crashes':>8}  {'Coll %':>7}")
    print("─" * (col_w + 32))
    total_ep = total_cr = 0
    for sn, results in results_by_session.items():
        n_ep = len(results)
        n_cr = sum(r["collided"] and r.get("n_steps", 0) > MIN_CRASH_STEPS for r in results)
        pct  = 100.0 * n_cr / n_ep if n_ep > 0 else float("nan")
        print(f"{sn:<{col_w}}  {n_ep:>9}  {n_cr:>8}  {pct:>6.1f}%")
        total_ep += n_ep
        total_cr += n_cr
    print("─" * (col_w + 32))
    pct_total = 100.0 * total_cr / total_ep if total_ep > 0 else float("nan")
    print(f"{'TOTAL':<{col_w}}  {total_ep:>9}  {total_cr:>8}  {pct_total:>6.1f}%")
    print(f"\nPolicy: gen {policy_generation}, fitness {policy_fitness:.1f}")


def plot_emulator_accuracy(
    results_by_session: Dict[str, List[dict]],
    out_dir: str,
) -> None:
    """
    For every episode, compare the last N steps of emulator vs geometric readings,
    split by crashed vs non-crashed.  Overlay both groups on the same scatter to
    reveal whether emulator error is specific to crash situations.
    """
    def _stats(true, pred):
        if len(true) == 0:
            return float("nan"), float("nan")
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        if np.std(true) == 0 or np.std(pred) == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(true, pred)[0, 1])
        return rmse, r

    col_w = 16
    print(f"\nEmulator accuracy — last {N_LAST_STEPS} steps (crash vs no-crash):")
    header = (f"{'Session':<{col_w}}  {'Group':<10}  {'N eps':>6}  {'Steps':>6}  "
              f"{'dist RMSE':>10}  {'dist r':>7}  {'IID RMSE':>9}  {'IID r':>6}")
    print(header)
    print("─" * len(header))

    for sn, results in results_by_session.items():
        groups = {"crash": [r for r in results if r.get("collided")
                            and r.get("n_steps", 0) > MIN_CRASH_STEPS],
                  "no-crash": [r for r in results if not r.get("collided")]}

        arrays: Dict[str, dict] = {}
        for grp, recs in groups.items():
            recs_with_steps = [r for r in recs if r.get("steps")]
            geo_d = np.array([s["geo_dist_mm"] for r in recs_with_steps for s in r["steps"]])
            emu_d = np.array([s["emu_dist_mm"] for r in recs_with_steps for s in r["steps"]])
            geo_i = np.array([s["geo_iid_db"]  for r in recs_with_steps for s in r["steps"]])
            emu_i = np.array([s["emu_iid_db"]  for r in recs_with_steps for s in r["steps"]])
            arrays[grp] = dict(geo_d=geo_d, emu_d=emu_d, geo_i=geo_i, emu_i=emu_i,
                               n_eps=len(recs_with_steps))
            d_rmse, d_r = _stats(geo_d, emu_d)
            i_rmse, i_r = _stats(geo_i, emu_i)
            print(f"{sn:<{col_w}}  {grp:<10}  {len(recs_with_steps):>6}  {len(geo_d):>6}  "
                  f"{d_rmse:>9.1f}mm  {d_r:>7.3f}  {i_rmse:>8.2f}dB  {i_r:>6.3f}")

        # ── Figure ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"{sn} — emulator vs geometric ground truth  "
            f"(last {N_LAST_STEPS} steps per episode, crashes after step {MIN_CRASH_STEPS} only)",
            fontsize=10,
        )

        styles = {
            "no-crash": dict(c="#4878d0", alpha=0.25, s=3, label="no-crash"),
            "crash":    dict(c="#d65f5f", alpha=0.6,  s=8, label="crash",
                             edgecolors="none"),
        }

        for col_idx, (xkey, ykey, xlabel, ylabel, unit) in enumerate([
            ("geo_dist_mm", "emu_dist_mm", "Geometric distance (mm)",
             "Emulator distance (mm)", "mm"),
            ("geo_iid_db", "emu_iid_db", "Geometric IID (dB)",
             "Emulator IID (dB)", "dB"),
        ]):
            ax = axes[col_idx]
            all_x, all_y = [], []
            for grp in ("no-crash", "crash"):
                a = arrays[grp]
                x = a["geo_d"] if "dist" in xkey else a["geo_i"]
                y = a["emu_d"] if "dist" in ykey else a["emu_i"]
                if len(x):
                    ax.scatter(x, y, linewidths=0, **styles[grp])
                    all_x.append(x); all_y.append(y)

            if all_x:
                flat_x = np.concatenate(all_x)
                flat_y = np.concatenate(all_y)
                lo = min(flat_x.min(), flat_y.min())
                hi = max(flat_x.max(), flat_y.max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=5)

            d_c, d_r_c = _stats(arrays["crash"]["geo_d"],    arrays["crash"]["emu_d"])
            d_n, d_r_n = _stats(arrays["no-crash"]["geo_d"], arrays["no-crash"]["emu_d"])
            i_c, i_r_c = _stats(arrays["crash"]["geo_i"],    arrays["crash"]["emu_i"])
            i_n, i_r_n = _stats(arrays["no-crash"]["geo_i"], arrays["no-crash"]["emu_i"])

            if col_idx == 0:
                ax.set_title(
                    f"Distance\n"
                    f"crash: RMSE={d_c:.0f}mm r={d_r_c:.3f}  |  "
                    f"no-crash: RMSE={d_n:.0f}mm r={d_r_n:.3f}",
                    fontsize=8,
                )
            else:
                ax.set_title(
                    f"IID\n"
                    f"crash: RMSE={i_c:.2f}dB r={i_r_c:.3f}  |  "
                    f"no-crash: RMSE={i_n:.2f}dB r={i_r_n:.3f}",
                    fontsize=8,
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect("equal")
            ax.legend(fontsize=8, markerscale=2)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"emulator_accuracy_{sn}.png")
        plt.savefig(out_path, dpi=130)
        plt.close(fig)

    print(f"\nEmulator accuracy plots saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "run", "session", "episode", "n_steps", "collided",
    "start_x", "start_y", "start_yaw",
    "crash_x", "crash_y", "crash_yaw",
]


def main() -> None:
    rng      = np.random.default_rng(SEED)
    run_name = os.path.basename(os.path.normpath(RUN_DIR))
    out_dir  = os.path.join("CollisionAssessment", run_name)
    os.makedirs(out_dir, exist_ok=True)

    cfg                      = load_cfg(RUN_DIR)
    policy, fitness, gen_num = load_best_policy(RUN_DIR, cfg)
    print(f"Policy:   gen {gen_num},  fitness {fitness:.1f}")
    print(f"Output:   {out_dir}/")

    all_sessions = list(cfg.train_session_names)
    if cfg.validation_session_name:
        all_sessions.append(cfg.validation_session_name)

    # Build job list: (session_name, episode_idx, seed)
    seeds = rng.integers(0, 2**31, size=len(all_sessions) * N_EPISODES_PER_SESSION)
    jobs  = [
        (sn, ep, int(seeds[i * N_EPISODES_PER_SESSION + ep]))
        for i, sn in enumerate(all_sessions)
        for ep in range(N_EPISODES_PER_SESSION)
    ]
    n_jobs = len(jobs)
    print(f"Dispatching {n_jobs} episodes across {len(all_sessions)} sessions...\n")

    results_by_session: Dict[str, List[dict]] = {sn: [] for sn in all_sessions}

    n_workers = NUM_WORKERS or os.cpu_count()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(dataclasses.asdict(cfg), policy.get_genome(), all_sessions),
    ) as executor:
        for sn, ep_idx, rec in tqdm(
            executor.map(_episode_worker, jobs),
            total=n_jobs,
            desc="Episodes",
        ):
            results_by_session[sn].append(rec)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "collisions.csv")
    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for sn, results in results_by_session.items():
            for r in sorted(results, key=lambda x: x["episode"]):
                writer.writerow({"run": run_name, "session": sn, **r})
    print(f"Saved {csv_path}")

    # ── Crash maps ────────────────────────────────────────────────────────────
    for sn, results in results_by_session.items():
        sim      = build_simulator(sn)
        map_path = os.path.join(out_dir, f"crash_map_{sn}.png")
        plot_crash_map(results, sim, sn, map_path, fitness, gen_num)
    print(f"Crash maps saved to {out_dir}/")

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary(results_by_session, fitness, gen_num)

    # ── Emulator accuracy on crash episodes ───────────────────────────────────
    plot_emulator_accuracy(results_by_session, out_dir)


if __name__ == "__main__":
    main()
