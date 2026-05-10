#!/usr/bin/env python3
"""
SCRIPT_Ablations.py — input-ablation diagnostic for the supervised RNN.

Supports both observation variants written by SCRIPT_TrainPolicy.py:
  use_sigma=True  → 7-input  [d_L, d_C, d_R, σ_L, σ_C, σ_R, prev_rot]
  use_sigma=False → 4-input  [d_L, d_C, d_R, prev_rot]

The actual layout is read from best_policy.json (in_dim + use_sigma).
σ-only conditions auto-collapse into 'full' for the 4-input variant.

For each ablation, the listed input *names* are clamped to their training-time
medians (so the policy gets a constant typical signal rather than zero or
NaN, matching how the old script handled "no_sonar"). Conditions:

  full           : nothing clamped (control)
  no_distances   : d_L, d_C, d_R       — does the policy use distance at all?
  no_sigmas      : σ_L, σ_C, σ_R       — does the policy actually use σ?
                                         (drops to 'full' when use_sigma=False)
  center_only    : d_L, d_R [+σ_L,σ_R] — can it navigate from center sensing only?
  sides_only     : d_C [+σ_C]          — does it need the center reading?
  no_prev_rot    : prev_rot            — does action feedback matter?
  blind          : everything          — dead-reckoning from hidden state alone

Per condition we record trajectories, collisions, max arc-length progress
(in laps), and mean cross-track error. Outputs a 2×4 trajectory grid plus a
summary table.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")

import json
from typing import Dict, List, Tuple

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(1)

from Library import Settings as _settings
_settings.data_folder = "TargetArenas"
from Library.EnvironmentSimulator import EnvironmentSimulator
from Library.TargetPath import load_target_path

from SCRIPT_TrainPolicy import (
    Config, RNNNet, _project_with_segment, _obs_from_cfg, make_starts,
)


# ── Settings ─────────────────────────────────────────────────────────────────
RUN_DIR    = "PolicyTraining/default_Target02_h32_nosigma"
N_ROLLOUTS = 12
SEED       = 1234

# Each condition specifies which input *names* to clamp. Resolved against the
# actual obs layout in main(); names not present in the layout are silently
# dropped (e.g. σs in the 4-input variant). Conditions whose clamps all drop
# out collapse to 'full' and are removed to avoid duplicate work.
CONDITIONS_BY_NAME: Dict[str, List[str]] = {
    "full":          [],
    "no_distances":  ["d_L", "d_C", "d_R"],
    "no_sigmas":     ["sigma_L", "sigma_C", "sigma_R"],
    "center_only":   ["d_L", "d_R", "sigma_L", "sigma_R"],
    "sides_only":    ["d_C", "sigma_C"],
    "no_prev_rot":   ["prev_rot"],
    "blind":         ["d_L", "d_C", "d_R", "sigma_L", "sigma_C", "sigma_R", "prev_rot"],
}


def input_names_for(use_sigma: bool) -> List[str]:
    return (
        ["d_L", "d_C", "d_R"]
        + (["sigma_L", "sigma_C", "sigma_R"] if use_sigma else [])
        + ["prev_rot"]
    )


def resolve_conditions(input_names: List[str]) -> Dict[str, List[int]]:
    """Map condition→list-of-indices against the actual input layout.
    Drop conditions that collapse to 'full' (i.e. all their clamps were σs
    that aren't present in this policy)."""
    out: Dict[str, List[int]] = {}
    for cond, names in CONDITIONS_BY_NAME.items():
        idx = [input_names.index(n) for n in names if n in input_names]
        if cond == "full" or names == [] or idx:
            out[cond] = idx
    return out

# Median measurement
N_MEDIAN_STEPS = 600


# ── Load policy + config ─────────────────────────────────────────────────────

def load_run(run_dir: str) -> Tuple[Config, RNNNet, List[str]]:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg_dict = json.load(f)
    valid_keys = set(Config.__dataclass_fields__.keys())
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

    with open(os.path.join(run_dir, "best_policy.json")) as f:
        pol = json.load(f)

    in_dim    = int(pol["in_dim"])
    use_sigma = bool(pol.get("use_sigma", in_dim == 7))   # fallback for older policies
    expected  = 7 if use_sigma else 4
    if in_dim != expected:
        raise RuntimeError(
            f"Policy in_dim={in_dim} disagrees with use_sigma={use_sigma} "
            f"(expected {expected}). Saved policy is inconsistent."
        )
    if cfg.use_sigma != use_sigma:
        # Keep them in lockstep so encode_obs in this run produces the same
        # input layout the policy was trained against.
        cfg.use_sigma = use_sigma

    h, IN, OUT = cfg.hidden_size, in_dim, RNNNet.OUT_DIM
    g = np.array(pol["genome"], dtype=np.float32)
    i = 0
    W_xh = g[i:i + h*IN].reshape(h, IN);    i += h*IN
    W_hh = g[i:i + h*h ].reshape(h, h);     i += h*h
    b_h  = g[i:i + h];                      i += h
    W_hy = g[i:i + OUT*h].reshape(OUT, h);  i += OUT*h
    b_y  = g[i:i + OUT];                    i += OUT

    net = RNNNet(cfg.hidden_size, cfg.max_rotate_deg, in_dim=in_dim)
    with torch.no_grad():
        net.W_xh.copy_(torch.from_numpy(W_xh))
        net.W_hh.copy_(torch.from_numpy(W_hh))
        net.b_h .copy_(torch.from_numpy(b_h))
        net.W_hy.copy_(torch.from_numpy(W_hy))
        net.b_y .copy_(torch.from_numpy(b_y))
    return cfg, net, input_names_for(use_sigma)


# ── Median sonar inputs (in normalised units) ────────────────────────────────

def measure_input_medians(sim: EnvironmentSimulator, path, starts,
                          cfg: Config, input_names: List[str],
                          rng: np.random.Generator,
                          n_steps: int = N_MEDIAN_STEPS) -> np.ndarray:
    """Walk the teacher around the loop and record medians of every input
    channel (in normalised units, matching the network's input scale).

    Returns a (in_dim,) array ordered by input_names. prev_rot's slot is
    pinned to 0 by hand for "no_prev_rot" semantics — its empirical median
    drifts with the path's curvature and isn't a useful baseline."""
    from SCRIPT_TrainPolicy import teacher_rotation_deg
    sim.reseed(int(rng.integers(2**31 - 1)))
    obs_log: List[np.ndarray] = []
    x, y, yaw = starts[0]
    prev_rot = 0.0
    for _ in range(n_steps):
        meas = sim.get_sonar_measurement(x, y, yaw)
        obs_log.append(np.asarray(
            _obs_from_cfg(meas, prev_rot, cfg),
            dtype=np.float32,
        ))
        rot = teacher_rotation_deg(path, x, y, yaw,
                                   cfg.teacher_lookahead_mm, cfg.max_rotate_deg)
        rot_motor   = rot
        drive_motor = cfg.fixed_drive_mm
        if cfg.motion_rotate_noise_deg > 0.0:
            rot_motor = float(np.clip(
                rot + rng.normal(0.0, cfg.motion_rotate_noise_deg),
                -cfg.max_rotate_deg, cfg.max_rotate_deg,
            ))
        if cfg.motion_drive_noise_mm > 0.0:
            drive_motor = max(0.0, cfg.fixed_drive_mm
                              + float(rng.normal(0.0, cfg.motion_drive_noise_mm)))
        action = {"rotate1_deg": 0.0, "rotate2_deg": rot_motor,
                  "drive_mm": drive_motor}
        r = sim.simulate_robot_movement(x, y, yaw, [action], compute_sonar=False)[0]
        x, y, yaw = float(r["position"]["x"]), float(r["position"]["y"]), \
                    float(r["orientation"])
        prev_rot = rot
        if bool(r["collision"]["drive_blocked"]):
            x, y, yaw = starts[int(rng.integers(len(starts)))]
            prev_rot = 0.0
    arr = np.stack(obs_log, axis=0)   # (n_steps, in_dim)
    medians = np.median(arr, axis=0)
    medians[input_names.index("prev_rot")] = 0.0
    return medians


# ── Rollout under ablation ───────────────────────────────────────────────────

def rollout_ablated(net: RNNNet, sim: EnvironmentSimulator, path,
                    start: Tuple[float, float, float], cfg: Config,
                    clamp_idx: List[int], medians: np.ndarray,
                    rng: np.random.Generator,
                    ) -> Dict:
    sim.reseed(int(rng.integers(2**31 - 1)))
    x, y, yaw = start
    positions: List[Tuple[float, float]] = [(x, y)]
    h = torch.zeros(1, net.hidden_size)
    prev_rot = 0.0
    collided = False
    cross_track: List[float] = []
    arc_progress: List[float] = []

    with torch.no_grad():
        for _ in range(cfg.max_steps):
            meas = sim.get_sonar_measurement(x, y, yaw)
            obs = np.asarray(
                _obs_from_cfg(meas, prev_rot, cfg),
                dtype=np.float32,
            )
            for k in clamp_idx:
                obs[k] = medians[k]

            x_in = torch.from_numpy(obs).unsqueeze(0)
            h    = torch.tanh(x_in @ net.W_xh.T + h @ net.W_hh.T + net.b_h)
            y_t  = torch.tanh(h @ net.W_hy.T + net.b_y) * net.max_rotate_deg
            rot  = float(np.clip(float(y_t.item()),
                                 -cfg.max_rotate_deg, cfg.max_rotate_deg))

            rot_motor   = rot
            drive_motor = cfg.fixed_drive_mm
            if cfg.motion_rotate_noise_deg > 0.0:
                rot_motor = float(np.clip(
                    rot + rng.normal(0.0, cfg.motion_rotate_noise_deg),
                    -cfg.max_rotate_deg, cfg.max_rotate_deg,
                ))
            if cfg.motion_drive_noise_mm > 0.0:
                drive_motor = max(0.0, cfg.fixed_drive_mm
                                  + float(rng.normal(0.0, cfg.motion_drive_noise_mm)))
            action = {"rotate1_deg": 0.0, "rotate2_deg": rot_motor,
                      "drive_mm": drive_motor}
            r = sim.simulate_robot_movement(x, y, yaw, [action],
                                            compute_sonar=False)[0]
            x = float(r["position"]["x"]); y = float(r["position"]["y"])
            yaw = float(r["orientation"])
            positions.append((x, y))
            prev_rot = rot

            ct, seg_i, t = _project_with_segment(path, x, y)
            arc = float(path.cum_arc[seg_i] +
                        t * (path.cum_arc[seg_i + 1] - path.cum_arc[seg_i]))
            cross_track.append(ct)
            arc_progress.append(arc)

            if bool(r["collision"]["drive_blocked"]):
                collided = True
                break

    arc_arr = np.array(arc_progress)
    arc0 = float(arc_arr[0]) if arc_arr.size else 0.0
    fwd = (arc_arr - arc0) % path.total_length
    laps = float(np.max(fwd) / path.total_length) if arc_arr.size else 0.0

    return {
        "positions":  positions,
        "collided":   collided,
        "laps":       laps,
        "ct_mean_mm": float(np.mean(cross_track)) if cross_track else 0.0,
        "ct_max_mm":  float(np.max(cross_track))  if cross_track else 0.0,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {RUN_DIR}")
    cfg, net, input_names = load_run(RUN_DIR)
    cfg.teacher_perturb_prob = 0.0   # no perturbation during eval

    conditions = resolve_conditions(input_names)
    print(f"  in_dim={net.in_dim}  use_sigma={cfg.use_sigma}  "
          f"layout={input_names}")
    print(f"  conditions: {list(conditions.keys())}")

    rng  = np.random.default_rng(SEED)
    sim  = EnvironmentSimulator(cfg.target_arena)
    path = load_target_path(cfg.target_arena, _settings.data_folder,
                            cfg.path_resample_mm)
    starts_pool = make_starts(path, cfg, rng)
    walls = sim.arena.walls

    print(f"Measuring input medians under teacher ({N_MEDIAN_STEPS} steps)")
    medians = measure_input_medians(sim, path, starts_pool, cfg, input_names, rng)
    print(f"  medians (normalised units):")
    for name, m in zip(input_names, medians):
        print(f"    {name:>9}: {m:+.3f}")

    starts = [starts_pool[int(rng.integers(len(starts_pool)))]
              for _ in range(N_ROLLOUTS)]

    results: Dict[str, List[Dict]] = {c: [] for c in conditions}
    for cond, clamp_idx in conditions.items():
        names_clamped = [input_names[k] for k in clamp_idx]
        print(f"\n→ {cond}  (clamping: {names_clamped})  × {N_ROLLOUTS} rollouts")
        for s in starts:
            results[cond].append(
                rollout_ablated(net, sim, path, s, cfg,
                                clamp_idx, medians, rng)
            )
        laps = [r["laps"] for r in results[cond]]
        coll = [r["collided"] for r in results[cond]]
        cts  = [r["ct_mean_mm"] for r in results[cond]]
        print(f"   laps:  mean={np.mean(laps):.2f}  median={np.median(laps):.2f}")
        print(f"   coll:  {sum(coll)}/{len(coll)}")
        print(f"   ct:    mean={np.mean(cts):6.0f} mm  max-of-mean={np.max(cts):6.0f} mm")

    # ── Summary printout ────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print(f"  {'condition':<14}  {'laps_mean':>9}  {'coll/n':>8}  {'⟨ct⟩ mm':>10}")
    print("─" * 72)
    for cond in conditions:
        rs = results[cond]
        laps_mean = float(np.mean([r["laps"] for r in rs]))
        n_coll    = sum(r["collided"] for r in rs)
        ct_mean   = float(np.mean([r["ct_mean_mm"] for r in rs]))
        print(f"  {cond:<14}  {laps_mean:>9.2f}  {n_coll:>3d}/{N_ROLLOUTS:<4d}  {ct_mean:>10.0f}")
    print("─" * 72)

    # ── Trajectory plot: 2×4 grid ───────────────────────────────────────────
    n_cond = len(conditions)
    fig, axes = plt.subplots(2, 4, figsize=(20, 11), sharex=True, sharey=True)
    cmap = plt.cm.tab10
    pts  = path.points
    for ax, cond in zip(axes.flat, conditions):
        if walls is not None and len(walls) > 0:
            ax.scatter(walls[:, 0], walls[:, 1], s=0.5, c="#aaaaaa",
                       linewidths=0, zorder=1)
        ax.plot(pts[:, 0], pts[:, 1], color="#d62728", linewidth=2.0, alpha=0.7,
                zorder=2, label="path")
        for i, r in enumerate(results[cond]):
            arr = np.array(r["positions"])
            c   = cmap(i % 10)
            ls  = "--" if r["collided"] else "-"
            ax.plot(arr[:, 0], arr[:, 1], color=c, linewidth=0.9,
                    linestyle=ls, alpha=0.85, zorder=3)
            ax.plot(arr[0,  0], arr[0,  1], "o", color=c, markersize=3, zorder=4)
            ax.plot(arr[-1, 0], arr[-1, 1], "x", color=c, markersize=4, zorder=4)
        laps = float(np.mean([r["laps"]     for r in results[cond]]))
        coll = sum(r["collided"]            for r in results[cond])
        ct   = float(np.mean([r["ct_mean_mm"] for r in results[cond]]))
        clamp_str = ",".join(input_names[k] for k in conditions[cond]) or "(none)"
        ax.set_title(f"{cond}\nclamp: {clamp_str}\n"
                     f"laps={laps:.2f}  coll={coll}/{N_ROLLOUTS}  ⟨ct⟩={ct:.0f}mm",
                     fontsize=9)
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
    # Hide unused panels
    for j in range(n_cond, 8):
        axes.flat[j].set_visible(False)
    fig.suptitle(f"Input ablations on {RUN_DIR}", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(RUN_DIR, "ablations.png")
    fig.savefig(out_path, dpi=120)
    fig.savefig(os.path.splitext(out_path)[0] + ".svg")
    plt.close(fig)
    print(f"\nSaved {out_path} (+ .svg)")


if __name__ == "__main__":
    main()
