#!/usr/bin/env python3
"""
SCRIPT_ParticleSLAM.py

Sequential multi-hypothesis place recognition using a RatSLAM-inspired
particle filter over a growing experience map.

A single long trajectory is collected. The experience map grows incrementally:
after step t, experience e_t = (pose_t, feat_t) is appended. At every step,
a particle filter maintains a belief over which past experience the robot
is currently near.

  - Particle state: integer index s ∈ {0, ..., t-1}
  - Prediction:     advance (s → s+1), stay (s → s), or random jump
                    (recovery / measurement-driven injection).
  - Update:         reweight by exp(-β · ||feat_t − feat_s||²).
  - Resample        when effective sample size drops below N/2.

The 'trivially-correct' answer at step t is s ≈ t-1 — the robot was just
there. The informative events are loop closures: particle mass that
concentrates at old s ≪ t-1 when the robot genuinely revisits a previously-
mapped region.

Output: SpatialInfo/<run>/particle_slam.png
  - Top: heatmap of particle weights over (past-index s, current-step t).
         Diagonal band (s ≈ t) is blanked out to highlight off-diagonal
         loop-closure activity.
  - Bottom: arena snapshots at SNAPSHOT_STEPS showing particles positioned
            at their hypothesised experience locations (blue dots), the
            growing map (grey path), and the current true pose (red X).

Usage:
  python SCRIPT_ParticleSLAM.py PolicyTraining/test_burst_h01
  python SCRIPT_ParticleSLAM.py PolicyTraining/test_burst_h01 --beta 50
"""

import argparse
import dataclasses
import os

import numpy as np
import matplotlib.pyplot as plt

from Library.SlamCore import (
    load_run, collect_data, build_windows, run_pf,
    N_PARTICLES, P_ADVANCE, P_STAY, P_JUMP,
)


# ── Config ────────────────────────────────────────────────────────────────────
RUN_DIR        = "PolicyTraining/test_burst_h01"
SESSION_NAME   = "sessionB02"
MAX_STEPS      = 500
WINDOW_LEN     = 5

BETA           = 30.0      # measurement-likelihood sharpness
MIN_LC_GAP     = 15        # heat-map: mask diagonal band of ±this half-width
SNAPSHOT_STEPS = [20, 80, 160, 240, 320, 400]


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(positions, history, heatmap, walls, run_name, output_dir, beta):
    N = len(positions)
    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1])

    # ── Top: heatmap (spans full width) ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    display = heatmap.copy()
    for t in range(N):
        lo = max(0, t - MIN_LC_GAP)
        hi = min(N, t + MIN_LC_GAP + 1)
        display[lo:hi, t] = 0.0

    im = ax.imshow(display, aspect="auto", origin="lower",
                   cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Current step t")
    ax.set_ylabel("Hypothesised past index s")
    ax.set_title(
        f"Particle weight over (past index s, current step t) — {run_name}  (β={beta:g})\n"
        f"diagonal band (|s − t| ≤ {MIN_LC_GAP}) blanked to highlight loop closures",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label="total particle weight")

    # ── Bottom: 6 arena snapshots ─────────────────────────────────────────────
    snap_axes = [fig.add_subplot(gs[1 + i // 3, i % 3]) for i in range(6)]

    for ax, t in zip(snap_axes, SNAPSHOT_STEPS):
        if t >= N:
            ax.axis("off")
            continue
        if len(walls):
            ax.scatter(walls[:, 0], walls[:, 1], s=0.3, c="#cccccc", linewidths=0)
        ax.plot(positions[:t + 1, 0], positions[:t + 1, 1],
                "-", color="#888888", linewidth=0.7, alpha=0.5)
        ax.scatter([positions[t, 0]], [positions[t, 1]],
                   s=120, marker="x", color="red", zorder=5, linewidths=2.5)

        particles, weights = history[t]
        p_pos  = positions[particles]
        w_norm = weights / weights.max() if weights.max() > 0 else weights
        sizes  = 3 + 40 * w_norm
        ax.scatter(p_pos[:, 0], p_pos[:, 1],
                   s=sizes, c="blue", alpha=0.35, linewidths=0, zorder=3)

        ax.set_title(f"t = {t}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "particle_slam.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir",    nargs="?",  default=RUN_DIR)
    parser.add_argument("--session",              default=SESSION_NAME)
    parser.add_argument("--beta",     type=float, default=BETA)
    parser.add_argument("--max_steps", type=int,  default=MAX_STEPS)
    parser.add_argument("--seed",     type=int,   default=0)
    args = parser.parse_args()

    rng        = np.random.default_rng(args.seed)
    run_name   = os.path.basename(args.run_dir.rstrip("/"))
    output_dir = os.path.join("SpatialInfo", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRun: {run_name}")
    mod, cfg, policy, is_burst = load_run(args.run_dir)
    print(f"  Type: {'burst' if is_burst else 'sonar'}")

    cfg_one = dataclasses.replace(cfg, train_session_names=[args.session])

    print(f"\nCollecting 1 long trajectory in {args.session}...")
    positions, yaws, meas_seq, traj_ids, simulators = collect_data(
        mod, cfg_one, policy, n_traj=1, max_steps=args.max_steps, rng=rng,
    )
    feats = build_windows(meas_seq, np.zeros(len(meas_seq), dtype=np.int32), WINDOW_LEN)
    print(f"  Trajectory length: {len(positions)} steps")

    print(f"\nRunning online particle filter  "
          f"(N={N_PARTICLES}, β={args.beta:g}, p_adv/stay/jump={P_ADVANCE}/{P_STAY}/{P_JUMP})...")
    history, heatmap = run_pf(feats, rng, args.beta)

    print("\nPlotting results...")
    walls = next(iter(simulators.values())).arena.walls
    plot_results(positions, history, heatmap, walls, run_name, output_dir, args.beta)

    print("\nDone.")


if __name__ == "__main__":
    main()
