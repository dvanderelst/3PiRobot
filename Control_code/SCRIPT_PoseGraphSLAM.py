#!/usr/bin/env python3
"""
SCRIPT_PoseGraphSLAM.py

Close the mapping loop: use loop closures produced by the RatSLAM-style
particle filter (SCRIPT_ParticleSLAM) to correct a drifted odometry map
via sparse pose-graph least-squares.

Pipeline:
  1. Collect one long trajectory in a training arena.
  2. Simulate noisy odometry along the true path (drifted map).
  3. Run the particle filter on the measurement stream to maintain belief
     over past experiences.
  4. Extract loop closures: for each step t, take argmax past index s
     outside a diagonal band; accept if particle weight at s exceeds
     LC_WEIGHT_THRESHOLD. Dedup via bucketing.
  5. Build a 2D pose graph:
        - Nodes: per-step positions, initialised at noisy odometry.
        - Odometry edges: x_{i+1} − x_i = d_i^noisy  (residual, weighted by w_odom)
        - Loop-closure edges: x_s − x_t = 0          (weighted by w_lc)
        - Anchor: x_0 fixed at noisy_odom[0].
     Solve x and y axes independently (linear least squares, sparse).
  6. Plot true / noisy-odometry / relaxed trajectories.

Output: SpatialInfo/<run>/posegraph_slam.png
"""

import argparse
import dataclasses
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.sparse
import scipy.sparse.linalg

from Library.SlamCore import (
    load_run, collect_data, build_windows, simulate_odometry, run_pf,
)


# ── Config ────────────────────────────────────────────────────────────────────
RUN_DIR             = "PolicyTraining/test_burst_h01"
SESSION_NAME        = "sessionB03"
MAX_STEPS           = 500
WINDOW_LEN          = 7
SEED                = 1          # RNG seed — controls start pick, noise, PF draws

ODOM_NOISE_XY_MM    = 30.0
ODOM_NOISE_YAW_DEG  = 5.0

PF_BETA             = 30.0
MIN_LC_GAP          = 30         # exclude past steps within this many of t
LC_WEIGHT_THRESHOLD = 0.35       # min particle weight at a candidate peak to accept
LC_DEDUP_BUCKET     = 8          # dedup loop closures by (s // bucket, t // bucket)
LC_ODOM_GATE_K      = 4.0        # reject LC if d_odom(s,t) > K·σ·√|t−s| (drift bound)

W_ODOM              = 1.0        # odometry edge weight (σ_odom = 30 mm/step)
W_LC                = 0.3        # loop closure edge weight (σ_LC ≈ 100 mm, softer than odom)
W_SMOOTH            = 0.75        # second-derivative smoothness prior weight
ANCHOR_WEIGHT       = 1e3


# ══════════════════════════════════════════════════════════════════════════════
# Loop closure extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_loop_closures(history, N, noisy_pos=None, odom_sigma=ODOM_NOISE_XY_MM):
    """
    For each step t, find the past index s ≠ t (outside diagonal band) with
    highest summed particle weight. Accept if weight exceeds the threshold
    AND (if noisy_pos given) the odometric distance between s and t is
    within the drift bound K·σ·√|t−s|. Dedup via 2D bucketing.
    """
    seen    = set()
    closures = []
    rejected_by_gate = 0
    for t, (particles, weights) in enumerate(history):
        if t <= MIN_LC_GAP:
            continue
        bins = np.zeros(N, dtype=np.float64)
        np.add.at(bins, particles, weights)
        lo = max(0, t - MIN_LC_GAP)
        hi = min(N, t + MIN_LC_GAP + 1)
        bins[lo:hi] = 0.0
        if bins.max() < LC_WEIGHT_THRESHOLD:
            continue
        s = int(bins.argmax())

        # Odometric drift gate
        if noisy_pos is not None:
            d_odom = float(np.hypot(*(noisy_pos[s] - noisy_pos[t])))
            max_drift = LC_ODOM_GATE_K * odom_sigma * np.sqrt(abs(t - s))
            if d_odom > max_drift:
                rejected_by_gate += 1
                continue

        key = (s // LC_DEDUP_BUCKET, t // LC_DEDUP_BUCKET)
        if key in seen:
            continue
        seen.add(key)
        closures.append((s, t))
    if rejected_by_gate:
        print(f"  {rejected_by_gate} candidate LCs rejected by odometric drift gate")
    return closures


# ══════════════════════════════════════════════════════════════════════════════
# Pose-graph linear least-squares (per axis)
# ══════════════════════════════════════════════════════════════════════════════

def solve_axis(noisy_disp_axis, loop_closures, N, anchor_val):
    """
    Linear least-squares on one coordinate axis. Returns length-N solution.
    Rows: anchor + odometry edges + loop closures + smoothness prior
          (second-derivative penalty on interior nodes).
    """
    n_smooth = max(0, N - 2)
    n_rows   = 1 + (N - 1) + len(loop_closures) + n_smooth
    rows, cols, data = [], [], []
    b = np.zeros(n_rows)
    r = 0

    # Anchor
    rows.append(r); cols.append(0); data.append(ANCHOR_WEIGHT)
    b[r] = ANCHOR_WEIGHT * anchor_val
    r += 1

    # Odometry: w_odom * (x_{i+1} − x_i) = w_odom * disp_i
    for i in range(N - 1):
        rows.append(r); cols.append(i);     data.append(-W_ODOM)
        rows.append(r); cols.append(i + 1); data.append(+W_ODOM)
        b[r] = W_ODOM * noisy_disp_axis[i]
        r += 1

    # Loop closures: w_lc * (x_s − x_t) = 0
    for s, t in loop_closures:
        rows.append(r); cols.append(s); data.append(+W_LC)
        rows.append(r); cols.append(t); data.append(-W_LC)
        b[r] = 0.0
        r += 1

    # Smoothness: w_smooth * (x_{i-1} − 2·x_i + x_{i+1}) = 0  for 1 ≤ i ≤ N-2
    for i in range(1, N - 1):
        rows.append(r); cols.append(i - 1); data.append(+W_SMOOTH)
        rows.append(r); cols.append(i);     data.append(-2.0 * W_SMOOTH)
        rows.append(r); cols.append(i + 1); data.append(+W_SMOOTH)
        b[r] = 0.0
        r += 1

    A = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n_rows, N)).tocsr()
    sol, *_ = scipy.sparse.linalg.lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=20000)
    return sol


def solve_pose_graph(noisy_pos, loop_closures):
    """
    Solve both axes. Anchor at noisy_pos[0].
    """
    N = len(noisy_pos)
    noisy_disp = np.diff(noisy_pos, axis=0)
    xs = solve_axis(noisy_disp[:, 0], loop_closures, N, noisy_pos[0, 0])
    ys = solve_axis(noisy_disp[:, 1], loop_closures, N, noisy_pos[0, 1])
    return np.stack([xs, ys], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(true_pos, noisy_pos, relaxed_pos, loop_closures,
                 walls, run_name, output_dir, errors):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Pose-graph SLAM — {run_name}  "
                 f"({len(loop_closures)} loop closures)", fontsize=11)

    def _walls(ax):
        if len(walls):
            ax.scatter(walls[:, 0], walls[:, 1], s=0.3, c="#cccccc", linewidths=0)

    def _fmt(ax):
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # Panel 1: true
    ax = axes[0, 0]
    _walls(ax)
    ax.plot(true_pos[:, 0], true_pos[:, 1], "-", color="#1f77b4", linewidth=1.0)
    ax.set_title("True trajectory", fontsize=9)
    _fmt(ax)

    # Panel 2: noisy odometry
    ax = axes[0, 1]
    _walls(ax)
    ax.plot(noisy_pos[:, 0], noisy_pos[:, 1], "--", color="#d62728", linewidth=1.0)
    ax.set_title(f"Noisy odometry  (σ={ODOM_NOISE_XY_MM:.0f} mm/step)", fontsize=9)
    _fmt(ax)

    # Panel 3: relaxed map
    ax = axes[0, 2]
    _walls(ax)
    ax.plot(relaxed_pos[:, 0], relaxed_pos[:, 1], "-", color="#2ca02c", linewidth=1.0)
    # Loop closure edges
    for s, t in loop_closures:
        ax.plot([relaxed_pos[s, 0], relaxed_pos[t, 0]],
                [relaxed_pos[s, 1], relaxed_pos[t, 1]],
                color="#ff7f0e", linewidth=0.5, alpha=0.5, zorder=2)
    ax.set_title("Pose-graph relaxed map  (orange = LC edges)", fontsize=9)
    _fmt(ax)

    # Panel 4: overlay
    ax = axes[1, 0]
    _walls(ax)
    ax.plot(true_pos[:, 0],    true_pos[:, 1],    "-",  color="#1f77b4",
            linewidth=1.0, label="true")
    ax.plot(noisy_pos[:, 0],   noisy_pos[:, 1],   "--", color="#d62728",
            linewidth=0.9, alpha=0.8, label="odometry")
    ax.plot(relaxed_pos[:, 0], relaxed_pos[:, 1], "-",  color="#2ca02c",
            linewidth=1.0, alpha=0.9, label="relaxed")
    ax.legend(fontsize=8, loc="best")
    ax.set_title("Overlay", fontsize=9)
    _fmt(ax)

    # Panel 5: loop closure edges on the true map (for interpretation)
    ax = axes[1, 1]
    _walls(ax)
    ax.plot(true_pos[:, 0], true_pos[:, 1], "-",
            color="#bbbbbb", linewidth=0.7, alpha=0.8)
    # True distance between paired steps colouring: green = close, red = far
    for s, t in loop_closures:
        d = float(np.hypot(true_pos[s, 0] - true_pos[t, 0],
                           true_pos[s, 1] - true_pos[t, 1]))
        color = "#00aa44" if d < 300 else "#cc2222"
        ax.plot([true_pos[s, 0], true_pos[t, 0]],
                [true_pos[s, 1], true_pos[t, 1]],
                color=color, linewidth=0.7, alpha=0.5, zorder=2)
    legend = [
        mlines.Line2D([], [], color="#00aa44", lw=1.5, label="LC: truly close (<300 mm)"),
        mlines.Line2D([], [], color="#cc2222", lw=1.5, label="LC: falsely paired"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="best")
    ax.set_title("Loop closures on true map", fontsize=9)
    _fmt(ax)

    # Panel 6: per-step position error (drift) vs step
    ax = axes[1, 2]
    ax.plot(errors["odom"],    "--", color="#d62728", label="odometry drift")
    ax.plot(errors["relaxed"], "-",  color="#2ca02c", label="relaxed error")
    ax.set_xlabel("Step")
    ax.set_ylabel("Position error vs. true (mm)")
    ax.set_title("Drift over time", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "posegraph_slam.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir",   nargs="?",  default=RUN_DIR)
    parser.add_argument("--session",             default=SESSION_NAME)
    parser.add_argument("--beta",    type=float, default=PF_BETA)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed",    type=int,   default=SEED)
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
    N = len(positions)
    print(f"  Trajectory length: {N} steps")

    print(f"\nSimulating noisy odometry "
          f"(σ_xy={ODOM_NOISE_XY_MM:.0f} mm, σ_yaw={ODOM_NOISE_YAW_DEG:.0f}°)...")
    noisy_pos, _ = simulate_odometry(
        positions, yaws, traj_ids, ODOM_NOISE_XY_MM, ODOM_NOISE_YAW_DEG, rng,
    )

    feats = build_windows(meas_seq, np.zeros(N, dtype=np.int32), WINDOW_LEN)

    print(f"\nRunning particle filter (β={args.beta:g})...")
    history, heatmap = run_pf(feats, rng, args.beta)

    print("\nExtracting loop closures...")
    loop_closures = extract_loop_closures(history, N, noisy_pos=noisy_pos)
    print(f"  Found {len(loop_closures)} loop closures")

    # TP/FP breakdown
    n_tp = sum(1 for s, t in loop_closures
               if np.hypot(*(positions[s] - positions[t])) < 300)
    n_fp = len(loop_closures) - n_tp
    prec = n_tp / max(1, len(loop_closures))
    print(f"  TP: {n_tp}  FP: {n_fp}  precision: {prec:.3f}")

    print("\nSolving pose graph...")
    relaxed_pos = solve_pose_graph(noisy_pos, loop_closures)

    # Errors
    errors = {
        "odom":    np.linalg.norm(noisy_pos   - positions, axis=1),
        "relaxed": np.linalg.norm(relaxed_pos - positions, axis=1),
    }
    print(f"  Final odometry drift: {errors['odom'][-1]:.0f} mm")
    print(f"  Final relaxed error : {errors['relaxed'][-1]:.0f} mm")
    print(f"  Mean  odometry drift: {errors['odom'].mean():.0f} mm")
    print(f"  Mean  relaxed error : {errors['relaxed'].mean():.0f} mm")

    print("\nPlotting...")
    walls = next(iter(simulators.values())).arena.walls
    plot_results(positions, noisy_pos, relaxed_pos, loop_closures,
                 walls, run_name, output_dir, errors)

    print("\nDone.")


if __name__ == "__main__":
    main()
