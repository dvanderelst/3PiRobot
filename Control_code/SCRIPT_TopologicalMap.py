#!/usr/bin/env python3
"""
SCRIPT_TopologicalMap.py

Demonstrate spatial information in emulated sonar measurements via loop closure
detection — a core component of topological SLAM.

Approach:
  1. Run the best policy, collect steps across multiple trajectories.
  2. Build windowed measurement feature vectors (same as SCRIPT_AnalyseSpatialInfo).
  3. Detect loop closures: step pairs whose measurement windows are similar
     enough (L2 distance < threshold).
  4. Validate against ground-truth positions:
       TP = loop closure where the robot was actually nearby (<= true_dist mm)
       FP = loop closure where it was not
  5. Simulate noisy dead-reckoning to show how raw odometry would drift without
     measurement-based loop closure correction.

Output:  SpatialInfo/<run_name>/topological_map.png
  Panel 1: arena map — true paths (solid) + dead-reckoning paths (dashed) +
           loop closure connections (green = TP, red = FP).
  Panel 2: precision-recall curve with the operating point marked.

Works for both sonar and burst runs.

Usage:
  python SCRIPT_TopologicalMap.py PolicyTraining/test_burst_h01
  python SCRIPT_TopologicalMap.py PolicyTraining/sonar_h01 --threshold 0.3
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from Library.SlamCore import load_run, collect_data, build_windows, simulate_odometry

# ── Run to analyse (used when running directly from PyCharm) ──────────────────
RUN_DIR = "PolicyTraining/test_burst_h01"

# ── Parameters ────────────────────────────────────────────────────────────────
WINDOW_LEN          = 5      # measurement window length (steps)
N_TRAJ              = 1      # trajectories per arena session — one long run
MAX_STEPS           = 500    # max steps per trajectory (may terminate earlier on collision)
MIN_STEP_GAP        = 15     # min step gap (within same traj) to be a LC candidate
LC_THRESHOLD        = 0.30   # measurement L2 distance threshold for loop closure
TRUE_LC_DIST_MM     = 300.0  # ground-truth dist (mm) below which a LC is "correct"
ODOM_NOISE_XY_MM    = 30.0   # Gaussian noise std added per step to dead-reckoning
ODOM_NOISE_YAW_DEG  = 5.0    # Gaussian noise std per step to dead-reckoning yaw
HEADING_GATE_DEG    = 45.0   # max heading difference (from odometry) for a valid LC
MIN_ODOM_DIST_MM    = 500.0  # min dead-reckoning distance for a non-trivial LC
MAX_LC_DRAW         = 600    # max loop closure lines to draw (sample if more)


# ══════════════════════════════════════════════════════════════════════════════
# Pairwise distances and loop closure detection
# ══════════════════════════════════════════════════════════════════════════════

def get_valid_pairs(traj_ids: np.ndarray, min_step_gap: int):
    """
    Return (i_idx, j_idx) upper-triangle pairs that are valid LC candidates:
    excludes same-trajectory pairs that are fewer than min_step_gap steps apart.
    """
    N = len(traj_ids)
    i_idx, j_idx = np.triu_indices(N, k=1)
    same_traj = traj_ids[i_idx] == traj_ids[j_idx]
    too_close = same_traj & ((j_idx - i_idx) < min_step_gap)
    valid = ~too_close
    return i_idx[valid], j_idx[valid]


def compute_meas_distances(feats: np.ndarray, i_idx, j_idx) -> np.ndarray:
    """Vectorised L2 distances between specified feature pairs."""
    diff = feats[i_idx].astype(np.float64) - feats[j_idx].astype(np.float64)
    return np.sqrt(np.sum(diff ** 2, axis=1)).astype(np.float32)


def compute_gt_distances(positions: np.ndarray, i_idx, j_idx) -> np.ndarray:
    """Euclidean distances in mm between specified position pairs."""
    diff = positions[i_idx] - positions[j_idx]
    return np.sqrt(np.sum(diff ** 2, axis=1))


def apply_heading_gate(i_idx, j_idx, yaws, heading_gate_deg):
    """Return boolean mask: True where heading difference <= heading_gate_deg."""
    diff = np.abs(yaws[i_idx] - yaws[j_idx]) % 360.0
    diff = np.minimum(diff, 360.0 - diff)   # wrap to [0, 180]
    return diff <= heading_gate_deg


def precision_recall_sweep(meas_dists, gt_dists, n_thresholds=200):
    """
    Sweep over measurement-distance thresholds.
    Returns (thresholds, precisions, recalls, total_true_positives).
    A ground-truth pair is considered a "true positive opportunity" if
    gt_dist < TRUE_LC_DIST_MM  (set by caller via the gt_dists array).
    """
    truly_close = gt_dists < TRUE_LC_DIST_MM
    total_tp    = int(truly_close.sum())

    t_max      = float(np.percentile(meas_dists, 99))
    thresholds = np.linspace(0.0, t_max, n_thresholds + 1)[1:]
    precisions = np.empty(n_thresholds, dtype=np.float32)
    recalls    = np.empty(n_thresholds, dtype=np.float32)

    for k, t in enumerate(thresholds):
        detected = meas_dists < t
        tp = int((detected & truly_close).sum())
        fp = int((detected & ~truly_close).sum())
        precisions[k] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recalls[k]    = tp / total_tp  if total_tp  > 0 else 0.0

    return thresholds, precisions, recalls, total_tp


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(positions, noisy_pos, traj_ids,
                 lc_i, lc_j, is_tp,
                 thresholds, precisions, recalls,
                 op_threshold, op_true_dist,
                 simulators, run_name, output_dir, rng):

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"Loop closure detection — {run_name}", fontsize=11)

    # Shared: walls, colours, stats
    walls = np.empty((0, 2))
    for sim in simulators.values():
        w = sim.arena.walls
        if len(w) > 0:
            walls = w
        break

    cmap    = plt.cm.tab20
    n_trajs = int(traj_ids.max()) + 1
    colours = [cmap(i % 20) for i in range(n_trajs)]

    n_tp = int(is_tp.sum())
    n_fp = int((~is_tp).sum())
    prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else float("nan")
    op_k = int(np.argmin(np.abs(thresholds - op_threshold)))
    rec  = float(recalls[op_k]) if len(recalls) else float("nan")

    def _draw_walls(ax):
        if len(walls):
            ax.scatter(walls[:, 0], walls[:, 1], s=0.3, c="#cccccc",
                       linewidths=0, zorder=1)

    def _format_map(ax):
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # ── Panel 1: true paths ───────────────────────────────────────────────────
    ax = axes[0]
    _draw_walls(ax)
    for t in range(n_trajs):
        idx = np.where(traj_ids == t)[0]
        ax.plot(positions[idx, 0], positions[idx, 1],
                color=colours[t], linewidth=0.7, alpha=0.6, zorder=2)
    ax.set_title("True paths", fontsize=9)
    _format_map(ax)

    # ── Panel 2: dead-reckoning paths ─────────────────────────────────────────
    ax = axes[1]
    _draw_walls(ax)
    for t in range(n_trajs):
        idx = np.where(traj_ids == t)[0]
        ax.plot(noisy_pos[idx, 0], noisy_pos[idx, 1],
                color=colours[t], linewidth=0.7, alpha=0.6, zorder=2)
    ax.set_title(f"Dead-reckoning  (noise={ODOM_NOISE_XY_MM:.0f} mm/step)", fontsize=9)
    _format_map(ax)

    # ── Panel 3: loop closure connections on true map ─────────────────────────
    ax = axes[2]
    _draw_walls(ax)
    for t in range(n_trajs):
        idx = np.where(traj_ids == t)[0]
        ax.plot(positions[idx, 0], positions[idx, 1],
                color=colours[t], linewidth=0.5, alpha=0.3, zorder=2)

    draw_idx = np.arange(len(lc_i))
    if len(lc_i) > MAX_LC_DRAW:
        draw_idx = rng.choice(len(lc_i), size=MAX_LC_DRAW, replace=False)
    for k in draw_idx:
        i, j  = lc_i[k], lc_j[k]
        color = "#00aa44" if is_tp[k] else "#cc2222"
        ax.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color=color, linewidth=0.8, alpha=0.6, zorder=3)

    legend_handles = [
        mlines.Line2D([], [], color="#00aa44", lw=1.5,
                      label=f"TP  (n={n_tp})"),
        mlines.Line2D([], [], color="#cc2222", lw=1.5,
                      label=f"FP  (n={n_fp})"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right")
    ax.set_title(
        f"Loop closures  –  threshold={op_threshold:.3f}\n"
        f"precision={prec:.2f}  recall={rec:.2f}",
        fontsize=8,
    )
    _format_map(ax)

    # ── Panel 4: precision-recall curve ──────────────────────────────────────
    ax = axes[3]
    ax.plot(recalls, precisions, "b-", linewidth=1.5, label="PR curve")
    ax.scatter([rec], [prec], color="red", s=60, zorder=5,
               label=f"threshold={op_threshold:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "topological_map.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Topological place mapping  (union-find over loop closures)
# ══════════════════════════════════════════════════════════════════════════════

def build_topological_map(lc_i, lc_j, n_steps: int, min_members: int = 3) -> np.ndarray:
    """
    Define "places" as connected components in the loop-closure graph.

    Each LC pair (i, j) asserts that step i and step j are at the same place.
    Union-find propagates that equivalence transitively.  Components with fewer
    than min_members steps are discarded (labelled -1 = corridor/unassigned).

    Returns place_labels (N,): int >= 0 for a known place, -1 for corridor.
    """
    parent = list(range(n_steps))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j in zip(lc_i.tolist(), lc_j.tolist()):
        union(i, j)

    # Group all steps by their root
    from collections import defaultdict
    components: dict = defaultdict(list)
    for s in range(n_steps):
        components[find(s)].append(s)

    # Assign place IDs to components large enough to be meaningful
    labels = np.full(n_steps, -1, dtype=np.int32)
    place_id = 0
    for members in sorted(components.values(), key=len, reverse=True):
        if len(members) >= min_members:
            for s in members:
                labels[s] = place_id
            place_id += 1

    return labels


def plot_place_map(positions, traj_ids, labels, lc_i, lc_j, simulators,
                   run_name, output_dir, lc_threshold, min_members):
    """
    Two-panel topological map built from loop closures + sequential structure.

      Left  — Arena map: steps coloured by place (union-find components).
               Corridor steps (no LC membership) shown in light grey.
               Each place = a set of steps linked by measurement similarity.

      Right — Topological graph: nodes = places (at centroid positions, sized
               by member count), edges = sequential transitions between places
               (line weight = transition count).
    """
    walls = np.empty((0, 2))
    for sim in simulators.values():
        w = sim.arena.walls
        if len(w) > 0:
            walls = w
        break

    place_ids  = sorted(set(labels.tolist()) - {-1})
    k          = len(place_ids)
    cmap       = plt.cm.tab20
    colour_map = {p: cmap(i % 20) for i, p in enumerate(place_ids)}

    n_corridor  = int((labels == -1).sum())
    n_placed    = int((labels >= 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Topological place map — {run_name}  "
        f"(threshold={lc_threshold}, {k} places, {n_corridor} corridor steps)",
        fontsize=11,
    )

    def _walls(ax):
        if len(walls):
            ax.scatter(walls[:, 0], walls[:, 1], s=0.3, c="#dddddd",
                       linewidths=0, zorder=1)

    def _fmt(ax):
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # ── Left: steps coloured by place ────────────────────────────────────────
    ax = axes[0]
    _walls(ax)
    # Corridor steps (background)
    corridor_mask = labels == -1
    if corridor_mask.any():
        ax.scatter(positions[corridor_mask, 0], positions[corridor_mask, 1],
                   s=3, color="#cccccc", alpha=0.4, linewidths=0, zorder=2)
    # Place steps — draw LC links first, then coloured dots on top
    for p in place_ids:
        mask = labels == p
        # Draw LC connections within this place
        place_set = set(np.where(mask)[0].tolist())
        for ii, jj in zip(lc_i.tolist(), lc_j.tolist()):
            if ii in place_set and jj in place_set:
                ax.plot([positions[ii, 0], positions[jj, 0]],
                        [positions[ii, 1], positions[jj, 1]],
                        color=colour_map[p], linewidth=0.5, alpha=0.3, zorder=2)
        ax.scatter(positions[mask, 0], positions[mask, 1],
                   s=8, color=colour_map[p], alpha=0.9, linewidths=0, zorder=3)

    ax.set_title(
        f"Steps coloured by place  ({k} places, {n_placed} steps)\n"
        f"grey = corridor  |  lines = loop closure links",
        fontsize=8,
    )
    _fmt(ax)

    # ── Right: topological graph ──────────────────────────────────────────────
    ax = axes[1]
    _walls(ax)

    centroids = {p: positions[labels == p].mean(axis=0) for p in place_ids}
    sizes     = {p: int((labels == p).sum())             for p in place_ids}

    # Count sequential transitions between places
    p_to_idx = {p: i for i, p in enumerate(place_ids)}
    trans = np.zeros((k, k), dtype=np.int32)
    for t in np.unique(traj_ids):
        idx  = np.where(traj_ids == t)[0]
        labs = labels[idx]
        for s in range(len(labs) - 1):
            a, b = int(labs[s]), int(labs[s + 1])
            if a >= 0 and b >= 0 and a != b:
                ia, ib = p_to_idx[a], p_to_idx[b]
                trans[ia, ib] += 1
                trans[ib, ia] += 1

    max_t  = trans.max() if trans.max() > 0 else 1
    cutoff = max(1, max_t * 0.03)
    for ii, pi in enumerate(place_ids):
        for jj, pj in enumerate(place_ids):
            if jj <= ii:
                continue
            if trans[ii, jj] >= cutoff:
                lw = 0.5 + 2.5 * trans[ii, jj] / max_t
                ax.plot([centroids[pi][0], centroids[pj][0]],
                        [centroids[pi][1], centroids[pj][1]],
                        color="#999999", linewidth=lw, alpha=0.6, zorder=2)

    max_size = max(sizes.values()) if sizes else 1
    for p in place_ids:
        node_s = 40 + 400 * sizes[p] / max_size
        ax.scatter(centroids[p][0], centroids[p][1],
                   s=node_s, color=colour_map[p],
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.text(centroids[p][0], centroids[p][1], str(p),
                fontsize=6, ha="center", va="center",
                fontweight="bold", zorder=4)

    ax.set_title("Topological graph  (nodes = places, edges = sequential transitions)",
                 fontsize=9)
    _fmt(ax)

    plt.tight_layout()
    path = os.path.join(output_dir, "place_map.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",     nargs="?", default=RUN_DIR)
    parser.add_argument("--window",    type=int,   default=WINDOW_LEN)
    parser.add_argument("--n_traj",    type=int,   default=N_TRAJ)
    parser.add_argument("--max_steps", type=int,   default=MAX_STEPS)
    parser.add_argument("--threshold", type=float, default=LC_THRESHOLD)
    parser.add_argument("--true_dist", type=float, default=TRUE_LC_DIST_MM)
    parser.add_argument("--noise_xy",      type=float, default=ODOM_NOISE_XY_MM)
    parser.add_argument("--noise_yaw",     type=float, default=ODOM_NOISE_YAW_DEG)
    parser.add_argument("--heading_gate",  type=float, default=HEADING_GATE_DEG,
                        help="Max heading difference (deg) for a valid LC (0 = disable)")
    parser.add_argument("--min_odom_dist", type=float, default=MIN_ODOM_DIST_MM,
                        help="Min dead-reckoning distance (mm) for a non-trivial LC (0 = disable)")
    parser.add_argument("--min_place_members", type=int, default=3,
                        help="Min LC-connected steps to qualify as a place (0 = skip place map)")
    parser.add_argument("--seed",          type=int,   default=0)
    args = parser.parse_args()

    rng        = np.random.default_rng(args.seed)
    run_name   = os.path.basename(args.run_dir.rstrip("/"))
    output_dir = os.path.join("SpatialInfo", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRun: {run_name}")
    mod, cfg, policy, is_burst = load_run(args.run_dir)
    print(f"  Type: {'burst' if is_burst else 'sonar'}  |  history_len={cfg.history_len}")

    print("\nCollecting trajectories...")
    positions, yaws, meas_seq, traj_ids, simulators = collect_data(
        mod, cfg, policy, args.n_traj, args.max_steps, rng
    )

    print(f"\nBuilding feature windows (w={args.window})...")
    feats = build_windows(meas_seq, traj_ids, args.window)

    print(f"\nSimulating noisy dead-reckoning "
          f"(noise_xy={args.noise_xy:.0f} mm/step, "
          f"noise_yaw={args.noise_yaw:.1f}°/step)...")
    noisy_pos, noisy_yaws = simulate_odometry(
        positions, yaws, traj_ids, args.noise_xy, args.noise_yaw, rng
    )

    print("\nComputing pairwise measurement distances...")
    i_valid, j_valid = get_valid_pairs(traj_ids, MIN_STEP_GAP)

    # Apply heading gate using noisy odometry yaw
    if args.heading_gate > 0:
        heading_mask = apply_heading_gate(i_valid, j_valid, noisy_yaws, args.heading_gate)
        i_valid = i_valid[heading_mask]
        j_valid = j_valid[heading_mask]
        print(f"  After heading gate (±{args.heading_gate:.0f}°): {len(i_valid):,} pairs")
    else:
        print(f"  Valid pairs (no heading gate): {len(i_valid):,}")

    # Apply minimum dead-reckoning distance gate — only non-trivial loop closures
    # (pairs odometry would not already flag as nearby)
    if args.min_odom_dist > 0:
        odom_dists  = compute_gt_distances(noisy_pos, i_valid, j_valid)
        nontrivial  = odom_dists >= args.min_odom_dist
        i_valid     = i_valid[nontrivial]
        j_valid     = j_valid[nontrivial]
        print(f"  After odometry distance gate (>{args.min_odom_dist:.0f} mm): {len(i_valid):,} pairs")

    meas_dists = compute_meas_distances(feats, i_valid, j_valid)
    gt_dists   = compute_gt_distances(positions, i_valid, j_valid)

    print("\nPrecision-recall sweep...")
    thresholds, precisions, recalls, total_tp = precision_recall_sweep(
        meas_dists, gt_dists
    )
    print(f"  Ground-truth close pairs (d < {args.true_dist:.0f} mm): {total_tp:,}")

    print(f"\nDetecting loop closures (threshold={args.threshold})...")
    lc_mask = meas_dists < args.threshold
    lc_i    = i_valid[lc_mask]
    lc_j    = j_valid[lc_mask]
    is_tp   = gt_dists[lc_mask] < args.true_dist
    n_tp    = int(is_tp.sum())
    n_fp    = int((~is_tp).sum())
    prec    = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else float("nan")
    op_k    = int(np.argmin(np.abs(thresholds - args.threshold)))
    rec     = float(recalls[op_k]) if len(recalls) else float("nan")
    print(f"  Detected: {len(lc_i):,}  |  TP: {n_tp:,}  FP: {n_fp:,}  "
          f"|  precision: {prec:.3f}  recall: {rec:.3f}")

    print("\nSaving loop closure plot...")
    plot_results(
        positions, noisy_pos, traj_ids,
        lc_i, lc_j, is_tp,
        thresholds, precisions, recalls,
        args.threshold, args.true_dist,
        simulators, run_name, output_dir, rng,
    )

    if args.min_place_members > 0:
        print(f"\nBuilding topological place map  "
              f"(threshold={args.threshold}, min_members={args.min_place_members})...")
        labels = build_topological_map(lc_i, lc_j, len(positions), args.min_place_members)
        n_places   = len(set(labels.tolist()) - {-1})
        n_corridor = int((labels == -1).sum())
        print(f"  Found {n_places} places, {n_corridor} corridor steps")
        plot_place_map(positions, traj_ids, labels, lc_i, lc_j, simulators,
                       run_name, output_dir, args.threshold, args.min_place_members)

    print("\nDone.")


if __name__ == "__main__":
    main()
