#!/usr/bin/env python3
"""
Spatial specificity assessment: landmark analysis via fingerprint clustering.

Runs N trials of a trained policy in a single session.  For each trajectory
step a sliding-window fingerprint is computed from the last W (IID, dist)
measurements.

Landmarks are found by clustering in fingerprint space (k-means, N_CLUSTERS
clusters), then assessing each cluster for:

  Recurrence    — how many distinct episodes contributed members?
                  High = the fingerprint pattern was seen in many runs.
  Spatial spread — mean pairwise pose distance among cluster members.
                  Low  = all members came from the same place → good landmark.
                  High = the fingerprint occurs at many different places → ambiguous.

Pose distance combines translation and rotation:
    pose_dist = sqrt(dx² + dy² + (YAW_SCALE_MM · dθ_deg)²)
"""

# ── Settings ──────────────────────────────────────────────────────────────────
POLICY_DIR   = "Policy/memory05"
GENERATION   = "last"      # integer, "last", or None for best_policy.json
SESSION      = "sessionB01"
N_TRIALS     = 20
MAX_STEPS    = 250
SEED         = 42
WINDOW_LEN   = 5          # fingerprint window length W
FEATURES     = "dist"      # "both" = (IID, dist)  |  "dist" = dist only
N_CLUSTERS   = 50          # number of k-means clusters in fingerprint space
YAW_SCALE_MM = 2.5         # mm per degree  (20° ≈ 100 mm robot body length)
OUTPUT_DIR   = "SpatialSpecificity"
# ─────────────────────────────────────────────────────────────────────────────

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


# ── I/O ───────────────────────────────────────────────────────────────────────

def find_json(policy_dir: str, generation) -> str:
    gen_dir = os.path.join(policy_dir, "generation_best")
    if generation is None:
        path = os.path.join(policy_dir, "best_policy.json")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"best_policy.json not found in {policy_dir}")
    if str(generation).lower() == "last":
        files = sorted(glob.glob(os.path.join(gen_dir, "gen_*_best_policy.json")))
        if not files:
            raise FileNotFoundError(f"No generation JSON files found in {gen_dir}")
        return files[-1]
    gen_num = int(generation)
    path = os.path.join(gen_dir, f"gen_{gen_num:03d}_best_policy.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    return path


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_trials(data: dict, session: str, n_trials: int, max_steps: int, seed: int):
    """Run n_trials episodes and return (sim, list_of_episodes)."""
    import random as _random
    from SCRIPT_TrainPolicy import HistoryNNPolicy, Evaluator, Config, build_simulator

    policy = HistoryNNPolicy(
        max_rotate1_deg=data["max_rotate1_deg"],
        max_rotate2_deg=data["max_rotate2_deg"],
        deadband_db=data["iid_deadband_db"],
        history_len=data["history_len"],
        hidden_sizes=tuple(data["hidden_sizes"]),
    )
    policy.set_genome(np.array(data["genome"], dtype=np.float32))

    sim = build_simulator(session)
    cfg = Config()
    cfg.max_steps               = max_steps
    cfg.history_len             = data["history_len"]
    cfg.max_rotate1_deg         = data["max_rotate1_deg"]
    cfg.max_rotate2_deg         = data["max_rotate2_deg"]
    cfg.iid_deadband_db         = data["iid_deadband_db"]
    cfg.quiet_setup             = True
    cfg.use_empirical_starts    = True
    cfg.randomize_empirical_yaw = True

    ev  = Evaluator(sim, cfg)
    rng = _random.Random(seed)

    episodes = []
    for i in range(n_trials):
        start = ev.sample_start(rng)
        ep    = ev.episode(policy, start)
        episodes.append(ep)
        print(f"  trial {i+1}/{n_trials}: {len(ep.get('trajectory', []))} steps, "
              f"fitness={ep['fitness']:.3f}")

    return sim, episodes


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_windows(episodes: list, window_len: int, features: str = "both"):
    """Extract sliding windows of length W at each trajectory step.

    features:
        "both" — normalised [iid, dist] interleaved  → fingerprint dim = 2*W
        "dist" — normalised [dist] only              → fingerprint dim = W

    Returns:
        fingerprints : np.ndarray (N, dim)  — normalised sequences
        poses        : np.ndarray (N, 3)    — (x, y, yaw_deg) at last step of window
        ep_ids       : np.ndarray (N,)      — episode index
    """
    fingerprints, poses, ep_ids = [], [], []
    for ep_i, ep in enumerate(episodes):
        traj = ep.get("trajectory", [])
        for t in range(window_len - 1, len(traj)):
            window = traj[t - window_len + 1 : t + 1]
            fp = []
            for step in window:
                dist_n = float(np.clip(step["distance_mm"] / 2000.0, 0.0, 2.0))
                if features == "both":
                    iid_n = float(np.clip(step["iid_db"] / 12.0, -2.0, 2.0))
                    fp.extend([iid_n, dist_n])
                else:
                    fp.append(dist_n)
            fingerprints.append(fp)
            poses.append([traj[t]["x"], traj[t]["y"], traj[t]["yaw_deg"]])
            ep_ids.append(ep_i)

    return (np.array(fingerprints, dtype=np.float32),
            np.array(poses,        dtype=np.float32),
            np.array(ep_ids,       dtype=np.int32))


# ── Pose distance ──────────────────────────────────────────────────────────────

def mean_pairwise_pose_dist(poses: np.ndarray, yaw_scale_mm: float) -> float:
    """Mean pairwise pose distance for a set of poses (N, 3)."""
    if len(poses) < 2:
        return 0.0
    xy  = poses[:, :2]
    yaw = poses[:, 2]
    sq  = np.sum(xy ** 2, axis=1, keepdims=True)
    D2  = np.maximum(sq + sq.T - 2.0 * (xy @ xy.T), 0.0)
    dyaw = yaw[:, None] - yaw[None, :]
    dyaw = (dyaw + 180.0) % 360.0 - 180.0
    pd   = np.sqrt(D2 + (yaw_scale_mm * dyaw) ** 2)
    # Upper triangle only
    n = len(poses)
    tri = np.triu_indices(n, k=1)
    return float(pd[tri].mean()) if len(tri[0]) > 0 else 0.0


# ── Clustering & assessment ────────────────────────────────────────────────────

def cluster_and_assess(fp: np.ndarray, poses: np.ndarray, ep_ids: np.ndarray,
                        n_clusters: int, yaw_scale_mm: float, seed: int = 42):
    """K-means cluster in fingerprint space, then assess each cluster spatially.

    Returns a list of dicts, one per cluster, sorted by spatial spread (ascending):
        label        : int   — cluster index
        size         : int   — number of member windows
        recurrence   : int   — distinct episodes represented
        spatial_spread: float — mean pairwise pose distance of members (mm)
        member_poses : np.ndarray (M, 3) — poses of all members
        member_ep_ids: np.ndarray (M,)   — episode ids of all members
        centroid     : np.ndarray (D,)   — fingerprint centroid
    """
    print(f"  Clustering {len(fp)} windows into {n_clusters} fingerprint clusters ...")
    km     = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(fp)

    clusters = []
    for c in range(n_clusters):
        mask        = labels == c
        m_poses     = poses[mask]
        m_ep_ids    = ep_ids[mask]
        recurrence  = int(len(np.unique(m_ep_ids)))
        spread      = mean_pairwise_pose_dist(m_poses, yaw_scale_mm)
        clusters.append({
            "label":         c,
            "size":          int(mask.sum()),
            "recurrence":    recurrence,
            "spatial_spread": spread,
            "member_poses":  m_poses,
            "member_ep_ids": m_ep_ids,
            "centroid":      km.cluster_centers_[c],
        })

    clusters.sort(key=lambda x: x["spatial_spread"])
    return clusters, labels


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_cluster_assessment(clusters: list, episodes: list, poses: np.ndarray,
                             labels: np.ndarray, sim,
                             session: str, gen, output_dir: str,
                             yaw_scale_mm: float, features: str) -> None:
    """Two-panel plot.

    Left:  arena map with all trajectory points coloured by cluster assignment.
    Right: scatter of recurrence vs spatial spread, one point per cluster.
           Good landmarks = top-left (high recurrence, low spread).
           Point size ∝ cluster size.
    """
    os.makedirs(output_dir, exist_ok=True)
    feat_label = "IID+dist" if features == "both" else "dist only"
    n_clusters = len(clusters)
    cmap       = plt.cm.get_cmap("tab20", n_clusters)
    walls      = getattr(sim.arena, "walls", None)
    pos        = poses[:, :2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: arena map coloured by cluster
    ax = axes[0]
    if walls is not None and len(walls) > 0:
        ax.scatter(walls[:, 0], walls[:, 1], s=1, color="#bdbdbd", alpha=0.3, zorder=0)
    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(pos[mask, 0], pos[mask, 1], s=6, color=cmap(c),
                   alpha=0.6, linewidths=0, zorder=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title("Arena map coloured by fingerprint cluster\n"
                 "(each colour = one k-means cluster)",
                 fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Right: recurrence vs spatial spread scatter
    ax = axes[1]
    n_eps   = len(episodes)
    sizes   = np.array([c["size"] for c in clusters], dtype=float)
    sizes_s = 30 + 200 * (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1)
    for i, cl in enumerate(clusters):
        ax.scatter(cl["recurrence"], cl["spatial_spread"],
                   s=float(sizes_s[i]), color=cmap(cl["label"]),
                   alpha=0.8, edgecolors="black", linewidths=0.4, zorder=3)
        ax.annotate(str(cl["label"]), (cl["recurrence"], cl["spatial_spread"]),
                    fontsize=6, ha="center", va="center", zorder=4)

    ax.set_xlabel("Recurrence (distinct episodes in cluster)", fontsize=9)
    ax.set_ylabel("Spatial spread — mean pairwise pose dist (mm)", fontsize=9)
    ax.set_title("Cluster quality\nBottom-right = high recurrence + low spread = good landmark",
                 fontsize=9, fontweight="bold")
    ax.axvline(n_eps * 0.5, color="#888888", linewidth=1, linestyle=":",
               label=f"50% of episodes ({n_eps//2})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    feat_tag = "both" if features == "both" else "dist"
    fig.suptitle(
        f"Fingerprint cluster assessment  [{feat_label}]  |  {session}  |  gen {gen}  |  W={WINDOW_LEN}\n"
        f"{n_clusters} clusters  |  yaw_scale={yaw_scale_mm} mm/°  |  point size ∝ cluster size",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(output_dir, f"plot_clusters_{feat_tag}_gen{gen}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    # Print summary table (top 10 by recurrence among low-spread clusters)
    print(f"\n  {'Cluster':>7}  {'Size':>6}  {'Recurrence':>11}  {'Spread (mm)':>12}")
    print("  " + "-" * 42)
    sorted_by_rec = sorted(clusters, key=lambda x: (-x["recurrence"], x["spatial_spread"]))
    for cl in sorted_by_rec[:15]:
        print(f"  {cl['label']:>7}  {cl['size']:>6}  {cl['recurrence']:>11}  "
              f"{cl['spatial_spread']:>12.0f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def abs_path(p):
        return p if os.path.isabs(p) else os.path.join(script_dir, p)

    json_path = find_json(abs_path(POLICY_DIR), GENERATION)
    print(f"Loading: {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    gen = data.get("generation", "best")

    print(f"Running {N_TRIALS} trials in {SESSION} ...")
    sim, episodes = run_trials(data, SESSION, N_TRIALS, MAX_STEPS, SEED)
    steps_total   = sum(len(ep.get("trajectory", [])) for ep in episodes)
    print(f"Total steps collected: {steps_total}")

    out = abs_path(OUTPUT_DIR)

    for features in ("both", "dist"):
        print(f"\n── features={features} ──")
        fp, poses, ep_ids = extract_windows(episodes, WINDOW_LEN, features=features)
        clusters, labels  = cluster_and_assess(fp, poses, ep_ids,
                                                N_CLUSTERS, YAW_SCALE_MM, SEED)
        plot_cluster_assessment(clusters, episodes, poses, labels, sim,
                                SESSION, gen, out, YAW_SCALE_MM, features)


if __name__ == "__main__":
    main()
