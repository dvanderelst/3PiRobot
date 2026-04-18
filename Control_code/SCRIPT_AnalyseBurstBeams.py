#!/usr/bin/env python3
"""
SCRIPT_AnalyseBurstBeams.py

Analyses the beam-aiming behaviour of a policy trained with
SCRIPT_TrainPolicy_Burst.py.

Loads best_policy.json (safe to run while training is still in progress)
and runs analysis episodes to answer:
  - Where does each look slot point (distribution per slot)?
  - Do later looks react to earlier measurements within the burst?
  - How does the drive rotation relate to the accumulated burst evidence?
  - How wide is the scan sweep within a step?

Plots are saved to <OUTPUT_DIR>/beam_analysis/.
"""

import collections
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, List, Optional, Tuple

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Library.EnvironmentSimulator import EnvironmentSimulator


# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR   = "PolicyTraining/burst_h03"  # path to training output directory
SESSION_NAME = None   # None = use first session from config.json
N_EPISODES   = 300    # number of analysis episodes to run
SEED         = 0
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Minimal MLP — matches SCRIPT_TrainPolicy_Burst.py exactly
# ══════════════════════════════════════════════════════════════════════════════

class MLP:
    def __init__(self, in_dim: int, hidden_sizes: Tuple[int, int], n_out: int):
        h1, h2 = hidden_sizes
        self.shapes = [
            (h1,    in_dim), (h1,),
            (h2,    h1),     (h2,),
            (n_out, h2),     (n_out,),
        ]
        self.params: List[np.ndarray] = [np.zeros(s, dtype=np.float32) for s in self.shapes]

    def set_genome(self, genome: np.ndarray) -> None:
        g = np.asarray(genome, dtype=np.float32).ravel()
        off = 0
        self.params = []
        for s in self.shapes:
            n = int(np.prod(s))
            self.params.append(g[off:off + n].reshape(s))
            off += n

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Returns all N_LOOKS+1 outputs as a (N_LOOKS+1,) array in [-1, 1]."""
        v = x.reshape(-1, 1)
        w1, b1, w2, b2, w3, b3 = self.params
        h = np.tanh(w1 @ v + b1.reshape(-1, 1))
        h = np.tanh(w2 @ h + b2.reshape(-1, 1))
        return np.tanh(w3 @ h + b3.reshape(-1, 1)).ravel()


# ══════════════════════════════════════════════════════════════════════════════
# Input construction — mirrors SCRIPT_TrainPolicy_Burst.py exactly
# ══════════════════════════════════════════════════════════════════════════════

def build_input(
    history: collections.deque,
    current_measurements: List[Tuple[float, float, float]],
    n_looks: int,
    max_dist_mm: float,
    max_iid_db: float,
    max_rotate1_deg: float,
    max_rotate2_deg: float,
) -> np.ndarray:
    """
    history entries: tuples of length 3*n_looks + 1
      (d1, i1, l1, d2, i2, l2, ..., dN, iN, lN, r2)
    current_measurements: list of (dist_mm, iid_canonical, r1_canonical) for
      completed looks this step.
    """
    md, mi, mr1, mr2 = max_dist_mm, max_iid_db, max_rotate1_deg, max_rotate2_deg
    parts: List[float] = []
    for k in range(n_looks):
        d_hist = [h[3 * k]     / md  for h in history]
        i_hist = [h[3 * k + 1] / mi  for h in history]
        l_hist = [h[3 * k + 2] / mr1 for h in history]
        if k < len(current_measurements):
            d_curr = current_measurements[k][0] / md
            i_curr = current_measurements[k][1] / mi
            l_curr = current_measurements[k][2] / mr1
        else:
            d_curr = i_curr = l_curr = 0.0
        parts += d_hist + [d_curr]
        parts += i_hist + [i_curr]
        parts += l_hist + [l_curr]
    parts += [h[3 * n_looks] / mr2 for h in history]
    return np.array(parts, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Episode runner (no noise, no overrides — clean emulator output)
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis_episodes(
    mlp: MLP,
    simulator: EnvironmentSimulator,
    starts: List[Tuple[float, float, float]],
    n_looks: int,
    history_len: int,
    max_rotate1_deg: float,
    max_rotate2_deg: float,
    max_net_rotation_deg: float,
    max_dist_mm: float,
    min_dist_mm: float,
    max_iid_db: float,
    fixed_drive_mm: float,
    max_steps: int,
    n_episodes: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """Run episodes and return per-step beam data."""
    empty_entry = (0.0,) * (3 * n_looks + 1)
    episodes: List[Dict] = []

    for _ in range(n_episodes):
        if not starts:
            break
        x, y, yaw = starts[int(rng.integers(len(starts)))]
        history: collections.deque = collections.deque(
            [empty_entry] * history_len, maxlen=history_len
        )
        last_physical_iid = 0.0
        positions = [(float(x), float(y))]
        collided = False
        steps: List[Dict] = []

        for step_idx in range(max_steps):
            original_yaw = yaw
            current_measurements: List[Tuple[float, float, float]] = []
            looks: List[Dict] = []

            # ── Look phase: plan all looks in one forward pass ────────────────
            flip = last_physical_iid < 0.0
            inp  = build_input(history, [], n_looks,
                               max_dist_mm, max_iid_db, max_rotate1_deg, max_rotate2_deg)
            raw  = mlp.forward(inp)   # (n_looks+1,) in [-1, 1]
            look_cans  = [float(raw[k]) * max_rotate1_deg for k in range(n_looks)]
            look_phys  = [-lc if flip else lc for lc in look_cans]

            # ── Measure at all planned look directions ────────────────────────
            last_meas_iid = last_physical_iid
            for k in range(n_looks):
                look_yaw = original_yaw + look_phys[k]

                meas = simulator.get_sonar_measurement(x, y, look_yaw)
                dist_mm  = max(min_dist_mm,
                               min(float(meas.get("distance_mm", max_dist_mm)), max_dist_mm))
                phys_iid = float(meas.get("iid_db", 0.0))
                iid_can  = abs(phys_iid)

                current_measurements.append((dist_mm, iid_can, look_cans[k]))
                last_meas_iid = phys_iid

                looks.append({
                    "r1_canonical":  look_cans[k],
                    "r1_physical":   look_phys[k],
                    "look_yaw":      look_yaw,
                    "dist_mm":       dist_mm,
                    "iid_physical":  phys_iid,
                    "iid_canonical": iid_can,
                    "flipped":       flip,
                })

            # ── Drive phase: second forward pass with current measurements ────
            inp     = build_input(history, current_measurements, n_looks,
                                  max_dist_mm, max_iid_db, max_rotate1_deg, max_rotate2_deg)
            raw     = mlp.forward(inp)
            r2_can  = float(raw[n_looks]) * max_rotate2_deg
            r2_phys = -r2_can if flip else r2_can
            r2_phys = float(np.clip(r2_phys, -max_net_rotation_deg, max_net_rotation_deg))
            r2_can  = -r2_phys if flip else r2_phys

            action = {"rotate1_deg": 0.0, "rotate2_deg": r2_phys, "drive_mm": fixed_drive_mm}
            result = simulator.simulate_robot_movement(
                x, y, original_yaw, [action], compute_sonar=False
            )[0]
            x   = float(result["position"]["x"])
            y   = float(result["position"]["y"])
            yaw = float(result["orientation"])
            blocked = bool(result["collision"]["drive_blocked"])

            positions.append((x, y))
            hist_entry = (
                tuple(v for d, i, l in current_measurements for v in (d, i, l))
                + (r2_can,)
            )
            history.append(hist_entry)
            last_physical_iid = last_meas_iid

            steps.append({
                "step":         step_idx,
                "yaw":          original_yaw,
                "looks":        looks,
                "r2_physical":  r2_phys,
                "r2_canonical": r2_can,
            })

            if blocked:
                collided = True
                break

        episodes.append({
            "positions": positions,
            "collided":  collided,
            "steps":     steps,
        })

    return episodes


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _binned_mean(x: np.ndarray, y: np.ndarray, n_bins: int = 15):
    """Return (bin_centres, bin_means) for scatter trend line."""
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.digitize(x, edges)
    bx, by = [], []
    for b in range(1, n_bins + 1):
        mask = idx == b
        if mask.sum() >= 5:
            bx.append(x[mask].mean())
            by.append(y[mask].mean())
    return np.array(bx), np.array(by)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_look_distributions(r1_can, n_looks, max_r1, out_dir):
    """Histogram of look angle per slot in canonical frame."""
    fig, axes = plt.subplots(1, n_looks, figsize=(4 * n_looks, 4), sharey=True)
    if n_looks == 1:
        axes = [axes]
    bins = np.linspace(-max_r1, max_r1, 40)
    for k, ax in enumerate(axes):
        vals = np.array(r1_can[k])
        ax.hist(vals, bins=bins, color=f"C{k}", alpha=0.75, edgecolor="white")
        ax.axvline(vals.mean(), color="k", linestyle="--", linewidth=1.2,
                   label=f"μ={vals.mean():.1f}°")
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"Look {k+1}\nμ={vals.mean():.1f}°  σ={vals.std():.1f}°", fontsize=10)
        ax.set_xlabel("r1_canonical (°)")
        if k == 0:
            ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Look angle distributions (canonical frame: IID≥0 = wall on right)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "look_distributions.png"), dpi=120)
    plt.close(fig)
    print("  Saved: look_distributions.png")


def plot_scan_summary(r1_can, r2_can_all, n_looks, out_dir):
    """Bar chart of mean ± std for each look slot and drive."""
    labels = [f"Look {k+1}" for k in range(n_looks)] + ["Drive\n(r2)"]
    all_vals = r1_can + [r2_can_all]
    means  = [np.mean(v) for v in all_vals]
    stds   = [np.std(v)  for v in all_vals]
    colors = [f"C{k}" for k in range(n_looks)] + ["C3"]

    fig, ax = plt.subplots(figsize=(max(5, 2 * (n_looks + 1)), 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.75, capsize=6, width=0.6)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Canonical rotation angle (°)  [mean ± std]")
    ax.set_title("Scan pattern summary — canonical frame")
    ax.grid(True, alpha=0.3, axis="y")
    # Annotate bars with mean value
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + (s + 1) * np.sign(m) if abs(m) > 1 else s + 2,
                f"{m:+.1f}°", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scan_summary.png"), dpi=120)
    plt.close(fig)
    print("  Saved: scan_summary.png")


def plot_sequential_reactivity(r1_can, dist, iid_can, n_looks, out_dir):
    """
    For each pair (look k → look k+1): scatter r1_{k+1} vs iid_k and dist_k.
    Shows whether the network redirects its gaze based on what look k found.
    """
    if n_looks < 2:
        return
    n_pairs = n_looks - 1
    fig, axes = plt.subplots(n_pairs, 2, figsize=(10, 4 * n_pairs), squeeze=False)
    fig.suptitle("Sequential reactivity: where does look k+1 point given look k's result?",
                 fontsize=11)

    for k in range(n_pairs):
        x_iid  = np.array(iid_can[k])
        x_dist = np.array(dist[k])
        y_r1   = np.array(r1_can[k + 1])

        for col, (x_vals, xlabel) in enumerate([
            (x_iid,  f"|IID| from look {k+1} (dB)"),
            (x_dist, f"Distance from look {k+1} (mm)"),
        ]):
            ax = axes[k, col]
            ax.scatter(x_vals, y_r1, s=1.5, alpha=0.15, color=f"C{k}", rasterized=True)
            bx, by = _binned_mean(x_vals, y_r1)
            if len(bx) > 1:
                ax.plot(bx, by, "k-o", markersize=4, linewidth=1.5, label="binned mean", zorder=3)
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(f"r1_canonical  look {k+2} (°)", fontsize=9)
            ax.grid(True, alpha=0.3)
            if len(bx) > 1:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sequential_reactivity.png"), dpi=120)
    plt.close(fig)
    print("  Saved: sequential_reactivity.png")


def plot_look_correlations(r1_can, n_looks, out_dir):
    """
    Scatter r1_{k+1} vs r1_k for each consecutive pair.
    Reveals systematic patterns (e.g. always opposite, always same direction).
    """
    if n_looks < 2:
        return
    n_pairs = n_looks - 1
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4), squeeze=False)
    fig.suptitle("Look-to-look angle correlation", fontsize=11)

    for k in range(n_pairs):
        ax = axes[0, k]
        x = np.array(r1_can[k])
        y = np.array(r1_can[k + 1])
        ax.scatter(x, y, s=1.5, alpha=0.15, color=f"C{k}", rasterized=True)
        bx, by = _binned_mean(x, y)
        if len(bx) > 1:
            ax.plot(bx, by, "k-o", markersize=4, linewidth=1.5, label="binned mean", zorder=3)
        # Diagonal reference lines
        lim = max(abs(x).max(), abs(y).max()) * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.plot([-lim, lim], [-lim, lim], "gray", linestyle="--",
                linewidth=0.7, label="same dir")
        ax.plot([-lim, lim], [lim, -lim], "gray", linestyle="-.",
                linewidth=0.7, label="opposite dir")
        ax.set_xlabel(f"r1_canonical  look {k+1} (°)", fontsize=9)
        ax.set_ylabel(f"r1_canonical  look {k+2} (°)", fontsize=9)
        ax.set_title(f"Look {k+1} → Look {k+2}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "look_correlations.png"), dpi=120)
    plt.close(fig)
    print("  Saved: look_correlations.png")


def plot_drive_vs_measurements(dist, iid_can, r2_can_all, n_looks, out_dir):
    """
    r2 vs IID and distance from the final look, and vs the overall min distance.
    """
    x_iid_last  = np.array(iid_can[n_looks - 1])
    x_dist_last = np.array(dist[n_looks - 1])
    # min distance across all looks (proxy for how close the nearest wall is)
    x_dist_min  = np.array([min(dist[k][i] for k in range(n_looks))
                             for i in range(len(r2_can_all))])
    y_r2 = np.array(r2_can_all)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Drive rotation r2 vs burst measurements (canonical frame)", fontsize=11)

    for ax, x_vals, xlabel in zip(
        axes,
        [x_iid_last, x_dist_last, x_dist_min],
        [f"|IID| from look {n_looks} (dB)",
         f"Distance from look {n_looks} (mm)",
         "Min distance across all looks (mm)"],
    ):
        ax.scatter(x_vals, y_r2, s=1.5, alpha=0.15, color="C3", rasterized=True)
        bx, by = _binned_mean(x_vals, y_r2)
        if len(bx) > 1:
            ax.plot(bx, by, "k-o", markersize=4, linewidth=1.5, label="binned mean", zorder=3)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("r2_canonical (°)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drive_vs_measurements.png"), dpi=120)
    plt.close(fig)
    print("  Saved: drive_vs_measurements.png")


def plot_scan_spread(all_steps, n_looks, out_dir):
    """
    Distribution of within-step scan spread (physical frame) and
    scatter of spread vs min distance.
    """
    spreads    = []
    min_dists  = []
    for s in all_steps:
        angles = [lk["r1_physical"] for lk in s["looks"]]
        dists  = [lk["dist_mm"]     for lk in s["looks"]]
        spreads.append(max(angles) - min(angles))
        min_dists.append(min(dists))

    spreads   = np.array(spreads)
    min_dists = np.array(min_dists)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Within-step scan spread (physical frame)", fontsize=11)

    axes[0].hist(spreads, bins=35, color="steelblue", alpha=0.75, edgecolor="white")
    axes[0].axvline(spreads.mean(), color="k", linestyle="--", linewidth=1.2,
                    label=f"mean={spreads.mean():.1f}°")
    axes[0].axvline(np.median(spreads), color="C1", linestyle="--", linewidth=1.2,
                    label=f"median={np.median(spreads):.1f}°")
    axes[0].set_xlabel("Scan spread (max − min look angle, °)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(min_dists, spreads, s=1.5, alpha=0.15, color="steelblue", rasterized=True)
    bx, by = _binned_mean(min_dists, spreads)
    if len(bx) > 1:
        axes[1].plot(bx, by, "k-o", markersize=4, linewidth=1.5, label="binned mean", zorder=3)
    axes[1].set_xlabel("Min distance across looks (mm)")
    axes[1].set_ylabel("Scan spread (°)")
    axes[1].set_title("Does the robot sweep more when walls are close?")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scan_spread.png"), dpi=120)
    plt.close(fig)
    print("  Saved: scan_spread.png")


def plot_example_trajectories(episodes, simulator, n_looks, out_dir, n_show=6):
    """
    Trajectory plot with arrows for every look direction, coloured by episode.
    One arrow per look per step (every 5 steps to avoid clutter).
    """
    arrow_len   = 150.0
    arrow_every = 5
    colours     = plt.cm.tab10(np.linspace(0, 1, n_show))

    fig, ax = plt.subplots(figsize=(8, 7))
    walls = simulator.arena.walls
    if len(walls) > 0:
        ax.scatter(walls[:, 0], walls[:, 1], s=0.5, c="#cccccc", linewidths=0, zorder=1)

    for ep, colour in zip(episodes[:n_show], colours):
        xs = [p[0] for p in ep["positions"]]
        ys = [p[1] for p in ep["positions"]]
        ax.plot(xs, ys, color=colour, linewidth=0.9,
                linestyle="--" if ep["collided"] else "-", zorder=2)
        ax.plot(xs[0],  ys[0],  "o", color=colour, markersize=4, zorder=3)
        ax.plot(xs[-1], ys[-1], "x", color=colour, markersize=5, zorder=3)

        for step, s in enumerate(ep["steps"]):
            if step % arrow_every != 0:
                continue
            px, py = xs[step], ys[step]
            for k, lk in enumerate(s["looks"]):
                alpha = 0.8 - 0.2 * k   # earlier looks slightly more opaque
                rad = np.deg2rad(lk["look_yaw"])
                ax.quiver(px, py,
                          np.cos(rad) * arrow_len, np.sin(rad) * arrow_len,
                          angles="xy", scale_units="xy", scale=1,
                          color=colour, alpha=alpha, width=0.002,
                          headwidth=3, headlength=3, zorder=4)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")
    ax.set_title(f"Example trajectories with look directions (every {arrow_every} steps)\n"
                 f"Opacity: look 1 > look 2 > look 3")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "example_trajectories.png"), dpi=120)
    plt.close(fig)
    print("  Saved: example_trajectories.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(SEED)

    # ── Load policy ───────────────────────────────────────────────────────────
    policy_path = os.path.join(OUTPUT_DIR, "best_policy.json")
    if not os.path.exists(policy_path):
        print(f"ERROR: no best_policy.json found in '{OUTPUT_DIR}'")
        print("Make sure training has completed at least one generation.")
        return

    with open(policy_path) as f:
        pol_data = json.load(f)

    n_looks          = int(pol_data["n_looks"])
    history_len      = int(pol_data["history_len"])
    hidden_sizes     = tuple(pol_data["hidden_sizes"])
    max_rotate1_deg  = float(pol_data["max_rotate1_deg"])
    max_rotate2_deg  = float(pol_data["max_rotate2_deg"])
    max_net_rot_deg  = float(pol_data.get("max_net_rotation_deg", max_rotate2_deg))
    max_dist_mm      = float(pol_data["max_dist_mm"])
    max_iid_db       = float(pol_data["max_iid_db"])
    fixed_drive_mm   = float(pol_data.get("fixed_drive_mm", 100.0))
    genome           = np.array(pol_data["genome"], dtype=np.float32)
    gen_saved        = int(pol_data.get("generation", -1))
    fitness_saved    = float(pol_data.get("fitness", float("nan")))

    # ── Load config for extra fields not saved in best_policy.json ────────────
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    max_steps = 75
    starts_dir = "ValidStarts"
    starts_suffixes = ["starts_headon", "starts_wall_left", "starts_wall_right"]
    min_dist_mm = 300.0
    train_sessions: List[str] = []

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        max_steps       = int(cfg.get("max_steps", max_steps))
        starts_dir      = cfg.get("starts_dir", starts_dir)
        starts_suffixes = cfg.get("starts_suffixes", starts_suffixes)
        min_dist_mm     = float(cfg.get("min_dist_mm", min_dist_mm))
        train_sessions  = cfg.get("train_session_names", [])

    # ── Choose session ────────────────────────────────────────────────────────
    session = SESSION_NAME
    if session is None:
        if not train_sessions:
            print("ERROR: could not determine session name. "
                  "Set SESSION_NAME at the top of the script.")
            return
        session = train_sessions[0]

    print(f"Policy:   generation {gen_saved}  fitness={fitness_saved:.1f}")
    print(f"Session:  {session}")
    print(f"N_LOOKS:  {n_looks}   history_len={history_len}")
    print(f"Episodes: {N_EPISODES}")

    # ── Build MLP ─────────────────────────────────────────────────────────────
    in_dim = 3 * n_looks * (history_len + 1) + history_len
    n_out  = n_looks + 1   # look1..lookN + r2
    mlp = MLP(in_dim, hidden_sizes, n_out)
    mlp.set_genome(genome)

    # ── Build simulator and load starts ──────────────────────────────────────
    print("Loading simulator ...", flush=True)
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        simulator = EnvironmentSimulator(session)

    all_starts: List[Tuple[float, float, float]] = []
    for suffix in starts_suffixes:
        path = os.path.join(starts_dir, f"{session}_{suffix}.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            data = json.load(f)
        all_starts.extend(
            (float(s["x"]), float(s["y"]), float(s["yaw_deg"]))
            for s in data.get("starts", [])
        )
    if not all_starts:
        print(f"ERROR: no starts found for session '{session}' in '{starts_dir}'")
        return
    print(f"Starts:   {len(all_starts)} loaded")

    # ── Run episodes ──────────────────────────────────────────────────────────
    print(f"Running {N_EPISODES} analysis episodes ...", flush=True)
    episodes = run_analysis_episodes(
        mlp=mlp,
        simulator=simulator,
        starts=all_starts,
        n_looks=n_looks,
        history_len=history_len,
        max_rotate1_deg=max_rotate1_deg,
        max_rotate2_deg=max_rotate2_deg,
        max_net_rotation_deg=max_net_rot_deg,
        max_dist_mm=max_dist_mm,
        min_dist_mm=min_dist_mm,
        max_iid_db=max_iid_db,
        fixed_drive_mm=fixed_drive_mm,
        max_steps=max_steps,
        n_episodes=N_EPISODES,
        rng=rng,
    )

    # ── Aggregate per-look data ───────────────────────────────────────────────
    all_steps = [s for ep in episodes for s in ep["steps"]]
    r1_can  = [[] for _ in range(n_looks)]
    r1_phys = [[] for _ in range(n_looks)]
    dist    = [[] for _ in range(n_looks)]
    iid_can = [[] for _ in range(n_looks)]
    r2_can_all: List[float] = []

    for s in all_steps:
        for k, lk in enumerate(s["looks"]):
            r1_can[k].append(lk["r1_canonical"])
            r1_phys[k].append(lk["r1_physical"])
            dist[k].append(lk["dist_mm"])
            iid_can[k].append(lk["iid_canonical"])
        r2_can_all.append(s["r2_canonical"])

    # ── Print summary ─────────────────────────────────────────────────────────
    n_ep = len(episodes)
    n_st = len(all_steps)
    n_coll = sum(ep["collided"] for ep in episodes)
    print(f"\n{'='*60}")
    print(f"Episodes: {n_ep}   Steps: {n_st}   "
          f"Collisions: {n_coll}/{n_ep} ({100*n_coll/n_ep:.0f}%)")
    print(f"\nCanonical look angles (mean ± std):")
    for k in range(n_looks):
        vals = np.array(r1_can[k])
        print(f"  Look {k+1}: {vals.mean():+.1f}° ± {vals.std():.1f}°  "
              f"range [{vals.min():.1f}°, {vals.max():.1f}°]")
    r2_arr = np.array(r2_can_all)
    print(f"  Drive r2: {r2_arr.mean():+.1f}° ± {r2_arr.std():.1f}°")

    spreads = [max(lk["r1_physical"] for lk in s["looks"]) -
               min(lk["r1_physical"] for lk in s["looks"])
               for s in all_steps]
    print(f"\nScan spread (physical):  mean={np.mean(spreads):.1f}°  "
          f"median={np.median(spreads):.1f}°  max={np.max(spreads):.1f}°")

    # ── Generate plots ────────────────────────────────────────────────────────
    out_dir = os.path.join(OUTPUT_DIR, "beam_analysis")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving plots to {out_dir}/")

    plot_look_distributions(r1_can, n_looks, max_rotate1_deg, out_dir)
    plot_scan_summary(r1_can, r2_can_all, n_looks, out_dir)
    plot_sequential_reactivity(r1_can, dist, iid_can, n_looks, out_dir)
    plot_look_correlations(r1_can, n_looks, out_dir)
    plot_drive_vs_measurements(dist, iid_can, r2_can_all, n_looks, out_dir)
    plot_scan_spread(all_steps, n_looks, out_dir)
    plot_example_trajectories(episodes, simulator, n_looks, out_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
