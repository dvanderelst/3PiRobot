#!/usr/bin/env python3
"""
SCRIPT_MakePlots.py

Publication-ready plots assembled from data produced by other scripts.
All outputs go to PaperPlots/.

Sections
--------
1. Training curves  — best and mean fitness per generation across runs
"""

import json
import os

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = "PaperPlots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Training curves
# ══════════════════════════════════════════════════════════════════════════════

TRAINING_RUN_DIRS = [
    ("TrainedPolicies/run_h00_baseline", "Baseline"),
    ("TrainedPolicies/run_h01", "Hist 1"),
    ("TrainedPolicies/run_h03", "Hist 3"),
    ("TrainedPolicies/run_h05", "Hist 5"),
]

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── Rotation strategy plot ranges ─────────────────────────────────────────────
ROT1_RANGE    = (-45, 45)   # x-axis: look angle (rotate1), degrees
NET_ROT_RANGE = (-30, 30)   # y-axis: net body rotation (rotate1 + rotate2), degrees


def _fitness_components(run_dir: str) -> dict | None:
    """
    Recompute coverage, jitter_factor, and collision_free_rate from the
    episodes CSV, returning one value per policy (rank) so callers can
    show variability across policies.

    Returns dict mapping component name → list of per-policy means,
    or None if the CSV does not exist.
    """
    import csv as _csv

    csv_path = os.path.join(
        "PolicyAssessment",
        f"{os.path.basename(run_dir)}_episodes.csv",
    )
    if not os.path.exists(csv_path):
        return None

    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path) as f:
        cfg_d = json.load(f)

    angular_bin_deg = float(cfg_d.get("angular_bin_deg", 10.0))
    w_smooth        = float(cfg_d.get("w_smooth",         0.25))
    max_rotate1_deg = float(cfg_d.get("max_rotate1_deg", 15.0))
    max_rotate2_deg = float(cfg_d.get("max_rotate2_deg", 15.0))
    max_jerk        = 2.0 * (max_rotate1_deg + max_rotate2_deg)
    n_bins          = int(round(360.0 / angular_bin_deg))

    # Group rows by (rank, session, episode)
    episodes: dict = {}
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            key = (row["rank"], row["session"], row["episode"])
            episodes.setdefault(key, []).append(row)

    # Compute per-episode metrics, then average per policy (rank)
    by_rank: dict = {}
    for (rank, _session, _ep), rows in episodes.items():
        xs  = np.array([float(r["x"]) for r in rows])
        ys  = np.array([float(r["y"]) for r in rows])
        r1  = np.array([float(r["rotate1_canonical"]) for r in rows])
        r2  = np.array([float(r["rotate2_canonical"]) for r in rows])
        blocked = any(int(r["drive_blocked"]) for r in rows)

        cx, cy  = xs.mean(), ys.mean()
        angles  = np.rad2deg(np.arctan2(ys - cy, xs - cx)) % 360.0
        dists   = np.hypot(xs - cx, ys - cy)
        bin_idx = (angles / angular_bin_deg).astype(int) % n_bins
        sum_d   = np.zeros(n_bins);  counts = np.zeros(n_bins)
        np.add.at(sum_d,  bin_idx, dists)
        np.add.at(counts, bin_idx, 1)
        coverage = float(np.where(counts > 0, sum_d / np.maximum(counts, 1), 0.0).mean())

        net_turns = r1 + r2
        jitter_f  = 1.0
        if w_smooth > 0.0 and len(net_turns) >= 2:
            mean_jerk_norm = float(np.mean(np.abs(np.diff(net_turns)))) / max(max_jerk, 1e-6)
            jitter_f = max(0.0, 1.0 - w_smooth * mean_jerk_norm)

        by_rank.setdefault(rank, {"coverages": [], "jitter_factors": [], "collided": []})
        by_rank[rank]["coverages"].append(coverage)
        by_rank[rank]["jitter_factors"].append(jitter_f)
        by_rank[rank]["collided"].append(float(blocked))

    # One value per policy
    coverages, jitter_factors, collision_free = [], [], []
    for rank_data in by_rank.values():
        coverages.append(float(np.mean(rank_data["coverages"])))
        jitter_factors.append(float(np.mean(rank_data["jitter_factors"])))
        collision_free.append(1.0 - float(np.mean(rank_data["collided"])))

    return {
        "coverage":            coverages,
        "jitter_factor":       jitter_factors,
        "collision_free_rate": collision_free,
    }


def plot_training_curves() -> None:
    fig = plt.figure(figsize=(12, 6))
    gs  = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.1, wspace=0.35)
    ax_fit  = fig.add_subplot(gs[0, 0])
    ax_col  = fig.add_subplot(gs[1, 0], sharex=ax_fit)
    ax_comp = fig.add_subplot(gs[:, 1])

    for (run_dir, label), color in zip(TRAINING_RUN_DIRS, COLORS):
        hist_path = os.path.join(run_dir, "training_history.json")
        if not os.path.exists(hist_path):
            print(f"  ⚠ Not found: {hist_path}")
            continue

        with open(hist_path) as f:
            hist = json.load(f)

        gens  = np.arange(len(hist["best"]))
        best  = np.array([v if v is not None else np.nan for v in hist["best"]])
        val   = np.array([v if v is not None else np.nan for v in hist["val"]])
        coll  = np.array([v if v is not None else np.nan for v in hist["collision_rate"]])

        ax_fit.plot(gens, best, color=color, lw=1.8, label=label)
        ax_fit.plot(gens, val,  color=color, lw=1.0, linestyle="--", alpha=0.6)
        ax_col.plot(gens, coll, color=color, lw=1.5, label=label)

    from matplotlib.lines import Line2D
    handles, labels = ax_fit.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", lw=1.0, linestyle="--", alpha=0.6))
    labels.append("Validation")
    ax_fit.legend(handles=handles, labels=labels, fontsize=8, ncol=2)

    ax_fit.set_ylabel("Fitness (a.u.)")
    ax_fit.grid(True, alpha=0.3)
    plt.setp(ax_fit.get_xticklabels(), visible=False)

    ax_col.set_xlabel("Generation")
    ax_col.set_ylabel("Collision rate")
    ax_col.set_ylim(0, 1)
    ax_col.legend(fontsize=8)
    ax_col.grid(True, alpha=0.3)

    # ── Fitness components panel ──────────────────────────────────────────────
    comp_labels = ["Coverage\n(norm.)", "Smoothness\n(jitter factor)", "Collision-free\nrate"]
    comp_keys   = ["coverage", "jitter_factor", "collision_free_rate"]

    run_names, all_comps, run_colors = [], [], []
    for (run_dir, label), color in zip(TRAINING_RUN_DIRS, COLORS):
        comps = _fitness_components(run_dir)
        if comps is None:
            continue
        run_names.append(label)
        all_comps.append(comps)
        run_colors.append(color)

    if all_comps:
        max_cov = max(max(c["coverage"]) for c in all_comps) or 1.0
        for c in all_comps:
            c["coverage"] = [v / max_cov for v in c["coverage"]]

        n_runs  = len(run_names)
        n_comp  = len(comp_keys)
        width   = 0.8 / n_runs
        offsets = np.linspace(-(n_runs - 1) / 2, (n_runs - 1) / 2, n_runs) * width
        xs      = np.arange(n_comp)

        for run_name, comps, color, offset in zip(run_names, all_comps, run_colors, offsets):
            for i, key in enumerate(comp_keys):
                bp = ax_comp.boxplot(
                    comps[key],
                    positions=[xs[i] + offset],
                    widths=width * 0.9,
                    patch_artist=True,
                    manage_ticks=False,
                    boxprops=dict(facecolor=color, alpha=0.7),
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    flierprops=dict(marker="o", color=color, ms=3),
                )
            # One legend handle per run (use the last box patch)
            bp["boxes"][0].set_label(run_name)

        ax_comp.set_xticks(xs)
        ax_comp.set_xticklabels(comp_labels, fontsize=9)
        ax_comp.set_ylim(0, 1.15)
        ax_comp.set_ylabel("Score (0–1)")
        ax_comp.legend(fontsize=8, loc="lower right")
        ax_comp.grid(axis="y", alpha=0.3)
        ax_comp.set_title("Fitness components (HOF policies)", fontsize=9)
    else:
        ax_comp.text(0.5, 0.5, "No episodes CSVs found\n(run SCRIPT_AssessPolicies.py first)",
                     ha="center", va="center", transform=ax_comp.transAxes, fontsize=9)

    out_path = os.path.join(PLOTS_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Look angle vs net body rotation
# ══════════════════════════════════════════════════════════════════════════════
#
# Episodes CSV expected at PolicyAssessment/{run_name}_episodes.csv
# (produced by SCRIPT_AssessPolicies.py).
# Uses the same TRAINING_RUN_DIRS list as section 1.

def plot_rotation_strategy() -> None:
    import csv

    # Only include runs for which an episodes CSV exists and which are not force_aligned (baseline)
    runs = []
    for run_dir, label in TRAINING_RUN_DIRS:
        if not os.path.exists(
            os.path.join("PolicyAssessment", f"{os.path.basename(run_dir)}_episodes.csv")
        ):
            continue
        cfg_path = os.path.join(run_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                if json.load(f).get("force_aligned", False):
                    continue
        runs.append((run_dir, label))
    if not runs:
        print("  ⚠ No episodes CSVs found — run SCRIPT_AssessPolicies.py first.")
        return

    n = len(runs)
    fig, axes2d = plt.subplots(2, n, figsize=(3.5 * n, 7),
                               gridspec_kw={"height_ratios": [1, 1.5]})
    if n == 1:
        axes2d = axes2d.reshape(2, 1)
    hist_axes   = axes2d[0]
    hexbin_axes = axes2d[1]

    all_rot1 = {}
    for ax_h, ax_hb, (run_dir, label) in zip(hist_axes, hexbin_axes, runs):
        csv_path = os.path.join("PolicyAssessment",
                                f"{os.path.basename(run_dir)}_episodes.csv")
        rot1, rot2 = [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rot1.append(float(row["rotate1_canonical"]))
                rot2.append(float(row["rotate2_canonical"]))

        rot1     = np.array(rot1)
        net_turn = rot1 + np.array(rot2)
        all_rot1[label] = rot1

        # Top row: histogram of rotate1
        ax_h.hist(rot1, bins=40, range=ROT1_RANGE, color="#4C72B0", edgecolor="none")
        ax_h.axvline(0, color="black", lw=0.8, linestyle=":", alpha=0.5)
        ax_h.set_xlim(ROT1_RANGE)
        ax_h.set_title(label, fontsize=11, fontweight="bold")
        ax_h.set_xlabel("Look angle (°)  [<0: away from wall]")
        ax_h.set_ylabel("Count") if ax_h is hist_axes[0] else None

        # Bottom row: hexbin
        hb = ax_hb.hexbin(rot1, net_turn, gridsize=40, cmap="Blues",
                          extent=[*ROT1_RANGE, *NET_ROT_RANGE], mincnt=5)
        ax_hb.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
        ax_hb.axvline(0, color="black", lw=0.8, linestyle=":",  alpha=0.5)
        ax_hb.set_xlabel("Look angle, rotate1 (°)")
        fig.colorbar(hb, ax=ax_hb, label="step count")

    hexbin_axes[0].set_ylabel("Net body rotation, rotate1+rotate2 (°)")
    fig.suptitle(
        "Look angle vs net body rotation  [canonical frame: wall on right]",
        fontsize=10,
    )
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "rotation_strategy.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Robot paths in arena
# ══════════════════════════════════════════════════════════════════════════════
#
# Rows = sessions, columns = history conditions.
# Each cell: one episode from a fixed central start in that arena.

PATH_SESSIONS = [
    ("sessionB01", (150.0,  -300.0, 0.0)),
    ("sessionB02", (  0.0,  -300.0, 0.0)),
    ("sessionB03", (-100.0, -900.0, 0.0)),
    ("sessionB04", (750.0,  -600.0, 0.0)),
    ("sessionB05", (-850.0, -650.0, 0.0)),
]

PATH_START = PATH_SESSIONS[0][1]   # kept for _run_single_episode default


def _run_single_episode(
    policy, cfg, simulator, start=None
) -> tuple[list[tuple[float, float]], list[float]]:
    """
    Run one episode from start (defaults to PATH_START).
    Returns (path, look_yaws) where path is a list of (x, y) positions and
    look_yaws is the sonar look direction (degrees) at each step.
    """
    import collections as _col
    from SCRIPT_TrainPolicy import build_input

    x, y, yaw         = start if start is not None else PATH_START
    history            = _col.deque([(0.0, 0.0, 0.0, 0.0)] * cfg.history_len,
                                    maxlen=cfg.history_len)
    last_physical_iid  = 0.0
    path               = [(x, y)]
    look_yaws          = []

    for _ in range(cfg.max_steps):
        original_yaw = yaw

        if cfg.force_aligned:
            rotate1_canonical = 0.0
            rotate1           = 0.0
        else:
            inp1              = build_input(history, 0.0, 0.0, 0.0, cfg)
            rotate1_canonical = policy.forward(inp1, cfg.max_rotate1_deg)
            flip1             = last_physical_iid < 0.0
            rotate1           = -rotate1_canonical if flip1 else rotate1_canonical
        look_yaw           = yaw + rotate1
        look_yaws.append(look_yaw)

        meas              = simulator.get_sonar_measurement(x, y, look_yaw)
        dist_mm           = min(float(meas.get("distance_mm", cfg.max_dist_mm)), cfg.max_dist_mm)
        physical_iid      = float(meas.get("iid_db", 0.0))
        flip2             = physical_iid < 0.0
        canonical_iid     = abs(physical_iid)

        inp2              = build_input(history, dist_mm, canonical_iid, rotate1_canonical, cfg)
        rotate2_canonical = policy.forward(inp2, cfg.max_rotate2_deg)
        rotate2           = -rotate2_canonical if flip2 else rotate2_canonical

        action = {"rotate1_deg": rotate1, "rotate2_deg": rotate2,
                  "drive_mm": cfg.fixed_drive_mm}
        result = simulator.simulate_robot_movement(
            x, y, original_yaw, [action], compute_sonar=False
        )[0]

        x   = float(result["position"]["x"])
        y   = float(result["position"]["y"])
        yaw = float(result["orientation"])
        path.append((x, y))

        history.append((dist_mm, canonical_iid, rotate1_canonical, rotate2_canonical))
        last_physical_iid = physical_iid

        if result["collision"]["drive_blocked"]:
            break

    return path, look_yaws


def plot_paths() -> None:
    from SCRIPT_AssessPolicies import load_cfg, load_hof, build_simulator

    runs = [
        (run_dir, label, color)
        for (run_dir, label), color in zip(TRAINING_RUN_DIRS, COLORS)
        if os.path.exists(run_dir)
    ]
    if not runs:
        print("  ⚠ No run directories found.")
        return

    n_runs     = len(runs)
    n_sessions = len(PATH_SESSIONS)
    fig, axes  = plt.subplots(n_sessions, n_runs,
                               figsize=(3.2 * n_runs, 3.2 * n_sessions))
    if n_sessions == 1:
        axes = axes[np.newaxis, :]
    if n_runs == 1:
        axes = axes[:, np.newaxis]

    # Load policies once
    policies = []
    for run_dir, label, color in runs:
        cfg    = load_cfg(run_dir)
        hof    = load_hof(run_dir, cfg, n=1)
        policies.append((cfg, hof[0]["policy"], label, color))

    for row, (session, start) in enumerate(PATH_SESSIONS):
        print(f"  {session}...")
        simulator = build_simulator(session)
        arena     = simulator.arena

        for col, (cfg, policy, label, color) in enumerate(policies):
            ax = axes[row, col]

            path, look_yaws = _run_single_episode(policy, cfg, simulator, start)
            xs, ys = zip(*path)

            # Arena walls
            if len(arena.walls) > 0:
                ax.scatter(arena.walls[:, 0], arena.walls[:, 1],
                           s=0.3, color="0.6", linewidths=0, zorder=1)

            # Path
            ax.plot(xs, ys, color=color, lw=1.2, alpha=0.9, zorder=2)

            # Look direction lines (every 5 steps); omit for baseline (always aligned with path)
            if not cfg.force_aligned:
                line_len = 200
                for i in range(0, len(look_yaws), 5):
                    angle_rad = np.deg2rad(look_yaws[i])
                    ax.plot([xs[i], xs[i] + line_len * np.cos(angle_rad)],
                            [ys[i], ys[i] + line_len * np.sin(angle_rad)],
                            color="black", lw=0.8, alpha=0.7, zorder=3)
                    ax.plot(xs[i], ys[i], "o", color="black", ms=2, zorder=4)

            # Start / end markers
            ax.plot(xs[0], ys[0], "o", color=color, ms=4, zorder=4)
            crashed    = len(path) < cfg.max_steps + 1
            ax.plot(xs[-1], ys[-1], "s" if crashed else "o", color=color,
                    ms=4, markerfacecolor=color if crashed else "white", zorder=4)

            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(label, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(session, fontsize=8)

    fig.suptitle(
        "Robot paths  [filled square = collision, open circle = max steps]",
        fontsize=9,
    )
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "paths.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: History usage
# ══════════════════════════════════════════════════════════════════════════════
#
# Reads PolicyAssessment/{run_name}_history_sensitivity.csv and plots
# sensitivity fraction vs input age for each channel, one line per run.

CHANNELS = ["dist", "iid", "r1", "r2"]
CHANNEL_LABELS = {"dist": "Distance", "iid": "IID", "r1": "Rotate1", "r2": "Rotate2"}


def plot_history_usage() -> None:
    import csv

    # Load sensitivity data for each run
    runs_data = []
    for (run_dir, label), color in zip(TRAINING_RUN_DIRS, COLORS):
        csv_path = os.path.join(
            "PolicyAssessment",
            f"{os.path.basename(run_dir)}_history_sensitivity.csv",
        )
        if not os.path.exists(csv_path):
            continue

        # channel -> list of (age, fraction, std) sorted oldest-first
        channels: dict = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                ch   = row["channel"]
                slot = row["slot"]
                age  = 0 if slot == "t" else int(slot.split("-")[1])
                channels.setdefault(ch, []).append(
                    (age, float(row["fraction"]), float(row["std"]))
                )
        for ch in channels:
            channels[ch].sort(key=lambda x: x[0], reverse=True)  # oldest first

        runs_data.append((label, color, channels))

    if not runs_data:
        print("  ⚠ No history sensitivity CSVs found — run SCRIPT_AssessPolicies.py first.")
        return

    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(3.5 * len(CHANNELS), 3.5), sharey=True)

    for ax, ch in zip(axes, CHANNELS):
        for label, color, channels in runs_data:
            if ch not in channels:
                continue
            entries  = channels[ch]
            ages     = [e[0] for e in entries]
            fracs    = [e[1] for e in entries]
            stds     = [e[2] for e in entries]
            max_f    = max(fracs) or 1.0
            fracs    = [f / max_f for f in fracs]
            stds     = [s / max_f for s in stds]
            ax.plot(ages, fracs, color=color, lw=1.8, marker="o", ms=4, label=label, alpha=0.6)
            ax.fill_between(ages,
                            [f - s for f, s in zip(fracs, stds)],
                            [f + s for f, s in zip(fracs, stds)],
                            color=color, alpha=0.15)

        ax.set_title(CHANNEL_LABELS[ch], fontsize=11, fontweight="bold")
        ax.set_xlabel("Input age (steps back)")
        ax.set_ylim(0, 1.1)
        ax.invert_xaxis()   # age=0 (current) on the right
        ax.axvline(0, color="black", lw=0.8, linestyle=":", alpha=0.5)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Sensitivity (normalised to peak)")
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        "History usage: input sensitivity by age  (rotate2 call)",
        fontsize=10,
    )
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "history_usage.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=== Section 1: Training curves ===")
    plot_training_curves()
    print("=== Section 2: Rotation strategy ===")
    plot_rotation_strategy()
    print("=== Section 3: Robot paths ===")
    plot_paths()
    print("=== Section 4: History usage ===")
    plot_history_usage()


if __name__ == "__main__":
    main()
