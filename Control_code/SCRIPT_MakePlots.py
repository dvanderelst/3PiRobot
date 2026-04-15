#!/usr/bin/env python3
"""
SCRIPT_MakePlots.py

Publication-ready plots assembled from data produced by other scripts.
All outputs go to PaperPlots/.

Data flow
---------
Section 1 — Training curves
    Reads:  {run_dir}/training_history.json          (SCRIPT_TrainPolicy.py or
                                                       SCRIPT_TrainPolicy_discriminability.py)
            Keys used: "best", "val", "collision_rate", and optionally "discriminability"
            (the last key is only present in discriminability-training runs)
            PolicyAssessment/{run_name}_episodes.csv  (SCRIPT_AssessPolicies.py)

Section 2 — Rotation strategy
    Reads:  PolicyAssessment/{run_name}_episodes.csv  (SCRIPT_AssessPolicies.py)

Section 3 — Robot paths
    Runs:   episodes live via EnvironmentSimulator    (no pre-computed data)
    Reads:  ValidStarts/{session}_valid_starts.json   (SCRIPT_ComputeValidStarts.py)
            {run_dir}/top_policies/rank*.json         (SCRIPT_TrainPolicy.py)

Section 4 — History usage
    Reads:  PolicyAssessment/{run_name}_history_sensitivity.csv  (SCRIPT_AssessPolicies.py)

Section 5 — Baseline policy map  (force_aligned runs only)
    Reads:  {run_dir}/top_policies/rank*.json  (SCRIPT_TrainPolicy.py)
    Note:   Only applicable to force_aligned policies (history_len=0, rotate1=0).
            Their input is exactly (dist, iid), so the full policy can be
            characterised by sweeping this 2-D space analytically — no simulation needed.
"""

import copy
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
# Settings
# ══════════════════════════════════════════════════════════════════════════════

# ── Runs (shared across all sections) ─────────────────────────────────────────
TRAINING_RUN_DIRS = [
    ("PolicyTraining/sonar_h00", "Baseline"),
    ("PolicyTraining/sonar_h01", "Hist 1"),
    ("PolicyTraining/sonar_h05", "Hist 5"),
    ("PolicyTraining/sonar_h10", "Hist 10"),
]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── Section 1: Fitness plots ──────────────────────────────────
FITNESS_RANGE = [0.9, 1.01]

# ── Section 2: Rotation strategy plot ranges ──────────────────────────────────
ROT1_RANGE    = (-45, 45)   # x-axis: look angle (rotate1), degrees
NET_ROT_RANGE = (-30, 30)   # y-axis: net body rotation (rotate1 + rotate2), degrees

# ── Section 3: Robot paths ────────────────────────────────────────────────────
PATH_SESSIONS   = ["sessionB01", "sessionB02", "sessionB03", "sessionB04", "sessionB05"]
PATH_N_EPISODES = 1     # episodes to run per (session, policy) combination
PATH_MAX_STEPS  = 100  # max steps per episode; None = use value from config.json
PATH_RANDOM_SEED = 42   # seed for start-position sampling (None = non-deterministic)

# ── Section 4: History usage ──────────────────────────────────────────────────
CHANNELS       = ["dist", "iid", "r1", "r2"]
CHANNEL_LABELS = {"dist": "Distance", "iid": "IID", "r1": "Rotate1", "r2": "Rotate2"}

# ── Section 5: Baseline policy map ───────────────────────────────────────────
POLICY_MAP_N_POLICIES = 5             # HOF policies to average over (None = all)
POLICY_MAP_DIST_RANGE = (0,   2000)   # mm  — x-axis
POLICY_MAP_IID_RANGE  = (0,   10)     # dB  — y-axis
POLICY_MAP_RESOLUTION = 200           # grid points per axis


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

    # coverage and jitter_factor: one value per policy (for boxplots)
    # collision_free_rate: single scalar over all episodes (for bar)
    coverages, jitter_factors, all_collided = [], [], []
    for rank_data in by_rank.values():
        coverages.append(float(np.mean(rank_data["coverages"])))
        jitter_factors.append(float(np.mean(rank_data["jitter_factors"])))
        all_collided.extend(rank_data["collided"])

    return {
        "coverage":            coverages,
        "jitter_factor":       jitter_factors,
        "collision_free_rate": 1.0 - float(np.mean(all_collided)),
    }


def plot_training_curves() -> None:
    fig = plt.figure(figsize=(12, 8))
    gs  = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.1, wspace=0.35)
    ax_fit  = fig.add_subplot(gs[0, 0])
    ax_col  = fig.add_subplot(gs[1, 0], sharex=ax_fit)
    ax_disc = fig.add_subplot(gs[2, 0], sharex=ax_fit)
    ax_comp = fig.add_subplot(gs[:, 1])

    has_disc = False
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

        disc_vals = hist.get("discriminability")
        if disc_vals:
            disc = np.array([v if v is not None else np.nan for v in disc_vals])
            ax_disc.plot(np.arange(len(disc)), disc, color=color, lw=1.5, label=label)
            has_disc = True

    from matplotlib.lines import Line2D
    handles, labels = ax_fit.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", lw=1.0, linestyle="--", alpha=0.6))
    labels.append("Validation")
    ax_fit.legend(handles=handles, labels=labels, fontsize=8, ncol=2)

    ax_fit.set_ylabel("Fitness (a.u.)")
    ax_fit.grid(True, alpha=0.3)
    plt.setp(ax_fit.get_xticklabels(), visible=False)

    ax_col.set_ylabel("Collision rate")
    ax_col.set_ylim(0, 1)
    ax_col.legend(fontsize=8)
    ax_col.grid(True, alpha=0.3)
    plt.setp(ax_col.get_xticklabels(), visible=False)

    ax_disc.set_xlabel("Generation")
    ax_disc.set_ylabel("Discriminability\n(Spearman r)")
    ax_disc.set_ylim(-1, 1)
    ax_disc.axhline(0, color="black", lw=0.8, linestyle=":", alpha=0.5)
    ax_disc.grid(True, alpha=0.3)
    if has_disc:
        ax_disc.legend(fontsize=8)
    else:
        ax_disc.text(0.5, 0.5, "No discriminability data\n(disc runs only)",
                     ha="center", va="center", transform=ax_disc.transAxes,
                     fontsize=9, color="0.5")

    # ── Fitness components panel ──────────────────────────────────────────────
    # coverage and jitter_factor: boxplots (distribution over HOF policies)
    # collision_free_rate: single bar (overall fraction across all episodes)
    boxplot_labels = ["Coverage\n(norm.)", "Smoothness\n(jitter factor)"]
    boxplot_keys   = ["coverage", "jitter_factor"]
    bar_label      = "Collision-free\nrate"
    bar_key        = "collision_free_rate"
    comp_labels    = boxplot_labels + [bar_label]

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
        n_comp  = len(comp_labels)
        width   = 0.8 / n_runs
        offsets = np.linspace(-(n_runs - 1) / 2, (n_runs - 1) / 2, n_runs) * width
        xs      = np.arange(n_comp)

        for run_name, comps, color, offset in zip(run_names, all_comps, run_colors, offsets):
            # Boxplots for coverage and jitter_factor
            for i, key in enumerate(boxplot_keys):
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
            # Bar for collision-free rate
            bar = ax_comp.bar(
                xs[-1] + offset, comps[bar_key],
                width=width * 0.9, color=color, alpha=0.7,
                label=run_name,
            )

        ax_comp.set_xticks(xs)
        ax_comp.set_xticklabels(comp_labels, fontsize=9)
        ax_comp.set_ylim(FITNESS_RANGE[0], FITNESS_RANGE[1])
        ax_comp.set_ylabel("Score (0–1)")
        ax_comp.legend(fontsize=8, loc="lower right")
        ax_comp.grid(axis="y", alpha=0.3)
        ax_comp.set_title("Fitness components (HOF policies)", fontsize=9)
    else:
        ax_comp.text(0.5, 0.5, "No episodes CSVs found\n(run SCRIPT_AssessPolicies.py first)",
                     ha="center", va="center", transform=ax_comp.transAxes, fontsize=9)

    out_path = os.path.join(PLOTS_DIR, "performance.png")
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
# Each cell: PATH_N_EPISODES episodes from a fixed central start in that arena.


def _run_single_episode(
    policy, cfg, simulator, start: tuple[float, float, float]
) -> tuple[list[tuple[float, float]], list[float]]:
    """
    Run one episode from start (defaults to PATH_START).
    Returns (path, look_yaws) where path is a list of (x, y) positions and
    look_yaws is the sonar look direction (degrees) at each step.
    """
    import collections as _col
    from SCRIPT_TrainPolicy import build_input

    x, y, yaw         = start
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


def _load_valid_starts(session: str) -> list[tuple[float, float, float]]:
    """Return list of (x, y, yaw_deg) from ValidStarts/<session>_valid_starts.json."""
    path = os.path.join("ValidStarts", f"{session}_valid_starts.json")
    with open(path) as f:
        data = json.load(f)
    return [(s["x"], s["y"], s["yaw_deg"]) for s in data["starts"]]


def plot_paths() -> None:
    from SCRIPT_AssessPolicies import load_cfg, load_hof, build_simulator

    rng = np.random.default_rng(PATH_RANDOM_SEED)

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

    for row, session in enumerate(PATH_SESSIONS):
        print(f"  {session}...")
        simulator     = build_simulator(session)
        arena         = simulator.arena
        valid_starts  = _load_valid_starts(session)
        chosen_starts = [
            valid_starts[i]
            for i in rng.choice(len(valid_starts), size=PATH_N_EPISODES, replace=False)
        ]

        for col, (cfg, policy, label, color) in enumerate(policies):
            ax = axes[row, col]

            effective_cfg = cfg
            if PATH_MAX_STEPS is not None:
                effective_cfg = copy.copy(cfg)
                effective_cfg.max_steps = PATH_MAX_STEPS

            all_paths = [
                _run_single_episode(policy, effective_cfg, simulator, start)
                for start in chosen_starts
            ]

            # Arena walls
            if len(arena.walls) > 0:
                ax.scatter(arena.walls[:, 0], arena.walls[:, 1],
                           s=0.3, color="0.6", linewidths=0, zorder=1)

            alpha_step = 0.9 / max(len(all_paths), 1)
            for ep_idx, (path, look_yaws) in enumerate(all_paths):
                xs, ys  = zip(*path)
                ep_alpha = 0.9 - ep_idx * alpha_step * 0.3  # fade slightly per episode

                # Path
                ax.plot(xs, ys, color=color, lw=1.2, alpha=ep_alpha, zorder=2)

                # Look direction lines (every 5 steps); omit for baseline (always aligned with path)
                if not effective_cfg.force_aligned:
                    line_len = 200
                    for i in range(0, len(look_yaws), 5):
                        angle_rad = np.deg2rad(look_yaws[i])
                        ax.plot([xs[i], xs[i] + line_len * np.cos(angle_rad)],
                                [ys[i], ys[i] + line_len * np.sin(angle_rad)],
                                color="black", lw=0.8, alpha=0.7 * ep_alpha, zorder=3)
                        ax.plot(xs[i], ys[i], "o", color="black", ms=2, zorder=4)

                # Start / end markers
                ax.plot(xs[0], ys[0], "o", color=color, ms=4, zorder=4)
                crashed = len(path) < effective_cfg.max_steps + 1
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
# Section 5: Baseline policy map
# ══════════════════════════════════════════════════════════════════════════════
#
# Only runs for force_aligned policies (input = (dist, iid) only).
# Sweeps the full 2-D input space analytically — no simulation needed.
# Mean across the top POLICY_MAP_N_POLICIES HOF policies, one panel per run.

def plot_policy_map() -> None:
    from SCRIPT_AssessPolicies import load_cfg, load_hof

    # Collect force_aligned runs
    runs = [
        (run_dir, label, color)
        for (run_dir, label), color in zip(TRAINING_RUN_DIRS, COLORS)
        if os.path.exists(run_dir)
        and json.load(open(os.path.join(run_dir, "config.json"))).get("force_aligned", False)
    ]
    if not runs:
        print("  ⚠ No force_aligned runs found — skipping policy map.")
        return

    dist_vals = np.linspace(*POLICY_MAP_DIST_RANGE, POLICY_MAP_RESOLUTION)
    iid_vals  = np.linspace(*POLICY_MAP_IID_RANGE,  POLICY_MAP_RESOLUTION)
    DD, II    = np.meshgrid(dist_vals, iid_vals)
    grid_pts  = np.stack([DD.ravel(), II.ravel()], axis=1)  # (res², 2)

    fig, axes = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 4),
                             squeeze=False)
    axes = axes[0]

    for ax, (run_dir, label, color) in zip(axes, runs):
        cfg      = load_cfg(run_dir)
        hof      = load_hof(run_dir, cfg, n=POLICY_MAP_N_POLICIES)
        norm_pts = (grid_pts / np.array([cfg.max_dist_mm, cfg.max_iid_db],
                                        dtype=np.float32)).astype(np.float32)

        maps = [
            np.array([
                entry["policy"].forward(pt, cfg.max_rotate2_deg) for pt in norm_pts
            ]).reshape(POLICY_MAP_RESOLUTION, POLICY_MAP_RESOLUTION)
            for entry in hof
        ]
        mean_map = np.mean(maps, axis=0)

        vmax = cfg.max_rotate2_deg
        im   = ax.imshow(mean_map, origin="lower", aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax,
                         extent=[*POLICY_MAP_DIST_RANGE, *POLICY_MAP_IID_RANGE])
        ax.contour(dist_vals, iid_vals, mean_map, levels=[0],
                   colors="black", linewidths=1.0, linestyles="--")
        fig.colorbar(im, ax=ax, label="rotate2 (°)", shrink=0.85)
        ax.set_xlabel("Distance (mm)")
        ax.set_ylabel("IID (dB)")
        ax.set_title(f"{label}  (mean of top {len(hof)} policies)", fontsize=10)

    fig.suptitle(
        "Baseline policy map: dist × IID → rotate2\n"
        "[red = toward wall, blue = away, dashed = zero boundary]",
        fontsize=10,
    )
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "policy_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
    print("=== Section 5: Baseline policy map ===")
    plot_policy_map()


if __name__ == "__main__":
    main()
