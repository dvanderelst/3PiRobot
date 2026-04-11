#!/usr/bin/env python3
"""
SCRIPT_AssessPolicies.py

Analyses whether a trained policy actually uses its history inputs by computing
input sensitivity: how much does the output change when each input slot changes?

Method: for each step of a simulated episode, compute |dy/dx_i| analytically
(backprop through the 3-layer tanh MLP) at the actual input vector the policy
received.  Average over all steps and policies.

A slot with high sensitivity is one the policy actively responds to.
A slot with near-zero sensitivity is one the policy ignores, regardless of
what weights connect to it.

Focuses on the rotate2 call (inp2), which receives real sonar measurements and
decides the final head rotation — this is where history is most relevant.

Input vector layout (see SCRIPT_TrainPolicy.py):
  [dist_{t-h}...dist_{t-1}, dist_t]   h+1 values  (t = current step)
  [iid_{t-h}...iid_{t-1},  iid_t]    h+1 values
  [r1_{t-h}...r1_{t-1},    r1_t]     h+1 values
  [r2_{t-h}...r2_{t-1}]              h   values    (no current r2 yet)

Outputs (to PolicyAssessment/<run_name>/):
  history_sensitivity.png    bar chart of mean ± std sensitivity per input slot
"""

import collections
import dataclasses
import glob
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from typing import List, Tuple

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Library.EnvironmentSimulator import EnvironmentSimulator
from SCRIPT_TrainPolicy import Config, MLPPolicy, build_input, load_starts


# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════

RUN_DIRS = [
    "PolicyTraining/policy_h00",
    "PolicyTraining/policy_h01",
    "PolicyTraining/policy_h03",
    "PolicyTraining/policy_h05",
]
N_POLICIES           = 5   # None = all HOF policies
EPISODES_PER_SESSION = 5   # episodes per policy per session
SEED                 = 42


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_cfg(run_dir: str) -> Config:
    with open(os.path.join(run_dir, "config.json")) as f:
        d = json.load(f)
    valid = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in valid})


def load_hof(run_dir: str, cfg: Config, n: int | None) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(run_dir, "top_policies", "rank*.json")))
    if not paths:
        raise FileNotFoundError(f"No rank*.json in {run_dir}/top_policies/")
    if n is not None:
        paths = paths[:n]
    entries = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        pol = MLPPolicy(cfg)
        pol.set_genome(np.array(data["genome"], dtype=np.float32))
        entries.append({"rank": data["rank"], "fitness": data["fitness"],
                        "generation": data.get("generation", -1), "policy": pol})
    return entries


def build_simulator(session_name: str) -> EnvironmentSimulator:
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        return EnvironmentSimulator(session_name)


# ══════════════════════════════════════════════════════════════════════════════
# Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def compute_sensitivity(policy: MLPPolicy, x: np.ndarray, max_rotate_deg: float) -> np.ndarray:
    """
    Analytical gradient |dy/dx_i| via backprop through the 3-layer tanh MLP.

    Forward:  z1 = W1@x+b1,  h1 = tanh(z1)
              z2 = W2@h1+b2, h2 = tanh(z2)
              y  = tanh(W3@h2+b3) * max_rotate_deg

    Backward: propagate scalar gradient back to x, take absolute value.
    """
    x = x.reshape(-1, 1)
    W1, b1, W2, b2, W3, b3 = policy.params

    z1 = W1 @ x + b1.reshape(-1, 1)
    h1 = np.tanh(z1)
    z2 = W2 @ h1 + b2.reshape(-1, 1)
    h2 = np.tanh(z2)
    z3 = W3 @ h2 + b3.reshape(-1, 1)

    g = max_rotate_deg * (1.0 - np.tanh(z3) ** 2) * W3   # (1, h2)
    g = g * (1.0 - h2 ** 2).T                             # (1, h2) element-wise
    g = g @ W2                                            # (1, h1)
    g = g * (1.0 - h1 ** 2).T                             # (1, h1) element-wise
    g = g @ W1                                            # (1, in_dim)

    return np.abs(g.ravel())


# ══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episodes(
    policy: MLPPolicy,
    simulator: EnvironmentSimulator,
    starts: List[Tuple[float, float, float]],
    cfg: Config,
    rng: np.random.Generator,
    n_episodes: int,
) -> tuple[np.ndarray, list[dict]]:
    """
    Run n_episodes episodes.  Returns:
      sens_r2   — np.ndarray of shape (n_steps, in_dim): |dy/dx| at each rotate2 call
      records   — list of per-step dicts with trajectory and action fields
    """
    all_sens, records = [], []

    for ep_idx in range(n_episodes):
        x, y, yaw = starts[int(rng.integers(len(starts)))]
        history = collections.deque(
            [(0.0, 0.0, 0.0, 0.0)] * cfg.history_len, maxlen=cfg.history_len
        )
        last_physical_iid = 0.0

        for step_idx in range(cfg.max_steps):
            original_yaw = yaw

            inp1 = build_input(history, 0.0, 0.0, 0.0, cfg)
            rotate1_canonical = policy.forward(inp1, cfg.max_rotate1_deg)
            flip1 = last_physical_iid < 0.0
            rotate1 = -rotate1_canonical if flip1 else rotate1_canonical
            look_yaw = yaw + rotate1

            meas          = simulator.get_sonar_measurement(x, y, look_yaw)
            dist_mm       = min(float(meas.get("distance_mm", cfg.max_dist_mm)), cfg.max_dist_mm)
            physical_iid  = float(meas.get("iid_db", 0.0))
            flip2         = physical_iid < 0.0
            canonical_iid = abs(physical_iid)

            inp2 = build_input(history, dist_mm, canonical_iid, rotate1_canonical, cfg)
            all_sens.append(compute_sensitivity(policy, inp2, cfg.max_rotate2_deg))
            rotate2_canonical = policy.forward(inp2, cfg.max_rotate2_deg)
            rotate2 = -rotate2_canonical if flip2 else rotate2_canonical

            action = {"rotate1_deg": rotate1, "rotate2_deg": rotate2,
                      "drive_mm": cfg.fixed_drive_mm}
            result = simulator.simulate_robot_movement(
                x, y, original_yaw, [action], compute_sonar=False
            )[0]

            new_x   = float(result["position"]["x"])
            new_y   = float(result["position"]["y"])
            new_yaw = float(result["orientation"])
            blocked = bool(result["collision"]["drive_blocked"])

            records.append({
                "episode":           ep_idx,
                "step":              step_idx,
                "x":                 x,
                "y":                 y,
                "yaw_deg":           original_yaw,
                "rotate1_canonical": rotate1_canonical,
                "rotate2_canonical": rotate2_canonical,
                "head_angle_deg":    -rotate2_canonical,
                "dist_mm":           dist_mm,
                "iid_canonical_db":  canonical_iid,
                "drive_blocked":     int(blocked),
            })

            history.append((dist_mm, canonical_iid, rotate1_canonical, rotate2_canonical))
            last_physical_iid = physical_iid
            x, y, yaw = new_x, new_y, new_yaw

            if blocked:
                break

    empty   = np.zeros((0, policy.in_dim), dtype=np.float32)
    sens_r2 = np.array(all_sens, dtype=np.float32) if all_sens else empty
    return sens_r2, records


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════

def input_channels(history_len: int, include_r1: bool = True):
    """Return list of (channel_name, slot_labels, start_index, size) in input-vector order."""
    h = history_len
    hist_labels = [f"t-{h - i}" for i in range(h)]
    ch_names  = ["dist", "iid", "r1", "r2"] if include_r1 else ["dist", "iid", "r2"]
    ch_sizes  = ([h + 1, h + 1, h + 1, h] if include_r1
                 else [h + 1, h + 1, h])
    ch_labels = ([hist_labels + ["t"], hist_labels + ["t"],
                  hist_labels + ["t"], hist_labels] if include_r1
                 else [hist_labels + ["t"], hist_labels + ["t"],
                       hist_labels])
    starts = [sum(ch_sizes[:i]) for i in range(len(ch_names))]
    return list(zip(ch_names, ch_labels, starts, ch_sizes))


def _plot_row(axes, mean_sens, std_sens, channels, row_label):
    """Fill one row of subplots with normalised sensitivity bars."""
    for ax, (ch_name, slot_labels, start, size) in zip(axes, channels):
        idx    = np.arange(start, start + size)
        raw    = mean_sens[idx]
        raw_sd = std_sens[idx]
        total  = raw.sum() if raw.sum() > 0 else 1.0
        mn, sd = raw / total, raw_sd / total
        xs     = np.arange(size)

        ax.bar(xs, mn, color="#4C72B0", alpha=0.85, zorder=3)
        ax.errorbar(xs, mn, yerr=sd, fmt="none", color="black",
                    capsize=3, lw=1, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(slot_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_xlabel("timestep")

    axes[0].set_ylabel(f"{row_label}\nfraction of channel total")


def make_plot(mean_r2, std_r2, cfg, n_policies, prefix):
    channels = input_channels(cfg.history_len, cfg.include_r1_in_input)
    n_ch = len(channels)

    fig, axes = plt.subplots(1, n_ch, figsize=(3.5 * n_ch, 4), sharey=True)
    if n_ch == 1:
        axes = [axes]

    for col, (ch_name, _, _, _) in enumerate(channels):
        axes[col].set_title(ch_name, fontsize=11, fontweight="bold")

    _plot_row(axes, mean_r2, std_r2, channels, "rotate2")

    run_name = os.path.basename(prefix)
    fig.suptitle(
        f"{run_name}  —  history_len={cfg.history_len},  {n_policies} HOF policies\n"
        "Input sensitivity (rotate2): fraction of each channel's influence per time slot",
        fontsize=10,
    )
    plt.tight_layout()
    out_path = f"{prefix}_history_sensitivity.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_csv(mean_r2, std_r2, cfg, run_name, prefix):
    """Save normalised sensitivities to CSV for later analysis."""
    import csv
    channels = input_channels(cfg.history_len, cfg.include_r1_in_input)
    rows = []
    for ch_name, slot_labels, start, size in channels:
        raw    = mean_r2[start:start+size]
        raw_sd = std_r2[start:start+size]
        total  = raw.sum() if raw.sum() > 0 else 1.0
        for label, mn, sd in zip(slot_labels, raw / total, raw_sd / total):
            rows.append({
                "run":         run_name,
                "history_len": cfg.history_len,
                "channel":     ch_name,
                "slot":        label,
                "fraction":    round(float(mn), 6),
                "std":         round(float(sd), 6),
            })
    out_path = f"{prefix}_history_sensitivity.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

EPISODE_CSV_FIELDS = [
    "run", "history_len", "rank", "fitness", "generation",
    "session", "episode", "step",
    "x", "y", "yaw_deg",
    "rotate1_canonical", "rotate2_canonical", "head_angle_deg",
    "dist_mm", "iid_canonical_db", "drive_blocked",
]


def run_one(run_dir: str, rng: np.random.Generator) -> None:
    import csv as _csv

    run_name = os.path.basename(os.path.normpath(run_dir))
    prefix   = os.path.join("PolicyAssessment", run_name)
    os.makedirs("PolicyAssessment", exist_ok=True)

    cfg = load_cfg(run_dir)
    hof = load_hof(run_dir, cfg, N_POLICIES)
    print(f"\nLoaded {len(hof)} HOF policies from {run_dir}  (history_len={cfg.history_len})")

    all_sessions = list(cfg.train_session_names)
    if cfg.validation_session_name:
        all_sessions.append(cfg.validation_session_name)

    print("Loading simulators...")
    simulators = {sn: build_simulator(sn) for sn in all_sessions}
    starts_by  = {sn: load_starts(sn, cfg, quiet=True) for sn in all_sessions}

    ep_csv_path = f"{prefix}_episodes.csv"
    means_r2    = []
    total_steps = 0

    with open(ep_csv_path, "w", newline="") as ep_f:
        writer = _csv.DictWriter(ep_f, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()

        for entry in tqdm(hof, desc="Policies"):
            rank, fitness, generation = entry["rank"], entry["fitness"], entry["generation"]
            policy = entry["policy"]

            for sn in all_sessions:
                sens_r2, records = run_episodes(
                    policy, simulators[sn], starts_by[sn],
                    cfg, rng, EPISODES_PER_SESSION,
                )
                if sens_r2.shape[0] > 0:
                    means_r2.append(sens_r2.mean(axis=0))

                for rec in records:
                    writer.writerow({
                        "run": run_name, "history_len": cfg.history_len,
                        "rank": rank, "fitness": fitness, "generation": generation,
                        "session": sn,
                        **rec,
                    })
                total_steps += len(records)

    print(f"Saved {ep_csv_path}  ({total_steps:,} steps)")

    means_r2 = np.stack(means_r2)
    mean_r2, std_r2 = means_r2.mean(axis=0), means_r2.std(axis=0)

    plot_path = make_plot(mean_r2, std_r2, cfg, len(hof), prefix)
    sens_path = save_csv(mean_r2, std_r2, cfg, run_name, prefix)
    print(f"Saved {plot_path}")
    print(f"Saved {sens_path}")

    channels = input_channels(cfg.history_len, cfg.include_r1_in_input)
    print(f"\n{'Channel':<6}  {'Slot':<6}  {'Fraction':>10}  {'Std':>8}")
    print("-" * 38)
    for ch_name, slot_labels, start, size in channels:
        raw    = mean_r2[start:start+size]
        raw_sd = std_r2[start:start+size]
        total  = raw.sum() if raw.sum() > 0 else 1.0
        for label, mn, sd in zip(slot_labels, raw / total, raw_sd / total):
            marker = " ◀ current" if label == "t" else ""
            print(f"{ch_name:<6}  {label:<6}  {mn:>10.3f}  {sd:>8.3f}{marker}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    for run_dir in RUN_DIRS:
        run_one(run_dir, rng)


if __name__ == "__main__":
    main()
