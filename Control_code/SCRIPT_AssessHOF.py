#!/usr/bin/env python3
"""
SCRIPT_AssessHOF.py

Loads HOF policies from a training run and runs assessment episodes across
all sessions.  Writes per-step data to a CSV and a histogram of the head
angle relative to body movement direction (-rotate2).

Head angle convention:
    head_angle_deg = -rotate2_canonical
    positive = measurement direction was to the RIGHT of body movement
    negative = measurement direction was to the LEFT  of body movement

Outputs (to PolicyAssessment/<run_name>/):
    episodes.csv        per-step data for all policies × sessions × episodes
    rotate2_hist.png    histogram of head_angle_deg
"""

import collections
import csv
import dataclasses
import glob
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, List, Tuple

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Library.EnvironmentSimulator import EnvironmentSimulator
from SCRIPT_TrainPolicy import Config, MLPPolicy, build_input, load_starts


# ══════════════════════════════════════════════════════════════════════════════
# Settings — edit these
# ══════════════════════════════════════════════════════════════════════════════

RUN_DIR              = "Policy/run5"
N_POLICIES           = 5    # how many top HOF policies to assess (None = all)
EPISODES_PER_SESSION = 3     # per policy per session
SEED                 = 42


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_cfg(run_dir: str) -> Config:
    with open(os.path.join(run_dir, "config.json")) as f:
        d = json.load(f)
    valid = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in valid})


def load_hof(run_dir: str, cfg: Config) -> List[dict]:
    """Load all rank*.json files from top_policies/. Returns list of dicts."""
    paths = sorted(glob.glob(os.path.join(run_dir, "top_policies", "rank*.json")))
    if not paths:
        raise FileNotFoundError(f"No rank*.json files found in {run_dir}/top_policies/")
    entries = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        pol = MLPPolicy(cfg)
        pol.set_genome(np.array(data["genome"], dtype=np.float32))
        entries.append({
            "rank":       int(data["rank"]),
            "fitness":    float(data["fitness"]),
            "generation": int(data["generation"]),
            "policy":     pol,
        })
    return entries


def build_simulator(session_name: str) -> EnvironmentSimulator:
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        return EnvironmentSimulator(session_name)


# ══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episode_record(
    policy: MLPPolicy,
    simulator: EnvironmentSimulator,
    starts: List[Tuple[float, float, float]],
    cfg: Config,
    rng: np.random.Generator,
) -> List[dict]:
    """
    Run one episode; return a list of per-step dicts.
    All rotation values are in the canonical (wall-on-right) frame.
    """
    if not starts:
        return []

    x, y, yaw = starts[int(rng.integers(len(starts)))]
    history = collections.deque(
        [(0.0, 0.0, 0.0, 0.0)] * cfg.history_len, maxlen=cfg.history_len
    )
    last_physical_iid = 0.0
    steps = []

    for step_idx in range(cfg.max_steps):
        original_yaw = yaw

        inp1 = build_input(history, 0.0, 0.0, 0.0, cfg)
        rotate1_canonical = policy.forward(inp1, cfg.max_rotate1_deg)
        flip1 = last_physical_iid < 0.0
        rotate1 = -rotate1_canonical if flip1 else rotate1_canonical
        look_yaw = original_yaw + rotate1

        meas         = simulator.get_sonar_measurement(x, y, look_yaw)
        dist_mm      = min(float(meas.get("distance_mm", cfg.max_dist_mm)), cfg.max_dist_mm)
        physical_iid = float(meas.get("iid_db", 0.0))

        flip2         = physical_iid < 0.0
        canonical_iid = abs(physical_iid)
        inp2 = build_input(history, dist_mm, canonical_iid, rotate1_canonical, cfg)
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

        steps.append({
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

    return steps


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "rank", "fitness", "generation",
    "session", "episode",
    "step", "x", "y", "yaw_deg",
    "rotate1_canonical", "rotate2_canonical", "head_angle_deg",
    "dist_mm", "iid_canonical_db", "drive_blocked",
]


def main() -> None:
    run_name   = os.path.basename(os.path.normpath(RUN_DIR))
    output_dir = os.path.join("PolicyAssessment", run_name)
    os.makedirs(output_dir, exist_ok=True)

    cfg = load_cfg(RUN_DIR)
    rng = np.random.default_rng(SEED)

    hof = load_hof(RUN_DIR, cfg)
    if N_POLICIES is not None:
        hof = hof[:N_POLICIES]
    print(f"Loaded {len(hof)} HOF policies from {RUN_DIR}/top_policies/")

    all_sessions = list(cfg.train_session_names)
    if cfg.validation_session_name:
        all_sessions.append(cfg.validation_session_name)

    print(f"Sessions: {all_sessions}")
    print("Loading simulators...", end=" ", flush=True)
    simulators        = {sn: build_simulator(sn)              for sn in all_sessions}
    starts_by_session = {sn: load_starts(sn, cfg, quiet=True) for sn in all_sessions}
    print("done.")

    n_episodes_total = len(hof) * len(all_sessions) * EPISODES_PER_SESSION
    print(f"Running {len(hof)} policies × {len(all_sessions)} sessions × "
          f"{EPISODES_PER_SESSION} episodes = {n_episodes_total} episodes\n")

    csv_path = os.path.join(output_dir, "episodes.csv")
    all_head_angles = []
    total_steps = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        with tqdm(total=n_episodes_total, unit="ep") as pbar:
            for entry in hof:
                rank       = entry["rank"]
                fitness    = entry["fitness"]
                generation = entry["generation"]
                policy     = entry["policy"]
                tqdm.write(f"  rank {rank:3d}  fitness={fitness:.1f}  gen={generation}")

                for sn in all_sessions:
                    for ep_idx in range(EPISODES_PER_SESSION):
                        steps = run_episode_record(
                            policy, simulators[sn], starts_by_session[sn], cfg, rng
                        )
                        for s in steps:
                            all_head_angles.append(s["head_angle_deg"])
                            writer.writerow({
                                "rank":       rank,
                                "fitness":    fitness,
                                "generation": generation,
                                "session":    sn,
                                "episode":    ep_idx,
                                **s,
                            })
                        total_steps += len(steps)
                        pbar.update(1)

    print(f"\nWrote {total_steps:,} steps to {csv_path}")

    # ── Histogram ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_head_angles, bins=60, edgecolor="none", alpha=0.85)
    ax.axvline(0, color="k", linestyle="--", lw=1, alpha=0.5)
    ax.set_xlabel(
        "Head angle relative to body movement direction (°)\n"
        "[positive = right of body,  negative = left of body]"
    )
    ax.set_ylabel("Step count")
    ax.set_title(
        f"{run_name}  —  {len(hof)} HOF policies, "
        f"{len(all_sessions)} sessions × {EPISODES_PER_SESSION} ep/session"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "rotate2_hist.png")
    plt.savefig(hist_path, dpi=120)
    plt.close(fig)
    print(f"Saved {hist_path}")


if __name__ == "__main__":
    main()
