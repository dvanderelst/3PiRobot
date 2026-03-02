#!/usr/bin/env python3
"""
Open-loop policy replay on real session data.

For each session: load real sonar measurements (IID, distance) at real recorded
positions, run the trained HistoryNNPolicy in open-loop, and visualise steering.

Usage (from Control_code/):
    python SCRIPT_ReplayPolicy.py
"""

# ============================================================================
# CONFIGURATION - Modify these settings to control replay behavior
# ============================================================================

# Generation to use (None for best policy, or specific number like 3 for gen_003)
SELECTED_GENERATION = 25  # Set to None to use best policy

# Sessions to replay (list of session names)
SELECTED_SESSIONS = ["sessionB01"]  # Modify this list

# Output directory (relative to Control_code/)
OUTPUT_DIR = "Replay"

# Visualization settings
ARROW_STRIDE = 3       # plot every Nth arrow
ARROW_LEN_MM = 150.0    # length of direction arrows in mm
IID_DEADBAND_DB = 0.25  # IID deadband in dB

# ============================================================================
# IMPORTS AND CODE - No need to modify below this line
# ============================================================================

import collections
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import Library.DataProcessor as DataProcessor
from SCRIPT_TrainPolicy import HistoryNNPolicy, safe_float


@dataclass
class ReplayConfig:
    policy_path: str = "Policy/best_policy.json"
    session_names: List[str] = field(default_factory=lambda: SELECTED_SESSIONS)
    output_dir: str = OUTPUT_DIR
    arrow_stride: int = ARROW_STRIDE
    arrow_len_mm: float = ARROW_LEN_MM
    iid_deadband_db: float = IID_DEADBAND_DB
    generation: Optional[int] = SELECTED_GENERATION


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy(policy_path: str) -> Tuple[HistoryNNPolicy, Dict[str, Any]]:
    """Load a HistoryNNPolicy from a JSON file."""
    with open(policy_path) as f:
        d = json.load(f)
    policy = HistoryNNPolicy(
        max_rotate1_deg=safe_float(d.get("max_rotate1_deg", 90.0), 90.0),
        max_rotate2_deg=safe_float(d.get("max_rotate2_deg", 90.0), 90.0),
        deadband_db=safe_float(d.get("iid_deadband_db", 0.25), 0.25),
        history_len=int(d["history_len"]),
        hidden_sizes=tuple(int(v) for v in d["hidden_sizes"]),
    )
    policy.set_genome(np.array(d["genome"], dtype=np.float32))
    return policy, d


def resolve_policy_path(cfg: ReplayConfig) -> str:
    """Return an existing policy JSON path, falling back to latest generation checkpoint."""
    # If specific generation is requested, use that
    if cfg.generation is not None:
        gen_str = f"gen_{cfg.generation:03d}"
        gen_policy_path = os.path.join("Policy", "generation_best", f"{gen_str}_best_policy.json")
        if os.path.exists(gen_policy_path):
            print(f"Using generation {cfg.generation} policy: {gen_policy_path}")
            return gen_policy_path
        else:
            print(f"ERROR: generation {cfg.generation} policy not found at '{gen_policy_path}'")
            sys.exit(1)
    
    # Otherwise use the configured policy path
    if os.path.exists(cfg.policy_path):
        return cfg.policy_path
    
    # Fallback to latest generation checkpoint
    gen_dir = os.path.join(os.path.dirname(cfg.policy_path), "generation_best")
    if os.path.isdir(gen_dir):
        candidates = sorted(
            f for f in os.listdir(gen_dir) if f.endswith("_best_policy.json")
        )
        if candidates:
            path = os.path.join(gen_dir, candidates[-1])
            print(f"best_policy.json not found; using: {path}")
            return path
    
    print(f"ERROR: no policy JSON found at '{cfg.policy_path}' or in '{gen_dir}'")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------

def replay_session(
    session_name: str,
    policy: HistoryNNPolicy,
) -> Dict[str, Any]:
    """
    Run open-loop policy replay on one session.

    Real IID and distance are used at each step; policy outputs steering angles.
    History state (prev_rot1, prev_rot2, …) is updated from the policy's own
    outputs so the history is self-consistent.
    """
    dc = DataProcessor.DataCollection([session_name], cache_dir=False)
    p = dc.processors[0]
    p.load_arena_metadata()

    rob_x = np.asarray(p.rob_x, dtype=np.float64)
    rob_y = np.asarray(p.rob_y, dtype=np.float64)
    rob_yaw_deg = np.asarray(p.rob_yaw_deg, dtype=np.float64)

    corrected_iid_db = np.asarray(
        dc.get_field("sonar_package", "corrected_iid"), dtype=np.float64
    )
    corrected_distance_mm = np.asarray(
        dc.get_field("sonar_package", "corrected_distance"), dtype=np.float64
    ) * 1000.0  # m → mm

    N = min(
        len(rob_x), len(rob_y), len(rob_yaw_deg),
        len(corrected_iid_db), len(corrected_distance_mm),
    )

    hist: collections.deque = collections.deque(maxlen=policy.history_len)
    prev_rot1_norm = 0.0
    prev_rot2_norm = 0.0
    prev_drive_norm = 0.0
    prev_blocked = 0.0

    sign_match_terms: List[float] = []
    steps: List[Dict[str, Any]] = []

    for t in range(N):
        iid_t = safe_float(corrected_iid_db[t], 0.0)
        dist_t = safe_float(corrected_distance_mm[t], 1800.0)
        iid_norm = float(np.clip(iid_t / 12.0, -2.0, 2.0))
        dist_norm = float(np.clip(dist_t / 2000.0, 0.0, 2.0))

        hist.append(np.array(
            [iid_norm, dist_norm, prev_rot1_norm, prev_rot2_norm,
             prev_drive_norm, prev_blocked],
            dtype=np.float32,
        ))
        pad_n = policy.history_len - len(hist)
        if pad_n > 0:
            hist_vec = np.concatenate(
                [np.zeros((pad_n * 6,), dtype=np.float32)] + list(hist), axis=0
            ).astype(np.float32)
        else:
            hist_vec = np.concatenate(list(hist), axis=0).astype(np.float32)

        rotate1, rotate2 = policy.action_degrees(hist_vec, iid_t)
        net_turn_deg = rotate1 + rotate2

        # Update action history from policy's own outputs (self-consistent)
        prev_rot1_norm = float(np.clip(rotate1 / policy.max_rotate1_deg, -1.0, 1.0))
        prev_rot2_norm = float(np.clip(rotate2 / policy.max_rotate2_deg, -1.0, 1.0))
        prev_drive_norm = 1.0   # real robot drove continuously
        prev_blocked = 0.0      # real robot was not blocked

        # Sign-match: same condition as Evaluator.episode()
        if abs(iid_norm) > 0.15 and abs(net_turn_deg) > 2.0:
            sign_match_terms.append(
                1.0 if np.sign(net_turn_deg) == -np.sign(iid_norm) else 0.0
            )

        steps.append({
            "x": float(rob_x[t]),
            "y": float(rob_y[t]),
            "yaw_deg": float(rob_yaw_deg[t]),
            "iid_db": iid_t,
            "distance_mm": dist_t,
            "rotate1_deg": rotate1,
            "rotate2_deg": rotate2,
        })

    sign_match_rate = float(np.mean(sign_match_terms)) if sign_match_terms else float("nan")
    return {
        "steps": steps,
        "sign_match_rate": sign_match_rate,
        "sign_match_n": len(sign_match_terms),
        "wall_x": p.wall_x,
        "wall_y": p.wall_y,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_replay(
    result: Dict[str, Any],
    session_name: str,
    cfg: ReplayConfig,
    policy_meta: Dict[str, Any],
    output_path: str,
) -> None:
    steps = result["steps"]
    if not steps:
        print(f"  (no steps to plot for {session_name})")
        return

    wall_x = result["wall_x"]
    wall_y = result["wall_y"]
    sign_match_rate = result["sign_match_rate"]
    sign_match_n = result["sign_match_n"]
    deadband = safe_float(
        policy_meta.get("iid_deadband_db", cfg.iid_deadband_db), cfg.iid_deadband_db
    )

    N = len(steps)
    xs = np.array([s["x"] for s in steps])
    ys = np.array([s["y"] for s in steps])
    iids = np.array([s["iid_db"] for s in steps])

    pos_mask = iids > deadband
    neg_mask = iids < -deadband
    dead_mask = ~pos_mask & ~neg_mask

    fig, ax = plt.subplots(figsize=(9, 7))

    # 1. Walls (grey scatter)
    if wall_x is not None and wall_y is not None and len(wall_x) > 0:
        ax.scatter(
            wall_x, wall_y, s=1, color="#9e9e9e", alpha=0.25,
            label="Walls", zorder=1,
        )

    # 2. Path coloured by IID sign
    ax.plot(xs, ys, "-", color="#cccccc", linewidth=1, alpha=0.5, zorder=2)
    if pos_mask.any():
        ax.scatter(
            xs[pos_mask], ys[pos_mask], s=8, color="#e65100", alpha=0.7,
            label="IID > 0 (right wall closer)", zorder=3,
        )
    if neg_mask.any():
        ax.scatter(
            xs[neg_mask], ys[neg_mask], s=8, color="#1565c0", alpha=0.7,
            label="IID < 0 (left wall closer)", zorder=3,
        )
    if dead_mask.any():
        ax.scatter(
            xs[dead_mask], ys[dead_mask], s=8, color="#757575", alpha=0.4,
            label="IID ≈ 0 (deadband)", zorder=3,
        )

    # 3. Look-direction quiver: real measured yaw (where sonar fires) — orange
    # 4. Drive-direction quiver: yaw + rotate1 + rotate2 (where policy steers) — teal
    stride = cfg.arrow_stride
    L = cfg.arrow_len_mm
    indices = range(0, N, stride)

    qx = [steps[i]["x"] for i in indices]
    qy = [steps[i]["y"] for i in indices]

    look_rad = [np.deg2rad(steps[i]["yaw_deg"] + steps[i]["rotate1_deg"]) for i in indices]
    lu = [L * np.cos(r) for r in look_rad]
    lv = [L * np.sin(r) for r in look_rad]
    ax.quiver(
        qx, qy, lu, lv,
        units="xy", angles="xy", scale_units="xy", scale=1,
        color="#ff9800", alpha=0.8, width=8.0, headwidth=4, headlength=5,
        label="Look dir (yaw + rotate1)", zorder=5,
    )

    drive_rad = [
        np.deg2rad(steps[i]["yaw_deg"] + steps[i]["rotate1_deg"] + steps[i]["rotate2_deg"])
        for i in indices
    ]
    du = [L * np.cos(r) for r in drive_rad]
    dv = [L * np.sin(r) for r in drive_rad]
    ax.quiver(
        qx, qy, du, dv,
        units="xy", angles="xy", scale_units="xy", scale=1,
        color="#00897b", alpha=0.8, width=8.0, headwidth=4, headlength=5,
        label="Drive dir (policy)", zorder=5,
    )

    # 5. Start / end markers
    ax.scatter(xs[0], ys[0], s=80, color="#2e7d32", zorder=6, label="Start")
    ax.scatter(
        xs[-1], ys[-1], s=80, marker="x", color="#c62828",
        linewidths=2, zorder=6, label="End",
    )

    # Axis limits
    all_x = list(xs)
    all_y = list(ys)
    if wall_x is not None and len(wall_x) > 0:
        all_x.extend(np.asarray(wall_x).tolist())
        all_y.extend(np.asarray(wall_y).tolist())
    min_x, max_x = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    min_y, max_y = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad_x = max(30.0, 0.05 * max(1.0, max_x - min_x))
    pad_y = max(30.0, 0.05 * max(1.0, max_y - min_y))
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    smr_str = f"{sign_match_rate:.3f}" if np.isfinite(sign_match_rate) else "n/a"
    ax.set_title(
        f"Open-loop replay: {session_name} | "
        f"sign_match_rate={smr_str} (n={sign_match_n})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = ReplayConfig()

    policy_path = resolve_policy_path(cfg)
    print(f"Loading policy from: {policy_path}")
    policy, policy_meta = load_policy(policy_path)

    # Override deadband from policy JSON
    if "iid_deadband_db" in policy_meta:
        cfg.iid_deadband_db = safe_float(policy_meta["iid_deadband_db"], cfg.iid_deadband_db)

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(
        f"Policy: history_len={policy.history_len}, hidden={policy.hidden_sizes}, "
        f"deadband={policy.deadband_db:.3f} dB, "
        f"max_rot1={policy.max_rotate1_deg:.0f} deg, "
        f"max_rot2={policy.max_rotate2_deg:.0f} deg"
    )
    print()

    all_smr: List[float] = []
    for session_name in cfg.session_names:
        print(f"Replaying {session_name}...", end=" ", flush=True)
        try:
            result = replay_session(session_name, policy)
            smr = result["sign_match_rate"]
            n = result["sign_match_n"]
            smr_str = f"{smr:.3f}" if np.isfinite(smr) else "n/a"
            print(f"sign_match_rate={smr_str} (n={n})")
            if np.isfinite(smr):
                all_smr.append(smr)
            out_path = os.path.join(cfg.output_dir, f"replay_{session_name}.png")
            plot_replay(result, session_name, cfg, policy_meta, out_path)
            print(f"  -> {out_path}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if all_smr:
        print(
            f"\nOverall mean sign_match_rate: {np.mean(all_smr):.3f} "
            f"(across {len(all_smr)} sessions)"
        )


if __name__ == "__main__":
    main()
