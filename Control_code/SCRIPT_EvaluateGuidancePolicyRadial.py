"""
SCRIPT_EvaluateGuidancePolicyRadial

Evaluate whether predicted occupancy provides better robot-guidance inputs than
baseline sonar (IID + distance), under one shared control policy.

The script:
1. Loads precomputed occupancy windows from SCRIPT_ComputeOccupancyRadial.py.
2. Builds per-step risk triples (left/center/right) from:
   - baseline sonar (window-integrated point proxies),
   - predicted occupancy heatmaps,
   - real profiles (ground-truth oracle source).
3. Applies one shared policy to each risk triple.
4. Scores baseline/occupancy policies against oracle actions and safety margins.
"""

import csv
import json
import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from Library import DataProcessor, DataStorage
from Library.ProfileInference import load_session_data

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")


# ============================================
# CONFIGURATION
# ============================================
session_to_evaluate = "sessionB05"

# Precomputed occupancy from SCRIPT_ComputeOccupancyRadial.py
occupancy_output_root = "OccupancyRadial"
occupancy_npz_path_override = None
occupancy_summary_path_override = None

# Training params source (if None, inferred from occupancy summary)
training_output_dir_override = None

# Output folder
evaluation_output_root = "EvaluationRadial"

# Ground-truth profile loading fallback
real_profile_method_fallback = "ray_center"
real_distance_cutoff_override_mm = None

# Occupancy -> point extraction
occupancy_min_rel_peak = 0.25
occupancy_min_abs = 1e-3

# Sector/risk geometry
center_band_deg = 8.0
side_proxy_frac_of_half_angle = 0.5  # baseline sonar proxy azimuth = +/- half_angle * frac

# Shared guidance policy parameters (applied to baseline/occupancy/truth)
policy_turn_distance_mm = 500.0
policy_side_bias_mm = 140.0
policy_stop_distance_mm = 120.0
policy_step_gain = 0.35
policy_step_min_mm = 0.0
policy_step_max_mm = 220.0

# Safety evaluation
safety_margin_mm = 80.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json_if_exists(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def resolve_occupancy_paths(session_name):
    npz_path = occupancy_npz_path_override
    if npz_path is None:
        npz_path = os.path.join(
            occupancy_output_root,
            session_name,
            f"{session_name}_occupancy_series_data.npz",
        )
    summary_path = occupancy_summary_path_override
    if summary_path is None:
        summary_path = os.path.join(
            occupancy_output_root,
            session_name,
            f"{session_name}_occupancy_summary.json",
        )
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Precomputed occupancy NPZ not found: {npz_path}")
    return npz_path, summary_path


def load_precomputed_occupancy(npz_path):
    data = np.load(npz_path)
    heatmaps_pred = data["heatmaps_pred"] if "heatmaps_pred" in data else data["heatmaps"]
    x_grid = data["x_grid"].astype(np.float32)
    y_grid = data["y_grid"].astype(np.float32)
    anchors = data["anchor_index"].astype(np.int32) if "anchor_index" in data else data["window_end"].astype(np.int32)
    starts = data["window_start"].astype(np.int32) if "window_start" in data else np.maximum(0, anchors)
    ends = data["window_end"].astype(np.int32) if "window_end" in data else anchors.copy()
    chunk_indices = data["chunk_indices"].astype(np.int32) if "chunk_indices" in data else None
    if chunk_indices is None:
        chunk_indices = np.expand_dims(anchors.copy(), axis=1)
    return {
        "heatmaps_pred": heatmaps_pred,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "anchor_index": anchors,
        "window_start": starts,
        "window_end": ends,
        "chunk_indices": chunk_indices,
    }


def load_training_params(training_output_dir):
    params_path = os.path.join(training_output_dir, "training_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Missing training params file: {params_path}")
    with open(params_path, "r") as f:
        return json.load(f)


def load_baseline_from_session(session_name, kept_indices):
    reader = DataStorage.DataReader(session_name)
    filenames = reader.get_all_filenames()
    side_codes = []
    dist_mm = []
    iid_db = []
    for fn in filenames:
        loaded = reader.load_data(fn)
        payload = loaded.get("data", loaded)
        sonar_package = payload.get("sonar_package", None) if isinstance(payload, dict) else None
        if isinstance(sonar_package, dict):
            side_codes.append(str(sonar_package.get("side_code", "U")))
            cd = sonar_package.get("corrected_distance", np.nan)
            iid = sonar_package.get("corrected_iid", np.nan)
            dist_mm.append(float(cd) * 1000.0 if cd is not None else np.nan)
            iid_db.append(float(iid) if iid is not None else np.nan)
        else:
            side_codes.append("U")
            dist_mm.append(np.nan)
            iid_db.append(np.nan)
    side_codes = np.asarray(side_codes, dtype=object)
    dist_mm = np.asarray(dist_mm, dtype=np.float32)
    iid_db = np.asarray(iid_db, dtype=np.float32)
    kept_indices = np.asarray(kept_indices, dtype=np.int32)
    if kept_indices.max(initial=-1) >= len(side_codes):
        raise ValueError("kept_indices exceed baseline sonar sample range.")
    return side_codes[kept_indices], dist_mm[kept_indices], iid_db[kept_indices]


def _risk_from_points(r_mm, az_deg, half_angle_deg, center_deg):
    r = np.asarray(r_mm, dtype=np.float32)
    az = np.asarray(az_deg, dtype=np.float32)
    in_fov = np.abs(az) <= float(half_angle_deg) + 1e-6
    if not np.any(in_fov):
        return np.inf, np.inf, np.inf
    r = r[in_fov]
    az = az[in_fov]
    finite = np.isfinite(r)
    if not np.any(finite):
        return np.inf, np.inf, np.inf
    r = r[finite]
    az = az[finite]

    left = az > float(center_deg)
    right = az < -float(center_deg)
    center = np.abs(az) <= float(center_deg)

    rL = float(np.min(r[left])) if np.any(left) else np.inf
    rR = float(np.min(r[right])) if np.any(right) else np.inf
    rC = float(np.min(r[center])) if np.any(center) else np.inf
    return rL, rC, rR


def risks_from_truth_profile(distance_row_mm, az_row_deg, half_angle_deg, center_deg, cutoff_mm=None):
    d = np.asarray(distance_row_mm, dtype=np.float32).copy()
    if cutoff_mm is not None:
        d[d > float(cutoff_mm)] = np.nan
    return _risk_from_points(d, az_row_deg, half_angle_deg=half_angle_deg, center_deg=center_deg)


def risks_from_occupancy_heatmap(hm, xx, yy, half_angle_deg, center_deg):
    hm = np.asarray(hm, dtype=np.float32)
    max_h = float(np.max(hm))
    if not np.isfinite(max_h) or max_h <= 0.0:
        return np.inf, np.inf, np.inf
    threshold = max(float(occupancy_min_abs), float(occupancy_min_rel_peak) * max_h)
    candidate = hm >= threshold
    if not np.any(candidate):
        return np.inf, np.inf, np.inf
    r = np.hypot(xx, yy)
    az = np.rad2deg(np.arctan2(yy, xx))
    return _risk_from_points(r[candidate], az[candidate], half_angle_deg=half_angle_deg, center_deg=center_deg)


def risks_from_baseline_window(
    window_indices,
    anchor_idx,
    baseline_side,
    baseline_dist_mm,
    rob_x_all,
    rob_y_all,
    rob_yaw_all,
    half_angle_deg,
    center_deg,
):
    proxy_az = {
        "L": +float(half_angle_deg) * float(side_proxy_frac_of_half_angle),
        "R": -float(half_angle_deg) * float(side_proxy_frac_of_half_angle),
        "C": 0.0,
    }
    anchor_x = float(rob_x_all[anchor_idx])
    anchor_y = float(rob_y_all[anchor_idx])
    anchor_yaw = float(rob_yaw_all[anchor_idx])

    r_list = []
    az_list = []
    for idx in window_indices:
        idx = int(idx)
        s = str(baseline_side[idx])
        d = float(baseline_dist_mm[idx]) if np.isfinite(baseline_dist_mm[idx]) else np.nan
        if s not in proxy_az or not np.isfinite(d):
            continue
        az_local = np.asarray([proxy_az[s]], dtype=np.float32)
        dist_local = np.asarray([d], dtype=np.float32)
        xw, yw = DataProcessor.robot2world(
            az_local,
            dist_local,
            float(rob_x_all[idx]),
            float(rob_y_all[idx]),
            float(rob_yaw_all[idx]),
        )
        xr, yr = DataProcessor.world2robot(xw, yw, anchor_x, anchor_y, anchor_yaw)
        r_list.append(float(np.hypot(xr[0], yr[0])))
        az_list.append(float(np.rad2deg(np.arctan2(yr[0], xr[0]))))

    if not r_list:
        return np.inf, np.inf, np.inf
    return _risk_from_points(
        np.asarray(r_list, dtype=np.float32),
        np.asarray(az_list, dtype=np.float32),
        half_angle_deg=half_angle_deg,
        center_deg=center_deg,
    )


def shared_policy(rL, rC, rR):
    # Treat no-evidence as open space.
    if not np.isfinite(rL):
        rL = np.inf
    if not np.isfinite(rR):
        rR = np.inf
    if not np.isfinite(rC):
        rC = np.inf

    safer_side = "L" if rL >= rR else "R"
    if rC < float(policy_turn_distance_mm):
        turn = safer_side
    elif abs(rL - rR) > float(policy_side_bias_mm):
        turn = safer_side
    else:
        turn = "S"

    if rC <= float(policy_stop_distance_mm):
        step_mm = 0.0
    else:
        step_mm = float(policy_step_gain) * (rC - float(policy_stop_distance_mm))
    step_mm = float(np.clip(step_mm, float(policy_step_min_mm), float(policy_step_max_mm)))
    return turn, step_mm


def _turn_to_int(turn):
    return {"L": -1, "S": 0, "R": 1}.get(str(turn), 0)


def evaluate_policy(rows, turn_col, step_col):
    oracle_turn = np.asarray([str(r["oracle_turn"]) for r in rows], dtype=object)
    pred_turn = np.asarray([str(r[turn_col]) for r in rows], dtype=object)
    oracle_step = np.asarray([r["oracle_step_mm"] for r in rows], dtype=np.float32)
    pred_step = np.asarray([r[step_col] for r in rows], dtype=np.float32)
    truth_rC = np.asarray([r["truth_rC_mm"] for r in rows], dtype=np.float32)

    turn_acc_all = float(np.mean(pred_turn == oracle_turn))
    decisive = oracle_turn != "S"
    turn_acc_decisive = float(np.mean(pred_turn[decisive] == oracle_turn[decisive])) if np.any(decisive) else np.nan

    safe_step_max = np.where(
        np.isfinite(truth_rC),
        np.maximum(truth_rC - float(safety_margin_mm), 0.0),
        float(policy_step_max_mm),
    )
    unsafe = pred_step > (safe_step_max + 1e-6)
    clearance_margin = safe_step_max - pred_step

    step_mae = float(np.mean(np.abs(pred_step - oracle_step)))
    turn_mae = float(np.mean(np.abs(np.vectorize(_turn_to_int)(pred_turn) - np.vectorize(_turn_to_int)(oracle_turn))))

    return {
        "n_rows": int(len(rows)),
        "turn_accuracy_all": turn_acc_all,
        "turn_accuracy_decisive": None if np.isnan(turn_acc_decisive) else float(turn_acc_decisive),
        "step_mae_vs_oracle_mm": step_mae,
        "turn_mae_vs_oracle_code": turn_mae,
        "unsafe_action_rate": float(np.mean(unsafe)),
        "mean_clearance_margin_mm": float(np.mean(clearance_margin)),
        "p10_clearance_margin_mm": float(np.quantile(clearance_margin, 0.10)),
        "mean_step_mm": float(np.mean(pred_step)),
    }


def _turn_confusion(oracle_turn, pred_turn):
    labels = ["L", "S", "R"]
    cm = np.zeros((3, 3), dtype=np.int32)
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            cm[i, j] = int(np.sum((oracle_turn == t) & (pred_turn == p)))
    return labels, cm


def create_plots(rows, out_dir, session_name, metrics_baseline, metrics_occ):
    ensure_dir(out_dir)
    oracle_turn = np.asarray([str(r["oracle_turn"]) for r in rows], dtype=object)
    base_turn = np.asarray([str(r["baseline_turn"]) for r in rows], dtype=object)
    occ_turn = np.asarray([str(r["occupancy_turn"]) for r in rows], dtype=object)
    oracle_step = np.asarray([r["oracle_step_mm"] for r in rows], dtype=np.float32)
    base_step = np.asarray([r["baseline_step_mm"] for r in rows], dtype=np.float32)
    occ_step = np.asarray([r["occupancy_step_mm"] for r in rows], dtype=np.float32)
    truth_rC = np.asarray([r["truth_rC_mm"] for r in rows], dtype=np.float32)
    safe_step_max = np.where(
        np.isfinite(truth_rC),
        np.maximum(truth_rC - float(safety_margin_mm), 0.0),
        float(policy_step_max_mm),
    )
    base_margin = safe_step_max - base_step
    occ_margin = safe_step_max - occ_step

    # Safety + turn summary bars
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    unsafe_vals = [
        float(metrics_baseline["unsafe_action_rate"]),
        float(metrics_occ["unsafe_action_rate"]),
    ]
    axes[0].bar(["Baseline", "Occupancy"], unsafe_vals, color=["tab:blue", "tab:orange"])
    axes[0].set_title("Unsafe Action Rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.25)

    turn_vals = [
        float(metrics_baseline["turn_accuracy_decisive"] or np.nan),
        float(metrics_occ["turn_accuracy_decisive"] or np.nan),
    ]
    axes[1].bar(["Baseline", "Occupancy"], turn_vals, color=["tab:blue", "tab:orange"])
    axes[1].set_title("Turn Accuracy (Oracle != Straight)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{session_name}_policy_summary_bars.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Step vs oracle scatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lim_max = float(np.nanmax(np.concatenate([oracle_step, base_step, occ_step])))
    for ax, pred, label, color in [
        (axes[0], base_step, "Baseline", "tab:blue"),
        (axes[1], occ_step, "Occupancy", "tab:orange"),
    ]:
        ax.scatter(oracle_step, pred, s=8, alpha=0.28, color=color)
        ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1.0)
        if np.std(oracle_step) > 1e-12 and np.std(pred) > 1e-12:
            corr = float(np.corrcoef(oracle_step, pred)[0, 1])
            ax.set_title(f"{label} Step vs Oracle (r={corr:.3f})")
        else:
            ax.set_title(f"{label} Step vs Oracle")
        ax.set_xlabel("Oracle step (mm)")
        ax.set_ylabel("Predicted step (mm)")
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{session_name}_policy_step_scatter.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Clearance margin distributions
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    base_margin_finite = base_margin[np.isfinite(base_margin)]
    occ_margin_finite = occ_margin[np.isfinite(occ_margin)]
    ax.hist(base_margin_finite, bins=45, alpha=0.5, label="Baseline", color="tab:blue", density=True)
    ax.hist(occ_margin_finite, bins=45, alpha=0.5, label="Occupancy", color="tab:orange", density=True)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax.set_title("Clearance Margin Distribution")
    ax.set_xlabel("Safe step max - proposed step (mm)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{session_name}_policy_clearance_margin_hist.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Turn confusion matrices (row-normalized by oracle turn)
    labels, cm_b = _turn_confusion(oracle_turn, base_turn)
    _, cm_o = _turn_confusion(oracle_turn, occ_turn)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, cm, title in [
        (axes[0], cm_b, "Baseline Turn vs Oracle (row-normalized)"),
        (axes[1], cm_o, "Occupancy Turn vs Oracle (row-normalized)"),
    ]:
        row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cmn = cm / row_sum
        im = ax.imshow(cmn, cmap="Blues", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("Predicted turn")
        ax.set_ylabel("Oracle turn")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n(n={cm[i, j]})", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{session_name}_policy_turn_confusion.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dir(evaluation_output_root)
    session_out_dir = os.path.join(evaluation_output_root, session_to_evaluate)
    ensure_dir(session_out_dir)

    occ_npz_path, occ_summary_path = resolve_occupancy_paths(session_to_evaluate)
    occ_summary = load_json_if_exists(occ_summary_path)
    occ = load_precomputed_occupancy(occ_npz_path)

    training_output_dir = (
        training_output_dir_override
        if training_output_dir_override is not None
        else occ_summary.get("training_output_dir", "TrainingRadial")
    )
    params = load_training_params(training_output_dir)
    profile_opening_angle = float(params["profile_opening_angle"])
    profile_steps = int(params["profile_steps"])
    half_angle_deg = 0.5 * float(profile_opening_angle)
    real_profile_method = str(occ_summary.get("real_profile_method", real_profile_method_fallback))
    real_cutoff_mm = params.get("distance_threshold", None)
    if real_distance_cutoff_override_mm is not None:
        real_cutoff_mm = float(real_distance_cutoff_override_mm)
    elif real_cutoff_mm is not None:
        real_cutoff_mm = float(real_cutoff_mm)

    sonar_data, real_distance_mm, profile_centers_deg, metadata = load_session_data(
        session_to_evaluate,
        profile_opening_angle,
        profile_steps,
        profile_method=real_profile_method,
    )
    print(f"Loaded session {session_to_evaluate}: {len(sonar_data)} filtered samples")

    baseline_side, baseline_dist_mm, baseline_iid_db = load_baseline_from_session(
        session_to_evaluate,
        metadata["kept_indices"],
    )

    x_grid = occ["x_grid"]
    y_grid = occ["y_grid"]
    xx, yy = np.meshgrid(x_grid, y_grid)
    heatmaps = occ["heatmaps_pred"]
    chunk_indices = occ["chunk_indices"]
    anchors = occ["anchor_index"]
    starts = occ["window_start"]
    ends = occ["window_end"]

    if heatmaps.shape[0] != len(anchors):
        raise ValueError("Mismatch: number of heatmaps != number of anchor indices.")
    if np.max(anchors, initial=-1) >= len(real_distance_mm):
        raise ValueError("Anchor index exceeds filtered session size.")

    rob_x_all = metadata["rob_x"]
    rob_y_all = metadata["rob_y"]
    rob_yaw_all = metadata["rob_yaw_deg"]

    rows = []
    for sidx in range(heatmaps.shape[0]):
        anchor_idx = int(anchors[sidx])
        win_idx = chunk_indices[sidx]
        truth_rL, truth_rC, truth_rR = risks_from_truth_profile(
            real_distance_mm[anchor_idx],
            profile_centers_deg[anchor_idx],
            half_angle_deg=half_angle_deg,
            center_deg=center_band_deg,
            cutoff_mm=real_cutoff_mm,
        )
        occ_rL, occ_rC, occ_rR = risks_from_occupancy_heatmap(
            heatmaps[sidx],
            xx,
            yy,
            half_angle_deg=half_angle_deg,
            center_deg=center_band_deg,
        )
        base_rL, base_rC, base_rR = risks_from_baseline_window(
            win_idx,
            anchor_idx=anchor_idx,
            baseline_side=baseline_side,
            baseline_dist_mm=baseline_dist_mm,
            rob_x_all=rob_x_all,
            rob_y_all=rob_y_all,
            rob_yaw_all=rob_yaw_all,
            half_angle_deg=half_angle_deg,
            center_deg=center_band_deg,
        )

        oracle_turn, oracle_step = shared_policy(truth_rL, truth_rC, truth_rR)
        base_turn, base_step = shared_policy(base_rL, base_rC, base_rR)
        occ_turn, occ_step = shared_policy(occ_rL, occ_rC, occ_rR)

        rows.append(
            {
                "series_idx": int(sidx),
                "window_start": int(starts[sidx]) if len(starts) > sidx else int(anchor_idx),
                "window_end": int(ends[sidx]) if len(ends) > sidx else int(anchor_idx),
                "anchor_idx_filtered": int(anchor_idx),
                "anchor_idx_original": int(metadata["kept_indices"][anchor_idx]),
                "truth_rL_mm": float(truth_rL),
                "truth_rC_mm": float(truth_rC),
                "truth_rR_mm": float(truth_rR),
                "baseline_rL_mm": float(base_rL),
                "baseline_rC_mm": float(base_rC),
                "baseline_rR_mm": float(base_rR),
                "occupancy_rL_mm": float(occ_rL),
                "occupancy_rC_mm": float(occ_rC),
                "occupancy_rR_mm": float(occ_rR),
                "oracle_turn": str(oracle_turn),
                "oracle_step_mm": float(oracle_step),
                "baseline_turn": str(base_turn),
                "baseline_step_mm": float(base_step),
                "occupancy_turn": str(occ_turn),
                "occupancy_step_mm": float(occ_step),
                "baseline_sonar_side": str(baseline_side[anchor_idx]),
                "baseline_sonar_distance_mm": float(baseline_dist_mm[anchor_idx]) if np.isfinite(baseline_dist_mm[anchor_idx]) else np.nan,
                "baseline_sonar_iid_db": float(baseline_iid_db[anchor_idx]) if np.isfinite(baseline_iid_db[anchor_idx]) else np.nan,
            }
        )

    csv_path = os.path.join(session_out_dir, f"{session_to_evaluate}_policy_eval.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved per-step policy CSV: {csv_path}")

    metrics_baseline = evaluate_policy(rows, turn_col="baseline_turn", step_col="baseline_step_mm")
    metrics_occupancy = evaluate_policy(rows, turn_col="occupancy_turn", step_col="occupancy_step_mm")

    create_plots(
        rows=rows,
        out_dir=session_out_dir,
        session_name=session_to_evaluate,
        metrics_baseline=metrics_baseline,
        metrics_occ=metrics_occupancy,
    )

    summary = {
        "session": session_to_evaluate,
        "occupancy_source_npz": occ_npz_path,
        "occupancy_source_summary": occ_summary_path if os.path.exists(occ_summary_path) else None,
        "training_output_dir": training_output_dir,
        "profile_opening_angle_deg": profile_opening_angle,
        "profile_steps": profile_steps,
        "real_profile_method": real_profile_method,
        "real_profile_distance_cutoff_mm": None if real_cutoff_mm is None else float(real_cutoff_mm),
        "n_steps_evaluated": int(len(rows)),
        "policy": {
            "turn_distance_mm": float(policy_turn_distance_mm),
            "side_bias_mm": float(policy_side_bias_mm),
            "stop_distance_mm": float(policy_stop_distance_mm),
            "step_gain": float(policy_step_gain),
            "step_min_mm": float(policy_step_min_mm),
            "step_max_mm": float(policy_step_max_mm),
        },
        "risk_definition": {
            "half_fov_deg": float(half_angle_deg),
            "center_band_deg": float(center_band_deg),
            "baseline_proxy_side_azimuth_deg": {
                "L": float(+half_angle_deg * side_proxy_frac_of_half_angle),
                "R": float(-half_angle_deg * side_proxy_frac_of_half_angle),
                "C": 0.0,
            },
            "occupancy_threshold": {
                "min_rel_peak": float(occupancy_min_rel_peak),
                "min_abs": float(occupancy_min_abs),
            },
        },
        "safety_evaluation": {
            "safe_step_max_mm": "max(truth_rC_mm - safety_margin_mm, 0)",
            "safety_margin_mm": float(safety_margin_mm),
        },
        "metrics": {
            "baseline_policy": metrics_baseline,
            "occupancy_policy": metrics_occupancy,
        },
        "output_csv": csv_path,
    }
    summary_path = os.path.join(session_out_dir, f"{session_to_evaluate}_policy_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
