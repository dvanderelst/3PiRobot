"""
SCRIPT_EvaluateClosestObstacleRadial

Evaluate closest-obstacle side and distance prediction quality for one session using
precomputed occupancy maps from SCRIPT_ComputeOccupancyRadial.py:
- Baseline sonar control signal (corrected IID + corrected distance)
- Predicted occupancy closest obstacle (from saved heatmaps)
Against real profiles as ground truth.
"""

import csv
import json
import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from Library import DataStorage
from Library.ProfileInference import load_session_data

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")


# ============================================
# CONFIGURATION
# ============================================
session_to_evaluate = "sessionB05"

# Use precomputed occupancy package from SCRIPT_ComputeOccupancyRadial.py
occupancy_output_root = "OccupancyRadial"
occupancy_npz_path_override = None
occupancy_summary_path_override = None

# Training params source (if None, inferred from occupancy summary)
training_output_dir_override = None

# Evaluation outputs
evaluation_output_root = "EvaluationRadial"

# Ground-truth profile settings fallback (used if occupancy summary does not specify)
real_profile_method_fallback = "ray_center"
real_distance_cutoff_override_mm = None

# Closest extraction settings from occupancy map.
occupancy_min_rel_peak = 0.25
occupancy_min_abs = 1e-3
center_band_mm = 25.0
iid_side_eps_db = 0.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def side_from_y(y_val, center_band):
    if y_val > center_band:
        return "L"
    if y_val < -center_band:
        return "R"
    return "C"


def side_from_azimuth(az_deg, center_band_deg=1.0):
    y_sign = np.sin(np.deg2rad(float(az_deg)))
    if y_sign > np.sin(np.deg2rad(center_band_deg)):
        return "L"
    if y_sign < -np.sin(np.deg2rad(center_band_deg)):
        return "R"
    return "C"


def side_from_iid_db(iid_db, eps_db=0.0):
    if not np.isfinite(iid_db):
        return "U"
    if iid_db < -float(eps_db):
        return "L"
    if iid_db > float(eps_db):
        return "R"
    return "C"


def load_json_if_exists(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def load_training_params(training_output_dir):
    params_path = os.path.join(training_output_dir, "training_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Missing training params file: {params_path}")
    with open(params_path, "r") as f:
        return json.load(f)


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

    starts = data["window_start"].astype(np.int32) if "window_start" in data else None
    ends = data["window_end"].astype(np.int32) if "window_end" in data else None
    anchor_idx = data["anchor_index"].astype(np.int32) if "anchor_index" in data else None

    if anchor_idx is None:
        if ends is not None:
            anchor_idx = ends.astype(np.int32)
        else:
            raise ValueError("Occupancy NPZ missing both 'anchor_index' and 'window_end'.")

    if starts is None:
        # Fallback for older files.
        starts = np.maximum(0, anchor_idx).astype(np.int32)

    return {
        "heatmaps_pred": heatmaps_pred,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "window_start": starts,
        "window_end": ends if ends is not None else anchor_idx,
        "anchor_index": anchor_idx,
    }


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
            dist_mm.append(float(cd) * 1000.0 if cd is not None else np.nan)
            iid = sonar_package.get("corrected_iid", np.nan)
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
        raise ValueError(
            f"kept_indices refer to original index {int(kept_indices.max())}, "
            f"but baseline has only {len(side_codes)} samples."
        )
    return side_codes[kept_indices], dist_mm[kept_indices], iid_db[kept_indices]


def closest_from_heatmap(hm, xx, yy, grid_mm):
    hm = np.asarray(hm, dtype=np.float32)
    max_h = float(np.max(hm))
    if not np.isfinite(max_h) or max_h <= 0.0:
        return "U", np.nan

    threshold = max(float(occupancy_min_abs), float(occupancy_min_rel_peak) * max_h)
    candidate = hm >= threshold
    if not np.any(candidate):
        return "U", np.nan

    r = np.hypot(xx, yy)
    r_masked = np.where(candidate, r, np.inf)
    min_r = float(np.min(r_masked))
    if not np.isfinite(min_r):
        return "U", np.nan

    tie = candidate & np.isclose(r, min_r, atol=0.5 * float(grid_mm))
    y_mean = float(np.mean(yy[tie])) if np.any(tie) else 0.0
    side = side_from_y(y_mean, center_band_mm)
    return side, min_r


def closest_from_profile(distance_row_mm, az_row_deg, max_distance_mm=None):
    d = np.asarray(distance_row_mm, dtype=np.float32).copy()
    if max_distance_mm is not None:
        d[d > float(max_distance_mm)] = np.nan
    valid = np.isfinite(d)
    if not np.any(valid):
        return "U", np.nan
    j = int(np.nanargmin(d))
    dist_mm = float(d[j])
    side = side_from_azimuth(float(az_row_deg[j]))
    return side, dist_mm


def compute_metrics(rows, pred_side_col, pred_dist_col):
    truth_dist = np.asarray([r["truth_distance_mm"] for r in rows], dtype=np.float32)
    pred_dist = np.asarray([r[pred_dist_col] for r in rows], dtype=np.float32)
    truth_side = np.asarray([str(r["truth_side"]) for r in rows], dtype=object)
    pred_side = np.asarray([str(r[pred_side_col]) for r in rows], dtype=object)

    truth_valid = np.isfinite(truth_dist)
    pred_valid = np.isfinite(pred_dist)
    both_valid = truth_valid & pred_valid

    truth_lr = np.isin(truth_side, ["L", "R"])
    side_acc = np.mean(pred_side[truth_lr] == truth_side[truth_lr]) if np.any(truth_lr) else np.nan

    mae = np.mean(np.abs(pred_dist[both_valid] - truth_dist[both_valid])) if np.any(both_valid) else np.nan
    if np.sum(both_valid) >= 2:
        td = truth_dist[both_valid]
        pd = pred_dist[both_valid]
        if np.std(td) > 1e-12 and np.std(pd) > 1e-12:
            corr = float(np.corrcoef(td, pd)[0, 1])
        else:
            corr = np.nan
    else:
        corr = np.nan

    side_correct = (pred_side == truth_side) & truth_lr & both_valid
    mae_when_side_correct = (
        np.mean(np.abs(pred_dist[side_correct] - truth_dist[side_correct]))
        if np.any(side_correct)
        else np.nan
    )

    return {
        "n_rows": int(len(rows)),
        "n_truth_valid": int(np.sum(truth_valid)),
        "n_both_valid": int(np.sum(both_valid)),
        "side_accuracy_lr": None if np.isnan(side_acc) else float(side_acc),
        "distance_mae_mm": None if np.isnan(mae) else float(mae),
        "distance_corr": None if np.isnan(corr) else float(corr),
        "distance_mae_mm_when_side_correct": None if np.isnan(mae_when_side_correct) else float(mae_when_side_correct),
    }


def compute_iid_sign_metrics(rows):
    truth_side = np.asarray([str(r["truth_side"]) for r in rows], dtype=object)
    iid_db = np.asarray([r["sonar_iid_db"] for r in rows], dtype=np.float32)
    iid_side = np.asarray([side_from_iid_db(v, eps_db=iid_side_eps_db) for v in iid_db], dtype=object)
    sonar_side = np.asarray([str(r["sonar_side"]) for r in rows], dtype=object)

    truth_lr = np.isin(truth_side, ["L", "R"])
    iid_valid = np.isfinite(iid_db)
    mask = truth_lr & iid_valid

    side_acc = np.mean(iid_side[mask] == truth_side[mask]) if np.any(mask) else np.nan
    match_side_code = np.mean(iid_side[iid_valid] == sonar_side[iid_valid]) if np.any(iid_valid) else np.nan
    return {
        "iid_eps_db": float(iid_side_eps_db),
        "n_truth_lr_with_iid": int(np.sum(mask)),
        "side_accuracy_lr_from_iid_sign": None if np.isnan(side_acc) else float(side_acc),
        "iid_sign_matches_side_code_rate": None if np.isnan(match_side_code) else float(match_side_code),
    }


def _confusion_counts(pred_side, truth_side):
    labels = ["L", "C", "R"]
    mat = np.zeros((3, 3), dtype=np.int32)
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            mat[i, j] = int(np.sum((truth_side == t) & (pred_side == p)))
    return labels, mat


def create_evaluation_plots(rows, out_dir, session_name, sonar_metrics, occ_metrics):
    ensure_dir(out_dir)
    truth_side = np.asarray([str(r["truth_side"]) for r in rows], dtype=object)
    sonar_side = np.asarray([str(r["sonar_side"]) for r in rows], dtype=object)
    occ_side = np.asarray([str(r["occ_side"]) for r in rows], dtype=object)
    truth_dist = np.asarray([r["truth_distance_mm"] for r in rows], dtype=np.float32)
    sonar_dist = np.asarray([r["sonar_distance_mm"] for r in rows], dtype=np.float32)
    occ_dist = np.asarray([r["occ_distance_mm"] for r in rows], dtype=np.float32)

    # Side confusion matrices
    labels, cm_sonar = _confusion_counts(sonar_side, truth_side)
    _, cm_occ = _confusion_counts(occ_side, truth_side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in [
        (axes[0], cm_sonar, "Baseline Sonar Side vs Truth (row-normalized)"),
        (axes[1], cm_occ, "Occupancy Side vs Truth (row-normalized)"),
    ]:
        row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_norm = cm / row_sum
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("Predicted side")
        ax.set_ylabel("Truth side")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{session_name}_side_confusion.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Distance scatter + MAE bars
    valid_sonar = np.isfinite(truth_dist) & np.isfinite(sonar_dist)
    valid_occ = np.isfinite(truth_dist) & np.isfinite(occ_dist)
    lim_max = np.nanmax(
        np.concatenate(
            [
                truth_dist[np.isfinite(truth_dist)],
                sonar_dist[np.isfinite(sonar_dist)],
                occ_dist[np.isfinite(occ_dist)],
            ]
        )
    )
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].scatter(truth_dist[valid_sonar], sonar_dist[valid_sonar], s=8, alpha=0.25, color="tab:blue")
    axes[0].plot([0, lim_max], [0, lim_max], "k--", linewidth=1.0)
    sonar_corr = sonar_metrics.get("distance_corr", None)
    axes[0].set_title(
        "Baseline Distance" + (f" (r={sonar_corr:.3f})" if sonar_corr is not None else "")
    )
    axes[0].set_xlabel("Truth (mm)")
    axes[0].set_ylabel("Pred (mm)")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(truth_dist[valid_occ], occ_dist[valid_occ], s=8, alpha=0.25, color="tab:orange")
    axes[1].plot([0, lim_max], [0, lim_max], "k--", linewidth=1.0)
    occ_corr = occ_metrics.get("distance_corr", None)
    axes[1].set_title(
        "Occupancy Distance" + (f" (r={occ_corr:.3f})" if occ_corr is not None else "")
    )
    axes[1].set_xlabel("Truth (mm)")
    axes[1].set_ylabel("Pred (mm)")
    axes[1].grid(True, alpha=0.25)

    mae_vals = [
        float(sonar_metrics.get("distance_mae_mm") or np.nan),
        float(occ_metrics.get("distance_mae_mm") or np.nan),
    ]
    axes[2].bar(["Baseline", "Occupancy"], mae_vals, color=["tab:blue", "tab:orange"])
    axes[2].set_title("Distance MAE (mm)")
    axes[2].grid(True, axis="y", alpha=0.25)

    corr_vals = [
        float(sonar_corr) if sonar_corr is not None else np.nan,
        float(occ_corr) if occ_corr is not None else np.nan,
    ]
    axes[3].bar(["Baseline", "Occupancy"], corr_vals, color=["tab:blue", "tab:orange"])
    axes[3].set_title("Distance Correlation (r)")
    axes[3].set_ylim(-1.0, 1.0)
    axes[3].grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{session_name}_distance_eval.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Error histogram
    sonar_err = np.abs(sonar_dist[valid_sonar] - truth_dist[valid_sonar])
    occ_err = np.abs(occ_dist[valid_occ] - truth_dist[valid_occ])
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(sonar_err, bins=40, alpha=0.5, label="Baseline", color="tab:blue", density=True)
    ax.hist(occ_err, bins=40, alpha=0.5, label="Occupancy", color="tab:orange", density=True)
    ax.set_title("Absolute Distance Error Distribution")
    ax.set_xlabel("Absolute error (mm)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{session_name}_distance_error_hist.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dir(evaluation_output_root)
    session_out_dir = os.path.join(evaluation_output_root, session_to_evaluate)
    ensure_dir(session_out_dir)

    occ_npz_path, occ_summary_path = resolve_occupancy_paths(session_to_evaluate)
    occ_summary = load_json_if_exists(occ_summary_path)
    occ_data = load_precomputed_occupancy(occ_npz_path)

    training_output_dir = (
        training_output_dir_override
        if training_output_dir_override is not None
        else occ_summary.get("training_output_dir", "TrainingRadial")
    )
    params = load_training_params(training_output_dir)

    profile_opening_angle = float(params["profile_opening_angle"])
    profile_steps = int(params["profile_steps"])

    real_profile_method = str(occ_summary.get("real_profile_method", real_profile_method_fallback))
    real_distance_cutoff_mm = params.get("distance_threshold", None)
    if real_distance_cutoff_override_mm is not None:
        real_distance_cutoff_mm = float(real_distance_cutoff_override_mm)
    elif real_distance_cutoff_mm is not None:
        real_distance_cutoff_mm = float(real_distance_cutoff_mm)

    sonar_data, real_distance_mm, profile_centers_deg, metadata = load_session_data(
        session_to_evaluate,
        profile_opening_angle,
        profile_steps,
        profile_method=real_profile_method,
    )
    print(f"Loaded session {session_to_evaluate}: {len(sonar_data)} filtered samples")

    baseline_side, baseline_distance_mm, baseline_iid_db = load_baseline_from_session(
        session_to_evaluate,
        metadata["kept_indices"],
    )

    x_grid = occ_data["x_grid"]
    y_grid = occ_data["y_grid"]
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_mm = float(np.mean(np.diff(x_grid))) if len(x_grid) >= 2 else 1.0

    heatmaps_pred = occ_data["heatmaps_pred"]
    starts = occ_data["window_start"]
    ends = occ_data["window_end"]
    anchors = occ_data["anchor_index"]

    if heatmaps_pred.shape[0] != len(anchors):
        raise ValueError("Mismatch: number of heatmaps does not match number of anchor indices.")

    max_anchor = int(np.max(anchors)) if len(anchors) else -1
    if max_anchor >= len(real_distance_mm):
        raise ValueError(
            f"Occupancy anchor index {max_anchor} exceeds filtered sample count {len(real_distance_mm)}."
        )

    rows = []
    for sidx in range(heatmaps_pred.shape[0]):
        anchor_idx = int(anchors[sidx])
        occ_side, occ_dist_mm = closest_from_heatmap(heatmaps_pred[sidx], xx, yy, grid_mm=grid_mm)
        truth_side, truth_dist_mm = closest_from_profile(
            real_distance_mm[anchor_idx],
            profile_centers_deg[anchor_idx],
            max_distance_mm=real_distance_cutoff_mm,
        )

        rows.append(
            {
                "series_idx": int(sidx),
                "window_start": int(starts[sidx]) if len(starts) > sidx else int(anchor_idx),
                "window_end": int(ends[sidx]) if len(ends) > sidx else int(anchor_idx),
                "anchor_idx_filtered": int(anchor_idx),
                "anchor_idx_original": int(metadata["kept_indices"][anchor_idx]),
                "truth_side": str(truth_side),
                "truth_distance_mm": float(truth_dist_mm) if np.isfinite(truth_dist_mm) else np.nan,
                "sonar_side": str(baseline_side[anchor_idx]),
                "sonar_distance_mm": float(baseline_distance_mm[anchor_idx]) if np.isfinite(baseline_distance_mm[anchor_idx]) else np.nan,
                "sonar_iid_db": float(baseline_iid_db[anchor_idx]) if np.isfinite(baseline_iid_db[anchor_idx]) else np.nan,
                "occ_side": str(occ_side),
                "occ_distance_mm": float(occ_dist_mm) if np.isfinite(occ_dist_mm) else np.nan,
            }
        )

    csv_path = os.path.join(session_out_dir, f"{session_to_evaluate}_closest_obstacle_eval.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved per-window evaluation CSV: {csv_path}")

    sonar_metrics = compute_metrics(rows, pred_side_col="sonar_side", pred_dist_col="sonar_distance_mm")
    occ_metrics = compute_metrics(rows, pred_side_col="occ_side", pred_dist_col="occ_distance_mm")
    iid_sign_metrics = compute_iid_sign_metrics(rows)

    create_evaluation_plots(
        rows=rows,
        out_dir=session_out_dir,
        session_name=session_to_evaluate,
        sonar_metrics=sonar_metrics,
        occ_metrics=occ_metrics,
    )

    summary = {
        "session": session_to_evaluate,
        "occupancy_source_npz": occ_npz_path,
        "occupancy_source_summary": occ_summary_path if os.path.exists(occ_summary_path) else None,
        "training_output_dir": training_output_dir,
        "model_target_type": params.get("target_type", "unknown"),
        "real_profile_method": real_profile_method,
        "real_profile_distance_cutoff_mm": None if real_distance_cutoff_mm is None else float(real_distance_cutoff_mm),
        "window_start_used": int(np.min(starts)) if len(starts) else None,
        "window_end_used": int(np.max(ends)) if len(ends) else None,
        "window_size": int(occ_summary.get("window_size", 1)) if occ_summary else None,
        "window_shift": int(occ_summary.get("window_shift", 1)) if occ_summary else None,
        "n_windows": int(len(rows)),
        "occupancy_closest_threshold": {
            "occupancy_min_rel_peak": float(occupancy_min_rel_peak),
            "occupancy_min_abs": float(occupancy_min_abs),
            "center_band_mm": float(center_band_mm),
        },
        "side_class_definition": {
            "truth_from_profile_azimuth": {
                "L": "sin(azimuth_deg) > sin(1 deg)",
                "R": "sin(azimuth_deg) < -sin(1 deg)",
                "C": "otherwise",
            },
            "occupancy_from_closest_heatmap_y_mm": {
                "L": f"closest y > {float(center_band_mm):.1f} mm",
                "R": f"closest y < -{float(center_band_mm):.1f} mm",
                "C": "otherwise",
            },
            "sonar_side_code": "from sonar_package['side_code']",
            "iid_sign_side": {
                "L": f"corrected_iid < -{float(iid_side_eps_db):.3f} dB",
                "R": f"corrected_iid > +{float(iid_side_eps_db):.3f} dB",
                "C": "otherwise",
            },
        },
        "metrics": {
            "baseline_sonar": sonar_metrics,
            "predicted_occupancy": occ_metrics,
            "iid_sign_side_eval": iid_sign_metrics,
        },
        "output_csv": csv_path,
    }
    summary_path = os.path.join(session_out_dir, f"{session_to_evaluate}_closest_obstacle_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
