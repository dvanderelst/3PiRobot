"""
SCRIPT_AssessOccupancy

Assess occupancy-derived curvature using precomputed occupancy tensors:
- Compare curvature from predicted occupancy vs real-profile occupancy.
- Compare occupancy curvature vs sonar IID and sonar distance.
"""

import json
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from Library import DataProcessor, Settings
from Library.CurvatureCalculation import plan_circles_from_heatmap, planner_to_curvature
from Library.SteeringConfigClass import SteeringConfig


# ============================================
# CONFIGURATION
# ============================================
session_to_assess = 'sessionB05'
occupancy_output_root = 'Occupancy'

# Planner settings used to convert occupancy maps -> curvature.
planner_config = SteeringConfig(
    grid_mm=Settings.occupancy_config.grid_mm,
    occ_block_threshold=Settings.curvature_config.occ_block_threshold,
    robot_radius_mm=Settings.curvature_config.robot_radius_mm,
    safety_margin_mm=Settings.curvature_config.safety_margin_mm,
    circle_radius_min_mm=Settings.curvature_config.circle_radius_min_mm,
    circle_radius_max_mm=Settings.curvature_config.circle_radius_max_mm,
    circle_radius_step_mm=Settings.curvature_config.circle_radius_step_mm,
    circle_arc_samples=Settings.curvature_config.circle_arc_samples,
    circle_horizon_x_mm=Settings.curvature_config.circle_horizon_x_mm,
    circle_radius_tie_mm=Settings.curvature_config.circle_radius_tie_mm,
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_corrcoef(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return np.nan
    x = x[valid]
    y = y[valid]
    rx = np.empty_like(x, dtype=float)
    ry = np.empty_like(y, dtype=float)
    rx[np.argsort(x, kind='mergesort')] = np.arange(len(x), dtype=float)
    ry[np.argsort(y, kind='mergesort')] = np.arange(len(y), dtype=float)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def load_filtered_iid_distance(session_name):
    """
    Recreate filtered-index alignment used by profile/occupancy generation:
    keep rows where sonar and profiles are finite.
    """
    dc = DataProcessor.DataCollection([session_name])
    sonar_data = dc.load_sonar(flatten=False)
    profiles_data, _ = dc.load_profiles(opening_angle=60, steps=11)

    sonar_iid = np.asarray(dc.get_field('sonar_package', 'corrected_iid'), dtype=np.float32)
    sonar_dist = np.asarray(dc.get_field('sonar_package', 'corrected_distance'), dtype=np.float32)

    finite_mask = np.isfinite(sonar_data).all(axis=(1, 2))
    finite_mask &= np.isfinite(profiles_data).all(axis=1)
    finite_mask &= np.isfinite(sonar_iid)
    finite_mask &= np.isfinite(sonar_dist)

    return sonar_iid[finite_mask], sonar_dist[finite_mask]


def curvature_from_heatmaps(heatmaps, x_grid, y_grid, cfg, progress_desc='Curvature from occupancy'):
    curv = np.zeros(len(heatmaps), dtype=np.float32)
    left_r = np.full(len(heatmaps), np.nan, dtype=np.float32)
    right_r = np.full(len(heatmaps), np.nan, dtype=np.float32)
    side = []
    for i, hm in enumerate(tqdm(heatmaps, desc=progress_desc)):
        planner = plan_circles_from_heatmap(hm_norm=hm, x_grid=x_grid, y_grid=y_grid, config=cfg)
        curv[i] = float(planner_to_curvature(planner))
        left_r[i] = np.nan if planner['left_radius_mm'] is None else float(planner['left_radius_mm'])
        right_r[i] = np.nan if planner['right_radius_mm'] is None else float(planner['right_radius_mm'])
        side.append(str(planner['chosen_side']))
    return curv, left_r, right_r, side


def plot_scatter_pred_vs_gt(pred, gt, real_left_r, real_right_r, out_path):
    lo = float(min(np.min(pred), np.min(gt)))
    hi = float(max(np.max(pred), np.max(gt)))
    pad = 0.05 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad
    r = safe_corrcoef(pred, gt)
    s = safe_spearman(pred, gt)
    mae = float(np.mean(np.abs(pred - gt)))
    sign_acc = float(np.mean(np.sign(pred) == np.sign(gt)))
    diam_diff = 2.0 * np.abs(np.asarray(real_left_r, dtype=float) - np.asarray(real_right_r, dtype=float))
    finite_c = np.isfinite(diam_diff)

    plt.figure(figsize=(6, 6))
    plt.axhline(0.0, color='gray', lw=1.0, alpha=0.6)
    plt.axvline(0.0, color='gray', lw=1.0, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], '--', color='black', lw=1.0, alpha=0.7)
    if np.any(finite_c):
        sc = plt.scatter(
            gt[finite_c],
            pred[finite_c],
            s=22,
            alpha=0.9,
            c=diam_diff[finite_c],
            cmap='viridis_r',
        )
        cbar = plt.colorbar(sc)
        cbar.set_label('Real occupancy |Δ circle diameter| (mm)')
    if np.any(~finite_c):
        plt.scatter(
            gt[~finite_c],
            pred[~finite_c],
            s=24,
            marker='x',
            color='black',
            alpha=0.9,
            label='non-finite radius diff',
        )
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel('Real-Profile Occupancy Curvature (1/mm)')
    plt.ylabel('Predicted Occupancy Curvature (1/mm)')
    plt.title('Predicted vs Real Occupancy Curvature')
    plt.text(
        0.03, 0.97,
        f'Pearson r = {r:.3f}\nSpearman ρ = {s:.3f}\nMAE = {mae:.5f}\nSign acc = {sign_acc:.3f}',
        transform=plt.gca().transAxes,
        va='top', ha='left',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
    )
    plt.grid(True, alpha=0.25)
    if np.any(~finite_c):
        plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()


def plot_curvature_vs_iid(pred, gt, iid_avg, out_path):
    plt.figure(figsize=(7, 6))
    plt.axhline(0.0, color='gray', lw=1.0, alpha=0.6)
    plt.axvline(0.0, color='gray', lw=1.0, alpha=0.6)
    plt.scatter(iid_avg, pred, s=18, alpha=0.8, label='Pred occupancy curvature')
    plt.scatter(iid_avg, gt, s=18, alpha=0.6, label='Real occupancy curvature')
    plt.xlabel('Window-Avg IID (dB)')
    plt.ylabel('Curvature (1/mm)')
    plt.title('Occupancy Curvature vs Window-Avg IID')
    plt.grid(True, alpha=0.25)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()


def plot_abs_curvature_vs_distance(pred, gt, dist_last, out_path):
    rp = safe_corrcoef(np.abs(pred), dist_last)
    rg = safe_corrcoef(np.abs(gt), dist_last)
    sp = safe_spearman(np.abs(pred), dist_last)
    sg = safe_spearman(np.abs(gt), dist_last)

    plt.figure(figsize=(7, 6))
    plt.scatter(dist_last, np.abs(pred), s=18, alpha=0.8, label='|Pred occupancy curvature|')
    plt.scatter(dist_last, np.abs(gt), s=18, alpha=0.6, label='|Real occupancy curvature|')
    plt.xlabel('Last Sonar Distance in Window')
    plt.ylabel('|Curvature| (1/mm)')
    plt.title('|Occupancy Curvature| vs Sonar Distance')
    plt.text(
        0.03, 0.97,
        f'Pred: Pearson={rp:.3f}, Spearman={sp:.3f}\nReal: Pearson={rg:.3f}, Spearman={sg:.3f}',
        transform=plt.gca().transAxes,
        va='top', ha='left',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
    )
    plt.grid(True, alpha=0.25)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()


def main():
    session_dir = os.path.join(occupancy_output_root, session_to_assess)
    occ_npz = os.path.join(session_dir, f'{session_to_assess}_occupancy_series_data.npz')
    if not os.path.exists(occ_npz):
        raise FileNotFoundError(f'Occupancy package not found: {occ_npz}')

    assess_dir = os.path.join(session_dir, 'assessment')
    ensure_dir(assess_dir)

    d = np.load(occ_npz)
    heatmaps_pred = d['heatmaps_pred'] if 'heatmaps_pred' in d else d['heatmaps']
    heatmaps_real = d['heatmaps_real'] if 'heatmaps_real' in d else d['heatmaps']
    x_grid = d['x_grid']
    y_grid = d['y_grid']
    chunk_indices = d['chunk_indices']
    anchor_idx = d['anchor_index']

    iid_filtered, dist_filtered = load_filtered_iid_distance(session_to_assess)
    if len(iid_filtered) <= int(np.max(anchor_idx)):
        raise ValueError('Filtered IID/distance length does not match occupancy index space.')

    iid_avg_window = np.asarray([np.mean(iid_filtered[idxs]) for idxs in chunk_indices], dtype=np.float32)
    dist_last = dist_filtered[anchor_idx]

    pred_curv, pred_left_r, pred_right_r, pred_side = curvature_from_heatmaps(
        heatmaps_pred, x_grid, y_grid, planner_config, progress_desc='Pred occupancy -> curvature'
    )
    real_curv, real_left_r, real_right_r, real_side = curvature_from_heatmaps(
        heatmaps_real, x_grid, y_grid, planner_config, progress_desc='Real occupancy -> curvature'
    )

    stats = {
        'n_windows': int(len(pred_curv)),
        'pred_vs_real_curvature': {
            'pearson': safe_corrcoef(pred_curv, real_curv),
            'spearman': safe_spearman(pred_curv, real_curv),
            'mae': float(np.mean(np.abs(pred_curv - real_curv))),
            'rmse': float(np.sqrt(np.mean((pred_curv - real_curv) ** 2))),
            'sign_agreement': float(np.mean(np.sign(pred_curv) == np.sign(real_curv))),
        },
        'curvature_vs_iid': {
            'pred_pearson': safe_corrcoef(pred_curv, iid_avg_window),
            'pred_spearman': safe_spearman(pred_curv, iid_avg_window),
            'real_pearson': safe_corrcoef(real_curv, iid_avg_window),
            'real_spearman': safe_spearman(real_curv, iid_avg_window),
        },
        'abs_curvature_vs_distance': {
            'pred_pearson': safe_corrcoef(np.abs(pred_curv), dist_last),
            'pred_spearman': safe_spearman(np.abs(pred_curv), dist_last),
            'real_pearson': safe_corrcoef(np.abs(real_curv), dist_last),
            'real_spearman': safe_spearman(np.abs(real_curv), dist_last),
        },
    }

    # Plots
    p1 = os.path.join(assess_dir, 'curvature_pred_vs_real_scatter.png')
    p2 = os.path.join(assess_dir, 'curvature_vs_iid.png')
    p3 = os.path.join(assess_dir, 'abs_curvature_vs_distance.png')
    plot_scatter_pred_vs_gt(pred_curv, real_curv, real_left_r, real_right_r, p1)
    plot_curvature_vs_iid(pred_curv, real_curv, iid_avg_window, p2)
    plot_abs_curvature_vs_distance(pred_curv, real_curv, dist_last, p3)

    # Save tabular window-level data.
    csv_path = os.path.join(assess_dir, 'occupancy_curvature_windows.csv')
    keys = [
        'series_idx', 'window_start', 'window_end', 'anchor_index',
        'curv_pred', 'curv_real',
        'left_radius_pred', 'right_radius_pred',
        'left_radius_real', 'right_radius_real',
        'side_pred', 'side_real',
        'iid_avg_window', 'sonar_distance_last',
    ]
    with open(csv_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for i in tqdm(range(len(pred_curv)), desc='Writing assessment CSV'):
            row = {
                'series_idx': int(i),
                'window_start': int(d['window_start'][i]),
                'window_end': int(d['window_end'][i]),
                'anchor_index': int(anchor_idx[i]),
                'curv_pred': float(pred_curv[i]),
                'curv_real': float(real_curv[i]),
                'left_radius_pred': float(pred_left_r[i]),
                'right_radius_pred': float(pred_right_r[i]),
                'left_radius_real': float(real_left_r[i]),
                'right_radius_real': float(real_right_r[i]),
                'side_pred': pred_side[i],
                'side_real': real_side[i],
                'iid_avg_window': float(iid_avg_window[i]),
                'sonar_distance_last': float(dist_last[i]),
            }
            f.write(','.join(str(row[k]) for k in keys) + '\n')

    summary = {
        'session': session_to_assess,
        'occupancy_npz': occ_npz,
        'assessment_dir': assess_dir,
        'plots': {
            'pred_vs_real_curvature_scatter': p1,
            'curvature_vs_iid': p2,
            'abs_curvature_vs_distance': p3,
        },
        'window_csv': csv_path,
        'stats': stats,
    }
    summary_path = os.path.join(assess_dir, 'occupancy_assessment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Saved assessment summary: {summary_path}')
    print('Pred vs Real occupancy curvature:')
    print(
        f"  Pearson={stats['pred_vs_real_curvature']['pearson']:.3f}, "
        f"Spearman={stats['pred_vs_real_curvature']['spearman']:.3f}, "
        f"MAE={stats['pred_vs_real_curvature']['mae']:.5f}, "
        f"Sign-agree={stats['pred_vs_real_curvature']['sign_agreement']:.3f}"
    )
    print('Curvature vs IID:')
    print(
        f"  Pred: Pearson={stats['curvature_vs_iid']['pred_pearson']:.3f}, "
        f"Spearman={stats['curvature_vs_iid']['pred_spearman']:.3f} | "
        f"Real: Pearson={stats['curvature_vs_iid']['real_pearson']:.3f}, "
        f"Spearman={stats['curvature_vs_iid']['real_spearman']:.3f}"
    )
    print('|Curvature| vs Sonar distance:')
    print(
        f"  Pred: Pearson={stats['abs_curvature_vs_distance']['pred_pearson']:.3f}, "
        f"Spearman={stats['abs_curvature_vs_distance']['pred_spearman']:.3f} | "
        f"Real: Pearson={stats['abs_curvature_vs_distance']['real_pearson']:.3f}, "
        f"Spearman={stats['abs_curvature_vs_distance']['real_spearman']:.3f}"
    )


if __name__ == '__main__':
    main()
