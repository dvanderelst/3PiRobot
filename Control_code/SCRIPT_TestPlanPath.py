import json
import os
import re
from glob import glob
import time

import numpy as np
from matplotlib import pyplot as plt

from Library import DataProcessor
from Library.SonarWallSteering import SonarWallSteering, SteeringConfig


# ============================================
# CONFIGURATION
# ============================================
# If None, use newest Prediction/heatmap_series_robot/*_robotframe_series_data.npz
series_npz_path = None
training_output_dir = 'Training'
output_dir = 'Prediction/PathPlanningExamples'

# Select frames by series index from the robotframe series file.
selected_series_indices = None
selected_series_range = (0, 500, 10)  # (start, stop, step)

# Optional quick benchmark.
run_benchmark = True
benchmark_runs = 80
benchmark_warmup = 10

# Ground-truth curvature settings.
# If None, use distance_threshold from Training/training_params.json.
gt_distance_cutoff_mm = None
# Compute GT-window comparison metrics only on samples with
# GT |Δ circle diameter| >= this cutoff (mm). Set to None to disable.
gtwindow_metrics_min_diam_diff_mm = 300.0

# Steering module configuration.
steering_config = SteeringConfig(
    window_size=5,
    presence_threshold_override=None,
    extent_mm=2500.0,
    grid_mm=20.0,
    sigma_perp_mm=40.0,
    sigma_para_mm=120.0,
    apply_heatmap_smoothing=False,
    occ_block_threshold=0.10,
    robot_radius_mm=80.0,
    safety_margin_mm=120.0,
    circle_radius_min_mm=250.0,
    circle_radius_max_mm=2500.0,
    circle_radius_step_mm=50.0,
    circle_arc_samples=220,
    circle_horizon_x_mm=1800.0,
    circle_radius_tie_mm=100.0,
)


# ============================================
# HELPERS
# ============================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_default_series_file():
    candidates = glob('Prediction/heatmap_series_robot/*_robotframe_series_data.npz')
    if not candidates:
        raise FileNotFoundError(
            "No robot-frame series data found. Run SCRIPT_PredictProfiles.py first."
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def infer_session_from_series_path(path):
    base = os.path.basename(path)
    m = re.match(r'(session[^_]+)_robotframe_series_data\.npz$', base)
    if not m:
        raise ValueError(f"Could not infer session from filename '{base}'")
    return m.group(1)


def resolve_indices(n_samples, idx_list, idx_range):
    selected = []
    if idx_list is not None:
        selected.extend(idx_list)
    if idx_range is not None:
        if len(idx_range) != 3:
            raise ValueError('selected_series_range must be (start, stop, step).')
        start, stop, step = idx_range
        selected.extend(list(range(int(start), int(stop), int(step))))

    if not selected:
        selected = list(np.linspace(0, n_samples - 1, num=min(6, n_samples), dtype=int))

    return sorted(set(int(i) for i in selected if 0 <= int(i) < n_samples))


def load_session_filtered(
    session_name,
    profile_opening_angle_deg,
    profile_steps,
    curvature_distance_cutoff_mm,
    curvature_config,
):
    dc = DataProcessor.DataCollection([session_name])
    sonar = dc.load_sonar(flatten=False)
    dc.load_profiles(opening_angle=float(profile_opening_angle_deg), steps=int(profile_steps), fill_nans=True)
    true_curvature = dc.load_curvatures(
        distance_cutoff_mm=float(curvature_distance_cutoff_mm),
        steering_config=curvature_config,
        show_progress=True,
        parallel=True,
    )
    sonar_iid = np.asarray(dc.get_field('sonar_package', 'corrected_iid'), dtype=np.float32)
    sonar_distance = np.asarray(dc.get_field('sonar_package', 'corrected_distance'), dtype=np.float32)

    finite_mask = np.isfinite(sonar).all(axis=(1, 2))
    finite_mask &= np.isfinite(dc.rob_x)
    finite_mask &= np.isfinite(dc.rob_y)
    finite_mask &= np.isfinite(dc.rob_yaw_deg)
    finite_mask &= np.isfinite(sonar_iid)
    finite_mask &= np.isfinite(sonar_distance)
    finite_mask &= np.isfinite(true_curvature)

    sonar = sonar[finite_mask]
    rob_x = dc.rob_x[finite_mask]
    rob_y = dc.rob_y[finite_mask]
    rob_yaw = dc.rob_yaw_deg[finite_mask]
    sonar_iid = sonar_iid[finite_mask]
    sonar_distance = sonar_distance[finite_mask]
    true_curvature = true_curvature[finite_mask]
    kept_indices = np.where(finite_mask)[0]

    return {
        'sonar': sonar,
        'profiles': dc.profiles[finite_mask],
        'profile_centers_deg': dc.profile_centers[finite_mask],
        'rob_x': rob_x,
        'rob_y': rob_y,
        'rob_yaw_deg': rob_yaw,
        'sonar_iid': sonar_iid,
        'sonar_distance': sonar_distance,
        'kept_indices': kept_indices,
        'n_total': len(finite_mask),
        'n_used': int(np.sum(finite_mask)),
    }


def summarize_times_ms(samples):
    vals = np.asarray(samples, dtype=np.float64)
    if vals.size == 0:
        return {'mean': np.nan, 'median': np.nan, 'p95': np.nan, 'min': np.nan, 'max': np.nan}
    return {
        'mean': float(np.mean(vals)),
        'median': float(np.median(vals)),
        'p95': float(np.percentile(vals, 95)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
    }


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

    # Rank transform (simple stable ranks; adequate here).
    rx = np.empty_like(x, dtype=float)
    ry = np.empty_like(y, dtype=float)
    rx[np.argsort(x, kind='mergesort')] = np.arange(len(x), dtype=float)
    ry[np.argsort(y, kind='mergesort')] = np.arange(len(y), dtype=float)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def make_iid_distance_comparison_plots(rows, out_dir):
    if len(rows) == 0:
        return []

    kappa = np.asarray([r['signed_curvature_inv_mm'] for r in rows], dtype=float)
    iid_avg = np.asarray([r['iid_avg_window'] for r in rows], dtype=float)
    dist_last = np.asarray([r['sonar_distance_last'] for r in rows], dtype=float)

    saved = []

    # Sign agreement diagnostic: kappa vs avg IID.
    plt.figure(figsize=(6, 6))
    plt.axhline(0.0, color='gray', linewidth=1.0, alpha=0.6)
    plt.axvline(0.0, color='gray', linewidth=1.0, alpha=0.6)
    plt.scatter(iid_avg, kappa, s=22, alpha=0.85)
    plt.xlabel('Average IID over window (dB)')
    plt.ylabel('Signed Curvature (1/mm)')
    plt.title('Curvature vs Window-Averaged IID')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    p1 = os.path.join(out_dir, 'comparison_curvature_vs_iid_avg.png')
    plt.savefig(p1, dpi=220, bbox_inches='tight')
    plt.close()
    saved.append(p1)

    # Magnitude relation: |kappa| vs last sonar distance.
    pearson_abs = safe_corrcoef(np.abs(kappa), dist_last)
    spearman_abs = safe_spearman(np.abs(kappa), dist_last)
    plt.figure(figsize=(6, 6))
    plt.scatter(dist_last, np.abs(kappa), s=22, alpha=0.85)
    plt.xlabel('Last Sonar Distance in Window')
    plt.ylabel('|Signed Curvature| (1/mm)')
    plt.title('|Curvature| vs Last Sonar Distance')
    plt.text(
        0.03,
        0.97,
        f'Pearson r = {pearson_abs:.3f}\\nSpearman ρ = {spearman_abs:.3f}',
        transform=plt.gca().transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none')
    )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    p2 = os.path.join(out_dir, 'comparison_abs_curvature_vs_last_distance.png')
    plt.savefig(p2, dpi=220, bbox_inches='tight')
    plt.close()
    saved.append(p2)

    return saved


def build_groundtruth_debug_for_window(
    steering,
    profile_dist_seq,
    profile_centers_seq,
    rob_x_seq,
    rob_y_seq,
    rob_yaw_deg_seq,
    distance_cutoff_mm,
):
    profile_dist_seq = np.asarray(profile_dist_seq, dtype=np.float32)
    profile_centers_seq = np.asarray(profile_centers_seq, dtype=np.float32)
    if profile_dist_seq.ndim != 2:
        raise ValueError('profile_dist_seq must be shape (N, profile_steps).')
    if profile_centers_seq.ndim != 2:
        raise ValueError('profile_centers_seq must be shape (N, profile_steps).')
    if profile_dist_seq.shape != profile_centers_seq.shape:
        raise ValueError('profile_dist_seq and profile_centers_seq shapes must match.')
    if profile_dist_seq.shape[1] != len(steering.profile_centers_deg):
        raise ValueError(
            f"Ground-truth profile steps ({profile_dist_seq.shape[1]}) do not match "
            f"steering model profile steps ({len(steering.profile_centers_deg)})."
        )

    # Ensure GT azimuth bins are aligned to steering bins.
    az_ref = np.asarray(steering.profile_centers_deg, dtype=np.float32)[None, :]
    az_diff = np.nanmax(np.abs(profile_centers_seq - az_ref))
    if not np.isfinite(az_diff) or az_diff > 1e-3:
        raise ValueError(f'Ground-truth profile centers do not match steering centers (max diff={az_diff}).')

    dist = profile_dist_seq.copy()
    dist[(dist < 0.0) | (dist > float(distance_cutoff_mm))] = np.nan
    presence_bin = np.isfinite(dist).astype(np.uint8)
    presence_probs = presence_bin.astype(np.float32)

    hm = steering._build_robot_frame_evidence(
        distance_mm=dist,
        presence_probs=presence_probs,
        presence_bin=presence_bin,
        rob_x_seq=np.asarray(rob_x_seq, dtype=np.float32),
        rob_y_seq=np.asarray(rob_y_seq, dtype=np.float32),
        rob_yaw_deg_seq=np.asarray(rob_yaw_deg_seq, dtype=np.float32),
    )
    planner = steering._plan_circles(hm)
    kappa = steering._planner_to_curvature(planner)

    return {
        'signed_curvature_inv_mm': float(kappa),
        'chosen_side': planner['chosen_side'],
        'chosen_radius_mm': float(planner['chosen_radius_mm']),
        'left_radius_mm': planner['left_radius_mm'],
        'right_radius_mm': planner['right_radius_mm'],
        'left_evidence_score': planner['left_evidence_score'],
        'right_evidence_score': planner['right_evidence_score'],
        'left_x': planner['left_x'],
        'left_y': planner['left_y'],
        'right_x': planner['right_x'],
        'right_y': planner['right_y'],
        'blocked_mask': planner['blocked_mask'],
        'evidence_map': hm,
        'x_grid': steering.x_grid,
        'y_grid': steering.y_grid,
    }


def _plot_debug_panel(ax, fig, debug, title):
    hm = debug['evidence_map']
    x_grid = debug['x_grid']
    y_grid = debug['y_grid']
    blocked = debug.get('blocked_mask', hm >= steering_config.occ_block_threshold)

    left_x = debug.get('left_x', None)
    left_y = debug.get('left_y', None)
    right_x = debug.get('right_x', None)
    right_y = debug.get('right_y', None)

    chosen_side = str(debug.get('chosen_side', 'straight'))
    if chosen_side.startswith('left'):
        path_x, path_y = left_x, left_y
    elif chosen_side.startswith('right'):
        path_x, path_y = right_x, right_y
    else:
        path_x = np.linspace(0.0, steering_config.circle_horizon_x_mm, steering_config.circle_arc_samples, dtype=np.float32)
        path_y = np.zeros_like(path_x)

    im = ax.imshow(
        hm,
        origin='lower',
        extent=[float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1])],
        cmap='magma',
        aspect='equal',
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Wall Evidence (norm)')

    ax.scatter([0.0], [0.0], c='cyan', s=45, label='Robot (anchor)')
    ax.arrow(0.0, 0.0, 200.0, 0.0, head_width=70, head_length=90, fc='cyan', ec='cyan')

    xx, yy = np.meshgrid(x_grid, y_grid)
    ax.contourf(xx, yy, blocked.astype(float), levels=[0.5, 1.5], colors=['red'], alpha=0.16)

    if left_x is not None:
        ax.plot(left_x, left_y, '--', color='deepskyblue', lw=1.4, alpha=0.8, label='Left candidate')
    if right_x is not None:
        ax.plot(right_x, right_y, '--', color='springgreen', lw=1.4, alpha=0.8, label='Right candidate')
    if path_x is not None and len(path_x) > 1:
        ax.plot(path_x, path_y, '-', color='white', lw=2.0, label='Chosen circle arc')

    txt = (
        f"side={debug['chosen_side']}\n"
        f"R={debug['chosen_radius_mm']:.0f} mm\n"
        f"curv={debug['signed_curvature_inv_mm']:+.5f} 1/mm\n"
        f"E(L,R)=({debug['left_evidence_score']:.4f},{debug['right_evidence_score']:.4f})"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=10, bbox=dict(facecolor='black', alpha=0.45, edgecolor='none'))

    ax.set_xlim(0.0, steering_config.circle_horizon_x_mm + 200.0)
    ax.set_ylim(float(y_grid[0]), float(y_grid[-1]))
    ax.set_xlabel('Robot X (mm)')
    ax.set_ylabel('Robot Y (mm)')
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=9)


def make_curvature_gtwindow_comparison_plots(rows, out_dir, min_diam_diff_mm=None):
    if len(rows) == 0:
        return [], None

    est = np.asarray([r['signed_curvature_inv_mm'] for r in rows], dtype=float)
    gtw = np.asarray([r['signed_curvature_gt_window_inv_mm'] for r in rows], dtype=float)
    gt_left = np.asarray([r['gt_left_radius_mm'] for r in rows], dtype=float)
    gt_right = np.asarray([r['gt_right_radius_mm'] for r in rows], dtype=float)
    gt_diam_diff = 2.0 * np.abs(gt_left - gt_right)
    idxs = np.asarray([r['series_idx'] for r in rows], dtype=int)
    err = est - gtw

    finite_core = np.isfinite(est) & np.isfinite(gtw) & np.isfinite(gt_diam_diff)
    if min_diam_diff_mm is not None:
        include = finite_core & (gt_diam_diff >= float(min_diam_diff_mm))
    else:
        include = finite_core

    if np.sum(include) >= 2:
        pearson = safe_corrcoef(est[include], gtw[include])
        spearman = safe_spearman(est[include], gtw[include])
        mae = float(np.mean(np.abs(err[include])))
        rmse = float(np.sqrt(np.mean(err[include] ** 2)))
        sign_acc = float(np.mean(np.sign(est[include]) == np.sign(gtw[include])))
    else:
        pearson = np.nan
        spearman = np.nan
        mae = np.nan
        rmse = np.nan
        sign_acc = np.nan

    stats = {
        'n_samples': int(len(rows)),
        'n_included_for_metrics': int(np.sum(include)),
        'min_gt_diam_diff_mm_for_metrics': None if min_diam_diff_mm is None else float(min_diam_diff_mm),
        'corr_est_vs_gtwindow': pearson,
        'spearman_est_vs_gtwindow': spearman,
        'mae': mae,
        'rmse': rmse,
        'sign_agreement_fraction': sign_acc,
    }

    saved = []
    lo = float(min(np.min(est), np.min(gtw)))
    hi = float(max(np.max(est), np.max(gtw)))
    pad = 0.05 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad

    plt.figure(figsize=(6, 6))
    plt.axhline(0.0, color='gray', linewidth=1.0, alpha=0.6)
    plt.axvline(0.0, color='gray', linewidth=1.0, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], '--', color='white', lw=1.2, alpha=0.8, label='y = x')
    finite_c = np.isfinite(gt_diam_diff)
    if np.any(finite_c):
        sc = plt.scatter(
            gtw[finite_c],
            est[finite_c],
            s=26,
            alpha=0.9,
            c=gt_diam_diff[finite_c],
            cmap='viridis_r',
        )
        cbar = plt.colorbar(sc)
        cbar.set_label('GT |Δ circle diameter| (mm)')
    if np.any(~finite_c):
        plt.scatter(
            gtw[~finite_c],
            est[~finite_c],
            s=28,
            alpha=0.9,
            marker='x',
            color='black',
            label='non-finite GT radius diff',
        )
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel('GT-Window Curvature (1/mm)')
    plt.ylabel('Estimated Curvature (1/mm)')
    plt.title('Estimated vs GT-Window Curvature')
    plt.text(
        0.03,
        0.97,
        (
            f'Pearson r = {pearson:.3f}\n'
            f'Spearman ρ = {spearman:.3f}\n'
            f'MAE = {mae:.5f}\n'
            f'RMSE = {rmse:.5f}\n'
            f'Sign acc = {sign_acc:.3f}\n'
            f'n incl = {int(np.sum(include))}/{len(rows)}\n'
            f'cutoff = {min_diam_diff_mm if min_diam_diff_mm is not None else "None"} mm'
        ),
        transform=plt.gca().transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none')
    )
    plt.grid(True, alpha=0.25)
    if np.any(~finite_c):
        plt.legend(loc='lower right')
    plt.tight_layout()
    p1 = os.path.join(out_dir, 'comparison_estimated_vs_gtwindow_curvature_scatter.png')
    plt.savefig(p1, dpi=220, bbox_inches='tight')
    plt.close()
    saved.append(p1)

    order = np.argsort(idxs)
    plt.figure(figsize=(10, 4))
    plt.plot(idxs[order], gtw[order], '-o', ms=3, lw=1.2, alpha=0.9, label='GT-window curvature')
    plt.plot(idxs[order], est[order], '-o', ms=3, lw=1.2, alpha=0.9, label='Estimated curvature')
    plt.xlabel('Series Index')
    plt.ylabel('Curvature (1/mm)')
    plt.title('Estimated vs GT-Window Curvature Across Selected Indices')
    plt.axhline(0.0, color='gray', linewidth=1.0, alpha=0.6)
    plt.grid(True, alpha=0.25)
    plt.legend(loc='best')
    plt.tight_layout()
    p2 = os.path.join(out_dir, 'comparison_estimated_vs_gtwindow_curvature_series.png')
    plt.savefig(p2, dpi=220, bbox_inches='tight')
    plt.close()
    saved.append(p2)

    return saved, stats


def plot_example(
    out_path,
    title,
    debug_est,
    debug_gt,
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _plot_debug_panel(axes[0], fig, debug_est, title='Estimated Profiles -> Walliness')
    _plot_debug_panel(axes[1], fig, debug_gt, title='Ground-Truth Profiles -> Walliness')
    fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


# ============================================
# MAIN
# ============================================
def main():
    ensure_dir(output_dir)

    npz_path = series_npz_path if series_npz_path else find_default_series_file()
    series = np.load(npz_path)
    session_name = infer_session_from_series_path(npz_path)

    window_start = series['window_start']
    window_end = series['window_end']
    n_series = len(window_start)
    selected = resolve_indices(n_series, selected_series_indices, selected_series_range)

    print(f'Loaded series: {npz_path}')
    print(f'Session inferred: {session_name}')
    print(f'Selected series indices for example plots: {selected}')

    training_params_path = os.path.join(training_output_dir, 'training_params.json')
    with open(training_params_path, 'r') as f:
        training_params = json.load(f)

    profile_opening_angle_deg = float(training_params['profile_opening_angle'])
    profile_steps = int(training_params['profile_steps'])
    gt_cutoff = (
        float(training_params.get('distance_threshold', 1500.0))
        if gt_distance_cutoff_mm is None else float(gt_distance_cutoff_mm)
    )

    steering = SonarWallSteering(training_output_dir=training_output_dir, config=steering_config)

    session_data = load_session_filtered(
        session_name=session_name,
        profile_opening_angle_deg=profile_opening_angle_deg,
        profile_steps=profile_steps,
        curvature_distance_cutoff_mm=gt_cutoff,
        curvature_config=steering_config,
    )
    sonar = session_data['sonar']
    profiles = session_data['profiles']
    profile_centers_deg = session_data['profile_centers_deg']
    rob_x = session_data['rob_x']
    rob_y = session_data['rob_y']
    rob_yaw = session_data['rob_yaw_deg']
    sonar_iid = session_data['sonar_iid']
    sonar_distance = session_data['sonar_distance']

    benchmark_result = None
    if run_benchmark:
        n = int(steering_config.window_size)
        max_start = len(sonar) - n
        if max_start > 0:
            t_samples = []
            total_runs = int(benchmark_runs)
            warmup = int(benchmark_warmup)
            for i in range(total_runs):
                start_i = i % max_start
                t0 = time.perf_counter()
                _ = steering.compute_curvature(
                    sonar_seq=sonar[start_i:start_i + n],
                    rob_x_seq=rob_x[start_i:start_i + n],
                    rob_y_seq=rob_y[start_i:start_i + n],
                    rob_yaw_deg_seq=rob_yaw[start_i:start_i + n],
                    return_debug=False,
                )
                t1 = time.perf_counter()
                t_samples.append((t1 - t0) * 1000.0)

            t_eval = t_samples[warmup:] if len(t_samples) > warmup else t_samples
            stats = summarize_times_ms(t_eval)
            benchmark_result = {
                'runs': total_runs,
                'warmup': warmup,
                'window_size': n,
                'timing_ms': stats,
            }
            print(
                f"Benchmark compute_curvature (window={n}, runs={total_runs}, warmup={warmup}): "
                f"mean={stats['mean']:.2f} ms, median={stats['median']:.2f} ms, p95={stats['p95']:.2f} ms"
            )
        else:
            print("Benchmark skipped: not enough sonar samples for selected window size.")

    selected_set = set(int(v) for v in selected)
    rows_all = []
    rows_examples = []
    n_valid_windows = 0
    for sidx in range(n_series):
        start = int(window_start[sidx])
        end = int(window_end[sidx])
        if start < 0 or end >= len(sonar) or start > end:
            continue
        n_valid_windows += 1

        kappa, dbg = steering.compute_curvature(
            sonar_seq=sonar[start:end + 1],
            rob_x_seq=rob_x[start:end + 1],
            rob_y_seq=rob_y[start:end + 1],
            rob_yaw_deg_seq=rob_yaw[start:end + 1],
            return_debug=True,
        )

        gt_dbg = build_groundtruth_debug_for_window(
            steering=steering,
            profile_dist_seq=profiles[start:end + 1],
            profile_centers_seq=profile_centers_deg[start:end + 1],
            rob_x_seq=rob_x[start:end + 1],
            rob_y_seq=rob_y[start:end + 1],
            rob_yaw_deg_seq=rob_yaw[start:end + 1],
            distance_cutoff_mm=gt_cutoff,
        )

        out_png = ''
        if sidx in selected_set:
            out_png = os.path.join(output_dir, f'seriesidx_{sidx:04d}_pathplan.png')
            title = f'series_idx={sidx} | chunk=[{start},{end}] | session={session_name}'
            plot_example(
                out_path=out_png,
                title=title,
                debug_est=dbg,
                debug_gt=gt_dbg,
            )

        row = {
            'series_idx': int(sidx),
            'window_start': start,
            'window_end': end,
            'anchor_index': end,
            'anchor_rob_x': float(rob_x[end]),
            'anchor_rob_y': float(rob_y[end]),
            'anchor_rob_yaw_deg': float(rob_yaw[end]),
            'chosen_side': str(dbg['chosen_side']),
            'chosen_radius_mm': float(dbg['chosen_radius_mm']),
            'left_radius_mm': float(dbg['left_radius_mm']) if dbg['left_radius_mm'] is not None else float('nan'),
            'right_radius_mm': float(dbg['right_radius_mm']) if dbg['right_radius_mm'] is not None else float('nan'),
            'left_evidence_score': float(dbg['left_evidence_score']),
            'right_evidence_score': float(dbg['right_evidence_score']),
            'signed_curvature_inv_mm': float(kappa),
            'signed_curvature_gt_window_inv_mm': float(gt_dbg['signed_curvature_inv_mm']),
            'gt_left_radius_mm': float(gt_dbg['left_radius_mm']) if gt_dbg['left_radius_mm'] is not None else float('nan'),
            'gt_right_radius_mm': float(gt_dbg['right_radius_mm']) if gt_dbg['right_radius_mm'] is not None else float('nan'),
            'iid_avg_window': float(np.mean(sonar_iid[start:end + 1])),
            'iid_last': float(sonar_iid[end]),
            'sonar_distance_last': float(sonar_distance[end]),
            'plot_file': out_png,
        }
        rows_all.append(row)
        if sidx in selected_set:
            rows_examples.append(row)

            print(
                f"example series_idx={sidx:4d} | side={row['chosen_side']:>12s} | "
                f"R={row['chosen_radius_mm']:.0f} | "
                f"kappa={row['signed_curvature_inv_mm']:+.5f} | "
                f"kappa_gt_window={row['signed_curvature_gt_window_inv_mm']:+.5f}"
            )

        if (n_valid_windows % 100) == 0:
            print(f'Processed {n_valid_windows}/{n_series} valid windows for metrics...')

    comparison = None
    comparison_plots = []
    gtwindow_comparison = None
    gtwindow_comparison_plots = []
    if len(rows_all) > 0:
        kappa = np.asarray([r['signed_curvature_inv_mm'] for r in rows_all], dtype=float)
        iid_avg = np.asarray([r['iid_avg_window'] for r in rows_all], dtype=float)
        dist_last = np.asarray([r['sonar_distance_last'] for r in rows_all], dtype=float)

        sign_kappa = np.sign(kappa)
        sign_iid = np.sign(iid_avg)
        comparable = sign_kappa != 0
        sign_agreement = float(np.mean(sign_kappa[comparable] == sign_iid[comparable])) if np.any(comparable) else np.nan

        comparison = {
            'n_samples': int(len(rows_all)),
            'sign_agreement_fraction': sign_agreement,
            'corr_curvature_vs_iid_avg': safe_corrcoef(kappa, iid_avg),
            'corr_abs_curvature_vs_last_distance': safe_corrcoef(np.abs(kappa), dist_last),
            'spearman_abs_curvature_vs_last_distance': safe_spearman(np.abs(kappa), dist_last),
            'corr_curvature_vs_last_distance': safe_corrcoef(kappa, dist_last),
        }
        comparison_plots = make_iid_distance_comparison_plots(rows_all, output_dir)
        print(
            "Comparison A (Estimated walliness curvature vs sonar summary): "
            f"turn-direction agreement with avg IID sign={comparison['sign_agreement_fraction']:.3f}, "
            f"corr(estimated signed curvature, avg IID)={comparison['corr_curvature_vs_iid_avg']:.3f}, "
            f"corr(|estimated curvature|, last sonar distance)={comparison['corr_abs_curvature_vs_last_distance']:.3f}"
        )
        gtwindow_comparison_plots, gtwindow_comparison = make_curvature_gtwindow_comparison_plots(
            rows_all,
            output_dir,
            min_diam_diff_mm=gtwindow_metrics_min_diam_diff_mm,
        )
        if gtwindow_comparison is not None:
            print(
                "Comparison B (Estimated walliness curvature vs GT-window curvature from real profiles, same planner/window): "
                f"Pearson corr={gtwindow_comparison['corr_est_vs_gtwindow']:.3f}, "
                f"Spearman={gtwindow_comparison['spearman_est_vs_gtwindow']:.3f}, "
                f"MAE={gtwindow_comparison['mae']:.5f}, "
                f"turn-direction agreement={gtwindow_comparison['sign_agreement_fraction']:.3f}, "
                f"included={gtwindow_comparison['n_included_for_metrics']}/{gtwindow_comparison['n_samples']} "
                f"(cutoff={gtwindow_comparison['min_gt_diam_diff_mm_for_metrics']} mm)"
            )

    summary = {
        'source_npz': npz_path,
        'session': session_name,
        'training_output_dir': training_output_dir,
        'groundtruth_distance_cutoff_mm': float(gt_cutoff),
        'gtwindow_metrics_min_diam_diff_mm': (
            None if gtwindow_metrics_min_diam_diff_mm is None else float(gtwindow_metrics_min_diam_diff_mm)
        ),
        'n_series_total': int(n_series),
        'n_valid_windows_for_metrics': int(len(rows_all)),
        'selected_series_indices': [int(i) for i in selected],
        'n_example_plots': int(len(rows_examples)),
        'window_size': int(steering_config.window_size),
        'benchmark': benchmark_result,
        'iid_distance_comparison': comparison,
        'iid_distance_comparison_plots': comparison_plots,
        'groundtruth_window_curvature_comparison': gtwindow_comparison,
        'groundtruth_window_curvature_comparison_plots': gtwindow_comparison_plots,
        'results_all_windows': rows_all,
        'results_example_windows': rows_examples,
    }

    summary_json = os.path.join(output_dir, 'path_planning_examples_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    summary_csv = os.path.join(output_dir, 'path_planning_examples_summary.csv')
    if rows_all:
        keys = [
            'series_idx', 'window_start', 'window_end', 'anchor_index',
            'anchor_rob_x', 'anchor_rob_y', 'anchor_rob_yaw_deg',
            'chosen_side', 'chosen_radius_mm', 'left_radius_mm', 'right_radius_mm',
            'left_evidence_score', 'right_evidence_score',
            'signed_curvature_inv_mm', 'signed_curvature_gt_window_inv_mm',
            'gt_left_radius_mm', 'gt_right_radius_mm',
            'iid_avg_window', 'iid_last', 'sonar_distance_last',
            'plot_file'
        ]
        with open(summary_csv, 'w') as f:
            f.write(','.join(keys) + '\n')
            for r in rows_all:
                f.write(','.join(str(r[k]) for k in keys) + '\n')

    print(f'Saved summary JSON: {summary_json}')
    print(f'Saved summary CSV:  {summary_csv}')
    print(f'Saved plots to:      {output_dir}')


if __name__ == '__main__':
    main()
