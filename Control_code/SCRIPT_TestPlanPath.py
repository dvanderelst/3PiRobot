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
selected_series_range = (0, 500, 5)  # (start, stop, step)

# Optional quick benchmark.
run_benchmark = True
benchmark_runs = 80
benchmark_warmup = 10

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


def load_session_filtered(session_name):
    dc = DataProcessor.DataCollection([session_name])
    sonar = dc.load_sonar(flatten=False)
    sonar_iid = np.asarray(dc.get_field('sonar_package', 'corrected_iid'), dtype=np.float32)
    sonar_distance = np.asarray(dc.get_field('sonar_package', 'corrected_distance'), dtype=np.float32)

    finite_mask = np.isfinite(sonar).all(axis=(1, 2))
    finite_mask &= np.isfinite(dc.rob_x)
    finite_mask &= np.isfinite(dc.rob_y)
    finite_mask &= np.isfinite(dc.rob_yaw_deg)
    finite_mask &= np.isfinite(sonar_iid)
    finite_mask &= np.isfinite(sonar_distance)

    sonar = sonar[finite_mask]
    rob_x = dc.rob_x[finite_mask]
    rob_y = dc.rob_y[finite_mask]
    rob_yaw = dc.rob_yaw_deg[finite_mask]
    sonar_iid = sonar_iid[finite_mask]
    sonar_distance = sonar_distance[finite_mask]
    kept_indices = np.where(finite_mask)[0]

    return {
        'sonar': sonar,
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
    plt.figure(figsize=(6, 6))
    plt.scatter(dist_last, np.abs(kappa), s=22, alpha=0.85)
    plt.xlabel('Last Sonar Distance in Window')
    plt.ylabel('|Signed Curvature| (1/mm)')
    plt.title('|Curvature| vs Last Sonar Distance')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    p2 = os.path.join(out_dir, 'comparison_abs_curvature_vs_last_distance.png')
    plt.savefig(p2, dpi=220, bbox_inches='tight')
    plt.close()
    saved.append(p2)

    return saved


def plot_example(out_path, title, debug):
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

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(
        hm,
        origin='lower',
        extent=[float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1])],
        cmap='magma',
        aspect='equal'
    )
    fig.colorbar(im, ax=ax, label='Wall Evidence (norm)')

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
    ax.legend(loc='upper right')
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
    print(f'Selected series indices: {selected}')

    session_data = load_session_filtered(session_name)
    sonar = session_data['sonar']
    rob_x = session_data['rob_x']
    rob_y = session_data['rob_y']
    rob_yaw = session_data['rob_yaw_deg']
    sonar_iid = session_data['sonar_iid']
    sonar_distance = session_data['sonar_distance']

    steering = SonarWallSteering(training_output_dir=training_output_dir, config=steering_config)

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

    rows = []
    for sidx in selected:
        start = int(window_start[sidx])
        end = int(window_end[sidx])
        if start < 0 or end >= len(sonar) or start > end:
            print(f'series_idx={sidx}: skipped invalid window [{start},{end}]')
            continue

        kappa, dbg = steering.compute_curvature(
            sonar_seq=sonar[start:end + 1],
            rob_x_seq=rob_x[start:end + 1],
            rob_y_seq=rob_y[start:end + 1],
            rob_yaw_deg_seq=rob_yaw[start:end + 1],
            return_debug=True,
        )

        out_png = os.path.join(output_dir, f'seriesidx_{sidx:04d}_pathplan.png')
        title = f'series_idx={sidx} | chunk=[{start},{end}] | session={session_name}'
        plot_example(out_png, title, dbg)

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
            'iid_avg_window': float(np.mean(sonar_iid[start:end + 1])),
            'iid_last': float(sonar_iid[end]),
            'sonar_distance_last': float(sonar_distance[end]),
            'plot_file': out_png,
        }
        rows.append(row)

        print(
            f"series_idx={sidx:4d} | side={row['chosen_side']:>12s} | "
            f"R={row['chosen_radius_mm']:.0f} | kappa={row['signed_curvature_inv_mm']:+.5f}"
        )

    comparison = None
    comparison_plots = []
    if len(rows) > 0:
        kappa = np.asarray([r['signed_curvature_inv_mm'] for r in rows], dtype=float)
        iid_avg = np.asarray([r['iid_avg_window'] for r in rows], dtype=float)
        dist_last = np.asarray([r['sonar_distance_last'] for r in rows], dtype=float)

        sign_kappa = np.sign(kappa)
        sign_iid = np.sign(iid_avg)
        comparable = sign_kappa != 0
        sign_agreement = float(np.mean(sign_kappa[comparable] == sign_iid[comparable])) if np.any(comparable) else np.nan

        comparison = {
            'n_samples': int(len(rows)),
            'sign_agreement_fraction': sign_agreement,
            'corr_curvature_vs_iid_avg': safe_corrcoef(kappa, iid_avg),
            'corr_abs_curvature_vs_last_distance': safe_corrcoef(np.abs(kappa), dist_last),
            'corr_curvature_vs_last_distance': safe_corrcoef(kappa, dist_last),
        }
        comparison_plots = make_iid_distance_comparison_plots(rows, output_dir)
        print(
            "Comparison stats: "
            f"sign_agreement={comparison['sign_agreement_fraction']:.3f}, "
            f"corr(kappa,iid_avg)={comparison['corr_curvature_vs_iid_avg']:.3f}, "
            f"corr(|kappa|,dist_last)={comparison['corr_abs_curvature_vs_last_distance']:.3f}"
        )

    summary = {
        'source_npz': npz_path,
        'session': session_name,
        'training_output_dir': training_output_dir,
        'n_series_total': int(n_series),
        'selected_series_indices': [int(i) for i in selected],
        'window_size': int(steering_config.window_size),
        'benchmark': benchmark_result,
        'iid_distance_comparison': comparison,
        'iid_distance_comparison_plots': comparison_plots,
        'results': rows,
    }

    summary_json = os.path.join(output_dir, 'path_planning_examples_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    summary_csv = os.path.join(output_dir, 'path_planning_examples_summary.csv')
    if rows:
        keys = [
            'series_idx', 'window_start', 'window_end', 'anchor_index',
            'anchor_rob_x', 'anchor_rob_y', 'anchor_rob_yaw_deg',
            'chosen_side', 'chosen_radius_mm', 'left_radius_mm', 'right_radius_mm',
            'left_evidence_score', 'right_evidence_score',
            'signed_curvature_inv_mm', 'iid_avg_window', 'iid_last', 'sonar_distance_last',
            'plot_file'
        ]
        with open(summary_csv, 'w') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r[k]) for k in keys) + '\n')

    print(f'Saved summary JSON: {summary_json}')
    print(f'Saved summary CSV:  {summary_csv}')
    print(f'Saved plots to:      {output_dir}')


if __name__ == '__main__':
    main()
