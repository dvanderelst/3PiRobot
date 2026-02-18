"""
SCRIPT_ComputeOccupancy

Compute robot-frame occupancy evidence maps for sliding windows in one session.
This script is split into two phases:
1) Compute + save occupancy tensors (parallelized)
2) Optional plotting from saved outputs (serial, matplotlib-safe)
"""

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch

from Library import DataProcessor, Settings
from Library.OccupancyCalculation import build_robot_frame_evidence, make_robot_frame_grid
from Library.ProfileInference import load_training_artifacts, load_session_data, predict_profiles


# ============================================
# CONFIGURATION
# ============================================
session_to_compute = 'sessionB05'
training_output_dir = 'Training'
occupancy_output_root = 'Occupancy'

# Sliding window settings (filtered index space).
window_start = 0
window_end = 500
window_shift = 1
window_size = Settings.occupancy_config.window_size

# Grid + integration settings for occupancy evidence.
extent_mm = Settings.occupancy_config.extent_mm       # Half-width/half-height of robot-frame occupancy map (mm): [-extent, +extent].
grid_mm = Settings.occupancy_config.grid_mm           # Occupancy grid cell size in mm (smaller = finer map, higher compute/memory).
sigma_perp_mm = Settings.occupancy_config.sigma_perp_mm     # Segment evidence spread perpendicular to predicted profile segments (mm).
sigma_para_mm = Settings.occupancy_config.sigma_para_mm    # Segment evidence soft spread along segment direction / beyond endpoints (mm).

# Model inference settings.
prediction_batch_size = 256
presence_threshold_override = None  # If None, use training params threshold.

# Parallel compute settings.
parallel_workers = 8

# Optional plotting from saved occupancy results.
save_plots = True
# Any iterable of series indices=
plot_series_selection = range(0, 500, 50)
plot_max_count = 30
plot_heatmap_vmin = 0.0
plot_heatmap_vmax = 1.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def reset_dir(path):
    if os.path.isdir(path):
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                shutil.rmtree(full)
            else:
                os.remove(full)
    os.makedirs(path, exist_ok=True)


def resolve_plot_indices(n_samples, selection):
    if selection is None:
        return []
    try:
        selected = [int(v) for v in selection]
    except TypeError as exc:
        raise ValueError("plot_series_selection must be an iterable of indices or None.") from exc
    out = sorted(set(int(v) for v in selected if 0 <= int(v) < n_samples))
    if plot_max_count is not None:
        out = out[: int(plot_max_count)]
    return out


def compute_occupancy_for_windows(
    profile_centers_deg,
    distance_mm,
    presence_bin,
    presence_probs,
    metadata,
    starts,
    x_grid,
    y_grid,
    xx,
    yy,
    progress_desc='Computing occupancy windows',
):
    rob_x_all = metadata['rob_x']
    rob_y_all = metadata['rob_y']
    rob_yaw_all = metadata['rob_yaw_deg']

    results = [None] * len(starts)

    def _task(seq_idx, win_start):
        win_end = win_start + window_size - 1
        indices = np.arange(win_start, win_end + 1, dtype=np.int32)

        dist_seq = distance_mm[indices].copy()
        bin_seq = presence_bin[indices]
        dist_seq[bin_seq == 0] = np.nan

        evidence_norm = build_robot_frame_evidence(
            profile_centers_deg_seq=profile_centers_deg[indices],
            distance_mm_seq=dist_seq,
            presence_probs_seq=presence_probs[indices],
            presence_bin_seq=bin_seq,
            rob_x_seq=rob_x_all[indices],
            rob_y_seq=rob_y_all[indices],
            rob_yaw_deg_seq=rob_yaw_all[indices],
            x_grid=x_grid,
            y_grid=y_grid,
            xx=xx,
            yy=yy,
            grid_mm=grid_mm,
            sigma_perp_mm=sigma_perp_mm,
            sigma_para_mm=sigma_para_mm,
            apply_smoothing=False,
        )
        anchor_idx = int(win_end)
        return seq_idx, win_start, win_end, indices, anchor_idx, evidence_norm

    max_workers = max(1, int(parallel_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_task, i, s) for i, s in enumerate(starts)]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=progress_desc):
            seq_idx, win_start, win_end, indices, anchor_idx, evidence_norm = fut.result()
            results[seq_idx] = (win_start, win_end, indices, anchor_idx, evidence_norm)

    return results


def save_occupancy_package(out_npz, results_pred, results_real, metadata, x_grid, y_grid):
    rob_x_all = metadata['rob_x']
    rob_y_all = metadata['rob_y']
    rob_yaw_all = metadata['rob_yaw_deg']

    heatmaps_pred = np.stack([r[4] for r in results_pred], axis=0).astype(np.float32)
    heatmaps_real = np.stack([r[4] for r in results_real], axis=0).astype(np.float32)
    chunk_indices = np.stack([r[2] for r in results_pred], axis=0).astype(np.int32)
    starts = np.asarray([r[0] for r in results_pred], dtype=np.int32)
    ends = np.asarray([r[1] for r in results_pred], dtype=np.int32)
    anchor_idx = np.asarray([r[3] for r in results_pred], dtype=np.int32)

    np.savez_compressed(
        out_npz,
        heatmaps_pred=heatmaps_pred,
        heatmaps_real=heatmaps_real,
        heatmaps=heatmaps_pred,  # backward-compatible alias
        x_grid=x_grid.astype(np.float32),
        y_grid=y_grid.astype(np.float32),
        chunk_indices=chunk_indices,
        window_start=starts,
        window_end=ends,
        anchor_index=anchor_idx,
        anchor_rob_x=rob_x_all[anchor_idx].astype(np.float32),
        anchor_rob_y=rob_y_all[anchor_idx].astype(np.float32),
        anchor_rob_yaw_deg=rob_yaw_all[anchor_idx].astype(np.float32),
        chunk_rob_x=rob_x_all[chunk_indices].astype(np.float32),
        chunk_rob_y=rob_y_all[chunk_indices].astype(np.float32),
        chunk_rob_yaw_deg=rob_yaw_all[chunk_indices].astype(np.float32),
        grid_mm=np.float32(grid_mm),
        extent_mm=np.float32(extent_mm),
        sigma_perp_mm=np.float32(sigma_perp_mm),
        sigma_para_mm=np.float32(sigma_para_mm),
    )


def write_session_readme(session_out_dir, session_name):
    readme_path = os.path.join(session_out_dir, 'README.md')
    txt = f"""# Occupancy Output: {session_name}

This folder contains precomputed robot-frame occupancy evidence for sliding windows in one session.

## Files

- `{session_name}_occupancy_series_data.npz`
  Main data package for downstream analysis.
- `{session_name}_occupancy_summary.json`
  Run/config summary (window settings, grid settings, threshold, etc.).
- `plots/` (optional)
  Example figures generated from the saved `.npz` package.

## NPZ Contents

- `heatmaps_pred`:
  Occupancy evidence maps from **predicted** profiles (shape: `[n_windows, H, W]`).
- `heatmaps_real`:
  Occupancy evidence maps from **real** profiles (same shape as above).
- `heatmaps`:
  Backward-compatible alias to `heatmaps_pred`.
- `x_grid`, `y_grid`:
  Robot-frame grid axes in mm used by all heatmaps.
- `chunk_indices`:
  Filtered sample indices used per window (shape: `[n_windows, window_size]`).
- `window_start`, `window_end`:
  Start/end filtered indices for each window.
- `anchor_index`:
  Anchor index for each window (equal to `window_end`).
- `anchor_rob_x`, `anchor_rob_y`, `anchor_rob_yaw_deg`:
  Anchor pose in world coordinates.
- `chunk_rob_x`, `chunk_rob_y`, `chunk_rob_yaw_deg`:
  Pose sequence (world frame) for each window.
- `grid_mm`, `extent_mm`, `sigma_perp_mm`, `sigma_para_mm`:
  Occupancy integration parameters used to generate this package.

## Notes

- Indices are in **filtered sample space** (after invalid rows were removed).
- Both predicted and real occupancy use identical windows/grid/settings for direct comparison.
"""
    with open(readme_path, 'w') as f:
        f.write(txt)
    print(f'Saved readme: {readme_path}')


def plot_examples_from_saved_package(npz_path, plot_dir, metadata, selected_series):
    if len(selected_series) == 0:
        print('No series indices selected for plotting.')
        return

    ensure_dir(plot_dir)
    data = np.load(npz_path)
    heatmaps_pred = data['heatmaps_pred'] if 'heatmaps_pred' in data else data['heatmaps']
    heatmaps_real = data['heatmaps_real'] if 'heatmaps_real' in data else data['heatmaps']
    x_grid = data['x_grid']
    y_grid = data['y_grid']
    starts = data['window_start']
    ends = data['window_end']
    chunk_indices = data['chunk_indices']

    rob_x_all = metadata['rob_x']
    rob_y_all = metadata['rob_y']
    rob_yaw_all = metadata['rob_yaw_deg']

    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    y_min, y_max = float(y_grid[0]), float(y_grid[-1])

    for sidx in selected_series:
        hm_pred = heatmaps_pred[sidx]
        hm_real = heatmaps_real[sidx]
        win_start = int(starts[sidx])
        win_end = int(ends[sidx])
        indices = chunk_indices[sidx]
        anchor_idx = int(win_end)
        anchor_x = float(rob_x_all[anchor_idx])
        anchor_y = float(rob_y_all[anchor_idx])
        anchor_yaw = float(rob_yaw_all[anchor_idx])

        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        im = axes[0].imshow(
            hm_pred,
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            cmap='magma',
            aspect='equal',
            vmin=plot_heatmap_vmin,
            vmax=plot_heatmap_vmax,
        )
        fig.colorbar(im, ax=axes[0], label='Pred Occupancy')
        axes[0].scatter([0], [0], c='cyan', s=50, label='Anchor robot')
        axes[0].arrow(0, 0, 250, 0, head_width=70, head_length=90, fc='cyan', ec='cyan', alpha=0.9)
        axes[0].set_title(f'Pred Occupancy\nseries_idx={sidx}, anchor={anchor_idx}')
        axes[0].set_xlabel('Robot X (mm)')
        axes[0].set_ylabel('Robot Y (mm)')
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc='upper right')

        im2 = axes[1].imshow(
            hm_real,
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            cmap='magma',
            aspect='equal',
            vmin=plot_heatmap_vmin,
            vmax=plot_heatmap_vmax,
        )
        fig.colorbar(im2, ax=axes[1], label='Real Occupancy')
        axes[1].scatter([0], [0], c='cyan', s=50, label='Anchor robot')
        axes[1].arrow(0, 0, 250, 0, head_width=70, head_length=90, fc='cyan', ec='cyan', alpha=0.9)
        axes[1].set_title('Real-Profile Occupancy')
        axes[1].set_xlabel('Robot X (mm)')
        axes[1].set_ylabel('Robot Y (mm)')
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc='upper right')

        axes[2].plot(rob_x_all, rob_y_all, color='lightgray', linewidth=1.0, label='Full trajectory')
        axes[2].scatter(rob_x_all[indices], rob_y_all[indices], c='tab:blue', s=28, label='Chunk poses')
        yaw_r = np.deg2rad(rob_yaw_all[indices])
        u = 140.0 * np.cos(yaw_r)
        v = 140.0 * np.sin(yaw_r)
        axes[2].quiver(
            rob_x_all[indices], rob_y_all[indices], u, v,
            angles='xy', scale_units='xy', scale=1, width=0.003, color='tab:blue', alpha=0.8
        )
        axes[2].scatter([anchor_x], [anchor_y], c='red', s=70, label='Anchor pose')
        axes[2].arrow(
            anchor_x, anchor_y,
            220.0 * np.cos(np.deg2rad(anchor_yaw)),
            220.0 * np.sin(np.deg2rad(anchor_yaw)),
            head_width=60, head_length=80, fc='red', ec='red', alpha=0.9
        )
        axes[2].set_title(f'World Context\nindices [{win_start}, {win_end}]')
        axes[2].set_xlabel('World X (mm)')
        axes[2].set_ylabel('World Y (mm)')
        axes[2].axis('equal')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='best')

        plt.tight_layout()
        out_png = os.path.join(
            plot_dir,
            f'{session_to_compute}_seriesidx_{sidx:04d}_chunk_{win_start:04d}_{win_end:04d}_occupancy.png'
        )
        plt.savefig(out_png, dpi=220, bbox_inches='tight')
        plt.close(fig)

    print(f'Saved {len(selected_series)} occupancy plots to: {plot_dir}')


def main():
    ensure_dir(occupancy_output_root)
    session_out_dir = os.path.join(occupancy_output_root, session_to_compute)
    reset_dir(session_out_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Loading model artifacts from: {training_output_dir}')
    model, y_scaler, params = load_training_artifacts(training_output_dir, device)

    profile_opening_angle = float(params['profile_opening_angle'])
    profile_steps = int(params['profile_steps'])
    threshold = float(params.get('presence_threshold', 0.5))
    if presence_threshold_override is not None:
        threshold = float(presence_threshold_override)

    sonar_data, real_distance_mm, profile_centers_deg, metadata = load_session_data(
        session_to_compute, profile_opening_angle, profile_steps
    )
    print(f'Loaded session {session_to_compute}: {len(sonar_data)} filtered samples')

    pred_presence_probs, pred_presence_bin, pred_distance_mm, _ = predict_profiles(
        model=model,
        y_scaler=y_scaler,
        sonar_data=sonar_data,
        batch_size=prediction_batch_size,
        threshold=threshold,
        device=device,
    )
    print('Predicted profiles for all samples.')

    n_samples = len(sonar_data)
    start = max(0, int(window_start))
    end = min(int(window_end), n_samples - 1)
    max_window_start = end - int(window_size) + 1
    if max_window_start < start:
        raise ValueError('window_size is larger than selected interval.')
    starts = list(range(start, max_window_start + 1, int(window_shift)))
    print(f'Computing occupancy for {len(starts)} windows (size={window_size}, shift={window_shift}).', flush=True)

    x_grid, y_grid, xx, yy = make_robot_frame_grid(extent_mm=extent_mm, grid_mm=grid_mm)
    print('Phase 1/2: predicted-profile occupancy...', flush=True)
    results_pred = compute_occupancy_for_windows(
        profile_centers_deg=profile_centers_deg,
        distance_mm=pred_distance_mm,
        presence_bin=pred_presence_bin,
        presence_probs=pred_presence_probs,
        metadata=metadata,
        starts=starts,
        x_grid=x_grid,
        y_grid=y_grid,
        xx=xx,
        yy=yy,
        progress_desc='Pred occupancy',
    )
    print('Phase 1/2 done: predicted-profile occupancy.', flush=True)

    real_presence_bin = np.isfinite(real_distance_mm).astype(np.uint8)
    real_presence_probs = real_presence_bin.astype(np.float32)
    print('Phase 2/2: real-profile occupancy...', flush=True)
    results_real = compute_occupancy_for_windows(
        profile_centers_deg=profile_centers_deg,
        distance_mm=real_distance_mm,
        presence_bin=real_presence_bin,
        presence_probs=real_presence_probs,
        metadata=metadata,
        starts=starts,
        x_grid=x_grid,
        y_grid=y_grid,
        xx=xx,
        yy=yy,
        progress_desc='Real occupancy',
    )
    print('Phase 2/2 done: real-profile occupancy.', flush=True)

    out_npz = os.path.join(session_out_dir, f'{session_to_compute}_occupancy_series_data.npz')
    save_occupancy_package(out_npz, results_pred, results_real, metadata, x_grid, y_grid)
    print(f'Saved occupancy package: {out_npz}')
    write_session_readme(session_out_dir, session_to_compute)

    summary = {
        'session': session_to_compute,
        'training_output_dir': training_output_dir,
        'n_samples_filtered': int(n_samples),
        'window_start': int(start),
        'window_end': int(end),
        'window_size': int(window_size),
        'window_shift': int(window_shift),
        'n_windows': int(len(starts)),
        'grid_mm': float(grid_mm),
        'extent_mm': float(extent_mm),
        'sigma_perp_mm': float(sigma_perp_mm),
        'sigma_para_mm': float(sigma_para_mm),
        'presence_threshold_used': float(threshold),
        'parallel_workers': int(parallel_workers),
        'output_npz': out_npz,
    }
    summary_path = os.path.join(session_out_dir, f'{session_to_compute}_occupancy_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved summary: {summary_path}')

    if save_plots:
        selected = resolve_plot_indices(len(starts), plot_series_selection)
        plot_dir = os.path.join(session_out_dir, 'plots')
        plot_examples_from_saved_package(
            npz_path=out_npz,
            plot_dir=plot_dir,
            metadata=metadata,
            selected_series=selected,
        )


if __name__ == '__main__':
    main()
