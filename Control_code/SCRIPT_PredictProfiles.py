#
# SCRIPT_PredictProfiles: Load trained two-head model and predict profiles for one session.
#
# This script loads a training output folder and predicts:
# 1) presence probabilities/binary predictions per azimuth bin
# 2) distance predictions per azimuth bin (in mm)
#

import json
import os
import shutil

import joblib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from Library import DataProcessor


# ============================================
# PREDICTION CONFIGURATION
# ============================================
session_to_predict = 'sessionB05'  # Session to predict profiles for
training_output_dir = 'Training'   # Directory containing trained model artifacts
prediction_output_dir = 'Prediction'  # Directory for prediction outputs
prediction_batch_size = 256
# If None, use threshold saved in training_params.json.
presence_threshold_override = None

# Optional index-based visualization
# Indices are in filtered sample space after invalid rows are removed.
plot_indices = [0]              # Example: [0, 10, 25]
plot_index_range = [210,220,1]         # Example: (100, 120, 2)
save_plots = True
save_overlay_plot = True
save_integrated_profile_map = True

# Segment integration settings (mm)
integration_grid_mm = 20.0
integration_sigma_perp_mm = 40.0   # spread across segment
integration_sigma_para_mm = 120.0  # soft falloff beyond segment ends
integration_margin_mm = 300.0

# Sliding heatmap series (filtered index space)
save_chunk_heatmap_series = False
chunk_series_start = 0
chunk_series_end = 500
chunk_series_shift = 1
chunk_series_size = 5
chunk_series_subdir = 'heatmap_series'

# Sliding heatmap series in robot frame (anchor = last index in each chunk)
save_robot_frame_chunk_heatmap_series = True
robot_chunk_series_subdir = 'heatmap_series_robot'
robot_frame_extent_mm = 2500.0
robot_frame_grid_mm = 20.0
# Fixed color scale for left panel in robot-frame series plots.
# Set either value to None to fall back to matplotlib auto-scaling.
robot_frame_heatmap_vmin = 0.0
robot_frame_heatmap_vmax = 1.0


def reset_prediction_output_dir(path):
    """Delete all existing files/subfolders in prediction output directory."""
    if os.path.isdir(path):
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
    os.makedirs(path, exist_ok=True)


# ============================================
# MODEL/SCALER DEFINITIONS (must match training)
# ============================================
class DistanceScaler:
    """Per-bin standardization fit on present-wall bins only."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, y_distance, y_presence):
        n_bins = y_distance.shape[1]
        self.mean_ = np.zeros(n_bins, dtype=np.float32)
        self.scale_ = np.ones(n_bins, dtype=np.float32)

        for bin_idx in range(n_bins):
            present_vals = y_distance[y_presence[:, bin_idx] > 0.5, bin_idx]
            if present_vals.size >= 2:
                std = np.std(present_vals)
                self.mean_[bin_idx] = np.mean(present_vals)
                self.scale_[bin_idx] = std if std > 1e-6 else 1.0
            elif present_vals.size == 1:
                self.mean_[bin_idx] = present_vals[0]
                self.scale_[bin_idx] = 1.0
            else:
                self.mean_[bin_idx] = 0.0
                self.scale_[bin_idx] = 1.0
        return self

    def transform(self, y):
        return (y - self.mean_) / self.scale_

    def inverse_transform(self, y):
        return y * self.scale_ + self.mean_


class TwoHeadedSonarCNN(nn.Module):
    def __init__(self, input_shape, profile_steps, conv_filters, kernel_size, pool_size, dropout_rate):
        super(TwoHeadedSonarCNN, self).__init__()

        self.shared = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=conv_filters[0], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate),

            nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate),

            nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate),
        )

        self._calculate_flattened_size(input_shape)

        # Presence logits
        self.presence_head = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, profile_steps),
        )

        # Distance regression
        self.distance_head = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, profile_steps),
        )

    def _calculate_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.shared(dummy_input)
            self.flattened_size = x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        features = self.shared(x)
        features = features.view(features.size(0), -1)
        presence_logits = self.presence_head(features)
        distance_scaled = self.distance_head(features)
        return presence_logits, distance_scaled


def load_training_artifacts(base_dir, device):
    params_path = os.path.join(base_dir, 'training_params.json')
    model_path = os.path.join(base_dir, 'best_model_pytorch.pth')
    scaler_path = os.path.join(base_dir, 'y_scaler.joblib')

    missing = [p for p in [params_path, model_path, scaler_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing training artifacts in '{base_dir}': {missing}")

    with open(params_path, 'r') as f:
        params = json.load(f)

    input_shape = tuple(params.get('input_shape', [2, 200]))
    profile_steps = int(params['profile_steps'])
    conv_filters = params['conv_filters']
    kernel_size = int(params['kernel_size'])
    pool_size = int(params['pool_size'])
    dropout_rate = float(params['dropout_rate'])

    model = TwoHeadedSonarCNN(
        input_shape=input_shape,
        profile_steps=profile_steps,
        conv_filters=conv_filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        dropout_rate=dropout_rate,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_scaler = joblib.load(scaler_path)
    return model, y_scaler, params


def load_session_data(session_name, profile_opening_angle, profile_steps):
    dc = DataProcessor.DataCollection([session_name])
    sonar_data = dc.load_sonar(flatten=False)  # shape: (N, 200, 2)
    profiles_data, profile_centers = dc.load_profiles(
        opening_angle=profile_opening_angle, steps=profile_steps
    )

    # Keep only rows with finite sonar/profile values.
    finite_mask = np.isfinite(sonar_data).all(axis=(1, 2))
    finite_mask &= np.isfinite(profiles_data).all(axis=1)
    kept_indices = np.where(finite_mask)[0]
    sonar_data = sonar_data[finite_mask]
    profiles_data = profiles_data[finite_mask]
    profile_centers = profile_centers[finite_mask]

    metadata = {
        'rob_x': dc.rob_x[finite_mask],
        'rob_y': dc.rob_y[finite_mask],
        'rob_yaw_deg': dc.rob_yaw_deg[finite_mask],
        'quadrants': dc.quadrants[finite_mask],
        'kept_indices': kept_indices,
        'total_samples_original': len(finite_mask),
        'total_samples_used': int(finite_mask.sum()),
    }
    return sonar_data, profiles_data, profile_centers, metadata


def resolve_plot_indices(n_samples, indices, index_range):
    selected = []
    if indices is not None:
        selected.extend(indices)
    if index_range is not None:
        if len(index_range) != 3:
            raise ValueError("plot_index_range must be (start, stop, step).")
        start, stop, step = index_range
        selected.extend(list(range(start, stop, step)))

    if len(selected) == 0:
        return []

    unique_sorted = sorted(set(int(i) for i in selected))
    valid = [i for i in unique_sorted if 0 <= i < n_samples]
    dropped = [i for i in unique_sorted if i < 0 or i >= n_samples]
    if dropped:
        print(f"Warning: dropped out-of-range indices: {dropped}")
    return valid


def plot_selected_predictions(
    session_name,
    out_dir,
    selected_indices,
    metadata,
    profile_centers_deg,
    real_distance_mm,
    pred_distance_mm,
    pred_presence_bin,
    verbose=True,
):
    if len(selected_indices) == 0:
        print("No selected indices to plot.")
        return

    plot_dir = os.path.join(out_dir, 'indices')
    os.makedirs(plot_dir, exist_ok=True)

    rob_x_all = metadata['rob_x']
    rob_y_all = metadata['rob_y']
    rob_yaw_all = metadata['rob_yaw_deg']

    for idx in selected_indices:
        rob_x_i = rob_x_all[idx]
        rob_y_i = rob_y_all[idx]
        rob_yaw_i = rob_yaw_all[idx]

        az_deg = profile_centers_deg[idx]
        true_dist = real_distance_mm[idx]
        pred_dist = pred_distance_mm[idx].copy()
        pred_dist[pred_presence_bin[idx] == 0] = np.nan

        true_x_w, true_y_w = DataProcessor.robot2world(az_deg, true_dist, rob_x_i, rob_y_i, rob_yaw_i)
        pred_x_w, pred_y_w = DataProcessor.robot2world(az_deg, pred_dist, rob_x_i, rob_y_i, rob_yaw_i)

        plt.figure(figsize=(8, 8))
        plt.plot(rob_x_all, rob_y_all, color='lightgray', linewidth=1.0, label='Robot trajectory')
        plt.scatter([rob_x_i], [rob_y_i], color='red', s=60, label='Robot position')

        yaw_rad = np.deg2rad(rob_yaw_i)
        arrow_len = 250.0
        plt.arrow(
            rob_x_i,
            rob_y_i,
            arrow_len * np.cos(yaw_rad),
            arrow_len * np.sin(yaw_rad),
            head_width=60,
            head_length=80,
            fc='red',
            ec='red',
            alpha=0.8,
        )

        plt.plot(true_x_w, true_y_w, 'o-', color='tab:blue', markersize=4, linewidth=1.5, label='True profile')
        plt.plot(pred_x_w, pred_y_w, 'x--', color='tab:orange', markersize=5, linewidth=1.2, label='Pred profile (presence-masked)')

        plt.title(f'{session_name} | index={idx} | yaw={rob_yaw_i:.1f}°')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()

        plot_path = os.path.join(plot_dir, f'{session_name}_profile_world_idx_{idx:04d}.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        if verbose:
            print(f"Saved plot: {plot_path}")
    if not verbose:
        print(f"Saved {len(selected_indices)} index plots to: {plot_dir}")


def plot_overlay_predictions(
    session_name,
    out_dir,
    selected_indices,
    metadata,
    profile_centers_deg,
    real_distance_mm,
    pred_distance_mm,
    pred_presence_bin,
    verbose=True,
):
    """Plot all selected indices overlaid in world coordinates with matching colors."""
    if len(selected_indices) == 0:
        return

    rob_x_all = metadata['rob_x']
    rob_y_all = metadata['rob_y']
    rob_yaw_all = metadata['rob_yaw_deg']

    # Qualitative colormap with discrete colors.
    cmap = plt.get_cmap('tab20')
    n_colors = max(1, len(selected_indices))

    plt.figure(figsize=(10, 10))
    plt.plot(rob_x_all, rob_y_all, color='lightgray', linewidth=1.0, label='Robot trajectory')

    for k, idx in enumerate(selected_indices):
        color = cmap(k % 20) if n_colors > 20 else cmap(k / max(1, n_colors - 1))

        rob_x_i = rob_x_all[idx]
        rob_y_i = rob_y_all[idx]
        rob_yaw_i = rob_yaw_all[idx]

        az_deg = profile_centers_deg[idx]
        true_dist = real_distance_mm[idx]
        pred_dist = pred_distance_mm[idx].copy()
        pred_dist[pred_presence_bin[idx] == 0] = np.nan

        true_x_w, true_y_w = DataProcessor.robot2world(az_deg, true_dist, rob_x_i, rob_y_i, rob_yaw_i)
        pred_x_w, pred_y_w = DataProcessor.robot2world(az_deg, pred_dist, rob_x_i, rob_y_i, rob_yaw_i)

        plt.scatter([rob_x_i], [rob_y_i], color=color, s=55)
        plt.plot(true_x_w, true_y_w, '-', color=color, linewidth=1.8, alpha=0.95)
        plt.plot(pred_x_w, pred_y_w, '--', color=color, linewidth=1.5, alpha=0.95)
        plt.text(rob_x_i, rob_y_i, f'{idx}', color=color, fontsize=8)

    # Legend semantics once, colors are index-specific and annotated at robot locations.
    plt.plot([], [], '-', color='black', linewidth=1.8, label='True profile')
    plt.plot([], [], '--', color='black', linewidth=1.5, label='Pred profile (presence-masked)')

    plt.title(f'{session_name} | Overlay of Selected Indices ({len(selected_indices)} samples)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    overlay_path = os.path.join(out_dir, f'{session_name}_profile_world_overlay_selected.png')
    plt.savefig(overlay_path, dpi=220, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"Saved overlay plot: {overlay_path}")


def create_segment_evidence_map(
    session_name,
    out_dir,
    selected_indices,
    metadata,
    profile_centers_deg,
    pred_distance_mm,
    pred_presence_bin,
    pred_presence_probs,
    real_distance_mm_raw,
    filename_prefix=None,
    title_suffix=None,
    save_npz=True,
    verbose=True,
    grid_mm=20.0,
    sigma_perp_mm=40.0,
    sigma_para_mm=120.0,
    margin_mm=300.0,
):
    """Integrate predicted profiles as anisotropic segment evidence on a 2D grid."""
    if len(selected_indices) == 0:
        return

    rob_x_all = metadata['rob_x']
    rob_y_all = metadata['rob_y']
    rob_yaw_all = metadata['rob_yaw_deg']

    segment_records = []
    endpoint_x = []
    endpoint_y = []

    for idx in selected_indices:
        az_deg = profile_centers_deg[idx]
        pred_dist = pred_distance_mm[idx].copy()
        pred_dist[pred_presence_bin[idx] == 0] = np.nan
        probs = pred_presence_probs[idx]

        xw, yw = DataProcessor.robot2world(az_deg, pred_dist, rob_x_all[idx], rob_y_all[idx], rob_yaw_all[idx])
        valid = np.isfinite(xw) & np.isfinite(yw)
        endpoint_x.extend(xw[valid].tolist())
        endpoint_y.extend(yw[valid].tolist())

        for j in range(len(az_deg) - 1):
            if not (np.isfinite(xw[j]) and np.isfinite(yw[j]) and np.isfinite(xw[j + 1]) and np.isfinite(yw[j + 1])):
                continue
            p0 = np.array([xw[j], yw[j]], dtype=float)
            p1 = np.array([xw[j + 1], yw[j + 1]], dtype=float)
            seg_vec = p1 - p0
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len < 1e-6:
                continue
            weight = float(0.5 * (probs[j] + probs[j + 1]))
            if weight <= 0.0:
                continue
            segment_records.append((p0, p1, seg_len, weight))

    if len(segment_records) == 0:
        if verbose:
            print("No valid predicted segments for integration.")
        return

    x_candidates = np.concatenate([rob_x_all[selected_indices], np.asarray(endpoint_x)]) if len(endpoint_x) > 0 else rob_x_all[selected_indices]
    y_candidates = np.concatenate([rob_y_all[selected_indices], np.asarray(endpoint_y)]) if len(endpoint_y) > 0 else rob_y_all[selected_indices]

    x_min = float(np.min(x_candidates) - margin_mm)
    x_max = float(np.max(x_candidates) + margin_mm)
    y_min = float(np.min(y_candidates) - margin_mm)
    y_max = float(np.max(y_candidates) + margin_mm)

    x_grid = np.arange(x_min, x_max + grid_mm, grid_mm)
    y_grid = np.arange(y_min, y_max + grid_mm, grid_mm)
    xx, yy = np.meshgrid(x_grid, y_grid)
    evidence = np.zeros_like(xx, dtype=np.float32)

    for p0, p1, seg_len, weight in segment_records:
        mid = 0.5 * (p0 + p1)
        tangent = (p1 - p0) / seg_len
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        half_len = 0.5 * seg_len

        local_margin = 3.0 * max(sigma_perp_mm, sigma_para_mm)
        bx0 = max(x_min, min(p0[0], p1[0]) - local_margin)
        bx1 = min(x_max, max(p0[0], p1[0]) + local_margin)
        by0 = max(y_min, min(p0[1], p1[1]) - local_margin)
        by1 = min(y_max, max(p0[1], p1[1]) + local_margin)

        ix0 = int(max(0, np.floor((bx0 - x_min) / grid_mm)))
        ix1 = int(min(len(x_grid) - 1, np.ceil((bx1 - x_min) / grid_mm)))
        iy0 = int(max(0, np.floor((by0 - y_min) / grid_mm)))
        iy1 = int(min(len(y_grid) - 1, np.ceil((by1 - y_min) / grid_mm)))

        sub_x = xx[iy0:iy1 + 1, ix0:ix1 + 1]
        sub_y = yy[iy0:iy1 + 1, ix0:ix1 + 1]
        dx = sub_x - mid[0]
        dy = sub_y - mid[1]

        d_para = dx * tangent[0] + dy * tangent[1]
        d_perp = dx * normal[0] + dy * normal[1]
        outside_para = np.maximum(np.abs(d_para) - half_len, 0.0)

        kernel = np.exp(
            -0.5 * (d_perp / sigma_perp_mm) ** 2
            -0.5 * (outside_para / sigma_para_mm) ** 2
        )
        evidence[iy0:iy1 + 1, ix0:ix1 + 1] += (weight * kernel).astype(np.float32)

    evidence_norm = evidence / max(len(selected_indices), 1)

    # Heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(
        evidence_norm,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap='magma',
        aspect='equal'
    )
    plt.colorbar(im, label='Integrated Segment Evidence (a.u.)')
    plt.plot(rob_x_all, rob_y_all, color='white', alpha=0.35, linewidth=0.8, label='Trajectory')
    plt.scatter(rob_x_all[selected_indices], rob_y_all[selected_indices], c='cyan', s=18, alpha=0.8, label='Selected poses')

    # Overlay raw ground-truth profiles (no cutoff) for selected indices.
    for idx in selected_indices:
        gt_dist = real_distance_mm_raw[idx]
        gt_x_w, gt_y_w = DataProcessor.robot2world(
            profile_centers_deg[idx],
            gt_dist,
            rob_x_all[idx],
            rob_y_all[idx],
            rob_yaw_all[idx],
        )
        valid_gt = np.isfinite(gt_x_w) & np.isfinite(gt_y_w)
        if np.any(valid_gt):
            plt.plot(gt_x_w[valid_gt], gt_y_w[valid_gt], '-', color='lime', linewidth=1.0, alpha=0.55)

    # Legend handle for ground truth overlay.
    plt.plot([], [], '-', color='lime', linewidth=1.0, alpha=0.8, label='Ground-truth profiles (raw)')
    full_title = f'{session_name} | Integrated Predicted-Wall Evidence (Segment Model)'
    if title_suffix:
        full_title = f'{full_title}\n{title_suffix}'
    plt.title(full_title)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend(loc='best')
    plt.tight_layout()
    base_name = filename_prefix if filename_prefix else f'{session_name}_profile_segment_evidence'
    heatmap_path = os.path.join(out_dir, f'{base_name}_heatmap.png')
    plt.savefig(heatmap_path, dpi=220, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"Saved integrated heatmap: {heatmap_path}")

    if save_npz:
        npz_path = os.path.join(out_dir, f'{base_name}_map.npz')
        np.savez_compressed(
            npz_path,
            x_grid=x_grid,
            y_grid=y_grid,
            evidence=evidence,
            evidence_norm=evidence_norm,
            selected_indices=np.asarray(selected_indices, dtype=int),
            sigma_perp_mm=float(sigma_perp_mm),
            sigma_para_mm=float(sigma_para_mm),
            grid_mm=float(grid_mm),
        )
        if verbose:
            print(f"Saved integrated map data: {npz_path}")


def create_chunk_heatmap_series(
    session_name,
    out_dir,
    metadata,
    profile_centers_deg,
    real_distance_mm_raw,
    pred_distance_mm,
    pred_presence_bin,
    pred_presence_probs,
    n_samples,
    start_idx,
    end_idx,
    shift,
    chunk_size,
    subdir='heatmap_series',
    grid_mm=20.0,
    sigma_perp_mm=40.0,
    sigma_para_mm=120.0,
    margin_mm=300.0,
):
    """Generate sliding-window integrated heatmaps over index chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_series_size must be >= 1.")
    if shift < 1:
        raise ValueError("chunk_series_shift must be >= 1.")

    start = max(0, int(start_idx))
    end = min(int(end_idx), n_samples - 1)
    if start > end:
        print("Chunk heatmap series skipped: start index is greater than end index after clipping.")
        return

    max_window_start = end - chunk_size + 1
    if max_window_start < start:
        print("Chunk heatmap series skipped: chunk size larger than selected interval.")
        return

    series_dir = os.path.join(out_dir, subdir)
    os.makedirs(series_dir, exist_ok=True)

    window_starts = list(range(start, max_window_start + 1, shift))
    n_maps = 0
    print(f"Generating world-frame chunk heatmap series: {len(window_starts)} windows")
    for seq_idx, win_start in enumerate(tqdm(window_starts, desc="World-frame chunks")):
        win_end = win_start + chunk_size - 1
        indices = list(range(win_start, win_end + 1))
        prefix = f'{session_name}_seriesidx_{seq_idx:04d}_chunk_{win_start:04d}_{win_end:04d}'
        subtitle = f'series_idx={seq_idx} | indices [{win_start}, {win_end}] (size={chunk_size}, shift={shift})'

        create_segment_evidence_map(
            session_name=session_name,
            out_dir=series_dir,
            selected_indices=indices,
            metadata=metadata,
            profile_centers_deg=profile_centers_deg,
            pred_distance_mm=pred_distance_mm,
            pred_presence_bin=pred_presence_bin,
            pred_presence_probs=pred_presence_probs,
            real_distance_mm_raw=real_distance_mm_raw,
            filename_prefix=prefix,
            title_suffix=subtitle,
            save_npz=False,
            verbose=False,
            grid_mm=grid_mm,
            sigma_perp_mm=sigma_perp_mm,
            sigma_para_mm=sigma_para_mm,
            margin_mm=margin_mm,
        )
        n_maps += 1

    print(f"Saved chunk heatmap series: {n_maps} maps in {series_dir}")


def create_robot_frame_chunk_heatmap_series(
    session_name,
    out_dir,
    metadata,
    profile_centers_deg,
    real_distance_mm_raw,
    pred_distance_mm,
    pred_presence_bin,
    pred_presence_probs,
    n_samples,
    start_idx,
    end_idx,
    shift,
    chunk_size,
    subdir='heatmap_series_robot',
    extent_mm=2500.0,
    grid_mm=20.0,
    sigma_perp_mm=40.0,
    sigma_para_mm=120.0,
    heatmap_vmin=0.0,
    heatmap_vmax=1.0,
):
    """Generate sliding-window integrated heatmaps in robot coordinates."""
    if chunk_size < 1:
        raise ValueError("chunk_series_size must be >= 1.")
    if shift < 1:
        raise ValueError("chunk_series_shift must be >= 1.")

    start = max(0, int(start_idx))
    end = min(int(end_idx), n_samples - 1)
    if start > end:
        print("Robot-frame chunk series skipped: start index is greater than end index after clipping.")
        return

    max_window_start = end - chunk_size + 1
    if max_window_start < start:
        print("Robot-frame chunk series skipped: chunk size larger than selected interval.")
        return

    series_dir = os.path.join(out_dir, subdir)
    os.makedirs(series_dir, exist_ok=True)

    rob_x_all = metadata['rob_x']
    rob_y_all = metadata['rob_y']
    rob_yaw_all = metadata['rob_yaw_deg']

    x_min, x_max = -float(extent_mm), float(extent_mm)
    y_min, y_max = -float(extent_mm), float(extent_mm)
    x_grid = np.arange(x_min, x_max + grid_mm, grid_mm)
    y_grid = np.arange(y_min, y_max + grid_mm, grid_mm)
    xx, yy = np.meshgrid(x_grid, y_grid)

    window_starts = list(range(start, max_window_start + 1, shift))
    n_maps = 0
    heatmap_stack = []
    chunk_indices_stack = []
    anchor_indices = []
    anchor_rob_x = []
    anchor_rob_y = []
    anchor_rob_yaw_deg = []
    chunk_rob_x_stack = []
    chunk_rob_y_stack = []
    chunk_rob_yaw_deg_stack = []
    window_starts_saved = []
    window_ends_saved = []
    print(f"Generating robot-frame chunk heatmap series: {len(window_starts)} windows")
    for seq_idx, win_start in enumerate(tqdm(window_starts, desc="Robot-frame chunks")):
        win_end = win_start + chunk_size - 1
        indices = list(range(win_start, win_end + 1))
        anchor_idx = win_end

        anchor_x = rob_x_all[anchor_idx]
        anchor_y = rob_y_all[anchor_idx]
        anchor_yaw = rob_yaw_all[anchor_idx]

        evidence = np.zeros_like(xx, dtype=np.float32)
        n_segments = 0

        for idx in indices:
            az_deg = profile_centers_deg[idx]
            pred_dist = pred_distance_mm[idx].copy()
            pred_dist[pred_presence_bin[idx] == 0] = np.nan
            probs = pred_presence_probs[idx]

            xw, yw = DataProcessor.robot2world(az_deg, pred_dist, rob_x_all[idx], rob_y_all[idx], rob_yaw_all[idx])
            xr, yr = DataProcessor.world2robot(xw, yw, anchor_x, anchor_y, anchor_yaw)

            for j in range(len(az_deg) - 1):
                if not (np.isfinite(xr[j]) and np.isfinite(yr[j]) and np.isfinite(xr[j + 1]) and np.isfinite(yr[j + 1])):
                    continue
                p0 = np.array([xr[j], yr[j]], dtype=float)
                p1 = np.array([xr[j + 1], yr[j + 1]], dtype=float)
                seg_vec = p1 - p0
                seg_len = float(np.linalg.norm(seg_vec))
                if seg_len < 1e-6:
                    continue
                weight = float(0.5 * (probs[j] + probs[j + 1]))
                if weight <= 0.0:
                    continue

                mid = 0.5 * (p0 + p1)
                tangent = (p1 - p0) / seg_len
                normal = np.array([-tangent[1], tangent[0]], dtype=float)
                half_len = 0.5 * seg_len

                local_margin = 3.0 * max(sigma_perp_mm, sigma_para_mm)
                bx0 = max(x_min, min(p0[0], p1[0]) - local_margin)
                bx1 = min(x_max, max(p0[0], p1[0]) + local_margin)
                by0 = max(y_min, min(p0[1], p1[1]) - local_margin)
                by1 = min(y_max, max(p0[1], p1[1]) + local_margin)

                ix0 = int(max(0, np.floor((bx0 - x_min) / grid_mm)))
                ix1 = int(min(len(x_grid) - 1, np.ceil((bx1 - x_min) / grid_mm)))
                iy0 = int(max(0, np.floor((by0 - y_min) / grid_mm)))
                iy1 = int(min(len(y_grid) - 1, np.ceil((by1 - y_min) / grid_mm)))

                sub_x = xx[iy0:iy1 + 1, ix0:ix1 + 1]
                sub_y = yy[iy0:iy1 + 1, ix0:ix1 + 1]
                dx = sub_x - mid[0]
                dy = sub_y - mid[1]

                d_para = dx * tangent[0] + dy * tangent[1]
                d_perp = dx * normal[0] + dy * normal[1]
                outside_para = np.maximum(np.abs(d_para) - half_len, 0.0)

                kernel = np.exp(
                    -0.5 * (d_perp / sigma_perp_mm) ** 2
                    -0.5 * (outside_para / sigma_para_mm) ** 2
                )
                evidence[iy0:iy1 + 1, ix0:ix1 + 1] += (weight * kernel).astype(np.float32)
                n_segments += 1

        evidence_norm = evidence / max(len(indices), 1)

        # 2-panel figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: robot-frame integrated evidence
        im = axes[0].imshow(
            evidence_norm,
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            cmap='magma',
            aspect='equal',
            vmin=heatmap_vmin,
            vmax=heatmap_vmax,
        )
        fig.colorbar(im, ax=axes[0], label='Integrated Evidence (a.u.)')
        axes[0].scatter([0], [0], c='cyan', s=50, label='Anchor robot')
        axes[0].arrow(0, 0, 250, 0, head_width=70, head_length=90, fc='cyan', ec='cyan', alpha=0.9)
        axes[0].set_title(f'Robot Frame Heatmap\nanchor idx={anchor_idx}, segments={n_segments}')
        axes[0].set_xlabel('Robot X (mm)')
        axes[0].set_ylabel('Robot Y (mm)')
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc='upper right')

        # Right: world trajectory + chunk poses and headings
        axes[1].plot(rob_x_all, rob_y_all, color='lightgray', linewidth=1.0, label='Full trajectory')
        axes[1].scatter(rob_x_all[indices], rob_y_all[indices], c='tab:blue', s=28, label='Chunk poses')
        yaw_r = np.deg2rad(rob_yaw_all[indices])
        u = 140.0 * np.cos(yaw_r)
        v = 140.0 * np.sin(yaw_r)
        axes[1].quiver(rob_x_all[indices], rob_y_all[indices], u, v, angles='xy', scale_units='xy', scale=1, width=0.003, color='tab:blue', alpha=0.8)
        # Ground-truth profiles in world coordinates for chunk indices.
        for idx in indices:
            gt_dist = real_distance_mm_raw[idx]
            gt_x_w, gt_y_w = DataProcessor.robot2world(
                profile_centers_deg[idx],
                gt_dist,
                rob_x_all[idx],
                rob_y_all[idx],
                rob_yaw_all[idx],
            )
            valid_gt = np.isfinite(gt_x_w) & np.isfinite(gt_y_w)
            if np.any(valid_gt):
                axes[1].plot(gt_x_w[valid_gt], gt_y_w[valid_gt], '-', color='lime', linewidth=1.0, alpha=0.45)
        axes[1].plot([], [], '-', color='lime', linewidth=1.0, alpha=0.75, label='GT profiles (raw)')
        axes[1].scatter([anchor_x], [anchor_y], c='red', s=70, label='Anchor pose')
        axes[1].arrow(anchor_x, anchor_y, 220.0 * np.cos(np.deg2rad(anchor_yaw)), 220.0 * np.sin(np.deg2rad(anchor_yaw)),
                      head_width=60, head_length=80, fc='red', ec='red', alpha=0.9)
        axes[1].set_title(f'World Frame Context\nindices [{win_start}, {win_end}]')
        axes[1].set_xlabel('World X (mm)')
        axes[1].set_ylabel('World Y (mm)')
        axes[1].axis('equal')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')

        plt.tight_layout()
        out_path = os.path.join(
            series_dir,
            f'{session_name}_seriesidx_{seq_idx:04d}_chunk_{win_start:04d}_{win_end:04d}_robotframe.png'
        )
        plt.savefig(out_path, dpi=220, bbox_inches='tight')
        plt.close(fig)

        # Save matrix data for downstream processing.
        heatmap_stack.append(evidence_norm.astype(np.float32))
        chunk_indices_stack.append(np.asarray(indices, dtype=np.int32))
        anchor_indices.append(int(anchor_idx))
        anchor_rob_x.append(float(anchor_x))
        anchor_rob_y.append(float(anchor_y))
        anchor_rob_yaw_deg.append(float(anchor_yaw))
        chunk_rob_x_stack.append(np.asarray(rob_x_all[indices], dtype=np.float32))
        chunk_rob_y_stack.append(np.asarray(rob_y_all[indices], dtype=np.float32))
        chunk_rob_yaw_deg_stack.append(np.asarray(rob_yaw_all[indices], dtype=np.float32))
        window_starts_saved.append(int(win_start))
        window_ends_saved.append(int(win_end))

        n_maps += 1

    if n_maps > 0:
        matrix_path = os.path.join(series_dir, f'{session_name}_robotframe_series_data.npz')
        np.savez_compressed(
            matrix_path,
            heatmaps=np.stack(heatmap_stack, axis=0),
            x_grid=x_grid.astype(np.float32),
            y_grid=y_grid.astype(np.float32),
            chunk_indices=np.stack(chunk_indices_stack, axis=0),
            window_start=np.asarray(window_starts_saved, dtype=np.int32),
            window_end=np.asarray(window_ends_saved, dtype=np.int32),
            anchor_index=np.asarray(anchor_indices, dtype=np.int32),
            anchor_rob_x=np.asarray(anchor_rob_x, dtype=np.float32),
            anchor_rob_y=np.asarray(anchor_rob_y, dtype=np.float32),
            anchor_rob_yaw_deg=np.asarray(anchor_rob_yaw_deg, dtype=np.float32),
            chunk_rob_x=np.stack(chunk_rob_x_stack, axis=0),
            chunk_rob_y=np.stack(chunk_rob_y_stack, axis=0),
            chunk_rob_yaw_deg=np.stack(chunk_rob_yaw_deg_stack, axis=0),
            grid_mm=np.float32(grid_mm),
            extent_mm=np.float32(extent_mm),
            sigma_perp_mm=np.float32(sigma_perp_mm),
            sigma_para_mm=np.float32(sigma_para_mm),
        )
        print(f"Saved robot-frame matrix data: {matrix_path}")

    print(f"Saved robot-frame chunk heatmap series: {n_maps} maps in {series_dir}")


def predict_profiles(model, y_scaler, sonar_data, batch_size, threshold, device):
    x_tensor = torch.FloatTensor(sonar_data).permute(0, 2, 1)  # (N, 2, 200)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False, pin_memory=(device.type == 'cuda'))

    all_presence_probs = []
    all_distance_scaled = []

    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device, non_blocking=True)
            presence_logits, distance_scaled = model(inputs)
            presence_probs = torch.sigmoid(presence_logits)

            all_presence_probs.append(presence_probs.cpu().numpy())
            all_distance_scaled.append(distance_scaled.cpu().numpy())

    presence_probs = np.concatenate(all_presence_probs, axis=0)
    presence_pred = (presence_probs >= threshold).astype(np.uint8)
    distance_scaled = np.concatenate(all_distance_scaled, axis=0)
    distance_mm = y_scaler.inverse_transform(distance_scaled)

    # Convenience output: distance masked by predicted presence
    distance_mm_masked = distance_mm.copy()
    distance_mm_masked[presence_pred == 0] = np.nan

    return presence_probs, presence_pred, distance_mm, distance_mm_masked


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    reset_prediction_output_dir(prediction_output_dir)
    print(f"Loading training artifacts from: {training_output_dir}")

    model, y_scaler, params = load_training_artifacts(training_output_dir, device)
    print("Loaded model + scaler artifacts.")

    print(f"Loading session data: {session_to_predict}")
    sonar_data, real_profiles_mm, profile_centers_deg, metadata = load_session_data(
        session_to_predict,
        profile_opening_angle=float(params['profile_opening_angle']),
        profile_steps=int(params['profile_steps']),
    )
    print(f"Sonar shape used for prediction: {sonar_data.shape}")
    
    # Match training target visibility: hide real points beyond training distance threshold.
    distance_threshold = float(params.get('distance_threshold', np.inf))
    real_profiles_mm_clipped = real_profiles_mm.copy()
    out_of_range_mask = (real_profiles_mm_clipped < 0) | (real_profiles_mm_clipped > distance_threshold)
    real_profiles_mm_clipped[out_of_range_mask] = np.nan

    if sonar_data.shape[0] == 0:
        raise ValueError("No valid sonar samples found for prediction.")

    # Use training-selected threshold unless user overrides it.
    presence_threshold = (
        float(params.get('presence_threshold', 0.5))
        if presence_threshold_override is None
        else float(presence_threshold_override)
    )
    print(f"Using presence threshold: {presence_threshold:.2f}")

    print("Running predictions...")
    presence_probs, presence_pred, distance_mm, distance_mm_masked = predict_profiles(
        model=model,
        y_scaler=y_scaler,
        sonar_data=sonar_data,
        batch_size=prediction_batch_size,
        threshold=presence_threshold,
        device=device,
    )

    out_npz = os.path.join(prediction_output_dir, f'predictions_{session_to_predict}.npz')
    np.savez_compressed(
        out_npz,
        session=session_to_predict,
        presence_threshold=presence_threshold,
        profile_steps=params['profile_steps'],
        profile_opening_angle=params['profile_opening_angle'],
        presence_probs=presence_probs,
        presence_pred=presence_pred,
        distance_mm=distance_mm,
        distance_mm_masked=distance_mm_masked,
        rob_x=metadata['rob_x'],
        rob_y=metadata['rob_y'],
        rob_yaw_deg=metadata['rob_yaw_deg'],
        quadrants=metadata['quadrants'],
        kept_indices=metadata['kept_indices'],
    )

    # Lightweight metadata summary
    summary = {
        'session': session_to_predict,
        'training_output_dir': training_output_dir,
        'prediction_output_dir': prediction_output_dir,
        'prediction_file': out_npz,
        'samples_original': metadata['total_samples_original'],
        'samples_used': metadata['total_samples_used'],
        'profile_steps': int(params['profile_steps']),
        'profile_opening_angle': float(params['profile_opening_angle']),
        'presence_threshold': float(presence_threshold),
        'distance_range_mm': [float(np.nanmin(distance_mm)), float(np.nanmax(distance_mm))],
        'mean_presence_probability': float(np.mean(presence_probs)),
        'predicted_presence_fraction': float(np.mean(presence_pred)),
    }
    summary_path = os.path.join(prediction_output_dir, f'predictions_{session_to_predict}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if save_plots:
        selected_indices = resolve_plot_indices(
            n_samples=sonar_data.shape[0],
            indices=plot_indices,
            index_range=plot_index_range,
        )
        plot_selected_predictions(
            session_name=session_to_predict,
            out_dir=prediction_output_dir,
            selected_indices=selected_indices,
            metadata=metadata,
            profile_centers_deg=profile_centers_deg,
            real_distance_mm=real_profiles_mm_clipped,
            pred_distance_mm=distance_mm,
            pred_presence_bin=presence_pred,
            verbose=False,
        )
        if save_overlay_plot:
            plot_overlay_predictions(
                session_name=session_to_predict,
                out_dir=prediction_output_dir,
                selected_indices=selected_indices,
                metadata=metadata,
                profile_centers_deg=profile_centers_deg,
                real_distance_mm=real_profiles_mm_clipped,
                pred_distance_mm=distance_mm,
                pred_presence_bin=presence_pred,
                verbose=True,
            )
        if save_integrated_profile_map:
            create_segment_evidence_map(
                session_name=session_to_predict,
                out_dir=prediction_output_dir,
                selected_indices=selected_indices,
                metadata=metadata,
                profile_centers_deg=profile_centers_deg,
                pred_distance_mm=distance_mm,
                pred_presence_bin=presence_pred,
                pred_presence_probs=presence_probs,
                real_distance_mm_raw=real_profiles_mm,
                grid_mm=integration_grid_mm,
                sigma_perp_mm=integration_sigma_perp_mm,
                sigma_para_mm=integration_sigma_para_mm,
                margin_mm=integration_margin_mm,
            )
        if save_chunk_heatmap_series:
            create_chunk_heatmap_series(
                session_name=session_to_predict,
                out_dir=prediction_output_dir,
                metadata=metadata,
                profile_centers_deg=profile_centers_deg,
                real_distance_mm_raw=real_profiles_mm,
                pred_distance_mm=distance_mm,
                pred_presence_bin=presence_pred,
                pred_presence_probs=presence_probs,
                n_samples=sonar_data.shape[0],
                start_idx=chunk_series_start,
                end_idx=chunk_series_end,
                shift=chunk_series_shift,
                chunk_size=chunk_series_size,
                subdir=chunk_series_subdir,
                grid_mm=integration_grid_mm,
                sigma_perp_mm=integration_sigma_perp_mm,
                sigma_para_mm=integration_sigma_para_mm,
                margin_mm=integration_margin_mm,
            )
        if save_robot_frame_chunk_heatmap_series:
            create_robot_frame_chunk_heatmap_series(
                session_name=session_to_predict,
                out_dir=prediction_output_dir,
                metadata=metadata,
                profile_centers_deg=profile_centers_deg,
                real_distance_mm_raw=real_profiles_mm,
                pred_distance_mm=distance_mm,
                pred_presence_bin=presence_pred,
                pred_presence_probs=presence_probs,
                n_samples=sonar_data.shape[0],
                start_idx=chunk_series_start,
                end_idx=chunk_series_end,
                shift=chunk_series_shift,
                chunk_size=chunk_series_size,
                subdir=robot_chunk_series_subdir,
                extent_mm=robot_frame_extent_mm,
                grid_mm=robot_frame_grid_mm,
                sigma_perp_mm=integration_sigma_perp_mm,
                sigma_para_mm=integration_sigma_para_mm,
                heatmap_vmin=robot_frame_heatmap_vmin,
                heatmap_vmax=robot_frame_heatmap_vmax,
            )

    print(f"Saved predictions to: {out_npz}")
    print(f"Saved summary to: {summary_path}")


if __name__ == '__main__':
    main()
