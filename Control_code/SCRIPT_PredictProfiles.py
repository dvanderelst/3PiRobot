#
# SCRIPT_PredictProfiles: Load trained two-head model and predict profiles for one session.
#
# This script loads a training output folder and predicts:
# 1) presence probabilities/binary predictions per azimuth bin
# 2) distance predictions per azimuth bin (in mm)
#

import json
import os

import joblib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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
):
    if len(selected_indices) == 0:
        print("No selected indices to plot.")
        return

    plot_dir = os.path.join(out_dir, f'prediction_plots_{session_name}')
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

        plot_path = os.path.join(plot_dir, f'profile_world_idx_{idx:04d}.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")


def plot_overlay_predictions(
    session_name,
    out_dir,
    selected_indices,
    metadata,
    profile_centers_deg,
    real_distance_mm,
    pred_distance_mm,
    pred_presence_bin,
):
    """Plot all selected indices overlaid in world coordinates with matching colors."""
    if len(selected_indices) == 0:
        return

    plot_dir = os.path.join(out_dir, f'prediction_plots_{session_name}')
    os.makedirs(plot_dir, exist_ok=True)

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

    overlay_path = os.path.join(plot_dir, 'profile_world_overlay_selected.png')
    plt.savefig(overlay_path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved overlay plot: {overlay_path}")


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
    os.makedirs(prediction_output_dir, exist_ok=True)
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
            )

    print(f"Saved predictions to: {out_npz}")
    print(f"Saved summary to: {summary_path}")


if __name__ == '__main__':
    main()
