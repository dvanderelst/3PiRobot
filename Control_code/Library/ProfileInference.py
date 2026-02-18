import json
import os
import sys

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Library import DataProcessor


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

        self.presence_head = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, profile_steps),
        )

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

    # Backward-compatible load: scaler may be serialized as __main__.DistanceScaler.
    main_mod = sys.modules.get('__main__')
    had_distance_scaler = hasattr(main_mod, 'DistanceScaler') if main_mod is not None else False
    old_distance_scaler = getattr(main_mod, 'DistanceScaler', None) if had_distance_scaler else None
    try:
        if main_mod is not None:
            setattr(main_mod, 'DistanceScaler', DistanceScaler)
        y_scaler = joblib.load(scaler_path)
    finally:
        if main_mod is not None:
            if had_distance_scaler:
                setattr(main_mod, 'DistanceScaler', old_distance_scaler)
            else:
                try:
                    delattr(main_mod, 'DistanceScaler')
                except AttributeError:
                    pass
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
