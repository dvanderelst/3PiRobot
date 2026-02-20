#
# SCRIPT_TrainProfilesRadial: Two-headed CNN with radial occupancy target
#
# Presence head: per-azimuth wall presence probability
# Radial head: per-azimuth radial occupancy logits over range bins
#

# ============================================
# CONFIGURATION SETTINGS
# ============================================

# Data Configuration
sessions = ['sessionB01', 'sessionB02', 'sessionB03', 'sessionB04']
profile_method = 'ray_center'  # target profile extraction method

# Spatial Splitting Configuration
train_quadrants = [0, 1, 2]
test_quadrant = 3

# Profile Parameters
profile_opening_angle = 60
profile_steps = 21

# Distance Threshold
distance_threshold = 2000.0

# Radial target parameters
radial_bins = 32
radial_sigma_mm = 75.0
radial_loss_weight = 1.0
radial_example_plots_count = 6
radial_example_samples_per_plot = 4

# Training Configuration
validation_split = 0.3
batch_size = 32
epochs = 100
patience = 10
seed = 42
debug = False
use_sample_weighting = True
sample_weight_power = 0.5
sample_weight_min = 0.25
sample_weight_max = 4.0
sample_weight_feature_bins = 8
sample_weight_shape_segments = 8

# Plot Configuration
plot_format = 'png'
plot_dpi = 300
show_plots = True

# Output Configuration
output_dir = 'TrainingRadial'

# Model Architecture
conv_filters = [32, 64, 128]
kernel_size = 5
pool_size = 2
dropout_rate = 0.3
l2_reg = 0.001
learning_rate = 0.001

# ============================================
# IMPORTS
# ============================================

import json
import os
import time

import matplotlib
if not os.environ.get('DISPLAY') and os.name != 'nt':
    matplotlib.use('Agg')

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from Library import DataProcessor


os.makedirs(output_dir, exist_ok=True)


def save_plot(filename):
    if plot_format in ['png', 'both']:
        png_filename = f"{output_dir}/{filename}.png"
        plt.savefig(png_filename, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved PNG plot: {png_filename}")

    if plot_format in ['svg', 'both']:
        svg_filename = f"{output_dir}/{filename}.svg"
        plt.savefig(svg_filename, bbox_inches='tight', facecolor='white')
        print(f"Saved SVG plot: {svg_filename}")


def maybe_show_plot():
    if show_plots and matplotlib.get_backend().lower() != 'agg':
        plt.show()
    else:
        plt.close()


def write_training_readme():
    readme_path = f"{output_dir}/README.md"
    txt = f"""# TrainingRadial Output

Artifacts from `SCRIPT_TrainProfilesRadial.py`.

## Core Artifacts

- `best_model_pytorch.pth`
  Best model checkpoint selected by validation loss.
- `training_params.json`
  Training/model parameters and radial target metadata for inference.

## Evaluation Tables

- `presence_confusion_test_set.csv`
- `presence_confusion_training_set.csv`

## Plots

- `training_curves.*`
- `bin_scatter_plots_test_set.*`
- `bin_scatter_plots_training_set.*`
- `radial_distribution_samples_test_set_batch_XX.*`
- `presence_confusion_test_set.*`
- `presence_confusion_training_set.*`
- `test_samples.*`
- `spatial_errors.*`

## Notes

- Model target type is `radial` and distances are recovered from predicted radial occupancy.
- No `y_scaler.joblib` is required for this model type.
"""
    with open(readme_path, 'w') as f:
        f.write(txt)
    print(f"Saved training readme: {readme_path}")


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Reproducibility
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TwoHeadedSonarRadialCNN(nn.Module):
    def __init__(self, input_shape, profile_steps, radial_bins, range_centers_mm):
        super(TwoHeadedSonarRadialCNN, self).__init__()
        self.profile_steps = int(profile_steps)
        self.radial_bins = int(radial_bins)
        self.register_buffer('range_centers_mm', torch.as_tensor(range_centers_mm, dtype=torch.float32))

        self.shared = nn.Sequential(
            nn.Conv1d(2, conv_filters[0], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout_rate),

            nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout_rate),

            nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout_rate),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            z = self.shared(dummy_input)
            flat = z.view(1, -1).shape[1]

        self.presence_head = nn.Sequential(
            nn.Linear(flat, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, profile_steps),
        )

        self.radial_head = nn.Sequential(
            nn.Linear(flat, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, profile_steps * radial_bins),
        )

    def forward(self, x):
        features = self.shared(x)
        features = features.view(features.size(0), -1)
        presence_logits = self.presence_head(features)
        radial_logits = self.radial_head(features).view(-1, self.profile_steps, self.radial_bins)
        return presence_logits, radial_logits











def get_range_centers(distance_threshold_mm, n_bins):
    return np.linspace(0.0, float(distance_threshold_mm), int(n_bins), dtype=np.float32)





def build_radial_targets(distance_mm, presence, range_centers_mm, sigma_mm):
    """
    distance_mm: (N, K)
    presence: (N, K)
    return: (N, K, R)
    """
    sigma = max(float(sigma_mm), 1e-3)
    r = range_centers_mm.reshape(1, 1, -1)
    d = distance_mm[:, :, None]
    gauss = np.exp(-0.5 * ((r - d) / sigma) ** 2)
    targets = gauss * presence[:, :, None]
    return targets.astype(np.float32)


def compute_profile_rarity_weights(distance_mm, presence):
    """
    Build per-sample weights from centered profile-shape signatures.
    Distances are centered per sample, so rarity focuses on shape rather than
    absolute range from the robot.
    """
    n = int(distance_mm.shape[0])
    k = int(distance_mm.shape[1])
    if n < 1:
        return np.ones((0,), dtype=np.float32)

    pres = presence.astype(np.float32)
    dist = distance_mm.astype(np.float32)
    present_counts = np.sum(pres, axis=1)
    present_any = present_counts > 0.5

    dist_masked = np.where(pres > 0.5, dist, np.nan)
    mean_present = np.nanmean(dist_masked, axis=1)
    mean_present = np.nan_to_num(mean_present, nan=0.0, posinf=0.0, neginf=0.0)

    # Center by per-profile mean so absolute range is removed from the rarity signal.
    centered = np.where(pres > 0.5, dist - mean_present[:, None], 0.0)
    std_present = np.nanstd(np.where(pres > 0.5, centered, np.nan), axis=1)
    std_present = np.nan_to_num(std_present, nan=1.0, posinf=1.0, neginf=1.0)
    std_present = np.maximum(std_present, 1e-3)
    normalized = np.where(pres > 0.5, centered / std_present[:, None], 0.0)

    n_seg = int(max(2, min(sample_weight_shape_segments, k)))
    seg_edges = np.linspace(0, k, n_seg + 1, dtype=np.int32)
    seg_shape = np.zeros((n, n_seg), dtype=np.float32)
    seg_coverage = np.zeros((n, n_seg), dtype=np.float32)
    for s in range(n_seg):
        a = int(seg_edges[s])
        b = int(seg_edges[s + 1])
        seg_pres = pres[:, a:b]
        seg_cov = seg_pres.mean(axis=1)
        seg_sum = (normalized[:, a:b] * seg_pres).sum(axis=1)
        seg_cnt = seg_pres.sum(axis=1)
        seg_mean = np.divide(seg_sum, np.maximum(seg_cnt, 1e-6))
        seg_shape[:, s] = np.where(seg_cnt > 0.0, seg_mean, 0.0)
        seg_coverage[:, s] = seg_cov

    present_frac = np.clip(present_counts / max(float(k), 1.0), 0.0, 1.0)
    shape_clip = np.clip(seg_shape, -3.0, 3.0)
    cov_clip = np.clip(seg_coverage, 0.0, 1.0)

    nb = int(max(2, sample_weight_feature_bins))
    q_shape = np.minimum((((shape_clip + 3.0) / 6.0) * (nb - 1)).astype(np.int32), nb - 1)
    q_cov = np.minimum((cov_clip * (nb - 1)).astype(np.int32), nb - 1)
    q_present = np.minimum((present_frac * (nb - 1)).astype(np.int32), nb - 1).reshape(-1, 1)

    feature_keys = np.concatenate([q_present, q_cov, q_shape], axis=1).astype(np.int32)
    feature_keys[~present_any, :] = 0
    _, inverse, counts = np.unique(feature_keys, axis=0, return_inverse=True, return_counts=True)
    key_counts = counts[inverse]

    raw = np.power(np.maximum(key_counts, 1), -float(sample_weight_power))
    raw /= np.mean(raw)
    clipped = np.clip(raw, float(sample_weight_min), float(sample_weight_max))
    weights = clipped / np.mean(clipped)
    return weights.astype(np.float32)


def radial_logits_to_distance_mm(radial_logits, range_centers_mm):
    probs = torch.softmax(radial_logits, dim=-1)
    r = torch.as_tensor(range_centers_mm, device=radial_logits.device, dtype=radial_logits.dtype).view(1, 1, -1)
    return torch.sum(probs * r, dim=-1)


def load_and_prepare_data():
    print('Loading data...')
    dc = DataProcessor.DataCollection(sessions)
    sonar_data = dc.load_sonar(flatten=False)
    profiles_data, _ = dc.load_profiles(
        opening_angle=profile_opening_angle,
        steps=profile_steps,
        profile_method=profile_method,
    )

    quadrants = dc.quadrants
    rob_x = dc.rob_x
    rob_y = dc.rob_y

    nan_mask = ~np.isnan(profiles_data).any(axis=1)
    sonar_data = sonar_data[nan_mask]
    profiles_data = profiles_data[nan_mask]
    quadrants = quadrants[nan_mask]
    rob_x = rob_x[nan_mask]
    rob_y = rob_y[nan_mask]

    print(f'   Sonar shape: {sonar_data.shape}')
    print(f'   Profiles shape: {profiles_data.shape}')
    print(f'   Quadrants shape: {quadrants.shape}')
    return sonar_data, profiles_data, quadrants, rob_x, rob_y


def create_spatial_split(sonar_data, profiles_data, quadrants):
    train_mask = np.isin(quadrants, train_quadrants)
    test_mask = (quadrants == test_quadrant)

    X_train, X_val, y_train, y_val = train_test_split(
        sonar_data[train_mask],
        profiles_data[train_mask],
        test_size=validation_split,
        random_state=seed,
    )

    X_test = sonar_data[test_mask]
    y_test = profiles_data[test_mask]
    return X_train, X_val, X_test, y_train, y_val, y_test, train_mask, test_mask


def preprocess_data(X_train, X_val, X_test, y_train, y_val, y_test, range_centers_mm):
    y_train_presence = (y_train <= distance_threshold).astype(np.float32)
    y_val_presence = (y_val <= distance_threshold).astype(np.float32)
    y_test_presence = (y_test <= distance_threshold).astype(np.float32)

    y_train_distance = np.clip(y_train, 0.0, distance_threshold).astype(np.float32)
    y_val_distance = np.clip(y_val, 0.0, distance_threshold).astype(np.float32)
    y_test_distance = np.clip(y_test, 0.0, distance_threshold).astype(np.float32)

    y_train_radial = build_radial_targets(y_train_distance, y_train_presence, range_centers_mm, radial_sigma_mm)
    y_val_radial = build_radial_targets(y_val_distance, y_val_presence, range_centers_mm, radial_sigma_mm)
    y_test_radial = build_radial_targets(y_test_distance, y_test_presence, range_centers_mm, radial_sigma_mm)

    X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1)
    X_val_tensor = torch.FloatTensor(X_val).permute(0, 2, 1)
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 2, 1)

    yp_train_tensor = torch.FloatTensor(y_train_presence)
    yp_val_tensor = torch.FloatTensor(y_val_presence)
    yp_test_tensor = torch.FloatTensor(y_test_presence)

    yr_train_tensor = torch.FloatTensor(y_train_radial)
    yr_val_tensor = torch.FloatTensor(y_val_radial)
    yr_test_tensor = torch.FloatTensor(y_test_radial)

    yd_train_tensor = torch.FloatTensor(y_train_distance)
    yd_val_tensor = torch.FloatTensor(y_val_distance)
    yd_test_tensor = torch.FloatTensor(y_test_distance)
    if use_sample_weighting:
        train_sample_weights = compute_profile_rarity_weights(y_train_distance, y_train_presence)
    else:
        train_sample_weights = np.ones((len(y_train_distance),), dtype=np.float32)
    w_train_tensor = torch.FloatTensor(train_sample_weights)
    w_val_tensor = torch.ones(len(y_val_distance), dtype=torch.float32)
    w_test_tensor = torch.ones(len(y_test_distance), dtype=torch.float32)

    class RadialDataset(torch.utils.data.Dataset):
        def __init__(self, x, yp, yr, yd, w=None):
            self.x = x
            self.yp = yp
            self.yr = yr
            self.yd = yd
            self.w = w

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            if self.w is None:
                return self.x[idx], self.yp[idx], self.yr[idx], self.yd[idx]
            return self.x[idx], self.yp[idx], self.yr[idx], self.yd[idx], self.w[idx]

    train_ds = RadialDataset(
        X_train_tensor,
        yp_train_tensor,
        yr_train_tensor,
        yd_train_tensor,
        w=w_train_tensor if use_sample_weighting else None,
    )
    val_ds = RadialDataset(
        X_val_tensor,
        yp_val_tensor,
        yr_val_tensor,
        yd_val_tensor,
        w=w_val_tensor if use_sample_weighting else None,
    )
    test_ds = RadialDataset(
        X_test_tensor,
        yp_test_tensor,
        yr_test_tensor,
        yd_test_tensor,
        w=w_test_tensor if use_sample_weighting else None,
    )

    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    if use_sample_weighting:
        print(
            "Sample weighting enabled "
            f"(mean={float(train_sample_weights.mean()):.3f}, "
            f"min={float(train_sample_weights.min()):.3f}, "
            f"max={float(train_sample_weights.max()):.3f})"
        )

    return train_loader, val_loader, test_loader





def train_model(model, train_loader, val_loader, range_centers_mm):
    presence_criterion = nn.BCEWithLogitsLoss(reduction='none')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    presence_losses = []
    radial_losses = []

    for epoch in range(epochs):
        model.train()
        run_train = 0.0
        run_presence = 0.0
        run_radial = 0.0

        for batch_data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} - Training'):
            if use_sample_weighting and len(batch_data) == 5:
                x, yp, yr, _yd, sample_w = batch_data
            else:
                x, yp, yr, _yd = batch_data
                sample_w = torch.ones(x.size(0), dtype=torch.float32)
            x = x.to(device, non_blocking=True)
            yp = yp.to(device, non_blocking=True)
            yr = yr.to(device, non_blocking=True)
            sample_w = sample_w.to(device, non_blocking=True)

            optimizer.zero_grad()
            presence_logits, radial_logits = model(x)

            presence_per_bin = presence_criterion(presence_logits, yp)
            presence_per_sample = presence_per_bin.mean(dim=1)
            presence_loss = (presence_per_sample * sample_w).sum() / sample_w.sum().clamp(min=1e-12)

            present_mask = (yp > 0.5).float()  # (B,K)
            present_count_per_sample = present_mask.sum(dim=1)
            valid_present = present_count_per_sample > 0
            if torch.any(valid_present):
                # Normalize gaussian targets to a probability distribution over range bins.
                target_dist = yr / yr.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                log_probs = torch.log_softmax(radial_logits, dim=-1)
                per_bin_kl = -(target_dist * log_probs).sum(dim=-1)  # (B,K)
                per_sample_kl = (per_bin_kl * present_mask).sum(dim=1) / present_count_per_sample.clamp(min=1e-12)
                radial_loss = (
                    per_sample_kl[valid_present] * sample_w[valid_present]
                ).sum() / sample_w[valid_present].sum().clamp(min=1e-12)
            else:
                radial_loss = radial_logits.sum() * 0.0

            loss = presence_loss + radial_loss_weight * radial_loss
            loss.backward()
            optimizer.step()

            n = x.size(0)
            run_train += loss.item() * n
            run_presence += presence_loss.item() * n
            run_radial += radial_loss.item() * n

        train_loss = run_train / len(train_loader.dataset)
        train_presence = run_presence / len(train_loader.dataset)
        train_radial = run_radial / len(train_loader.dataset)

        train_losses.append(train_loss)
        presence_losses.append(train_presence)
        radial_losses.append(train_radial)

        model.eval()
        run_val = 0.0
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} - Validation'):
                if use_sample_weighting and len(batch_data) == 5:
                    x, yp, yr, _yd, _sample_w = batch_data
                else:
                    x, yp, yr, _yd = batch_data
                x = x.to(device, non_blocking=True)
                yp = yp.to(device, non_blocking=True)
                yr = yr.to(device, non_blocking=True)

                presence_logits, radial_logits = model(x)
                presence_per_bin = presence_criterion(presence_logits, yp)
                presence_loss = presence_per_bin.mean()
                present_mask = (yp > 0.5).float()
                present_count = present_mask.sum()
                if present_count.item() > 0:
                    target_dist = yr / yr.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                    log_probs = torch.log_softmax(radial_logits, dim=-1)
                    per_bin_kl = -(target_dist * log_probs).sum(dim=-1)
                    radial_loss = (per_bin_kl * present_mask).sum() / present_count
                else:
                    radial_loss = torch.tensor(0.0, device=x.device)
                loss = presence_loss + radial_loss_weight * radial_loss
                run_val += loss.item() * x.size(0)

        val_loss = run_val / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train={train_loss:.6f} "
            f"(Presence={train_presence:.6f}, Radial={train_radial:.6f}) "
            f"Val={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'{output_dir}/best_model_pytorch.pth')
            print(f"Saved best model to {output_dir}/best_model_pytorch.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return train_losses, val_losses, presence_losses, radial_losses


def collect_predictions(model, data_loader, range_centers_mm):
    model.eval()
    all_presence_probs = []
    all_distance_preds = []
    all_presence_targets = []
    all_distance_targets = []

    with torch.no_grad():
        for batch_data in data_loader:
            # Handle both weighted and unweighted batches
            if use_sample_weighting and len(batch_data) == 5:
                x, yp, _yr, yd, _weights = batch_data
            else:
                x, yp, _yr, yd = batch_data

            x = x.to(device, non_blocking=True)
            yp = yp.to(device, non_blocking=True)
            yd = yd.to(device, non_blocking=True)

            presence_logits, radial_logits = model(x)
            presence_probs = torch.sigmoid(presence_logits)
            distance_mm = radial_logits_to_distance_mm(radial_logits, range_centers_mm)

            all_presence_probs.append(presence_probs.cpu().numpy())
            all_distance_preds.append(distance_mm.cpu().numpy())
            all_presence_targets.append(yp.cpu().numpy())
            all_distance_targets.append(yd.cpu().numpy())

    presence_probs = np.concatenate(all_presence_probs, axis=0)
    distance_preds_mm = np.concatenate(all_distance_preds, axis=0)
    presence_targets = np.concatenate(all_presence_targets, axis=0)
    distance_targets_mm = np.concatenate(all_distance_targets, axis=0)
    return presence_probs, distance_preds_mm, presence_targets, distance_targets_mm


def find_best_presence_threshold(model, val_loader, range_centers_mm):
    presence_probs, _, presence_targets, _ = collect_predictions(model, val_loader, range_centers_mm)

    probs = presence_probs.reshape(-1)
    targets = (presence_targets.reshape(-1) > 0.5)

    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.linspace(0.05, 0.95, 91):
        preds = probs >= thr
        tp = np.sum(preds & targets)
        fp = np.sum(preds & ~targets)
        fn = np.sum(~preds & targets)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

        if (f1 > best_f1) or (np.isclose(f1, best_f1) and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_thr = float(thr)
            best_f1 = float(f1)

    print(f"Best validation presence threshold: {best_thr:.2f} (F1={best_f1:.4f})")
    return best_thr


def create_scatter_plots(model, data_loader, range_centers_mm, dataset_name):
    print(f"Creating scatter plots for {dataset_name}...")
    presence_probs, distance_preds_mm, presence_targets, distance_targets_mm = collect_predictions(
        model, data_loader, range_centers_mm
    )

    present_mask = presence_targets > 0.5

    plt.figure(figsize=(15, 10))
    for bin_idx in range(min(12, presence_probs.shape[1])):
        plt.subplot(4, 3, bin_idx + 1)
        mask = present_mask[:, bin_idx]
        if np.sum(mask) > 0:
            x = distance_targets_mm[mask, bin_idx]
            y = distance_preds_mm[mask, bin_idx]
            plt.scatter(x, y, alpha=0.3, s=10)
            target_range = [np.min(x), np.max(x)]
            plt.plot(target_range, target_range, 'r--', alpha=0.5)

            corr = np.corrcoef(y, x)[0, 1]
            mae = np.mean(np.abs(y - x))
            mse = np.mean((y - x) ** 2)

            plt.title(f'Bin {bin_idx}: corr={corr:.3f}')
            plt.text(0.05, 0.95, f'MAE={mae:.1f}', transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)
            plt.text(0.05, 0.88, f'MSE={mse:.1f}', transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)
        else:
            plt.title(f'Bin {bin_idx}: no data')
        plt.xlabel('True (mm)')
        plt.ylabel('Predicted (mm)')
        plt.grid(True, alpha=0.3)

    plt.suptitle(f'Distance Prediction Scatter Plots by Azimuth Bin - {dataset_name}', fontsize=16)
    plt.tight_layout()
    save_plot(f'bin_scatter_plots_{dataset_name.lower().replace(" ", "_")}')
    maybe_show_plot()


def create_presence_confusion_plots(model, data_loader, range_centers_mm, dataset_name, threshold=0.5):
    presence_probs, _, presence_targets_raw, _ = collect_predictions(model, data_loader, range_centers_mm)
    presence_targets = presence_targets_raw.astype(bool)
    presence_preds = (presence_probs >= threshold)

    tp = np.sum(presence_preds & presence_targets, axis=0)
    fp = np.sum(presence_preds & ~presence_targets, axis=0)
    tn = np.sum(~presence_preds & ~presence_targets, axis=0)
    fn = np.sum(~presence_preds & presence_targets, axis=0)

    confusion_df = pd.DataFrame({
        'bin': np.arange(presence_targets.shape[1]),
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
    })

    confusion_df['precision'] = confusion_df['TP'] / np.maximum(confusion_df['TP'] + confusion_df['FP'], 1)
    confusion_df['recall'] = confusion_df['TP'] / np.maximum(confusion_df['TP'] + confusion_df['FN'], 1)
    confusion_df['specificity'] = confusion_df['TN'] / np.maximum(confusion_df['TN'] + confusion_df['FP'], 1)
    confusion_df['f1'] = (
        2 * confusion_df['precision'] * confusion_df['recall']
    ) / np.maximum(confusion_df['precision'] + confusion_df['recall'], 1e-12)

    csv_name = f"{output_dir}/presence_confusion_{dataset_name.lower().replace(' ', '_')}.csv"
    confusion_df.to_csv(csv_name, index=False)
    print(f"Saved presence confusion table: {csv_name}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    count_matrix = confusion_df[['TP', 'FP', 'TN', 'FN']].values
    sns.heatmap(
        count_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        yticklabels=[f"Bin {i}" for i in confusion_df['bin']],
        xticklabels=['TP', 'FP', 'TN', 'FN'],
    )
    plt.title(f'Presence Confusion Counts - {dataset_name}')

    plt.subplot(1, 2, 2)
    row_sums = np.maximum(count_matrix.sum(axis=1, keepdims=True), 1)
    norm_matrix = count_matrix / row_sums
    sns.heatmap(
        norm_matrix,
        annot=True,
        fmt='.2f',
        cmap='Greens',
        vmin=0,
        vmax=1,
        yticklabels=[f"Bin {i}" for i in confusion_df['bin']],
        xticklabels=['TP', 'FP', 'TN', 'FN'],
    )
    plt.title(f'Presence Confusion Fractions - {dataset_name}')

    plt.tight_layout()
    save_plot(f'presence_confusion_{dataset_name.lower().replace(" ", "_")}')
    maybe_show_plot()


def create_radial_distribution_examples(
    model,
    data_loader,
    range_centers_mm,
    dataset_name,
    num_plots=6,
    samples_per_plot=4,
    random_seed=42,
):
    """Visualize multiple random batches of predicted/target radial distributions."""
    model.eval()
    all_presence_probs = []
    all_radial_probs = []
    all_presence_targets = []
    all_radial_targets = []
    all_distance_targets = []
    with torch.no_grad():
        for batch_data in data_loader:
            # Handle both weighted and unweighted batches
            if use_sample_weighting and len(batch_data) == 5:
                x, yp, yr, yd, _weights = batch_data
            else:
                x, yp, yr, yd = batch_data

            x = x.to(device, non_blocking=True)
            presence_logits, radial_logits = model(x)
            all_presence_probs.append(torch.sigmoid(presence_logits).cpu().numpy())
            all_radial_probs.append(torch.softmax(radial_logits, dim=-1).cpu().numpy())
            all_presence_targets.append(yp.cpu().numpy())
            all_radial_targets.append(yr.cpu().numpy())
            all_distance_targets.append(yd.cpu().numpy())

    if len(all_presence_probs) == 0:
        print(f'No samples for radial distribution plotting: {dataset_name}')
        return

    presence_probs_all = np.concatenate(all_presence_probs, axis=0)
    radial_probs_all = np.concatenate(all_radial_probs, axis=0)
    presence_targets_all = np.concatenate(all_presence_targets, axis=0)
    radial_targets_all = np.concatenate(all_radial_targets, axis=0)
    distance_targets_all = np.concatenate(all_distance_targets, axis=0)

    n_total = presence_probs_all.shape[0]
    if n_total < 1:
        print(f'No samples for radial distribution plotting: {dataset_name}')
        return

    rng = np.random.default_rng(int(random_seed))
    for pidx in range(int(max(1, num_plots))):
        n_pick = int(max(1, samples_per_plot))
        replace = (n_total < n_pick)
        chosen = rng.choice(n_total, size=n_pick, replace=replace)

        fig, axes = plt.subplots(n_pick, 4, figsize=(21, 3.8 * n_pick))
        if n_pick == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, sidx in enumerate(chosen):
            presence_probs = presence_probs_all[sidx]
            radial_probs = radial_probs_all[sidx]
            yp = presence_targets_all[sidx]
            yr = radial_targets_all[sidx]
            yd = distance_targets_all[sidx]
            yr_norm = yr / np.maximum(yr.sum(axis=1, keepdims=True), 1e-12)
            pred_occ = radial_probs * presence_probs[:, None]

            ax0 = axes[row, 0]
            ax0.imshow(
                pred_occ,
                aspect='auto',
                origin='lower',
                extent=[range_centers_mm[0], range_centers_mm[-1], 0, profile_steps - 1],
                cmap='magma',
                vmin=0.0,
                vmax=np.max(pred_occ),
            )
            ax0.set_title(f'Pred Occupancy Dist (presence-weighted) - idx {sidx}')
            ax0.set_xlabel('Range (mm)')
            ax0.set_ylabel('Azimuth bin')

            ax1 = axes[row, 1]
            ax1.imshow(
                yr_norm,
                aspect='auto',
                origin='lower',
                extent=[range_centers_mm[0], range_centers_mm[-1], 0, profile_steps - 1],
                cmap='magma',
                vmin=0.0,
                vmax=np.max(yr_norm),
            )
            ax1.set_title('Target Radial Dist')
            ax1.set_xlabel('Range (mm)')
            ax1.set_ylabel('Azimuth bin')

            ax2 = axes[row, 2]
            pred_distance = np.sum(radial_probs * range_centers_mm.reshape(1, -1), axis=1)
            ax2.plot(yd, 'g-', label='True dist (mm)')
            ax2.plot(pred_distance, 'm--', label='Pred dist (mm)')
            ax2.set_title('Distance outputs')
            ax2.set_ylim(0, distance_threshold * 1.05)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8, loc='upper right')

            ax3 = axes[row, 3]
            ax3.plot(yp, 'b-', label='True presence')
            ax3.plot(presence_probs, 'r--', label='Pred presence')
            ax3.set_title('Presence outputs')
            ax3.set_ylim(-0.05, 1.05)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        save_plot(f'radial_distribution_samples_{dataset_name.lower().replace(" ", "_")}_batch_{pidx:02d}')
        maybe_show_plot()


def main():
    print('Starting radial profile training')

    sonar_data, profiles_data, quadrants, rob_x, rob_y = load_and_prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test, train_mask, test_mask = create_spatial_split(
        sonar_data, profiles_data, quadrants
    )

    range_centers_mm = get_range_centers(distance_threshold, radial_bins)

    train_loader, val_loader, test_loader = preprocess_data(
        X_train, X_val, X_test, y_train, y_val, y_test, range_centers_mm
    )

    # Use the original CNN architecture (compatible with downstream scripts)
    print("Using original CNN architecture")
    model = TwoHeadedSonarRadialCNN(
        input_shape=(2, 200),
        profile_steps=profile_steps,
        radial_bins=radial_bins,
        range_centers_mm=range_centers_mm,
    ).to(device)

    start_time = time.time()
    
    # Normal training
    train_losses, val_losses, presence_losses, radial_losses = train_model(
        model, train_loader, val_loader, range_centers_mm
    )
    
    training_time = time.time() - start_time

    print(f'Training completed in {training_time:.2f} seconds')

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Total Loss')
    plt.plot(val_losses, label='Val Total Loss')
    plt.plot(presence_losses, label='Train Presence Loss')
    plt.plot(radial_losses, label='Train Radial Loss')
    plt.title('Training Progress (Radial)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot('training_curves')
    maybe_show_plot()

    model.load_state_dict(torch.load(f'{output_dir}/best_model_pytorch.pth', map_location=device))

    presence_threshold = find_best_presence_threshold(model, val_loader, range_centers_mm)

    training_params = {
        'target_type': 'radial',
        'train_quadrants': train_quadrants,
        'test_quadrant': test_quadrant,
        'profile_method': profile_method,
        'profile_opening_angle': profile_opening_angle,
        'profile_steps': profile_steps,
        'distance_threshold': distance_threshold,
        'presence_threshold': presence_threshold,
        'conv_filters': conv_filters,
        'kernel_size': kernel_size,
        'pool_size': pool_size,
        'dropout_rate': dropout_rate,
        'input_shape': [2, 200],
        'output_shape': profile_steps,
        'radial_bins': radial_bins,
        'range_centers_mm': range_centers_mm.tolist(),
        'radial_sigma_mm': radial_sigma_mm,
        'radial_loss_weight': radial_loss_weight,
        'use_sample_weighting': bool(use_sample_weighting),
        'sample_weight_power': float(sample_weight_power),
        'sample_weight_min': float(sample_weight_min),
        'sample_weight_max': float(sample_weight_max),
        'sample_weight_feature_bins': int(sample_weight_feature_bins),
        'sample_weight_shape_segments': int(sample_weight_shape_segments),
        'model_architecture': 'original',
    }

    with open(f'{output_dir}/training_params.json', 'w') as f:
        json.dump(training_params, f, indent=2)
    print(f"Saved training parameters to {output_dir}/training_params.json")

    write_training_readme()

    create_scatter_plots(model, test_loader, range_centers_mm, 'Test Set')
    create_scatter_plots(model, train_loader, range_centers_mm, 'Training Set')

    create_presence_confusion_plots(model, test_loader, range_centers_mm, 'Test Set', threshold=presence_threshold)
    create_presence_confusion_plots(model, train_loader, range_centers_mm, 'Training Set', threshold=presence_threshold)
    create_radial_distribution_examples(
        model,
        test_loader,
        range_centers_mm,
        'Test Set',
        num_plots=radial_example_plots_count,
        samples_per_plot=radial_example_samples_per_plot,
        random_seed=seed,
    )

    presence_preds, distance_preds_mm, presence_targets, distance_targets_mm = collect_predictions(
        model, test_loader, range_centers_mm
    )

    presence_acc = np.mean((presence_preds > presence_threshold) == (presence_targets > 0.5))

    present_mask = presence_targets > 0.5
    if np.any(present_mask):
        distance_mse = np.mean((distance_preds_mm[present_mask] - distance_targets_mm[present_mask]) ** 2)
        distance_mae = np.mean(np.abs(distance_preds_mm[present_mask] - distance_targets_mm[present_mask]))
    else:
        distance_mse = float('nan')
        distance_mae = float('nan')

    print('\nTest Set Performance:')
    print(f'   Presence Accuracy: {presence_acc:.4f}')
    print(f'   Distance MSE (present only): {distance_mse:.4f}')
    print(f'   Distance MAE (present only): {distance_mae:.4f}')

    plt.figure(figsize=(12, 8))
    for i in range(min(6, len(presence_preds))):
        ax1 = plt.subplot(3, 2, i + 1)
        ax1.plot(presence_targets[i], 'b-', label='True Presence')
        ax1.plot(presence_preds[i], 'r--', label='Pred Presence')

        ax2 = ax1.twinx()
        if np.any(presence_targets[i] > 0.5):
            ax2.plot(distance_targets_mm[i], 'g-', label='True Distance (mm)')
            ax2.plot(distance_preds_mm[i], 'm--', label='Pred Distance (mm)')
            ax2.set_ylim(0, distance_threshold * 1.05)
        ax2.set_ylabel('Distance (mm)')

        ax1.set_title(f'Test Sample {i}')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
        ax1.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    save_plot('test_samples')
    maybe_show_plot()

    sample_errors = []
    for i in range(len(presence_preds)):
        if np.any(present_mask[i]):
            per_bin_error = np.abs(distance_preds_mm[i] - distance_targets_mm[i])
            sample_errors.append(np.mean(per_bin_error[present_mask[i]]))
        else:
            sample_errors.append(0)

    plt.figure(figsize=(12, 8))
    test_rob_x = rob_x[test_mask]
    test_rob_y = rob_y[test_mask]

    scatter = plt.scatter(test_rob_x, test_rob_y, c=sample_errors, cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Distance Error (mm)')

    mean_x = np.mean(rob_x)
    mean_y = np.mean(rob_y)
    plt.axvline(mean_x, color='red', linestyle='--', alpha=0.5, label='Mean X')
    plt.axhline(mean_y, color='blue', linestyle='--', alpha=0.5, label='Mean Y')

    plt.title('Spatial Distribution of Test Errors')
    plt.xlabel('X position (mm)')
    plt.ylabel('Y position (mm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    save_plot('spatial_errors')
    maybe_show_plot()

    return model


if __name__ == '__main__':
    model = main()
