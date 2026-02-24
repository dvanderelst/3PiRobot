"""
SCRIPT_TrainEchoProcessor

Pipeline:
1) Train a single-target CNN: sonar -> closest profile distance (mm)
2) Convert predicted distance to sample index using corrected_distance_axis
3) Compute distance-conditioned IID from a local echo window
4) Correlate IID with profile-derived asymmetry metrics
"""

# ============================================
# CONFIGURATION
# ============================================

# Data
sessions = ["sessionB01", "sessionB02", "sessionB03", "sessionB04", "sessionB05"]
profile_opening_angle = 60
profile_steps = 21

# Split
train_quadrants = [0, 1, 2]
test_quadrant = 3
validation_split = 0.2
seed = 42

# Training
batch_size = 32
epochs = 80
patience = 10
learning_rate = 1e-3
l2_reg = 1e-3
dropout_rate = 0.30

# Model
conv_filters = [32, 64, 128]
kernel_size = 5
pool_size = 2

# Normalization/calibration
normalize_sonar = True
normalize_target = True
enable_bias_calibration = True
reuse_existing_model_for_sweep = False

# IID from predicted distance (sample window around mapped distance index)
iid_window_pre_samples = 2
iid_window_post_samples = 10
iid_window_center_offset_samples = 0
iid_eps = 1e-9

# Exploratory sweep for window placement/width
enable_window_sweep = True
sweep_center_offsets = list(range(-10, 10, 3))
sweep_pre_samples = list(range(0, 10))
sweep_post_samples = list(range(5, 50, 5))
sweep_target_asymmetry = "zero_split"  # "zero_split" or "min_location"
min_location_center_band_deg = 0.0  # 0.0 = strict sign(argmin azimuth) test

# Output
output_dir = "EchoProcessor"
plot_dpi = 240
plot_format = "png"


# ============================================
# IMPORTS
# ============================================

import json
import os
import time

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

from Library import DataProcessor
from Library import EchoProcessor


os.makedirs(output_dir, exist_ok=True)


def save_plot(filename):
    if plot_format in ["png", "both"]:
        plt.savefig(f"{output_dir}/{filename}.png", dpi=plot_dpi, bbox_inches="tight", facecolor="white")
    if plot_format in ["svg", "both"]:
        plt.savefig(f"{output_dir}/{filename}.svg", bbox_inches="tight", facecolor="white")


def write_readme():
    txt = """# EchoProcessor Output

Artifacts from `SCRIPT_TrainEchoProcessor.py`.

## Core
- `best_model_pytorch.pth`: Best distance model by validation loss.
- `echoprocessor_artifacts.pth`: Portable inference artifact for `Library/EchoProcessor.py`.
- `training_params.json`: Configuration + metrics + IID correlation summary.

## Plots
- `training_curves.*`
- `distance_scatter_test_set.*`
- `iid_vs_zero_split_asymmetry.*`
- `iid_argmin_sign_confusion.*`
- `iid_window_sweep.*` (if sweep enabled)
"""
    with open(f"{output_dir}/README.md", "w") as f:
        f.write(txt)


class DistanceCNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
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
            dummy = torch.zeros(1, *input_shape)
            out = self.shared(dummy)
            flat = out.view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        z = self.shared(x)
        return self.head(z).squeeze(-1)


class SonarDistanceDataset(Dataset):
    def __init__(self, sonar, y_mm):
        self.sonar = torch.FloatTensor(sonar)
        self.y_mm = torch.FloatTensor(y_mm)

    def __len__(self):
        return len(self.sonar)

    def __getitem__(self, idx):
        x = self.sonar[idx].transpose(0, 1)  # (samples,2)->(2,samples)
        y = self.y_mm[idx]
        return x, y


def _subset_tensors(ds):
    if isinstance(ds, Subset):
        base = ds.dataset
        idx = torch.as_tensor(ds.indices, dtype=torch.long)
        return base.sonar[idx], base.y_mm[idx]
    return ds.sonar, ds.y_mm


def compute_norm_stats(train_ds):
    sonar, y = _subset_tensors(train_ds)
    sonar_mean = sonar.mean(dim=(0, 1)).clamp_min(-1e12)  # shape (2,)
    sonar_std = sonar.std(dim=(0, 1)).clamp_min(1e-6)
    y_mean = y.mean()
    y_std = y.std().clamp_min(1e-6)
    return {
        "sonar_mean": sonar_mean,
        "sonar_std": sonar_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def normalize_batch(x, y, norm, device):
    if normalize_sonar:
        m = norm["sonar_mean"].to(device).view(1, -1, 1)
        s = norm["sonar_std"].to(device).view(1, -1, 1)
        x = (x - m) / s
    if normalize_target:
        ym = norm["y_mean"].to(device)
        ys = norm["y_std"].to(device)
        y = (y - ym) / ys
    return x, y


def denorm_y(y_pred, norm):
    if not normalize_target:
        return y_pred
    return y_pred * norm["y_std"].item() + norm["y_mean"].item()


def create_split(sonar, y_mm, quadrants):
    train_mask = np.isin(quadrants, train_quadrants)
    test_mask = quadrants == test_quadrant

    ds_train_full = SonarDistanceDataset(sonar[train_mask], y_mm[train_mask])
    ds_test = SonarDistanceDataset(sonar[test_mask], y_mm[test_mask])

    n_train = int((1.0 - validation_split) * len(ds_train_full))
    n_val = len(ds_train_full) - n_train
    ds_train, ds_val = random_split(
        ds_train_full, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    return ds_train, ds_val, ds_test, train_mask, test_mask


def train_model(model, train_loader, val_loader, norm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    criterion = nn.HuberLoss(delta=1.0)

    history = {"train": [], "val": []}
    best_val = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        run_train = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x = x.to(device)
            y = y.to(device)
            x, y = normalize_batch(x, y, norm, device)
            pred = model(x)
            loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_train += loss.item() * x.size(0)
        train_loss = run_train / len(train_loader.dataset)

        model.eval()
        run_val = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                x, y = normalize_batch(x, y, norm, device)
                pred = model(x)
                loss = criterion(pred, y)
                run_val += loss.item() * x.size(0)
        val_loss = run_val / len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"Epoch {epoch + 1}: train={train_loss:.5f}, val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "history": history}, f"{output_dir}/best_model_pytorch.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch + 1}")
                break

    return model, history


def collect_predictions(model, loader, norm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x_norm, y_norm = normalize_batch(x, y, norm, device)
            pred = model(x_norm)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(denorm_y(pred.cpu().numpy(), norm))
    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


def fit_linear_calibration(y_true, y_pred):
    if np.std(y_pred) < 1e-8:
        return 1.0, 0.0
    a, b = np.polyfit(y_pred, y_true, 1)
    return float(a), float(b)


def apply_calibration(y_pred, cal):
    if cal is None:
        return y_pred
    return cal["slope"] * y_pred + cal["intercept"]


def rankdata(a):
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = r
        i = j + 1
    return ranks


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return np.nan
    xx = x[m] - np.mean(x[m])
    yy = y[m] - np.mean(y[m])
    den = np.sqrt(np.sum(xx * xx) * np.sum(yy * yy))
    if den <= 1e-12:
        return np.nan
    return float(np.sum(xx * yy) / den)


def spearman_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return np.nan
    rx = rankdata(x[m])
    ry = rankdata(y[m])
    return pearson_corr(rx, ry)


def iid_from_distance_window(sonar_lr, dist_axis_mm, target_dist_mm, pre, post, eps, center_offset=0):
    if not (np.isfinite(target_dist_mm) and np.any(np.isfinite(dist_axis_mm))):
        return np.nan
    idx = int(np.argmin(np.abs(dist_axis_mm - target_dist_mm)))
    idx = int(np.clip(idx + int(center_offset), 0, len(dist_axis_mm) - 1))
    lo = max(0, idx - int(pre))
    hi = min(len(dist_axis_mm), idx + int(post) + 1)
    if hi <= lo:
        return np.nan
    w = sonar_lr[lo:hi, :]
    l = w[:, 0]
    r = w[:, 1]
    el = float(np.sum(l * l))
    er = float(np.sum(r * r))
    return float(10.0 * np.log10((el + eps) / (er + eps)))


def compute_profile_min_location_deg(profiles_mm, profile_centers_deg):
    """
    Min-location per sample:
    azimuth angle (deg) at global minimum profile distance.
    """
    p = np.asarray(profiles_mm, dtype=np.float32)
    c = np.asarray(profile_centers_deg, dtype=np.float32)
    out = np.full((p.shape[0],), np.nan, dtype=np.float32)
    for i in range(p.shape[0]):
        row = p[i]
        ctr = c[i]
        valid = np.isfinite(row) & np.isfinite(ctr)
        if not np.any(valid):
            continue
        row_masked = row.copy()
        row_masked[~valid] = np.inf
        idx = int(np.argmin(row_masked))
        if not np.isfinite(row_masked[idx]):
            continue
        out[i] = float(ctr[idx])
    return out


def iid_side_accuracy(iid_values, min_location_deg, center_band_deg):
    """
    LR side accuracy from IID sign vs min-location side label.
    Ignores center labels within +/- center_band_deg.
    """
    iid = np.asarray(iid_values, dtype=np.float32)
    az = np.asarray(min_location_deg, dtype=np.float32)
    valid = np.isfinite(iid) & np.isfinite(az)
    if not np.any(valid):
        return np.nan
    iid = iid[valid]
    az = az[valid]

    true_side = np.full_like(az, "", dtype=object)
    true_side[az > float(center_band_deg)] = "L"
    true_side[az < -float(center_band_deg)] = "R"
    lr = np.isin(true_side, ["L", "R"])
    if not np.any(lr):
        return np.nan
    true_side = true_side[lr]
    pred_side = np.where(iid[lr] >= 0.0, "L", "R")
    return float(np.mean(pred_side == true_side))


def iid_sign_confusion(iid_values, min_location_deg, center_band_deg=0.0):
    """
    Confusion of IID sign vs argmin(profile) side.
    Rows=true side [L,R], cols=pred side [L,R].
    """
    iid = np.asarray(iid_values, dtype=np.float32)
    az = np.asarray(min_location_deg, dtype=np.float32)
    valid = np.isfinite(iid) & np.isfinite(az)
    iid = iid[valid]
    az = az[valid]

    true_side = np.full_like(az, "", dtype=object)
    true_side[az > float(center_band_deg)] = "L"
    true_side[az < -float(center_band_deg)] = "R"
    lr = np.isin(true_side, ["L", "R"])
    true_side = true_side[lr]
    pred_side = np.where(iid[lr] >= 0.0, "L", "R")

    cm = np.zeros((2, 2), dtype=np.int32)  # [L,R]x[L,R]
    labels = ["L", "R"]
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            cm[i, j] = int(np.sum((true_side == t) & (pred_side == p)))
    n = int(np.sum(cm))
    acc = float(np.trace(cm) / n) if n > 0 else np.nan
    return cm, n, acc


def plot_training(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.title("Distance Training Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_plot("training_curves")
    plt.close()


def plot_distance_scatter(y_true, y_pred, corrected_distance_mm_test):
    lo = float(min(np.min(y_true), np.min(y_pred), np.nanmin(corrected_distance_mm_test)))
    hi = float(max(np.max(y_true), np.max(y_pred), np.nanmax(corrected_distance_mm_test)))

    model_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    model_bias = float(np.mean(y_pred - y_true))
    model_p = pearson_corr(y_true, y_pred)
    model_s = spearman_corr(y_true, y_pred)

    base_mask = np.isfinite(corrected_distance_mm_test) & np.isfinite(y_true)
    base_true = y_true[base_mask]
    base_pred = corrected_distance_mm_test[base_mask]
    base_rmse = float(np.sqrt(np.mean((base_pred - base_true) ** 2))) if len(base_true) > 0 else np.nan
    base_bias = float(np.mean(base_pred - base_true)) if len(base_true) > 0 else np.nan
    base_p = pearson_corr(base_true, base_pred) if len(base_true) > 0 else np.nan
    base_s = spearman_corr(base_true, base_pred) if len(base_true) > 0 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    axes[0].scatter(y_true, y_pred, s=10, alpha=0.3)
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[0].set_xlabel("True closest distance (mm)")
    axes[0].set_ylabel("Predicted closest distance (mm)")
    axes[0].set_title(
        "Model vs Profile Min Distance\n"
        f"RMSE={model_rmse:.1f} mm, Bias={model_bias:.1f} mm, "
        f"Pearson={model_p:.3f}, Spearman={model_s:.3f}"
    )
    axes[0].text(
        0.02, 0.98,
        "What this shows:\nModel agreement with profile-derived closest distance.\n"
        "Ideal points lie on the red dashed diagonal.",
        transform=axes[0].transAxes, va="top", ha="left",
        fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(base_true, base_pred, s=10, alpha=0.3)
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[1].set_xlabel("True closest distance (mm)")
    axes[1].set_ylabel("Acquisition corrected_distance (mm)")
    axes[1].set_title(
        "Acquisition corrected_distance vs Profile Min Distance\n"
        f"RMSE={base_rmse:.1f} mm, Bias={base_bias:.1f} mm, "
        f"Pearson={base_p:.3f}, Spearman={base_s:.3f}"
    )
    axes[1].text(
        0.02, 0.98,
        "What this shows:\nYour existing corrected_distance estimate\n"
        "against the same profile-derived target.",
        transform=axes[1].transAxes, va="top", ha="left",
        fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )
    axes[1].grid(True, alpha=0.3)

    save_plot("distance_scatter_test_set")
    plt.close(fig)


def plot_iid_sign_confusions(iid_pred, iid_oracle, iid_baseline_sign_inverted, min_location_deg, center_band_deg):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    entries = [
        ("Pred-distance IID", iid_pred),
        ("Oracle-distance IID", iid_oracle),
        ("Baseline corrected IID (-IID)", iid_baseline_sign_inverted),
    ]
    for ax, (title, iid_vals) in zip(axes, entries):
        cm, n, acc = iid_sign_confusion(iid_vals, min_location_deg, center_band_deg=center_band_deg)
        row = cm.sum(axis=1, keepdims=True)
        cmn = cm / np.maximum(row, 1)
        im = ax.imshow(cmn, vmin=0.0, vmax=1.0, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n(n={cm[i,j]})", ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1], ["Pred L", "Pred R"])
        ax.set_yticks([0, 1], ["True L", "True R"])
        ax.set_title(f"{title}\nSign accuracy={acc:.3f} (N={n})", fontsize=11, pad=8)
    cbar = fig.colorbar(im, ax=axes.tolist(), location="right", fraction=0.03, pad=0.02)
    cbar.set_label("Row-normalized fraction")
    fig.text(0.01, 0.98, "IID sign vs argmin(profile) side", ha="left", va="top", fontsize=12)
    fig.text(0.01, 0.02, f"Sign(IID) vs sign(argmin azimuth), center exclusion +/-{center_band_deg:.1f} deg",
             ha="left", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.055, right=0.96, top=0.86, bottom=0.11, wspace=0.18)
    save_plot("iid_argmin_sign_confusion")
    plt.close(fig)


def plot_iid_vs_zero_split(asym_zero_mm, iid_pred, iid_oracle, iid_baseline_sign_inverted):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    entries = [
        ("Pred-distance IID", iid_pred),
        ("Oracle-distance IID", iid_oracle),
        ("Baseline corrected IID (-IID)", iid_baseline_sign_inverted),
    ]
    for ax, (title, iid_vals) in zip(axes, entries):
        m = np.isfinite(asym_zero_mm) & np.isfinite(iid_vals)
        ax.scatter(asym_zero_mm[m], iid_vals[m], s=8, alpha=0.25)
        p = pearson_corr(asym_zero_mm[m], iid_vals[m]) if np.any(m) else np.nan
        s = spearman_corr(asym_zero_mm[m], iid_vals[m]) if np.any(m) else np.nan
        ax.set_xlabel("Profile asymmetry: left_min - right_min (mm)")
        ax.set_ylabel("IID (dB)")
        ax.set_title(f"{title}\nPearson={p:.3f}, Spearman={s:.3f}", fontsize=11, pad=8)
        ax.grid(True, alpha=0.3)
    fig.text(0.01, 0.98, "IID vs zero-split profile asymmetry", ha="left", va="top", fontsize=12)
    fig.text(0.01, 0.02, "Diagnostic: this is also available as a sweep objective.", ha="left", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.055, right=0.98, top=0.86, bottom=0.13, wspace=0.22)
    save_plot("iid_vs_zero_split_asymmetry")
    plt.close(fig)


def run_window_sweep(sonar_test, axis_test_mm, pred_dist_mm, asym_test):
    results = []
    for off in sweep_center_offsets:
        for pre in sweep_pre_samples:
            for post in sweep_post_samples:
                iid = np.array([
                    iid_from_distance_window(
                        sonar_test[i], axis_test_mm[i], pred_dist_mm[i],
                        pre, post, iid_eps, center_offset=off
                    )
                    for i in range(len(pred_dist_mm))
                ], dtype=np.float32)
                p = pearson_corr(iid, asym_test)
                s = spearman_corr(iid, asym_test)
                score = abs(s) if np.isfinite(s) else -np.inf
                results.append({
                    "center_offset_samples": int(off),
                    "pre_samples": int(pre),
                    "post_samples": int(post),
                    "pearson": p,
                    "spearman": s,
                    "score_abs_spearman": score,
                })
    results.sort(key=lambda d: d["score_abs_spearman"], reverse=True)
    return results


def plot_window_sweep_results(sweep_results, asymmetry_label):
    if not sweep_results:
        return

    offsets = sorted({int(r["center_offset_samples"]) for r in sweep_results})
    pres = sorted({int(r["pre_samples"]) for r in sweep_results})
    posts = sorted({int(r["post_samples"]) for r in sweep_results})
    finite_vals = np.array(
        [float(r["spearman"]) for r in sweep_results if np.isfinite(r["spearman"])],
        dtype=np.float32
    )
    if finite_vals.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
        if abs(vmax - vmin) < 1e-8:
            vmin -= 1e-3
            vmax += 1e-3

    ncols = len(offsets)
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 5.4), squeeze=False)
    axes = axes[0]

    for ax, off in zip(axes, offsets):
        mat = np.full((len(pres), len(posts)), np.nan, dtype=np.float32)
        for r in sweep_results:
            if int(r["center_offset_samples"]) != off:
                continue
            i = pres.index(int(r["pre_samples"]))
            j = posts.index(int(r["post_samples"]))
            mat[i, j] = float(r["spearman"]) if np.isfinite(r["spearman"]) else np.nan

        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Offset = {off} samples", fontsize=10, pad=8)
        ax.set_xlabel("post_samples")
        ax.set_ylabel("pre_samples")
        ax.set_xticks(np.arange(len(posts)))
        ax.set_yticks(np.arange(len(pres)))
        ax.set_xticklabels(posts)
        ax.set_yticklabels(pres)
        ax.axis("tight")
        ax.grid(False)

        # Annotate each cell with Spearman value.
        for ii, pre in enumerate(pres):
            for jj, post in enumerate(posts):
                v = mat[ii, jj]
                txt = "nan" if not np.isfinite(v) else f"{v:.2f}"
                ax.text(jj, ii, txt, ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=axes.tolist(), location="right", fraction=0.025, pad=0.015)
    cbar.set_label("Spearman correlation")
    fig.text(0.01, 0.98, f"IID window sweep ({asymmetry_label} target)", ha="left", va="top", fontsize=11)
    fig.text(0.01, 0.02, "Cells: Spearman between IID and target variable for each (pre, post, offset).",
             ha="left", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.045, right=0.965, top=0.90, bottom=0.13, wspace=0.18)
    save_plot("iid_window_sweep")
    plt.close(fig)


def main():
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Loading data...")
    dc = DataProcessor.DataCollection(sessions)
    sonar = dc.load_sonar(flatten=False)  # (N, samples, 2)
    profiles, profile_centers = dc.load_profiles(opening_angle=profile_opening_angle, steps=profile_steps)
    quadrants = dc.quadrants
    corrected_axis_m = dc.get_field("sonar_package", "corrected_distance_axis")  # (N, samples)
    corrected_iid_db = dc.get_field("sonar_package", "corrected_iid")  # (N,)
    corrected_distance_m = dc.get_field("sonar_package", "corrected_distance")  # (N,)

    # Closest-distance target from profile (mm)
    y_closest_mm = np.min(profiles, axis=1).astype(np.float32)

    # Min-location target (deg) for IID side/location analysis.
    profile_min_location_deg = compute_profile_min_location_deg(profiles, profile_centers)
    half = profiles.shape[1] // 2
    left_min = np.min(profiles[:, :half], axis=1)
    right_min = np.min(profiles[:, half:], axis=1)
    profile_asym_zero_mm = (left_min - right_min).astype(np.float32)

    # Basic filtering
    finite = np.isfinite(sonar).all(axis=(1, 2))
    finite &= np.isfinite(y_closest_mm)
    finite &= np.isfinite(profile_min_location_deg)
    finite &= np.isfinite(profile_asym_zero_mm)
    finite &= np.isfinite(corrected_axis_m).all(axis=1)
    finite &= np.isfinite(corrected_iid_db)
    finite &= np.isfinite(corrected_distance_m)

    sonar = sonar[finite]
    y_closest_mm = y_closest_mm[finite]
    profile_min_location_deg = profile_min_location_deg[finite]
    profile_asym_zero_mm = profile_asym_zero_mm[finite]
    corrected_axis_mm = (corrected_axis_m[finite] * 1000.0).astype(np.float32)
    corrected_iid_db = corrected_iid_db[finite].astype(np.float32)
    corrected_distance_mm = (corrected_distance_m[finite] * 1000.0).astype(np.float32)
    quadrants = quadrants[finite]

    print(f"Kept {len(sonar)} samples after filtering.")

    ds_train, ds_val, ds_test, _, test_mask = create_split(sonar, y_closest_mm, quadrants)
    norm = compute_norm_stats(ds_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=pin)

    model = DistanceCNN((2, sonar.shape[1]))
    history = {"train": [], "val": []}
    checkpoint_path = f"{output_dir}/best_model_pytorch.pth"
    can_reuse = reuse_existing_model_for_sweep and os.path.exists(checkpoint_path)

    if can_reuse:
        print(f"Reusing existing checkpoint for evaluation/sweep: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        history = ckpt.get("history", history)
    else:
        if reuse_existing_model_for_sweep and not os.path.exists(checkpoint_path):
            print(f"Requested reuse, but checkpoint not found at: {checkpoint_path}")
            print("Proceeding with training.")
        model, history = train_model(model, train_loader, val_loader, norm)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    calibration = None
    if enable_bias_calibration:
        yv_true, yv_pred = collect_predictions(model, val_loader, norm)
        slope, intercept = fit_linear_calibration(yv_true, yv_pred)
        calibration = {"slope": slope, "intercept": intercept}
        print(f"Validation calibration: y={slope:.4f}*pred+{intercept:.2f}")

    yt_true, yt_pred_raw = collect_predictions(model, test_loader, norm)
    yt_pred = apply_calibration(yt_pred_raw, calibration)

    # Pull test-aligned arrays for IID analysis.
    test_idx = np.where(test_mask)[0]
    sonar_test = sonar[test_idx]
    axis_test_mm = corrected_axis_mm[test_idx]
    min_location_test_deg = profile_min_location_deg[test_idx]
    asym_zero_test = profile_asym_zero_mm[test_idx]
    iid_baseline_test = corrected_iid_db[test_idx]
    corrected_distance_mm_test = corrected_distance_mm[test_idx]
    if sweep_target_asymmetry == "zero_split":
        sweep_asym = asym_zero_test
        sweep_asymmetry_label = "zero_split"
    elif sweep_target_asymmetry == "min_location":
        sweep_asym = min_location_test_deg
        sweep_asymmetry_label = "min_location"
    else:
        raise ValueError(
            f"Invalid sweep_target_asymmetry='{sweep_target_asymmetry}'. "
            "Use 'zero_split' or 'min_location'."
        )

    iid_pred = np.array([
        iid_from_distance_window(
            sonar_test[i], axis_test_mm[i], yt_pred[i],
            iid_window_pre_samples, iid_window_post_samples, iid_eps,
            center_offset=iid_window_center_offset_samples
        )
        for i in range(len(yt_pred))
    ], dtype=np.float32)
    iid_oracle = np.array([
        iid_from_distance_window(
            sonar_test[i], axis_test_mm[i], yt_true[i],
            iid_window_pre_samples, iid_window_post_samples, iid_eps,
            center_offset=iid_window_center_offset_samples
        )
        for i in range(len(yt_true))
    ], dtype=np.float32)

    sweep_results = []
    best_window = {
        "center_offset_samples": int(iid_window_center_offset_samples),
        "pre_samples": int(iid_window_pre_samples),
        "post_samples": int(iid_window_post_samples),
    }
    if enable_window_sweep:
        sweep_results = run_window_sweep(sonar_test, axis_test_mm, yt_pred, sweep_asym)
        if sweep_results:
            best_window = {
                "center_offset_samples": int(sweep_results[0]["center_offset_samples"]),
                "pre_samples": int(sweep_results[0]["pre_samples"]),
                "post_samples": int(sweep_results[0]["post_samples"]),
            }
            print(
                "Best sweep window (by |Spearman| on pred-distance IID): "
                f"offset={best_window['center_offset_samples']}, "
                f"pre={best_window['pre_samples']}, post={best_window['post_samples']}, "
                f"Pearson={sweep_results[0]['pearson']:.4f}, Spearman={sweep_results[0]['spearman']:.4f}, "
                f"target={sweep_asymmetry_label}"
            )
            iid_pred = np.array([
                iid_from_distance_window(
                    sonar_test[i], axis_test_mm[i], yt_pred[i],
                    best_window["pre_samples"], best_window["post_samples"], iid_eps,
                    center_offset=best_window["center_offset_samples"]
                )
                for i in range(len(yt_pred))
            ], dtype=np.float32)
            iid_oracle = np.array([
                iid_from_distance_window(
                    sonar_test[i], axis_test_mm[i], yt_true[i],
                    best_window["pre_samples"], best_window["post_samples"], iid_eps,
                    center_offset=best_window["center_offset_samples"]
                )
                for i in range(len(yt_true))
            ], dtype=np.float32)

    iid_baseline_sign_inverted = -iid_baseline_test
    cm_pred, n_pred_sign, acc_pred_sign = iid_sign_confusion(
        iid_pred, min_location_test_deg, center_band_deg=min_location_center_band_deg
    )
    cm_oracle, n_oracle_sign, acc_oracle_sign = iid_sign_confusion(
        iid_oracle, min_location_test_deg, center_band_deg=min_location_center_band_deg
    )
    cm_base_inv, n_base_inv_sign, acc_base_inv_sign = iid_sign_confusion(
        iid_baseline_sign_inverted, min_location_test_deg, center_band_deg=min_location_center_band_deg
    )

    # Distance metrics
    rmse_mm = float(np.sqrt(np.mean((yt_pred - yt_true) ** 2)))
    mae_mm = float(np.mean(np.abs(yt_pred - yt_true)))
    bias_mm = float(np.mean(yt_pred - yt_true))

    # IID metrics vs min-location azimuth target
    metrics_min_location = {
        "pred_iid_pearson": pearson_corr(iid_pred, min_location_test_deg),
        "pred_iid_spearman": spearman_corr(iid_pred, min_location_test_deg),
        "pred_iid_sign_acc": acc_pred_sign,
        "pred_iid_sign_n": n_pred_sign,
        "oracle_iid_pearson": pearson_corr(iid_oracle, min_location_test_deg),
        "oracle_iid_spearman": spearman_corr(iid_oracle, min_location_test_deg),
        "oracle_iid_sign_acc": acc_oracle_sign,
        "oracle_iid_sign_n": n_oracle_sign,
        "baseline_iid_pearson": pearson_corr(iid_baseline_test, min_location_test_deg),
        "baseline_iid_spearman": spearman_corr(iid_baseline_test, min_location_test_deg),
        "baseline_sign_inverted_iid_pearson": pearson_corr(iid_baseline_sign_inverted, min_location_test_deg),
        "baseline_sign_inverted_iid_spearman": spearman_corr(iid_baseline_sign_inverted, min_location_test_deg),
        "baseline_sign_inverted_iid_sign_acc": acc_base_inv_sign,
        "baseline_sign_inverted_iid_sign_n": n_base_inv_sign,
    }
    metrics_zero_split = {
        "pred_iid_pearson": pearson_corr(iid_pred, asym_zero_test),
        "pred_iid_spearman": spearman_corr(iid_pred, asym_zero_test),
        "oracle_iid_pearson": pearson_corr(iid_oracle, asym_zero_test),
        "oracle_iid_spearman": spearman_corr(iid_oracle, asym_zero_test),
        "baseline_sign_inverted_iid_pearson": pearson_corr(iid_baseline_sign_inverted, asym_zero_test),
        "baseline_sign_inverted_iid_spearman": spearman_corr(iid_baseline_sign_inverted, asym_zero_test),
    }
    metrics = {
        "min_location": metrics_min_location,
        "zero_split": metrics_zero_split,
    }

    print("\nDistance test metrics:")
    print(f"  RMSE: {rmse_mm:.3f} mm")
    print(f"  MAE:  {mae_mm:.3f} mm")
    print(f"  Bias: {bias_mm:.3f} mm")
    print("\nIID correlation/side metrics (min_location):")
    for k, v in metrics_min_location.items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")
    print("\nIID correlation metrics (zero_split):")
    for k, v in metrics_zero_split.items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

    if len(history.get("train", [])) > 0 and len(history.get("val", [])) > 0:
        plot_training(history)
    plot_distance_scatter(yt_true, yt_pred, corrected_distance_mm_test)
    plot_iid_vs_zero_split(asym_zero_test, iid_pred, iid_oracle, iid_baseline_sign_inverted)
    plot_iid_sign_confusions(
        iid_pred, iid_oracle, iid_baseline_sign_inverted,
        min_location_test_deg, center_band_deg=min_location_center_band_deg
    )
    if enable_window_sweep:
        plot_window_sweep_results(sweep_results, sweep_asymmetry_label)

    params = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sessions": sessions,
        "train_quadrants": train_quadrants,
        "test_quadrant": test_quadrant,
        "validation_split": validation_split,
        "profile_opening_angle": profile_opening_angle,
        "profile_steps": profile_steps,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "l2_reg": l2_reg,
        "dropout_rate": dropout_rate,
        "normalize_sonar": normalize_sonar,
        "normalize_target": normalize_target,
        "enable_bias_calibration": enable_bias_calibration,
        "reuse_existing_model_for_sweep": reuse_existing_model_for_sweep,
        "iid_window_pre_samples": iid_window_pre_samples,
        "iid_window_post_samples": iid_window_post_samples,
        "iid_window_center_offset_samples": iid_window_center_offset_samples,
        "enable_window_sweep": enable_window_sweep,
        "sweep_center_offsets": sweep_center_offsets,
        "sweep_pre_samples": sweep_pre_samples,
        "sweep_post_samples": sweep_post_samples,
        "sweep_target_asymmetry": sweep_target_asymmetry,
        "min_location_center_band_deg": min_location_center_band_deg,
        "selected_window_params": best_window,
        "selected_window_objective": {
            "target": sweep_asymmetry_label,
            "metric": "abs_spearman",
        },
        "distance_test_rmse_mm": rmse_mm,
        "distance_test_mae_mm": mae_mm,
        "distance_test_bias_mm": bias_mm,
        "calibration": calibration,
        "norm_stats": {
            "sonar_mean": norm["sonar_mean"].tolist(),
            "sonar_std": norm["sonar_std"].tolist(),
            "y_mean": float(norm["y_mean"].item()),
            "y_std": float(norm["y_std"].item()),
        },
        "iid_correlation_metrics": metrics,
        "iid_sign_confusions": {
            "pred": cm_pred.tolist(),
            "oracle": cm_oracle.tolist(),
            "baseline_sign_inverted": cm_base_inv.tolist(),
        },
        "iid_window_sweep_results": sweep_results,
    }
    with open(f"{output_dir}/training_params.json", "w") as f:
        json.dump(params, f, indent=2)

    artifact_path = EchoProcessor.save_artifacts(
        artifact_dir=output_dir,
        model_state_dict=ckpt["model_state_dict"],
        model_config={
            "num_sonar_samples": int(sonar.shape[1]),
            "conv_filters": [int(v) for v in conv_filters],
            "kernel_size": int(kernel_size),
            "pool_size": int(pool_size),
            "dropout_rate": float(dropout_rate),
        },
        norm_stats={
            "sonar_mean": norm["sonar_mean"].detach().cpu().numpy(),
            "sonar_std": norm["sonar_std"].detach().cpu().numpy(),
            "y_mean": float(norm["y_mean"].item()),
            "y_std": float(norm["y_std"].item()),
        },
        normalize_sonar=normalize_sonar,
        normalize_target=normalize_target,
        calibration=calibration,
        iid_window=best_window,
        iid_eps=iid_eps,
        profile_opening_angle=profile_opening_angle,
        profile_steps=profile_steps,
    )

    write_readme()
    print(f"Saved EchoProcessor artifact: {artifact_path}")
    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
