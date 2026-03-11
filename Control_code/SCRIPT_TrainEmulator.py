"""
Train an environment emulator that predicts sonar measurements from profiles.

This script trains a neural network that maps geometric profiles to sonar cues.
It learns to predict both distance and IID from profile data, effectively
creating a "world model" that can simulate what sonar measurements would
be received from different positions in an environment.

Supervision targets:
- distance_mm: corrected_distance from sonar_package (converted to mm)
- iid_db: corrected_iid from sonar_package
"""

# ============================================
# CONFIGURATION
# ============================================
sessions = ["sessionB01", "sessionB02", "sessionB03"]
profile_opening_angle = 180
profile_steps = 121

output_dir = "Emulator"

train_quadrants = [0, 2]
val_quadrants   = [1, 3]
# Test set = all original (non-flipped) samples across all quadrants.
seed = 42

batch_size = 64
epochs = 120
patience = 12
learning_rate = 1e-3
l2_reg = 1e-4
dropout = 0.1
hidden_sizes = [128, 128]
head_hidden_size = 96

normalize_x = True
normalize_y = True
use_feature_augmentation = True

# Multi-task loss settings
distance_loss_weight = 1.0
iid_loss_weight = 1.5
distance_huber_delta = 1.0
iid_huber_delta = 2.0
enable_output_calibration = True

# IID sample weighting (applied to IID loss term)
use_iid_sample_weighting = True
iid_positive_weight = 1.0        # set to 1.0: flip augmentation balances pos/neg IID
iid_near_zero_abs_db = 1.5
iid_near_zero_weight = 1.25
iid_tail_abs_db = 5.0
iid_tail_weight = 1.25

# Profile flip augmentation: for each training sample add a horizontally mirrored
# copy (profile bins reversed) with IID sign negated and distance unchanged.
# This forces the emulator to learn a perfectly symmetric profile->IID mapping
# regardless of any wall-side bias in the original data collection.
use_profile_flip_augmentation = True

# No-echo detection threshold: samples with distance_mm >= this value are treated as
# "no echo detected" (sonar returned max range). Determined empirically from data gap.
no_echo_min_distance_mm = 3500.0
echo_present_loss_weight = 1.5


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
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from Library import DataProcessor


os.makedirs(output_dir, exist_ok=True)


def save_plot(filename):
    plt.savefig(f"{output_dir}/{filename}.png", dpi=240, bbox_inches="tight", facecolor="white")


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
    return pearson_corr(rankdata(x[m]), rankdata(y[m]))


class ProfileTargetDataset(Dataset):
    def __init__(self, x_profiles, y_targets, echo_present):
        self.x = torch.as_tensor(x_profiles, dtype=torch.float32)
        self.y = torch.as_tensor(y_targets, dtype=torch.float32)
        self.ep = torch.as_tensor(echo_present, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.ep[idx]


def build_profile_features(profiles):
    """
    Feature augmentation from profiles only.
    Keeps raw profile bins and appends summary descriptors.
    """
    p = np.asarray(profiles, dtype=np.float32)
    if not use_feature_augmentation:
        return p

    n, steps = p.shape
    half = steps // 2
    idx = np.arange(steps, dtype=np.float32)
    idx_grid = np.broadcast_to(idx[None, :], p.shape)

    min_val = np.min(p, axis=1)
    argmin = np.argmin(p, axis=1).astype(np.float32)
    argmin_norm = argmin / max(steps - 1, 1)
    # Exclude center bin for odd step counts so the split is symmetric under
    # left-right reversal: asym(flipped) == -asym(original).
    if steps % 2 == 1:
        center = steps // 2
        right_min = np.min(p[:, :center], axis=1)
        left_min = np.min(p[:, center + 1:], axis=1)
    else:
        right_min = np.min(p[:, :half], axis=1)
        left_min = np.min(p[:, half:], axis=1)
    asym = left_min - right_min

    # local slope around minimum bin (simple finite difference)
    argmin_i = argmin.astype(np.int64)
    prev_i = np.clip(argmin_i - 1, 0, steps - 1)
    next_i = np.clip(argmin_i + 1, 0, steps - 1)
    local_slope = p[np.arange(n), next_i] - p[np.arange(n), prev_i]

    # weighted center-of-mass with inverse distance weights
    w = 1.0 / np.clip(p, 1e-3, None)
    wsum = np.sum(w, axis=1)
    com = np.sum(w * idx_grid, axis=1) / np.clip(wsum, 1e-6, None)
    com_norm = com / max(steps - 1, 1)

    extras = np.stack([min_val, argmin_norm, asym, local_slope, com_norm], axis=1).astype(np.float32)
    return np.concatenate([p, extras], axis=1).astype(np.float32)


def _subset_tensors(ds):
    if isinstance(ds, Subset):
        base = ds.dataset
        idx = torch.as_tensor(ds.indices, dtype=torch.long)
        return base.x[idx], base.y[idx], base.ep[idx]
    return ds.x, ds.y, ds.ep


def compute_norm_stats(train_ds):
    x, y, _ = _subset_tensors(train_ds)
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp_min(1e-6)
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0).clamp_min(1e-6)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def normalize_batch(x, y, norm, device):
    if normalize_x:
        x = (x - norm["x_mean"].to(device)) / norm["x_std"].to(device)
    if normalize_y:
        y = (y - norm["y_mean"].to(device)) / norm["y_std"].to(device)
    return x, y


def denorm_y(y_pred, norm):
    if not normalize_y:
        return y_pred
    ym = norm["y_mean"].cpu().numpy().reshape(1, -1)
    ys = norm["y_std"].cpu().numpy().reshape(1, -1)
    return y_pred * ys + ym


class ProfileMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        layers = []
        dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim = h
        self.trunk = nn.Sequential(*layers)
        self.echo_present_head = nn.Sequential(
            nn.Linear(dim, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )
        self.echo_distance_head = nn.Sequential(
            nn.Linear(dim, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )
        self.iid_head = nn.Sequential(
            nn.Linear(dim, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )

    def forward(self, x):
        z = self.trunk(x)
        ep = self.echo_present_head(z)   # logit (raw, no sigmoid)
        d = self.echo_distance_head(z)
        i = self.iid_head(z)
        return {"echo_logit": ep, "reg": torch.cat([d, i], dim=1)}


def collect_predictions(model, loader, norm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    ep_true = []
    ep_pred = []
    with torch.no_grad():
        for x, y, ep in loader:
            x = x.to(device)
            y = y.to(device)
            ep = ep.to(device)
            x, y = normalize_batch(x, y, norm, device)
            pred_out = model(x)
            pred = pred_out["reg"].cpu().numpy()
            ep_prob = torch.sigmoid(pred_out["echo_logit"]).cpu().numpy()
            y_pred.append(pred)
            y_true.append(y.cpu().numpy())
            ep_pred.append(ep_prob)
            ep_true.append(ep.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    ep_true = np.concatenate(ep_true, axis=0)
    ep_pred = np.concatenate(ep_pred, axis=0).squeeze(1)
    if normalize_y:
        y_true = denorm_y(y_true, norm)
        y_pred = denorm_y(y_pred, norm)
    return y_true.astype(np.float32), y_pred.astype(np.float32), ep_true.astype(np.float32), ep_pred.astype(np.float32)


def fit_targetwise_calibration(y_true, y_pred):
    cal = []
    for k in range(y_true.shape[1]):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        m = np.isfinite(yt) & np.isfinite(yp)
        if np.sum(m) < 2 or np.std(yp[m]) < 1e-8:
            cal.append({"slope": 1.0, "intercept": 0.0})
            continue
        a, b = np.polyfit(yp[m], yt[m], 1)
        cal.append({"slope": float(a), "intercept": float(b)})
    return cal


def apply_targetwise_calibration(y_pred, calibration):
    if calibration is None:
        return y_pred
    out = np.asarray(y_pred, dtype=np.float32).copy()
    for k, c in enumerate(calibration):
        out[:, k] = float(c["slope"]) * out[:, k] + float(c["intercept"])
    return out


def iid_sample_weights(iid_true_raw):
    """
    Build per-sample IID weights from raw (de-normalized) IID target in dB.
    """
    w = torch.ones_like(iid_true_raw)
    if not use_iid_sample_weighting:
        return w
    w = w * torch.where(iid_true_raw >= 0.0, float(iid_positive_weight), 1.0)
    w = w * torch.where(torch.abs(iid_true_raw) <= float(iid_near_zero_abs_db), float(iid_near_zero_weight), 1.0)
    w = w * torch.where(torch.abs(iid_true_raw) >= float(iid_tail_abs_db), float(iid_tail_weight), 1.0)
    return w


def multitask_loss(pred_out, y, y_raw, ep, dist_criterion, iid_criterion, bce_criterion):
    # Echo presence classification (BCE on raw logit)
    loss_ep = bce_criterion(pred_out["echo_logit"][:, 0], ep)

    # Distance regression — only on echo-present samples
    echo_mask = ep > 0.5
    if echo_mask.sum() > 0:
        loss_dist = dist_criterion(pred_out["reg"][:, 0][echo_mask], y[:, 0][echo_mask])
    else:
        loss_dist = torch.tensor(0.0, device=y.device)

    # IID regression (unchanged)
    loss_iid_per_sample = iid_criterion(pred_out["reg"][:, 1], y[:, 1])
    w_iid = iid_sample_weights(y_raw[:, 1])
    loss_iid = torch.sum(w_iid * loss_iid_per_sample) / torch.clamp(torch.sum(w_iid), min=1.0)

    return echo_present_loss_weight * loss_ep + distance_loss_weight * loss_dist + iid_loss_weight * loss_iid


def train_model(model, train_loader, val_loader, norm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    dist_criterion = nn.HuberLoss(delta=distance_huber_delta)
    iid_criterion = nn.HuberLoss(delta=iid_huber_delta, reduction="none")
    bce_criterion = nn.BCEWithLogitsLoss()
    history = {"train": [], "val": []}

    best_val = float("inf")
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        run_train = 0.0
        for x, y, ep in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x = x.to(device)
            y = y.to(device)
            ep = ep.to(device)
            y_raw = y.clone()
            x, y = normalize_batch(x, y, norm, device)
            pred_out = model(x)
            loss = multitask_loss(pred_out, y, y_raw, ep, dist_criterion, iid_criterion, bce_criterion)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_train += loss.item() * x.size(0)
        train_loss = run_train / len(train_loader.dataset)

        model.eval()
        run_val = 0.0
        with torch.no_grad():
            for x, y, ep in val_loader:
                x = x.to(device)
                y = y.to(device)
                ep = ep.to(device)
                y_raw = y.clone()
                x, y = normalize_batch(x, y, norm, device)
                pred_out = model(x)
                loss = multitask_loss(pred_out, y, y_raw, ep, dist_criterion, iid_criterion, bce_criterion)
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
    return history


def plot_training(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.title("Profile -> corrected_distance/iid training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_plot("training_curves")
    plt.close()


def _scatter_panel(ax, yt, yp, title):
    lo = float(min(np.min(yt), np.min(yp)))
    hi = float(max(np.max(yt), np.max(yp)))
    ax.scatter(yt, yp, s=10, alpha=0.3)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    p = pearson_corr(yt, yp)
    s = spearman_corr(yt, yp)
    bias = float(np.mean(yp - yt))
    ax.set_xlabel(f"True {title}")
    ax.set_ylabel(f"Pred {title}")
    if "iid" in title.lower():
        sign_acc = float(np.mean(np.sign(yp) == np.sign(yt)))
        pos_acc  = float(np.mean(yp[yt >= 0] >= 0)) if np.any(yt >= 0) else float("nan")
        neg_acc  = float(np.mean(yp[yt <  0] <  0)) if np.any(yt <  0) else float("nan")
        ax.set_title(
            f"{title}\nPearson={p:.3f}, Spearman={s:.3f}, "
            f"Bias={bias:+.3f} dB\n"
            f"SignAcc={sign_acc:.3f}  PosAcc={pos_acc:.3f}  NegAcc={neg_acc:.3f}"
        )
    else:
        ax.set_title(f"{title}\nPearson={p:.3f}, Spearman={s:.3f}, Bias={bias:+.1f} mm")
    ax.grid(True, alpha=0.3)


def plot_scatter(y_true, y_pred, ep_true, ep_pred_prob):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Distance — echo-present samples only
    echo_mask = ep_true > 0.5
    if echo_mask.sum() > 1:
        _scatter_panel(axes[0], y_true[echo_mask, 0], y_pred[echo_mask, 0], "corrected_distance (mm) [echo only]")
    else:
        axes[0].set_title("corrected_distance (mm)\n(no echo-present samples)")

    # IID
    _scatter_panel(axes[1], y_true[:, 1], y_pred[:, 1], "corrected_iid (dB)")

    # Echo present classification
    ep_pred_bin = (ep_pred_prob >= 0.5).astype(np.float32)
    acc = float(np.mean(ep_pred_bin == ep_true))
    n_pos = int(np.sum(ep_true > 0.5))
    n_neg = int(np.sum(ep_true <= 0.5))
    axes[2].scatter(ep_true, ep_pred_prob, s=10, alpha=0.3)
    axes[2].axhline(0.5, color='r', linestyle='--', linewidth=1)
    axes[2].set_xlabel("True echo_present")
    axes[2].set_ylabel("Pred echo_present prob")
    axes[2].set_title(f"echo_present\nAcc={acc:.3f}  N_echo={n_pos}  N_no_echo={n_neg}")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot("test_scatter")
    plt.close(fig)


def main():
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Loading data...")
    dc = DataProcessor.DataCollection(sessions)
    profiles, _ = dc.load_profiles(
        opening_angle=profile_opening_angle,
        steps=profile_steps,
    )
    quadrants = dc.quadrants

    finite = np.isfinite(profiles).all(axis=1)
    profiles = profiles[finite].astype(np.float32)
    quadrants = quadrants[finite]
    print(f"Kept {len(profiles)} samples after filtering.")

    distance_m = dc.get_field('sonar_package', 'corrected_distance').astype(np.float32)[finite]
    iid        = dc.get_field('sonar_package', 'corrected_iid').astype(np.float32)[finite]
    targets    = np.stack([distance_m * 1000.0, iid], axis=1)
    valid_target = np.isfinite(targets).all(axis=1)
    profiles  = profiles[valid_target]
    targets   = targets[valid_target]
    quadrants = quadrants[valid_target]
    print(f"Kept {len(profiles)} samples after target filtering.")

    echo_present = (targets[:, 0] < no_echo_min_distance_mm).astype(np.float32)
    n_echo = int(np.sum(echo_present))
    print(f"Echo present: {n_echo}/{len(echo_present)} samples ({100*n_echo/len(echo_present):.1f}% echo, {100*(1-n_echo/len(echo_present)):.1f}% no-echo)")

    # Flip augmentation applied to all data before splitting.
    # Each split (train/val/test) gets both original and flipped samples.
    if use_profile_flip_augmentation:
        profiles_flipped = profiles[:, ::-1].copy()
        targets_flipped  = targets.copy()
        targets_flipped[:, 1] *= -1.0          # negate IID; distance unchanged
        echo_present_flipped = echo_present.copy()   # echo_present unchanged by flip
        profiles_aug      = np.concatenate([profiles,      profiles_flipped],      axis=0)
        targets_aug       = np.concatenate([targets,       targets_flipped],       axis=0)
        echo_present_aug  = np.concatenate([echo_present,  echo_present_flipped],  axis=0)
        quadrants_aug     = np.concatenate([quadrants,     quadrants],             axis=0)
        print(f"Profile flip augmentation: {len(profiles)} -> {len(profiles_aug)} samples.")
    else:
        profiles_aug     = profiles
        targets_aug      = targets
        echo_present_aug = echo_present
        quadrants_aug    = quadrants

    # Quadrant-based split on augmented data.
    train_mask = np.isin(quadrants_aug, train_quadrants)
    val_mask   = np.isin(quadrants_aug, val_quadrants)

    x_train = build_profile_features(profiles_aug[train_mask])
    y_train = targets_aug[train_mask]
    x_val   = build_profile_features(profiles_aug[val_mask])
    y_val   = targets_aug[val_mask]
    x_test  = build_profile_features(profiles_aug)   # all quadrants, orig + flipped
    y_test  = targets_aug

    ep_train = echo_present_aug[train_mask]
    ep_val   = echo_present_aug[val_mask]
    ep_test  = echo_present_aug

    ds_train = ProfileTargetDataset(x_train, y_train, ep_train)
    ds_val   = ProfileTargetDataset(x_val,   y_val,   ep_val)
    ds_test  = ProfileTargetDataset(x_test,  y_test,  ep_test)

    norm = compute_norm_stats(ds_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=pin)

    model = ProfileMLP(in_dim=x_train.shape[1])
    history = train_model(model, train_loader, val_loader, norm)
    ckpt = torch.load(f"{output_dir}/best_model_pytorch.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    calibration = None
    if enable_output_calibration:
        yv_true, yv_pred, _, _ = collect_predictions(model, val_loader, norm)
        calibration = fit_targetwise_calibration(yv_true, yv_pred)
        print(
            "Validation calibration: "
            f"distance y={calibration[0]['slope']:.4f}*pred+{calibration[0]['intercept']:.2f}, "
            f"iid y={calibration[1]['slope']:.4f}*pred+{calibration[1]['intercept']:.2f}"
        )

    y_true, y_pred_raw, ep_true, ep_pred = collect_predictions(model, test_loader, norm)
    y_pred = apply_targetwise_calibration(y_pred_raw, calibration)

    echo_mask = ep_true > 0.5
    dist_true, dist_pred = y_true[echo_mask, 0], y_pred[echo_mask, 0]
    iid_true, iid_pred = y_true[:, 1], y_pred[:, 1]
    true_pos = iid_true >= 0.0
    true_neg = iid_true < 0.0
    pred_pos = iid_pred >= 0.0
    pred_neg = iid_pred < 0.0
    tp = int(np.sum(true_pos & pred_pos))
    tn = int(np.sum(true_neg & pred_neg))
    fp = int(np.sum(true_neg & pred_pos))
    fn = int(np.sum(true_pos & pred_neg))
    n_sign = int(len(iid_true))
    acc_pos = float(tp / max(1, int(np.sum(true_pos))))
    acc_neg = float(tn / max(1, int(np.sum(true_neg))))
    sign_acc = float((tp + tn) / max(1, n_sign))

    ep_pred_bin = (ep_pred >= 0.5).astype(np.float32)
    ep_acc = float(np.mean(ep_pred_bin == ep_true))
    ep_tp = int(np.sum((ep_true > 0.5) & (ep_pred_bin > 0.5)))
    ep_tn = int(np.sum((ep_true <= 0.5) & (ep_pred_bin <= 0.5)))
    ep_fp = int(np.sum((ep_true <= 0.5) & (ep_pred_bin > 0.5)))
    ep_fn = int(np.sum((ep_true > 0.5) & (ep_pred_bin <= 0.5)))

    metrics = {
        "distance": {
            "rmse_mm": float(np.sqrt(np.mean((dist_pred - dist_true) ** 2))) if len(dist_true) > 0 else float("nan"),
            "mae_mm": float(np.mean(np.abs(dist_pred - dist_true))) if len(dist_true) > 0 else float("nan"),
            "bias_mm": float(np.mean(dist_pred - dist_true)) if len(dist_true) > 0 else float("nan"),
            "pearson": pearson_corr(dist_true, dist_pred),
            "spearman": spearman_corr(dist_true, dist_pred),
            "n_echo_samples": int(len(dist_true)),
        },
        "iid": {
            "rmse_db": float(np.sqrt(np.mean((iid_pred - iid_true) ** 2))),
            "mae_db": float(np.mean(np.abs(iid_pred - iid_true))),
            "bias_db": float(np.mean(iid_pred - iid_true)),
            "pearson": pearson_corr(iid_true, iid_pred),
            "spearman": spearman_corr(iid_true, iid_pred),
            "sign_accuracy": sign_acc,
            "sign_confusion_counts": {
                "true_pos_pred_pos": tp,
                "true_pos_pred_neg": fn,
                "true_neg_pred_pos": fp,
                "true_neg_pred_neg": tn,
            },
            "sign_accuracy_true_pos": acc_pos,
            "sign_accuracy_true_neg": acc_neg,
            "sign_n": n_sign,
        },
        "echo_present": {
            "accuracy": ep_acc,
            "tp": ep_tp, "tn": ep_tn, "fp": ep_fp, "fn": ep_fn,
            "n_echo": int(np.sum(ep_true > 0.5)),
            "n_no_echo": int(np.sum(ep_true <= 0.5)),
        },
    }

    print("\nTest metrics:")
    print(
        f"Distance (echo-present only, N={metrics['distance']['n_echo_samples']}): "
        f"RMSE={metrics['distance']['rmse_mm']:.2f} mm, "
        f"MAE={metrics['distance']['mae_mm']:.2f} mm, Bias={metrics['distance']['bias_mm']:.2f} mm, "
        f"Pearson={metrics['distance']['pearson']:.3f}, Spearman={metrics['distance']['spearman']:.3f}"
    )
    print(
        f"IID: RMSE={metrics['iid']['rmse_db']:.3f} dB, "
        f"MAE={metrics['iid']['mae_db']:.3f} dB, Bias={metrics['iid']['bias_db']:.3f} dB, "
        f"Pearson={metrics['iid']['pearson']:.3f}, Spearman={metrics['iid']['spearman']:.3f}, "
        f"SignAcc={metrics['iid']['sign_accuracy']:.3f}"
    )
    print(
        "IID sign confusion "
        f"(N={metrics['iid']['sign_n']}): "
        f"TP={tp}, FN={fn}, FP={fp}, TN={tn}, "
        f"PosAcc={acc_pos:.3f}, NegAcc={acc_neg:.3f}"
    )
    print(
        f"Echo present: Acc={ep_acc:.3f}, "
        f"TP={ep_tp}, TN={ep_tn}, FP={ep_fp}, FN={ep_fn}, "
        f"N_echo={metrics['echo_present']['n_echo']}, N_no_echo={metrics['echo_present']['n_no_echo']}"
    )

    plot_training(history)
    plot_scatter(y_true, y_pred, ep_true, ep_pred)

    params = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sessions": sessions,
        "profile_opening_angle": profile_opening_angle,
        "profile_steps": profile_steps,
        "train_quadrants": train_quadrants,
        "val_quadrants": val_quadrants,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "l2_reg": l2_reg,
        "dropout": dropout,
        "hidden_sizes": hidden_sizes,
        "head_hidden_size": head_hidden_size,
        "normalize_x": normalize_x,
        "normalize_y": normalize_y,
        "use_feature_augmentation": use_feature_augmentation,
        "distance_loss_weight": distance_loss_weight,
        "iid_loss_weight": iid_loss_weight,
        "distance_huber_delta": distance_huber_delta,
        "iid_huber_delta": iid_huber_delta,
        "enable_output_calibration": enable_output_calibration,
        "use_profile_flip_augmentation": use_profile_flip_augmentation,
        "no_echo_min_distance_mm": no_echo_min_distance_mm,
        "echo_present_loss_weight": echo_present_loss_weight,
        "use_iid_sample_weighting": use_iid_sample_weighting,
        "iid_positive_weight": iid_positive_weight,
        "iid_near_zero_abs_db": iid_near_zero_abs_db,
        "iid_near_zero_weight": iid_near_zero_weight,
        "iid_tail_abs_db": iid_tail_abs_db,
        "iid_tail_weight": iid_tail_weight,
        "num_train": len(ds_train),
        "num_val": len(ds_val),
        "num_test": len(ds_test),
        "input_feature_dim": int(x_train.shape[1]),
        "metrics": metrics,
        "calibration": calibration,
        "norm_stats": {
            "x_mean": norm["x_mean"].tolist(),
            "x_std": norm["x_std"].tolist(),
            "y_mean": norm["y_mean"].tolist(),
            "y_std": norm["y_std"].tolist(),
        },
    }
    with open(f"{output_dir}/training_params.json", "w") as f:
        json.dump(params, f, indent=2)

    readme = """# Emulator Output

Artifacts from `SCRIPT_TrainEmulator.py`.

This emulator serves as a "world model" that predicts what sonar measurements
(distance and IID) would be received from different positions in an environment,
based on profile data.

## Core
- `best_model_pytorch.pth`: Best profile->(distance,iid) emulator model by validation loss.
- `training_params.json`: Configuration + normalization + test metrics.

## Plots
- `training_curves.png`: Training and validation loss curves
- `test_scatter.png`: Predicted vs true distance and IID on test set

## Usage
This emulator can be used to:
1. Simulate robot navigation through environments
2. Generate synthetic training data for policy learning
3. Enable "imagination-based" planning and learning
"""
    with open(f"{output_dir}/README.md", "w") as f:
        f.write(readme)

    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
