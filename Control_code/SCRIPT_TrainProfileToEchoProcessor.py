"""
Train a profile -> EchoProcessor emulator.

Supervision target:
- distance_mm predicted by EchoProcessor
- iid_db computed by EchoProcessor
"""

# ============================================
# CONFIGURATION
# ============================================
sessions = ["sessionB01", "sessionB02", "sessionB03", "sessionB04", "sessionB05"]
# Taken from EchoProcessor artifact at runtime (kept None here intentionally).
profile_opening_angle = None
profile_steps = None

echo_artifact_dir = "EchoProcessor"
output_dir = "ProfileToEchoProcessor"

train_quadrants = [0, 1, 2]
test_quadrant = 3
validation_split = 0.2
seed = 42

batch_size = 64
epochs = 120
patience = 12
learning_rate = 1e-3
l2_reg = 1e-4
dropout = 0.1
hidden_sizes = [128, 128, 64]
head_hidden_size = 96

normalize_x = True
normalize_y = True
teacher_batch_size = 512
use_feature_augmentation = True

# Multi-task loss settings
distance_loss_weight = 1.0
iid_loss_weight = 1.5
distance_huber_delta = 1.0
iid_huber_delta = 2.0
enable_output_calibration = True

# IID sample weighting (applied to IID loss term)
use_iid_sample_weighting = True
iid_positive_weight = 1.25
iid_near_zero_abs_db = 1.5
iid_near_zero_weight = 1.25
iid_tail_abs_db = 5.0
iid_tail_weight = 1.25


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
from Library.EchoProcessor import EchoProcessor


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
    def __init__(self, x_profiles, y_targets):
        self.x = torch.as_tensor(x_profiles, dtype=torch.float32)
        self.y = torch.as_tensor(y_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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
    left_min = np.min(p[:, :half], axis=1)
    right_min = np.min(p[:, half:], axis=1)
    asym = left_min - right_min

    # local slope around minimum bin (simple finite difference)
    argmin_i = argmin.astype(np.int64)
    left_i = np.clip(argmin_i - 1, 0, steps - 1)
    right_i = np.clip(argmin_i + 1, 0, steps - 1)
    local_slope = p[np.arange(n), right_i] - p[np.arange(n), left_i]

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
        return base.x[idx], base.y[idx]
    return ds.x, ds.y


def compute_norm_stats(train_ds):
    x, y = _subset_tensors(train_ds)
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
        self.distance_head = nn.Sequential(
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
        d = self.distance_head(z)
        i = self.iid_head(z)
        return {"reg": torch.cat([d, i], dim=1)}


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
            x, y = normalize_batch(x, y, norm, device)
            pred_out = model(x)
            pred = pred_out["reg"].cpu().numpy()
            y_pred.append(pred)
            y_true.append(y.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    if normalize_y:
        y_true = denorm_y(y_true, norm)
        y_pred = denorm_y(y_pred, norm)
    return y_true.astype(np.float32), y_pred.astype(np.float32)


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


def multitask_loss(pred_out, y, y_raw, dist_criterion, iid_criterion):
    pred = pred_out["reg"]
    loss_dist = dist_criterion(pred[:, 0], y[:, 0])
    loss_iid_per_sample = iid_criterion(pred[:, 1], y[:, 1])
    w_iid = iid_sample_weights(y_raw[:, 1])
    loss_iid = torch.sum(w_iid * loss_iid_per_sample) / torch.clamp(torch.sum(w_iid), min=1.0)
    return distance_loss_weight * loss_dist + iid_loss_weight * loss_iid


def train_model(model, train_loader, val_loader, norm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    dist_criterion = nn.HuberLoss(delta=distance_huber_delta)
    iid_criterion = nn.HuberLoss(delta=iid_huber_delta, reduction="none")
    history = {"train": [], "val": []}

    best_val = float("inf")
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        run_train = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x = x.to(device)
            y = y.to(device)
            y_raw = y.clone()
            x, y = normalize_batch(x, y, norm, device)
            pred_out = model(x)
            loss = multitask_loss(pred_out, y, y_raw, dist_criterion, iid_criterion)
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
                y_raw = y.clone()
                x, y = normalize_batch(x, y, norm, device)
                pred_out = model(x)
                loss = multitask_loss(pred_out, y, y_raw, dist_criterion, iid_criterion)
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
    plt.title("Profile -> EchoProcessor training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_plot("training_curves")
    plt.close()


def plot_scatter(y_true, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [("Distance (mm)", 0), ("IID (dB)", 1)]
    for ax, (title, k) in zip(axes, labels):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        lo = float(min(np.min(yt), np.min(yp)))
        hi = float(max(np.max(yt), np.max(yp)))
        ax.scatter(yt, yp, s=10, alpha=0.3)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        p = pearson_corr(yt, yp)
        s = spearman_corr(yt, yp)
        ax.set_xlabel(f"True {title}")
        ax.set_ylabel(f"Pred {title}")
        if k == 1:
            sign_acc = float(np.mean(np.sign(yp) == np.sign(yt)))
            ax.set_title(f"{title}\nPearson={p:.3f}, Spearman={s:.3f}, SignAcc={sign_acc:.3f}")
        else:
            ax.set_title(f"{title}\nPearson={p:.3f}, Spearman={s:.3f}")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("test_scatter")
    plt.close(fig)


def infer_teacher_targets(ep, sonar, axis_mm, chunk_size):
    d_all = []
    i_all = []
    for i in range(0, len(sonar), chunk_size):
        j = min(i + chunk_size, len(sonar))
        out = ep.predict(sonar[i:j], axis_mm[i:j])
        d_all.append(out["distance_mm"])
        i_all.append(out["iid_db"])
    d = np.concatenate(d_all).astype(np.float32)
    iid = np.concatenate(i_all).astype(np.float32)
    return np.stack([d, iid], axis=1)


def main():
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Loading EchoProcessor artifact...")
    ep = EchoProcessor.load(echo_artifact_dir)
    if ep.profile_opening_angle is None or ep.profile_steps is None:
        raise ValueError(
            "EchoProcessor artifact is missing profile_opening_angle/profile_steps. "
            "Retrain SCRIPT_TrainEchoProcessor.py to regenerate artifacts."
        )
    effective_profile_opening_angle = float(ep.profile_opening_angle)
    effective_profile_steps = int(ep.profile_steps)
    print(
        f"Using profile settings from EchoProcessor artifact: "
        f"opening_angle={effective_profile_opening_angle}, steps={effective_profile_steps}"
    )

    print("Loading data...")
    dc = DataProcessor.DataCollection(sessions)
    profiles, _ = dc.load_profiles(
        opening_angle=effective_profile_opening_angle,
        steps=effective_profile_steps,
    )
    sonar = dc.load_sonar(flatten=False)
    axis_mm = (dc.get_field("sonar_package", "corrected_distance_axis") * 1000.0).astype(np.float32)
    quadrants = dc.quadrants

    finite = np.isfinite(profiles).all(axis=1)
    finite &= np.isfinite(sonar).all(axis=(1, 2))
    finite &= np.isfinite(axis_mm).all(axis=1)
    profiles = profiles[finite].astype(np.float32)
    sonar = sonar[finite].astype(np.float32)
    axis_mm = axis_mm[finite].astype(np.float32)
    quadrants = quadrants[finite]
    print(f"Kept {len(profiles)} samples after filtering.")

    print("Running EchoProcessor teacher inference...")
    targets = infer_teacher_targets(ep, sonar, axis_mm, chunk_size=teacher_batch_size)
    valid_target = np.isfinite(targets).all(axis=1)
    profiles = profiles[valid_target]
    targets = targets[valid_target]
    quadrants = quadrants[valid_target]
    print(f"Kept {len(profiles)} samples after teacher target filtering.")

    x_features = build_profile_features(profiles)

    train_mask = np.isin(quadrants, train_quadrants)
    test_mask = quadrants == test_quadrant

    ds_train_full = ProfileTargetDataset(x_features[train_mask], targets[train_mask])
    ds_test = ProfileTargetDataset(x_features[test_mask], targets[test_mask])
    n_train = int((1.0 - validation_split) * len(ds_train_full))
    n_val = len(ds_train_full) - n_train
    ds_train, ds_val = random_split(
        ds_train_full, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    norm = compute_norm_stats(ds_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=pin)

    model = ProfileMLP(in_dim=x_features.shape[1])
    history = train_model(model, train_loader, val_loader, norm)
    ckpt = torch.load(f"{output_dir}/best_model_pytorch.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    calibration = None
    if enable_output_calibration:
        yv_true, yv_pred = collect_predictions(model, val_loader, norm)
        calibration = fit_targetwise_calibration(yv_true, yv_pred)
        print(
            "Validation calibration: "
            f"distance y={calibration[0]['slope']:.4f}*pred+{calibration[0]['intercept']:.2f}, "
            f"iid y={calibration[1]['slope']:.4f}*pred+{calibration[1]['intercept']:.2f}"
        )

    y_true, y_pred_raw = collect_predictions(model, test_loader, norm)
    y_pred = apply_targetwise_calibration(y_pred_raw, calibration)

    dist_true, dist_pred = y_true[:, 0], y_pred[:, 0]
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

    metrics = {
        "distance": {
            "rmse_mm": float(np.sqrt(np.mean((dist_pred - dist_true) ** 2))),
            "mae_mm": float(np.mean(np.abs(dist_pred - dist_true))),
            "bias_mm": float(np.mean(dist_pred - dist_true)),
            "pearson": pearson_corr(dist_true, dist_pred),
            "spearman": spearman_corr(dist_true, dist_pred),
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
    }

    print("\nTest metrics:")
    print(
        f"Distance: RMSE={metrics['distance']['rmse_mm']:.2f} mm, "
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

    plot_training(history)
    plot_scatter(y_true, y_pred)

    params = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sessions": sessions,
        "profile_opening_angle": effective_profile_opening_angle,
        "profile_steps": effective_profile_steps,
        "echo_artifact_dir": echo_artifact_dir,
        "train_quadrants": train_quadrants,
        "test_quadrant": test_quadrant,
        "validation_split": validation_split,
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
        "use_iid_sample_weighting": use_iid_sample_weighting,
        "iid_positive_weight": iid_positive_weight,
        "iid_near_zero_abs_db": iid_near_zero_abs_db,
        "iid_near_zero_weight": iid_near_zero_weight,
        "iid_tail_abs_db": iid_tail_abs_db,
        "iid_tail_weight": iid_tail_weight,
        "teacher_batch_size": teacher_batch_size,
        "num_train": len(ds_train),
        "num_val": len(ds_val),
        "num_test": len(ds_test),
        "input_feature_dim": int(x_features.shape[1]),
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

    readme = """# ProfileToEchoProcessor Output

Artifacts from `SCRIPT_TrainProfileToEchoProcessor.py`.

## Core
- `best_model_pytorch.pth`: Best profile->(distance,iid) model by validation loss.
- `training_params.json`: Configuration + normalization + test metrics.

## Plots
- `training_curves.png`
- `test_scatter.png`
"""
    with open(f"{output_dir}/README.md", "w") as f:
        f.write(readme)

    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
