"""
Train an environment emulator that predicts sonar measurements from profiles.

This script trains a neural network that maps geometric profiles to sonar cues.
It predicts IID and echo presence probability from profile data.
Distance is computed geometrically (minimum over the profile opening angle)
rather than being regressed, so it is not a training target here.

Supervision targets:
- iid_db: corrected_iid from sonar_package
- echo_present: binary label (distance_mm < no_echo_min_distance_mm)
"""

# ============================================
# CONFIGURATION
# ============================================
sessions = ["sessionB01", "sessionB02", "sessionB03"]
profile_opening_angle = 90
profile_steps = 61

output_dir = "Emulator"

val_fraction = 0.15   # random holdout fraction for validation
seed = 42

batch_size = 64
epochs = 120
patience = 12
learning_rate = 1e-3
l2_reg = 1e-4
# CNN architecture
conv_channels = [16, 32, 32]   # channels per conv layer
conv_kernel    = 7              # kernel size (same for all layers)
fc_hidden      = 64             # FC hidden size after conv

# Loss settings
iid_huber_delta = 2.0
echo_present_loss_weight = 1.0  # relative weight of echo_present vs IID loss

# IID sample weighting (applied to IID loss term)
iid_near_zero_abs_db = 1.5
iid_near_zero_weight = 1.25
iid_tail_abs_db = 5.0
iid_tail_weight = 1.25

# No-echo detection threshold: samples with distance_mm >= this value are treated as
# "no echo detected" (sonar returned max range). Determined empirically from data gap.
no_echo_min_distance_mm = 3500.0


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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Library import DataProcessor
from Library import CodeLogger


os.makedirs(output_dir, exist_ok=True)
CodeLogger.log_code(output_dir, ['.', 'Library'], label='TrainEmulator')


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



def compute_norm_stats(ds):
    x_mean = ds.x.mean(dim=0)
    x_std  = ds.x.std(dim=0).clamp_min(1e-6)
    y_mean = ds.y.mean(dim=0)
    y_std  = ds.y.std(dim=0).clamp_min(1e-6)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def normalize_batch(x, y, norm, device):
    x = (x - norm["x_mean"].to(device)) / norm["x_std"].to(device)
    y = (y - norm["y_mean"].to(device)) / norm["y_std"].to(device)
    return x, y


def denorm_y(y_pred, norm):
    ym = norm["y_mean"].cpu().numpy().reshape(1, -1)
    ys = norm["y_std"].cpu().numpy().reshape(1, -1)
    return y_pred * ys + ym


class ProfileCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [nn.Conv1d(in_ch, out_ch, conv_kernel, padding=conv_kernel // 2), nn.ReLU()]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc   = nn.Sequential(nn.Linear(conv_channels[-1] * 8, fc_hidden), nn.ReLU())
        self.echo_present_head = nn.Linear(fc_hidden, 1)
        self.iid_head          = nn.Linear(fc_hidden, 1)

    def forward(self, x):
        z = self.conv(x.unsqueeze(1))   # (batch, C, L)
        z = self.pool(z).flatten(1)
        z = self.fc(z)
        return {"echo_logit": self.echo_present_head(z), "iid": self.iid_head(z)}


def collect_predictions(model, loader, norm):
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred, ep_true, ep_pred = [], [], [], []
    with torch.no_grad():
        for x, y, ep in loader:
            x, y, ep = x.to(device), y.to(device), ep.to(device)
            x, y = normalize_batch(x, y, norm, device)
            out = model(x)
            y_pred.append(out["iid"].cpu().numpy())
            y_true.append(y.cpu().numpy())
            ep_pred.append(torch.sigmoid(out["echo_logit"]).cpu().numpy())
            ep_true.append(ep.cpu().numpy())
    y_true  = denorm_y(np.concatenate(y_true,  axis=0), norm)
    y_pred  = denorm_y(np.concatenate(y_pred,  axis=0), norm)
    ep_true = np.concatenate(ep_true, axis=0)
    ep_pred = np.concatenate(ep_pred, axis=0).squeeze(1)
    return y_true.astype(np.float32), y_pred.astype(np.float32), ep_true.astype(np.float32), ep_pred.astype(np.float32)


def fit_calibration(y_true, y_pred):
    """Linear calibration: fit pred -> true for each output column."""
    cal = []
    for k in range(y_true.shape[1]):
        yt, yp = y_true[:, k], y_pred[:, k]
        m = np.isfinite(yt) & np.isfinite(yp)
        if np.sum(m) < 2 or np.std(yp[m]) < 1e-8:
            cal.append({"slope": 1.0, "intercept": 0.0})
        else:
            a, b = np.polyfit(yp[m], yt[m], 1)
            cal.append({"slope": float(a), "intercept": float(b)})
    return cal


def apply_calibration(y_pred, calibration):
    out = np.asarray(y_pred, dtype=np.float32).copy()
    for k, c in enumerate(calibration):
        out[:, k] = float(c["slope"]) * out[:, k] + float(c["intercept"])
    return out


def iid_sample_weights(iid_true_raw):
    """Per-sample IID loss weights from raw (de-normalised) IID in dB."""
    w = torch.ones_like(iid_true_raw)
    w = w * torch.where(torch.abs(iid_true_raw) <= float(iid_near_zero_abs_db), float(iid_near_zero_weight), 1.0)
    w = w * torch.where(torch.abs(iid_true_raw) >= float(iid_tail_abs_db), float(iid_tail_weight), 1.0)
    return w


def multitask_loss(pred_out, y, y_raw, ep, iid_criterion, bce_criterion):
    loss_ep = bce_criterion(pred_out["echo_logit"][:, 0], ep)
    loss_iid_per_sample = iid_criterion(pred_out["iid"][:, 0], y[:, 0])
    w_iid = iid_sample_weights(y_raw[:, 0])
    loss_iid = torch.sum(w_iid * loss_iid_per_sample) / torch.clamp(torch.sum(w_iid), min=1.0)
    return echo_present_loss_weight * loss_ep + loss_iid


def train_model(model, train_loader, val_loader, norm):
    device = next(model.parameters()).device
    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    iid_criterion = nn.HuberLoss(delta=iid_huber_delta, reduction="none")
    bce_criterion = nn.BCEWithLogitsLoss()
    history = {"train": [], "val": []}

    best_val = float("inf")
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        run_train = 0.0
        for x, y, ep in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x, y, ep = x.to(device), y.to(device), ep.to(device)
            y_raw = y.clone()
            x, y = normalize_batch(x, y, norm, device)
            loss = multitask_loss(model(x), y, y_raw, ep, iid_criterion, bce_criterion)
            opt.zero_grad(); loss.backward(); opt.step()
            run_train += loss.item() * x.size(0)
        train_loss = run_train / len(train_loader.dataset)

        model.eval()
        run_val = 0.0
        with torch.no_grad():
            for x, y, ep in val_loader:
                x, y, ep = x.to(device), y.to(device), ep.to(device)
                y_raw = y.clone()
                x, y = normalize_batch(x, y, norm, device)
                run_val += multitask_loss(model(x), y, y_raw, ep, iid_criterion, bce_criterion).item() * x.size(0)
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
    plt.xlabel("Epoch"); plt.ylabel("Huber loss")
    plt.title("Profile -> IID training")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    save_plot("training_curves"); plt.close()


def plot_scatter(y_true, y_pred, ep_true, ep_pred_prob, min_dist):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # IID — coloured by min_dist(profile)
    ax = axes[0]
    yt, yp = y_true[:, 0], y_pred[:, 0]
    lo, hi = float(min(np.min(yt), np.min(yp))), float(max(np.max(yt), np.max(yp)))
    sc = ax.scatter(yt, yp, c=min_dist, cmap="plasma_r", s=10, alpha=0.5,
                    vmin=np.percentile(min_dist, 2), vmax=np.percentile(min_dist, 98))
    plt.colorbar(sc, ax=ax, label="min profile dist (mm)")
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    p = pearson_corr(yt, yp)
    s = spearman_corr(yt, yp)
    bias = float(np.mean(yp - yt))
    sign_acc = float(np.mean(np.sign(yp) == np.sign(yt)))
    pos_acc = float(np.mean(yp[yt >= 0] >= 0)) if np.any(yt >= 0) else float("nan")
    neg_acc = float(np.mean(yp[yt < 0] < 0)) if np.any(yt < 0) else float("nan")
    ax.set_xlabel("True IID (dB)"); ax.set_ylabel("Pred IID (dB)")
    ax.set_title(
        f"corrected_iid (dB)\nPearson={p:.3f}, Spearman={s:.3f}, Bias={bias:+.3f} dB\n"
        f"SignAcc={sign_acc:.3f}  PosAcc={pos_acc:.3f}  NegAcc={neg_acc:.3f}"
    )
    ax.grid(True, alpha=0.3)

    # Echo present classification
    ep_pred_bin = (ep_pred_prob >= 0.5).astype(np.float32)
    acc = float(np.mean(ep_pred_bin == ep_true))
    n_pos, n_neg = int(np.sum(ep_true > 0.5)), int(np.sum(ep_true <= 0.5))
    axes[1].scatter(ep_true, ep_pred_prob, s=10, alpha=0.3)
    axes[1].axhline(0.5, color='r', linestyle='--', linewidth=1)
    axes[1].set_xlabel("True echo_present"); axes[1].set_ylabel("Pred echo_present prob")
    axes[1].set_title(f"echo_present\nAcc={acc:.3f}  N_echo={n_pos}  N_no_echo={n_neg}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout(); save_plot("test_scatter"); plt.close(fig)


def main():
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Loading data...")
    dc = DataProcessor.DataCollection(sessions)
    profiles, _ = dc.load_profiles(opening_angle=profile_opening_angle, steps=profile_steps)

    finite = np.isfinite(profiles).all(axis=1)
    profiles  = profiles[finite].astype(np.float32)
    print(f"Kept {len(profiles)} samples after filtering.")

    distance_m = dc.get_field('sonar_package', 'corrected_distance').astype(np.float32)[finite]
    iid        = dc.get_field('sonar_package', 'corrected_iid').astype(np.float32)[finite]
    distance_mm = distance_m * 1000.0
    targets     = iid.reshape(-1, 1)   # IID is the only regression target
    valid_target = np.isfinite(targets[:, 0]) & np.isfinite(distance_mm)
    profiles    = profiles[valid_target]
    targets     = targets[valid_target]
    distance_mm = distance_mm[valid_target]
    print(f"Kept {len(profiles)} samples after target filtering.")

    echo_present = (distance_mm < no_echo_min_distance_mm).astype(np.float32)
    n_echo = int(np.sum(echo_present))
    print(f"Echo present: {n_echo}/{len(echo_present)} samples ({100*n_echo/len(echo_present):.1f}% echo, {100*(1-n_echo/len(echo_present)):.1f}% no-echo)")

    # Flip augmentation: mirror profile + negate IID, balancing pos/neg IID perfectly.
    profiles_flipped     = profiles[:, ::-1].copy()
    targets_flipped      = targets.copy(); targets_flipped[:, 0] *= -1.0
    echo_present_flipped = echo_present.copy()
    profiles_aug     = np.concatenate([profiles,      profiles_flipped],      axis=0)
    targets_aug      = np.concatenate([targets,       targets_flipped],       axis=0)
    echo_present_aug = np.concatenate([echo_present,  echo_present_flipped],  axis=0)
    print(f"Profile flip augmentation: {len(profiles)} -> {len(profiles_aug)} samples.")

    n_aug  = len(profiles_aug)
    idx    = np.random.permutation(n_aug)
    n_val  = max(1, int(val_fraction * n_aug))
    val_mask   = np.zeros(n_aug, dtype=bool); val_mask[idx[:n_val]]  = True
    train_mask = ~val_mask
    print(f"Random split: {train_mask.sum()} train, {val_mask.sum()} val ({val_fraction:.0%} holdout).")

    # Keep original distances for scatter plot colouring (before normalisation).
    min_dist_test = np.min(profiles_aug, axis=1)

    # Normalise each profile by its mean so the CNN sees shape (relative asymmetry)
    # rather than absolute distances. This makes IID prediction scale-invariant.
    profile_means = np.mean(profiles_aug, axis=1, keepdims=True)
    profiles_aug  = profiles_aug / np.clip(profile_means, 1e-6, None)

    ds_train = ProfileTargetDataset(profiles_aug[train_mask], targets_aug[train_mask], echo_present_aug[train_mask])
    ds_val   = ProfileTargetDataset(profiles_aug[val_mask],   targets_aug[val_mask],   echo_present_aug[val_mask])
    ds_test  = ProfileTargetDataset(profiles_aug,             targets_aug,             echo_present_aug)

    norm = compute_norm_stats(ds_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, pin_memory=pin)

    model = ProfileCNN().to(device)
    history = train_model(model, train_loader, val_loader, norm)
    ckpt = torch.load(f"{output_dir}/best_model_pytorch.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    yv_true, yv_pred, _, _ = collect_predictions(model, val_loader, norm)
    calibration = fit_calibration(yv_true, yv_pred)
    print(f"Validation calibration: iid y={calibration[0]['slope']:.4f}*pred+{calibration[0]['intercept']:.2f}")

    y_true, y_pred_raw, ep_true, ep_pred = collect_predictions(model, test_loader, norm)
    y_pred = apply_calibration(y_pred_raw, calibration)

    iid_true, iid_pred = y_true[:, 0], y_pred[:, 0]
    true_pos = iid_true >= 0.0; true_neg = iid_true < 0.0
    pred_pos = iid_pred >= 0.0; pred_neg = iid_pred < 0.0
    tp = int(np.sum(true_pos & pred_pos)); tn = int(np.sum(true_neg & pred_neg))
    fp = int(np.sum(true_neg & pred_pos)); fn = int(np.sum(true_pos & pred_neg))
    n_sign   = int(len(iid_true))
    acc_pos  = float(tp / max(1, int(np.sum(true_pos))))
    acc_neg  = float(tn / max(1, int(np.sum(true_neg))))
    sign_acc = float((tp + tn) / max(1, n_sign))

    ep_pred_bin = (ep_pred >= 0.5).astype(np.float32)
    ep_acc = float(np.mean(ep_pred_bin == ep_true))
    ep_tp  = int(np.sum((ep_true > 0.5) & (ep_pred_bin > 0.5)))
    ep_tn  = int(np.sum((ep_true <= 0.5) & (ep_pred_bin <= 0.5)))
    ep_fp  = int(np.sum((ep_true <= 0.5) & (ep_pred_bin > 0.5)))
    ep_fn  = int(np.sum((ep_true > 0.5) & (ep_pred_bin <= 0.5)))

    metrics = {
        "iid": {
            "rmse_db": float(np.sqrt(np.mean((iid_pred - iid_true) ** 2))),
            "mae_db":  float(np.mean(np.abs(iid_pred - iid_true))),
            "bias_db": float(np.mean(iid_pred - iid_true)),
            "pearson":  pearson_corr(iid_true, iid_pred),
            "spearman": spearman_corr(iid_true, iid_pred),
            "sign_accuracy": sign_acc,
            "sign_confusion_counts": {
                "true_pos_pred_pos": tp, "true_pos_pred_neg": fn,
                "true_neg_pred_pos": fp, "true_neg_pred_neg": tn,
            },
            "sign_accuracy_true_pos": acc_pos,
            "sign_accuracy_true_neg": acc_neg,
            "sign_n": n_sign,
        },
        "echo_present": {
            "accuracy": ep_acc,
            "tp": ep_tp, "tn": ep_tn, "fp": ep_fp, "fn": ep_fn,
            "n_echo":    int(np.sum(ep_true > 0.5)),
            "n_no_echo": int(np.sum(ep_true <= 0.5)),
        },
    }

    print("\nTest metrics:")
    print(
        f"IID: RMSE={metrics['iid']['rmse_db']:.3f} dB, "
        f"MAE={metrics['iid']['mae_db']:.3f} dB, Bias={metrics['iid']['bias_db']:.3f} dB, "
        f"Pearson={metrics['iid']['pearson']:.3f}, Spearman={metrics['iid']['spearman']:.3f}, "
        f"SignAcc={metrics['iid']['sign_accuracy']:.3f}"
    )
    print(
        f"IID sign confusion (N={n_sign}): "
        f"TP={tp}, FN={fn}, FP={fp}, TN={tn}, PosAcc={acc_pos:.3f}, NegAcc={acc_neg:.3f}"
    )
    print(
        f"Echo present: Acc={ep_acc:.3f}, "
        f"TP={ep_tp}, TN={ep_tn}, FP={ep_fp}, FN={ep_fn}, "
        f"N_echo={metrics['echo_present']['n_echo']}, N_no_echo={metrics['echo_present']['n_no_echo']}"
    )

    plot_training(history)
    plot_scatter(y_true, y_pred, ep_true, ep_pred, min_dist_test)

    params = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sessions": sessions,
        "profile_opening_angle": profile_opening_angle,
        "profile_steps": profile_steps,
        "val_fraction": val_fraction,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "l2_reg": l2_reg,
        "conv_channels": conv_channels,
        "conv_kernel": conv_kernel,
        "fc_hidden": fc_hidden,
        "iid_huber_delta": iid_huber_delta,
        "echo_present_loss_weight": echo_present_loss_weight,
        "no_echo_min_distance_mm": no_echo_min_distance_mm,
        "num_train": len(ds_train),
        "num_val": len(ds_val),
        "num_test": len(ds_test),
        "metrics": metrics,
        "calibration": calibration,
        "norm_stats": {
            "x_mean": norm["x_mean"].tolist(),
            "x_std":  norm["x_std"].tolist(),
            "y_mean": norm["y_mean"].tolist(),
            "y_std":  norm["y_std"].tolist(),
        },
    }
    with open(f"{output_dir}/training_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
