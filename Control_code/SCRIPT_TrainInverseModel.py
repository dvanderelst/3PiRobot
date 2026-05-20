#!/usr/bin/env python3
"""
SCRIPT_TrainInverseModel.py

Train the two-headed cross-modal inverse on AcquisitionSessions data:
  Head A — wall vs pole classifier (2-way softmax).
  Head B — wall 3-slice depth profile (mean, log_var) per slice.
  Head C — pole azimuth (mean, log_var), in normalised [-1, 1] space.

Loss combines (cross-entropy on class) + (masked GNLL on wall slices, applied
only to wall-class samples) + (masked GNLL on pole azimuth, applied only to
pole-class samples). Per-component normalisation is by that component's mask
count, not by batch size, so class imbalance doesn't suppress the minority
regression signal.

Pole azimuth is normalised by dividing by CONE_HALF_DEG so the model outputs
land roughly in [-1, 1] and the antisymmetry-by-construction in
SonarSlicesUQ_TwoHeaded is preserved exactly (a fixed scaling, not a
learned shift+scale).

Outputs in SonarModel/  (prefix `inverse_`):
  inverse_best_model.pth
  inverse_feature_params.json
  inverse_results.json
  inverse_confusion.png            confusion matrix + per-class probability
  inverse_pole_azimuth_scatter.png pred vs true pole bearing
  inverse_wall_scatter.png         per-slice wall depth scatter (wall-class)
  inverse_calibration.png          per-slice + pole-az σ calibration
"""

import copy
import json
import os

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Library.AcquisitionSessionLoader import load_data_inverse
from Library.SonarModel import SonarSlicesUQ_TwoHeaded, SLICE_NAMES as _LIB_SLICE_NAMES


# ── Settings ──────────────────────────────────────────────────────────────────

ACQUISITION_SESSIONS = ["Acquisition01A"]
ACQUISITIONS_ROOT    = "AcquisitionSessions"

OPENING_ANGLE  = 270.0
PROFILE_STEPS  = 90
PROFILE_METHOD = "ray_center"
CONE_HALF_DEG  = 35.0   # also used to normalise pole azimuth

# Validation: hold out one quadrant per session, same convention as the
# wall-only trainer. With one session this gives ~25% val / 75% train.
VALIDATION_QUADRANTS = {
    "Acquisition01A": [0],
}

# Architecture (mirrors SCRIPT_TrainSonarModel.py defaults)
SONAR_CONV_CHANNELS = [8, 16]
SONAR_CONV_KERNEL   = 7
SONAR_POOL_OUT      = 8
SONAR_FC_HIDDEN     = 32
SONAR_HEAD_HIDDEN   = 16

# Loss weights. Tune by inspecting per-component val losses if imbalanced.
LOSS_W_CLASS = 1.0
LOSS_W_WALL  = 1.0
LOSS_W_POLE  = 1.0

LR             = 1e-3
BATCH_SIZE     = 64
EPOCHS         = 150
WARMUP_EPOCHS  = 20
LOG_VAR_MIN    = -6.0
LOG_VAR_MAX    = 4.0
SEED           = 42

CLASS_NAMES = ["wall", "pole"]

OUTPUT_DIR      = "SonarModel"
ARTIFACT_PREFIX = "inverse"

SLICE_NAMES = list(_LIB_SLICE_NAMES)


# ── Geometry ──────────────────────────────────────────────────────────────────

def profile_bin_centers(opening_angle, profile_steps):
    edges = np.linspace(-opening_angle / 2, opening_angle / 2, profile_steps + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def slice_masks(bin_centers, cone_half_deg):
    third = 2.0 * cone_half_deg / 3.0
    right_lo, right_hi = -cone_half_deg,                -cone_half_deg + third
    cent_lo,  cent_hi  = -cone_half_deg + third,        -cone_half_deg + 2.0 * third
    left_lo,  left_hi  = -cone_half_deg + 2.0 * third,  cone_half_deg
    return [
        (bin_centers >= right_lo) & (bin_centers <  right_hi),
        (bin_centers >= cent_lo)  & (bin_centers <  cent_hi),
        (bin_centers >= left_lo)  & (bin_centers <= left_hi),
    ]


def compute_slice_targets(profiles, bin_centers, cone_half_deg):
    """Per-slice min wall distance from the full profile.

    Wall-class samples will use these as regression targets. Pole-class
    samples carry NaN slice targets that the masked loss ignores.
    """
    masks = slice_masks(bin_centers, cone_half_deg)
    cols = []
    for m in masks:
        sub = profiles[:, m]
        all_nan = np.isnan(sub).all(axis=1)
        col = np.where(all_nan, np.nan,
                       np.nanmin(np.where(all_nan[:, None], np.inf, sub), axis=1))
        cols.append(col)
    return np.stack(cols, axis=1).astype(np.float32)


# ── Loss ──────────────────────────────────────────────────────────────────────

def masked_gnll(pred_mean, pred_log_var, target, mask,
                in_warmup,
                log_var_min=LOG_VAR_MIN, log_var_max=LOG_VAR_MAX):
    """GNLL averaged over masked elements only. Returns 0 when mask is empty
    so it contributes no gradient. During warmup, use MSE on the mean only
    to stabilise before log_var carries gradient."""
    n = float(mask.sum().item())
    if n == 0:
        return pred_mean.sum() * 0.0  # zero tensor that backprops cleanly
    m = mask.float()
    if in_warmup:
        return (((pred_mean - target) ** 2) * m).sum() / n
    log_var = pred_log_var.clamp(log_var_min, log_var_max)
    inv_var = torch.exp(-log_var)
    per = 0.5 * (log_var + (target - pred_mean) ** 2 * inv_var)
    return (per * m).sum() / n


# ── Data ──────────────────────────────────────────────────────────────────────

def load_and_filter():
    """Returns sonar, slice_targets, classes, pole_az_deg, quads, sess, bin_centers
    with NaN-class rows dropped (cone-empty pings)."""
    sonar, profiles, classes, pole_az, quads, sess, bin_centers = load_data_inverse(
        ACQUISITION_SESSIONS,
        acquisitions_root=ACQUISITIONS_ROOT,
        opening_angle=OPENING_ANGLE,
        profile_steps=PROFILE_STEPS,
        profile_method=PROFILE_METHOD,
        cone_half_deg=CONE_HALF_DEG,
    )
    keep = ~np.isnan(classes)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"  dropping {n_dropped} pings with empty cone")
    slice_t = compute_slice_targets(profiles, bin_centers, CONE_HALF_DEG)
    return (sonar[keep], slice_t[keep], classes[keep].astype(np.int64),
            pole_az[keep], quads[keep], sess[keep], bin_centers)


def split_indices(quads, sess):
    is_val = np.zeros(len(quads), dtype=bool)
    for s_name, val_q in VALIDATION_QUADRANTS.items():
        is_val |= (sess == s_name) & np.isin(quads, list(val_q))
    return is_val


# ── Training ──────────────────────────────────────────────────────────────────

def make_loader(s, t_wall, cls, t_pole_n, batch_size, shuffle):
    L = torch.as_tensor(s[..., 0],   dtype=torch.float32)
    R = torch.as_tensor(s[..., 1],   dtype=torch.float32)
    Tw = torch.as_tensor(t_wall,     dtype=torch.float32)
    C  = torch.as_tensor(cls,        dtype=torch.long)
    Tp = torch.as_tensor(t_pole_n,   dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(L, R, Tw, C, Tp)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def combined_loss(out, T_wall_n, C, T_pole_n, in_warmup):
    """Returns total loss + per-component breakdown for logging."""
    # Class loss: CE on (B, 2) logits against integer labels
    loss_class = F.cross_entropy(out["class_logits"], C)

    # Per-sample masks
    wall_mask = (C == 0)
    pole_mask = (C == 1)

    # Wall slice GNLL — per-sample mask AND per-slice NaN-safe
    # (a wall-class sample may still have a slice with no wall in that bin)
    wall_losses = []
    for i, name in enumerate(SLICE_NAMES):
        target = T_wall_n[:, i]
        valid_slice = ~torch.isnan(target)
        m = wall_mask & valid_slice
        safe_target = torch.where(valid_slice, target, torch.zeros_like(target))
        wall_losses.append(masked_gnll(
            out[f"{name}_mean"].squeeze(1),
            out[f"{name}_log_var"].squeeze(1),
            safe_target, m, in_warmup,
        ))
    loss_wall = sum(wall_losses)

    # Pole azimuth GNLL
    loss_pole = masked_gnll(
        out["pole_az_mean"].squeeze(1),
        out["pole_az_log_var"].squeeze(1),
        T_pole_n, pole_mask, in_warmup,
    )

    total = (LOSS_W_CLASS * loss_class
             + LOSS_W_WALL * loss_wall
             + LOSS_W_POLE * loss_pole)
    return total, {"class": loss_class.detach(),
                   "wall":  loss_wall.detach() if torch.is_tensor(loss_wall) else torch.tensor(0.0),
                   "pole":  loss_pole.detach()}


def train(tr_s, tr_tw, tr_c, tr_tp_n,
          va_s, va_tw, va_c, va_tp_n,
          sonar_stats, wall_stats, device, save_path):
    s_mean, s_std = sonar_stats
    t_mean, t_std = wall_stats

    def norm_sonar(x): return ((x - s_mean) / s_std).astype(np.float32)
    def norm_wall(x):  return ((x - t_mean) / t_std).astype(np.float32)

    train_loader = make_loader(norm_sonar(tr_s), norm_wall(tr_tw),
                               tr_c, tr_tp_n, BATCH_SIZE, True)
    val_loader   = make_loader(norm_sonar(va_s), norm_wall(va_tw),
                               va_c, va_tp_n, BATCH_SIZE, False)

    torch.manual_seed(SEED)
    model = SonarSlicesUQ_TwoHeaded(
        samples=tr_s.shape[1],
        conv_channels=SONAR_CONV_CHANNELS, conv_kernel=SONAR_CONV_KERNEL,
        pool_out=SONAR_POOL_OUT, fc_hidden=SONAR_FC_HIDDEN,
        head_hidden=SONAR_HEAD_HIDDEN,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_state, best_val, best_epoch = None, float('inf'), -1
    for epoch in range(1, EPOCHS + 1):
        in_warmup = epoch <= WARMUP_EPOCHS
        model.train()
        for L, R, Tw, C, Tp in train_loader:
            L, R, Tw, C, Tp = (L.to(device), R.to(device),
                               Tw.to(device), C.to(device), Tp.to(device))
            out = model(L, R)
            loss, _ = combined_loss(out, Tw, C, Tp, in_warmup)
            opt.zero_grad(); loss.backward(); opt.step()

        if not in_warmup:
            model.eval()
            v_total = []
            v_breakdown = {"class": [], "wall": [], "pole": []}
            with torch.no_grad():
                for L, R, Tw, C, Tp in val_loader:
                    L, R, Tw, C, Tp = (L.to(device), R.to(device),
                                       Tw.to(device), C.to(device), Tp.to(device))
                    out = model(L, R)
                    total, parts = combined_loss(out, Tw, C, Tp, in_warmup=False)
                    v_total.append(float(total.item()))
                    for k in v_breakdown:
                        v_breakdown[k].append(float(parts[k].item()))
            val_total = float(np.mean(v_total))
            if val_total < best_val:
                best_val, best_epoch = val_total, epoch
                best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1 or epoch == WARMUP_EPOCHS + 1:
            tag = " *" if epoch == best_epoch else (" [warmup]" if in_warmup else "")
            if best_val < float('inf'):
                bk = {k: float(np.mean(v)) for k, v in v_breakdown.items()} if not in_warmup else None
                msg = (f"  Epoch {epoch:3d}/{EPOCHS}{tag}"
                       f"  val={best_val:+.4f}")
                if bk:
                    msg += f"  (class={bk['class']:+.3f}  wall={bk['wall']:+.3f}  pole={bk['pole']:+.3f})"
                print(msg)
            else:
                print(f"  Epoch {epoch:3d}/{EPOCHS}{tag}")

    model.load_state_dict(best_state)
    model.eval()
    torch.save({'model_state_dict': best_state}, save_path)
    print(f"  Best epoch: {best_epoch}  val total = {best_val:.4f}")
    return model, best_val, best_epoch


# ── Prediction & evaluation ───────────────────────────────────────────────────

def predict(model, sonar, sonar_stats, wall_stats, device):
    s_mean, s_std = sonar_stats
    t_mean, t_std = wall_stats
    s = ((sonar - s_mean) / s_std).astype(np.float32)
    L = torch.as_tensor(s[..., 0], dtype=torch.float32).to(device)
    R = torch.as_tensor(s[..., 1], dtype=torch.float32).to(device)

    out_means_n = {k: [] for k in SLICE_NAMES}
    out_logv_n  = {k: [] for k in SLICE_NAMES}
    cls_logits  = []
    pole_mean_n = []
    pole_logv   = []
    with torch.no_grad():
        for st in range(0, len(L), 256):
            ed = min(st + 256, len(L))
            o = model(L[st:ed], R[st:ed])
            for k in SLICE_NAMES:
                out_means_n[k].append(o[f"{k}_mean"].cpu().squeeze(1).numpy())
                out_logv_n[k].append(o[f"{k}_log_var"].cpu().squeeze(1).numpy())
            cls_logits.append(o["class_logits"].cpu().numpy())
            pole_mean_n.append(o["pole_az_mean"].cpu().squeeze(1).numpy())
            pole_logv.append(o["pole_az_log_var"].cpu().squeeze(1).numpy())

    means_n = np.stack([np.concatenate(out_means_n[k]) for k in SLICE_NAMES], axis=1)
    logv_n  = np.stack([np.concatenate(out_logv_n[k])  for k in SLICE_NAMES], axis=1)
    logv_n  = np.clip(logv_n, LOG_VAR_MIN, LOG_VAR_MAX)
    wall_pred_mean = means_n * t_std + t_mean
    wall_pred_std  = np.exp(logv_n / 2.0) * t_std

    cls_logits = np.concatenate(cls_logits, axis=0)
    cls_probs  = np.exp(cls_logits - cls_logits.max(axis=1, keepdims=True))
    cls_probs  = cls_probs / cls_probs.sum(axis=1, keepdims=True)
    cls_pred   = cls_probs.argmax(axis=1)

    pole_mean_n = np.concatenate(pole_mean_n)
    pole_logv   = np.clip(np.concatenate(pole_logv), LOG_VAR_MIN, LOG_VAR_MAX)
    pole_pred_az_deg  = pole_mean_n * CONE_HALF_DEG  # de-normalise
    pole_pred_az_std  = np.exp(pole_logv / 2.0) * CONE_HALF_DEG

    return {
        "wall_pred_mean": wall_pred_mean,
        "wall_pred_std":  wall_pred_std,
        "cls_logits":     cls_logits,
        "cls_probs":      cls_probs,
        "cls_pred":       cls_pred,
        "pole_pred_az_deg": pole_pred_az_deg,
        "pole_pred_az_std": pole_pred_az_std,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion(true_cls, pred_cls, probs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: confusion matrix
    ax = axes[0]
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(true_cls, pred_cls):
        cm[int(t), int(p)] += 1
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, weight="bold")
    acc = float(np.mean(pred_cls == true_cls))
    ax.set_title(f"Confusion matrix  (val acc = {acc:.2%})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: predicted pole-class probability by true class
    ax = axes[1]
    p_pole = probs[:, 1]
    bins = np.linspace(0, 1, 21)
    ax.hist(p_pole[true_cls == 0], bins=bins, alpha=0.6, color="#377eb8",
            label=f"true wall (n={int((true_cls == 0).sum())})")
    ax.hist(p_pole[true_cls == 1], bins=bins, alpha=0.6, color="#e41a1c",
            label=f"true pole (n={int((true_cls == 1).sum())})")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("predicted P(pole)")
    ax.set_ylabel("count")
    ax.set_title("Predicted pole probability by true class")
    ax.legend()

    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def plot_pole_azimuth(true_az, pred_az, pred_std, out_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(true_az, pred_az, c=pred_std, s=22, alpha=0.7, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="predicted σ (deg)")
    lo = min(true_az.min(), pred_az.min()) - 2
    hi = max(true_az.max(), pred_az.max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    if len(true_az):
        rmse = float(np.sqrt(((pred_az - true_az) ** 2).mean()))
        mae  = float(np.abs(pred_az - true_az).mean())
        ax.set_title(f"Pole azimuth (pole-class val samples)\n"
                     f"RMSE={rmse:.2f}°  MAE={mae:.2f}°  n={len(true_az)}")
    else:
        ax.set_title("Pole azimuth — no pole-class samples in val")
    ax.set_xlabel("true az (deg)")
    ax.set_ylabel("pred az (deg)")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def plot_wall_scatter(true_w, pred_w, std_w, wall_mask, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    n_wall = int(wall_mask.sum())
    for i, (ax, name) in enumerate(zip(axes, SLICE_NAMES)):
        t = true_w[wall_mask, i]
        m = pred_w[wall_mask, i]
        s = std_w[wall_mask, i]
        valid = ~np.isnan(t)
        t, m, s = t[valid], m[valid], s[valid]
        if len(t) == 0:
            ax.set_title(f"{name.capitalize()} — no valid wall samples")
            continue
        sc = ax.scatter(t, m, c=s, s=10, alpha=0.55, cmap="viridis")
        plt.colorbar(sc, ax=ax, label="σ (mm)")
        lo = float(min(t.min(), m.min()))
        hi = float(max(t.max(), m.max())) * 1.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        rmse = float(np.sqrt(((m - t) ** 2).mean()))
        mae  = float(np.abs(m - t).mean())
        ax.set_title(f"{name.capitalize()}   RMSE={rmse:.0f}   MAE={mae:.0f} mm   n={len(t)}")
        ax.set_xlabel("true (mm)")
        if i == 0: ax.set_ylabel("predicted (mm)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Wall depth (wall-class samples only, n={n_wall})", y=1.02)
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


def plot_calibration(true_w, pred_w, std_w, true_az, pred_az, pred_az_std,
                     wall_mask, out_path, n_bins=10):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    def cal_panel(ax, t, m, s, title, unit):
        residuals = m - t
        sorted_idx = np.argsort(s)
        bin_size = max(len(t) // n_bins, 1)
        bps, bemp = [], []
        for k in range(n_bins):
            start = k * bin_size
            end = (k + 1) * bin_size if k < n_bins - 1 else len(t)
            idx = sorted_idx[start:end]
            bps.append(float(s[idx].mean()))
            bemp.append(float(np.sqrt((residuals[idx] ** 2).mean())))
        ax.scatter(bps, bemp, s=70, color="steelblue", edgecolor="white", linewidth=1)
        lo = min(min(bps), min(bemp)) * 0.85 if bps else 0
        hi = max(max(bps), max(bemp)) * 1.15 if bps else 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"predicted σ ({unit})")
        ax.set_ylabel(f"empirical RMSE ({unit})")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)

    for i, name in enumerate(SLICE_NAMES):
        t = true_w[wall_mask, i]
        m = pred_w[wall_mask, i]
        s = std_w[wall_mask, i]
        valid = ~np.isnan(t)
        t, m, s = t[valid], m[valid], s[valid]
        if len(t) >= n_bins:
            cal_panel(axes[i], t, m, s, f"{name.capitalize()} σ calib", "mm")
        else:
            axes[i].set_title(f"{name.capitalize()} — too few samples")

    if len(true_az) >= n_bins:
        cal_panel(axes[3], true_az, pred_az, pred_az_std, "Pole-az σ calib", "deg")
    else:
        axes[3].set_title("Pole-az — too few samples")

    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)

    print("[1/5] Loading data")
    sonar, slice_t, classes, pole_az_deg, quads, sess, bin_centers = load_and_filter()
    print(f"  {len(sonar)} pings retained, "
          f"wall={int((classes == 0).sum())}, pole={int((classes == 1).sum())}")

    pole_az_n = (pole_az_deg / CONE_HALF_DEG).astype(np.float32)
    # NaN azimuth (wall-class samples) is fine — the masked loss ignores it.
    # But torch.as_tensor doesn't like NaN propagation in CE so we fill with 0.
    pole_az_n_safe = np.where(np.isnan(pole_az_n), 0.0, pole_az_n).astype(np.float32)

    is_val = split_indices(quads, sess)
    tr_s, va_s         = sonar[~is_val],         sonar[is_val]
    tr_tw, va_tw       = slice_t[~is_val],       slice_t[is_val]
    tr_c, va_c         = classes[~is_val],       classes[is_val]
    tr_tp_n, va_tp_n   = pole_az_n_safe[~is_val], pole_az_n_safe[is_val]
    print(f"  train: {len(tr_s)} (wall={int((tr_c==0).sum())}, pole={int((tr_c==1).sum())})")
    print(f"  val:   {len(va_s)} (wall={int((va_c==0).sum())}, pole={int((va_c==1).sum())})")

    s_mean = float(tr_s.mean()); s_std = max(float(tr_s.std()), 1e-8)
    # Wall target stats: pool over wall-class training rows + non-NaN bins
    tr_wall_only = tr_tw[tr_c == 0]
    valid = ~np.isnan(tr_wall_only)
    t_mean = float(tr_wall_only[valid].mean()) if valid.any() else 0.0
    t_std  = max(float(tr_wall_only[valid].std()), 1e-8) if valid.any() else 1.0
    print(f"  sonar  mean={s_mean:.0f}, std={s_std:.0f}")
    print(f"  wall   target mean={t_mean:.0f} mm, std={t_std:.0f} mm")
    print(f"  pole   az normalised by /{CONE_HALF_DEG:.0f}°")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_best_model.pth")
    print(f"\n[2/5] Training on {device} ({EPOCHS} epochs, {WARMUP_EPOCHS} warmup)")
    model, best_val, best_epoch = train(
        tr_s, tr_tw, tr_c, tr_tp_n,
        va_s, va_tw, va_c, va_tp_n,
        (s_mean, s_std), (t_mean, t_std), device, save_path)

    print("\n[3/5] Evaluating on val")
    pred = predict(model, va_s, (s_mean, s_std), (t_mean, t_std), device)

    # Class metrics
    cls_acc = float(np.mean(pred["cls_pred"] == va_c))
    per_class = {}
    for ci, name in enumerate(CLASS_NAMES):
        truth = (va_c == ci)
        if truth.any():
            prec = float(np.mean(va_c[pred["cls_pred"] == ci] == ci)) if (pred["cls_pred"] == ci).any() else 0.0
            rec  = float(np.mean(pred["cls_pred"][truth] == ci))
            per_class[name] = {"precision": prec, "recall": rec, "n_true": int(truth.sum())}
    print(f"  Class accuracy: {cls_acc:.3f}")
    for name, m in per_class.items():
        print(f"    {name:>5}: precision={m['precision']:.3f}  recall={m['recall']:.3f}  n_true={m['n_true']}")

    # Pole azimuth metrics (pole-class val samples only)
    pole_val = (va_c == 1)
    pole_metrics = {}
    if pole_val.any():
        t_az = pole_az_deg[is_val][pole_val]
        p_az = pred["pole_pred_az_deg"][pole_val]
        residuals_az = p_az - t_az
        pole_metrics = {
            "n": int(pole_val.sum()),
            "rmse_deg": float(np.sqrt((residuals_az ** 2).mean())),
            "mae_deg":  float(np.abs(residuals_az).mean()),
            "pred_std_median_deg": float(np.median(pred["pole_pred_az_std"][pole_val])),
        }
        print(f"  Pole azimuth (n={pole_metrics['n']}): "
              f"RMSE={pole_metrics['rmse_deg']:.2f}°  MAE={pole_metrics['mae_deg']:.2f}°  "
              f"σ_med={pole_metrics['pred_std_median_deg']:.2f}°")

    # Wall slice metrics (wall-class val samples, non-NaN)
    wall_val = (va_c == 0)
    wall_metrics = {}
    for i, name in enumerate(SLICE_NAMES):
        t = va_tw[wall_val, i]; m = pred["wall_pred_mean"][wall_val, i]; s = pred["wall_pred_std"][wall_val, i]
        valid = ~np.isnan(t)
        t, m, s = t[valid], m[valid], s[valid]
        if len(t):
            rmse = float(np.sqrt(((m - t) ** 2).mean()))
            mae  = float(np.abs(m - t).mean())
            wall_metrics[name] = {"n": int(len(t)), "rmse_mm": rmse, "mae_mm": mae,
                                  "pred_std_median_mm": float(np.median(s))}
            print(f"  Wall {name:>6} (n={len(t)}): RMSE={rmse:.0f} mm  MAE={mae:.0f} mm  σ_med={np.median(s):.0f}")

    print("\n[4/5] Saving plots")
    plot_confusion(va_c, pred["cls_pred"], pred["cls_probs"],
                   os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_confusion.png"))
    if pole_val.any():
        plot_pole_azimuth(pole_az_deg[is_val][pole_val],
                          pred["pole_pred_az_deg"][pole_val],
                          pred["pole_pred_az_std"][pole_val],
                          os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_pole_azimuth_scatter.png"))
    plot_wall_scatter(va_tw, pred["wall_pred_mean"], pred["wall_pred_std"],
                      wall_val, os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_wall_scatter.png"))
    if pole_val.any():
        plot_calibration(va_tw, pred["wall_pred_mean"], pred["wall_pred_std"],
                         pole_az_deg[is_val][pole_val],
                         pred["pole_pred_az_deg"][pole_val],
                         pred["pole_pred_az_std"][pole_val],
                         wall_val, os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_calibration.png"))

    print("\n[5/5] Saving params + results")
    feature_params = {
        "cone_half_deg":  CONE_HALF_DEG,
        "slice_definitions": {
            "left":   [-CONE_HALF_DEG,            -CONE_HALF_DEG + 2*CONE_HALF_DEG/3],
            "center": [-CONE_HALF_DEG + 2*CONE_HALF_DEG/3, CONE_HALF_DEG - 2*CONE_HALF_DEG/3],
            "right":  [CONE_HALF_DEG - 2*CONE_HALF_DEG/3,  CONE_HALF_DEG],
        },
        "sonar_norm":     {"mean": s_mean, "std": s_std},
        "target_norm":    {"mean": t_mean, "std": t_std},
        "pole_az_norm":   {"divide_by_deg": CONE_HALF_DEG},
        "class_names":    CLASS_NAMES,
        "envelope_norm":  {"kind": "none"},
        "log_var_clamp":  [LOG_VAR_MIN, LOG_VAR_MAX],
        "architecture": {
            "samples":       int(sonar.shape[1]),
            "conv_channels": SONAR_CONV_CHANNELS,
            "conv_kernel":   SONAR_CONV_KERNEL,
            "pool_out":      SONAR_POOL_OUT,
            "fc_hidden":     SONAR_FC_HIDDEN,
            "head_hidden":   SONAR_HEAD_HIDDEN,
            "model_class":   "SonarSlicesUQ_TwoHeaded",
            "n_classes":     len(CLASS_NAMES),
        },
        "profile": {
            "opening_angle":  OPENING_ANGLE,
            "profile_steps":  PROFILE_STEPS,
            "profile_method": PROFILE_METHOD,
        },
        "loss_weights": {"class": LOSS_W_CLASS, "wall": LOSS_W_WALL, "pole": LOSS_W_POLE},
    }
    with open(os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_feature_params.json"), "w") as f:
        json.dump(feature_params, f, indent=2)

    results = {
        "metrics": {
            "best_epoch":   best_epoch,
            "val_total":    best_val,
            "class_acc":    cls_acc,
            "per_class":    per_class,
            "pole_az":      pole_metrics,
            "wall_per_slice": wall_metrics,
        },
        "data": {
            "n_train": int(len(tr_s)), "n_val": int(len(va_s)),
            "validation_quadrants": VALIDATION_QUADRANTS,
            "sessions": ACQUISITION_SESSIONS,
        },
    }
    with open(os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Artifacts in {OUTPUT_DIR}/  (prefix '{ARTIFACT_PREFIX}_'):")
    print(f"  {ARTIFACT_PREFIX}_best_model.pth, {ARTIFACT_PREFIX}_feature_params.json,"
          f" {ARTIFACT_PREFIX}_results.json")
    print(f"  {ARTIFACT_PREFIX}_confusion.png, {ARTIFACT_PREFIX}_pole_azimuth_scatter.png,")
    print(f"  {ARTIFACT_PREFIX}_wall_scatter.png, {ARTIFACT_PREFIX}_calibration.png")


if __name__ == "__main__":
    main()
