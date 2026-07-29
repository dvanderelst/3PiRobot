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
from Library.SonarModel import (
    SonarSlicesUQ_TwoHeaded, SonarSlicesUQ_Wall3, SLICE_NAMES as _LIB_SLICE_NAMES,
)


# ── Settings ──────────────────────────────────────────────────────────────────

ACQUISITION_SESSIONS = ["Acquisition01A", "Acquisition02A","Acquisition03A","Acquisition04A","Acquisition05A"]
ACQUISITIONS_ROOT    = "AcquisitionSessions"

OPENING_ANGLE  = 270.0
PROFILE_STEPS  = 90
PROFILE_METHOD = "ray_center"
CONE_HALF_DEG  = 35.0   # also used to normalise pole azimuth
MAX_RANGE_MM   = 1000.0  # drop pings whose nearest reflector is beyond this.
                         # Restricts the task to the close-range regime where
                         # the narrowband sonar carries discriminative pole
                         # signal (mid-range 1-1.7 m is the empirical dead zone).

# 4-fold cross-validation: for each q in CV_QUADRANTS, hold out that quadrant
# from every session as the val set and train on the rest. Same per-fold
# convention as the wall-only trainer; here the loop is wired directly in main()
# rather than requiring four separate runs.
CV_QUADRANTS = [0, 1, 2, 3]

# Deployment training: a SINGLE spatial holdout, not cross-validation. We make no
# generalization claim for the inverse (its real test is the behavioral
# experiments); this held-out region is purely an overfitting guard, and we report
# in-sample vs held-out side by side. Per session, the HOLDOUT_FRAC of pings
# nearest a seeded random anchor pose form a contiguous held-out patch; the model
# trains on the rest with early stopping on the patch. The deployed inverse_ model
# (fold name "deploy") comes from this path, not from the CV folds. main_deploy()
# is the script entry point; main() (the CV path) is kept for diagnostics/EXPT.
HOLDOUT_FRAC = 0.15
HOLDOUT_SEED = 0   # chosen for balanced per-session pole coverage in the holdout
                   # (>=14 poles/session), before training; not tuned on results.

# Canonical inverse architecture. B = SonarSlicesUQ_Wall3(symmetric=True): one
# 3-output wall head, applied to both ear orderings (LR/RL) with the flanking
# bins swapped and the center averaged, so the left-right mirror symmetry of the
# sensor is enforced uniformly across all three slices. Chosen over the base
# side-head + z_sym-center arrangement (SonarSlicesUQ_TwoHeaded): performance is
# within per-fold noise (see Performance notes 2026-06-22) but the single-head
# construction is simpler to state and matches the symmetry logic already used
# by the class and pole-azimuth heads. EXPT_head_variants.py overrides these to
# sweep architectures. MODEL_CLASS_NAME is recorded in feature_params so
# InverseModel.load reconstructs the right class at deploy time.
MODEL_CLASS      = SonarSlicesUQ_Wall3
MODEL_KWARGS     = {"symmetric": True, "pole_dist_head": True}
MODEL_CLASS_NAME = "SonarSlicesUQ_Wall3"

# Architecture (carries forward the wall-only SonarSlicesUQ defaults)
SONAR_CONV_CHANNELS = [8, 16]
SONAR_CONV_KERNEL   = 7
SONAR_POOL_OUT      = 8
SONAR_FC_HIDDEN     = 32
SONAR_HEAD_HIDDEN   = 16

# Loss weights. Tune by inspecting per-component val losses if imbalanced.
LOSS_W_CLASS = 1.0
LOSS_W_WALL  = 1.0
LOSS_W_POLE  = 1.0
# Pole RANGE term. The controller's terminal approach stops on this output, so
# it has to be trained, not inferred from the wall head (which is masked to
# wall-class pings). Normalised by MAX_RANGE_MM, like pole azimuth is by
# CONE_HALF_DEG, so all regression targets sit on a comparable scale and one
# weight of 1.0 does not silently dominate.
LOSS_W_POLE_DIST = 1.0

LR             = 1e-3
BATCH_SIZE     = 64
EPOCHS         = 60
WARMUP_EPOCHS  = 20
LOG_VAR_MIN    = -6.0
LOG_VAR_MAX    = 4.0
SEED           = 42

CLASS_NAMES = ["wall", "pole", "none"]
WALL_CLASS, POLE_CLASS, NONE_CLASS = 0, 1, 2   # "none" = nothing within MAX_RANGE_MM

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

def load_and_filter(with_poses: bool = False):
    """Returns sonar, slice_targets, classes (0=wall, 1=pole, 2=none), pole_az,
    pole_dist_mm, quads, sess, bin_centers (poses appended when with_poses=True).

    Cone-empty pings and pings whose nearest in-cone reflector is beyond
    MAX_RANGE_MM are RELABELLED to the 'none' class (nothing actionable within
    range) rather than dropped. This gives the classifier an explicit abstain
    option and confines the wall-slice / pole-azimuth regression to the in-range
    regime where the narrowband sonar is reliable (wall/pole masks key off
    classes 0/1, so 'none' samples feed only the cross-entropy)."""
    loaded = load_data_inverse(
        ACQUISITION_SESSIONS,
        acquisitions_root=ACQUISITIONS_ROOT,
        opening_angle=OPENING_ANGLE,
        profile_steps=PROFILE_STEPS,
        profile_method=PROFILE_METHOD,
        cone_half_deg=CONE_HALF_DEG,
        return_poses=with_poses,
    )
    if with_poses:
        sonar, profiles, classes, pole_az, near_dist, quads, sess, bin_centers, poses = loaded
    else:
        sonar, profiles, classes, pole_az, near_dist, quads, sess, bin_centers = loaded
    classes = np.asarray(classes, dtype=np.float64)
    empty = np.isnan(classes)
    if np.isfinite(MAX_RANGE_MM):
        in_range = np.isfinite(near_dist) & (near_dist <= MAX_RANGE_MM)
    else:
        in_range = np.isfinite(near_dist)
    none_mask = empty | (~in_range)
    labels = classes.copy()
    labels[none_mask] = NONE_CLASS
    print(f"  relabelled {int(none_mask.sum())} pings to 'none' "
          f"(empty_cone={int(empty.sum())}, "
          f"beyond_{MAX_RANGE_MM:.0f}mm={int((none_mask & ~empty).sum())})")
    slice_t = compute_slice_targets(profiles, bin_centers, CONE_HALF_DEG)
    # near_dist is the range to whichever reflector won the cone; for pole-class
    # pings that is the pole SURFACE distance (centre - radius), the same
    # quantity the deployed stop rule compares against. It was already computed
    # here for the 'none' relabelling and then discarded.
    base = (sonar, slice_t, labels.astype(np.int64),
            pole_az, np.asarray(near_dist, dtype=np.float32),
            quads, sess, bin_centers)
    if with_poses:
        return base + (poses,)
    return base


def split_indices(quads, sess, val_quadrants):
    # Guard against config drift: val_quadrants keys must exactly match the
    # loaded sessions. Otherwise a renamed/missing key silently leaks the whole
    # session into training (no holdout) or references nothing.
    sess_loaded = set(np.unique(sess).tolist())
    vq_keys = set(val_quadrants)
    extra   = vq_keys - sess_loaded
    missing = sess_loaded - vq_keys
    if extra or missing:
        msgs = []
        if extra:
            msgs.append(f"val_quadrants references sessions not loaded: "
                        f"{sorted(extra)}")
        if missing:
            msgs.append(f"Loaded sessions missing from val_quadrants "
                        f"(would leak entirely into training): {sorted(missing)}")
        raise ValueError(
            "val_quadrants keys must exactly match the loaded sessions.\n  "
            + "\n  ".join(msgs)
        )
    is_val = np.zeros(len(quads), dtype=bool)
    for s_name, val_q in val_quadrants.items():
        is_val |= (sess == s_name) & np.isin(quads, list(val_q))
    return is_val


# ── Training ──────────────────────────────────────────────────────────────────

def make_loader(s, t_wall, cls, t_pole_n, batch_size, shuffle, t_pdist_n=None):
    L = torch.as_tensor(s[..., 0],   dtype=torch.float32)
    R = torch.as_tensor(s[..., 1],   dtype=torch.float32)
    Tw = torch.as_tensor(t_wall,     dtype=torch.float32)
    C  = torch.as_tensor(cls,        dtype=torch.long)
    Tp = torch.as_tensor(t_pole_n,   dtype=torch.float32)
    # Zeros when the range head is off: the tuple shape stays fixed, and the
    # loss ignores it because combined_loss only reads it when the head exists.
    Td = torch.as_tensor(t_pdist_n if t_pdist_n is not None
                         else np.zeros_like(t_pole_n), dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(L, R, Tw, C, Tp, Td)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def combined_loss(out, T_wall_n, C, T_pole_n, in_warmup, T_pdist_n=None):
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

    # Pole range GNLL — same masking as azimuth (pole-class pings only)
    loss_pole_dist = torch.tensor(0.0, device=out["class_logits"].device)
    if "pole_dist_mean" in out and T_pdist_n is not None:
        loss_pole_dist = masked_gnll(
            out["pole_dist_mean"].squeeze(1),
            out["pole_dist_log_var"].squeeze(1),
            T_pdist_n, pole_mask, in_warmup,
        )

    total = (LOSS_W_CLASS * loss_class
             + LOSS_W_WALL * loss_wall
             + LOSS_W_POLE * loss_pole
             + LOSS_W_POLE_DIST * loss_pole_dist)
    return total, {"class": loss_class.detach(),
                   "wall":  loss_wall.detach() if torch.is_tensor(loss_wall) else torch.tensor(0.0),
                   "pole":  loss_pole.detach(),
                   "pole_dist": loss_pole_dist.detach()}


def train(tr_s, tr_tw, tr_c, tr_tp_n,
          va_s, va_tw, va_c, va_tp_n,
          sonar_stats, wall_stats, device, save_path,
          tr_td_n=None, va_td_n=None):
    s_mean, s_std = sonar_stats
    t_mean, t_std = wall_stats

    def norm_sonar(x): return ((x - s_mean) / s_std).astype(np.float32)
    def norm_wall(x):  return ((x - t_mean) / t_std).astype(np.float32)

    train_loader = make_loader(norm_sonar(tr_s), norm_wall(tr_tw),
                               tr_c, tr_tp_n, BATCH_SIZE, True, t_pdist_n=tr_td_n)
    val_loader   = make_loader(norm_sonar(va_s), norm_wall(va_tw),
                               va_c, va_tp_n, BATCH_SIZE, False, t_pdist_n=va_td_n)

    torch.manual_seed(SEED)
    model = MODEL_CLASS(
        samples=tr_s.shape[1],
        conv_channels=SONAR_CONV_CHANNELS, conv_kernel=SONAR_CONV_KERNEL,
        pool_out=SONAR_POOL_OUT, fc_hidden=SONAR_FC_HIDDEN,
        head_hidden=SONAR_HEAD_HIDDEN,
        n_classes=len(CLASS_NAMES),
        **MODEL_KWARGS,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_state, best_val, best_epoch = None, float('inf'), -1
    for epoch in range(1, EPOCHS + 1):
        in_warmup = epoch <= WARMUP_EPOCHS
        model.train()
        for L, R, Tw, C, Tp, Td in train_loader:
            L, R, Tw, C, Tp, Td = (L.to(device), R.to(device), Tw.to(device),
                                   C.to(device), Tp.to(device), Td.to(device))
            out = model(L, R)
            loss, _ = combined_loss(out, Tw, C, Tp, in_warmup, T_pdist_n=Td)
            opt.zero_grad(); loss.backward(); opt.step()

        if not in_warmup:
            model.eval()
            v_total = []
            v_breakdown = {"class": [], "wall": [], "pole": [], "pole_dist": []}
            with torch.no_grad():
                for L, R, Tw, C, Tp, Td in val_loader:
                    L, R, Tw, C, Tp, Td = (L.to(device), R.to(device), Tw.to(device),
                                           C.to(device), Tp.to(device), Td.to(device))
                    out = model(L, R)
                    total, parts = combined_loss(out, Tw, C, Tp, in_warmup=False,
                                                 T_pdist_n=Td)
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
                    msg += (f"  (class={bk['class']:+.3f}  wall={bk['wall']:+.3f}  "
                            f"pole={bk['pole']:+.3f}  pdist={bk.get('pole_dist', 0.0):+.3f})")
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
    pdist_mean_n, pdist_logv = [], []
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
            if "pole_dist_mean" in o:
                pdist_mean_n.append(o["pole_dist_mean"].cpu().squeeze(1).numpy())
                pdist_logv.append(o["pole_dist_log_var"].cpu().squeeze(1).numpy())

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

    pole_pred_dist_mm = pole_pred_dist_std = None
    if pdist_mean_n:
        pd_n  = np.concatenate(pdist_mean_n)
        pd_lv = np.clip(np.concatenate(pdist_logv), LOG_VAR_MIN, LOG_VAR_MAX)
        pole_pred_dist_mm  = pd_n * MAX_RANGE_MM        # de-normalise
        pole_pred_dist_std = np.exp(pd_lv / 2.0) * MAX_RANGE_MM

    return {
        "pole_pred_dist_mm":  pole_pred_dist_mm,
        "pole_pred_dist_std": pole_pred_dist_std,
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

    # Panel 1: confusion matrix (n×n over CLASS_NAMES)
    ax = axes[0]
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_cls, pred_cls):
        cm[int(t), int(p)] += 1
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(n):
        for j in range(n):
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
    colors = {"wall": "#377eb8", "pole": "#e41a1c", "none": "#4daf4a"}
    for ci, name in enumerate(CLASS_NAMES):
        sel = true_cls == ci
        if sel.any():
            ax.hist(p_pole[sel], bins=bins, alpha=0.55, color=colors.get(name),
                    label=f"true {name} (n={int(sel.sum())})")
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


# ── Per-fold runner ───────────────────────────────────────────────────────────

def run_fold(q, sonar, slice_t, classes, pole_az_deg, pole_az_n_safe,
             quads, sess, device, sub_prefix, pole_dist_mm=None,
             pole_dist_n_safe=None):
    """Train + evaluate one CV fold (hold out quadrant `q` from every session).

    Saves model, plots, and per-fold JSON to OUTPUT_DIR with file prefix
    `{ARTIFACT_PREFIX}_{sub_prefix}`. Returns a metrics dict.
    """
    val_quadrants = {s: [q] for s in ACQUISITION_SESSIONS}
    is_val = split_indices(quads, sess, val_quadrants)
    tr_s, va_s       = sonar[~is_val],         sonar[is_val]
    tr_tw, va_tw     = slice_t[~is_val],       slice_t[is_val]
    tr_c, va_c       = classes[~is_val],       classes[is_val]
    tr_tp_n, va_tp_n = pole_az_n_safe[~is_val], pole_az_n_safe[is_val]
    tr_td_n = va_td_n = None
    if pole_dist_n_safe is not None:
        tr_td_n, va_td_n = pole_dist_n_safe[~is_val], pole_dist_n_safe[is_val]
    print(f"  train: {len(tr_s)} (wall={int((tr_c==0).sum())}, "
          f"pole={int((tr_c==1).sum())}, none={int((tr_c==2).sum())})")
    print(f"  val:   {len(va_s)} (wall={int((va_c==0).sum())}, "
          f"pole={int((va_c==1).sum())}, none={int((va_c==2).sum())})")

    s_mean = float(tr_s.mean()); s_std = max(float(tr_s.std()), 1e-8)
    tr_wall_only = tr_tw[tr_c == 0]
    valid = ~np.isnan(tr_wall_only)
    t_mean = float(tr_wall_only[valid].mean()) if valid.any() else 0.0
    t_std  = max(float(tr_wall_only[valid].std()), 1e-8) if valid.any() else 1.0

    save_path = os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_{sub_prefix}_best_model.pth")
    model, best_val, best_epoch = train(
        tr_s, tr_tw, tr_c, tr_tp_n,
        va_s, va_tw, va_c, va_tp_n,
        (s_mean, s_std), (t_mean, t_std), device, save_path,
        tr_td_n=tr_td_n, va_td_n=va_td_n)

    pred = predict(model, va_s, (s_mean, s_std), (t_mean, t_std), device)

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

    wall_val = (va_c == 0)
    wall_metrics = {}
    for i, name in enumerate(SLICE_NAMES):
        t = va_tw[wall_val, i]; m = pred["wall_pred_mean"][wall_val, i]; s = pred["wall_pred_std"][wall_val, i]
        v = ~np.isnan(t)
        t, m, s = t[v], m[v], s[v]
        if len(t):
            rmse = float(np.sqrt(((m - t) ** 2).mean()))
            mae  = float(np.abs(m - t).mean())
            wall_metrics[name] = {"n": int(len(t)), "rmse_mm": rmse, "mae_mm": mae,
                                  "pred_std_median_mm": float(np.median(s))}
            print(f"  Wall {name:>6} (n={len(t)}): RMSE={rmse:.0f} mm  MAE={mae:.0f} mm  σ_med={np.median(s):.0f}")

    out_prefix = os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_{sub_prefix}")
    plot_confusion(va_c, pred["cls_pred"], pred["cls_probs"], f"{out_prefix}_confusion.png")
    if pole_val.any():
        plot_pole_azimuth(pole_az_deg[is_val][pole_val],
                          pred["pole_pred_az_deg"][pole_val],
                          pred["pole_pred_az_std"][pole_val],
                          f"{out_prefix}_pole_azimuth_scatter.png")
    plot_wall_scatter(va_tw, pred["wall_pred_mean"], pred["wall_pred_std"],
                      wall_val, f"{out_prefix}_wall_scatter.png")
    if pole_val.any():
        plot_calibration(va_tw, pred["wall_pred_mean"], pred["wall_pred_std"],
                         pole_az_deg[is_val][pole_val],
                         pred["pole_pred_az_deg"][pole_val],
                         pred["pole_pred_az_std"][pole_val],
                         wall_val, f"{out_prefix}_calibration.png")

    fold_result = {
        "quadrant":   q,
        "best_epoch": best_epoch,
        "val_total":  best_val,
        "class_acc":  cls_acc,
        "per_class":  per_class,
        "pole_az":    pole_metrics,
        "wall_per_slice": wall_metrics,
        "n_train":    int(len(tr_s)),
        "n_val":      int(len(va_s)),
        "sonar_norm": {"mean": s_mean, "std": s_std},
        "target_norm":{"mean": t_mean, "std": t_std},
    }
    with open(f"{out_prefix}_results.json", "w") as f:
        json.dump(fold_result, f, indent=2)
    return fold_result


# ── Deployment training (single spatial holdout) ──────────────────────────────

def _pole_dist_metrics(c, dist_mm, pred):
    """RMSE/MAE of the range head on pole-class pings, plus the constant-mean
    baseline it has to beat (204 mm on the current data)."""
    if pred.get("pole_pred_dist_mm") is None or dist_mm is None:
        return None
    m = (c == 1) & np.isfinite(dist_mm)
    if not m.any():
        return None
    err = pred["pole_pred_dist_mm"][m] - dist_mm[m]
    baseline = float(np.std(dist_mm[m]))
    return {
        "n": int(m.sum()),
        "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "mae_mm": float(np.mean(np.abs(err))),
        "pred_std_median_mm": (float(np.median(pred["pole_pred_dist_std"][m]))
                               if pred.get("pole_pred_dist_std") is not None else None),
        "constant_mean_baseline_rmse_mm": baseline,
    }


def _subset_metrics(c, tw, az_deg, pred):
    """Class / pole-az / wall-slice metrics for one subset. c, tw, az_deg are that
    subset's labels, wall targets, and true pole azimuth (deg); pred is predict()
    run on the same subset. Returns (class_acc, per_class, pole_az, wall_per_slice)."""
    cls_acc = float(np.mean(pred["cls_pred"] == c))
    per_class = {}
    for ci, name in enumerate(CLASS_NAMES):
        truth = (c == ci)
        if truth.any():
            prec = float(np.mean(c[pred["cls_pred"] == ci] == ci)) if (pred["cls_pred"] == ci).any() else 0.0
            rec  = float(np.mean(pred["cls_pred"][truth] == ci))
            per_class[name] = {"precision": prec, "recall": rec, "n_true": int(truth.sum())}
    pole_metrics = {}
    pole_m = (c == 1)
    if pole_m.any():
        res = pred["pole_pred_az_deg"][pole_m] - az_deg[pole_m]
        pole_metrics = {"n": int(pole_m.sum()),
                        "rmse_deg": float(np.sqrt((res ** 2).mean())),
                        "mae_deg":  float(np.abs(res).mean()),
                        "pred_std_median_deg": float(np.median(pred["pole_pred_az_std"][pole_m]))}
    wall_metrics = {}
    wall_m = (c == 0)
    for i, name in enumerate(SLICE_NAMES):
        t = tw[wall_m, i]; m = pred["wall_pred_mean"][wall_m, i]; s = pred["wall_pred_std"][wall_m, i]
        v = ~np.isnan(t); t, m, s = t[v], m[v], s[v]
        if len(t):
            wall_metrics[name] = {"n": int(len(t)),
                                  "rmse_mm": float(np.sqrt(((m - t) ** 2).mean())),
                                  "mae_mm":  float(np.abs(m - t).mean()),
                                  "pred_std_median_mm": float(np.median(s))}
    return cls_acc, per_class, pole_metrics, wall_metrics


def spatial_holdout_mask(poses, sess, frac, seed):
    """Per session, hold out the contiguous patch of `frac` of pings nearest a
    seeded random anchor pose. Returns a boolean is_val mask."""
    rng = np.random.default_rng(seed)
    xy = np.asarray(poses, dtype=float)[:, :2]
    is_val = np.zeros(len(sess), dtype=bool)
    for name in ACQUISITION_SESSIONS:
        idx = np.where(sess == name)[0]
        if len(idx) == 0:
            continue
        anchor = xy[idx[rng.integers(len(idx))]]
        d = np.linalg.norm(xy[idx] - anchor, axis=1)
        k = int(np.ceil(frac * len(idx)))
        is_val[idx[np.argsort(d)[:k]]] = True
    return is_val


def run_deploy(sonar, slice_t, classes, pole_az_deg, pole_az_n_safe,
               poses, sess, device, sub_prefix="deploy", pole_dist_mm=None,
               pole_dist_n_safe=None):
    """Train the single deployment model on the spatial-holdout split and report
    in-sample (train) vs held-out (val) metrics. Writes loadable
    `{ARTIFACT_PREFIX}_deploy_*` artifacts + the shared feature_params."""
    is_val = spatial_holdout_mask(poses, sess, HOLDOUT_FRAC, HOLDOUT_SEED)
    print(f"  spatial holdout: {HOLDOUT_FRAC*100:.0f}% per session, seed={HOLDOUT_SEED}")
    for name in ACQUISITION_SESSIONS:
        vm = (sess == name) & is_val
        print(f"    {name}: held out {int(vm.sum())}/{int((sess==name).sum())} "
              f"(pole={int((classes[vm]==1).sum())}, wall={int((classes[vm]==0).sum())}, "
              f"none={int((classes[vm]==2).sum())})")

    tr_s, va_s       = sonar[~is_val],          sonar[is_val]
    tr_tw, va_tw     = slice_t[~is_val],        slice_t[is_val]
    tr_c, va_c       = classes[~is_val],        classes[is_val]
    tr_tp_n, va_tp_n = pole_az_n_safe[~is_val], pole_az_n_safe[is_val]
    tr_td_n = va_td_n = None
    if pole_dist_n_safe is not None:
        tr_td_n, va_td_n = pole_dist_n_safe[~is_val], pole_dist_n_safe[is_val]
    print(f"  train: {len(tr_s)} (wall={int((tr_c==0).sum())}, pole={int((tr_c==1).sum())}, "
          f"none={int((tr_c==2).sum())})")
    print(f"  held-out: {len(va_s)} (wall={int((va_c==0).sum())}, pole={int((va_c==1).sum())}, "
          f"none={int((va_c==2).sum())})")

    s_mean = float(tr_s.mean()); s_std = max(float(tr_s.std()), 1e-8)
    tr_wall_only = tr_tw[tr_c == 0]
    valid = ~np.isnan(tr_wall_only)
    t_mean = float(tr_wall_only[valid].mean()) if valid.any() else 0.0
    t_std  = max(float(tr_wall_only[valid].std()), 1e-8) if valid.any() else 1.0

    save_path = os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_{sub_prefix}_best_model.pth")
    model, best_val, best_epoch = train(
        tr_s, tr_tw, tr_c, tr_tp_n,
        va_s, va_tw, va_c, va_tp_n,
        (s_mean, s_std), (t_mean, t_std), device, save_path,
        tr_td_n=tr_td_n, va_td_n=va_td_n)

    pred_va = predict(model, va_s, (s_mean, s_std), (t_mean, t_std), device)
    pred_tr = predict(model, tr_s, (s_mean, s_std), (t_mean, t_std), device)
    keys = ("class_acc", "per_class", "pole_az", "wall_per_slice")
    held     = dict(zip(keys, _subset_metrics(va_c, va_tw, pole_az_deg[is_val],  pred_va)))
    insample = dict(zip(keys, _subset_metrics(tr_c, tr_tw, pole_az_deg[~is_val], pred_tr)))
    if pole_dist_mm is not None:
        held["pole_dist"]     = _pole_dist_metrics(va_c, pole_dist_mm[is_val],  pred_va)
        insample["pole_dist"] = _pole_dist_metrics(tr_c, pole_dist_mm[~is_val], pred_tr)

    def _show(tag, mtr):
        print(f"  [{tag}] class acc {mtr['class_acc']*100:.1f}%")
        for n, m in mtr["per_class"].items():
            print(f"      {n:>5}: prec {m['precision']*100:.1f}  rec {m['recall']*100:.1f}  n={m['n_true']}")
        if mtr["pole_az"]:
            print(f"      pole-az RMSE {mtr['pole_az']['rmse_deg']:.2f} deg  (n={mtr['pole_az']['n']})")
        for n, m in mtr["wall_per_slice"].items():
            print(f"      wall {n:>6} RMSE {m['rmse_mm']:.0f} mm  (n={m['n']})")
    print(f"  best_epoch={best_epoch}")
    _show("held-out", held); _show("in-sample", insample)

    out_prefix = os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_{sub_prefix}")
    plot_confusion(va_c, pred_va["cls_pred"], pred_va["cls_probs"], f"{out_prefix}_confusion.png")
    pole_val = (va_c == 1)
    if pole_val.any():
        plot_pole_azimuth(pole_az_deg[is_val][pole_val], pred_va["pole_pred_az_deg"][pole_val],
                          pred_va["pole_pred_az_std"][pole_val], f"{out_prefix}_pole_azimuth_scatter.png")
    plot_wall_scatter(va_tw, pred_va["wall_pred_mean"], pred_va["wall_pred_std"],
                      (va_c == 0), f"{out_prefix}_wall_scatter.png")

    feature_params = {
        "cone_half_deg":  CONE_HALF_DEG,
        "slice_definitions": {
            "left":   [-CONE_HALF_DEG,            -CONE_HALF_DEG + 2*CONE_HALF_DEG/3],
            "center": [-CONE_HALF_DEG + 2*CONE_HALF_DEG/3, CONE_HALF_DEG - 2*CONE_HALF_DEG/3],
            "right":  [CONE_HALF_DEG - 2*CONE_HALF_DEG/3,  CONE_HALF_DEG],
        },
        "pole_az_norm":   {"divide_by_deg": CONE_HALF_DEG},
        "pole_dist_norm": ({"divide_by_mm": MAX_RANGE_MM}
                           if MODEL_KWARGS.get("pole_dist_head") else None),
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
            "model_class":   MODEL_CLASS_NAME,
            "wall3_symmetric": bool(MODEL_KWARGS.get("symmetric", True)),
            "wall3_pole_dist": bool(MODEL_KWARGS.get("pole_dist_head", False)),
            "n_classes":     len(CLASS_NAMES),
        },
        "profile": {
            "opening_angle":  OPENING_ANGLE,
            "profile_steps":  PROFILE_STEPS,
            "profile_method": PROFILE_METHOD,
        },
        "loss_weights": {"class": LOSS_W_CLASS, "wall": LOSS_W_WALL,
                         "pole": LOSS_W_POLE, "pole_dist": LOSS_W_POLE_DIST},
    }
    with open(os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_feature_params.json"), "w") as f:
        json.dump(feature_params, f, indent=2)

    result = {
        "mode":         "spatial_holdout_deploy",
        "holdout_frac": HOLDOUT_FRAC,
        "holdout_seed": HOLDOUT_SEED,
        "best_epoch":   best_epoch,
        "val_total":    best_val,
        "held_out":     held,
        "in_sample":    insample,
        "n_train":      int(len(tr_s)),
        "n_val":        int(len(va_s)),
        "sonar_norm":   {"mean": s_mean, "std": s_std},
        "target_norm":  {"mean": t_mean, "std": t_std},
        "config": {"sessions": ACQUISITION_SESSIONS, "max_range_mm": MAX_RANGE_MM,
                   "cone_half_deg": CONE_HALF_DEG},
    }
    with open(f"{out_prefix}_results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)

    print("[1/4] Loading data")
    sonar, slice_t, classes, pole_az_deg, pole_dist_mm, quads, sess, bin_centers = \
        load_and_filter()
    print(f"  {len(sonar)} pings: wall={int((classes == 0).sum())}, "
          f"pole={int((classes == 1).sum())}, none={int((classes == 2).sum())}")

    pole_az_n = (pole_az_deg / CONE_HALF_DEG).astype(np.float32)
    # NaN azimuth (wall-class samples) is fine — the masked loss ignores it.
    # But torch.as_tensor doesn't like NaN propagation in CE so we fill with 0.
    pole_az_n_safe = np.where(np.isnan(pole_az_n), 0.0, pole_az_n).astype(np.float32)
    # Pole range target, normalised by MAX_RANGE_MM so it sits on the same scale
    # as the other regression heads. NaNs (non-pole pings) become 0 and are
    # excluded by the pole mask in the loss, exactly as for azimuth.
    pole_dist_n = (pole_dist_mm / MAX_RANGE_MM).astype(np.float32)
    pole_dist_n_safe = np.where(np.isnan(pole_dist_n), 0.0, pole_dist_n).astype(np.float32)
    print(f"  pole az normalised by /{CONE_HALF_DEG:.0f}°")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2/4] {len(CV_QUADRANTS)}-fold CV on {device} "
          f"({EPOCHS} epochs, {WARMUP_EPOCHS} warmup per fold)")

    fold_results = []
    for q in CV_QUADRANTS:
        print(f"\n--- Fold q={q} ---")
        fold_results.append(run_fold(
            q, sonar, slice_t, classes, pole_az_deg, pole_az_n_safe,
            quads, sess, device, sub_prefix=f"q{q}"))

    print("\n[3/4] CV summary")

    def _ms(vals):
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std())

    per_fold_acc = [f["class_acc"] for f in fold_results]
    acc_m, acc_s = _ms(per_fold_acc)
    print(f"  Class accuracy: {acc_m*100:.1f}% ± {acc_s*100:.1f}%  "
          f"(per-fold: {' / '.join(f'{a*100:.0f}' for a in per_fold_acc)})")

    cv_class = {}
    for name in CLASS_NAMES:
        precs = [f["per_class"][name]["precision"] for f in fold_results if name in f["per_class"]]
        recs  = [f["per_class"][name]["recall"]    for f in fold_results if name in f["per_class"]]
        if precs:
            pm, ps = _ms(precs); rm, rs = _ms(recs)
            cv_class[name] = {"precision_mean": pm, "precision_std": ps,
                              "recall_mean": rm, "recall_std": rs}
            print(f"    {name:>5}: precision={pm*100:.1f}% ± {ps*100:.1f}%  "
                  f"recall={rm*100:.1f}% ± {rs*100:.1f}%")

    cv_pole_az = {}
    pole_rmses = [f["pole_az"]["rmse_deg"] for f in fold_results if f["pole_az"]]
    if pole_rmses:
        pm, ps = _ms(pole_rmses)
        cv_pole_az = {"rmse_deg_mean": pm, "rmse_deg_std": ps,
                      "per_fold": pole_rmses}
        print(f"  Pole-az RMSE: {pm:.2f}° ± {ps:.2f}°  "
              f"(per-fold: {' / '.join(f'{r:.1f}' for r in pole_rmses)})")

    cv_wall = {}
    for name in SLICE_NAMES:
        rmses = [f["wall_per_slice"][name]["rmse_mm"] for f in fold_results if name in f["wall_per_slice"]]
        if rmses:
            rm, rs = _ms(rmses)
            cv_wall[name] = {"rmse_mm_mean": rm, "rmse_mm_std": rs, "per_fold": rmses}
            print(f"  Wall {name:>6} RMSE: {rm:.0f} ± {rs:.0f} mm  "
                  f"(per-fold: {' / '.join(f'{r:.0f}' for r in rmses)})")

    print("\n[4/4] Saving params + CV results")
    feature_params = {
        "cone_half_deg":  CONE_HALF_DEG,
        "slice_definitions": {
            "left":   [-CONE_HALF_DEG,            -CONE_HALF_DEG + 2*CONE_HALF_DEG/3],
            "center": [-CONE_HALF_DEG + 2*CONE_HALF_DEG/3, CONE_HALF_DEG - 2*CONE_HALF_DEG/3],
            "right":  [CONE_HALF_DEG - 2*CONE_HALF_DEG/3,  CONE_HALF_DEG],
        },
        "pole_az_norm":   {"divide_by_deg": CONE_HALF_DEG},
        "pole_dist_norm": ({"divide_by_mm": MAX_RANGE_MM}
                           if MODEL_KWARGS.get("pole_dist_head") else None),
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
            "model_class":   MODEL_CLASS_NAME,
            "wall3_symmetric": bool(MODEL_KWARGS.get("symmetric", True)),
            "wall3_pole_dist": bool(MODEL_KWARGS.get("pole_dist_head", False)),
            "n_classes":     len(CLASS_NAMES),
        },
        "profile": {
            "opening_angle":  OPENING_ANGLE,
            "profile_steps":  PROFILE_STEPS,
            "profile_method": PROFILE_METHOD,
        },
        "loss_weights": {"class": LOSS_W_CLASS, "wall": LOSS_W_WALL,
                         "pole": LOSS_W_POLE, "pole_dist": LOSS_W_POLE_DIST},
    }
    with open(os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_feature_params.json"), "w") as f:
        json.dump(feature_params, f, indent=2)

    cv_results = {
        "cv_summary": {
            "class_acc_mean":  acc_m,
            "class_acc_std":   acc_s,
            "per_class":       cv_class,
            "pole_az":         cv_pole_az,
            "wall_per_slice":  cv_wall,
        },
        "folds":  fold_results,
        "config": {
            "sessions":      ACQUISITION_SESSIONS,
            "cv_quadrants":  CV_QUADRANTS,
            "max_range_mm":  MAX_RANGE_MM,
            "cone_half_deg": CONE_HALF_DEG,
        },
    }
    with open(os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_cv_results.json"), "w") as f:
        json.dump(cv_results, f, indent=2)

    print(f"\nDone. Artifacts in {OUTPUT_DIR}/")
    print(f"  per-fold: {ARTIFACT_PREFIX}_q{{0..{len(CV_QUADRANTS)-1}}}_"
          f"{{best_model.pth, results.json, confusion.png, "
          f"pole_azimuth_scatter.png, wall_scatter.png, calibration.png}}")
    print(f"  aggregated: {ARTIFACT_PREFIX}_feature_params.json, "
          f"{ARTIFACT_PREFIX}_cv_results.json")


def main_deploy():
    """Canonical entry point: train the single deployed inverse on the spatial
    holdout and report in-sample vs held-out. The CV path main() is kept for
    diagnostics (and EXPT_head_variants)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)

    print("[1/3] Loading data (with poses)")
    sonar, slice_t, classes, pole_az_deg, pole_dist_mm, quads, sess, bin_centers, poses = \
        load_and_filter(with_poses=True)
    print(f"  {len(sonar)} pings: wall={int((classes == 0).sum())}, "
          f"pole={int((classes == 1).sum())}, none={int((classes == 2).sum())}")

    pole_az_n = (pole_az_deg / CONE_HALF_DEG).astype(np.float32)
    pole_az_n_safe = np.where(np.isnan(pole_az_n), 0.0, pole_az_n).astype(np.float32)
    # Pole range target, normalised by MAX_RANGE_MM so it sits on the same scale
    # as the other regression heads. NaNs (non-pole pings) become 0 and are
    # excluded by the pole mask in the loss, exactly as for azimuth.
    pole_dist_n = (pole_dist_mm / MAX_RANGE_MM).astype(np.float32)
    pole_dist_n_safe = np.where(np.isnan(pole_dist_n), 0.0, pole_dist_n).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2/3] Training deployment model on {device} "
          f"({EPOCHS} epochs, {WARMUP_EPOCHS} warmup; {HOLDOUT_FRAC*100:.0f}% spatial holdout)")
    run_deploy(sonar, slice_t, classes, pole_az_deg, pole_az_n_safe, poses, sess, device,
               pole_dist_mm=pole_dist_mm, pole_dist_n_safe=pole_dist_n_safe)

    print(f"\n[3/3] Done. Deployed model: {ARTIFACT_PREFIX}_deploy_* "
          f"+ {ARTIFACT_PREFIX}_feature_params.json (load with fold='deploy')")


if __name__ == "__main__":
    main_deploy()
