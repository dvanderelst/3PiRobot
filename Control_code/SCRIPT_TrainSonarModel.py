#!/usr/bin/env python3
"""
SCRIPT_TrainSonarModel.py

Train the 3-slice SonarSlicesUQ model on visually-guided acquisition data.
Reads AcquisitionSessions/<name>/ folders via Library.AcquisitionSessionLoader,
predicts per-slice (mean, σ²) for the right/center/left thirds of the
±CONE_HALF_DEG forward cone, and writes the canonical model artifacts to
SonarModel/ for downstream simulator/policy use.

Slice labels follow the project +az = LEFT convention (azimuth CCW from
robot forward), so increasing bin index goes from physical right to left:

  right  = [-cone,   -cone/3)        (negative az → robot's physical right)
  center = [-cone/3, +cone/3)
  left   = [+cone/3, +cone]          (positive az → robot's physical left)

Each slice gets a (mean, σ) pair → 6 output heads total. Symmetry is
baked into the architecture: center heads see (z_L + z_R) / 2; side heads
share weights and are applied with channels swapped, so swapping L/R
sonar inputs exactly swaps left/right predictions.

Outputs in SonarModel/  (prefix `slices_`):
  slices_best_model.pth
  slices_feature_params.json
  slices_scatter.png              3-panel: per-slice pred vs true
  slices_calibration.png          3-panel: per-slice σ calibration
  slices_sigma_sim_fit.png        3-panel: per-slice σ_sim(d) fit
  slices_overall_min_scatter.png  pred-vs-true min over the cone
  slices_collapse_check.png       L−R asymmetry diagnostic + per-sample spread
  slices_results.json
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
from scipy.optimize import minimize

from Library.AcquisitionSessionLoader import load_data as _load_acquisition_data
from Library.SonarModel import SonarSlicesUQ, SLICE_NAMES as _LIB_SLICE_NAMES
from Library.SonarModel import normalize_envelope_per_ping


# ── Settings ──────────────────────────────────────────────────────────────────

ACQUISITION_SESSIONS = ["Acquisition01A", "Acquisition02A", "Acquisition03A", "Acquisition04A"]
ACQUISITIONS_ROOT    = "AcquisitionSessions"

OPENING_ANGLE  = 270.0
PROFILE_STEPS  = 90
PROFILE_METHOD = "ray_center"

CONE_HALF_DEG  = 35.0    # full cone; split into 3 equal angular slices

# Validation: hold out one quadrant per session. Loader's quadrants are by
# per-session median split on (x, y); 0..3 covers all four quadrants.
VALIDATION_QUADRANTS = {
    "Acquisition01A": [0],
    "Acquisition02A": [1],
    "Acquisition03A": [2],
    "Acquisition04A": [3],
}

SONAR_CONV_CHANNELS = [8, 16]
SONAR_CONV_KERNEL   = 7
SONAR_POOL_OUT      = 8
SONAR_FC_HIDDEN     = 32
SONAR_HEAD_HIDDEN   = 16

# Per-ping envelope normalisation was originally added to defend against
# day-to-day emission-strength drift (battery state, analog-receive-path
# drift). That drift has been addressed at the hardware/firmware level, so
# new acquisitions don't need it. Keep the setting but default to off; flip
# back to "per_ping_minmax" if amplitude variability ever returns. The
# trainer's z-score normalisation (using sonar_norm in feature_params.json)
# still applies regardless and keeps inputs at the network in roughly
# [-3, 3] range.
ENVELOPE_NORM_KIND       = None
ENVELOPE_NORM_OUT_MIN    = 0.0
ENVELOPE_NORM_OUT_MAX    = 1.0
ENVELOPE_NORM_REF_WINDOW = 10

LR             = 1e-3
BATCH_SIZE     = 64
EPOCHS         = 150
WARMUP_EPOCHS  = 20
LOG_VAR_MIN    = -6.0
LOG_VAR_MAX    = 4.0
SEED           = 42

N_SIGMA_BINS   = 8

OUTPUT_DIR      = "SonarModel"
ARTIFACT_PREFIX = "slices"

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
    """Per-slice min profile distance, robust to a few NaN bins (gaps where
    no wall point fell into a profile cell). Uses np.nanmin so a slice with
    even one valid bin still yields a usable target — only fully-empty
    slices remain NaN, and those rows are dropped before training."""
    masks = slice_masks(bin_centers, cone_half_deg)
    cols = []
    for m in masks:
        sub = profiles[:, m]
        all_nan = np.isnan(sub).all(axis=1)
        col = np.where(all_nan, np.nan, np.nanmin(np.where(all_nan[:, None], np.inf, sub), axis=1))
        cols.append(col)
    return np.stack(cols, axis=1).astype(np.float32)


# ── Loss ──────────────────────────────────────────────────────────────────────

def gnll_loss(pred_mean, pred_log_var, target,
              log_var_min=LOG_VAR_MIN, log_var_max=LOG_VAR_MAX):
    log_var = pred_log_var.clamp(log_var_min, log_var_max)
    inv_var = torch.exp(-log_var)
    return 0.5 * (log_var + (target - pred_mean) ** 2 * inv_var).mean()


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    """Returns sonar (N, T, 2), profiles (N, profile_steps), quads (N,),
    sess (N,), bin_centers (profile_steps,)."""
    return _load_acquisition_data(
        ACQUISITION_SESSIONS,
        acquisitions_root=ACQUISITIONS_ROOT,
        opening_angle=OPENING_ANGLE,
        profile_steps=PROFILE_STEPS,
        profile_method=PROFILE_METHOD,
    )


def split_indices(quads, sess):
    is_val = np.zeros(len(quads), dtype=bool)
    for s_name, val_q in VALIDATION_QUADRANTS.items():
        is_val |= (sess == s_name) & np.isin(quads, list(val_q))
    return is_val


# ── Training ──────────────────────────────────────────────────────────────────

def make_loader(s, t, batch_size, shuffle):
    L = torch.as_tensor(s[..., 0], dtype=torch.float32)
    R = torch.as_tensor(s[..., 1], dtype=torch.float32)
    T = torch.as_tensor(t,         dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(L, R, T)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def total_loss(out, T_norm, in_warmup):
    means    = [out["right_mean"].squeeze(1),   out["center_mean"].squeeze(1),   out["left_mean"].squeeze(1)]
    log_vars = [out["right_log_var"].squeeze(1),out["center_log_var"].squeeze(1),out["left_log_var"].squeeze(1)]
    targets  = [T_norm[:, 0], T_norm[:, 1], T_norm[:, 2]]
    if in_warmup:
        return sum(((m - t) ** 2).mean() for m, t in zip(means, targets))
    return sum(gnll_loss(m, lv, t) for m, lv, t in zip(means, log_vars, targets))


def train(tr_s, tr_t, va_s, va_t, sonar_stats, target_stats, device, save_path):
    s_mean, s_std = sonar_stats
    t_mean, t_std = target_stats
    tr_sn = ((tr_s - s_mean) / s_std).astype(np.float32)
    va_sn = ((va_s - s_mean) / s_std).astype(np.float32)
    tr_tn = ((tr_t - t_mean) / t_std).astype(np.float32)
    va_tn = ((va_t - t_mean) / t_std).astype(np.float32)

    train_loader = make_loader(tr_sn, tr_tn, BATCH_SIZE, True)
    val_loader   = make_loader(va_sn, va_tn, BATCH_SIZE, False)

    torch.manual_seed(SEED)
    model = SonarSlicesUQ(
        samples=tr_s.shape[1],
        conv_channels=SONAR_CONV_CHANNELS, conv_kernel=SONAR_CONV_KERNEL,
        pool_out=SONAR_POOL_OUT, fc_hidden=SONAR_FC_HIDDEN,
        head_hidden=SONAR_HEAD_HIDDEN,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_state, best_val_nll, best_epoch = None, float('inf'), -1
    for epoch in range(1, EPOCHS + 1):
        in_warmup = epoch <= WARMUP_EPOCHS
        model.train()
        for L, R, T in train_loader:
            L, R, T = L.to(device), R.to(device), T.to(device)
            out  = model(L, R)
            loss = total_loss(out, T, in_warmup)
            opt.zero_grad(); loss.backward(); opt.step()

        if not in_warmup:
            model.eval()
            v_nll = []
            with torch.no_grad():
                for L, R, T in val_loader:
                    L, R, T = L.to(device), R.to(device), T.to(device)
                    out = model(L, R)
                    v_nll.append(float(total_loss(out, T, in_warmup=False).item()))
            val_nll = float(np.mean(v_nll))
            if val_nll < best_val_nll:
                best_val_nll, best_epoch = val_nll, epoch
                best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1 or epoch == WARMUP_EPOCHS + 1:
            tag = " *" if epoch == best_epoch else (" [warmup]" if in_warmup else "")
            extra = f"  val_nll={best_val_nll:+.4f}" if best_val_nll < float('inf') else ""
            print(f"  Epoch {epoch:3d}/{EPOCHS}{tag}{extra}")

    model.load_state_dict(best_state)
    model.eval()
    torch.save({'model_state_dict': best_state}, save_path)
    print(f"  Best epoch: {best_epoch}  val NLL = {best_val_nll:.4f}")
    return model, best_val_nll, best_epoch


def predict(model, sonar, sonar_stats, target_stats, device):
    s_mean, s_std = sonar_stats
    t_mean, t_std = target_stats
    s = ((sonar - s_mean) / s_std).astype(np.float32)
    L = torch.as_tensor(s[..., 0], dtype=torch.float32).to(device)
    R = torch.as_tensor(s[..., 1], dtype=torch.float32).to(device)
    means_n = {k: [] for k in SLICE_NAMES}
    log_vars_n = {k: [] for k in SLICE_NAMES}
    with torch.no_grad():
        for st in range(0, len(L), 256):
            ed = min(st + 256, len(L))
            out = model(L[st:ed], R[st:ed])
            for k in SLICE_NAMES:
                means_n[k].append(out[f"{k}_mean"].cpu().squeeze(1).numpy())
                log_vars_n[k].append(out[f"{k}_log_var"].cpu().squeeze(1).numpy())
    means = np.stack([np.concatenate(means_n[k])    for k in SLICE_NAMES], axis=1)
    log_vars = np.stack([np.concatenate(log_vars_n[k]) for k in SLICE_NAMES], axis=1)
    log_vars = np.clip(log_vars, LOG_VAR_MIN, LOG_VAR_MAX)
    pred_mean = means * t_std + t_mean
    pred_std  = np.exp(log_vars / 2.0) * t_std
    return pred_mean, pred_std


# ── σ_sim fitting (per slice) ─────────────────────────────────────────────────

def fit_empirical_sigma(true, pred_mean, n_bins=N_SIGMA_BINS):
    residuals = pred_mean - true
    sorted_idx = np.argsort(true)
    bin_size = max(len(true) // n_bins, 1)
    bin_centers, bin_sigmas, bin_n = [], [], []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(true)
        idx = sorted_idx[start:end]
        bin_centers.append(float(true[idx].mean()))
        bin_sigmas.append(float(residuals[idx].std()))
        bin_n.append(int(len(idx)))
    return bin_centers, bin_sigmas, bin_n


def fit_parametric_sigma(bin_centers, bin_sigmas):
    bc = np.asarray(bin_centers, dtype=float)
    bs = np.asarray(bin_sigmas,  dtype=float)
    def predict_(params, d):
        floor, knee, slope = params
        return floor + slope * np.maximum(0.0, d - knee)
    def loss(params):
        if params[0] < 0 or params[2] < 0:
            return 1e12
        return float(((predict_(params, bc) - bs) ** 2).mean())
    floor_init = float(np.min(bs))
    over = np.where(bs > floor_init * 1.5)[0]
    knee_init = float(bc[over[0]]) if len(over) > 0 else float(bc[len(bc) // 2])
    slope_init = max(0.0, float((bs[-1] - floor_init) / max(bc[-1] - knee_init, 1.0)))
    res = minimize(loss, x0=[floor_init, knee_init, slope_init], method="Nelder-Mead")
    floor, knee, slope = [float(v) for v in res.x]
    return {"sigma_floor_mm": floor, "d_knee_mm": knee, "slope": slope}


def parametric_sigma(d_mm, params):
    return params["sigma_floor_mm"] + params["slope"] * np.maximum(
        0.0, d_mm - params["d_knee_mm"])


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_scatter_per_slice(true, pred_mean, pred_std, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, (ax, name) in enumerate(zip(axes, SLICE_NAMES)):
        t = true[:, i]; m = pred_mean[:, i]; s = pred_std[:, i]
        sc = ax.scatter(t, m, c=s, s=10, alpha=0.55, cmap='viridis')
        plt.colorbar(sc, ax=ax, label="σ (mm)")
        lo = float(min(t.min(), m.min())); hi = float(max(t.max(), m.max())) * 1.02
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        rmse = float(np.sqrt(((m - t) ** 2).mean()))
        mae  = float(np.abs(m - t).mean())
        ax.set_title(f"{name.capitalize()}   RMSE={rmse:.0f}   MAE={mae:.0f} mm",
                     fontsize=11)
        ax.set_xlabel("true (mm)")
        if i == 0: ax.set_ylabel("predicted (mm)")
        ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def plot_calibration_per_slice(true, pred_mean, pred_std, out_path, n_bins=10):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, (ax, name) in enumerate(zip(axes, SLICE_NAMES)):
        t = true[:, i]; m = pred_mean[:, i]; s = pred_std[:, i]
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
        ax.scatter(bps, bemp, s=70, color='steelblue', edgecolor='white', linewidth=1)
        for k, (x, y) in enumerate(zip(bps, bemp)):
            ax.annotate(f"d{k+1}", (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=7)
        lo = min(min(bps), min(bemp)) * 0.85
        hi = max(max(bps), max(bemp)) * 1.15
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("predicted σ (mm)")
        if i == 0: ax.set_ylabel("empirical RMSE (mm)")
        ax.set_title(f"{name.capitalize()} calibration", fontsize=11)
        ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def plot_sigma_sim_fit_per_slice(true, pred_mean, fits, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, (ax, name) in enumerate(zip(axes, SLICE_NAMES)):
        t = true[:, i]; m = pred_mean[:, i]
        residuals = m - t
        bc, bs, _ = fits[name]["empirical"]
        params = fits[name]["parametric"]
        ax.scatter(t, np.abs(residuals), s=4, alpha=0.3, color='steelblue', label='|residual|')
        ax.scatter(bc, bs, s=90, color='black', edgecolor='white', linewidth=1.3,
                   zorder=10, label='empirical σ')
        d_range = np.linspace(float(t.min()), float(t.max()), 300)
        ax.plot(d_range, parametric_sigma(d_range, params), color='red', lw=2,
                label=(f"σ(d) = {params['sigma_floor_mm']:.0f}"
                       f" + {params['slope']:.2f}·max(0, d−{params['d_knee_mm']:.0f})"))
        ax.set_xlabel("true distance (mm)")
        if i == 0: ax.set_ylabel("|residual| / σ (mm)")
        ax.set_title(f"{name.capitalize()} σ_sim(d)", fontsize=11)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def collapse_check(true, pred_mean):
    pred_spread = pred_mean.std(axis=1)
    true_spread = true.std(axis=1)
    n_collapsed = int(np.sum(pred_spread < 5.0))
    true_LR = true[:, 0]      - true[:, 2]
    pred_LR = pred_mean[:, 0] - pred_mean[:, 2]
    pearson_LR = float(np.corrcoef(true_LR, pred_LR)[0, 1])
    side_acc_by_threshold = {}
    for thr in (0, 50, 100, 200, 400):
        mask = np.abs(true_LR) > thr
        if mask.any():
            acc = float(np.mean(np.sign(pred_LR[mask]) == np.sign(true_LR[mask])))
            side_acc_by_threshold[f"|trueLR|>{thr}mm"] = {"acc": acc, "n": int(mask.sum())}
    return {
        "true_spread_mean_mm":         float(true_spread.mean()),
        "true_spread_median_mm":       float(np.median(true_spread)),
        "pred_spread_mean_mm":         float(pred_spread.mean()),
        "pred_spread_median_mm":       float(np.median(pred_spread)),
        "spread_ratio_pred_over_true": float(pred_spread.mean() / max(true_spread.mean(), 1e-8)),
        "n_samples":                   int(len(pred_spread)),
        "n_collapsed_under_5mm":       n_collapsed,
        "pearson_LR_pred_vs_true":     pearson_LR,
        "side_acc_by_LR_threshold":    side_acc_by_threshold,
        "_arrays": {"true_LR": true_LR, "pred_LR": pred_LR,
                    "true_spread": true_spread, "pred_spread": pred_spread,
                    "pearson_LR": pearson_LR},
    }


def plot_collapse_check(diag, out_path):
    a = diag["_arrays"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(a["true_LR"], a["pred_LR"], s=8, alpha=0.5, color='steelblue')
    lo = float(min(a["true_LR"].min(), a["pred_LR"].min()))
    hi = float(max(a["true_LR"].max(), a["pred_LR"].max()))
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1, alpha=0.5,
            label='diagonal (perfect)')
    ax.axhline(0, color='red', lw=1, alpha=0.5, label='pred = 0 (collapse)')
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("true (left − right) (mm)")
    ax.set_ylabel("pred (left − right) (mm)")
    ax.set_title(f"L−R asymmetry: pred vs true   r = {a['pearson_LR']:+.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    bins = np.linspace(0, max(a["true_spread"].max(), a["pred_spread"].max()), 40)
    ax.hist(a["true_spread"], bins=bins, alpha=0.55, color='gray',
            label=f"true spread (mean {a['true_spread'].mean():.0f})")
    ax.hist(a["pred_spread"], bins=bins, alpha=0.65, color='steelblue',
            label=f"pred spread (mean {a['pred_spread'].mean():.0f})")
    ax.set_xlabel("per-sample std across L/C/R (mm)")
    ax.set_ylabel("count")
    ax.set_title("Per-sample spread across slices  (collapse → mass at 0)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def plot_overall_min_comparison(true, pred_mean, out_path):
    true_overall = true.min(axis=1)
    pred_overall = pred_mean.min(axis=1)
    rmse = float(np.sqrt(((pred_overall - true_overall) ** 2).mean()))
    mae  = float(np.abs(pred_overall - true_overall).mean())
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_overall, pred_overall, s=10, alpha=0.5, color='steelblue')
    lo = float(min(true_overall.min(), pred_overall.min()))
    hi = float(max(true_overall.max(), pred_overall.max())) * 1.02
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("true min over ±35° cone (mm)")
    ax.set_ylabel("min(left_pred, center_pred, right_pred) (mm)")
    ax.set_title(f"Overall-min recovery from 3-slice model\n"
                 f"RMSE={rmse:.0f} mm  MAE={mae:.0f} mm  "
                 f"(reference: distance-only model = 142 mm)")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()
    return {"rmse_mm": rmse, "mae_mm": mae}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)

    print("[1/5] Loading data")
    sonar, profiles, quads, sess, bin_centers = load_data()
    print(f"  loaded {len(sonar)} pings from {len(set(sess.tolist()))} session(s)")

    if ENVELOPE_NORM_KIND == "per_ping_minmax":
        n_before = sonar.shape[0]
        sonar = normalize_envelope_per_ping(
            sonar,
            ENVELOPE_NORM_OUT_MIN,
            ENVELOPE_NORM_OUT_MAX,
            ref_window=ENVELOPE_NORM_REF_WINDOW,
        )
        print(f"  envelope normalised per-ping per-channel to "
              f"[{ENVELOPE_NORM_OUT_MIN:.1f}, {ENVELOPE_NORM_OUT_MAX:.1f}] "
              f"using first {ENVELOPE_NORM_REF_WINDOW} samples for max  (n={n_before})")
    elif ENVELOPE_NORM_KIND not in (None, "", "none"):
        raise ValueError(f"Unknown ENVELOPE_NORM_KIND={ENVELOPE_NORM_KIND!r}")

    is_val = split_indices(quads, sess)
    targets = compute_slice_targets(profiles, bin_centers, CONE_HALF_DEG)
    tr_s, va_s = sonar[~is_val],  sonar[is_val]
    tr_t, va_t = targets[~is_val], targets[is_val]
    print(f"  train: {len(tr_s)},  val: {len(va_s)},  cone: ±{CONE_HALF_DEG:.0f}°")
    masks = slice_masks(bin_centers, CONE_HALF_DEG)
    for name, m in zip(SLICE_NAMES, masks):
        print(f"    {name:>6}: {m.sum()} bins  "
              f"(target mean={targets[:, SLICE_NAMES.index(name)].mean():.0f} mm,  "
              f"std={targets[:, SLICE_NAMES.index(name)].std():.0f} mm)")

    s_mean = float(tr_s.mean()); s_std = max(float(tr_s.std()), 1e-8)
    t_mean = float(tr_t.mean()); t_std = max(float(tr_t.std()), 1e-8)
    print(f"  sonar  mean={s_mean:.0f}, std={s_std:.0f}")
    print(f"  target (pooled) mean={t_mean:.0f} mm, std={t_std:.0f} mm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_best_model.pth")
    print(f"\n[2/5] Training on {device} ({EPOCHS} epochs, {WARMUP_EPOCHS} warmup)")
    model, best_nll, best_epoch = train(
        tr_s, tr_t, va_s, va_t, (s_mean, s_std), (t_mean, t_std), device, save_path)

    print("\n[3/5] Evaluating on val")
    pred_mean, pred_std = predict(model, va_s, (s_mean, s_std), (t_mean, t_std), device)
    per_slice_metrics = {}
    for i, name in enumerate(SLICE_NAMES):
        t = va_t[:, i]; m = pred_mean[:, i]; s = pred_std[:, i]
        residuals = m - t
        rmse = float(np.sqrt((residuals ** 2).mean()))
        mae  = float(np.abs(residuals).mean())
        cor  = float(np.corrcoef(s, np.abs(residuals))[0, 1])
        f1s  = float(np.mean(np.abs(residuals) <= s))
        f2s  = float(np.mean(np.abs(residuals) <= 2 * s))
        per_slice_metrics[name] = {
            "val_rmse_mm": rmse, "val_mae_mm": mae,
            "val_pearson_sigma_residual": cor,
            "val_frac_within_1sigma": f1s, "val_frac_within_2sigma": f2s,
            "pred_std_median_mm": float(np.median(s)),
        }
        print(f"  {name:>6}:  RMSE={rmse:5.0f}  MAE={mae:4.0f}  "
              f"corr(σ,|r|)={cor:+.3f}  frac_1σ={f1s:.2f}  σ_med={np.median(s):.0f}")

    true_overall = va_t.min(axis=1)
    pred_overall = pred_mean.min(axis=1)
    overall_rmse = float(np.sqrt(((pred_overall - true_overall) ** 2).mean()))
    overall_mae  = float(np.abs(pred_overall - true_overall).mean())
    print(f"\n  Overall min:")
    print(f"    RMSE = {overall_rmse:.0f} mm  (reference distance-only model: 142 mm)")
    print(f"    MAE  = {overall_mae:.0f} mm")

    diag = collapse_check(va_t, pred_mean)
    print(f"\n  Collapse check:")
    print(f"    Per-sample spread across slices:")
    print(f"      true:  mean={diag['true_spread_mean_mm']:.0f} mm,  "
          f"median={diag['true_spread_median_mm']:.0f} mm")
    print(f"      pred:  mean={diag['pred_spread_mean_mm']:.0f} mm,  "
          f"median={diag['pred_spread_median_mm']:.0f} mm   "
          f"(pred/true = {diag['spread_ratio_pred_over_true']:.2f})")
    print(f"    Samples with predicted spread < 5 mm (collapsed): "
          f"{diag['n_collapsed_under_5mm']}/{diag['n_samples']}")
    print(f"    Pearson(pred L−R, true L−R) = {diag['pearson_LR_pred_vs_true']:+.3f}")
    print(f"  Side accuracy by |true L−R| threshold:")
    for k, v in diag["side_acc_by_LR_threshold"].items():
        print(f"    {k:>16}:  acc = {v['acc']:.3f}  (n={v['n']})")

    print("\n[4/5] Fitting σ_sim(d) per slice")
    fits = {}
    for i, name in enumerate(SLICE_NAMES):
        t = va_t[:, i]; m = pred_mean[:, i]
        bc, bs, bn = fit_empirical_sigma(t, m)
        params = fit_parametric_sigma(bc, bs)
        fits[name] = {"empirical":  (bc, bs, bn), "parametric": params}
        print(f"  {name:>6}:  σ(d) = {params['sigma_floor_mm']:.0f} "
              f"+ {params['slope']:.3f}·max(0, d − {params['d_knee_mm']:.0f})")

    print("\n[5/5] Saving plots and JSON")
    plot_scatter_per_slice(va_t, pred_mean, pred_std,
                           os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_scatter.png"))
    plot_calibration_per_slice(va_t, pred_mean, pred_std,
                               os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_calibration.png"))
    plot_sigma_sim_fit_per_slice(va_t, pred_mean, fits,
                                 os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_sigma_sim_fit.png"))
    plot_overall_min_comparison(va_t, pred_mean,
                                os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_overall_min_scatter.png"))
    plot_collapse_check(diag,
                        os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_collapse_check.png"))

    feature_params = {
        "cone_half_deg":  CONE_HALF_DEG,
        "slice_definitions": {
            "left":   [-CONE_HALF_DEG,            -CONE_HALF_DEG + 2*CONE_HALF_DEG/3],
            "center": [-CONE_HALF_DEG + 2*CONE_HALF_DEG/3, CONE_HALF_DEG - 2*CONE_HALF_DEG/3],
            "right":  [CONE_HALF_DEG - 2*CONE_HALF_DEG/3,  CONE_HALF_DEG],
        },
        "sonar_norm":     {"mean": s_mean, "std": s_std},
        "target_norm":    {"mean": t_mean, "std": t_std},
        "envelope_norm":  ({"kind":       "per_ping_minmax",
                            "out_min":    ENVELOPE_NORM_OUT_MIN,
                            "out_max":    ENVELOPE_NORM_OUT_MAX,
                            "ref_window": ENVELOPE_NORM_REF_WINDOW}
                           if ENVELOPE_NORM_KIND == "per_ping_minmax"
                           else {"kind": "none"}),
        "log_var_clamp":  [LOG_VAR_MIN, LOG_VAR_MAX],
        "architecture": {
            "samples":       int(tr_s.shape[1]),
            "conv_channels": SONAR_CONV_CHANNELS,
            "conv_kernel":   SONAR_CONV_KERNEL,
            "pool_out":      SONAR_POOL_OUT,
            "fc_hidden":     SONAR_FC_HIDDEN,
            "head_hidden":   SONAR_HEAD_HIDDEN,
            "model_class":   "SonarSlicesUQ",
        },
        "profile": {
            "opening_angle":  OPENING_ANGLE,
            "profile_steps":  PROFILE_STEPS,
            "profile_method": PROFILE_METHOD,
        },
        "sigma_sim_per_slice": {
            name: {
                "parametric": fits[name]["parametric"],
                "empirical": {
                    "bin_centers_mm": fits[name]["empirical"][0],
                    "bin_sigmas_mm":  fits[name]["empirical"][1],
                    "bin_n":          fits[name]["empirical"][2],
                },
            } for name in SLICE_NAMES
        },
    }
    with open(os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_feature_params.json"), "w") as f:
        json.dump(feature_params, f, indent=2)

    results = {
        "metrics": {
            "best_epoch":         best_epoch,
            "val_nll_summed":     best_nll,
            "per_slice":          per_slice_metrics,
            "overall_min_rmse_mm":overall_rmse,
            "overall_min_mae_mm": overall_mae,
            "collapse_check": {k: v for k, v in diag.items() if not k.startswith("_")},
        },
        "data": {
            "n_train": int(len(tr_t)), "n_val": int(len(va_t)),
            "validation_quadrants": VALIDATION_QUADRANTS,
            "sessions": ACQUISITION_SESSIONS,
        },
    }
    with open(os.path.join(OUTPUT_DIR, f"{ARTIFACT_PREFIX}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Artifacts in {OUTPUT_DIR}/  (all prefixed with '{ARTIFACT_PREFIX}_'):")
    print(f"  {ARTIFACT_PREFIX}_best_model.pth, {ARTIFACT_PREFIX}_feature_params.json,"
          f" {ARTIFACT_PREFIX}_results.json")
    print(f"  {ARTIFACT_PREFIX}_scatter.png, {ARTIFACT_PREFIX}_calibration.png,")
    print(f"  {ARTIFACT_PREFIX}_sigma_sim_fit.png, {ARTIFACT_PREFIX}_overall_min_scatter.png,")
    print(f"  {ARTIFACT_PREFIX}_collapse_check.png")


if __name__ == "__main__":
    main()
