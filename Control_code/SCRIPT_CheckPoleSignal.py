#!/usr/bin/env python3
"""
SCRIPT_CheckPoleSignal.py

Diagnostic: do six hand-crafted physical features in the sonar envelope
discriminate wall-class from pole-class pings, regardless of model capacity
or data quantity?

If a logistic regression on these features substantially beats the prior,
the signal is unambiguously in the envelope and the CNN classifier is just
data-starved. If it doesn't, the discrimination problem is harder than
expected and more sessions alone won't fix it.

Features per ping (post-emit region only, samples >= EMIT_PULSE_SAMPLES):
  peak_max         max amplitude across both channels (strong echo)
  peak_min         min amplitude across both channels (do both ears hear it?)
  lr_asym          |peak_L − peak_R| / (peak_L + peak_R)  side-asymmetry
  width_max        broader of L/R echo widths (samples above peak/2)
  width_min        narrower of L/R echo widths
  toa              time-of-arrival sample index (mean argmax across channels)

Output: TempOutput/check_pole_signal/{summary.txt, features_by_class.png,
                                       pca_scatter.png, confusion.png}
and a stdout summary.
"""

import os
from pathlib import Path

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from Library.AcquisitionSessionLoader import load_data_inverse


ACQUISITION_SESSIONS = ["Acquisition01A", "Acquisition02A"]
ACQUISITIONS_ROOT    = "AcquisitionSessions"
CONE_HALF_DEG        = 35.0

EMIT_PULSE_SAMPLES = 10   # skip the saturated emit-pulse region

OUT_DIR = Path("TempOutput") / "check_pole_signal"

FEATURE_NAMES = ["peak_max", "peak_min", "lr_asym",
                 "width_max", "width_min", "toa"]

# Per-feature print format. lr_asym lives in [0, 1] so a 1-decimal format
# (used for the ~25 000-range peaks) rounds typical values to 0.0; .4f keeps
# the actual signal visible.
FEATURE_FORMATS = {
    "peak_max":  ".1f",
    "peak_min":  ".1f",
    "lr_asym":   ".4f",
    "width_max": ".1f",
    "width_min": ".1f",
    "toa":       ".1f",
}


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(sonar: np.ndarray) -> np.ndarray:
    """sonar (N, T, 2) → features (N, 6).

    Uses only samples >= EMIT_PULSE_SAMPLES so the emit pulse doesn't
    dominate the peak amplitude.
    """
    post = sonar[:, EMIT_PULSE_SAMPLES:, :].astype(np.float32)  # (N, T', 2)
    L, R = post[..., 0], post[..., 1]

    peak_L = L.max(axis=1)
    peak_R = R.max(axis=1)
    arg_L  = L.argmax(axis=1).astype(np.float32)
    arg_R  = R.argmax(axis=1).astype(np.float32)

    # Echo width: count of samples above peak/2, per channel
    half_L = (L >= peak_L[:, None] * 0.5)
    half_R = (R >= peak_R[:, None] * 0.5)
    width_L = half_L.sum(axis=1).astype(np.float32)
    width_R = half_R.sum(axis=1).astype(np.float32)

    sum_peak = np.maximum(peak_L + peak_R, 1.0)
    lr_asym  = np.abs(peak_L - peak_R) / sum_peak

    return np.column_stack([
        np.maximum(peak_L, peak_R),       # peak_max
        np.minimum(peak_L, peak_R),       # peak_min
        lr_asym,                          # lr_asym
        np.maximum(width_L, width_R),     # width_max
        np.minimum(width_L, width_R),     # width_min
        0.5 * (arg_L + arg_R),            # toa
    ]).astype(np.float32)


# ── Distance helper (rough) ───────────────────────────────────────────────────

def nearest_reflector_distance(walls, poles, pole_radius_mm,
                               rob_x, rob_y, rob_yaw_deg, cone_half_deg):
    """Return distance (mm) to the nearest reflector inside the cone; np.inf if empty."""
    yaw_rad = np.deg2rad(rob_yaw_deg)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    best = np.inf
    if walls.size:
        dx = walls[:, 0] - rob_x
        dy = walls[:, 1] - rob_y
        rx = dx * c + dy * s; ry = -dx * s + dy * c
        ang = np.rad2deg(np.arctan2(ry, rx))
        in_cone = np.abs(ang) <= cone_half_deg
        if in_cone.any():
            best = min(best, float(np.hypot(rx[in_cone], ry[in_cone]).min()))
    if poles.size:
        dx = poles[:, 0] - rob_x
        dy = poles[:, 1] - rob_y
        rx = dx * c + dy * s; ry = -dx * s + dy * c
        ang = np.rad2deg(np.arctan2(ry, rx))
        in_cone = np.abs(ang) <= cone_half_deg
        if in_cone.any():
            d_surf = np.hypot(rx[in_cone], ry[in_cone]) - pole_radius_mm
            best = min(best, float(d_surf.min()))
    return best


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_features_by_class(X, y, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for k, ax in enumerate(axes.flat):
        wall = X[y == 0, k]; pole = X[y == 1, k]
        lo = min(wall.min(), pole.min())
        hi = max(wall.max(), pole.max())
        bins = np.linspace(lo, hi, 25)
        ax.hist(wall, bins=bins, alpha=0.55, color="#377eb8", label="wall")
        ax.hist(pole, bins=bins, alpha=0.55, color="#e41a1c", label="pole")
        # Brunner-Munzel / separation effect size: |Δ mean| / pooled σ
        sep = abs(wall.mean() - pole.mean()) / max(0.5 * (wall.std() + pole.std()), 1e-6)
        ax.set_title(f"{FEATURE_NAMES[k]}   separation d = {sep:.2f}")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_pca(X, y, out_path):
    Xs = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pcs[y == 0, 0], pcs[y == 0, 1], s=18, alpha=0.6,
               color="#377eb8", label=f"wall (n={int((y == 0).sum())})")
    ax.scatter(pcs[y == 1, 0], pcs[y == 1, 1], s=18, alpha=0.6,
               color="#e41a1c", label=f"pole (n={int((y == 1).sum())})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    ax.set_title("PCA of hand features (z-scored)")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_confusion(y, y_pred, out_path, acc):
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y, y_pred):
        cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["wall", "pole"]); ax.set_yticklabels(["wall", "pole"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, weight="bold")
    ax.set_title(f"LogReg 5-fold CV  acc = {acc:.1%}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] Loading data")
    sonar, profiles, classes, pole_az, near_dist, quads, sess, bin_centers = load_data_inverse(
        ACQUISITION_SESSIONS,
        acquisitions_root=ACQUISITIONS_ROOT,
        cone_half_deg=CONE_HALF_DEG,
    )
    keep = ~np.isnan(classes)
    sonar = sonar[keep]; classes = classes[keep].astype(int)
    pole_az = pole_az[keep]
    n_wall = int((classes == 0).sum()); n_pole = int((classes == 1).sum())
    print(f"  {len(sonar)} pings: wall={n_wall}, pole={n_pole}, "
          f"prior(pole)={n_pole/len(sonar):.3f}")

    # Also compute nearest-reflector distance for stratified failure analysis.
    # The loader has already dropped tracker-miss pings, so we need to
    # recompute poses by reloading the per-session features + walking pings.
    # Simpler shortcut: re-derive from the loader's intermediate. For this
    # diagnostic we don't strictly need exact per-ping distance; use
    # post-emit TOA as a distance proxy.
    print()

    print("[2/4] Extracting features")
    X = extract_features(sonar)
    y = classes
    print(f"  feature matrix: {X.shape}")
    for i, name in enumerate(FEATURE_NAMES):
        w = X[y == 0, i]; p = X[y == 1, i]
        fmt = FEATURE_FORMATS[name]
        print(f"    {name:>9}: wall μ={w.mean():{fmt}} σ={w.std():{fmt}}   "
              f"pole μ={p.mean():{fmt}} σ={p.std():{fmt}}")
    print()

    print("[3/4] 5-fold stratified logistic regression")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipe, X, y, cv=skf)
    acc = float(np.mean(y_pred == y))
    print(f"  CV accuracy: {acc:.1%}")

    # Per-class precision/recall
    for ci, name in enumerate(["wall", "pole"]):
        truth = (y == ci)
        pred  = (y_pred == ci)
        prec = float((truth & pred).sum()) / max(int(pred.sum()), 1)
        rec  = float((truth & pred).sum()) / max(int(truth.sum()), 1)
        print(f"    {name}: precision={prec:.2f}  recall={rec:.2f}  n_true={int(truth.sum())}")

    # Fit a single model on all data to read feature importance
    pipe.fit(X, y)
    coefs = pipe.named_steps["lr"].coef_[0]
    print("\n  Feature coefficients (z-scored space, positive = pole-leaning):")
    order = np.argsort(np.abs(coefs))[::-1]
    for k in order:
        print(f"    {FEATURE_NAMES[k]:>9}: {coefs[k]:+.3f}")

    # Stratify accuracy by TOA bin (distance proxy) and pole azimuth
    print("\n  Failure stratification:")
    toa = X[:, FEATURE_NAMES.index("toa")]
    for lo, hi in [(0, 30), (30, 60), (60, 100), (100, 190)]:
        mask = (toa >= lo) & (toa < hi)
        if mask.any():
            local_acc = float((y_pred[mask] == y[mask]).mean())
            n_w = int(((y == 0) & mask).sum()); n_p = int(((y == 1) & mask).sum())
            print(f"    TOA [{lo:>3}-{hi:>3}]: acc={local_acc:.1%}  "
                  f"(n_wall={n_w}, n_pole={n_p})")

    pole_mask = (y == 1)
    if pole_mask.any():
        az = pole_az[pole_mask]; pred_on_pole = (y_pred[pole_mask] == 1)
        for lo, hi in [(0, 10), (10, 20), (20, 35)]:
            mask = (np.abs(az) >= lo) & (np.abs(az) < hi)
            if mask.any():
                local = float(pred_on_pole[mask].mean())
                print(f"    |pole_az| [{lo:>2}-{hi:>2}°]: pole-recall={local:.1%}  (n={int(mask.sum())})")
    print()

    print("[4/4] Saving plots")
    plot_features_by_class(X, y, OUT_DIR / "features_by_class.png")
    plot_pca(X, y, OUT_DIR / "pca_scatter.png")
    plot_confusion(y, y_pred, OUT_DIR / "confusion.png", acc)
    print(f"  → {OUT_DIR}/")

    print("\nVerdict heuristic:")
    if acc >= 0.75:
        print(f"  acc={acc:.1%} ≥ 75% — signal is clearly in the envelope.")
        print(f"  The CNN's poor performance is data-starvation, not a physical limit.")
    elif acc >= 0.62:
        print(f"  acc={acc:.1%} 62-75% — moderate signal. CNN should improve with more data,")
        print(f"  but the discrimination problem may have intrinsic limits.")
    else:
        print(f"  acc={acc:.1%} < 62% — weak signal. Check feature distributions: the classes")
        print(f"  may overlap too much in this feature space. Could mean (a) the sonar truly")
        print(f"  doesn't discriminate well at the typical distances/angles in this arena, or")
        print(f"  (b) richer features (e.g. raw envelope, time-frequency) are needed.")


if __name__ == "__main__":
    main()
