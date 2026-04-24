#!/usr/bin/env python3
"""
SCRIPT_SonarSpatialInfo.py

Diagnostic: which scalar sonar features carry the most spatial information?

For every pair of steps in a real-robot run, computes:
  - physical XY distance
  - heading difference |Δyaw|
  - sensory distance under several scalar feature representations

Two analyses:
  1. Spearman ρ between sensory distance and XY distance, by heading bin.
  2. Precision@K place recognition (fixed "same place" threshold, heading known).

Output:
  SpatialInfo/<run>/sonar_spatial_info.png
  SpatialInfo/<run>/sonar_precision_at_k.png
"""

import glob
import os

import dill
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ── Run to analyse ────────────────────────────────────────────────────────────
REAL_RUN_DIR    = "PolicyRuns/session_test2_h01_arena1_01"

# Minimum step gap between paired steps (avoid trivially-similar neighbours)
MIN_STEP_GAP    = 15

# Maximum pairs per heading bin (random subsample for speed)
MAX_PAIRS_PER_BIN = 5000
SEED            = 0

# Heading bins (degrees)
HEADING_BINS_DEG = [0, 15, 30, 60, 90, 180]

# Window sizes to test
WINDOW_SIZES    = [1, 4, 8]

# ── Precision@K parameters ────────────────────────────────────────────────────
SAME_PLACE_XY_MM   = 300.0
SAME_PLACE_YAW_DEG = 30.0
CAND_YAW_DEG       = 30.0
K_VALUES           = [1, 3, 5, 10]


# ══════════════════════════════════════════════════════════════════════════════
# Load run
# ══════════════════════════════════════════════════════════════════════════════

def load_run(run_dir):
    """
    Returns per-step arrays:
      positions    (N, 2)  XY in mm
      yaws_rad     (N,)
      scalars_norm (N, 15) z-scored
      scalars_raw  (N, 15) raw values

    Scalar columns
    --------------
    0  dist_mm        corrected distance to echo (mm)
    1  iid_db         calibrated IID (dB)           — echo window, calibrated
    2  log_L          log left-ear echo integral     — echo window only
    3  log_R          log right-ear echo integral
    4  r1_deg         rotate1 (deg)
    5  r2_deg         rotate2 (deg)
    6  log_sumL_total log of left-channel envelope sum (all 200 bins)
    7  log_sumR_total log of right-channel envelope sum
    8  prom_L         echo prominence left  = echo_integral / total_sum
    9  prom_R         echo prominence right
    10 iid_total      IID of full envelope  = log(sumL_total / sumR_total)
    11 iid_post       IID of reverb tail    = log(sumL_post  / sumR_post)
    12 peak_L         max amplitude in echo window, left ear
    13 peak_R         max amplitude in echo window, right ear
    14 xcorr          Pearson r(L, R) over echo window  — reflection complexity
    """
    files = sorted(glob.glob(os.path.join(run_dir, "data*.dill")))
    if not files:
        raise FileNotFoundError(f"No data*.dill in {run_dir}")
    print(f"Loading {len(files)} steps …")

    xs, ys, yaws_deg = [], [], []
    rows = []

    for p in files:
        with open(p, "rb") as f:
            d = dill.load(f)
        pos = d["data"]["position"]
        mot = d["data"]["motion"]
        sp  = d["data"]["sonar_package"]
        env = sp["sonar_data"][:, :2].astype(np.float64)   # (200, 2) L and R
        onset  = int(sp["onset"])
        offset = int(sp["offset"])

        sum_total = env.sum(axis=0) + 1.0              # (2,) avoid log(0)
        sum_post  = env[offset:, :].sum(axis=0) + 1.0
        integrals = np.array(sp["integrals"], dtype=np.float64) + 1.0
        echo_win  = env[onset:offset, :]               # echo window
        peak      = echo_win.max(axis=0)               # (2,) peak per ear
        lw, rw = echo_win[:, 0], echo_win[:, 1]
        lw_c, rw_c = lw - lw.mean(), rw - rw.mean()
        denom = np.sqrt((lw_c**2).sum() * (rw_c**2).sum())
        xcorr = float((lw_c * rw_c).sum() / denom) if denom > 1e-9 else 0.0

        xs.append(float(pos.get("x") or 0.0))
        ys.append(float(pos.get("y") or 0.0))
        yaws_deg.append(float(pos.get("yaw_deg") or 0.0))
        rows.append([
            float(sp["corrected_distance"]) * 1000.0,   # 0  dist_mm
            float(sp["corrected_iid"]),                  # 1  iid_db
            float(sp["log_integrals"][0]),               # 2  log_L
            float(sp["log_integrals"][1]),               # 3  log_R
            float(mot["rotate1"]),                       # 4  r1_deg
            float(mot["rotate2"]),                       # 5  r2_deg
            float(np.log(sum_total[0])),                 # 6  log_sumL_total
            float(np.log(sum_total[1])),                 # 7  log_sumR_total
            float(integrals[0] / sum_total[0]),          # 8  prom_L
            float(integrals[1] / sum_total[1]),          # 9  prom_R
            float(np.log(sum_total[0] / sum_total[1])),  # 10 iid_total
            float(np.log(sum_post[0]  / sum_post[1])),   # 11 iid_post
            float(peak[0]),                              # 12 peak_L
            float(peak[1]),                              # 13 peak_R
            xcorr,                                       # 14 xcorr
        ])

    positions = np.column_stack([xs, ys]).astype(np.float64)
    yaws_rad  = np.radians(yaws_deg)
    scalars   = np.array(rows, dtype=np.float64)        # (N, 15)

    means = scalars.mean(axis=0)
    stds  = scalars.std(axis=0)
    stds[stds < 1e-9] = 1.0
    scalars_norm = ((scalars - means) / stds).astype(np.float32)

    col_names = ["dist_mm", "iid_db", "log_L", "log_R", "r1", "r2",
                 "log_sumL_total", "log_sumR_total",
                 "prom_L", "prom_R",
                 "iid_total", "iid_post",
                 "peak_L", "peak_R", "xcorr"]
    print(f"  {len(positions)} steps  ({len(col_names)} scalar features)")
    for i, name in enumerate(col_names):
        print(f"    {name:<18} {scalars[:,i].min():>10.3f} – {scalars[:,i].max():.3f}")

    return positions, yaws_rad, scalars_norm, scalars


# ══════════════════════════════════════════════════════════════════════════════
# Feature builders  (all return L2-comparable (N, D) float32 arrays)
# ══════════════════════════════════════════════════════════════════════════════
# Column indices in scalars_norm:
#  0=dist    1=iid     2=log_L   3=log_R   4=r1      5=r2
#  6=logSL_tot  7=logSR_tot
#  8=prom_L  9=prom_R  10=iid_total  11=iid_post  12=peak_L  13=peak_R  14=xcorr

FEATURE_SETS = {
    # ── Baselines ─────────────────────────────────────────────────────────────
    "dist+IID+r1+r2":                       [0, 1, 4, 5],
    "dist+logL+logR+r1+r2":                 [0, 2, 3, 4, 5],

    # ── Best from previous round ──────────────────────────────────────────────
    "dist+logL+logR+prom+r1+r2":            [0, 2, 3, 8, 9, 4, 5],
    "dist+IID+sumLR_total+r1+r2":           [0, 1, 6, 7, 4, 5],
    "dist+logL+logR+sumLR_total+r1+r2":     [0, 2, 3, 6, 7, 4, 5],

    # ── IID at different temporal scales ─────────────────────────────────────
    "dist+IID+iid_total+r1+r2":             [0, 1, 10, 4, 5],
    "dist+IID+iid_post+r1+r2":              [0, 1, 11, 4, 5],
    "dist+IID+iid_total+iid_post+r1+r2":    [0, 1, 10, 11, 4, 5],
    "dist+logL+logR+iid_total+iid_post+r1+r2": [0, 2, 3, 10, 11, 4, 5],

    # ── Peak amplitude per ear ────────────────────────────────────────────────
    "dist+IID+peak+r1+r2":                  [0, 1, 12, 13, 4, 5],
    "dist+logL+logR+peak+r1+r2":            [0, 2, 3, 12, 13, 4, 5],

    # ── Cross-correlation (reflection complexity) ─────────────────────────────
    "dist+IID+xcorr+r1+r2":                 [0, 1, 14, 4, 5],
    "dist+logL+logR+xcorr+r1+r2":           [0, 2, 3, 14, 4, 5],

    # ── Combinations ─────────────────────────────────────────────────────────
    "dist+logL+logR+prom+iid_total+iid_post+r1+r2":
                                            [0, 2, 3, 8, 9, 10, 11, 4, 5],
    "dist+logL+logR+prom+peak+xcorr+r1+r2": [0, 2, 3, 8, 9, 12, 13, 14, 4, 5],
    "dist+logL+logR+prom+iid_total+peak+xcorr+r1+r2":
                                            [0, 2, 3, 8, 9, 10, 12, 13, 14, 4, 5],
}


def build_feature_matrices(scalars_norm):
    """Returns dict: name → (N, D) float32 array (not yet windowed)."""
    feat = {}
    for name, cols in FEATURE_SETS.items():
        feat[name] = scalars_norm[:, cols].astype(np.float32)

    # Monaural level (log_L + log_R) as a single derived column
    log_mono = scalars_norm[:, 2:4].sum(axis=1, keepdims=True)
    feat["dist+mono+IID+r1+r2"] = np.hstack([
        scalars_norm[:, [0]], log_mono, scalars_norm[:, [1, 4, 5]]
    ]).astype(np.float32)
    feat["dist+mono+IID+prom+iid_total+iid_post+r1+r2"] = np.hstack([
        scalars_norm[:, [0]], log_mono,
        scalars_norm[:, [1, 8, 9, 10, 11, 4, 5]]
    ]).astype(np.float32)

    return feat


# ══════════════════════════════════════════════════════════════════════════════
# Windowing
# ══════════════════════════════════════════════════════════════════════════════

def apply_window(mat, window):
    N, D = mat.shape
    out = np.zeros((N, window * D), dtype=np.float64)
    for i in range(N):
        for w in range(window):
            j = i - w
            if j >= 0:
                out[i, w * D:(w + 1) * D] = mat[j]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (out / norms).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Pairwise analysis helpers
# ══════════════════════════════════════════════════════════════════════════════

def wrap_rad(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def compute_pairwise(positions, yaws_rad, feat_dict, rng):
    N = len(positions)
    ii, jj = np.triu_indices(N, k=MIN_STEP_GAP)
    print(f"  {len(ii):,} candidate pairs")

    xy_dist  = np.linalg.norm(positions[ii] - positions[jj], axis=1)
    dheading = np.abs(np.degrees(wrap_rad(yaws_rad[ii] - yaws_rad[jj])))
    hbin_ids = np.digitize(dheading, HEADING_BINS_DEG[1:])

    results = {}
    for name, feat_mat in feat_dict.items():
        print(f"  {name} …", end="", flush=True)
        for hb in range(len(HEADING_BINS_DEG) - 1):
            idx = np.where(hbin_ids == hb)[0]
            if len(idx) < 10:
                continue
            if len(idx) > MAX_PAIRS_PER_BIN:
                idx = rng.choice(idx, MAX_PAIRS_PER_BIN, replace=False)
            diff  = feat_mat[ii[idx]].astype(np.float64) - feat_mat[jj[idx]].astype(np.float64)
            sdist = np.sqrt(np.sum(diff ** 2, axis=1))
            results[(name, hb)] = (xy_dist[idx], sdist)
        print(" done")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Spearman analysis
# ══════════════════════════════════════════════════════════════════════════════

def spearman_table(results, feat_names):
    n_bins = len(HEADING_BINS_DEG) - 1
    rho_mat = np.full((len(feat_names), n_bins), np.nan)
    for fi, name in enumerate(feat_names):
        for hb in range(n_bins):
            key = (name, hb)
            if key not in results:
                continue
            xy, sd = results[key]
            if len(xy) >= 10:
                rho_mat[fi, hb], _ = spearmanr(sd, xy)
    return rho_mat


def plot_spearman(feat_names, rho_mat, run_name, output_dir):
    bin_labels = [f"{HEADING_BINS_DEG[i]}–{HEADING_BINS_DEG[i+1]}°"
                  for i in range(len(HEADING_BINS_DEG) - 1)]
    n_bins, n_feats = len(bin_labels), len(feat_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n_feats * 0.32 + 2)))
    fig.suptitle(f"Sonar spatial info (Spearman ρ) — {run_name}", fontsize=11)
    cmap = plt.cm.tab10

    ax = axes[0]
    for fi, name in enumerate(feat_names):
        v = rho_mat[fi]
        valid = ~np.isnan(v)
        ax.plot(np.arange(n_bins)[valid], v[valid], "o-",
                label=name, color=cmap(fi % 10), linewidth=1.5, markersize=5)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels, rotation=20, ha="right")
    ax.set_ylabel("Spearman ρ  (sensory dist vs XY dist)")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(-0.2, 1.0)

    ax = axes[1]
    im = ax.imshow(rho_mat, aspect="auto", vmin=-0.1, vmax=0.8,
                   cmap="RdYlGn", interpolation="nearest")
    ax.set_xticks(range(n_bins)); ax.set_xticklabels(bin_labels, rotation=20, ha="right")
    ax.set_yticks(range(n_feats)); ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_title("Spearman ρ heatmap", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    for fi in range(n_feats):
        for hb in range(n_bins):
            v = rho_mat[fi, hb]
            if not np.isnan(v):
                ax.text(hb, fi, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < 0.5 else "white")
    plt.tight_layout()
    path = os.path.join(output_dir, "sonar_spatial_info.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")

    col_w = max(len(n) for n in feat_names) + 2
    header = f"{'Feature':<{col_w}}" + "".join(f"{lb:>12}" for lb in bin_labels)
    print("\nSpearman ρ"); print(header); print("-" * len(header))
    order = np.argsort(rho_mat[:, 0])[::-1]
    for fi in order:
        row = f"{feat_names[fi]:<{col_w}}"
        for hb in range(n_bins):
            v = rho_mat[fi, hb]
            row += f"{'—':>12}" if np.isnan(v) else f"{v:>12.3f}"
        print(row)


# ══════════════════════════════════════════════════════════════════════════════
# Precision@K
# ══════════════════════════════════════════════════════════════════════════════

def precision_at_k(positions, yaws_rad, feat_dict):
    N = len(positions)
    prec     = {name: np.zeros(len(K_VALUES)) for name in feat_dict}
    n_queries = 0

    for t in range(N):
        gaps  = np.abs(np.arange(N) - t)
        dyaw  = np.abs(np.degrees(wrap_rad(yaws_rad - yaws_rad[t])))
        cand  = np.where((gaps >= MIN_STEP_GAP) & (dyaw < CAND_YAW_DEG))[0]
        if len(cand) == 0:
            continue
        dxy      = np.linalg.norm(positions[cand] - positions[t], axis=1)
        dyaw_c   = np.abs(np.degrees(wrap_rad(yaws_rad[cand] - yaws_rad[t])))
        pos_mask = (dxy < SAME_PLACE_XY_MM) & (dyaw_c < SAME_PLACE_YAW_DEG)
        if pos_mask.sum() == 0:
            continue
        n_queries += 1
        for name, feat_mat in feat_dict.items():
            diff  = feat_mat[cand].astype(np.float64) - feat_mat[t].astype(np.float64)
            sdist = np.sqrt(np.sum(diff ** 2, axis=1))
            order = np.argsort(sdist)
            for ki, k in enumerate(K_VALUES):
                prec[name][ki] += pos_mask[order[:k]].sum() / k

    if n_queries:
        for name in prec:
            prec[name] /= n_queries
    return prec, n_queries


def plot_precision(prec, feat_names, n_queries, run_name, output_dir):
    mat = np.array([prec[n] for n in feat_names])

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(feat_names) * 0.32 + 2)))
    fig.suptitle(
        f"Place-recognition precision@K — {run_name}\n"
        f"same place: XY<{SAME_PLACE_XY_MM:.0f}mm & |Δyaw|<{SAME_PLACE_YAW_DEG:.0f}°, "
        f"candidates: |Δyaw|<{CAND_YAW_DEG:.0f}°,  n_queries={n_queries}",
        fontsize=10,
    )

    ax = axes[0]
    p1    = mat[:, 0]
    order = np.argsort(p1)[::-1]
    bars  = ax.barh(range(len(feat_names)), p1[order], color="steelblue", height=0.6)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels([feat_names[i] for i in order], fontsize=8)
    ax.set_xlabel("Precision@1")
    ax.set_title("Precision@1 (ranked)", fontsize=10)
    ax.set_xlim(0, min(1.0, p1.max() * 1.15))
    ax.grid(True, alpha=0.3, axis="x")
    for i, bar in enumerate(bars):
        v = p1[order[i]]
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=7)

    ax = axes[1]
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=min(mat.max() * 1.1, 1.0),
                   cmap="YlGn", interpolation="nearest")
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([f"@{k}" for k in K_VALUES])
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_title("Precision@K heatmap", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    for fi in range(len(feat_names)):
        for ki in range(len(K_VALUES)):
            v = mat[fi, ki]
            ax.text(ki, fi, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if v < 0.5 else "white")
    plt.tight_layout()
    path = os.path.join(output_dir, "sonar_precision_at_k.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")

    col_w = max(len(n) for n in feat_names) + 2
    header = f"{'Feature':<{col_w}}" + "".join(f"{f'@{k}':>8}" for k in K_VALUES)
    print(f"\nPrecision@K  (n_queries={n_queries})")
    print(header); print("-" * len(header))
    for fi in np.argsort(mat[:, 0])[::-1]:
        row = f"{feat_names[fi]:<{col_w}}" + "".join(f"{mat[fi,ki]:>8.3f}" for ki in range(len(K_VALUES)))
        print(row)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng        = np.random.default_rng(SEED)
    run_name   = os.path.basename(REAL_RUN_DIR.rstrip("/"))
    output_dir = os.path.join("SpatialInfo", run_name)
    os.makedirs(output_dir, exist_ok=True)

    positions, yaws_rad, scalars_norm, scalars_raw = load_run(REAL_RUN_DIR)

    print("\nBuilding feature matrices …")
    base_feats = build_feature_matrices(scalars_norm)

    # Expand with windowed variants
    all_feats = {}
    for name, mat in base_feats.items():
        for w in WINDOW_SIZES:
            label = name if w == 1 else f"{name} w={w}"
            all_feats[label] = apply_window(mat, w) if w > 1 else mat
    feat_names = list(all_feats.keys())
    print(f"  {len(feat_names)} feature variants")

    print("\nComputing pairwise distances …")
    results = compute_pairwise(positions, yaws_rad, all_feats, rng)

    print("\nSpearman correlations …")
    rho_mat = spearman_table(results, feat_names)
    plot_spearman(feat_names, rho_mat, run_name, output_dir)

    print("\nPrecision@K …")
    prec, n_queries = precision_at_k(positions, yaws_rad, all_feats)
    plot_precision(prec, feat_names, n_queries, run_name, output_dir)


if __name__ == "__main__":
    main()
