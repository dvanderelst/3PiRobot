#!/usr/bin/env python3
"""
SCRIPT_DiagnoseSigmaGap.py

Compare the two σ distributions the policy sees on the same input channel:

  σ_sim    = SonarModel.sigma_sim(slice, d_true_in_slice)
             — sim-time noise model (deterministic given geometric truth).
             What EnvironmentSimulator.get_sonar_measurement uses to noise
             distance observations in predict_from_profile.

  σ_deploy = SonarModel.predict_from_envelope(L, R)["sigma_<slice>_mm"]
             — network's per-ping σ at deploy.
             What the real-robot path will feed the policy.

Both feed slot 3-5 of the policy obs vector. If they disagree in marginal
scale or condition on different things, a policy trained on σ_sim may
misread σ_deploy.

Runs on the same val split as SCRIPT_TrainSonarModel.py
(VALIDATION_QUADRANTS) so the network has not seen these samples.

Outputs:
  SonarModel/sigma_sim_vs_deploy.png   3 rows × 3 slices
    row 0: marginal histograms of σ_sim and σ_deploy
    row 1: per-sample scatter σ_deploy vs σ_sim(d_true)  +  Pearson r
    row 2: σ binned by d_true   σ_sim curve vs σ_deploy mean ± std band
  SonarModel/sigma_sim_vs_deploy.json   per-slice summary stats
"""

import json
import os

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Library.DataProcessor import DataCollection
from Library.SonarModel import SonarModel, SLICE_NAMES


# ── Settings (mirrors SCRIPT_TrainSonarModel.py) ───────────────────────

SESSION_PATHS = ["sessionB01", "sessionB02", "sessionB03", "sessionB04", "sessionB05"]
CACHE_DIR     = "Cache"

VALIDATION_QUADRANTS = {
    "sessionB01": [0],
    "sessionB02": [1],
    "sessionB03": [2],
    "sessionB04": [3],
    "sessionB05": [0],
}

MODEL_DIR    = "SonarModel"
OUT_PNG      = os.path.join(MODEL_DIR, "sigma_sim_vs_deploy.png")
OUT_JSON     = os.path.join(MODEL_DIR, "sigma_sim_vs_deploy.json")
N_DIST_BINS  = 8


# ── Data loading ─────────────────────────────────────────────────────────────

def load_val_data(model: SonarModel):
    """Load the val split exactly the way the training script does. Profile
    geometry is taken from the model artifact, so retraining with different
    settings auto-flows here."""
    pp = model.get_profile_params()
    dc = DataCollection(SESSION_PATHS, cache_dir=CACHE_DIR)
    dc.load_profiles(
        opening_angle=pp["opening_angle"],
        steps=pp["profile_steps"],
        profile_method=pp["profile_method"],
    )
    sonar_l, prof_l, quad_l, sess_l = [], [], [], []
    for proc in dc.processors:
        proc.load_sonar(flatten=False)
        sonar_l.append(np.asarray(proc.sonar_data, dtype=np.float32))
        prof_l.append(np.asarray(proc.profiles,    dtype=np.float32))
        quad_l.append(proc.quadrants)
        sess_l.append(np.array([os.path.basename(proc.session)] * proc.n))
    sonar    = np.concatenate(sonar_l, axis=0)
    profiles = np.concatenate(prof_l,  axis=0)
    quads    = np.concatenate(quad_l,  axis=0)
    sess     = np.concatenate(sess_l,  axis=0)

    is_val = np.zeros(len(quads), dtype=bool)
    for s_name, val_q in VALIDATION_QUADRANTS.items():
        is_val |= (sess == s_name) & np.isin(quads, list(val_q))
    return sonar[is_val], profiles[is_val]


# ── σ extraction per sample, per slice ───────────────────────────────────────

def slice_true_min(profiles: np.ndarray, model: SonarModel) -> np.ndarray:
    """(N, 3) — true min profile distance per slice [left, center, right]."""
    cols = [profiles[:, m].min(axis=1) for m in model.slice_masks]
    return np.stack(cols, axis=1).astype(np.float32)


def sigma_sim_per_sample(d_true: np.ndarray, model: SonarModel) -> np.ndarray:
    """(N, 3) — σ_sim(d_true) per slice, what the simulator adds to truth."""
    cols = [model.sigma_sim(name, d_true[:, i])
            for i, name in enumerate(SLICE_NAMES)]
    return np.stack(cols, axis=1).astype(np.float32)


def sigma_deploy_per_sample(sonar: np.ndarray, model: SonarModel) -> np.ndarray:
    """(N, 3) — network's per-ping σ per slice.

    sonar shape: (N, samples, 3). Channels [..., 0] = L, [..., 1] = R, matching
    SCRIPT_TrainSonarModel.make_loader and SonarModel.predict_from_envelope.
    """
    L = sonar[..., 0]
    R = sonar[..., 1]
    out = model.predict_from_envelope(L, R)
    return np.stack(
        [out[f"sigma_{name}_mm"] for name in SLICE_NAMES], axis=1
    ).astype(np.float32)


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_diagnostic(d_true, sigma_sim, sigma_dep, out_path, n_bins=N_DIST_BINS):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    summary = {}
    for i, name in enumerate(SLICE_NAMES):
        d  = d_true[:, i]
        ss = sigma_sim[:, i]
        sd = sigma_dep[:, i]

        # Row 0 — marginal histograms
        ax = axes[0, i]
        bins = np.linspace(0.0, max(ss.max(), sd.max()) * 1.05, 40)
        ax.hist(ss, bins=bins, alpha=0.55, color='gray',
                label=f"σ_sim  (mean {ss.mean():.0f})")
        ax.hist(sd, bins=bins, alpha=0.55, color='steelblue',
                label=f"σ_deploy (mean {sd.mean():.0f})")
        ax.set_xlabel("σ (mm)")
        if i == 0:
            ax.set_ylabel("count")
        ax.set_title(f"{name.capitalize()} — marginal σ", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Row 1 — per-sample scatter
        ax = axes[1, i]
        ax.scatter(ss, sd, s=8, alpha=0.45, color='steelblue')
        lo = float(min(ss.min(), sd.min()))
        hi = float(max(ss.max(), sd.max())) * 1.05
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6, label='diagonal')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        r     = float(np.corrcoef(ss, sd)[0, 1])
        ratio = float((sd / np.maximum(ss, 1e-6)).mean())
        ax.set_xlabel("σ_sim (mm)")
        if i == 0:
            ax.set_ylabel("σ_deploy (mm)")
        ax.set_title(f"per-sample agreement   r = {r:+.3f}   "
                     f"⟨σ_dep/σ_sim⟩ = {ratio:.2f}",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Row 2 — σ binned by d_true
        ax = axes[2, i]
        sorted_idx = np.argsort(d)
        bin_size = max(len(d) // n_bins, 1)
        bin_d, sim_curve, dep_mean, dep_std = [], [], [], []
        for k in range(n_bins):
            lo_k = k * bin_size
            hi_k = (k + 1) * bin_size if k < n_bins - 1 else len(d)
            idx = sorted_idx[lo_k:hi_k]
            bin_d.append(float(d[idx].mean()))
            sim_curve.append(float(ss[idx].mean()))   # ≈ σ_sim(bin_d)
            dep_mean.append(float(sd[idx].mean()))
            dep_std.append(float(sd[idx].std()))
        bin_d     = np.array(bin_d)
        sim_curve = np.array(sim_curve)
        dep_mean  = np.array(dep_mean)
        dep_std   = np.array(dep_std)
        ax.plot(bin_d, sim_curve, 'k-', lw=2, label='σ_sim(d_true)')
        ax.fill_between(bin_d, dep_mean - dep_std, dep_mean + dep_std,
                        alpha=0.25, color='steelblue', label='σ_deploy mean ± std')
        ax.plot(bin_d, dep_mean, 'o-', color='steelblue', lw=1.5)
        ax.set_xlabel("d_true in slice (mm)")
        if i == 0:
            ax.set_ylabel("σ (mm)")
        ax.set_title(f"σ vs distance   ⟨within-bin std σ_dep⟩ = "
                     f"{dep_std.mean():.0f} mm",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        summary[name] = {
            "n_samples":                     int(len(d)),
            "sigma_sim_mean_mm":             float(ss.mean()),
            "sigma_sim_median_mm":           float(np.median(ss)),
            "sigma_deploy_mean_mm":          float(sd.mean()),
            "sigma_deploy_median_mm":        float(np.median(sd)),
            "ratio_dep_over_sim_mean":       ratio,
            "pearson_dep_vs_sim":            r,
            "within_bin_std_deploy_mean_mm": float(dep_std.mean()),
        }

    fig.suptitle(
        "σ_sim vs σ_deploy on val split  "
        "(σ_sim = simulator noise model;  σ_deploy = network per-ping σ)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[1/4] Loading SonarModel")
    model = SonarModel.load(model_dir=MODEL_DIR, device="cpu")
    print(f"  {model}")

    print("\n[2/4] Loading val split (same as training)")
    sonar, profiles = load_val_data(model)
    print(f"  n_val = {len(sonar)}   sonar shape = {sonar.shape}   "
          f"profile shape = {profiles.shape}")

    print("\n[3/4] Computing σ_sim(d_true) and σ_deploy per sample")
    d_true    = slice_true_min(profiles, model)
    sigma_sim = sigma_sim_per_sample(d_true, model)
    sigma_dep = sigma_deploy_per_sample(sonar, model)

    print("\n[4/4] Plotting and writing summary")
    summary = plot_diagnostic(d_true, sigma_sim, sigma_dep, OUT_PNG)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    for name, s in summary.items():
        print(f"  {name:>6}:  σ_sim mean = {s['sigma_sim_mean_mm']:5.0f} mm   "
              f"σ_dep mean = {s['sigma_deploy_mean_mm']:5.0f} mm   "
              f"ratio = {s['ratio_dep_over_sim_mean']:.2f}   "
              f"r = {s['pearson_dep_vs_sim']:+.3f}   "
              f"within-bin σ_dep std = {s['within_bin_std_deploy_mean_mm']:.0f} mm")

    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
