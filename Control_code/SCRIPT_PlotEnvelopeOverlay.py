#!/usr/bin/env python3
"""
SCRIPT_PlotEnvelopeOverlay.py

Overlay a live deploy envelope on top of each training session's mean
envelope, to spot systematic gain or shape drift between train and deploy.

Inputs:
  - DEPLOY_DUMP: path to a `_envelope_step0.npz` saved by SCRIPT_RunPolicy
                 (writes step-0 L and R arrays, shape (200,) each).
  - SESSION_PATHS: training sessions; each contributes its per-sample mean
                 envelope (and a 10–90 percentile band).

Output:
  - A 2-panel figure (L on top, R on bottom) with x = sample index,
    training-session means as thin lines, the live deploy envelope as a
    bold line, and the training sonar_norm mean ±1σ as a horizontal band.
  - Saved next to the DEPLOY_DUMP as `_envelope_overlay.png`.
"""

import os
import sys

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import Library.Settings as _settings
_settings.data_folder = "SonarSessions"
from Library.DataProcessor import DataProcessor
from Library.SonarModel   import SonarModel


SESSION_PATHS = ["sessionB01", "sessionB02", "sessionB03", "sessionB04", "sessionB05"]
DEFAULT_DEPLOY_DUMP = (
    "PolicyRuns/rnn_sup_loop2_h32_nosigma_run01/_envelope_step0.npz"
)


def _training_means():
    """Return dict session_name -> (mean_L, mean_R, p10_L, p90_L, p10_R, p90_R)
    where each array has shape (samples,). Loads from session caches."""
    out = {}
    for s in SESSION_PATHS:
        proc = DataProcessor(s, cache_dir="Cache", force_recompute=False)
        proc.load_sonar(flatten=False)
        sd = np.asarray(proc.sonar_data, dtype=np.float32)  # (N, T, 2)
        L, R = sd[:, :, 0], sd[:, :, 1]
        out[s] = (
            L.mean(axis=0), R.mean(axis=0),
            np.percentile(L, 10, axis=0), np.percentile(L, 90, axis=0),
            np.percentile(R, 10, axis=0), np.percentile(R, 90, axis=0),
        )
    return out


def main(deploy_dump_path: str = DEFAULT_DEPLOY_DUMP):
    if not os.path.exists(deploy_dump_path):
        raise FileNotFoundError(f"deploy dump not found: {deploy_dump_path}")

    dump = np.load(deploy_dump_path)
    L_live = np.asarray(dump["L"], dtype=np.float64)
    R_live = np.asarray(dump["R"], dtype=np.float64)
    n_samples = len(L_live)
    x = np.arange(n_samples)
    print(f"Loaded deploy dump: {deploy_dump_path}  (samples={n_samples})")

    m = SonarModel.load("SonarModel")
    norm_mu, norm_std = m._s_mean, m._s_std
    print(f"Saved sonar_norm: ({norm_mu:.0f}, {norm_std:.0f})")

    print("Loading training session means…")
    means = _training_means()

    fig, (axL, axR) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, (s, (mL, mR, pL10, pL90, pR10, pR90)) in enumerate(means.items()):
        c = cmap(i)
        axL.fill_between(x, pL10, pL90, color=c, alpha=0.10)
        axL.plot(x, mL, color=c, lw=1.0, alpha=0.8, label=f"{s}")
        axR.fill_between(x, pR10, pR90, color=c, alpha=0.10)
        axR.plot(x, mR, color=c, lw=1.0, alpha=0.8, label=f"{s}")

    axL.plot(x, L_live, color="black", lw=2.0, label="live (step 0)")
    axR.plot(x, R_live, color="black", lw=2.0, label="live (step 0)")

    for ax in (axL, axR):
        ax.axhline(norm_mu, color="grey", linestyle=":", lw=0.8, alpha=0.7,
                   label=f"sonar_norm μ={norm_mu:.0f}")
        ax.axhspan(norm_mu - norm_std, norm_mu + norm_std,
                   color="grey", alpha=0.06)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("envelope (raw counts)")

    axL.set_title(
        f"L channel  —  live mean={L_live.mean():.0f}  std={L_live.std():.0f}"
    )
    axR.set_title(
        f"R channel  —  live mean={R_live.mean():.0f}  std={R_live.std():.0f}"
    )
    axR.set_xlabel("sample index")
    axL.legend(loc="upper right", fontsize=8, ncol=2)
    axR.legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(
        "Live deploy envelope (step 0) vs. training-session mean envelopes "
        f"\n(emit pulse at samples 0–9; bands = 10–90 percentile per training session)"
    )
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(deploy_dump_path),
                            "_envelope_overlay.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DEPLOY_DUMP
    main(path)
