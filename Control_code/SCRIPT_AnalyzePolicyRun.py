#!/usr/bin/env python3
"""
SCRIPT_AnalyzePolicyRun.py

Read the per-step TSV produced by SCRIPT_RunPolicy.py and produce a
diagnostic figure showing where the SonarModel's live distance/σ estimates
disagree with the simulator's geometric truth at the tracker pose.

Usage:
  .venv/bin/python SCRIPT_AnalyzePolicyRun.py [SESSION]

Where SESSION is the folder name under PolicyRuns/, e.g.
  rnn_sup_loop2_h32_nosigma_run05
If omitted, defaults to SESSION at the top of this script.

Outputs (alongside the input TSV):
  step_metrics_analysis.png — 3-row figure:
     row 1: per-slice scatter (d_live vs d_clean) coloured by step index.
     row 2: per-slice residual (d_live − d_clean) and σ vs step.
     row 3: envelope max per channel + commanded rotation, vs step.
"""
import os
import sys

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SESSION    = "rnn_sup_loop2_h32_nosigma_run05"
DATA_ROOT  = "PolicyRuns"
SLICES     = ("right", "center", "left")
SLICE_COLOUR = {"right": "tab:red", "center": "tab:gray", "left": "tab:blue"}


def _load(session: str) -> pd.DataFrame:
    path = os.path.join(DATA_ROOT, session, "step_metrics.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"step_metrics.tsv not found at {path}")
    df = pd.read_csv(path, sep="\t")
    print(f"Loaded {len(df)} steps from {path}")
    return df


def _summary(df: pd.DataFrame) -> None:
    print("\nLive – clean residuals (mm):")
    print(f"  {'slice':<7} {'n':>4} {'mean':>7} {'std':>6} {'median':>7} "
          f"{'|abs| med':>9} {'σ_live mean':>12} {'σ_live max':>11}")
    for s in SLICES:
        res = df[f"d_live_{s}_mm"] - df[f"d_clean_{s}_mm"]
        sig = df[f"s_live_{s}_mm"]
        n   = res.notna().sum()
        print(f"  {s:<7} {n:>4d} "
              f"{res.mean():>+7.0f} {res.std():>6.0f} "
              f"{res.median():>+7.0f} {res.abs().median():>9.0f} "
              f"{sig.mean():>12.0f} {sig.max():>11.0f}")

    # Whether the right-slice bias is concentrated in any pose region
    res_r = df["d_live_right_mm"] - df["d_clean_right_mm"]
    if res_r.notna().any():
        print(f"\nRight-slice residual by trajectory third:")
        third = len(df) // 3
        for label, sub in [("first",  df.iloc[:third]),
                           ("middle", df.iloc[third:2*third]),
                           ("last",   df.iloc[2*third:])]:
            r = sub["d_live_right_mm"] - sub["d_clean_right_mm"]
            print(f"  {label:<7} n={len(sub):>3}  mean={r.mean():>+5.0f}  std={r.std():>5.0f}")


def _plot(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))

    step = df["step"].values

    # Row 1: scatter d_live vs d_clean per slice, coloured by step
    for j, s in enumerate(SLICES):
        ax = axes[0, j]
        x = df[f"d_clean_{s}_mm"].values
        y = df[f"d_live_{s}_mm"].values
        valid = np.isfinite(x) & np.isfinite(y)
        sc = ax.scatter(x[valid], y[valid], c=step[valid],
                        cmap="viridis", s=20, alpha=0.85)
        lim = float(np.nanmax([x[valid].max() if valid.any() else 0,
                               y[valid].max() if valid.any() else 0])) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"d_clean_{s}_mm  (geometric truth)")
        ax.set_ylabel(f"d_live_{s}_mm  (model)")
        ax.set_title(f"{s} slice", color=SLICE_COLOUR[s])
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("step", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Row 2: residual (live - clean) per slice over step, with ±σ_live band
    for j, s in enumerate(SLICES):
        ax = axes[1, j]
        res = df[f"d_live_{s}_mm"] - df[f"d_clean_{s}_mm"]
        sig = df[f"s_live_{s}_mm"]
        ax.fill_between(step, -sig, sig, color="grey", alpha=0.18,
                        label="±σ_live")
        ax.plot(step, res, color=SLICE_COLOUR[s], lw=1.0, marker="o", ms=3,
                label="live − clean")
        ax.axhline(0, color="black", lw=0.6, alpha=0.6)
        ax.set_xlabel("step")
        ax.set_ylabel("residual (mm)")
        ax.set_title(f"{s}: live − clean over time")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # Row 3: envelope max per channel + rotation command, all vs step
    ax = axes[2, 0]
    ax.plot(step, df["envL_max"], color="steelblue", lw=1.0, label="envL max")
    ax.plot(step, df["envR_max"], color="firebrick", lw=1.0, label="envR max")
    ax.set_xlabel("step")
    ax.set_ylabel("envelope max")
    ax.set_title("Per-step envelope peaks")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[2, 1]
    ax.plot(step, df["rot_deg"], color="purple", lw=1.0, marker="o", ms=3)
    ax.axhline(0, color="black", lw=0.6, alpha=0.6)
    ax.set_xlabel("step")
    ax.set_ylabel("commanded rotation (°)")
    ax.set_title("Policy rotation command per step")
    ax.grid(True, alpha=0.3)

    # Trajectory overlay (x, y), coloured by step.
    ax = axes[2, 2]
    sc = ax.scatter(df["x_mm"], df["y_mm"], c=step, cmap="viridis", s=14)
    ax.plot(df["x_mm"], df["y_mm"], color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel("x_mm")
    ax.set_ylabel("y_mm")
    ax.set_aspect("equal")
    ax.set_title("Trajectory (coloured by step)")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label("step", fontsize=8)

    fig.suptitle(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main(session: str) -> None:
    df = _load(session)
    _summary(df)
    out_path = os.path.join(DATA_ROOT, session, "step_metrics_analysis.png")
    _plot(df, out_path)


if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else SESSION
    main(session)
