#!/usr/bin/env python3
"""
SCRIPT_AssessSonarSessions.py

Per-session quality check on the SonarSessions/sessionBxx data used to train
the 3-slice SonarModel.

Goal: catch sessions where the sonar transducer was bumped/rotated and the
ping pipeline ends up reporting `corrected_distance` far shorter than any
wall in the arena could explain — i.e. the echo locator is locking onto a
ground or chassis return.

Per session it prints:
  - sample count, L/R envelope mean/std (compare to saved sonar_norm).
  - corrected_distance summary.
  - profile min over the full 270° opening (any direction the arena has a
    wall) and over the ±cone_half_deg forward cone (what a forward sonar
    should be able to echo off).
  - fraction of pings with `corrected_distance` below the full-profile
    floor — physically impossible in a wall-only world.

Per session it writes a scatter figure:
  - x: forward-cone min wall distance (mm).
  - y: corrected_distance (mm).
  - y = x diagonal for reference.
  - points coloured by floor-violation severity.

Outputs land in `SonarSessions/_assessment/`.
"""

import os

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import Library.Settings as _settings
_settings.data_folder = "SonarSessions"
from Library.DataProcessor import DataProcessor
from Library.SonarModel   import SonarModel


SESSION_PATHS  = ["sessionB01", "sessionB02", "sessionB03", "sessionB04", "sessionB05"]
OUT_DIR        = "SonarSessions/_assessment"
OPENING_ANGLE  = 270.0
PROFILE_STEPS  = 90
PROFILE_METHOD = "ray_center"

GROUND_THRESH  = 0.7   # cd < this × profile_min flagged "severe" (likely ground echo)

# Sample indices probed for the mean-envelope shape table (200-sample ping)
SHAPE_PROBE_IDX = [0, 2, 5, 10, 20, 40, 60, 80, 100, 120, 150, 180, 199]

# Per-session example-traces figure: how many pings to plot, and grid shape
EXAMPLE_GRID = (3, 3)   # → 9 example traces evenly spaced through the session


def _profile_geom(opening_angle: float, profile_steps: int):
    edges   = np.linspace(-opening_angle / 2, opening_angle / 2, profile_steps + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers


def assess(session_name: str, sonar_norm: tuple, cone_half_deg: float) -> dict:
    proc = DataProcessor(session_name, cache_dir="Cache", force_recompute=False)
    proc.load_sonar(flatten=False)
    proc.load_profiles(opening_angle=OPENING_ANGLE, steps=PROFILE_STEPS,
                       profile_method=PROFILE_METHOD)

    sd       = np.asarray(proc.sonar_data, dtype=np.float32)
    profiles = np.asarray(proc.profiles,   dtype=np.float32)
    cd_mm    = np.asarray(proc.get_field("sonar_package", "corrected_distance"),
                          dtype=np.float64) * 1000.0

    n = len(sd)
    L = sd[:, :, 0]; R = sd[:, :, 1]

    centers     = _profile_geom(OPENING_ANGLE, PROFILE_STEPS)
    cone_mask   = np.abs(centers) <= cone_half_deg
    full_min_mm = np.nanmin(profiles, axis=1)
    cone_min_mm = np.nanmin(profiles[:, cone_mask], axis=1)

    valid       = np.isfinite(cd_mm) & np.isfinite(full_min_mm) & np.isfinite(cone_min_mm)
    below_floor = valid & (cd_mm < full_min_mm)
    severe      = valid & (cd_mm < GROUND_THRESH * full_min_mm)

    # Per-ping argmax: where does the envelope peak land? Samples 0–9 are the
    # emit-pulse leak; a healthy session has the bulk of pings peaking in there
    # because it's by far the loudest signal. A ping that peaks deeper means a
    # very strong wall echo (or saturation/noise).
    argmax_L = L.argmax(axis=1)
    argmax_R = R.argmax(axis=1)
    mean_env_L = L.mean(axis=0)
    mean_env_R = R.mean(axis=0)
    probe = [k for k in SHAPE_PROBE_IDX if 0 <= k < L.shape[1]]

    s_mu, s_std = sonar_norm
    print(f"\n── {session_name} ──")
    print(f"  samples:           {n}")
    print(f"  L envelope:        mean={L.mean():.0f}  std={L.std():.0f}  "
          f"max={L.max():.0f}  (saved norm: {s_mu:.0f}, {s_std:.0f})")
    print(f"  R envelope:        mean={R.mean():.0f}  std={R.std():.0f}  "
          f"max={R.max():.0f}")
    print(f"  argmax L:          median={int(np.median(argmax_L))}  "
          f"mean={argmax_L.mean():.1f}  count<10={int((argmax_L < 10).sum())}/{n}")
    print(f"  argmax R:          median={int(np.median(argmax_R))}  "
          f"mean={argmax_R.mean():.1f}  count<10={int((argmax_R < 10).sum())}/{n}")
    print(f"  mean envelope shape  (idx → L | R):")
    for k in probe:
        print(f"     sample {k:>3d}:    {mean_env_L[k]:>6.0f}  |  {mean_env_R[k]:>6.0f}")
    print(f"  cd (mm):           median={np.median(cd_mm):.0f}  mean={cd_mm.mean():.0f}  "
          f"std={cd_mm.std():.0f}  range=[{cd_mm.min():.0f}, {cd_mm.max():.0f}]")
    print(f"  profile_min full:  median={np.nanmedian(full_min_mm):.0f}  "
          f"mean={np.nanmean(full_min_mm):.0f}")
    print(f"  profile_min cone:  median={np.nanmedian(cone_min_mm):.0f}  "
          f"mean={np.nanmean(cone_min_mm):.0f}")
    print(f"  cd < profile_min:        {below_floor.sum():>3}/{n}  "
          f"({100 * below_floor.mean():.1f}%)")
    print(f"  cd < {GROUND_THRESH:.1f}×profile_min: {severe.sum():>3}/{n}  "
          f"({100 * severe.mean():.1f}%)  ← likely ground echoes")

    fig, ax = plt.subplots(figsize=(7, 6))
    ok_mask = valid & ~below_floor
    mid     = valid & below_floor & ~severe
    ax.scatter(cone_min_mm[ok_mask], cd_mm[ok_mask],
               s=10, alpha=0.5, color="steelblue", label="cd ≥ profile_min")
    ax.scatter(cone_min_mm[mid], cd_mm[mid],
               s=14, alpha=0.7, color="orange", label="cd < profile_min")
    ax.scatter(cone_min_mm[severe], cd_mm[severe],
               s=18, alpha=0.9, color="red",
               label=f"cd < {GROUND_THRESH:.1f}×profile_min (likely ground)")
    lim = float(np.nanmax([cone_min_mm[valid].max(), cd_mm[valid].max()])) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("forward-cone min wall distance (mm)")
    ax.set_ylabel("corrected sonar distance (mm)")
    ax.set_title(f"{session_name}  —  {n} pings, "
                 f"{below_floor.sum()} below floor, {severe.sum()} severe")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{session_name}_cd_vs_profile.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → {out_path}")

    # Example envelope traces: a small grid of individual pings sampled
    # evenly across the session, with cone-min / cd annotated for context.
    rows, cols = EXAMPLE_GRID
    n_show = rows * cols
    pick = np.linspace(0, n - 1, n_show, dtype=int)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.0 * rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    x = np.arange(L.shape[1])
    for ax, idx in zip(axes, pick):
        Li, Ri = L[idx], R[idx]
        ax.plot(x, Li, color="steelblue", lw=0.8, alpha=0.9, label="L")
        ax.plot(x, Ri, color="firebrick", lw=0.8, alpha=0.9, label="R")
        ax.set_title(
            f"#{idx}  cd={cd_mm[idx]:.0f}  cone_min={cone_min_mm[idx]:.0f}\n"
            f"L [{Li.min():.0f},{Li.max():.0f}]  R [{Ri.min():.0f},{Ri.max():.0f}]",
            fontsize=7,
        )
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=7)
    for ax in axes[-cols:]: ax.set_xlabel("sample idx", fontsize=7)
    for ax in axes[::cols]: ax.set_ylabel("envelope", fontsize=7)
    fig.suptitle(f"{session_name}  —  example envelopes (L blue, R red), {n_show}/{n} pings")
    fig.tight_layout()
    ex_path = os.path.join(OUT_DIR, f"{session_name}_envelope_examples.png")
    fig.savefig(ex_path, dpi=120)
    plt.close(fig)
    print(f"  → {ex_path}")

    return {
        "n":            n,
        "below_floor":  int(below_floor.sum()),
        "severe":       int(severe.sum()),
        "L_mean":       float(L.mean()), "L_std": float(L.std()),
        "R_mean":       float(R.mean()), "R_std": float(R.std()),
        "cd_median_mm": float(np.median(cd_mm)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m          = SonarModel.load("SonarModel")
    sonar_norm = (m._s_mean, m._s_std)
    cone_half  = m.get_cone_half_deg()
    print(f"Saved sonar_norm: ({sonar_norm[0]:.0f}, {sonar_norm[1]:.0f})")
    print(f"Cone half-angle:  ±{cone_half:.0f}°  (forward cone for cone_min)")

    summary = {}
    for s in SESSION_PATHS:
        try:
            summary[s] = assess(s, sonar_norm, cone_half)
        except Exception as exc:
            print(f"\n── {s} ──\n  ❌ failed: {exc}")

    print("\n──────── summary ────────")
    print(f"  {'session':<14} {'n':>4} {'L mean':>7} {'L std':>6} "
          f"{'R mean':>7} {'R std':>6} {'cd_med':>7} {'below':>6} {'severe':>7}")
    for s, info in summary.items():
        print(f"  {s:<14} {info['n']:>4} "
              f"{info['L_mean']:>7.0f} {info['L_std']:>6.0f} "
              f"{info['R_mean']:>7.0f} {info['R_std']:>6.0f} "
              f"{info['cd_median_mm']:>7.0f} "
              f"{info['below_floor']:>6} {info['severe']:>7}")


if __name__ == "__main__":
    main()
