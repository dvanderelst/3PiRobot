#!/usr/bin/env python3
"""
SCRIPT_SweepSLAM.py — sweep SLAM parameters and report results compactly.

Caches the ingest + odometry once, then iterates over:
  * PF-shape settings  (WINDOW_LEN, PF_BETA) — each requires a fresh PF
  * LC-filter settings (heading gate, weight threshold, dedup bucket)
                       — each reuses the cached PF history (fast)

For every combination runs LC extraction + SE(2) solve and prints one row:
  WL  β   hdg  wth dd   LCs  TP  FP  prec   final  mean

Only synthetic odometry is supported (matches the behavioural-experiment
workflow). Edit the GRID block below to change what gets swept.
"""

import contextlib
import io
import itertools
import os

import numpy as np

import SCRIPT_PoseGraphSLAM_SE2 as slam
from Library.SlamCore import build_windows, run_pf, umeyama_align


# ── Grid ────────────────────────────────────────────────────────────────
# PF-shape settings: each combo requires re-running the PF.
WINDOW_LENS     = [8]
PF_BETAS        = [2.0]

# LC-filter settings: cheap to sweep (PF history reused).
# HEADING_GATE_K is the σ_rot multiplier for the heading gate — threshold is
# K · σ_rot · √|t−s| degrees (see SCRIPT_PoseGraphSLAM_SE2.py).
HEADING_GATE_KS = [1.0, 1.5, 2.0]
WEIGHT_THRESH   = [0.25]
DEDUP_BUCKETS   = [3]

# Injected body-frame odometry noise. Sweep σ_rot to see how the SLAM
# degrades as rotation noise grows (the behavioural-experiment axis).
# σ_drive is held constant; add values here if you want to sweep it too.
SIGMA_DRIVE_MMS = [slam.SIGMA_DRIVE_MM]
SIGMA_ROT_DEGS  = [1.0, 2.0, 3.0, 4.0]

# Ground-truth radius for TP/FP labelling (same as main script's plot).
TP_RADIUS_MM    = 300.0

SEED            = slam.SEED


def _silently(fn, *args, **kwargs):
    """Run fn, swallowing its stdout chatter."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def main():
    rng = np.random.default_rng(SEED)

    # ── Ingest once ─────────────────────────────────────────────────────
    if slam.DATA_SOURCE == "real":
        run_label = f"{os.path.basename(slam.REAL_RUN_DIR.rstrip('/'))}  (real)"
        positions, yaws_rad, meas_seq, _walls, _name, commanded = slam.ingest_real(
            slam.REAL_RUN_DIR,
        )
        drive_mm_per_step = (float(np.mean(commanded["dr_mm"]))
                             if len(commanded["dr_mm"]) else 0.0)
    else:
        run_label = f"{slam.SIM_RUN_DIR}  session={slam.SIM_SESSION_NAME}  (sim)"
        positions, yaws_rad, meas_seq, _walls, _name, drive_mm_per_step = slam.ingest_sim(
            slam.SIM_RUN_DIR, slam.SIM_SESSION_NAME, slam.SIM_MAX_STEPS, rng,
        )

    N = len(positions)

    print(f"\nRun: {run_label}")
    print(f"  N={N} steps, drive={drive_mm_per_step:.0f} mm/step")

    # ── Sweep ───────────────────────────────────────────────────────────
    # Loop order (outer → inner):
    #   (WL, β)      — re-runs the PF; slow
    #   (σ_drive, σ_rot) — regenerates synthetic odometry; patches module
    #                     constants so the solver and heading gate use the
    #                     new σ; fast
    #   (hgK, wth, dd)   — relays on cached PF + fresh odometry; fastest
    header = (f"{'WL':>3} {'β':>4}  {'σd':>4} {'σr':>4}  "
              f"{'odom':>5}  "
              f"{'hgK':>4} {'wth':>4} {'dd':>3}  "
              f"{'LCs':>4} {'TP':>3} {'FP':>3} {'prec':>5}  "
              f"{'final':>6} {'mean':>5}")
    print(header)
    print("─" * len(header))

    lc_combos    = list(itertools.product(HEADING_GATE_KS, WEIGHT_THRESH, DEDUP_BUCKETS))
    sigma_combos = list(itertools.product(SIGMA_DRIVE_MMS, SIGMA_ROT_DEGS))

    for wl, beta in itertools.product(WINDOW_LENS, PF_BETAS):
        feats = build_windows(meas_seq, np.zeros(N, dtype=np.int32), wl)
        history, _ = _silently(run_pf, feats, np.random.default_rng(SEED), beta)

        for σd, σr in sigma_combos:
            # Regenerate synthetic odometry for this σ combo.
            noisy_pos, noisy_yaw, dθ_meas, dr_meas = slam.simulate_odometry_se2(
                positions, yaws_rad, σd, np.radians(σr),
                np.random.default_rng(SEED),
            )
            odom_mean = float(np.linalg.norm(noisy_pos - positions, axis=1).mean())

            # Patch module constants that depend on σ (used by the solver
            # and by the heading gate).
            slam.SIGMA_DRIVE_MM = σd
            slam.SIGMA_ROT_DEG  = σr
            slam.W_ODOM_POS     = 1.0 / σd
            slam.W_ODOM_ROT     = 1.0 / np.radians(σr)

            for hg_k, wt, dd in lc_combos:
                slam.LC_HEADING_GATE_K   = hg_k
                slam.LC_WEIGHT_THRESHOLD = wt
                slam.LC_DEDUP_BUCKET     = dd

                closures = _silently(
                    slam.extract_loop_closures, history, N,
                    noisy_pos=noisy_pos, noisy_yaw=noisy_yaw,
                    drive_mm_per_step=drive_mm_per_step,
                )
                relaxed, _relaxed_yaw, pruned = _silently(
                    slam.solve_pose_graph_se2,
                    noisy_pos, noisy_yaw, dθ_meas, dr_meas, closures,
                )
                aligned, _ = umeyama_align(relaxed, positions, with_scale=True)
                err = np.linalg.norm(aligned - positions, axis=1)

                tp = sum(1 for s, t in pruned
                         if np.hypot(*(positions[s] - positions[t])) < TP_RADIUS_MM)
                fp = len(pruned) - tp
                prec = tp / max(1, len(pruned))

                print(f"{wl:>3} {beta:>4.1f}  "
                      f"{σd:>4.1f} {σr:>4.1f}  "
                      f"{odom_mean:>5.0f}  "
                      f"{hg_k:>4.1f} {wt:>4.2f} {dd:>3}  "
                      f"{len(pruned):>4} {tp:>3} {fp:>3} {prec:>5.2f}  "
                      f"{err[-1]:>6.0f} {err.mean():>5.0f}")


if __name__ == "__main__":
    main()
