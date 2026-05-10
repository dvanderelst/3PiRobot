#!/usr/bin/env python3
"""
SCRIPT_MeasureTrackerNoise.py

Empirically characterise the overhead-tracker noise floor for a stationary
robot. Outputs the per-channel std on (x, y, yaw_deg) so we can size
`wait_for_stable_pose`'s tolerances against data instead of guesswork.

Why: `wait_for_stable_pose` declares a settle when the spread of recent
distinct tracker reads is below `yaw_tol_deg` / `pos_tol_mm`. If those
tolerances are tighter than the camera+aruco intrinsic noise, the function
will time out on every settle even though the robot is genuinely
stationary — the bit-equality filter (which guards against premature
settle on stale buffered frames) exposes this floor by forcing each
counted read to be a genuinely-fresh frame.

Workflow:
  1. Place the robot inside the tracker view and DO NOT TOUCH it.
  2. Run this script. It collects ~30 s of reads at 10 Hz, separates
     bit-identical buffered re-serves (from the polling-rate-mismatch
     issue) from genuinely-distinct frames, and reports the std of the
     distinct subset — that's the per-frame noise floor.
  3. Use the reported std to set `wait_for_stable_pose` tolerances at
     ~3× the std (so most stationary settles converge cleanly).

Output: Diagnostics/tracker_noise_<timestamp>/{samples.tsv, results.png}
"""
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from Library import LorexTracker


ROBOT_ID = 1
N_READS  = 300       # ~30 s at 10 Hz polling
POLL_S   = 0.1
OUTPUT_DIR = f"Diagnostics/tracker_noise_{datetime.now().strftime('%Y%m%dT%H%M%S')}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tracker = LorexTracker.LorexTracker()

    print("Place the robot in tracker view, then DO NOT TOUCH it.")
    print(f"Collecting {N_READS} reads at {1/POLL_S:.0f} Hz "
          f"(~{N_READS*POLL_S:.0f} s). Press Enter when ready.")
    input()

    samples: list = []   # list of (x, y, yaw) tuples; (None, None, None) on miss
    for _ in range(N_READS):
        pos = tracker.get_position(ROBOT_ID)
        if pos is None or pos.get("x") is None:
            samples.append((None, None, None))
        else:
            samples.append((float(pos["x"]),
                            float(pos["y"]),
                            float(pos["yaw_deg"])))
        time.sleep(POLL_S)

    # ── Save raw samples ─────────────────────────────────────────────────────
    tsv_path = os.path.join(OUTPUT_DIR, "samples.tsv")
    with open(tsv_path, "w") as f:
        f.write("idx\tx_mm\ty_mm\tyaw_deg\tdistinct\n")
        last = None
        for i, s in enumerate(samples):
            x, y, yaw = s
            if (x, y, yaw) == (None, None, None):
                f.write(f"{i}\t\t\t\t-\n")
                continue
            distinct = "1" if (x, y, yaw) != last else "0"
            f.write(f"{i}\t{x}\t{y}\t{yaw}\t{distinct}\n")
            last = (x, y, yaw)
    print(f"\nLog: {tsv_path}")

    # ── Stats ───────────────────────────────────────────────────────────────
    valid = [s for s in samples if (None, None, None) != s]
    distinct = []
    last = None
    for s in valid:
        if s != last:
            distinct.append(s)
            last = s
    distinct_arr = np.array(distinct, dtype=float) if distinct else np.zeros((0, 3))

    n_total    = len(samples)
    n_valid    = len(valid)
    n_dropped  = n_total - n_valid
    n_distinct = len(distinct)
    n_repeats  = n_valid - n_distinct

    print(f"Reads: {n_total} total | {n_valid} valid | "
          f"{n_distinct} distinct | {n_repeats} bit-identical re-serves "
          f"({100*n_repeats/max(n_valid,1):.0f}%) | "
          f"{n_dropped} marker-not-found")

    if n_distinct < 2:
        print("⚠️  Not enough distinct reads to compute statistics.")
        return

    print(f"\nStationary noise (distinct reads only, n={n_distinct}):")
    std_yaw = std_x = std_y = 0.0
    for col, name in zip(range(3), ["x_mm", "y_mm", "yaw_deg"]):
        v = distinct_arr[:, col]
        mean = v.mean()
        std  = v.std()
        rng  = v.max() - v.min()
        print(f"  {name:7}  mean = {mean:+9.2f}  std = {std:6.3f}  "
              f"range = {rng:6.3f}")
        if name == "yaw_deg": std_yaw = std
        if name == "x_mm":    std_x   = std
        if name == "y_mm":    std_y   = std

    pos_std = float(np.hypot(std_x, std_y))
    print(f"\nSuggested wait_for_stable_pose tolerances (~3×std):")
    print(f"  yaw_tol_deg ≈ {3*std_yaw:.2f}")
    print(f"  pos_tol_mm  ≈ {3*pos_std:.2f}  (combined x/y; "
          f"current default is 8.0)")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, name in zip(range(3), ["x (mm)", "y (mm)", "yaw (°)"]):
        v = distinct_arr[:, col]
        axes[0, col].plot(v, '.', alpha=0.6)
        axes[0, col].axhline(v.mean(), color='red', linestyle='--', alpha=0.5)
        axes[0, col].set_title(f"{name} time series  (mean {v.mean():+.2f})")
        axes[0, col].set_xlabel("distinct read #")
        axes[0, col].grid(alpha=0.3)
        axes[1, col].hist(v - v.mean(), bins=30, alpha=0.7)
        axes[1, col].set_title(f"{name} centered  std={v.std():.3f}")
        axes[1, col].set_xlabel(f"value − mean")
        axes[1, col].grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "results.png")
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
