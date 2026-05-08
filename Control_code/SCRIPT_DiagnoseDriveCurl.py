#!/usr/bin/env python3
"""
SCRIPT_DiagnoseDriveCurl.py

Diagnostic for drive-induced yaw bias ("drive curl"). Three phases, all using
the motion-required `wait_for_stable_pose` so settled reads cannot lock onto
pre-motion lag frames.

  Phase 1 — pure straight drive:
      N consecutive forward drives, no rotation between them. If the drive
      primitive is unbiased the per-step Δyaw is flat. A constant non-zero
      mean Δyaw means the robot curls during drive (the leading hypothesis
      for the −7°/step bias seen in default_Target02_h32_nosigma_run01).

  Phase 2 — CW circle:  alternate rotate(−A°), drive(D mm) for K rotations.
  Phase 3 — CCW circle: alternate rotate(+A°), drive(D mm) for K rotations.
      A bias-free drive primitive traces same-radius circles in both
      directions and the start↔end gap is small. Drive curl in one direction
      makes the circle in that sense tighter than the other.

Output: Diagnostics/drive_curl_<timestamp>/{poses.tsv, results.png}
"""
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from Library import Client
from Library import LorexTracker
from Library.TrackerNav import wait_for_stable_pose


# ── Settings ──────────────────────────────────────────────────────────────────
ROBOT_ID = 1

N_STRAIGHT          = 10
STRAIGHT_DRIVE_MM   = 125.0   # matches Policy.fixed_drive_mm in the run01 trace

N_CIRCLE_STEPS      = 12      # 12 × 30° = 360°
CIRCLE_ROTATION_DEG = 30.0
CIRCLE_DRIVE_MM     = 100.0
# Polygon side = 100 mm at 30° interior turn → inscribed radius ≈ 187 mm.
# Robot needs ≥ 400 mm clear in every direction from its starting pose.

OUTPUT_DIR = f"Diagnostics/drive_curl_{datetime.now().strftime('%Y%m%dT%H%M%S')}"


def settled(tracker, prior=None):
    """Settled tracker read; if `prior` is given, requires observed motion
    before declaring settled (defeats pre-motion stale-frame lock-on)."""
    return wait_for_stable_pose(
        tracker, ROBOT_ID,
        prior_pose=prior,
        strict_motion=False,
        verbose=True,
    )


def run_phase(client, tracker, label, actions, log):
    """Run a list of (kind, value) actions; kind ∈ {'rotate' (deg), 'drive' (mm)}.
    Logs each settled post-step pose. Returns the final pose."""
    pose = settled(tracker)
    if pose is None:
        raise RuntimeError(f"{label}: tracker missed initial pose")
    print(f"  init pose: ({pose[0]:.0f}, {pose[1]:.0f}, {pose[2]:+.1f}°)")
    log.append((label, -1, "init", 0.0, pose[0], pose[1], pose[2]))
    for i, (kind, value) in enumerate(actions):
        prior = pose
        try:
            if kind == "rotate":
                client.step(angle=value)
            elif kind == "drive":
                client.step(distance=value / 1000.0)
            else:
                raise ValueError(kind)
        except RuntimeError as e:
            print(f"  step {i:3d} {kind}={value:+.1f}: aborted: {e}")
            continue
        pose = settled(tracker, prior=prior)
        if pose is None:
            print(f"  step {i:3d} {kind}={value:+.1f}: tracker timed out")
            continue
        delta_yaw = ((pose[2] - prior[2] + 180.0) % 360.0) - 180.0
        log.append((label, i, kind, value, pose[0], pose[1], pose[2]))
        print(f"  step {i:3d} {kind}={value:+6.1f}: "
              f"pose=({pose[0]:7.0f},{pose[1]:7.0f},{pose[2]:+7.1f}°)  "
              f"Δyaw={delta_yaw:+6.2f}°")
    return pose


def fit_circle(xs, ys):
    """Algebraic least-squares circle fit. Returns (xc, yc, r)."""
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = -(xs**2 + ys**2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a_, b_, c_ = sol
    xc, yc = -a_/2, -b_/2
    r = float(np.sqrt(max(a_**2/4 + b_**2/4 - c_, 0.0)))
    return float(xc), float(yc), r


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = Client.Client(robot_number=ROBOT_ID)
    tracker = LorexTracker.LorexTracker()

    log: list = []

    print(f"\n{'='*72}\nPhase 1: pure forward drive (no rotation between drives)")
    print(f"Robot needs ≥ {N_STRAIGHT * STRAIGHT_DRIVE_MM:.0f} mm of clear space "
          f"in the direction it currently faces. Press Enter.")
    input()
    run_phase(client, tracker, "straight",
              [("drive", STRAIGHT_DRIVE_MM)] * N_STRAIGHT, log)

    print(f"\n{'='*72}\nPhase 2: CW circle "
          f"({N_CIRCLE_STEPS} × (rotate {-CIRCLE_ROTATION_DEG:+.0f}°, "
          f"drive {CIRCLE_DRIVE_MM:.0f} mm))")
    print("Reposition robot — needs ≥ 400 mm clear radius. Press Enter.")
    input()
    actions = []
    for _ in range(N_CIRCLE_STEPS):
        actions += [("rotate", -CIRCLE_ROTATION_DEG),
                    ("drive",  CIRCLE_DRIVE_MM)]
    run_phase(client, tracker, "cw", actions, log)

    print(f"\n{'='*72}\nPhase 3: CCW circle "
          f"({N_CIRCLE_STEPS} × (rotate {+CIRCLE_ROTATION_DEG:+.0f}°, "
          f"drive {CIRCLE_DRIVE_MM:.0f} mm))")
    print("Reposition robot — needs ≥ 400 mm clear radius. Press Enter.")
    input()
    actions = []
    for _ in range(N_CIRCLE_STEPS):
        actions += [("rotate", +CIRCLE_ROTATION_DEG),
                    ("drive",  CIRCLE_DRIVE_MM)]
    run_phase(client, tracker, "ccw", actions, log)

    # ── Save log ──────────────────────────────────────────────────────────────
    tsv_path = os.path.join(OUTPUT_DIR, "poses.tsv")
    with open(tsv_path, "w") as f:
        f.write("label\tstep\tkind\tvalue\tx_mm\ty_mm\tyaw_deg\n")
        for row in log:
            f.write("\t".join(str(v) for v in row) + "\n")
    print(f"\nLog: {tsv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    arr    = np.array([r[4:] for r in log], dtype=float)
    labels = np.array([r[0] for r in log])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for label, color in [("straight", "C0"), ("cw", "C1"), ("ccw", "C2")]:
        m = labels == label
        if not m.any():
            continue
        xs, ys = arr[m, 0], arr[m, 1]
        ax.plot(xs, ys, '-o', label=label, alpha=0.7, color=color, markersize=4)
        ax.scatter([xs[0]], [ys[0]], color=color, s=80, zorder=3, marker='s',
                   edgecolor='black', label=f"{label} start")
        if label in ("cw", "ccw") and len(xs) >= 4:
            xc, yc, r = fit_circle(xs, ys)
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(xc + r*np.cos(theta), yc + r*np.sin(theta),
                    color=color, linestyle=":", alpha=0.5,
                    label=f"{label} fit (r={r:.0f}mm)")
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title('Trajectories with circle fits')
    ax.legend(fontsize=8, loc='best'); ax.grid(alpha=0.3)

    ax = axes[1]
    m = labels == "straight"
    if m.sum() >= 2:
        yaws = arr[m, 2]
        deltas = ((np.diff(yaws) + 180.0) % 360.0) - 180.0
        ax.plot(np.arange(1, len(deltas)+1), deltas, 'o-')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axhline(deltas.mean(), color='red', linestyle='--',
                   label=f'mean = {deltas.mean():+.2f}°  (std {deltas.std():.2f}°)')
        ax.set_xlabel('drive step'); ax.set_ylabel('Δ yaw per pure drive (°)')
        ax.set_title(f'Pure-drive yaw drift  ({STRAIGHT_DRIVE_MM:.0f} mm/step, '
                     f'commanded rotation = 0)')
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "results.png")
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Plot: {plot_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("Summary")
    print("="*72)
    m = labels == "straight"
    if m.sum() >= 2:
        yaws   = arr[m, 2]
        deltas = ((np.diff(yaws) + 180.0) % 360.0) - 180.0
        print(f"Pure-drive yaw drift per {STRAIGHT_DRIVE_MM:.0f} mm step "
              f"(commanded rotation = 0):")
        print(f"  mean  = {deltas.mean():+.2f}°")
        print(f"  std   = {deltas.std():.2f}°")
        print(f"  range = {deltas.min():+.2f}° to {deltas.max():+.2f}°")
        if abs(deltas.mean()) > 1.0 and deltas.std() < 3.0:
            print("  → drive curl detected (consistent non-zero bias).")
        elif deltas.std() > 5.0:
            print("  → high variance — investigate tracker or surface.")
        else:
            print("  → drive primitive looks unbiased.")

    for label in ("cw", "ccw"):
        m = labels == label
        if m.sum() < 4:
            continue
        xs, ys = arr[m, 0], arr[m, 1]
        xc, yc, r = fit_circle(xs, ys)
        gap = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
        residuals = np.hypot(xs - xc, ys - yc) - r
        print(f"\n{label.upper()} circle:  fitted radius = {r:.0f} mm  "
              f"(residual std {residuals.std():.0f} mm)")
        print(f"  closure gap (start → end): {gap:.0f} mm")

    # Direct CW vs CCW comparison
    cw_pts  = arr[labels == "cw"][:, :2]
    ccw_pts = arr[labels == "ccw"][:, :2]
    if len(cw_pts) >= 4 and len(ccw_pts) >= 4:
        _, _, r_cw  = fit_circle(cw_pts[:, 0],  cw_pts[:, 1])
        _, _, r_ccw = fit_circle(ccw_pts[:, 0], ccw_pts[:, 1])
        print(f"\nCW radius − CCW radius = {r_cw - r_ccw:+.0f} mm  "
              f"(non-zero → drive bias couples with rotation direction)")


if __name__ == "__main__":
    main()
