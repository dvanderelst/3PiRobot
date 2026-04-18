#!/usr/bin/env python3
"""
SCRIPT_PoseGraphSLAM_SE2.py

Full SE(2) pose-graph SLAM with realistic body-frame odometry noise.

Differences vs. SCRIPT_PoseGraphSLAM.py:
  - Odometry noise is applied in the body frame: per-step rotation and drive
    distance each get Gaussian noise, then integrated through the estimated
    heading. Yaw error therefore propagates into position drift — the
    characteristic spiral/hook pattern of real robot odometry.
  - Pose graph nodes are (x, y, θ). Odometry edges constrain the full
    body-frame relative pose between consecutive nodes. Loop closures
    constrain position only (we don't infer relative heading from the PF).
  - Solved with Gauss-Newton on a sparse analytic Jacobian.

Output: SpatialInfo/<run>/posegraph_slam_se2.png
"""

import argparse
import dataclasses
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.sparse
import scipy.sparse.linalg

from Library.SlamCore import load_run, collect_data, build_windows, run_pf, umeyama_align


# ── Config ────────────────────────────────────────────────────────────────────
RUN_DIR             = "PolicyTraining/burst_h03"
SESSION_NAME        = "sessionB05"
MAX_STEPS           = 500
WINDOW_LEN          = 5
SEED                = 1

# Realistic body-frame odometry noise
SIGMA_DRIVE_MM      = 10.0        # per-step drive-distance noise (σ)
SIGMA_ROT_DEG       = 5.0        # per-step rotation noise (σ)

# Particle filter
PF_BETA             = 30.0
MIN_LC_GAP          = 30
LC_WEIGHT_THRESHOLD = 0.35
LC_DEDUP_BUCKET     = 8
LC_ODOM_GATE_K      = 3.0        # reject LC if d_odom(s,t) > K · expected_drift
DRIVE_MM_PER_STEP   = 50.0       # used by the drift-gate bound

# Pose graph: edge weights (1/σ)
W_ODOM_POS          = 1.0 / SIGMA_DRIVE_MM                # per-coord
W_ODOM_ROT          = 1.0 / np.radians(SIGMA_ROT_DEG)     # rad⁻¹
W_LC_POS            = 1.0 / 50.0                          # σ_LC ≈ 50 mm
W_LC_ROT            = 1.0 / np.radians(15.0)              # σ_LC_θ ≈ 15° (same-direction revisits)
W_SMOOTH_POS        = 0.02
W_SMOOTH_ROT        = 0.05
ANCHOR_WEIGHT       = 1e3

# Gauss-Newton with step damping
GN_MAX_ITERS        = 60
GN_TOL              = 1e-3
GN_MAX_STEP_XY_MM   = 200.0     # cap per-node position update per iteration
GN_MAX_STEP_ROT_RAD = 0.3       # cap per-node rotation update per iteration

# Huber robust loss on loop-closure residuals (outlier rejection via IRLS).
# δ is in σ-units of the LC residual (weighted magnitude). Residuals with
# weighted 2D magnitude > δ get progressively demoted in subsequent GN iters.
HUBER_DELTA_LC      = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# Realistic odometry
# ══════════════════════════════════════════════════════════════════════════════

def wrap_rad(x: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [−π, π]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def reconstruct_body_motion(positions, yaws_rad):
    """From world-frame trajectory, recover per-step body motion (dθ, dr)."""
    N   = len(positions)
    dθ  = wrap_rad(np.diff(yaws_rad))
    dr  = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return dθ, dr


def simulate_odometry_se2(positions, yaws_rad, σ_drive, σ_rot_rad, rng):
    """
    Apply body-frame noise to (dθ, dr), integrate through estimated heading.

    Returns
    -------
    noisy_pos  (N, 2)
    noisy_yaw  (N,) radians
    dθ_meas, dr_meas  — the noisy body-frame measurements (the edges the
                         pose graph will try to satisfy).
    """
    N = len(positions)
    dθ_true, dr_true = reconstruct_body_motion(positions, yaws_rad)

    dθ_meas = dθ_true + rng.normal(0.0, σ_rot_rad,  N - 1)
    dr_meas = dr_true + rng.normal(0.0, σ_drive,    N - 1)

    noisy_pos = np.zeros_like(positions, dtype=np.float64)
    noisy_yaw = np.zeros(N, dtype=np.float64)
    noisy_pos[0] = positions[0]
    noisy_yaw[0] = yaws_rad[0]
    for i in range(1, N):
        noisy_yaw[i] = noisy_yaw[i - 1] + dθ_meas[i - 1]
        h = noisy_yaw[i]                                        # after-rotation heading
        noisy_pos[i] = noisy_pos[i - 1] + dr_meas[i - 1] * np.array([np.cos(h), np.sin(h)])
    noisy_yaw = wrap_rad(noisy_yaw)
    return noisy_pos, noisy_yaw, dθ_meas, dr_meas


# ══════════════════════════════════════════════════════════════════════════════
# Loop-closure extraction (PF weight threshold only — no odom gate here)
# ══════════════════════════════════════════════════════════════════════════════

def _expected_drift_mm(n_steps):
    """
    Expected position drift between two poses n_steps apart under the
    current (realistic) odometry noise. Drive-noise component scales as
    √n, rotation-noise component scales as n (dominant).
    """
    drive_component = SIGMA_DRIVE_MM * np.sqrt(max(n_steps, 1))
    rot_component   = n_steps * np.radians(SIGMA_ROT_DEG) * DRIVE_MM_PER_STEP
    return drive_component + rot_component


def extract_loop_closures(history, N, noisy_pos=None):
    seen, closures = set(), []
    rejected_by_gate = 0
    for t, (particles, weights) in enumerate(history):
        if t <= MIN_LC_GAP:
            continue
        bins = np.zeros(N, dtype=np.float64)
        np.add.at(bins, particles, weights)
        lo = max(0, t - MIN_LC_GAP)
        hi = min(N, t + MIN_LC_GAP + 1)
        bins[lo:hi] = 0.0
        if bins.max() < LC_WEIGHT_THRESHOLD:
            continue
        s = int(bins.argmax())

        if noisy_pos is not None:
            d_odom   = float(np.hypot(*(noisy_pos[s] - noisy_pos[t])))
            max_drift = LC_ODOM_GATE_K * _expected_drift_mm(abs(t - s))
            if d_odom > max_drift:
                rejected_by_gate += 1
                continue

        key = (s // LC_DEDUP_BUCKET, t // LC_DEDUP_BUCKET)
        if key in seen:
            continue
        seen.add(key)
        closures.append((s, t))
    if rejected_by_gate:
        print(f"  {rejected_by_gate} candidate LCs rejected by drift gate")
    return closures


# ══════════════════════════════════════════════════════════════════════════════
# SE(2) residuals and Jacobian
# ══════════════════════════════════════════════════════════════════════════════

def build_residuals_and_jacobian(x, y, θ, dθ_m, dr_m, loop_closures, anchor):
    """
    Compute stacked residual vector r and sparse Jacobian J for current poses.

    Pose vector v is [x_0, y_0, θ_0, x_1, y_1, θ_1, ...] (length 3N).
    Residuals, in order:
        anchor           (3)
        odometry × M     (3 each = 3M)
        loop closures    (2 each = 2L)
        smoothness       (3 each = 3(N−2))
    """
    N = len(x)
    M = N - 1
    L = len(loop_closures)
    S = max(0, N - 2)
    n_rows = 3 + 3 * M + 3 * L + 3 * S   # LC: 2 pos + 1 heading per closure
    n_vars = 3 * N

    r = np.zeros(n_rows)
    rj, cj, dj = [], [], []

    def add(row, col, val):
        rj.append(row); cj.append(col); dj.append(val)

    row = 0

    # ── Anchor ────────────────────────────────────────────────────────────────
    r[row + 0] = ANCHOR_WEIGHT * (x[0] - anchor[0])
    r[row + 1] = ANCHOR_WEIGHT * (y[0] - anchor[1])
    r[row + 2] = ANCHOR_WEIGHT * wrap_rad(θ[0] - anchor[2])
    add(row + 0, 0, ANCHOR_WEIGHT)
    add(row + 1, 1, ANCHOR_WEIGHT)
    add(row + 2, 2, ANCHOR_WEIGHT)
    row += 3

    # ── Odometry ──────────────────────────────────────────────────────────────
    for i in range(M):
        c, s = np.cos(θ[i + 1]), np.sin(θ[i + 1])
        r[row + 0] = W_ODOM_ROT * wrap_rad(θ[i + 1] - θ[i] - dθ_m[i])
        r[row + 1] = W_ODOM_POS * ((x[i + 1] - x[i]) - dr_m[i] * c)
        r[row + 2] = W_ODOM_POS * ((y[i + 1] - y[i]) - dr_m[i] * s)

        # θ residual
        add(row + 0, 3 * i + 2,        -W_ODOM_ROT)
        add(row + 0, 3 * (i + 1) + 2,  +W_ODOM_ROT)

        # x residual
        add(row + 1, 3 * i + 0,        -W_ODOM_POS)
        add(row + 1, 3 * (i + 1) + 0,  +W_ODOM_POS)
        add(row + 1, 3 * (i + 1) + 2,  +W_ODOM_POS * dr_m[i] * s)   # ∂(−dr·cos θ)/∂θ = dr·sin θ

        # y residual
        add(row + 2, 3 * i + 1,        -W_ODOM_POS)
        add(row + 2, 3 * (i + 1) + 1,  +W_ODOM_POS)
        add(row + 2, 3 * (i + 1) + 2,  -W_ODOM_POS * dr_m[i] * c)   # ∂(−dr·sin θ)/∂θ = −dr·cos θ

        row += 3

    # ── Loop closures (position + heading; assume same-direction revisits) ────
    for s_idx, t_idx in loop_closures:
        r[row + 0] = W_LC_POS * (x[t_idx] - x[s_idx])
        r[row + 1] = W_LC_POS * (y[t_idx] - y[s_idx])
        r[row + 2] = W_LC_ROT * wrap_rad(θ[t_idx] - θ[s_idx])
        add(row + 0, 3 * s_idx + 0, -W_LC_POS)
        add(row + 0, 3 * t_idx + 0, +W_LC_POS)
        add(row + 1, 3 * s_idx + 1, -W_LC_POS)
        add(row + 1, 3 * t_idx + 1, +W_LC_POS)
        add(row + 2, 3 * s_idx + 2, -W_LC_ROT)
        add(row + 2, 3 * t_idx + 2, +W_LC_ROT)
        row += 3

    # ── Smoothness on (x, y, θ) ───────────────────────────────────────────────
    for i in range(1, N - 1):
        for comp, w, val in [
            (0, W_SMOOTH_POS, x[i - 1] - 2 * x[i] + x[i + 1]),
            (1, W_SMOOTH_POS, y[i - 1] - 2 * y[i] + y[i + 1]),
            (2, W_SMOOTH_ROT, wrap_rad(θ[i - 1] - 2 * θ[i] + θ[i + 1])),
        ]:
            r[row] = w * val
            add(row, 3 * (i - 1) + comp, +w)
            add(row, 3 *  i      + comp, -2 * w)
            add(row, 3 * (i + 1) + comp, +w)
            row += 1

    J = scipy.sparse.coo_matrix((dj, (rj, cj)), shape=(n_rows, n_vars)).tocsr()
    lc_row_start = 3 + 3 * M                 # first LC residual row
    lc_row_end   = lc_row_start + 3 * L      # first post-LC row  (3 rows per LC)
    return r, J, lc_row_start, lc_row_end


def _apply_huber_reweighting(r, J, lc_row_start, lc_row_end, δ):
    """
    IRLS reweighting: for each LC (3 consecutive rows = Δx, Δy, Δθ), if the
    2D *position* residual magnitude exceeds δ, scale all three rows by √w
    where w = δ / |r_lc_pos|. Heading is demoted alongside position so a
    position-outlier LC doesn't retain rotational pull.
    """
    n_demoted, weight_sum = 0, 0.0
    r = r.copy()
    J = J.tolil(copy=True)
    for k in range(lc_row_start, lc_row_end, 3):
        mag = float(np.hypot(r[k], r[k + 1]))
        if mag <= δ:
            continue
        w = δ / max(mag, 1e-12)
        sqw = np.sqrt(w)
        for off in (0, 1, 2):
            r[k + off] *= sqw
            J[k + off] *= sqw
        n_demoted += 1
        weight_sum += w
    avg_w = (weight_sum / n_demoted) if n_demoted else 1.0
    return r, J.tocsr(), n_demoted, avg_w


def solve_pose_graph_se2(noisy_pos, noisy_yaw, dθ_m, dr_m, loop_closures):
    N = len(noisy_pos)
    x = noisy_pos[:, 0].astype(np.float64).copy()
    y = noisy_pos[:, 1].astype(np.float64).copy()
    θ = noisy_yaw.astype(np.float64).copy()
    anchor = (float(x[0]), float(y[0]), float(θ[0]))

    print(f"  Gauss-Newton on {3 * N} variables, "
          f"{N - 1} odom + {len(loop_closures)} LC + {N - 2} smooth edges")
    prev_r = np.inf
    for it in range(GN_MAX_ITERS):
        r, J, lc_lo, lc_hi = build_residuals_and_jacobian(
            x, y, θ, dθ_m, dr_m, loop_closures, anchor,
        )
        total_r = float(np.linalg.norm(r))

        # Huber IRLS on LC rows
        r, J, n_dem, avg_w = _apply_huber_reweighting(r, J, lc_lo, lc_hi, HUBER_DELTA_LC)

        dx, *_ = scipy.sparse.linalg.lsqr(
            J, -r, atol=1e-10, btol=1e-10, iter_lim=3000,
        )
        step = dx.reshape(N, 3)

        # Damp: scale step so no single node exceeds position / rotation caps
        max_xy  = float(np.max(np.hypot(step[:, 0], step[:, 1])))
        max_rot = float(np.max(np.abs(step[:, 2])))
        scale   = min(1.0,
                      GN_MAX_STEP_XY_MM  / max(max_xy,  1e-9),
                      GN_MAX_STEP_ROT_RAD / max(max_rot, 1e-9))
        step *= scale

        x += step[:, 0]
        y += step[:, 1]
        θ = wrap_rad(θ + step[:, 2])

        norm_dx = float(np.linalg.norm(step))
        print(f"    it {it:2d}: ||r||={total_r:>10.2f}  ||step||={norm_dx:>8.2f}  "
              f"scale={scale:.3f}  huber_demoted={n_dem}")
        if abs(prev_r - total_r) < GN_TOL:
            break
        prev_r = total_r

    return np.stack([x, y], axis=1), θ


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(true_pos, noisy_pos, aligned_pos, loop_closures,
                 walls, run_name, output_dir, errors):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"SE(2) pose-graph SLAM — {run_name}  "
        f"({len(loop_closures)} loop closures, "
        f"σ_drive={SIGMA_DRIVE_MM:.1f} mm, σ_rot={SIGMA_ROT_DEG:.1f}°)",
        fontsize=11,
    )

    def _walls(ax):
        if len(walls):
            ax.scatter(walls[:, 0], walls[:, 1], s=0.3, c="#cccccc", linewidths=0)

    def _fmt(ax):
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    ax = axes[0, 0]; _walls(ax)
    ax.plot(true_pos[:, 0], true_pos[:, 1], "-", color="#1f77b4", linewidth=1.0)
    ax.set_title("True trajectory", fontsize=9); _fmt(ax)

    ax = axes[0, 1]; _walls(ax)
    ax.plot(noisy_pos[:, 0], noisy_pos[:, 1], "--", color="#d62728", linewidth=1.0)
    ax.set_title(
        f"Body-frame odometry  (σ_drive={SIGMA_DRIVE_MM:.1f} mm, σ_rot={SIGMA_ROT_DEG:.1f}°)",
        fontsize=9,
    ); _fmt(ax)

    ax = axes[0, 2]; _walls(ax)
    ax.plot(aligned_pos[:, 0], aligned_pos[:, 1], "-", color="#2ca02c", linewidth=1.0)
    for s, t in loop_closures:
        ax.plot([aligned_pos[s, 0], aligned_pos[t, 0]],
                [aligned_pos[s, 1], aligned_pos[t, 1]],
                color="#ff7f0e", linewidth=0.4, alpha=0.4, zorder=2)
    ax.set_title("Aligned SE(2) relaxed map (similarity-aligned to true)",
                 fontsize=9); _fmt(ax)

    ax = axes[1, 0]; _walls(ax)
    ax.plot(true_pos[:, 0],    true_pos[:, 1],    "-",  color="#1f77b4",
            linewidth=1.0, label="true")
    ax.plot(noisy_pos[:, 0],   noisy_pos[:, 1],   "--", color="#d62728",
            linewidth=0.9, alpha=0.8, label="odometry")
    ax.plot(aligned_pos[:, 0], aligned_pos[:, 1], "-",  color="#2ca02c",
            linewidth=1.0, alpha=0.9, label="relaxed (aligned)")
    ax.legend(fontsize=8, loc="best")
    ax.set_title("Overlay", fontsize=9); _fmt(ax)

    ax = axes[1, 1]; _walls(ax)
    ax.plot(true_pos[:, 0], true_pos[:, 1], "-",
            color="#bbbbbb", linewidth=0.7, alpha=0.8)
    for s, t in loop_closures:
        d = float(np.hypot(true_pos[s, 0] - true_pos[t, 0],
                           true_pos[s, 1] - true_pos[t, 1]))
        color = "#00aa44" if d < 300 else "#cc2222"
        ax.plot([true_pos[s, 0], true_pos[t, 0]],
                [true_pos[s, 1], true_pos[t, 1]],
                color=color, linewidth=0.7, alpha=0.5, zorder=2)
    ax.legend(handles=[
        mlines.Line2D([], [], color="#00aa44", lw=1.5, label="LC: truly close (<300 mm)"),
        mlines.Line2D([], [], color="#cc2222", lw=1.5, label="LC: falsely paired"),
    ], fontsize=7, loc="best")
    ax.set_title("Loop closures on true map", fontsize=9); _fmt(ax)

    ax = axes[1, 2]
    ax.plot(errors["odom"],    "--", color="#d62728", label="odometry drift")
    ax.plot(errors["relaxed"], ":",  color="#888888", label="raw relaxed error")
    ax.plot(errors["aligned"], "-",  color="#2ca02c", label="aligned relaxed error")
    ax.set_xlabel("Step"); ax.set_ylabel("Position error vs. true (mm)")
    ax.set_title("Drift over time", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "posegraph_slam_se2.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir",    nargs="?",  default=RUN_DIR)
    parser.add_argument("--session",              default=SESSION_NAME)
    parser.add_argument("--beta",     type=float, default=PF_BETA)
    parser.add_argument("--max_steps", type=int,  default=MAX_STEPS)
    parser.add_argument("--seed",     type=int,   default=SEED)
    args = parser.parse_args()

    rng        = np.random.default_rng(args.seed)
    run_name   = os.path.basename(args.run_dir.rstrip("/"))
    output_dir = os.path.join("SpatialInfo", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRun: {run_name}")
    mod, cfg, policy, is_burst = load_run(args.run_dir)
    print(f"  Type: {'burst' if is_burst else 'sonar'}")

    cfg_one = dataclasses.replace(cfg, train_session_names=[args.session])

    print(f"\nCollecting 1 long trajectory in {args.session}...")
    positions, yaws_deg, meas_seq, traj_ids, simulators = collect_data(
        mod, cfg_one, policy, n_traj=1, max_steps=args.max_steps, rng=rng,
    )
    N = len(positions)
    positions = positions.astype(np.float64)
    yaws_rad  = np.radians(yaws_deg.astype(np.float64))
    print(f"  Trajectory length: {N} steps")

    print(f"\nSimulating realistic body-frame odometry "
          f"(σ_drive={SIGMA_DRIVE_MM} mm, σ_rot={SIGMA_ROT_DEG}°)...")
    noisy_pos, noisy_yaw, dθ_meas, dr_meas = simulate_odometry_se2(
        positions, yaws_rad, SIGMA_DRIVE_MM, np.radians(SIGMA_ROT_DEG), rng,
    )

    feats = build_windows(meas_seq, np.zeros(N, dtype=np.int32), WINDOW_LEN)

    print(f"\nRunning particle filter (β={args.beta:g})...")
    history, _ = run_pf(feats, rng, args.beta)

    print("\nExtracting loop closures...")
    loop_closures = extract_loop_closures(history, N, noisy_pos=noisy_pos)
    n_tp = sum(1 for s, t in loop_closures
               if np.hypot(*(positions[s] - positions[t])) < 300)
    n_fp = len(loop_closures) - n_tp
    prec = n_tp / max(1, len(loop_closures))
    print(f"  Found {len(loop_closures)} LCs  "
          f"(TP {n_tp}, FP {n_fp}, precision {prec:.3f})")

    print("\nSolving SE(2) pose graph (Gauss-Newton)...")
    relaxed_pos, relaxed_yaw = solve_pose_graph_se2(
        noisy_pos, noisy_yaw, dθ_meas, dr_meas, loop_closures,
    )

    # Align relaxed map to true via similarity transform (the SLAM map is
    # only recoverable up to rotation / translation / uniform scale).
    aligned_pos, align_params = umeyama_align(relaxed_pos, positions, with_scale=True)
    print(f"\n  Alignment: scale={align_params['scale']:.4f}  "
          f"translation=({align_params['t'][0]:.0f}, {align_params['t'][1]:.0f})")

    errors = {
        "odom":    np.linalg.norm(noisy_pos   - positions, axis=1),
        "relaxed": np.linalg.norm(relaxed_pos - positions, axis=1),
        "aligned": np.linalg.norm(aligned_pos - positions, axis=1),
    }
    print(f"  Final odom drift      : {errors['odom'][-1]:.0f} mm")
    print(f"  Final raw-relaxed err : {errors['relaxed'][-1]:.0f} mm")
    print(f"  Final aligned err     : {errors['aligned'][-1]:.0f} mm")
    print(f"  Mean  odom drift      : {errors['odom'].mean():.0f} mm")
    print(f"  Mean  raw-relaxed err : {errors['relaxed'].mean():.0f} mm")
    print(f"  Mean  aligned err     : {errors['aligned'].mean():.0f} mm")

    print("\nPlotting...")
    walls = next(iter(simulators.values())).arena.walls
    plot_results(positions, noisy_pos, aligned_pos, loop_closures,
                 walls, run_name, output_dir, errors)

    print("\nDone.")


if __name__ == "__main__":
    main()
