#!/usr/bin/env python3
"""
SCRIPT_PoseGraphSLAM_SE2_Envelope.py

SE(2) pose-graph SLAM on real-robot data using raw sonar envelopes as the
place-recognition signal instead of the derived (dist_mm, IID_dB) scalars.

Differences vs. SCRIPT_PoseGraphSLAM_SE2.py:
  - Data source is always real (no sim mode).
  - Each step's sonar measurement is the left+right ear envelopes (200 bins),
    restricted to the echo-relevant distance range (ECHO_BIN_LO:ECHO_BIN_HI).
  - Each channel is L1-normalised to a probability distribution and then
    cumsum'd to a CDF. The PF measures similarity via the L1 norm of CDF
    differences — this equals the 1-D Earth Mover's Distance (Wasserstein-1)
    up to a factor of bin_width, which β absorbs.
  - The Gauss-Newton pose-graph solver, odometry model, loop-closure gating,
    and Umeyama alignment are unchanged.

Output: SpatialInfo/<run>/posegraph_slam_se2_envelope.png
"""

import glob
import json
import os

import dill
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.sparse
import scipy.sparse.linalg

from Library.SlamCore import build_windows, umeyama_align

# ── Data / odometry ───────────────────────────────────────────────────────────
REAL_RUN_DIR    = "PolicyRuns/session_test2_h01_arena1_01"
#   "commanded" : dead-reckon from the robot's own motion commands.
#   "synthetic" : Gaussian noise on ground truth (sanity check only).
ODOM_SOURCE     = "synthetic"     # "commanded" | "synthetic"

# Body-frame odometry noise (used when ODOM_SOURCE="synthetic")
SIGMA_DRIVE_MM  = 5.0
SIGMA_ROT_DEG   = 2.0

# ── Envelope feature extraction ───────────────────────────────────────────────
# Bin range to keep from the 200-sample envelope (17.1 mm/bin, 0–3408 mm).
# Bins 0–4  : dominated by emission bleed.
# Bins 5–130: 85 mm – 2.22 m, covers all arena echoes.
ECHO_BIN_LO     = 5
ECHO_BIN_HI     = 130
N_ECHO_BINS     = ECHO_BIN_HI - ECHO_BIN_LO   # 125 bins per channel

# ── PF / SLAM parameters ──────────────────────────────────────────────────────
WINDOW_LEN          = 4       # smaller than default: envelopes are richer per step
SEED                = 1

# PF_BETA: likelihood = exp(−β · L1(CDF_t, CDF_s)).
# Empirical L1 distances for this run (window=4, 1000 dims):
#   same-dir revisits (<30° Δheading, <200mm) : mean ≈  8.7  → exp(−0.15·8.7)  ≈ 0.27
#   random pairs                               : mean ≈ 15.8  → exp(−0.15·15.8) ≈ 0.094
#   per-step likelihood ratio ≈ 2.9×; absolute likelihoods safely above the reset floor.
PF_BETA             = 0.15

MIN_LC_GAP          = 10
LC_WEIGHT_THRESHOLD = 0.25
LC_DEDUP_BUCKET     = 3
LC_ODOM_GATE_K      = 1.5
LC_HEADING_GATE_K   = 1.0

# Pose-graph edge weights
W_ODOM_POS      = 1.0 / SIGMA_DRIVE_MM
W_ODOM_ROT      = 1.0 / np.radians(SIGMA_ROT_DEG)
W_LC_POS        = 1.0 / 50.0
W_LC_ROT        = 1.0 / np.radians(15.0)
W_SMOOTH_POS    = 0.02
W_SMOOTH_ROT    = 0.05
ANCHOR_WEIGHT   = 1e3

# Gauss-Newton
GN_MAX_ITERS        = 60
GN_TOL              = 1e-3
GN_MAX_STEP_XY_MM   = 200.0
GN_MAX_STEP_ROT_RAD = 0.3
HUBER_DELTA_LC      = 5.0
LC_PRUNE_DIST_MM    = 150.0
GN_CLEAN_ITERS      = 20


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def wrap_rad(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def reconstruct_body_motion(positions, yaws_rad):
    dθ = wrap_rad(np.diff(yaws_rad))
    dr = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return dθ, dr


def simulate_odometry_se2(positions, yaws_rad, σ_drive, σ_rot_rad, rng):
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
        h = noisy_yaw[i]
        noisy_pos[i] = noisy_pos[i - 1] + dr_meas[i - 1] * np.array([np.cos(h), np.sin(h)])
    noisy_yaw = wrap_rad(noisy_yaw)
    return noisy_pos, noisy_yaw, dθ_meas, dr_meas


def odom_from_commanded(positions, yaws_rad, commanded):
    N = len(positions)
    dθ = np.asarray(commanded["dθ_rad"], dtype=np.float64)
    dr = np.asarray(commanded["dr_mm"],  dtype=np.float64)
    assert len(dθ) == N - 1 and len(dr) == N - 1
    noisy_pos = np.zeros_like(positions, dtype=np.float64)
    noisy_yaw = np.zeros(N, dtype=np.float64)
    noisy_pos[0] = positions[0]
    noisy_yaw[0] = yaws_rad[0]
    for k in range(1, N):
        noisy_yaw[k] = noisy_yaw[k - 1] + dθ[k - 1]
        h = noisy_yaw[k]
        noisy_pos[k] = noisy_pos[k - 1] + dr[k - 1] * np.array([np.cos(h), np.sin(h)])
    noisy_yaw = wrap_rad(noisy_yaw)
    return noisy_pos, noisy_yaw, dθ, dr


# ══════════════════════════════════════════════════════════════════════════════
# Envelope → CDF feature
# ══════════════════════════════════════════════════════════════════════════════

def envelope_to_cdf_feature(sonar_data: np.ndarray) -> np.ndarray:
    """
    Convert a (200, 3) sonar_data array into a (2 * N_ECHO_BINS,) CDF feature.

    Steps:
      1. Slice bins [ECHO_BIN_LO:ECHO_BIN_HI] from channels 0 (left) and 1 (right).
      2. Clip negative values to 0 (baseline noise can go slightly negative after
         DC removal in some processing pipelines).
      3. L1-normalise each channel to a probability distribution.
      4. Cumsum → CDF; values in [0, 1].
      5. Concatenate left and right CDFs.

    The L1 norm between two such features equals the 1-D Earth Mover's Distance
    (Wasserstein-1) between the two echo-energy distributions, multiplied by the
    bin count (bin_width is absorbed into the β parameter).
    """
    env = sonar_data[ECHO_BIN_LO:ECHO_BIN_HI, :2].astype(np.float32)
    env = np.clip(env, 0.0, None)
    for ch in range(2):
        total = env[:, ch].sum()
        if total > 1e-6:
            env[:, ch] /= total
        else:
            env[:, ch] = 1.0 / N_ECHO_BINS     # flat if no signal
    cdf = np.cumsum(env, axis=0)                # shape (N_ECHO_BINS, 2)
    return cdf.flatten()                        # shape (2 * N_ECHO_BINS,)


# ══════════════════════════════════════════════════════════════════════════════
# Real-data ingestion (sonar format)
# ══════════════════════════════════════════════════════════════════════════════

def _interpolate_gaps(arr, valid):
    out = arr.astype(np.float64).copy()
    if valid.all():
        return out
    idx = np.arange(len(out))
    if not valid.any():
        raise ValueError("no valid samples to interpolate from")
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def _unwrap_and_interp_yaw_deg(yaws_deg, valid):
    y = np.asarray(yaws_deg, dtype=np.float64).copy()
    if not valid.any():
        raise ValueError("no valid yaw samples")
    unwrapped = np.degrees(np.unwrap(np.radians(y[valid])))
    y[valid] = unwrapped
    y = _interpolate_gaps(y, valid)
    return ((y + 180.0) % 360.0) - 180.0


def _arena_rect_walls(bounds_mm, spacing_mm=20.0):
    x0, x1 = bounds_mm["min_x"], bounds_mm["max_x"]
    y0, y1 = bounds_mm["min_y"], bounds_mm["max_y"]
    xs_h = np.arange(x0, x1 + spacing_mm, spacing_mm)
    ys_v = np.arange(y0, y1 + spacing_mm, spacing_mm)
    top    = np.column_stack([xs_h, np.full_like(xs_h, y1)])
    bottom = np.column_stack([xs_h, np.full_like(xs_h, y0)])
    left   = np.column_stack([np.full_like(ys_v, x0), ys_v])
    right  = np.column_stack([np.full_like(ys_v, x1), ys_v])
    return np.vstack([top, bottom, left, right]).astype(np.float32)


def ingest_real_envelope(run_dir: str):
    """
    Load per-step data from a sonar-format PolicyRuns/ directory.
    Returns the same tuple as ingest_real in SE2 except meas_seq is now the
    CDF envelope feature (shape N × 2*N_ECHO_BINS) rather than (dist, IID, r1, r2).

    Only the legacy sonar schema is supported (single sonar_package per step).
    """
    files = sorted(glob.glob(os.path.join(run_dir, "data*.dill")))
    if not files:
        raise FileNotFoundError(f"No data*.dill files in {run_dir}")
    print(f"  Loading {len(files)} .dill steps from {run_dir}")

    with open(files[0], "rb") as f:
        first_keys = dill.load(f)["data"].keys()
    if "sonar_packages" in first_keys:
        raise NotImplementedError("Burst format not yet supported; use sonar format.")
    print("  Format: sonar")
    print(f"  Envelope feature: bins {ECHO_BIN_LO}–{ECHO_BIN_HI} "
          f"({ECHO_BIN_LO * 17.1:.0f}–{ECHO_BIN_HI * 17.1:.0f} mm), "
          f"2 channels × {N_ECHO_BINS} bins = {2 * N_ECHO_BINS} dims/step")

    xs, ys, yaws_deg, valid = [], [], [], []
    r1_deg, r2_deg, drive_mm = [], [], []
    cdf_feats = []

    for p in files:
        with open(p, "rb") as f:
            d = dill.load(f)
        pos = d["data"]["position"]
        mot = d["data"]["motion"]
        sp  = d["data"]["sonar_package"]

        x, y, yaw = pos.get("x"), pos.get("y"), pos.get("yaw_deg")
        ok = (x is not None and y is not None
              and yaw is not None and np.isfinite(float(yaw)))
        xs.append(np.nan if x is None else float(x))
        ys.append(np.nan if y is None else float(y))
        yaws_deg.append(np.nan if yaw is None else float(yaw))
        valid.append(ok)

        r1_deg.append(float(mot["rotate1"]))
        r2_deg.append(float(mot["rotate2"]))
        drive_mm.append(float(mot["drive_mm"]))

        cdf_feats.append(envelope_to_cdf_feature(sp["sonar_data"]))

    valid_arr = np.array(valid, dtype=bool)
    n_gaps = int((~valid_arr).sum())
    if n_gaps:
        print(f"  Interpolating {n_gaps}/{len(files)} GT gaps")

    positions = np.column_stack([
        _interpolate_gaps(np.asarray(xs, dtype=np.float64), valid_arr),
        _interpolate_gaps(np.asarray(ys, dtype=np.float64), valid_arr),
    ])
    yaws_rad = np.radians(_unwrap_and_interp_yaw_deg(
        np.asarray(yaws_deg, dtype=np.float64), valid_arr
    ))
    meas_seq = np.array(cdf_feats, dtype=np.float32)

    # Commanded body-frame motion (same as SE2 sonar branch)
    r1 = np.asarray(r1_deg, dtype=np.float64)
    r2 = np.asarray(r2_deg, dtype=np.float64)
    dθ_rad = -np.radians(r2[:-1] + r1[1:])
    dr_mm_ = np.asarray(drive_mm[:-1], dtype=np.float64)
    commanded = {"dθ_rad": dθ_rad, "dr_mm": dr_mm_}

    # Walls
    env_dirs = sorted(p for p in os.listdir(run_dir)
                      if p.startswith("env_") and os.path.isdir(os.path.join(run_dir, p)))
    walls = None
    if env_dirs:
        npz_path = os.path.join(run_dir, env_dirs[0], "arena_walls.npz")
        if os.path.isfile(npz_path):
            data = np.load(npz_path)
            walls = np.column_stack([data["x_mm"], data["y_mm"]]).astype(np.float32)
    if walls is None:
        bounds = None
        if env_dirs:
            meta_path = os.path.join(run_dir, env_dirs[0], "meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    bounds = json.load(f).get("arena_bounds_mm")
        if bounds is None:
            pad = 500.0
            bounds = {"min_x": float(positions[:, 0].min() - pad),
                      "max_x": float(positions[:, 0].max() + pad),
                      "min_y": float(positions[:, 1].min() - pad),
                      "max_y": float(positions[:, 1].max() + pad)}
        walls = _arena_rect_walls(bounds)

    run_name = os.path.basename(run_dir.rstrip("/"))
    return positions, yaws_rad, meas_seq, walls, run_name, commanded


# ══════════════════════════════════════════════════════════════════════════════
# Particle filter with L1 / EMD likelihood
# ══════════════════════════════════════════════════════════════════════════════

# PF motion-model constants (same as SlamCore defaults)
_N_PARTICLES = 1000
_P_ADVANCE   = 0.60
_P_STAY      = 0.25
_P_JUMP      = 0.15


def _propagate(particles, t, rng):
    r = rng.random(len(particles))
    advance = r < _P_ADVANCE
    jump    = r >= (_P_ADVANCE + _P_STAY)
    new = particles.copy()
    new[advance] = np.minimum(particles[advance] + 1, t - 1)
    if jump.any():
        new[jump] = rng.integers(0, t, size=int(jump.sum()))
    return new


def _resample(particles, weights, rng):
    idx = rng.choice(len(particles), size=len(particles), p=weights)
    return particles[idx], np.ones_like(weights) / len(weights)


def run_pf_emd(feats: np.ndarray, rng, beta: float):
    """
    Particle filter identical to SlamCore.run_pf except the likelihood uses
    L1 distance on CDF features (= Earth Mover's Distance) instead of L2.

    ll_i = exp(−β · ||feats[particle_i] − feats[t]||_1)
    """
    N = len(feats)
    particles = np.zeros(_N_PARTICLES, dtype=np.int32)
    weights   = np.ones(_N_PARTICLES,  dtype=np.float64) / _N_PARTICLES

    history = [(particles.copy(), weights.copy())]
    heatmap = np.zeros((N, N), dtype=np.float32)

    for t in range(1, N):
        particles = _propagate(particles, t, rng)

        diff = feats[particles] - feats[t]
        ll   = np.exp(-beta * np.sum(np.abs(diff), axis=1))   # L1, not L2
        weights = weights * ll
        total = weights.sum()
        if total < 1e-20:
            particles = rng.integers(0, t, size=_N_PARTICLES)
            weights   = np.ones(_N_PARTICLES, dtype=np.float64) / _N_PARTICLES
        else:
            weights = weights / total

        ess = 1.0 / float(np.sum(weights ** 2))
        if ess < _N_PARTICLES / 2:
            particles, weights = _resample(particles, weights, rng)

        history.append((particles.copy(), weights.copy()))
        np.add.at(heatmap[:, t], particles, weights.astype(np.float32))

    return history, heatmap


# ══════════════════════════════════════════════════════════════════════════════
# Loop-closure extraction (unchanged from SE2)
# ══════════════════════════════════════════════════════════════════════════════

def _expected_drift_mm(n_steps, drive_mm):
    drive_component = SIGMA_DRIVE_MM * np.sqrt(max(n_steps, 1))
    rot_component   = n_steps * np.radians(SIGMA_ROT_DEG) * drive_mm
    return drive_component + rot_component


def extract_loop_closures(history, N, noisy_pos=None, noisy_yaw=None,
                          drive_mm_per_step=None):
    seen, closures = set(), []
    rejected_by_drift   = 0
    rejected_by_heading = 0
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
            d_odom    = float(np.hypot(*(noisy_pos[s] - noisy_pos[t])))
            max_drift = LC_ODOM_GATE_K * _expected_drift_mm(abs(t - s), drive_mm=drive_mm_per_step)
            if d_odom > max_drift:
                rejected_by_drift += 1
                continue

        if noisy_yaw is not None:
            d_yaw_deg = float(abs(np.degrees(wrap_rad(noisy_yaw[s] - noisy_yaw[t]))))
            max_yaw   = LC_HEADING_GATE_K * SIGMA_ROT_DEG * np.sqrt(max(abs(t - s), 1))
            if d_yaw_deg > max_yaw:
                rejected_by_heading += 1
                continue

        key = (s // LC_DEDUP_BUCKET, t // LC_DEDUP_BUCKET)
        if key in seen:
            continue
        seen.add(key)
        closures.append((s, t))
    if rejected_by_drift:
        print(f"  {rejected_by_drift} candidate LCs rejected by drift gate")
    if rejected_by_heading:
        print(f"  {rejected_by_heading} candidate LCs rejected by heading gate "
              f"(K={LC_HEADING_GATE_K:g}·σ_rot·√|t−s|)")
    return closures


# ══════════════════════════════════════════════════════════════════════════════
# SE(2) pose graph — unchanged from SE2 script
# ══════════════════════════════════════════════════════════════════════════════

def build_residuals_and_jacobian(x, y, θ, dθ_m, dr_m, loop_closures, anchor):
    N = len(x)
    M = N - 1
    L = len(loop_closures)
    S = max(0, N - 2)
    n_rows = 3 + 3 * M + 3 * L + 3 * S
    n_vars = 3 * N

    r = np.zeros(n_rows)
    rj, cj, dj = [], [], []

    def add(row, col, val):
        rj.append(row); cj.append(col); dj.append(val)

    row = 0

    r[row + 0] = ANCHOR_WEIGHT * (x[0] - anchor[0])
    r[row + 1] = ANCHOR_WEIGHT * (y[0] - anchor[1])
    r[row + 2] = ANCHOR_WEIGHT * wrap_rad(θ[0] - anchor[2])
    add(row + 0, 0, ANCHOR_WEIGHT)
    add(row + 1, 1, ANCHOR_WEIGHT)
    add(row + 2, 2, ANCHOR_WEIGHT)
    row += 3

    for i in range(M):
        c, s = np.cos(θ[i + 1]), np.sin(θ[i + 1])
        r[row + 0] = W_ODOM_ROT * wrap_rad(θ[i + 1] - θ[i] - dθ_m[i])
        r[row + 1] = W_ODOM_POS * ((x[i + 1] - x[i]) - dr_m[i] * c)
        r[row + 2] = W_ODOM_POS * ((y[i + 1] - y[i]) - dr_m[i] * s)
        add(row + 0, 3 * i + 2,        -W_ODOM_ROT)
        add(row + 0, 3 * (i + 1) + 2,  +W_ODOM_ROT)
        add(row + 1, 3 * i + 0,        -W_ODOM_POS)
        add(row + 1, 3 * (i + 1) + 0,  +W_ODOM_POS)
        add(row + 1, 3 * (i + 1) + 2,  +W_ODOM_POS * dr_m[i] * s)
        add(row + 2, 3 * i + 1,        -W_ODOM_POS)
        add(row + 2, 3 * (i + 1) + 1,  +W_ODOM_POS)
        add(row + 2, 3 * (i + 1) + 2,  -W_ODOM_POS * dr_m[i] * c)
        row += 3

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
    lc_row_start = 3 + 3 * M
    lc_row_end   = lc_row_start + 3 * L
    return r, J, lc_row_start, lc_row_end


def _apply_huber_reweighting(r, J, lc_row_start, lc_row_end, δ):
    n_demoted, weight_sum = 0, 0.0
    r = r.copy()
    J = J.tolil(copy=True)
    for k in range(lc_row_start, lc_row_end, 3):
        mag = float(np.hypot(r[k], r[k + 1]))
        if mag <= δ:
            continue
        w   = δ / max(mag, 1e-12)
        sqw = np.sqrt(w)
        for off in (0, 1, 2):
            r[k + off] *= sqw
            J[k + off] *= sqw
        n_demoted  += 1
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
        r, J, n_dem, avg_w = _apply_huber_reweighting(r, J, lc_lo, lc_hi, HUBER_DELTA_LC)
        dx, *_ = scipy.sparse.linalg.lsqr(J, -r, atol=1e-10, btol=1e-10, iter_lim=3000)
        step = dx.reshape(N, 3)
        max_xy  = float(np.max(np.hypot(step[:, 0], step[:, 1])))
        max_rot = float(np.max(np.abs(step[:, 2])))
        scale   = min(1.0,
                      GN_MAX_STEP_XY_MM  / max(max_xy,  1e-9),
                      GN_MAX_STEP_ROT_RAD / max(max_rot, 1e-9))
        step *= scale
        x += step[:, 0]
        y += step[:, 1]
        θ  = wrap_rad(θ + step[:, 2])
        norm_dx = float(np.linalg.norm(step))
        print(f"    it {it:2d}: ||r||={total_r:>10.2f}  ||step||={norm_dx:>8.2f}  "
              f"scale={scale:.3f}  huber_demoted={n_dem}")
        if abs(prev_r - total_r) < GN_TOL:
            break
        prev_r = total_r

    r_final, _, lc_lo, _ = build_residuals_and_jacobian(
        x, y, θ, dθ_m, dr_m, loop_closures, anchor,
    )
    pruned = []
    for k, (s_idx, t_idx) in enumerate(loop_closures):
        row = lc_lo + 3 * k
        mag = float(np.hypot(r_final[row] / W_LC_POS, r_final[row + 1] / W_LC_POS))
        if mag <= LC_PRUNE_DIST_MM:
            pruned.append((s_idx, t_idx))
    n_pruned = len(loop_closures) - len(pruned)
    print(f"  Pruned {n_pruned} outlier LCs (>{LC_PRUNE_DIST_MM:.0f} mm residual); "
          f"{len(pruned)} remain — re-solving clean...")
    prev_r = np.inf
    for it in range(GN_CLEAN_ITERS):
        r, J, _, _ = build_residuals_and_jacobian(x, y, θ, dθ_m, dr_m, pruned, anchor)
        total_r = float(np.linalg.norm(r))
        dx, *_ = scipy.sparse.linalg.lsqr(J, -r, atol=1e-10, btol=1e-10, iter_lim=3000)
        step = dx.reshape(N, 3)
        max_xy  = float(np.max(np.hypot(step[:, 0], step[:, 1])))
        max_rot = float(np.max(np.abs(step[:, 2])))
        scale   = min(1.0,
                      GN_MAX_STEP_XY_MM  / max(max_xy,  1e-9),
                      GN_MAX_STEP_ROT_RAD / max(max_rot, 1e-9))
        step *= scale
        x += step[:, 0]
        y += step[:, 1]
        θ  = wrap_rad(θ + step[:, 2])
        norm_dx = float(np.linalg.norm(step))
        print(f"    clean it {it:2d}: ||r||={total_r:>10.2f}  ||step||={norm_dx:>8.2f}  scale={scale:.3f}")
        if abs(prev_r - total_r) < GN_TOL:
            break
        prev_r = total_r

    return np.stack([x, y], axis=1), θ, pruned


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(true_pos, noisy_pos, aligned_pos, loop_closures,
                 walls, run_name, output_dir, errors):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"SE(2) SLAM (envelope+EMD) — {run_name}  "
        f"({len(loop_closures)} loop closures, "
        f"bins {ECHO_BIN_LO}–{ECHO_BIN_HI}, window={WINDOW_LEN}, β={PF_BETA})",
        fontsize=11,
    )

    def _walls(ax):
        if len(walls):
            ax.scatter(walls[:, 0], walls[:, 1], s=0.3, c="#cccccc", linewidths=0)

    def _fmt(ax):
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2)

    ax = axes[0, 0]; _walls(ax)
    ax.plot(true_pos[:, 0], true_pos[:, 1], "-", color="#1f77b4", linewidth=1.0)
    ax.set_title("True trajectory", fontsize=9); _fmt(ax)

    ax = axes[0, 1]; _walls(ax)
    ax.plot(noisy_pos[:, 0], noisy_pos[:, 1], "--", color="#d62728", linewidth=1.0)
    ax.set_title(
        f"Odometry  ({ODOM_SOURCE}, σ_drive={SIGMA_DRIVE_MM:.1f} mm, σ_rot={SIGMA_ROT_DEG:.1f}°)",
        fontsize=9,
    ); _fmt(ax)

    ax = axes[0, 2]; _walls(ax)
    ax.plot(aligned_pos[:, 0], aligned_pos[:, 1], "-", color="#2ca02c", linewidth=1.0)
    for s, t in loop_closures:
        ax.plot([aligned_pos[s, 0], aligned_pos[t, 0]],
                [aligned_pos[s, 1], aligned_pos[t, 1]],
                color="#ff7f0e", linewidth=0.4, alpha=0.4, zorder=2)
    ax.set_title("Aligned SE(2) relaxed map", fontsize=9); _fmt(ax)

    ax = axes[1, 0]; _walls(ax)
    ax.plot(true_pos[:, 0],    true_pos[:, 1],    "-",  color="#1f77b4", linewidth=1.0, label="true")
    ax.plot(noisy_pos[:, 0],   noisy_pos[:, 1],   "--", color="#d62728", linewidth=0.9, alpha=0.8, label="odometry")
    ax.plot(aligned_pos[:, 0], aligned_pos[:, 1], "-",  color="#2ca02c", linewidth=1.0, alpha=0.9, label="relaxed (aligned)")
    ax.legend(fontsize=8, loc="best")
    ax.set_title("Overlay", fontsize=9); _fmt(ax)

    ax = axes[1, 1]; _walls(ax)
    ax.plot(true_pos[:, 0], true_pos[:, 1], "-", color="#bbbbbb", linewidth=0.7, alpha=0.8)
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
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "posegraph_slam_se2_envelope.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(SEED)

    run_dir = REAL_RUN_DIR
    print(f"\nRun: {os.path.basename(run_dir.rstrip('/'))}  (real, envelope features)")
    positions, yaws_rad, meas_seq, walls, run_name, commanded = ingest_real_envelope(run_dir)
    drive_mm_per_step = float(np.mean(commanded["dr_mm"])) if len(commanded["dr_mm"]) else 0.0
    print(f"  Mean commanded drive per step: {drive_mm_per_step:.0f} mm")

    N = len(positions)
    output_dir = os.path.join("SpatialInfo", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Trajectory length: {N} steps")

    # Odometry
    if ODOM_SOURCE == "commanded":
        print("\nUsing commanded-motion odometry (real dead-reckoning)...")
        noisy_pos, noisy_yaw, dθ_meas, dr_meas = odom_from_commanded(
            positions, yaws_rad, commanded,
        )
    else:
        print(f"\nSimulating body-frame odometry "
              f"(σ_drive={SIGMA_DRIVE_MM} mm, σ_rot={SIGMA_ROT_DEG}°)...")
        noisy_pos, noisy_yaw, dθ_meas, dr_meas = simulate_odometry_se2(
            positions, yaws_rad, SIGMA_DRIVE_MM, np.radians(SIGMA_ROT_DEG), rng,
        )

    # PF place recognition with envelope EMD
    feats = build_windows(meas_seq, np.zeros(N, dtype=np.int32), WINDOW_LEN)
    print(f"\nRunning particle filter (β={PF_BETA:g}, L1/EMD, "
          f"feature dim={feats.shape[1]})...")
    history, _ = run_pf_emd(feats, rng, PF_BETA)

    print("\nExtracting loop closures...")
    loop_closures = extract_loop_closures(
        history, N,
        noisy_pos=noisy_pos, noisy_yaw=noisy_yaw,
        drive_mm_per_step=drive_mm_per_step,
    )
    n_tp  = sum(1 for s, t in loop_closures
                if np.hypot(*(positions[s] - positions[t])) < 300)
    n_fp  = len(loop_closures) - n_tp
    prec  = n_tp / max(1, len(loop_closures))
    print(f"  Found {len(loop_closures)} LCs  "
          f"(TP {n_tp}, FP {n_fp}, precision {prec:.3f})")

    print("\nSolving SE(2) pose graph (Gauss-Newton)...")
    relaxed_pos, relaxed_yaw, pruned_closures = solve_pose_graph_se2(
        noisy_pos, noisy_yaw, dθ_meas, dr_meas, loop_closures,
    )
    n_pruned_tp = sum(1 for s, t in pruned_closures
                      if np.hypot(*(positions[s] - positions[t])) < 300)
    print(f"  Final LC set: {len(pruned_closures)} "
          f"(TP {n_pruned_tp}, FP {len(pruned_closures) - n_pruned_tp})")

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
    plot_results(positions, noisy_pos, aligned_pos, pruned_closures,
                 walls, run_name, output_dir, errors)

    print("\nDone.")


if __name__ == "__main__":
    main()
