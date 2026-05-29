#!/usr/bin/env python3
"""
SCRIPT_PoseGraphSLAM_SE2.py

Full SE(2) pose-graph SLAM with realistic body-frame odometry noise.

Runs on either simulated policy rollouts (DATA_SOURCE="sim") or recorded
real-robot data from PolicyRuns/ (DATA_SOURCE="real"). The SLAM core —
PF place recognition, LC extraction, SE(2) Gauss-Newton, Umeyama alignment,
plotting — is identical in both modes; only the ingestion and the odometry
source differ.

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

import dataclasses
import glob
import json
import os

import dill
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.sparse
import scipy.sparse.linalg

from SlamCore import load_run, collect_data, build_windows, run_pf, umeyama_align


# ── Data source ───────────────────────────────────────────────────────────────
DATA_SOURCE         = "real"      # "sim" | "real"

# Sim-mode inputs (ignored if DATA_SOURCE == "real")
SIM_RUN_DIR         = "PolicyTraining/sonar_h01"
SIM_SESSION_NAME    = "sessionB01"
SIM_MAX_STEPS       = 500

# Real-mode inputs (ignored if DATA_SOURCE == "sim")
REAL_RUN_DIR        = "PolicyRuns/session_test2_h01_arena1_01"
#   "commanded" uses the robot's own motion commands as dead-reckoning — the
#       honest test: residual error between commanded and executed motion is
#       the odometry drift the SLAM must cancel. Real mode only.
#   "synthetic" applies Gaussian noise to ground truth (same as sim mode).
#       Useful as a plumbing sanity check; not a real test of real-data SLAM.
ODOM_SOURCE         = "synthetic"     # "commanded" | "synthetic"

# Feature normalisation for real data (matches the training-config defaults;
# only affects the PF feature space — change only if the deployed policy
# used different caps).
REAL_MAX_DIST_MM    = 2000.0
REAL_MAX_IID_DB     = 12.0
REAL_MAX_R1_DEG     = 90.0
REAL_MAX_R2_DEG     = 90.0

# ── SLAM parameters ───────────────────────────────────────────────────────────
WINDOW_LEN          = 8
SEED                = 1

# Body-frame odometry noise (used by ODOM_SOURCE="synthetic")
SIGMA_DRIVE_MM      = 5.0        # per-step drive-distance noise (σ)
SIGMA_ROT_DEG       = 3.0        # per-step rotation noise (σ)

# Particle filter
# PF_BETA: sharpness of the likelihood weighting — higher = more selective matches.
#   Particle weight ∝ exp(−β · feature_distance). Raise if too many false LCs;
#   lower if genuine revisits are not recognised.
#   For z-scored features (real sonar mode): random-pair E[L2²] ≈ 112 (56-dim window),
#   same-place E[L2²] ≈ 5–15. β=0.05 gives likelihood ratio ~100×.
#   For fixed-normalisation features (sim / burst): β=2 worked well.
PF_BETA             = 0.025

# MIN_LC_GAP: minimum step separation between the two ends of a loop closure.
#   Prevents the PF from "recognising" a place it just left (features are trivially
#   similar between consecutive steps). Scale with step size: at 160 mm/step,
#   10 steps ≈ 1600 mm exclusion zone.
MIN_LC_GAP          = 10

# LC_WEIGHT_THRESHOLD: minimum PF weight a candidate step must accumulate before
#   it is accepted as a loop-closure partner. Higher = fewer but more confident LCs.
LC_WEIGHT_THRESHOLD = 0.15

# LC_DEDUP_BUCKET: two loop closures that map to the same (s//bucket, t//bucket)
#   cell are merged into one. Avoids flooding the pose graph with near-duplicate
#   edges for the same physical revisit. At 160 mm/step, 3 steps ≈ 480 mm cell size.
LC_DEDUP_BUCKET     = 3

# LC_ODOM_GATE_K: a candidate LC (s→t) is rejected if the odometry distance between
#   s and t exceeds K × expected_drift(|t−s|). Guards against false matches when
#   the odometry is still close to ground truth early in the trajectory.
LC_ODOM_GATE_K      = 1.5

# LC_HEADING_GATE_K: reject an LC candidate if the odometry-estimated |Δyaw|
#   between s and t exceeds K · σ_rot · √|t−s| degrees. Sonar features are
#   directional — the spatial-info diagnostic showed feature distance is a
#   (place, heading) descriptor, not a pure place descriptor — so a revisit
#   from the opposite direction will have a different feature window and
#   shouldn't be accepted even if the PF happens to vote for it.
#
#   The threshold scales with the accumulated odometry yaw error between
#   s and t (≈ σ_rot·√|t−s|) so that the gate self-adjusts when σ_rot changes
#   across behavioural conditions. A σ-sweep (K∈{1.0, 1.5, 2.0} × σ_rot∈1..4°)
#   showed K=1.0 is robust from σ_rot=1° to σ_rot=4°: it's nearly as good as
#   K=1.5 at low σ and avoids the FP blow-up seen at K≥1.5 when σ_rot≥3°.
LC_HEADING_GATE_K   = 1.0

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

# Hard outlier pruning after the main GN loop.
# LCs whose 2D position residual (in mm) exceeds this threshold are removed
# and a short clean re-solve is run without Huber.
LC_PRUNE_DIST_MM    = 150.0   # ~3× W_LC_POS σ of 50 mm
GN_CLEAN_ITERS      = 20


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

def _expected_drift_mm(n_steps, drive_mm):
    """
    Expected position drift between two poses n_steps apart under the
    current (realistic) odometry noise. Drive-noise component scales as
    √n, rotation-noise component scales as n (dominant).
    """
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
            d_odom   = float(np.hypot(*(noisy_pos[s] - noisy_pos[t])))
            max_drift = LC_ODOM_GATE_K * _expected_drift_mm(abs(t - s), drive_mm=drive_mm_per_step)
            if d_odom > max_drift:
                rejected_by_drift += 1
                continue

        if noisy_yaw is not None:
            d_yaw_deg = float(abs(np.degrees(wrap_rad(noisy_yaw[s] - noisy_yaw[t]))))
            max_yaw = LC_HEADING_GATE_K * SIGMA_ROT_DEG * np.sqrt(max(abs(t - s), 1))
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
# Real-data ingestion
# ══════════════════════════════════════════════════════════════════════════════

def _interpolate_gaps(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear interpolation through NaN/None gaps, flat-extrapolation at edges."""
    out = arr.astype(np.float64).copy()
    if valid.all():
        return out
    idx = np.arange(len(out))
    if not valid.any():
        raise ValueError("no valid samples to interpolate from")
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def _unwrap_and_interp_yaw_deg(yaws_deg: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Unwrap valid yaw samples, interpolate through gaps, return wrapped degrees."""
    y = np.asarray(yaws_deg, dtype=np.float64).copy()
    if not valid.any():
        raise ValueError("no valid yaw samples")
    unwrapped = np.degrees(np.unwrap(np.radians(y[valid])))
    y[valid] = unwrapped
    y = _interpolate_gaps(y, valid)
    return ((y + 180.0) % 360.0) - 180.0


def _arena_rect_walls(bounds_mm: dict, spacing_mm: float = 20.0) -> np.ndarray:
    """Densely-sampled rectangle outline of the arena bounds (for plotting only)."""
    x0, x1 = bounds_mm["min_x"], bounds_mm["max_x"]
    y0, y1 = bounds_mm["min_y"], bounds_mm["max_y"]
    xs_h = np.arange(x0, x1 + spacing_mm, spacing_mm)
    ys_v = np.arange(y0, y1 + spacing_mm, spacing_mm)
    top    = np.column_stack([xs_h, np.full_like(xs_h, y1)])
    bottom = np.column_stack([xs_h, np.full_like(xs_h, y0)])
    left   = np.column_stack([np.full_like(ys_v, x0), ys_v])
    right  = np.column_stack([np.full_like(ys_v, x1), ys_v])
    return np.vstack([top, bottom, left, right]).astype(np.float32)


def ingest_real(run_dir: str):
    """
    Load per-step (GT pose, commanded motion, sonar features) from a PolicyRuns/
    directory of .dill files. Supports both schemas:

      - sonar (legacy):  data.sonar_package (dict); motion has drive_mm, rotate1, rotate2
      - burst:           data.sonar_packages (list of N_LOOKS dicts);
                         motion has rotate2, net_rotation, intra_burst_drive_mm,
                         inter_burst_drive_mm, look_physicals (N_LOOKS)

    Returns:
        positions    (N, 2) float64  — GT x,y (gaps interpolated)
        yaws_rad     (N,)  float64   — GT yaw in radians, wrapped to [-π,π]
        meas_seq     (N, D) float32  — normalised features; sonar D=4,
                                       burst D = 3·N_LOOKS + 1
        walls        (W, 2) float32  — arena-bounds rectangle for plotting
        run_name     str
        commanded    dict with "dθ_rad" (N-1,) and "dr_mm" (N-1,) from motion commands
    """
    files = sorted(glob.glob(os.path.join(run_dir, "data*.dill")))
    if not files:
        raise FileNotFoundError(f"No data*.dill files in {run_dir}")
    print(f"  Loading {len(files)} .dill steps from {run_dir}")

    # Detect schema from the first file. Burst runs emit a list of N_LOOKS
    # sonar packages per step; legacy sonar runs emit a single package.
    with open(files[0], "rb") as f:
        first_keys = dill.load(f)["data"].keys()
    is_burst = "sonar_packages" in first_keys
    print(f"  Format: {'burst' if is_burst else 'sonar'}")

    xs, ys, yaws_deg, valid = [], [], [], []

    # Sonar-specific per-step arrays
    drive_mm = []
    dist_mm, r1_deg = [], []
    log_L_list, log_R_list = [], []
    prom_L_list, prom_R_list = [], []
    # Shared / burst-specific
    r2_deg = []
    intra_mm, inter_mm = [], []
    per_step_looks = []        # burst: list[ list[(dist_mm, iid_db, look_deg)] ] or None for skip
    n_looks_seen = set()

    for p in files:
        with open(p, "rb") as f:
            d = dill.load(f)
        pos = d["data"]["position"]
        mot = d["data"]["motion"]

        x, y, yaw = pos.get("x"), pos.get("y"), pos.get("yaw_deg")
        ok = (x is not None and y is not None
              and yaw is not None and np.isfinite(float(yaw)))
        xs.append(np.nan if x is None else float(x))
        ys.append(np.nan if y is None else float(y))
        yaws_deg.append(np.nan if (yaw is None) else float(yaw))
        valid.append(ok)

        if is_burst:
            r2_deg.append(float(mot.get("rotate2", 0.0) or 0.0))
            intra_mm.append(float(mot.get("intra_burst_drive_mm", 0.0) or 0.0))
            inter_mm.append(float(mot.get("inter_burst_drive_mm", 0.0) or 0.0))
            sps = d["data"].get("sonar_packages")
            if sps:
                look_angles = mot.get("look_physicals") or []
                looks = []
                for k, sp in enumerate(sps):
                    looks.append((
                        float(sp["corrected_distance"]) * 1000.0,   # m → mm
                        float(sp["corrected_iid"]),                 # signed dB
                        float(look_angles[k]) if k < len(look_angles) else 0.0,
                    ))
                n_looks_seen.add(len(looks))
                per_step_looks.append(looks)
            else:
                per_step_looks.append(None)     # skip-step: no pings this step
        else:
            sp = d["data"]["sonar_package"]
            drive_mm.append(float(mot["drive_mm"]))
            dist_mm.append(float(sp["corrected_distance"]) * 1000.0)
            r1_deg.append(float(mot["rotate1"]))
            r2_deg.append(float(mot["rotate2"]))
            env       = sp["sonar_data"][:, :2].astype(np.float64)
            integrals = np.array(sp["integrals"], dtype=np.float64) + 1.0
            sum_total = env.sum(axis=0) + 1.0
            log_L_list.append(float(sp["log_integrals"][0]))
            log_R_list.append(float(sp["log_integrals"][1]))
            prom_L_list.append(float(integrals[0] / sum_total[0]))
            prom_R_list.append(float(integrals[1] / sum_total[1]))

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

    if is_burst:
        if not n_looks_seen:
            raise ValueError(f"no non-skip burst steps in {run_dir}")
        if len(n_looks_seen) > 1:
            print(f"  Warning: N_LOOKS varies across steps {sorted(n_looks_seen)}; "
                  f"padding short bursts with zeros")
        n_looks = max(n_looks_seen)

        # Feature vector (matches SlamCore.collect_data burst branch): per look
        # sorted by angle ascending: [d/max_dist, iid/max_iid, r1/max_r1], then
        # one trailing [r2/max_r2]. Skip-steps emit zeros; the PF will simply
        # not match them.
        meas_seq = np.zeros((len(files), 3 * n_looks + 1), dtype=np.float32)
        for i, looks in enumerate(per_step_looks):
            if looks is None:
                continue
            looks_sorted = sorted(looks, key=lambda t: t[2])
            padded = looks_sorted + [(0.0, 0.0, 0.0)] * (n_looks - len(looks_sorted))
            for k, (d_mm, i_db, r1) in enumerate(padded):
                j = 3 * k
                meas_seq[i, j    ] = np.clip(d_mm, 0.0, REAL_MAX_DIST_MM) / REAL_MAX_DIST_MM
                meas_seq[i, j + 1] = i_db / REAL_MAX_IID_DB
                meas_seq[i, j + 2] = r1   / REAL_MAX_R1_DEG
            meas_seq[i, -1] = r2_deg[i] / REAL_MAX_R2_DEG

        # Commanded body-frame motion in burst mode.
        # pose[k] is the end-of-step GT for step k (recorded after r2 +
        # inter_burst_drive). The transition pose[k] → pose[k+1] is driven
        # entirely by the motion of step k+1:
        #   1. Burst phase: N_LOOKS rotations with rotate-backs, net heading
        #      change = 0; net drive = intra_burst_drive_mm along h_k.
        #   2. rotate2 (net_rotation).
        #   3. inter_burst_drive_mm along h_k + r2.
        #
        # The solver's edge model is "rotate then drive", so we lump:
        #     dθ[k] = rotate2[k+1]
        #     dr[k] = intra_burst_drive_mm[k+1] + inter_burst_drive_mm[k+1]
        # This attributes the intra-burst drive to the post-r2 heading, a
        # small bias on non-zero-r2 steps that SLAM absorbs as odom residual.
        #
        # Both commanded rotations and overhead-camera yaw are CCW-positive
        # (post-May-2026 firmware flip; see Convention block on the Notion
        # "Lorex Camera System" page), so no sign flip is needed here.
        r2 = np.asarray(r2_deg, dtype=np.float64)
        intra = np.asarray(intra_mm, dtype=np.float64)
        inter = np.asarray(inter_mm, dtype=np.float64)
        dθ_rad = np.radians(r2[1:])
        dr_mm_ = intra[1:] + inter[1:]
    else:
        # Feature set: dist_mm, log_L, log_R, prom_L, prom_R, r1_deg, r2_deg
        # Best-performing scalar set from SCRIPT_SonarSpatialInfo (Precision@1=0.726 w=8).
        # Z-scored across the run so all features have equal variance (~1) regardless
        # of raw scale; this also makes PF_BETA independent of feature units.
        raw = np.column_stack([
            np.asarray(dist_mm),
            np.asarray(log_L_list),
            np.asarray(log_R_list),
            np.asarray(prom_L_list),
            np.asarray(prom_R_list),
            np.asarray(r1_deg),
            np.asarray(r2_deg),
        ]).astype(np.float64)
        means = raw.mean(axis=0)
        stds  = raw.std(axis=0)
        stds[stds < 1e-9] = 1.0
        meas_seq = ((raw - means) / stds).astype(np.float32)

        # Commanded body-frame motion: dθ, dr between consecutive recorded poses.
        # Per SCRIPT_RunPolicy, pose[k] is read AFTER rotate1[k], BEFORE rotate2[k]
        # + drive[k].  So the body-frame edge from pose[k] to pose[k+1] is:
        #     rotate2[k]  →  drive[k]  →  rotate1[k+1]
        # In the solver's rotate-then-drive convention this collapses to
        #     dθ[k] = rotate2[k] + rotate1[k+1]
        #     dr[k] = drive[k]
        # (the drive's actual heading is yaw[k]+rotate2[k] vs. the solver's
        #  assumed yaw[k]+dθ[k]; the rotate1[k+1] mis-attribution is small when
        #  rotations are small per step.)
        #
        # Both commanded rotations and the solver's trig are CCW-positive
        # (post-May-2026 firmware flip; see Convention block on the Notion
        # "Lorex Camera System" page), so no sign flip is needed here.
        r1 = np.asarray(r1_deg, dtype=np.float64)
        r2 = np.asarray(r2_deg, dtype=np.float64)
        dθ_rad = np.radians(r2[:-1] + r1[1:])
        dr_mm_ = np.asarray(drive_mm[:-1], dtype=np.float64)

    commanded = {"dθ_rad": dθ_rad, "dr_mm": dr_mm_}

    # Walls for plotting — prefer arena_features.npz (per-camera back-projected
    # walls + poles); otherwise fall back to the arena-bounds rectangle from
    # meta.json. Pole points are excluded here because the SLAM plot wants
    # wall geometry only.
    env_dirs = sorted(p for p in os.listdir(run_dir)
                      if p.startswith("env_") and os.path.isdir(os.path.join(run_dir, p)))
    walls = None
    if env_dirs:
        npz_path = os.path.join(run_dir, env_dirs[0], "arena_features.npz")
        if os.path.isfile(npz_path):
            data = np.load(npz_path)
            x_all = np.asarray(data["x_mm"])
            y_all = np.asarray(data["y_mm"])
            kind = np.asarray(data["kind"]) if "kind" in data.files else np.zeros_like(x_all, dtype=np.uint8)
            wall_sel = kind == 0
            walls = np.column_stack([x_all[wall_sel], y_all[wall_sel]]).astype(np.float32)
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


def ingest_sim(run_dir: str, session_name: str, max_steps: int, rng):
    """Thin wrapper around SlamCore.collect_data for the sim path."""
    mod, cfg, policy, is_burst = load_run(run_dir)
    print(f"  Type: {'burst' if is_burst else 'sonar'}")
    cfg_one = dataclasses.replace(cfg, train_session_names=[session_name])
    positions, yaws_deg, meas_seq, traj_ids, simulators = collect_data(
        mod, cfg_one, policy, n_traj=1, max_steps=max_steps, rng=rng,
    )
    positions = positions.astype(np.float64)
    yaws_rad  = np.radians(yaws_deg.astype(np.float64))
    walls     = next(iter(simulators.values())).arena.walls
    run_name  = os.path.basename(run_dir.rstrip("/"))

    if is_burst:
        drive_mm_per_step = cfg.intra_burst_drive_mm + cfg.inter_burst_drive_mm
    else:
        drive_mm_per_step = cfg.fixed_drive_mm
    print(f"  Drive per step: {drive_mm_per_step:.0f} mm")

    return positions, yaws_rad, meas_seq, walls, run_name, drive_mm_per_step


def odom_from_commanded(positions, yaws_rad, commanded):
    """
    Integrate the robot's commanded motion forward from the first GT pose.
    Returns (noisy_pos, noisy_yaw, dθ_meas, dr_meas) in the same layout as
    simulate_odometry_se2. This is the *real* dead-reckoning: any drift
    between the commanded and the executed motion is the odometry error
    that the pose graph will try to cancel via loop closures.
    """
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

    # ── Hard prune: remove LCs with large final residuals, re-solve clean ─────
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
        r, J, _, _ = build_residuals_and_jacobian(
            x, y, θ, dθ_m, dr_m, pruned, anchor,
        )
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
        θ = wrap_rad(θ + step[:, 2])
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
    rng = np.random.default_rng(SEED)

    # ── Ingest ──────────────────────────────────────────────────────────────
    if DATA_SOURCE == "sim":
        run_dir = SIM_RUN_DIR
        print(f"\nRun: {os.path.basename(run_dir.rstrip('/'))}  (sim)")
        positions, yaws_rad, meas_seq, walls, run_name, drive_mm_per_step = ingest_sim(
            run_dir, SIM_SESSION_NAME, SIM_MAX_STEPS, rng,
        )
        commanded = None
    elif DATA_SOURCE == "real":
        run_dir = REAL_RUN_DIR
        print(f"\nRun: {os.path.basename(run_dir.rstrip('/'))}  (real)")
        positions, yaws_rad, meas_seq, walls, run_name, commanded = ingest_real(run_dir)
        drive_mm_per_step = float(np.mean(commanded["dr_mm"])) if len(commanded["dr_mm"]) else 0.0
        print(f"  Mean commanded drive per step: {drive_mm_per_step:.0f} mm")
    else:
        raise ValueError(f"unknown DATA_SOURCE={DATA_SOURCE!r}")

    N = len(positions)
    output_dir = os.path.join("SpatialInfo", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Trajectory length: {N} steps")

    # ── Odometry ────────────────────────────────────────────────────────────
    use_commanded = (DATA_SOURCE == "real" and ODOM_SOURCE == "commanded")
    if use_commanded:
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

    # ── PF place recognition ────────────────────────────────────────────────
    feats = build_windows(meas_seq, np.zeros(N, dtype=np.int32), WINDOW_LEN)

    print(f"\nRunning particle filter (β={PF_BETA:g})...")
    history, _ = run_pf(feats, rng, PF_BETA)

    print("\nExtracting loop closures...")
    loop_closures = extract_loop_closures(history, N, noisy_pos=noisy_pos,
                                          noisy_yaw=noisy_yaw,
                                          drive_mm_per_step=drive_mm_per_step)
    n_tp = sum(1 for s, t in loop_closures
               if np.hypot(*(positions[s] - positions[t])) < 300)
    n_fp = len(loop_closures) - n_tp
    prec = n_tp / max(1, len(loop_closures))
    print(f"  Found {len(loop_closures)} LCs  "
          f"(TP {n_tp}, FP {n_fp}, precision {prec:.3f})")

    # ── SE(2) pose-graph solve ──────────────────────────────────────────────
    print("\nSolving SE(2) pose graph (Gauss-Newton)...")
    relaxed_pos, relaxed_yaw, pruned_closures = solve_pose_graph_se2(
        noisy_pos, noisy_yaw, dθ_meas, dr_meas, loop_closures,
    )
    n_pruned_tp = sum(1 for s, t in pruned_closures
                      if np.hypot(*(positions[s] - positions[t])) < 300)
    print(f"  Final LC set: {len(pruned_closures)} "
          f"(TP {n_pruned_tp}, FP {len(pruned_closures) - n_pruned_tp})")

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
    plot_results(positions, noisy_pos, aligned_pos, pruned_closures,
                 walls, run_name, output_dir, errors)

    print("\nDone.")


if __name__ == "__main__":
    main()
