"""
Library/SlamCore.py

Shared building blocks for SLAM experiments on policy-trained trajectories.

Exports:
    load_run(run_dir)
        Load a trained policy and its config. Auto-detects sonar vs burst.

    collect_data(mod, cfg, policy, n_traj, max_steps, rng)
        Run the policy in each training session and record per-step
        (position, yaw, feature vector). The feature vector concatenates
        dist/IID/r1 for every look (N_LOOKS for burst, 1 for sonar) plus
        a final r2.

    build_windows(meas_seq, traj_ids, window)
        Stack each step with the (window−1) preceding steps (zero-padded
        at trajectory boundaries).

    simulate_odometry(positions, yaws, traj_ids, σ_xy, σ_yaw, rng)
        Isotropic world-frame noise on per-step displacement and yaw.
        (The SE(2) script uses its own body-frame integration instead.)

    run_pf(feats, rng, beta)
        RatSLAM-style particle filter over a growing experience map.
        Returns (history, heatmap) — see docstring.

PF tuning constants (N_PARTICLES, P_ADVANCE, P_STAY, P_JUMP) are exposed
at module scope. Importers can override by assigning to them before
calling run_pf, or (cleanest) pass overrides through a wrapper.
"""

import dataclasses
import importlib
import json
import os

import numpy as np


# ── Particle filter defaults ──────────────────────────────────────────────────
N_PARTICLES = 1000
P_ADVANCE   = 0.60      # s → s+1
P_STAY      = 0.25      # s → s
P_JUMP      = 0.15      # s → uniform({0, ..., t-1})  — recovery / injection


# ══════════════════════════════════════════════════════════════════════════════
# Load a trained policy
# ══════════════════════════════════════════════════════════════════════════════

def load_run(run_dir: str):
    """Load config, detect sonar vs burst, import training module, load policy."""
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg_dict = json.load(f)
    with open(os.path.join(run_dir, "best_policy.json")) as f:
        pol_dict = json.load(f)
    is_burst = "n_looks" in cfg_dict
    mod = importlib.import_module(
        "SCRIPT_TrainPolicy_Burst" if is_burst else "SCRIPT_TrainPolicy"
    )
    valid_keys = {f.name for f in dataclasses.fields(mod.Config)}
    cfg = mod.Config(**{k: v for k, v in cfg_dict.items() if k in valid_keys})
    policy = mod.MLPPolicy(cfg)
    policy.set_genome(np.array(pol_dict["genome"], dtype=np.float32))
    return mod, cfg, policy, is_burst


# ══════════════════════════════════════════════════════════════════════════════
# Collect trajectories
# ══════════════════════════════════════════════════════════════════════════════

def collect_data(mod, cfg, policy, n_traj: int, max_steps: int, rng):
    """
    Run policy in all training arenas; extract per-step (pos, yaw, measurements).
    Also returns one simulator per session for wall plotting.
    """
    override_cfg = dataclasses.replace(cfg, max_steps=max_steps,
                                       plot_trajectories_every_n=0)
    all_pos, all_yaw, all_meas, all_traj = [], [], [], []
    simulators = {}
    traj_counter = 0

    for sn in cfg.train_session_names:
        sim    = mod.build_simulator(sn, quiet=True)
        simulators[sn] = sim
        starts = mod.load_starts(sn, cfg, quiet=True)
        if not starts:
            continue
        trajs = mod.record_trajectories(policy, sim, starts, override_cfg, n_traj, rng)

        for traj in trajs:
            steps = traj.get("steps", [])
            if not steps:
                continue
            for step in steps:
                pos = traj["positions"][step["step"]]
                yaw = traj["body_yaws"][step["step"]]
                if "looks" in step:   # burst
                    raw = sorted(
                        [(lk["emu_dist_mm"], lk["emu_iid_db"], lk["r1_deg"])
                         for lk in step["looks"]],
                        key=lambda t: t[2],  # sort by look angle: leftmost first
                    )
                else:                 # sonar
                    raw = [(step["emu_dist_mm"], step["emu_iid_db"], step["rotate1_deg"])]
                meas = []
                max_r1 = cfg.max_rotate1_deg + getattr(cfg, "max_burst_spread_deg", 0.0) / 2
                for d, i, r1 in raw:
                    meas.append(d / cfg.max_dist_mm)
                    meas.append(i / cfg.max_iid_db)
                    meas.append(r1 / max_r1)
                meas.append(step["rotate2_deg"] / cfg.max_rotate2_deg)
                all_pos.append(pos)
                all_yaw.append(yaw)
                all_meas.append(meas)
                all_traj.append(traj_counter)
            traj_counter += 1

    positions = np.array(all_pos,  dtype=np.float32)
    yaws      = np.array(all_yaw,  dtype=np.float32)
    meas_seq  = np.array(all_meas, dtype=np.float32)
    traj_ids  = np.array(all_traj, dtype=np.int32)

    print(f"  Collected {len(positions)} steps across "
          f"{traj_counter} trajectories, {len(cfg.train_session_names)} sessions")
    return positions, yaws, meas_seq, traj_ids, simulators


def build_windows(meas_seq: np.ndarray, traj_ids: np.ndarray, window: int) -> np.ndarray:
    """Concatenate current + (window-1) previous step measurements; zero-pad at boundaries."""
    N, meas_dim = meas_seq.shape
    feats = np.zeros((N, window * meas_dim), dtype=np.float32)
    for i in range(N):
        for w in range(window):
            j = i - w
            if j >= 0 and traj_ids[j] == traj_ids[i]:
                feats[i, w * meas_dim:(w + 1) * meas_dim] = meas_seq[j]
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# Noisy dead-reckoning (isotropic world-frame noise — used by the 2D pose graph)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_odometry(positions, yaws, traj_ids, noise_xy_mm, noise_yaw_deg, rng):
    """
    Simulate dead-reckoning: integrate ground-truth displacements with additive
    per-step Gaussian noise. Returns noisy_pos (N,2) and noisy_yaws (N,).
    """
    noisy_pos  = positions.copy().astype(np.float32)
    noisy_yaws = yaws.copy().astype(np.float32)

    for t in np.unique(traj_ids):
        idx = np.where(traj_ids == t)[0]
        for k in range(1, len(idx)):
            disp     = positions[idx[k]] - positions[idx[k - 1]]
            yaw_disp = float(yaws[idx[k]]) - float(yaws[idx[k - 1]])
            noisy_pos[idx[k]]  = (noisy_pos[idx[k - 1]]
                                  + disp
                                  + rng.normal(0.0, noise_xy_mm, 2).astype(np.float32))
            noisy_yaws[idx[k]] = (noisy_yaws[idx[k - 1]]
                                  + yaw_disp
                                  + float(rng.normal(0.0, noise_yaw_deg)))
    return noisy_pos, noisy_yaws


# ══════════════════════════════════════════════════════════════════════════════
# RatSLAM-style particle filter over a growing experience map
# ══════════════════════════════════════════════════════════════════════════════

def _propagate(particles: np.ndarray, t: int, rng) -> np.ndarray:
    """Advance each particle by one of: s→s+1, s→s, or s→uniform random."""
    r = rng.random(len(particles))
    advance = r < P_ADVANCE
    jump    = r >= (P_ADVANCE + P_STAY)

    new = particles.copy()
    new[advance] = np.minimum(particles[advance] + 1, t - 1)
    if jump.any():
        new[jump] = rng.integers(0, t, size=int(jump.sum()))
    return new


def _resample(particles, weights, rng):
    idx = rng.choice(len(particles), size=len(particles), p=weights)
    return particles[idx], np.ones_like(weights) / len(weights)


def umeyama_align(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """
    Similarity (or rigid) alignment of 2D point cloud `src` onto `dst`
    (Umeyama 1991). Finds the (scale c, rotation R, translation t) that
    minimises Σ‖c·R·src_i + t − dst_i‖².

    Parameters
    ----------
    src : (N, 2) array — points to be aligned (e.g. SLAM output)
    dst : (N, 2) array — reference points (e.g. ground truth)
    with_scale : if False, solve rigid alignment (c = 1)

    Returns
    -------
    aligned : (N, 2) — src mapped into dst's frame
    params  : dict with 'scale', 'R' (2×2), 't' (2,)
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, d = src.shape
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    sx, sy = src - mu_src, dst - mu_dst
    cov = (sy.T @ sx) / n                              # (d, d)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_src = float((sx * sx).sum() / n)
        c = float(np.trace(np.diag(D) @ S)) / var_src if var_src > 0 else 1.0
    else:
        c = 1.0
    t = mu_dst - c * R @ mu_src
    aligned = c * (src @ R.T) + t
    return aligned, {"scale": c, "R": R, "t": t}


def run_pf(feats: np.ndarray, rng, beta: float):
    """
    Online particle filter over a growing experience map.

    Returns
    -------
    history : list of (particles, weights) per step
    heatmap : (N, N) array — heatmap[s, t] = total particle weight at past
              index s during step t
    """
    N = len(feats)
    particles = np.zeros(N_PARTICLES, dtype=np.int32)
    weights   = np.ones(N_PARTICLES, dtype=np.float64) / N_PARTICLES

    history = [(particles.copy(), weights.copy())]
    heatmap = np.zeros((N, N), dtype=np.float32)

    for t in range(1, N):
        # Predict
        particles = _propagate(particles, t, rng)

        # Update
        diff = feats[particles] - feats[t]
        ll   = np.exp(-beta * np.sum(diff ** 2, axis=1))
        weights = weights * ll
        total = weights.sum()
        if total < 1e-20:
            particles = rng.integers(0, t, size=N_PARTICLES)
            weights   = np.ones(N_PARTICLES, dtype=np.float64) / N_PARTICLES
        else:
            weights = weights / total

        # Resample on low ESS
        ess = 1.0 / float(np.sum(weights ** 2))
        if ess < N_PARTICLES / 2:
            particles, weights = _resample(particles, weights, rng)

        history.append((particles.copy(), weights.copy()))
        np.add.at(heatmap[:, t], particles, weights.astype(np.float32))

    return history, heatmap
