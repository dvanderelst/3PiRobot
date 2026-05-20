"""
Load AcquisitionSessions/<name>/ folders into the (sonar, profiles, quads,
sess, bin_centers) shape that SCRIPT_TrainSonarModel's load_data() returns,
so the SonarModel training pipeline can consume the new visually-guided
acquisition output without changes downstream of the data layer.

Differences from the legacy DataProcessor pipeline:
  - Each ping is its own .dill (DataWriter file), not a per-step
    (sonar, position, motion) tuple in an env_*/ subfolder.
  - Pose comes from `executed_pose` (tracker reading at ping time) saved
    alongside `sonar_package`, not from a separate `position` field. If the
    tracker missed at ping time we fall back to the planned pose.
  - Ground-truth profile is computed at load time from the session's copy
    of arena_features.npz (wall points only) + the executed pose; we don't
    precompute and stash profiles. Cost is negligible at SonarModel-training
    data sizes. Poles in arena_features.npz are ignored here — the legacy
    SonarModel is a wall-only profile predictor; pole-aware label generation
    for the two-headed inverse lives in a separate (future) loader.
  - Quadrants for train/val split are by sign of (x − x_med, y − y_med)
    inside each session, so each session's held-out quadrant carries
    roughly 25% of its pings regardless of the arena's shape.
"""

import json
from pathlib import Path
from typing import List, Tuple

import dill
import numpy as np


# ─── Paths and IO ────────────────────────────────────────────────────────────

def _resolve(p) -> Path:
    p = Path(p)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _list_ping_files(session_dir: Path) -> List[Path]:
    """Numbered DataWriter ping files in this session, in order."""
    return sorted(p for p in session_dir.iterdir()
                  if p.is_file() and p.name.startswith("data") and p.suffix == ".dill")


def _read_ping(path: Path) -> dict:
    """Return the per-ping payload dict.

    DataWriter wraps each save under a top-level `{"data": ..., "meta": ...}`
    structure (the inner `meta` is DataWriter's own bookkeeping — timestamp,
    file number — not our session metadata). We unwrap to the payload here
    so callers see the fields they were saved with.
    """
    with open(path, "rb") as f:
        rec = dill.load(f)
    return rec["data"] if isinstance(rec, dict) and "data" in rec else rec


def _load_walls_for_session(session_dir: Path) -> np.ndarray:
    """Load wall points from arena_features.npz for this session.

    Preferred location is the session root itself (newer runs copy the file in
    so each session is self-contained). For older sessions that predate the
    auto-copy, fall back to the arena referenced in `session_meta.json`,
    walking its env_*/ subfolder for the features file.

    Pole entries (kind == 1) are filtered out: this loader feeds the wall-only
    SonarModel training pipeline.
    """
    def _walls_from_npz(path: Path) -> np.ndarray:
        d = np.load(path)
        x_all = np.asarray(d["x_mm"])
        y_all = np.asarray(d["y_mm"])
        kind = np.asarray(d["kind"]) if "kind" in d.files else np.zeros_like(x_all, dtype=np.uint8)
        wall_sel = kind == 0
        return np.column_stack([x_all[wall_sel], y_all[wall_sel]]).astype(np.float32)

    features_path = session_dir / "arena_features.npz"
    if features_path.exists():
        return _walls_from_npz(features_path)

    meta_path = session_dir / "session_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        arena_dir = meta.get("arena_dir")
        if arena_dir:
            arena_dir = Path(arena_dir)
            env_dirs = sorted(p for p in arena_dir.iterdir()
                              if p.is_dir() and p.name.startswith("env_"))
            for env in reversed(env_dirs):  # newest first
                wp = env / "arena_features.npz"
                if wp.exists():
                    return _walls_from_npz(wp)

    raise FileNotFoundError(
        f"arena_features.npz not found in {session_dir} or via session_meta.json's "
        f"arena_dir reference."
    )


def _load_features_for_session(session_dir: Path) -> dict:
    """Load walls AND poles from arena_features.npz for this session.

    Same lookup path as `_load_walls_for_session` (session root first, then
    session_meta.json's arena_dir). Returns a dict with keys:
        walls           (N, 2) float32  wall points (kind == 0)
        poles           (M, 2) float32  pole centres (kind == 1)
        pole_radius_mm  float           physical pole radius (0.0 if absent)
    """
    def _features_from_npz(path: Path) -> dict:
        d = np.load(path)
        x_all = np.asarray(d["x_mm"])
        y_all = np.asarray(d["y_mm"])
        kind = np.asarray(d["kind"]) if "kind" in d.files else np.zeros_like(x_all, dtype=np.uint8)
        wall_sel = kind == 0
        pole_sel = kind == 1
        walls = np.column_stack([x_all[wall_sel], y_all[wall_sel]]).astype(np.float32)
        poles = np.column_stack([x_all[pole_sel], y_all[pole_sel]]).astype(np.float32)
        pole_radius_mm = float(d["pole_radius_mm"]) if "pole_radius_mm" in d.files else 0.0
        return {"walls": walls, "poles": poles, "pole_radius_mm": pole_radius_mm}

    features_path = session_dir / "arena_features.npz"
    if features_path.exists():
        return _features_from_npz(features_path)

    meta_path = session_dir / "session_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        arena_dir = meta.get("arena_dir")
        if arena_dir:
            arena_dir = Path(arena_dir)
            env_dirs = sorted(p for p in arena_dir.iterdir()
                              if p.is_dir() and p.name.startswith("env_"))
            for env in reversed(env_dirs):
                wp = env / "arena_features.npz"
                if wp.exists():
                    return _features_from_npz(wp)

    raise FileNotFoundError(
        f"arena_features.npz not found in {session_dir} or via session_meta.json's "
        f"arena_dir reference."
    )


# ─── Profile geometry ────────────────────────────────────────────────────────

def profile_bin_centers(opening_angle: float, profile_steps: int) -> np.ndarray:
    edges = np.linspace(-opening_angle / 2, opening_angle / 2, profile_steps + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def compute_profile(walls: np.ndarray,
                    rob_x: float, rob_y: float, rob_yaw_deg: float,
                    opening_angle: float, profile_steps: int,
                    profile_method: str = "ray_center") -> np.ndarray:
    """Geometric distance profile at (rob_x, rob_y, rob_yaw_deg) against the
    arena wall point cloud (`walls`, an (N, 2) array of (x, y) world mm).

    Returns a length-`profile_steps` array of per-bin minimum distance (mm),
    NaN for bins that no wall point falls into. Mirrors
    `ArenaLayout.compute_profile` from EnvironmentSimulator.py — kept here
    as a free function so the loader doesn't depend on the legacy
    session-folder layout that ArenaLayout is wired for.
    """
    yaw_rad = np.deg2rad(rob_yaw_deg)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    dx = walls[:, 0] - rob_x
    dy = walls[:, 1] - rob_y
    rel_x = dx * cos_y + dy * sin_y
    rel_y = -dx * sin_y + dy * cos_y
    angles_deg = np.rad2deg(np.arctan2(rel_y, rel_x))
    distances = np.hypot(rel_x, rel_y).astype(np.float32)

    edges = np.linspace(-opening_angle / 2, opening_angle / 2, profile_steps + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(profile_steps, np.nan, dtype=np.float32)

    if profile_method == "min_bin":
        bin_idx = np.digitize(angles_deg, edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < profile_steps)
        if valid.any():
            tmp = np.full(profile_steps, np.inf, dtype=np.float32)
            np.minimum.at(tmp, bin_idx[valid], distances[valid])
            out = np.where(np.isinf(tmp), np.nan, tmp).astype(np.float32)
    elif profile_method == "ray_center":
        half_bin = 0.5 * (edges[1] - edges[0]) if profile_steps > 1 else 180.0
        for i, center_deg in enumerate(centers):
            on_ray = np.abs(angles_deg - center_deg) <= half_bin
            if on_ray.any():
                out[i] = float(distances[on_ray].min())
    else:
        raise ValueError(f"Unknown profile_method={profile_method!r}")

    return out


def nearest_reflector_in_cone(walls: np.ndarray,
                              poles: np.ndarray,
                              pole_radius_mm: float,
                              rob_x: float, rob_y: float, rob_yaw_deg: float,
                              cone_half_deg: float) -> Tuple[float, float]:
    """Class of the nearest physical reflector in the forward ±cone, and the
    pole's signed bearing if a pole is the nearest.

    Per-ping label for the two-headed inverse:
      - Walls represented by their point cloud; distance is to the nearest
        wall point in the cone.
      - Poles represented by their centres; distance is (centre distance −
        pole_radius_mm), i.e. distance to the pole's surface, which is what
        the sonar actually echoes off.
      - Inclusion in the cone uses centre angle for both (cheap, exact for
        walls; an approximation for poles, but accurate to within ~5° at
        the planner's minimum approach distance of 275 mm).
      - Ties go to wall (the safer error: an "approach pole" policy that
        misclassifies wall→pole would drive into the wall).

    Returns:
        class_label       0.0 = wall, 1.0 = pole, NaN if cone is empty.
        pole_azimuth_deg  signed bearing to the nearest pole if class == pole,
                          NaN otherwise. Angle convention matches
                          `compute_profile`: +ccw from robot forward, in [-180, 180).
    """
    yaw_rad = np.deg2rad(rob_yaw_deg)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)

    wall_min_dist = np.inf
    if walls.size:
        dx = walls[:, 0] - rob_x
        dy = walls[:, 1] - rob_y
        rel_x = dx * cos_y + dy * sin_y
        rel_y = -dx * sin_y + dy * cos_y
        ang = np.rad2deg(np.arctan2(rel_y, rel_x))
        in_cone = np.abs(ang) <= cone_half_deg
        if in_cone.any():
            d = np.hypot(rel_x[in_cone], rel_y[in_cone])
            wall_min_dist = float(d.min())

    pole_min_dist = np.inf
    pole_min_az = float("nan")
    if poles.size:
        dx = poles[:, 0] - rob_x
        dy = poles[:, 1] - rob_y
        rel_x = dx * cos_y + dy * sin_y
        rel_y = -dx * sin_y + dy * cos_y
        ang = np.rad2deg(np.arctan2(rel_y, rel_x))
        d_surface = np.hypot(rel_x, rel_y) - pole_radius_mm
        in_cone = np.abs(ang) <= cone_half_deg
        if in_cone.any():
            d_in = d_surface[in_cone]
            ang_in = ang[in_cone]
            j = int(np.argmin(d_in))
            pole_min_dist = float(d_in[j])
            pole_min_az = float(ang_in[j])

    if not np.isfinite(wall_min_dist) and not np.isfinite(pole_min_dist):
        return float("nan"), float("nan")
    if wall_min_dist <= pole_min_dist:  # tie → wall
        return 0.0, float("nan")
    return 1.0, pole_min_az


# ─── Public loader API ───────────────────────────────────────────────────────

def load_session(session_name: str,
                 acquisitions_root: str = "AcquisitionSessions",
                 opening_angle: float = 270.0,
                 profile_steps: int = 90,
                 profile_method: str = "ray_center",
                 drop_pose_fallback: bool = True,
                 verbose: bool = True
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one AcquisitionSessions/<name>/ folder.

    `drop_pose_fallback` (default True): skip pings where the tracker missed
    at ping time. Those pings have `executed_pose=None`, and we don't know
    where the robot actually was — substituting the planned pose would feed
    the trainer a wrong (x, y, yaw) and therefore a wrong geometric profile,
    teaching the SonarModel to associate the sonar input with the wrong
    distance. Set False to keep them with planned-pose fallback if the
    fraction is small and label noise is acceptable.

    Returns:
        sonar    (N, samples, 2) — last dim is [L, R]; emitter channel dropped
                                    to match the legacy training pipeline.
        profiles (N, profile_steps) — geometric profile per ping (NaNs allowed)
        quads    (N,) int32 — quadrant label by sign of (x − x_med, y − y_med)
                              within this session
        poses    (N, 3) float32 — executed (or planned-fallback) (x, y, yaw_deg)
    """
    session_dir = _resolve(acquisitions_root) / session_name
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    walls = _load_walls_for_session(session_dir)
    ping_files = _list_ping_files(session_dir)
    if not ping_files:
        raise FileNotFoundError(f"No data*.dill files in {session_dir}")
    if verbose:
        print(f"  {session_name}: {len(ping_files)} pings  "
              f"(walls: {walls.shape[0]:,} pts)")

    sonar_list, profile_list, pose_list = [], [], []
    fallback_pose_count = 0
    dropped_pose_count = 0
    for pf in ping_files:
        rec = _read_ping(pf)
        sp = rec["sonar_package"]
        executed = rec.get("executed_pose")
        if executed is None:
            if drop_pose_fallback:
                dropped_pose_count += 1
                continue
            planned_xy = rec["planned_position"]
            planned_yaw = rec["planned_yaw"]
            executed = (planned_xy[0], planned_xy[1], planned_yaw)
            fallback_pose_count += 1
        x, y, yaw = float(executed[0]), float(executed[1]), float(executed[2])
        sd = np.asarray(sp["sonar_data"], dtype=np.float32)  # (samples, 3) = [emit, L, R]
        sonar_list.append(sd[:, [1, 2]])
        profile_list.append(
            compute_profile(walls, x, y, yaw,
                            opening_angle, profile_steps, profile_method)
        )
        pose_list.append((x, y, yaw))

    if verbose:
        if dropped_pose_count > 0:
            print(f"    {dropped_pose_count} pings dropped "
                  f"(tracker missed at ping time, no ground-truth pose)")
        if fallback_pose_count > 0:
            print(f"    {fallback_pose_count} pings used planned-pose fallback "
                  f"(tracker missed at ping time; drop_pose_fallback=False)")

    sonar = np.stack(sonar_list, axis=0).astype(np.float32)
    profiles = np.stack(profile_list, axis=0).astype(np.float32)
    poses = np.array(pose_list, dtype=np.float32)

    # Quadrant by per-session median split — robust to arena shape.
    x_med = float(np.median(poses[:, 0]))
    y_med = float(np.median(poses[:, 1]))
    quads = ((poses[:, 0] < x_med).astype(np.int32) +
             2 * (poses[:, 1] < y_med).astype(np.int32))
    return sonar, profiles, quads, poses


def load_data(session_names: List[str],
              acquisitions_root: str = "AcquisitionSessions",
              opening_angle: float = 270.0,
              profile_steps: int = 90,
              profile_method: str = "ray_center",
              drop_pose_fallback: bool = True,
              verbose: bool = True
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multi-session loader matching the return shape of
    SCRIPT_TrainSonarModel.load_data().

    Returns:
        sonar       (N, samples, 2)
        profiles    (N, profile_steps)
        quads       (N,) int32
        sess        (N,) object array of session names
        bin_centers (profile_steps,) azimuth bin centers (deg)
    """
    s_l, p_l, q_l, sess_l = [], [], [], []
    for name in session_names:
        sonar, profiles, quads, _ = load_session(
            name, acquisitions_root,
            opening_angle, profile_steps, profile_method,
            drop_pose_fallback=drop_pose_fallback,
            verbose=verbose,
        )
        s_l.append(sonar)
        p_l.append(profiles)
        q_l.append(quads)
        sess_l.append(np.array([name] * len(sonar)))
    return (np.concatenate(s_l, axis=0),
            np.concatenate(p_l, axis=0),
            np.concatenate(q_l, axis=0),
            np.concatenate(sess_l, axis=0),
            profile_bin_centers(opening_angle, profile_steps))


# ─── Two-headed inverse loader (walls + poles) ───────────────────────────────

def load_session_inverse(session_name: str,
                         acquisitions_root: str = "AcquisitionSessions",
                         opening_angle: float = 270.0,
                         profile_steps: int = 90,
                         profile_method: str = "ray_center",
                         cone_half_deg: float = 35.0,
                         drop_pose_fallback: bool = True,
                         verbose: bool = True
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray, np.ndarray]:
    """Load one session for two-headed inverse training.

    Same per-ping pipeline as `load_session`, but also computes the
    nearest-reflector class and pole azimuth from the env's poles. Walls
    used for `profiles` are wall-only (kind == 0); the class label is
    against the wall + pole geometry in the ±cone_half_deg forward cone.

    Returns:
        sonar         (N, samples, 2)
        profiles      (N, profile_steps) — wall-only profile (NaNs allowed)
        class_labels  (N,) float32 — 0.0=wall, 1.0=pole, NaN if cone empty
        pole_az_deg   (N,) float32 — bearing to nearest pole when class==pole;
                                     NaN otherwise. Same +ccw convention.
        quads         (N,) int32   — per-session median-split quadrant
        poses         (N, 3) float32 — (x, y, yaw_deg)
    """
    session_dir = _resolve(acquisitions_root) / session_name
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    feat = _load_features_for_session(session_dir)
    walls = feat["walls"]
    poles = feat["poles"]
    pole_radius_mm = feat["pole_radius_mm"]
    ping_files = _list_ping_files(session_dir)
    if not ping_files:
        raise FileNotFoundError(f"No data*.dill files in {session_dir}")
    if verbose:
        print(f"  {session_name}: {len(ping_files)} pings  "
              f"(walls: {walls.shape[0]:,} pts, poles: {poles.shape[0]}, "
              f"pole_r={pole_radius_mm:.0f} mm)")

    sonar_list:    List[np.ndarray]      = []
    profile_list:  List[np.ndarray]      = []
    class_list:    List[float]           = []
    pole_az_list:  List[float]           = []
    pose_list:     List[Tuple[float, float, float]] = []
    fallback_pose_count = 0
    dropped_pose_count = 0
    for pf in ping_files:
        rec = _read_ping(pf)
        sp = rec["sonar_package"]
        executed = rec.get("executed_pose")
        if executed is None:
            if drop_pose_fallback:
                dropped_pose_count += 1
                continue
            planned_xy = rec["planned_position"]
            planned_yaw = rec["planned_yaw"]
            executed = (planned_xy[0], planned_xy[1], planned_yaw)
            fallback_pose_count += 1
        x, y, yaw = float(executed[0]), float(executed[1]), float(executed[2])
        sd = np.asarray(sp["sonar_data"], dtype=np.float32)
        sonar_list.append(sd[:, [1, 2]])
        profile_list.append(
            compute_profile(walls, x, y, yaw,
                            opening_angle, profile_steps, profile_method)
        )
        cls, pole_az = nearest_reflector_in_cone(
            walls, poles, pole_radius_mm, x, y, yaw, cone_half_deg
        )
        class_list.append(cls)
        pole_az_list.append(pole_az)
        pose_list.append((x, y, yaw))

    if verbose:
        if dropped_pose_count > 0:
            print(f"    {dropped_pose_count} pings dropped "
                  f"(tracker missed at ping time, no ground-truth pose)")
        if fallback_pose_count > 0:
            print(f"    {fallback_pose_count} pings used planned-pose fallback "
                  f"(tracker missed at ping time; drop_pose_fallback=False)")

    sonar       = np.stack(sonar_list, axis=0).astype(np.float32)
    profiles    = np.stack(profile_list, axis=0).astype(np.float32)
    class_arr   = np.asarray(class_list, dtype=np.float32)
    pole_az_arr = np.asarray(pole_az_list, dtype=np.float32)
    poses       = np.asarray(pose_list, dtype=np.float32)

    x_med = float(np.median(poses[:, 0]))
    y_med = float(np.median(poses[:, 1]))
    quads = ((poses[:, 0] < x_med).astype(np.int32) +
             2 * (poses[:, 1] < y_med).astype(np.int32))

    if verbose:
        valid = ~np.isnan(class_arr)
        n_valid = int(valid.sum())
        n_wall = int(((class_arr == 0.0) & valid).sum())
        n_pole = int(((class_arr == 1.0) & valid).sum())
        print(f"    class balance: wall={n_wall}, pole={n_pole}, "
              f"empty_cone={len(class_arr) - n_valid}  "
              f"(pole frac = {n_pole / max(n_valid, 1):.1%})")

    return sonar, profiles, class_arr, pole_az_arr, quads, poses


def load_data_inverse(session_names: List[str],
                      acquisitions_root: str = "AcquisitionSessions",
                      opening_angle: float = 270.0,
                      profile_steps: int = 90,
                      profile_method: str = "ray_center",
                      cone_half_deg: float = 35.0,
                      drop_pose_fallback: bool = True,
                      verbose: bool = True
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multi-session two-headed-inverse loader.

    Returns:
        sonar        (N, samples, 2)
        profiles     (N, profile_steps) — wall-only
        class_labels (N,) float32 — 0.0=wall, 1.0=pole, NaN=empty cone
        pole_az_deg  (N,) float32 — bearing to nearest pole (NaN if class != pole)
        quads        (N,) int32
        sess         (N,) object array of session names
        bin_centers  (profile_steps,) azimuth bin centers (deg)
    """
    s_l, p_l, c_l, az_l, q_l, sess_l = [], [], [], [], [], []
    for name in session_names:
        sonar, profiles, classes, pole_az, quads, _ = load_session_inverse(
            name, acquisitions_root,
            opening_angle, profile_steps, profile_method,
            cone_half_deg=cone_half_deg,
            drop_pose_fallback=drop_pose_fallback,
            verbose=verbose,
        )
        s_l.append(sonar)
        p_l.append(profiles)
        c_l.append(classes)
        az_l.append(pole_az)
        q_l.append(quads)
        sess_l.append(np.array([name] * len(sonar)))
    return (np.concatenate(s_l, axis=0),
            np.concatenate(p_l, axis=0),
            np.concatenate(c_l, axis=0),
            np.concatenate(az_l, axis=0),
            np.concatenate(q_l, axis=0),
            np.concatenate(sess_l, axis=0),
            profile_bin_centers(opening_angle, profile_steps))
