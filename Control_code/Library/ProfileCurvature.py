import numpy as np

from Library.SonarWallSteering import SteeringConfig


def _smooth_heatmap(hm):
    k = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    k /= np.sum(k)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), axis=1, arr=hm)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), axis=0, arr=out)
    return out


def _dilate_binary(mask, radius_px):
    if radius_px <= 0:
        return mask.copy()
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    rr2 = radius_px * radius_px
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            if dx * dx + dy * dy > rr2:
                continue
            y_src0 = max(0, -dy)
            y_src1 = min(h, h - dy)
            x_src0 = max(0, -dx)
            x_src1 = min(w, w - dx)
            if y_src1 <= y_src0 or x_src1 <= x_src0:
                continue
            y_dst0 = y_src0 + dy
            y_dst1 = y_src1 + dy
            x_dst0 = x_src0 + dx
            x_dst1 = x_src1 + dx
            out[y_dst0:y_dst1, x_dst0:x_dst1] |= mask[y_src0:y_src1, x_src0:x_src1]
    return out


def _sample_tangent_circle(radius_mm, side_sign, horizon_x_mm, n_samples):
    if radius_mm <= 1e-6:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    phi_max = np.arcsin(np.clip(horizon_x_mm / radius_mm, 0.0, 1.0))
    if phi_max <= 1e-6:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    phi = np.linspace(0.0, phi_max, int(n_samples), dtype=np.float32)
    x = radius_mm * np.sin(phi)
    y = side_sign * radius_mm * (1.0 - np.cos(phi))
    return x.astype(np.float32), y.astype(np.float32)


def _path_collision_free(path_x, path_y, x_grid, y_grid, blocked):
    h, w = blocked.shape
    for x, y in zip(path_x, path_y):
        ix = int(np.argmin(np.abs(x_grid - x)))
        iy = int(np.argmin(np.abs(y_grid - y)))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return False
        if blocked[iy, ix]:
            return False
    return True


def _path_evidence_score(path_x, path_y, x_grid, y_grid, hm):
    if path_x is None or len(path_x) == 0:
        return np.inf
    vals = []
    h, w = hm.shape
    for x, y in zip(path_x, path_y):
        ix = int(np.argmin(np.abs(x_grid - x)))
        iy = int(np.argmin(np.abs(y_grid - y)))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            continue
        vals.append(float(hm[iy, ix]))
    if len(vals) == 0:
        return np.inf
    return float(np.mean(vals))


def _planner_to_curvature(planner):
    side = str(planner.get('chosen_side', 'straight'))
    radius = float(planner.get('chosen_radius_mm', np.inf))
    if not np.isfinite(radius) or radius <= 1e-6:
        return 0.0
    if side.startswith('left'):
        return 1.0 / radius
    if side.startswith('right'):
        return -1.0 / radius
    return 0.0


def _accumulate_segment(evidence, xx, yy, x_grid, y_grid, config, p0, p1, seg_len):
    tangent = (p1 - p0) / seg_len
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    mid = 0.5 * (p0 + p1)
    half_len = 0.5 * seg_len

    grid_x_min = float(x_grid[0])
    grid_y_min = float(y_grid[0])
    local_margin = 3.0 * max(config.sigma_perp_mm, config.sigma_para_mm)
    bx0 = max(float(x_grid[0]), min(p0[0], p1[0]) - local_margin)
    bx1 = min(float(x_grid[-1]), max(p0[0], p1[0]) + local_margin)
    by0 = max(float(y_grid[0]), min(p0[1], p1[1]) - local_margin)
    by1 = min(float(y_grid[-1]), max(p0[1], p1[1]) + local_margin)

    ix0 = int(max(0, np.floor((bx0 - grid_x_min) / config.grid_mm)))
    ix1 = int(min(len(x_grid) - 1, np.ceil((bx1 - grid_x_min) / config.grid_mm)))
    iy0 = int(max(0, np.floor((by0 - grid_y_min) / config.grid_mm)))
    iy1 = int(min(len(y_grid) - 1, np.ceil((by1 - grid_y_min) / config.grid_mm)))

    sub_x = xx[iy0:iy1 + 1, ix0:ix1 + 1]
    sub_y = yy[iy0:iy1 + 1, ix0:ix1 + 1]
    dx = sub_x - mid[0]
    dy = sub_y - mid[1]

    d_para = dx * tangent[0] + dy * tangent[1]
    d_perp = dx * normal[0] + dy * normal[1]
    outside_para = np.maximum(np.abs(d_para) - half_len, 0.0)

    kernel = np.exp(
        -0.5 * (d_perp / config.sigma_perp_mm) ** 2
        -0.5 * (outside_para / config.sigma_para_mm) ** 2
    )
    evidence[iy0:iy1 + 1, ix0:ix1 + 1] += kernel.astype(np.float32)


def profile2curvature(az_deg, dist_mm, config: SteeringConfig | None = None, return_debug=False):
    """
    Compute signed curvature from a single robot-frame obstacle profile.

    Parameters
    ----------
    az_deg : array-like
        Profile azimuth centers in robot frame (degrees).
    dist_mm : array-like
        First-obstacle distance per azimuth bin (mm). NaN values are ignored.
    config : SteeringConfig, optional
        Planner/grid parameters. If None, defaults are used.
    return_debug : bool, optional
        If True, returns (curvature, debug_dict) instead of just curvature.

    Returns
    -------
    float or tuple
        Signed curvature (1/mm). Positive=left, negative=right.
    """
    cfg = config if config is not None else SteeringConfig()
    az_deg = np.asarray(az_deg, dtype=np.float32)
    dist_mm = np.asarray(dist_mm, dtype=np.float32)
    if az_deg.ndim != 1 or dist_mm.ndim != 1 or len(az_deg) != len(dist_mm):
        raise ValueError('az_deg and dist_mm must be 1D arrays of equal length.')

    x_min, x_max = -float(cfg.extent_mm), float(cfg.extent_mm)
    y_min, y_max = -float(cfg.extent_mm), float(cfg.extent_mm)
    x_grid = np.arange(x_min, x_max + cfg.grid_mm, cfg.grid_mm, dtype=np.float32)
    y_grid = np.arange(y_min, y_max + cfg.grid_mm, cfg.grid_mm, dtype=np.float32)
    xx, yy = np.meshgrid(x_grid, y_grid)

    az_rad = np.deg2rad(az_deg)
    profile_x = dist_mm * np.cos(az_rad)
    profile_y = dist_mm * np.sin(az_rad)

    evidence = np.zeros_like(xx, dtype=np.float32)
    for j in range(len(az_deg) - 1):
        if not (
            np.isfinite(profile_x[j]) and np.isfinite(profile_y[j]) and
            np.isfinite(profile_x[j + 1]) and np.isfinite(profile_y[j + 1])
        ):
            continue
        p0 = np.array([profile_x[j], profile_y[j]], dtype=float)
        p1 = np.array([profile_x[j + 1], profile_y[j + 1]], dtype=float)
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len < 1e-6:
            continue
        _accumulate_segment(
            evidence=evidence, xx=xx, yy=yy, x_grid=x_grid, y_grid=y_grid, config=cfg, p0=p0, p1=p1, seg_len=seg_len
        )

    hm_norm = evidence.copy()
    if cfg.apply_heatmap_smoothing:
        hm_norm = _smooth_heatmap(hm_norm)
    max_v = float(np.max(hm_norm))
    if max_v > 1e-12:
        hm_norm = hm_norm / max_v
    hm_norm = hm_norm.astype(np.float32)

    wall_mask = hm_norm >= cfg.occ_block_threshold
    grid_mm = float(np.mean(np.diff(x_grid))) if len(x_grid) > 1 else cfg.grid_mm
    inflate_px = int(np.ceil((cfg.robot_radius_mm + cfg.safety_margin_mm) / max(grid_mm, 1e-6)))
    blocked = _dilate_binary(wall_mask, inflate_px)

    radii = np.arange(
        cfg.circle_radius_min_mm,
        cfg.circle_radius_max_mm + cfg.circle_radius_step_mm,
        cfg.circle_radius_step_mm,
        dtype=np.float32,
    )
    result = {}
    for side_name, side_sign in [('left', 1.0), ('right', -1.0)]:
        best_r = None
        best_xy = (None, None)
        touched_obstacle = False
        for r in radii:
            px, py = _sample_tangent_circle(float(r), side_sign, cfg.circle_horizon_x_mm, cfg.circle_arc_samples)
            ok = _path_collision_free(px, py, x_grid, y_grid, blocked)
            if ok:
                best_r = float(r)
                best_xy = (px, py)
            else:
                touched_obstacle = True
                break
        result[side_name] = {
            'radius': best_r,
            'x': best_xy[0],
            'y': best_xy[1],
            'touched_obstacle': touched_obstacle,
        }

    left_score = _path_evidence_score(result['left']['x'], result['left']['y'], x_grid, y_grid, hm_norm)
    right_score = _path_evidence_score(result['right']['x'], result['right']['y'], x_grid, y_grid, hm_norm)

    r_left = result['left']['radius'] if result['left']['radius'] is not None else -np.inf
    r_right = result['right']['radius'] if result['right']['radius'] is not None else -np.inf
    left_never_touched = not result['left']['touched_obstacle']
    right_never_touched = not result['right']['touched_obstacle']

    if (r_left < 0) and (r_right < 0):
        chosen_side = 'straight'
        chosen_radius = np.inf
    elif left_never_touched and right_never_touched:
        chosen_side = 'straight_open'
        chosen_radius = np.inf
    else:
        if abs(float(r_left - r_right)) <= float(cfg.circle_radius_tie_mm):
            if left_score <= right_score:
                chosen_side = 'left'
                chosen_radius = r_left
            else:
                chosen_side = 'right'
                chosen_radius = r_right
        elif r_left > r_right:
            chosen_side = 'left'
            chosen_radius = r_left
        else:
            chosen_side = 'right'
            chosen_radius = r_right

    planned = {
        'chosen_side': chosen_side,
        'chosen_radius_mm': float(chosen_radius) if np.isfinite(chosen_radius) else float('inf'),
        'left_radius_mm': None if result['left']['radius'] is None else float(result['left']['radius']),
        'right_radius_mm': None if result['right']['radius'] is None else float(result['right']['radius']),
        'left_evidence_score': float(left_score) if np.isfinite(left_score) else float('nan'),
        'right_evidence_score': float(right_score) if np.isfinite(right_score) else float('nan'),
        'left_x': result['left']['x'],
        'left_y': result['left']['y'],
        'right_x': result['right']['x'],
        'right_y': result['right']['y'],
        'blocked_mask': blocked,
    }
    curvature = _planner_to_curvature(planned)

    if not return_debug:
        return curvature

    debug = {
        'signed_curvature_inv_mm': float(curvature),
        'evidence_map': hm_norm,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'profile_x': profile_x,
        'profile_y': profile_y,
        **planned,
    }
    return curvature, debug
