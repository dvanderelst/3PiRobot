import numpy as np


def dilate_binary(mask, radius_px):
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


def sample_tangent_circle(radius_mm, side_sign, horizon_x_mm, n_samples):
    if radius_mm <= 1e-6:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    phi_max = np.arcsin(np.clip(horizon_x_mm / radius_mm, 0.0, 1.0))
    if phi_max <= 1e-6:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    phi = np.linspace(0.0, phi_max, int(n_samples), dtype=np.float32)
    x = radius_mm * np.sin(phi)
    y = side_sign * radius_mm * (1.0 - np.cos(phi))
    return x.astype(np.float32), y.astype(np.float32)


def path_collision_free(path_x, path_y, x_grid, y_grid, blocked):
    h, w = blocked.shape
    for x, y in zip(path_x, path_y):
        ix = int(np.argmin(np.abs(x_grid - x)))
        iy = int(np.argmin(np.abs(y_grid - y)))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return False
        if blocked[iy, ix]:
            return False
    return True


def path_evidence_score(path_x, path_y, x_grid, y_grid, hm):
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


def plan_circles_from_heatmap(hm_norm, x_grid, y_grid, config):
    wall_mask = hm_norm >= float(config.occ_block_threshold)
    grid_mm = float(np.mean(np.diff(x_grid))) if len(x_grid) > 1 else float(config.grid_mm)
    inflate_px = int(np.ceil((float(config.robot_radius_mm) + float(config.safety_margin_mm)) / max(grid_mm, 1e-6)))
    blocked = dilate_binary(wall_mask, inflate_px)

    radii = np.arange(
        float(config.circle_radius_min_mm),
        float(config.circle_radius_max_mm) + float(config.circle_radius_step_mm),
        float(config.circle_radius_step_mm),
        dtype=np.float32,
    )
    result = {}
    for side_name, side_sign in [('left', 1.0), ('right', -1.0)]:
        best_r = None
        best_xy = (None, None)
        touched_obstacle = False
        for r in radii:
            px, py = sample_tangent_circle(float(r), side_sign, float(config.circle_horizon_x_mm), int(config.circle_arc_samples))
            ok = path_collision_free(px, py, x_grid, y_grid, blocked)
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

    left_score = path_evidence_score(result['left']['x'], result['left']['y'], x_grid, y_grid, hm_norm)
    right_score = path_evidence_score(result['right']['x'], result['right']['y'], x_grid, y_grid, hm_norm)

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
        if abs(float(r_left - r_right)) <= float(config.circle_radius_tie_mm):
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

    return {
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


def planner_to_curvature(planner):
    side = str(planner.get('chosen_side', 'straight'))
    radius = float(planner.get('chosen_radius_mm', np.inf))
    if not np.isfinite(radius) or radius <= 1e-6:
        return 0.0
    if side.startswith('left'):
        return 1.0 / radius
    if side.startswith('right'):
        return -1.0 / radius
    return 0.0
