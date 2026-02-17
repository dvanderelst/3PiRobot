import json
import os
from glob import glob

import numpy as np
from matplotlib import pyplot as plt


# ============================================
# CONFIGURATION
# ============================================
series_npz_path = None  # If None, newest *_robotframe_series_data.npz is used.
output_dir = 'Prediction/PathPlanningExamples'

# Select frames by series index (index in heatmap stack). Use either explicit list or range.
selected_series_indices = None
selected_series_range = (20, 50, 1)  # (start, stop, step)

apply_heatmap_smoothing = False

# Occupancy threshold for obstacle extraction.
occ_block_threshold = 0.10

# Circle planner parameters.
robot_radius_mm = 80.0
safety_margin_mm = 120.0
circle_radius_min_mm = 250.0
circle_radius_max_mm = 2500.0
circle_radius_step_mm = 50.0
circle_arc_samples = 220
circle_horizon_x_mm = 1800.0
circle_radius_tie_mm = 100.0

speed_min = 0.10
speed_max = 1.00
clearance_slow_mm = 250.0
clearance_fast_mm = 1500.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_default_series_file():
    candidates = glob('Prediction/heatmap_series_robot/*_robotframe_series_data.npz')
    if not candidates:
        raise FileNotFoundError(
            "No robot-frame series data found. Run SCRIPT_PredictProfiles.py first."
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def resolve_indices(n_samples, idx_list, idx_range):
    selected = []
    if idx_list is not None:
        selected.extend(idx_list)
    if idx_range is not None:
        if len(idx_range) != 3:
            raise ValueError('selected_series_range must be (start, stop, step).')
        start, stop, step = idx_range
        selected.extend(list(range(int(start), int(stop), int(step))))

    if not selected:
        selected = list(np.linspace(0, n_samples - 1, num=min(6, n_samples), dtype=int))

    uniq = sorted(set(int(i) for i in selected if 0 <= int(i) < n_samples))
    return uniq


def smooth_heatmap(hm):
    # Lightweight separable smoothing without scipy dependency.
    k = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    k /= np.sum(k)

    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), axis=1, arr=hm)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), axis=0, arr=out)
    return out


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


def sample_tangent_circle_path(radius_mm, side_sign, horizon_x_mm, n_samples):
    # Circle center at (0, side_sign*R), path passes through (0,0) with heading +x.
    # Parametrization: x = R*sin(phi), y = side_sign*R*(1-cos(phi)).
    if radius_mm <= 1e-6:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    phi_max = np.arcsin(np.clip(horizon_x_mm / radius_mm, 0.0, 1.0))
    if phi_max <= 1e-6:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    phi = np.linspace(0.0, phi_max, n_samples, dtype=np.float32)
    x = radius_mm * np.sin(phi)
    y = side_sign * radius_mm * (1.0 - np.cos(phi))
    return x.astype(np.float32), y.astype(np.float32)


def is_path_collision_free(path_x, path_y, x_grid, y_grid, blocked_mask):
    if len(path_x) == 0:
        return False
    h, w = blocked_mask.shape
    for x, y in zip(path_x, path_y):
        ix = int(np.argmin(np.abs(x_grid - x)))
        iy = int(np.argmin(np.abs(y_grid - y)))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return False
        if blocked_mask[iy, ix]:
            return False
    return True


def path_evidence_score(path_x, path_y, x_grid, y_grid, hm_norm):
    if path_x is None or len(path_x) == 0:
        return np.inf
    vals = []
    h, w = hm_norm.shape
    for x, y in zip(path_x, path_y):
        ix = int(np.argmin(np.abs(x_grid - x)))
        iy = int(np.argmin(np.abs(y_grid - y)))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            continue
        vals.append(float(hm_norm[iy, ix]))
    if len(vals) == 0:
        return np.inf
    return float(np.mean(vals))


def plan_path_circles(heatmap, x_grid, y_grid):
    hm = heatmap.astype(np.float32)
    if apply_heatmap_smoothing:
        hm = smooth_heatmap(hm)
    hm = hm - np.min(hm)
    hm = hm / (np.max(hm) + 1e-12)

    wall_mask = hm >= occ_block_threshold
    grid_mm = float(np.mean(np.diff(x_grid))) if len(x_grid) > 1 else 20.0
    inflate_px = int(np.ceil((robot_radius_mm + safety_margin_mm) / max(grid_mm, 1e-6)))
    blocked = dilate_binary(wall_mask, inflate_px)

    radii = np.arange(circle_radius_min_mm, circle_radius_max_mm + circle_radius_step_mm, circle_radius_step_mm, dtype=np.float32)
    result = {}
    for side_name, side_sign in [('left', 1.0), ('right', -1.0)]:
        best_r = None
        best_xy = (None, None)
        touched_obstacle = False
        for r in radii:
            px, py = sample_tangent_circle_path(
                radius_mm=float(r),
                side_sign=side_sign,
                horizon_x_mm=float(circle_horizon_x_mm),
                n_samples=int(circle_arc_samples),
            )
            ok = is_path_collision_free(px, py, x_grid, y_grid, blocked)
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
            'sign': side_sign,
            'touched_obstacle': touched_obstacle,
        }

    left_score = path_evidence_score(result['left']['x'], result['left']['y'], x_grid, y_grid, hm)
    right_score = path_evidence_score(result['right']['x'], result['right']['y'], x_grid, y_grid, hm)

    # Choose larger feasible radius; for near-ties, choose lower wall evidence along arc.
    r_left = result['left']['radius'] if result['left']['radius'] is not None else -np.inf
    r_right = result['right']['radius'] if result['right']['radius'] is not None else -np.inf
    left_never_touched = not result['left']['touched_obstacle']
    right_never_touched = not result['right']['touched_obstacle']

    if (r_left < 0) and (r_right < 0):
        path_x = np.linspace(0.0, circle_horizon_x_mm, int(circle_arc_samples), dtype=np.float32)
        path_y = np.zeros_like(path_x, dtype=np.float32)
        chosen_side = 'straight'
        chosen_radius = np.inf
    elif left_never_touched and right_never_touched:
        # Open space on both sides: keep going straight.
        path_x = np.linspace(0.0, circle_horizon_x_mm, int(circle_arc_samples), dtype=np.float32)
        path_y = np.zeros_like(path_x, dtype=np.float32)
        chosen_side = 'straight_open'
        chosen_radius = np.inf
    else:
        radius_gap = abs(float(r_left - r_right))
        if radius_gap <= float(circle_radius_tie_mm):
            # Evidence-based tie-break for ambiguous radius decisions.
            if left_score <= right_score:
                path_x = result['left']['x']
                path_y = result['left']['y']
                chosen_side = 'left'
                chosen_radius = r_left
            else:
                path_x = result['right']['x']
                path_y = result['right']['y']
                chosen_side = 'right'
                chosen_radius = r_right
        elif r_left > r_right:
            path_x = result['left']['x']
            path_y = result['left']['y']
            chosen_side = 'left'
            chosen_radius = r_left
        else:
            path_x = result['right']['x']
            path_y = result['right']['y']
            chosen_side = 'right'
            chosen_radius = r_right

    # Confidence: lower near blocked cells (sampled from non-dilated map).
    conf = []
    for x, y in zip(path_x, path_y):
        ix = int(np.argmin(np.abs(x_grid - x)))
        iy = int(np.argmin(np.abs(y_grid - y)))
        conf.append(1.0 - float(hm[iy, ix]))
    conf = np.asarray(conf, dtype=np.float32)

    extras = {
        'blocked_mask': blocked,
        'left_radius_mm': None if result['left']['radius'] is None else float(result['left']['radius']),
        'right_radius_mm': None if result['right']['radius'] is None else float(result['right']['radius']),
        'left_touched_obstacle': bool(result['left']['touched_obstacle']),
        'right_touched_obstacle': bool(result['right']['touched_obstacle']),
        'left_evidence_score': float(left_score) if np.isfinite(left_score) else float('nan'),
        'right_evidence_score': float(right_score) if np.isfinite(right_score) else float('nan'),
        'chosen_side': chosen_side,
        'chosen_radius_mm': float(chosen_radius) if np.isfinite(chosen_radius) else float('inf'),
        'left_x': result['left']['x'],
        'left_y': result['left']['y'],
        'right_x': result['right']['x'],
        'right_y': result['right']['y'],
    }
    return path_x.astype(np.float32), path_y.astype(np.float32), conf, hm, extras


def enforce_origin_anchor(x_points, y_points, conf_points):
    """Ensure path includes robot origin in robot frame."""
    if len(x_points) == 0:
        return (
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        )

    if np.isclose(x_points[0], 0.0):
        y_points = y_points.copy()
        y_points[0] = 0.0
        return x_points.astype(np.float32), y_points.astype(np.float32), conf_points.astype(np.float32)

    x_aug = np.concatenate([np.asarray([0.0], dtype=np.float32), x_points.astype(np.float32)])
    y_aug = np.concatenate([np.asarray([0.0], dtype=np.float32), y_points.astype(np.float32)])
    c0 = conf_points[0] if len(conf_points) > 0 else 1.0
    c_aug = np.concatenate([np.asarray([c0], dtype=np.float32), conf_points.astype(np.float32)])
    return x_aug, y_aug, c_aug


def compute_forward_clearance_mm(hm_norm, x_grid, y_grid):
    x_mask = x_grid > 0.0
    y_mask = np.abs(y_grid) <= 220.0
    if not np.any(x_mask) or not np.any(y_mask):
        return np.nan

    sub = hm_norm[np.ix_(y_mask, x_mask)]
    xx = x_grid[x_mask]

    # Dynamic threshold from front-sector intensity.
    thr = max(0.25, float(np.percentile(sub, 85)))
    occupied = sub >= thr
    if not np.any(occupied):
        return float(clearance_fast_mm)

    x_occ = np.where(np.any(occupied, axis=0))[0]
    if len(x_occ) == 0:
        return float(clearance_fast_mm)

    return float(xx[x_occ[0]])


def circle_turn_and_speed(planner_extras, hm_norm, x_grid, y_grid):
    side = str(planner_extras.get('chosen_side', 'straight'))
    radius = float(planner_extras.get('chosen_radius_mm', np.inf))

    if side.startswith('left') and np.isfinite(radius) and radius > 1e-6:
        turn_direction = 1
        curvature_inv_mm = 1.0 / radius
    elif side.startswith('right') and np.isfinite(radius) and radius > 1e-6:
        turn_direction = -1
        curvature_inv_mm = -1.0 / radius
    else:
        turn_direction = 0
        curvature_inv_mm = 0.0

    max_curvature = 1.0 / max(circle_radius_min_mm, 1e-6)
    turn_strength = float(np.clip(abs(curvature_inv_mm) / max_curvature, 0.0, 1.0))

    clearance = compute_forward_clearance_mm(hm_norm, x_grid, y_grid)
    if not np.isfinite(clearance):
        speed = speed_min
    else:
        alpha = (clearance - clearance_slow_mm) / max(clearance_fast_mm - clearance_slow_mm, 1e-6)
        speed = float(np.clip(alpha, speed_min, speed_max))

    return {
        'turn_direction': int(turn_direction),   # left=+1, straight=0, right=-1
        'curvature_inv_mm': float(curvature_inv_mm),
        'turn_strength': float(turn_strength),
        'forward_clearance_mm': float(clearance),
        'speed_cmd': float(speed),
    }


def plot_example(
    out_path,
    title,
    hm_norm,
    x_grid,
    y_grid,
    x_points,
    y_path,
    control,
    planner_extras=None,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(
        hm_norm,
        origin='lower',
        extent=[float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1])],
        cmap='magma',
        aspect='equal'
    )
    fig.colorbar(im, ax=ax, label='Wall Evidence (norm)')

    ax.scatter([0.0], [0.0], c='cyan', s=45, label='Robot (anchor)')
    ax.arrow(0.0, 0.0, 200.0, 0.0, head_width=70, head_length=90, fc='cyan', ec='cyan')

    if len(x_points) > 0:
        ax.plot(x_points, y_path, 'o', ms=4, color='lime', label='Path samples')

    # Overlay blocked occupancy to verify path stays in free space.
    xx, yy = np.meshgrid(x_grid, y_grid)
    blocked = hm_norm >= occ_block_threshold
    if planner_extras is not None and ('blocked_mask' in planner_extras):
        blocked = planner_extras['blocked_mask']
    ax.contourf(
        xx,
        yy,
        blocked.astype(float),
        levels=[0.5, 1.5],
        colors=['red'],
        alpha=0.16
    )

    # For circle mode, show both candidate arcs.
    if planner_extras is not None and ('left_x' in planner_extras):
        if planner_extras['left_x'] is not None:
            ax.plot(planner_extras['left_x'], planner_extras['left_y'], '--', color='deepskyblue', lw=1.4, alpha=0.8, label='Left candidate')
        if planner_extras['right_x'] is not None:
            ax.plot(planner_extras['right_x'], planner_extras['right_y'], '--', color='springgreen', lw=1.4, alpha=0.8, label='Right candidate')

    if len(x_points) > 1:
        ax.plot(x_points, y_path, '-', color='white', lw=2.0, label='Chosen circle arc')

    txt = (
        f"turn_dir={control['turn_direction']:+d}\n"
        f"curv={control['curvature_inv_mm']:+.5f} 1/mm\n"
        f"turn_strength={control['turn_strength']:.3f}\n"
        f"speed={control['speed_cmd']:.3f}\n"
        f"clearance={control['forward_clearance_mm']:.0f} mm"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=10, bbox=dict(facecolor='black', alpha=0.45, edgecolor='none'))

    ax.set_xlim(0.0, circle_horizon_x_mm + 200.0)
    ax.set_ylim(float(y_grid[0]), float(y_grid[-1]))
    ax.set_xlabel('Robot X (mm)')
    ax.set_ylabel('Robot Y (mm)')
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_dir(output_dir)

    npz_path = series_npz_path if series_npz_path else find_default_series_file()
    data = np.load(npz_path)

    heatmaps = data['heatmaps']
    x_grid = data['x_grid']
    y_grid = data['y_grid']
    anchor_idx = data['anchor_index']
    chunk_start = data['window_start']
    chunk_end = data['window_end']
    anchor_x = data['anchor_rob_x']
    anchor_y = data['anchor_rob_y']
    anchor_yaw = data['anchor_rob_yaw_deg']

    n = heatmaps.shape[0]
    selected = resolve_indices(n, selected_series_indices, selected_series_range)

    print(f'Loaded: {npz_path}')
    print(f'Heatmaps: {n}, selected for examples: {selected}')

    rows = []

    for sidx in selected:
        hm = heatmaps[sidx]
        x_pts, y_path, conf, hm_norm, planner_extras = plan_path_circles(hm, x_grid, y_grid)
        x_pts, y_path, conf = enforce_origin_anchor(x_pts, y_path, conf)
        control = circle_turn_and_speed(planner_extras, hm_norm, x_grid, y_grid)

        out_png = os.path.join(output_dir, f'seriesidx_{sidx:04d}_pathplan.png')
        title = (
            f'series_idx={sidx} | chunk=[{int(chunk_start[sidx])},{int(chunk_end[sidx])}] | '
            f'anchor_idx={int(anchor_idx[sidx])}'
        )
        plot_example(out_png, title, hm_norm, x_grid, y_grid, x_pts, y_path, control, planner_extras=planner_extras)

        row = {
            'series_idx': int(sidx),
            'window_start': int(chunk_start[sidx]),
            'window_end': int(chunk_end[sidx]),
            'anchor_index': int(anchor_idx[sidx]),
            'anchor_rob_x': float(anchor_x[sidx]),
            'anchor_rob_y': float(anchor_y[sidx]),
            'anchor_rob_yaw_deg': float(anchor_yaw[sidx]),
            'n_path_samples': int(len(x_pts)),
            'mean_path_confidence': float(np.mean(conf)) if len(conf) > 0 else float('nan'),
            'planner_mode': 'circle',
            'chosen_side': planner_extras.get('chosen_side', 'n/a'),
            'chosen_radius_mm': float(planner_extras.get('chosen_radius_mm', np.nan)),
            'left_radius_mm': float(planner_extras['left_radius_mm']) if planner_extras.get('left_radius_mm') is not None else float('nan'),
            'right_radius_mm': float(planner_extras['right_radius_mm']) if planner_extras.get('right_radius_mm') is not None else float('nan'),
            'left_evidence_score': float(planner_extras.get('left_evidence_score', np.nan)),
            'right_evidence_score': float(planner_extras.get('right_evidence_score', np.nan)),
            'turn_direction': int(control['turn_direction']),
            'curvature_inv_mm': float(control['curvature_inv_mm']),
            'turn_strength': float(control['turn_strength']),
            'speed_cmd': float(control['speed_cmd']),
            'forward_clearance_mm': float(control['forward_clearance_mm']),
            'plot_file': out_png,
        }
        rows.append(row)
        print(
            f"series_idx={sidx:4d} | side={row['chosen_side']:>8s} | R={row['chosen_radius_mm']:.0f} | "
            f"turn_dir={row['turn_direction']:+d} | kappa={row['curvature_inv_mm']:+.5f} | speed={row['speed_cmd']:.3f}"
        )

    summary_json = {
        'source_npz': npz_path,
        'n_heatmaps_total': int(n),
        'selected_series_indices': [int(i) for i in selected],
        'planner_mode': 'circle',
        'results': rows,
    }

    summary_path = os.path.join(output_dir, 'path_planning_examples_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_json, f, indent=2)

    csv_path = os.path.join(output_dir, 'path_planning_examples_summary.csv')
    if rows:
        keys = [
            'series_idx', 'window_start', 'window_end', 'anchor_index',
            'anchor_rob_x', 'anchor_rob_y', 'anchor_rob_yaw_deg',
            'n_path_samples', 'mean_path_confidence',
            'planner_mode', 'chosen_side', 'chosen_radius_mm', 'left_radius_mm', 'right_radius_mm',
            'left_evidence_score', 'right_evidence_score',
            'turn_direction', 'curvature_inv_mm', 'turn_strength', 'speed_cmd',
            'forward_clearance_mm', 'plot_file'
        ]
        with open(csv_path, 'w') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                vals = []
                for k in keys:
                    v = r[k]
                    if isinstance(v, str):
                        vals.append(v)
                    else:
                        vals.append(str(v))
                f.write(','.join(vals) + '\n')

    print(f'Saved summary JSON: {summary_path}')
    print(f'Saved summary CSV:  {csv_path}')
    print(f'Saved example plots to: {output_dir}')


if __name__ == '__main__':
    main()
