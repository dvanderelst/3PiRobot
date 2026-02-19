import numpy as np

from Library import DataProcessor


def make_robot_frame_grid(extent_mm, grid_mm):
    x_min, x_max = -float(extent_mm), float(extent_mm)
    y_min, y_max = -float(extent_mm), float(extent_mm)
    x_grid = np.arange(x_min, x_max + float(grid_mm), float(grid_mm), dtype=np.float32)
    y_grid = np.arange(y_min, y_max + float(grid_mm), float(grid_mm), dtype=np.float32)
    xx, yy = np.meshgrid(x_grid, y_grid)
    return x_grid, y_grid, xx, yy


def smooth_heatmap(hm):
    k = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    k /= np.sum(k)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), axis=1, arr=hm)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), axis=0, arr=out)
    return out


def accumulate_segment(
    evidence,
    xx,
    yy,
    x_grid,
    y_grid,
    grid_mm,
    sigma_perp_mm,
    sigma_para_mm,
    p0,
    p1,
    seg_len,
    weight=1.0,
):
    tangent = (p1 - p0) / seg_len
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    mid = 0.5 * (p0 + p1)
    half_len = 0.5 * seg_len

    grid_x_min = float(x_grid[0])
    grid_y_min = float(y_grid[0])
    local_margin = 3.0 * max(float(sigma_perp_mm), float(sigma_para_mm))
    bx0 = max(float(x_grid[0]), min(p0[0], p1[0]) - local_margin)
    bx1 = min(float(x_grid[-1]), max(p0[0], p1[0]) + local_margin)
    by0 = max(float(y_grid[0]), min(p0[1], p1[1]) - local_margin)
    by1 = min(float(y_grid[-1]), max(p0[1], p1[1]) + local_margin)

    ix0 = int(max(0, np.floor((bx0 - grid_x_min) / float(grid_mm))))
    ix1 = int(min(len(x_grid) - 1, np.ceil((bx1 - grid_x_min) / float(grid_mm))))
    iy0 = int(max(0, np.floor((by0 - grid_y_min) / float(grid_mm))))
    iy1 = int(min(len(y_grid) - 1, np.ceil((by1 - grid_y_min) / float(grid_mm))))

    sub_x = xx[iy0:iy1 + 1, ix0:ix1 + 1]
    sub_y = yy[iy0:iy1 + 1, ix0:ix1 + 1]
    dx = sub_x - mid[0]
    dy = sub_y - mid[1]

    d_para = dx * tangent[0] + dy * tangent[1]
    d_perp = dx * normal[0] + dy * normal[1]
    outside_para = np.maximum(np.abs(d_para) - half_len, 0.0)

    kernel = np.exp(
        -0.5 * (d_perp / float(sigma_perp_mm)) ** 2
        -0.5 * (outside_para / float(sigma_para_mm)) ** 2
    )
    evidence[iy0:iy1 + 1, ix0:ix1 + 1] += (float(weight) * kernel).astype(np.float32)


def accumulate_point(
    evidence,
    xx,
    yy,
    x_grid,
    y_grid,
    grid_mm,
    sigma_mm,
    p,
    weight=1.0,
):
    grid_x_min = float(x_grid[0])
    grid_y_min = float(y_grid[0])
    local_margin = 3.0 * float(sigma_mm)
    bx0 = max(float(x_grid[0]), float(p[0]) - local_margin)
    bx1 = min(float(x_grid[-1]), float(p[0]) + local_margin)
    by0 = max(float(y_grid[0]), float(p[1]) - local_margin)
    by1 = min(float(y_grid[-1]), float(p[1]) + local_margin)

    ix0 = int(max(0, np.floor((bx0 - grid_x_min) / float(grid_mm))))
    ix1 = int(min(len(x_grid) - 1, np.ceil((bx1 - grid_x_min) / float(grid_mm))))
    iy0 = int(max(0, np.floor((by0 - grid_y_min) / float(grid_mm))))
    iy1 = int(min(len(y_grid) - 1, np.ceil((by1 - grid_y_min) / float(grid_mm))))

    sub_x = xx[iy0:iy1 + 1, ix0:ix1 + 1]
    sub_y = yy[iy0:iy1 + 1, ix0:ix1 + 1]
    dx = sub_x - float(p[0])
    dy = sub_y - float(p[1])

    kernel = np.exp(-0.5 * ((dx * dx + dy * dy) / (float(sigma_mm) ** 2)))
    evidence[iy0:iy1 + 1, ix0:ix1 + 1] += (float(weight) * kernel).astype(np.float32)


def _centers_to_edges_deg(centers_deg):
    centers_deg = np.asarray(centers_deg, dtype=np.float32)
    if centers_deg.size == 1:
        c = float(centers_deg[0])
        return np.asarray([c - 0.5, c + 0.5], dtype=np.float32)
    mids = 0.5 * (centers_deg[:-1] + centers_deg[1:])
    left = centers_deg[0] - 0.5 * (centers_deg[1] - centers_deg[0])
    right = centers_deg[-1] + 0.5 * (centers_deg[-1] - centers_deg[-2])
    return np.concatenate([[left], mids, [right]]).astype(np.float32)


def accumulate_annular_sector(
    evidence,
    xx,
    yy,
    x_grid,
    y_grid,
    grid_mm,
    rob_x,
    rob_y,
    rob_yaw_deg,
    anchor_x,
    anchor_y,
    anchor_yaw_deg,
    az_left_deg,
    az_right_deg,
    r_inner_mm,
    r_outer_mm,
    weight=1.0,
):
    # Conservative bounding box using outer arc endpoints + robot origin.
    az_samples = np.asarray([az_left_deg, az_right_deg], dtype=np.float32)
    dist_samples = np.asarray([r_outer_mm, r_outer_mm], dtype=np.float32)
    xw_o, yw_o = DataProcessor.robot2world(
        az_samples, dist_samples, float(rob_x), float(rob_y), float(rob_yaw_deg)
    )
    xr_o, yr_o = DataProcessor.world2robot(xw_o, yw_o, float(anchor_x), float(anchor_y), float(anchor_yaw_deg))
    x0r, y0r = DataProcessor.world2robot(
        np.asarray([float(rob_x)], dtype=np.float32),
        np.asarray([float(rob_y)], dtype=np.float32),
        float(anchor_x),
        float(anchor_y),
        float(anchor_yaw_deg),
    )
    x0 = float(x0r[0])
    y0 = float(y0r[0])

    bx0 = min(float(np.min(xr_o)), x0)
    bx1 = max(float(np.max(xr_o)), x0)
    by0 = min(float(np.min(yr_o)), y0)
    by1 = max(float(np.max(yr_o)), y0)
    margin = 2.0 * float(grid_mm)
    bx0 = max(float(x_grid[0]), bx0 - margin)
    bx1 = min(float(x_grid[-1]), bx1 + margin)
    by0 = max(float(y_grid[0]), by0 - margin)
    by1 = min(float(y_grid[-1]), by1 + margin)

    grid_x_min = float(x_grid[0])
    grid_y_min = float(y_grid[0])
    ix0 = int(max(0, np.floor((bx0 - grid_x_min) / float(grid_mm))))
    ix1 = int(min(len(x_grid) - 1, np.ceil((bx1 - grid_x_min) / float(grid_mm))))
    iy0 = int(max(0, np.floor((by0 - grid_y_min) / float(grid_mm))))
    iy1 = int(min(len(y_grid) - 1, np.ceil((by1 - grid_y_min) / float(grid_mm))))
    if ix1 < ix0 or iy1 < iy0:
        return

    sub_x = xx[iy0:iy1 + 1, ix0:ix1 + 1]
    sub_y = yy[iy0:iy1 + 1, ix0:ix1 + 1]

    # Convert anchor-frame Cartesian grid points -> world via polar conversion.
    # robot2world expects (azimuth, distance), not Cartesian coordinates.
    sub_az = np.rad2deg(np.arctan2(sub_y, sub_x)).reshape(-1)
    sub_dist = np.hypot(sub_x, sub_y).reshape(-1)
    xw, yw = DataProcessor.robot2world(
        sub_az,
        sub_dist,
        float(anchor_x),
        float(anchor_y),
        float(anchor_yaw_deg),
    )
    x_cur, y_cur = DataProcessor.world2robot(
        xw,
        yw,
        float(rob_x),
        float(rob_y),
        float(rob_yaw_deg),
    )
    x_cur = x_cur.reshape(sub_x.shape)
    y_cur = y_cur.reshape(sub_y.shape)

    r = np.hypot(x_cur, y_cur)
    az = np.rad2deg(np.arctan2(y_cur, x_cur))

    # Robust wrapped-angle inclusion for the azimuth interval.
    az_center = 0.5 * (float(az_left_deg) + float(az_right_deg))
    half_width = 0.5 * ((float(az_right_deg) - float(az_left_deg)) % 360.0)
    if half_width > 180.0:
        half_width = 360.0 - half_width
    d_az = (az - az_center + 180.0) % 360.0 - 180.0

    in_sector = (
        (r >= float(r_inner_mm))
        & (r <= float(r_outer_mm))
        & (np.abs(d_az) <= half_width + 1e-6)
    )
    if np.any(in_sector):
        evidence[iy0:iy1 + 1, ix0:ix1 + 1][in_sector] += float(weight)


def build_robot_frame_evidence(
    profile_centers_deg_seq,
    distance_mm_seq,
    presence_probs_seq,
    presence_bin_seq,
    rob_x_seq,
    rob_y_seq,
    rob_yaw_deg_seq,
    x_grid,
    y_grid,
    xx,
    yy,
    grid_mm,
    sigma_perp_mm,
    sigma_para_mm,
    apply_smoothing=False,
):
    profile_centers_deg_seq = np.asarray(profile_centers_deg_seq, dtype=np.float32)
    distance_mm_seq = np.asarray(distance_mm_seq, dtype=np.float32)
    presence_probs_seq = np.asarray(presence_probs_seq, dtype=np.float32)
    presence_bin_seq = np.asarray(presence_bin_seq, dtype=np.uint8)
    rob_x_seq = np.asarray(rob_x_seq, dtype=np.float32)
    rob_y_seq = np.asarray(rob_y_seq, dtype=np.float32)
    rob_yaw_deg_seq = np.asarray(rob_yaw_deg_seq, dtype=np.float32)

    n = distance_mm_seq.shape[0]
    anchor_x = float(rob_x_seq[-1])
    anchor_y = float(rob_y_seq[-1])
    anchor_yaw = float(rob_yaw_deg_seq[-1])
    evidence = np.zeros_like(xx, dtype=np.float32)

    for i in range(n):
        az_deg = profile_centers_deg_seq[i]
        dist_i = distance_mm_seq[i].copy()
        dist_i[presence_bin_seq[i] == 0] = np.nan
        probs_i = presence_probs_seq[i]

        xw, yw = DataProcessor.robot2world(az_deg, dist_i, float(rob_x_seq[i]), float(rob_y_seq[i]), float(rob_yaw_deg_seq[i]))
        xr, yr = DataProcessor.world2robot(xw, yw, anchor_x, anchor_y, anchor_yaw)

        for j in range(len(az_deg) - 1):
            if not (np.isfinite(xr[j]) and np.isfinite(yr[j]) and np.isfinite(xr[j + 1]) and np.isfinite(yr[j + 1])):
                continue
            p0 = np.array([xr[j], yr[j]], dtype=float)
            p1 = np.array([xr[j + 1], yr[j + 1]], dtype=float)
            seg_vec = p1 - p0
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len < 1e-6:
                continue
            weight = float(0.5 * (probs_i[j] + probs_i[j + 1]))
            if weight <= 0.0:
                continue
            accumulate_segment(
                evidence=evidence,
                xx=xx,
                yy=yy,
                x_grid=x_grid,
                y_grid=y_grid,
                grid_mm=grid_mm,
                sigma_perp_mm=sigma_perp_mm,
                sigma_para_mm=sigma_para_mm,
                p0=p0,
                p1=p1,
                seg_len=seg_len,
                weight=weight,
            )

    evidence_norm = evidence / max(n, 1)
    if apply_smoothing:
        evidence_norm = smooth_heatmap(evidence_norm)
    max_v = float(np.max(evidence_norm))
    if max_v > 1e-12:
        evidence_norm = evidence_norm / max_v
    return evidence_norm.astype(np.float32)


def build_robot_frame_evidence_from_radial(
    profile_centers_deg_seq,
    radial_probs_seq,
    range_centers_mm,
    presence_probs_seq,
    rob_x_seq,
    rob_y_seq,
    rob_yaw_deg_seq,
    x_grid,
    y_grid,
    xx,
    yy,
    grid_mm,
    sigma_point_mm,
    min_point_weight=1e-4,
    apply_smoothing=False,
):
    profile_centers_deg_seq = np.asarray(profile_centers_deg_seq, dtype=np.float32)
    radial_probs_seq = np.asarray(radial_probs_seq, dtype=np.float32)
    range_centers_mm = np.asarray(range_centers_mm, dtype=np.float32)
    presence_probs_seq = np.asarray(presence_probs_seq, dtype=np.float32)
    rob_x_seq = np.asarray(rob_x_seq, dtype=np.float32)
    rob_y_seq = np.asarray(rob_y_seq, dtype=np.float32)
    rob_yaw_deg_seq = np.asarray(rob_yaw_deg_seq, dtype=np.float32)

    n = radial_probs_seq.shape[0]
    anchor_x = float(rob_x_seq[-1])
    anchor_y = float(rob_y_seq[-1])
    anchor_yaw = float(rob_yaw_deg_seq[-1])
    evidence = np.zeros_like(xx, dtype=np.float32)

    range_edges = _centers_to_edges_deg(range_centers_mm.astype(np.float32))
    # _centers_to_edges_deg is angle-centric; reuse logic numerically for distances:
    # values are still monotonic centers -> monotonic edges.
    range_edges = np.asarray(range_edges, dtype=np.float32)
    if range_edges.size != range_centers_mm.size + 1:
        raise ValueError('range_edges size mismatch in radial occupancy integration.')

    for i in range(n):
        az_deg = profile_centers_deg_seq[i]  # (K,)
        az_edges = _centers_to_edges_deg(az_deg)  # (K+1,)
        probs_kr = radial_probs_seq[i]       # (K,R)
        presence_k = presence_probs_seq[i]   # (K,)

        for k in range(len(az_deg)):
            p_pres = float(presence_k[k])
            if p_pres <= 0.0:
                continue

            weights_r = p_pres * probs_kr[k]
            valid_r = weights_r > float(min_point_weight)
            if not np.any(valid_r):
                continue

            az_left = float(az_edges[k])
            az_right = float(az_edges[k + 1])
            dist_vec = range_centers_mm[valid_r]
            weight_vec = weights_r[valid_r]

            for r_idx in range(dist_vec.size):
                dist = float(dist_vec[r_idx])
                w = float(weight_vec[r_idx])
                if w <= 0.0:
                    continue
                ridx_full = np.where(valid_r)[0][r_idx]
                r_inner = float(max(0.0, range_edges[ridx_full]))
                r_outer = float(max(r_inner, range_edges[ridx_full + 1]))
                if r_outer <= r_inner + 1e-9:
                    continue

                accumulate_annular_sector(
                    evidence=evidence,
                    xx=xx,
                    yy=yy,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    grid_mm=grid_mm,
                    rob_x=float(rob_x_seq[i]),
                    rob_y=float(rob_y_seq[i]),
                    rob_yaw_deg=float(rob_yaw_deg_seq[i]),
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                    anchor_yaw_deg=anchor_yaw,
                    az_left_deg=az_left,
                    az_right_deg=az_right,
                    r_inner_mm=r_inner,
                    r_outer_mm=r_outer,
                    weight=w,
                )

    evidence_norm = evidence / max(n, 1)
    if apply_smoothing:
        evidence_norm = smooth_heatmap(evidence_norm)
    max_v = float(np.max(evidence_norm))
    if max_v > 1e-12:
        evidence_norm = evidence_norm / max_v
    return evidence_norm.astype(np.float32)
