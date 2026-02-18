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
