import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_closing, binary_dilation, label


def world_to_pixel(x, y, meta):
    min_x = meta["arena_bounds_mm"]["min_x"]
    max_y = meta["arena_bounds_mm"]["max_y"]
    mm_per_px = float(meta["map_mm_per_px"])
    col = (np.asarray(x) - min_x) / mm_per_px - 0.5
    row = (max_y - np.asarray(y)) / mm_per_px - 0.5
    return col, row


def pixel_to_world(col, row, meta):
    min_x = meta["arena_bounds_mm"]["min_x"]
    max_y = meta["arena_bounds_mm"]["max_y"]
    mm_per_px = float(meta["map_mm_per_px"])
    x = min_x + (np.asarray(col) + 0.5) * mm_per_px
    y = max_y - (np.asarray(row) + 0.5) * mm_per_px
    return x, y


def segment_inside_region(wall_mask, seed_xy=None, close_iter=2, dilate_iter=2):
    mask = wall_mask.astype(bool)
    if close_iter and close_iter > 0:
        mask = binary_closing(mask, iterations=close_iter)
    if dilate_iter and dilate_iter > 0:
        mask = binary_dilation(mask, iterations=dilate_iter)

    free = ~mask
    labeled, num = label(free)
    if num == 0:
        raise ValueError("No free space detected in wall mask")

    if seed_xy is not None:
        h, w = mask.shape
        sx = int(np.clip(seed_xy[0], 0, w - 1))
        sy = int(np.clip(seed_xy[1], 0, h - 1))
        if labeled[sy, sx] == 0:
            ys, xs = np.nonzero(free)
            if xs.size == 0:
                raise ValueError("No free pixels found")
            d2 = (xs - sx) ** 2 + (ys - sy) ** 2
            idx = int(np.argmin(d2))
            sx = int(xs[idx])
            sy = int(ys[idx])
        main_label = labeled[sy, sx]
        inside = labeled == main_label
    else:
        border_labels = np.unique(
            np.concatenate(
                [labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]]
            )
        )
        outside = np.isin(labeled, border_labels)
        inside_candidates = free & ~outside
        if inside_candidates.any():
            labels_inside = labeled * inside_candidates
            label_ids, counts = np.unique(
                labels_inside[labels_inside > 0], return_counts=True
            )
            main_label = label_ids[np.argmax(counts)]
            inside = labeled == main_label
        else:
            inside = free

    return inside, mask


def compute_distance_field(wall_mask, seed_xy=None, close_iter=2, dilate_iter=2):
    inside, mask = segment_inside_region(
        wall_mask, seed_xy=seed_xy, close_iter=close_iter, dilate_iter=dilate_iter
    )
    dist = distance_transform_edt(inside).astype(np.float32)
    return dist, inside, mask


def compute_repulsive_field(dist):
    grad_y, grad_x = np.gradient(dist)
    mag = np.hypot(grad_x, grad_y)
    mag = np.maximum(mag, 1e-6)
    vx = grad_x / mag
    vy = grad_y / mag
    return vx, vy


def wrap_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def wall_vector_field_batch(
    x_vec,
    y_vec,
    yaw_deg_vec,
    wall_mask,
    meta,
    seed_xy=None,
    close_iter=2,
    dilate_iter=2,
    visualize=False,
    arrow_len=50.0,
    invert_y=False,
    title=None,
):
    x_arr = np.asarray(x_vec, dtype=float)
    y_arr = np.asarray(y_vec, dtype=float)
    yaw_arr = np.asarray(yaw_deg_vec, dtype=float)
    if not (x_arr.shape == y_arr.shape == yaw_arr.shape):
        raise ValueError("x_vec, y_vec, yaw_deg_vec must have the same shape")

    dist, inside, mask = compute_distance_field(
        wall_mask, seed_xy=seed_xy, close_iter=close_iter, dilate_iter=dilate_iter
    )
    vx, vy = compute_repulsive_field(dist)

    col, row = world_to_pixel(x_arr, y_arr, meta)
    h, w = dist.shape
    col_i = np.clip(np.round(col).astype(int), 0, w - 1)
    row_i = np.clip(np.round(row).astype(int), 0, h - 1)

    dir_x_img = vx[row_i, col_i]
    dir_y_img = vy[row_i, col_i]
    dir_x = dir_x_img
    dir_y = -dir_y_img

    yaw_des_deg = np.rad2deg(np.arctan2(dir_y, dir_x))
    yaw_err_deg = wrap_angle_deg(yaw_des_deg - yaw_arr)
    dist_mm = dist[row_i, col_i] * float(meta["map_mm_per_px"])

    rows = []
    for x, y, yaw_deg, dx, dy, yd, ye, dist_val in zip(
        x_arr, y_arr, yaw_arr, dir_x, dir_y, yaw_des_deg, yaw_err_deg, dist_mm
    ):
        rows.append(
            {
                "x": float(x),
                "y": float(y),
                "yaw_deg": float(yaw_deg),
                "yaw_des_deg": float(yd),
                "yaw_err_deg": float(ye),
                "dir_x": float(dx),
                "dir_y": float(dy),
                "dist_mm": float(dist_val),
            }
        )

    df = pd.DataFrame(rows)

    if visualize:
        plt.figure(figsize=(7, 7))
        plt.scatter(x_arr, y_arr, s=12, label="poses", zorder=5)
        yaw_rad = np.deg2rad(df["yaw_deg"].to_numpy(dtype=float))
        hx = np.cos(yaw_rad)
        hy = np.sin(yaw_rad)
        plt.quiver(
            df["x"],
            df["y"],
            hx,
            hy,
            angles="xy",
            scale_units="xy",
            scale=1.0 / (arrow_len * 0.8),
            width=0.003,
            color="tab:blue",
            label="current heading",
            zorder=3,
        )
        plt.quiver(
            df["x"],
            df["y"],
            df["dir_x"],
            df["dir_y"],
            angles="xy",
            scale_units="xy",
            scale=1.0 / arrow_len,
            width=0.004,
            color="tab:orange",
            label="desired heading",
            zorder=4,
        )
        if invert_y:
            plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title if title else "Wall repulsion field (poses)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df, dist, inside, mask
