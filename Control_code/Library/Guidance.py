import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_guidance_field(processor, d=500, plots=False):
    polyline = build_polyline_from_xy(processor.path_x, processor.path_y, plot=plots)
    x = processor.rob_x
    y = processor.rob_y
    yaw_deg = processor.rob_yaw_deg
    df = guidance_vector_field_batch(x, y, yaw_deg, polyline, d, visualize=plots)
    return df

def build_polyline_from_xy(
    x,
    y,
    closed=True,
    densify_step=None,    # e.g. 10.0 pixels; None = no densification
    sort_method="angle", # "angle" or "none"
    plot=False,
    plot_equal=True
):
    """
    Build an ordered polyline from x/y point coordinates.

    Parameters
    ----------
    x, y : array-like
        Coordinates of path points (same length).
    closed : bool
        Whether to treat the polyline as a closed loop.
    densify_step : float or None
        If given, resample segments so points are ~densify_step apart.
    sort_method : str
        "angle" = sort points around centroid (recommended for loops)
        "none"  = assume points already ordered
    plot : bool
        If True, plot original points, ordered polyline, and densified polyline.
    plot_equal : bool
        If True, enforce equal aspect ratio in plot.

    Returns
    -------
    polyline : ndarray, shape (N, 2)
        Ordered (and optionally densified) polyline points.
    """

    pts_raw = np.column_stack([x, y]).astype(float)

    if pts_raw.shape[0] < 2:
        raise ValueError("Need at least two points to build a polyline")

    pts = pts_raw.copy()

    # ---- 1. Order points into a loop ----
    if sort_method == "angle":
        c = pts.mean(axis=0)
        angles = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])
        order = np.argsort(angles)
        pts = pts[order]
    elif sort_method != "none":
        raise ValueError("sort_method must be 'angle' or 'none'")

    pts_ordered = pts.copy()

    # ---- 2. Densify if requested ----
    if densify_step is not None and densify_step > 0:
        dense = []
        n = len(pts)
        last = n if not closed else n + 1

        for i in range(1, last):
            a = pts[i-1]
            b = pts[i % n]
            v = b - a
            L = np.linalg.norm(v)

            if L < 1e-9:
                continue

            dense.append(a)
            k = int(np.floor(L / densify_step))
            for j in range(1, k + 1):
                dense.append(a + (j * densify_step / L) * v)

        pts = np.array(dense, dtype=float)

    polyline = pts

    # ---- 3. Optional plot ----
    if plot:
        plt.figure(figsize=(6, 6))

        # original points
        plt.scatter(
            pts_raw[:,0], pts_raw[:,1],
            c="red", s=50, label="original points", zorder=3
        )

        # ordered polyline
        xo, yo = pts_ordered[:,0], pts_ordered[:,1]
        if closed:
            xo = np.r_[xo, xo[0]]
            yo = np.r_[yo, yo[0]]
        plt.plot(
            xo, yo,
            "k--", linewidth=1.5, label="ordered polyline"
        )

        # densified polyline
        if densify_step is not None and densify_step > 0:
            xd, yd = polyline[:,0], polyline[:,1]
            if closed:
                xd = np.r_[xd, xd[0]]
                yd = np.r_[yd, yd[0]]
            plt.plot(
                xd, yd,
                "b-", linewidth=2, label="densified polyline"
            )

        if plot_equal:
            plt.axis("equal")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("Polyline construction")
        plt.tight_layout()
        plt.show()

    return polyline

import numpy as np

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def guidance_vector_field(
    x, y, yaw_deg,
    polyline_xy,
    d0,
    closed=True,
    d_min=1e-6,
    align_tangent_to_yaw=True
):
    """
    Vector-field guidance toward a (closed) polyline.

    Parameters
    ----------
    x, y : float
        Robot position in same coordinate system as polyline.
    yaw_deg : float
        Robot orientation.
    polyline_xy : ndarray (N,2)
        Ordered (optionally densified) path points.
    d0 : float
        Capture distance (same units as x,y). Bigger => smoother / more tangent-following.
    closed : bool
        Treat path as closed loop (connect last -> first).
    d_min : float
        Threshold for treating robot as "on path" to avoid division by ~0.
    align_tangent_to_yaw : bool
        If True, flip the segment tangent so it best matches the robot's current heading.

    Returns
    -------
    yaw_des : float
        Desired heading (deg).
    yaw_err : float
        Wrapped heading error yaw_des - yaw in degrees.
    dir_xy : tuple(float,float)
        Unit desired direction vector (dx, dy).
    closest_xy : tuple(float,float)
        Closest point on the polyline.
    dist_to_path : float
        Euclidean distance to path.
    """

    pts = np.asarray(polyline_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("polyline_xy must have shape (N,2) with N>=2")

    r = np.array([x, y], dtype=float)

    # Build segments A->B
    if closed:
        A = pts
        B = np.vstack([pts[1:], pts[:1]])
    else:
        A = pts[:-1]
        B = pts[1:]

    v = B - A
    vv = np.sum(v*v, axis=1)
    vv = np.where(vv < 1e-12, 1e-12, vv)

    w = r - A
    t = np.sum(w*v, axis=1) / vv
    t = np.clip(t, 0.0, 1.0)

    Q = A + t[:, None] * v
    diff = Q - r
    dist2 = np.sum(diff*diff, axis=1)
    k = int(np.argmin(dist2))

    q = Q[k]
    dist = float(np.sqrt(dist2[k]))

    # Tangent from chosen segment
    seg = v[k]
    seg_norm = np.linalg.norm(seg)
    if seg_norm < 1e-12:
        t_hat = np.array([1.0, 0.0])
    else:
        t_hat = seg / seg_norm

    if align_tangent_to_yaw:
        h = np.array([np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))])
        if np.dot(t_hat, h) < 0:
            t_hat = -t_hat

    # Blend tangent with "to-path" direction
    if dist < d_min:
        u = t_hat
    else:
        e_hat = (q - r) / dist
        alpha = dist / (dist + d0)  # 0 near path, ->1 far away
        u = (1.0 - alpha) * t_hat + alpha * e_hat  # convex blend

        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            u = t_hat
        else:
            u = u / u_norm

    yaw_rad = np.deg2rad(yaw_deg)

    yaw_des_rad = float(np.arctan2(u[1], u[0]))
    yaw_err_rad = float(wrap_to_pi(yaw_des_rad - yaw_rad))

    yaw_des_deg = float(np.rad2deg(yaw_des_rad))
    yaw_err_deg = float(np.rad2deg(yaw_err_rad))

    dir_xy = (float(u[0]), float(u[1]))
    closest_xy = (float(q[0]), float(q[1]))

    return yaw_des_deg, yaw_err_deg, dir_xy, closest_xy, dist



def visualize_guidance_result(
    x, y, yaw_deg,
    polyline_xy,
    guidance_result,
    *,
    degrees=True,
    arrow_len=50.0,          # in same units as x,y (pixels if working in image coords)
    show_closest=True,
    show_cross_track=True,
    show_next_step=True,
    invert_y=False,          # set True if your y-axis is image row coords (downwards)
    title=None
):
    """
    Visualize robot pose + path + guidance output.

    Parameters
    ----------
    x, y : float
        Robot position.
    yaw_deg : float
        Robot yaw (degrees if degrees=True, otherwise radians).
    polyline_xy : ndarray (N,2)
        Path polyline points.
    guidance_result : tuple
        Output from PathProcessing.guidance_vector_field.
        Expected (common) formats:
          A) (yaw_des, yaw_err, (dx,dy), (qx,qy), dist)
          B) (yaw_des, yaw_err)  # minimal
        where yaw_des/yaw_err are degrees if degrees=True (or radians if degrees=False).
    degrees : bool
        Whether angles are in degrees (True) or radians (False).
    arrow_len : float
        Length of heading arrows.
    invert_y : bool
        If True, invert y-axis (useful when plotting on top of images where y increases downward).
    """

    pts = np.asarray(polyline_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("polyline_xy must have shape (N,2)")

    # ---- Unpack guidance_result robustly ----
    yaw_des = None
    yaw_err = None
    dir_xy = None
    closest_xy = None
    dist = None

    if isinstance(guidance_result, (list, tuple)) and len(guidance_result) >= 2:
        yaw_des = guidance_result[0]
        yaw_err = guidance_result[1]
        if len(guidance_result) >= 3: dir_xy = guidance_result[2]
        if len(guidance_result) >= 4: closest_xy = guidance_result[3]
        if len(guidance_result) >= 5: dist = guidance_result[4]
    else:
        raise ValueError("guidance_result must be a tuple/list with at least (yaw_des, yaw_err)")

    # Convert angles to radians for plotting vectors
    if degrees:
        yaw_rad = np.deg2rad(yaw_deg)
        yaw_des_rad = np.deg2rad(yaw_des)
    else:
        yaw_rad = yaw_deg
        yaw_des_rad = yaw_des

    # Current heading vector
    hx, hy = np.cos(yaw_rad), np.sin(yaw_rad)
    # Desired heading vector
    dx, dy = np.cos(yaw_des_rad), np.sin(yaw_des_rad)

    # If guidance already returned direction vector, use it (can differ slightly if you had custom logic)
    if dir_xy is not None:
        dx, dy = float(dir_xy[0]), float(dir_xy[1])
        n = np.hypot(dx, dy)
        if n > 1e-12:
            dx, dy = dx / n, dy / n

    # ---- Plot ----
    plt.figure(figsize=(7, 7))
    plt.plot(pts[:,0], pts[:,1], '-', linewidth=2, label='polyline')

    # Robot position
    plt.scatter([x], [y], s=20, label='robot', zorder=5)

    # Current heading arrow
    plt.arrow(
        x, y, arrow_len * hx, arrow_len * hy,
        length_includes_head=True, head_width=arrow_len * 0.15,
        linewidth=2, label='current heading'
    )

    # Desired heading arrow
    plt.arrow(
        x, y, arrow_len * dx, arrow_len * dy,
        length_includes_head=True, head_width=arrow_len * 0.15,
        linewidth=2, label='desired heading'
    )

    # Closest point + cross-track vector
    if show_closest and closest_xy is not None:
        qx, qy = closest_xy
        plt.scatter([qx], [qy], s=20, label='closest point', zorder=6)

        if show_cross_track:
            plt.plot([x, qx], [y, qy], '--', linewidth=2, label='cross-track')

    # Next-step point (where you'd go if you followed desired direction for one "step")
    if show_next_step:
        step_x = x + arrow_len * dx
        step_y = y + arrow_len * dy
        plt.scatter([step_x], [step_y], s=10, label='next step', zorder=6)

    # Annotations
    txt = []
    if yaw_des is not None: txt.append(f"yaw_des={yaw_des:.1f}{'deg' if degrees else 'rad'}")
    if yaw_err is not None: txt.append(f"yaw_err={yaw_err:.1f}{'deg' if degrees else 'rad'}")
    if dist is not None: txt.append(f"dist={dist:.2f}")
    if txt:
        plt.text(0.02, 0.02, " | ".join(txt), transform=plt.gca().transAxes)

    if invert_y:
        plt.gca().invert_yaxis()

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title if title else "Guidance visualization")
    plt.legend()
    plt.tight_layout()
    plt.show()


def guidance_vector_field_batch(
    x_vec, y_vec, yaw_deg_vec,
    polyline_xy,
    d0,
    closed=True,
    d_min=1e-6,
    align_tangent_to_yaw=True,
    visualize=False,
    *,
    degrees=True,
    arrow_len=50.0,
    invert_y=False,
    title=None
):
    """
    Batch guidance for multiple robot poses.

    Returns a pandas DataFrame with pose and guidance outputs.
    Optionally visualizes the path and desired headings.
    """
    x_arr = np.asarray(x_vec, dtype=float)
    y_arr = np.asarray(y_vec, dtype=float)
    yaw_arr = np.asarray(yaw_deg_vec, dtype=float)

    if not (x_arr.shape == y_arr.shape == yaw_arr.shape):
        raise ValueError("x_vec, y_vec, yaw_deg_vec must have the same shape")

    rows = []
    for x, y, yaw_deg in zip(x_arr, y_arr, yaw_arr):
        yaw_des, yaw_err, (dx, dy), (qx, qy), dist = guidance_vector_field(
            x, y, yaw_deg,
            polyline_xy,
            d0,
            closed=closed,
            d_min=d_min,
            align_tangent_to_yaw=align_tangent_to_yaw
        )
        rows.append(
            {
                "x": float(x),
                "y": float(y),
                "yaw_deg": float(yaw_deg),
                "yaw_des_deg": float(yaw_des),
                "yaw_err_deg": float(yaw_err),
                "dir_x": float(dx),
                "dir_y": float(dy),
                "closest_x": float(qx),
                "closest_y": float(qy),
                "dist": float(dist),
            }
        )

    df = pd.DataFrame(rows)

    if visualize:
        pts = np.asarray(polyline_xy, dtype=float)
        plt.figure(figsize=(7, 7))
        plt.plot(pts[:, 0], pts[:, 1], "-", linewidth=2, label="polyline")
        plt.scatter(df["x"], df["y"], s=12, label="poses", zorder=5)
        plt.quiver(
            df["x"],
            df["y"],
            df["dir_x"],
            df["dir_y"],
            angles="xy",
            scale_units="xy",
            scale=1.0 / arrow_len,
            width=0.003,
            label="desired heading",
        )
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
            scale=1.0 / arrow_len,
            width=0.003,
            color="tab:blue",
            label="current heading",
        )

        if invert_y:
            plt.gca().invert_yaxis()

        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title if title else "Guidance batch visualization")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df
