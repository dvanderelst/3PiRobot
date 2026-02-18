import numpy as np

from Library.SteeringConfigClass import SteeringConfig
from Library.OccupancyCalculation import build_robot_frame_evidence, make_robot_frame_grid
from Library.CurvatureCalculation import plan_circles_from_heatmap, planner_to_curvature


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

    x_grid, y_grid, xx, yy = make_robot_frame_grid(extent_mm=float(cfg.extent_mm), grid_mm=float(cfg.grid_mm))

    # Single profile as a single-frame sequence in anchor robot coordinates.
    centers_seq = az_deg[None, :]
    dist_seq = dist_mm[None, :]
    presence_bin = np.isfinite(dist_seq).astype(np.uint8)
    presence_probs = presence_bin.astype(np.float32)
    rob_x_seq = np.array([0.0], dtype=np.float32)
    rob_y_seq = np.array([0.0], dtype=np.float32)
    rob_yaw_deg_seq = np.array([0.0], dtype=np.float32)

    hm_norm = build_robot_frame_evidence(
        profile_centers_deg_seq=centers_seq,
        distance_mm_seq=dist_seq,
        presence_probs_seq=presence_probs,
        presence_bin_seq=presence_bin,
        rob_x_seq=rob_x_seq,
        rob_y_seq=rob_y_seq,
        rob_yaw_deg_seq=rob_yaw_deg_seq,
        x_grid=x_grid,
        y_grid=y_grid,
        xx=xx,
        yy=yy,
        grid_mm=float(cfg.grid_mm),
        sigma_perp_mm=float(cfg.sigma_perp_mm),
        sigma_para_mm=float(cfg.sigma_para_mm),
        apply_smoothing=bool(cfg.apply_heatmap_smoothing),
    )

    planned = plan_circles_from_heatmap(
        hm_norm=hm_norm,
        x_grid=x_grid,
        y_grid=y_grid,
        config=cfg,
    )
    curvature = planner_to_curvature(planned)

    if not return_debug:
        return curvature

    az_rad = np.deg2rad(az_deg)
    profile_x = dist_mm * np.cos(az_rad)
    profile_y = dist_mm * np.sin(az_rad)
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
