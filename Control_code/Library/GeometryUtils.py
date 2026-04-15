"""
Library/GeometryUtils.py
========================
Pure-geometry helpers for robot-arena spatial reasoning.

Functions
---------
min_wall_distances      — vectorised min distance from positions to wall cloud
inside_arena_mask       — vectorised inside-arena test (angular sector coverage)
valid_headings          — headings at (x, y) that have no wall dead ahead
wall_lateral_angle      — angle from robot to nearest-wall centroid in robot frame
"""

import numpy as np
from scipy.ndimage import label as _nd_label
from scipy.spatial import KDTree as _KDTree

__all__ = [
    "min_wall_distances",
    "inside_arena_mask",
    "largest_component_mask",
    "valid_headings",
    "wall_lateral_angle",
]


def largest_component_mask(flat_mask: np.ndarray, grid_shape: tuple) -> np.ndarray:
    """
    Keep only the largest connected component in a flat boolean position mask.

    Reshapes the mask to `grid_shape`, runs 4-connected labelling, returns the
    flat mask with only the largest region set True.  This removes isolated
    islands such as positions inside enclosed obstacles that pass the distance
    and sector tests but are physically unreachable from the main arena floor.

    Parameters
    ----------
    flat_mask  : (M,) bool array  — valid-position mask in ravel order
    grid_shape : (n_rows, n_cols)  — shape of the meshgrid before ravelling

    Returns
    -------
    (M,) bool array
    """
    mask_2d = flat_mask.reshape(grid_shape)
    labeled, n = _nd_label(mask_2d)
    if n == 0:
        return flat_mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0                          # ignore background label
    return (labeled == sizes.argmax()).ravel()


def min_wall_distances(walls: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Minimum distance from each (xs[i], ys[i]) to the wall point cloud.

    Uses a KDTree for O(M log N) performance instead of O(M·N) brute force.

    Parameters
    ----------
    walls : (N, 2) float array  — wall point cloud in mm
    xs, ys : (M,) float arrays  — query positions

    Returns
    -------
    (M,) float array of minimum distances in mm
    """
    tree = _KDTree(walls)
    pts = np.column_stack([xs, ys])
    dists, _ = tree.query(pts, workers=-1)
    return dists


def inside_arena_mask(walls: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                      n_sectors: int = 8) -> np.ndarray:
    """
    Vectorised inside-arena test using angular sector coverage.

    A point is considered inside the arena if the wall point cloud contains at
    least one point in every angular sector around it.  A point outside the
    arena will have at least one empty sector (the one pointing away into open
    space), so this correctly rejects exterior positions even when they happen
    to be far from the nearest wall point.

    Parameters
    ----------
    walls    : (N, 2) wall point cloud in mm
    xs, ys   : (M,) candidate positions
    n_sectors: number of equal angular slices (8 → 45° each)

    Returns
    -------
    (M,) bool array — True where the point is enclosed by walls on all sides
    """
    if len(walls) == 0:
        return np.ones(len(xs), dtype=bool)

    dx = walls[:, 0][np.newaxis, :] - xs[:, np.newaxis]   # (M, N)
    dy = walls[:, 1][np.newaxis, :] - ys[:, np.newaxis]   # (M, N)
    angles = (np.degrees(np.arctan2(dy, dx)) % 360)        # (M, N) in [0, 360)

    sector_size = 360.0 / n_sectors
    inside = np.ones(len(xs), dtype=bool)
    for i in range(n_sectors):
        lo = i * sector_size
        hi = (i + 1) * sector_size
        has_wall = np.any((angles >= lo) & (angles < hi), axis=1)
        inside &= has_wall

    return inside


def valid_headings(x: float, y: float, walls: np.ndarray,
                   heading_step: int = 10,
                   min_clearance: float = 450.0,
                   cone_half: float = 15.0) -> np.ndarray:
    """
    Return array of valid heading angles (degrees, 0 = +X / East) at (x, y).

    A heading h is *invalid* if any wall point lies within `min_clearance` mm
    inside a cone of ±cone_half degrees around h.

    Parameters
    ----------
    x, y          : robot position in mm
    walls         : (N, 2) wall point cloud in mm
    heading_step  : angular resolution of the sweep (degrees)
    min_clearance : wall within this range in the cone blocks the heading (mm)
    cone_half     : half-angle of the forward-clearance cone (degrees)

    Returns
    -------
    (K,) float array of valid headings in degrees
    """
    headings = np.arange(0, 360, heading_step, dtype=float)
    if len(walls) == 0:
        return headings

    dx = walls[:, 0] - x
    dy = walls[:, 1] - y
    wall_angles = np.degrees(np.arctan2(dy, dx))   # (N,) in [-180, 180]
    wall_dists  = np.hypot(dx, dy)                 # (N,)

    valid = []
    for h in headings:
        dang = wall_angles - h
        dang = (dang + 180.0) % 360.0 - 180.0      # wrap to [-180, 180]
        in_cone = np.abs(dang) <= cone_half
        if not np.any(in_cone):
            valid.append(h)
        elif wall_dists[in_cone].min() > min_clearance:
            valid.append(h)

    return np.array(valid, dtype=float)


def wall_lateral_angle(x: float, y: float, heading_deg: float,
                       walls: np.ndarray,
                       k_nearest: int = 10) -> float:
    """
    Estimate the lateral angle (degrees) from the robot to the nearest wall.

    Uses the centroid of the K nearest wall points for robustness against
    sparse / unevenly sampled wall point clouds.

    Parameters
    ----------
    x, y        : robot position in mm
    heading_deg : robot heading (degrees, 0 = +X / East)
    walls       : (N, 2) wall point cloud in mm
    k_nearest   : number of nearest wall points used for centroid estimate

    Returns
    -------
    float in [-180, 180]
        Positive  → wall centroid is to the LEFT  of the heading
        Negative  → wall centroid is to the RIGHT of the heading
    """
    if len(walls) == 0:
        return 0.0

    dx = walls[:, 0] - x
    dy = walls[:, 1] - y
    dists = np.hypot(dx, dy)

    k = min(k_nearest, len(walls))
    if k >= len(walls):
        idx = np.arange(len(walls))
    else:
        idx = np.argpartition(dists, k)[:k]
    cx = dx[idx].mean()
    cy = dy[idx].mean()

    abs_angle = np.degrees(np.arctan2(cy, cx))
    rel_angle = abs_angle - heading_deg
    return float((rel_angle + 180.0) % 360.0 - 180.0)   # wrap to [-180, 180]
