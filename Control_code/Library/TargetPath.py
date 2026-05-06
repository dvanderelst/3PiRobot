"""
Target-path representation used by the GA fitness function.

A TargetPath is a closed polygonal loop in arena (x, y) coordinates, densified
at a fixed spacing and parameterised by cumulative arc-length.  Every step of
a robot trajectory can be projected onto the path to recover both the
perpendicular distance to the loop and the arc-length coordinate s ∈ [0, L).

The projection is vectorised over all line segments — fast enough to call
once per step per episode without becoming a bottleneck.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TargetPath:
    points: np.ndarray   # (M, 2) densified path points; first point is repeated at the end
    cum_arc: np.ndarray  # (M,)   cumulative arc length to each point; cum_arc[-1] = total_length
    total_length: float
    # Optional release-box and direction-arrow used by training start sampling.
    # Box: (x_min_mm, y_min_mm, x_max_mm, y_max_mm). Arrow: (base_x, base_y, tip_x, tip_y).
    start_box:   Optional[Tuple[float, float, float, float]] = None
    start_arrow: Optional[Tuple[float, float, float, float]] = None

    @property
    def n_segments(self) -> int:
        return self.points.shape[0] - 1


def _densify_closed_loop(waypoints: np.ndarray, resample_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Densify a closed polygonal loop so successive points are at most resample_mm apart.
    Returns (dense_pts (M, 2), cum_arc (M,)) where dense_pts[0] == dense_pts[-1].
    """
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError(f"waypoints must be (N, 2), got {waypoints.shape}")
    if waypoints.shape[0] < 3:
        raise ValueError(f"need at least 3 waypoints for a loop, got {waypoints.shape[0]}")

    pts = [waypoints[0]]
    arc = [0.0]
    n = waypoints.shape[0]
    for i in range(n):
        a = waypoints[i]
        b = waypoints[(i + 1) % n]
        seg_len = float(np.linalg.norm(b - a))
        if seg_len < 1e-9:
            continue
        n_sub = max(1, int(np.ceil(seg_len / resample_mm)))
        step = seg_len / n_sub
        for j in range(1, n_sub + 1):
            t = j / n_sub
            pts.append(a * (1.0 - t) + b * t)
            arc.append(arc[-1] + step)
    return np.asarray(pts, dtype=np.float64), np.asarray(arc, dtype=np.float64)


def load_target_path(arena: str, data_folder: str, resample_mm: float) -> TargetPath:
    """Load TargetArenas/<arena>/target_path.json and densify."""
    path_json = os.path.join(data_folder, arena, "target_path.json")
    if not os.path.isfile(path_json):
        raise FileNotFoundError(f"Target path not found: {path_json}")
    with open(path_json) as f:
        data = json.load(f)
    waypoints = np.array(
        [[w["x_mm"], w["y_mm"]] for w in data["waypoints"]],
        dtype=np.float64,
    )
    points, cum_arc = _densify_closed_loop(waypoints, resample_mm)

    start_box = None
    if isinstance(data.get("start_box"), dict):
        b = data["start_box"]
        start_box = (
            float(b["x_min_mm"]), float(b["y_min_mm"]),
            float(b["x_max_mm"]), float(b["y_max_mm"]),
        )

    start_arrow = None
    if isinstance(data.get("start_arrow"), dict):
        a = data["start_arrow"]
        start_arrow = (
            float(a["base_x_mm"]), float(a["base_y_mm"]),
            float(a["tip_x_mm"]),  float(a["tip_y_mm"]),
        )

    return TargetPath(
        points=points, cum_arc=cum_arc, total_length=float(cum_arc[-1]),
        start_box=start_box, start_arrow=start_arrow,
    )


def project(path: TargetPath, x: float, y: float) -> Tuple[float, float]:
    """
    Project (x, y) onto the closest segment of the path.
    Returns (perp_dist_mm, arc_length_mm).
    """
    pts = path.points
    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    seg_len_sq = np.einsum("ij,ij->i", ab, ab)
    pos = np.array([x, y], dtype=np.float64)
    ap = pos - a
    t = np.einsum("ij,ij->i", ap, ab) / np.maximum(seg_len_sq, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    foot = a + t[:, None] * ab
    diffs = foot - pos
    dists_sq = np.einsum("ij,ij->i", diffs, diffs)
    i_min = int(np.argmin(dists_sq))
    perp = float(np.sqrt(dists_sq[i_min]))
    arc  = float(path.cum_arc[i_min] + t[i_min] * (path.cum_arc[i_min + 1] - path.cum_arc[i_min]))
    return perp, arc


def shortest_signed_delta(s_new: float, s_old: float, total_length: float) -> float:
    """
    Signed arc-length step from s_old to s_new on a closed loop of total_length.
    Returns a value in (-L/2, L/2]; positive = forward in parameterisation order.
    """
    L = total_length
    d = (s_new - s_old) % L
    if d > L / 2:
        d -= L
    return d
