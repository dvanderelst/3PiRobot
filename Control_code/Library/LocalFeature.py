"""
LocalFeature — the one feature every controller in this project acts on, and the
geometric truth behind it.

Both experiments are built on the claim that the controller is indifferent to
which modality produced its input. That only holds if every producer emits the
same object, so the definition lives here rather than in any one script:

    sonar   the deployed inverse's output          (SCRIPT_RunDirectPolicy)
    vision  geometry read at the robot's true pose (`feature_from_geometry`)
    sim     geometry, then corrupted by the        (EnvironmentSimulator +
            measured error model                    Library.InverseErrorModel)

`feature_from_geometry` was previously defined inside SCRIPT_RunDirectPolicy.
The path-following simulator needs exactly the same truth before corrupting it,
and a second implementation would be free to drift from the first — at which
point the vision teacher and the simulated sensor would disagree about what the
robot can see, silently.

Conventions: azimuth in degrees, CCW-positive, 0 = straight ahead. Distances in
mm. Pole range is to the pole SURFACE (centre minus radius), matching both the
inverse's training target and the acquisition labels.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from Library.AcquisitionSessionLoader import compute_profile, nearest_reflector_in_cone

# Rays cast when reducing arena geometry to a wall-depth profile. Only affects
# the vision/sim path; the sonar path gets its slices from the network.
GEOM_PROFILE_STEPS = 30

SLICE_NAMES = ["right", "center", "left"]


@dataclass
class LocalFeature:
    """The single input to every controller. Fixed shape regardless of source."""
    cls: str                              # "pole" | "wall" | "empty"
    pole_az_deg: float = float("nan")     # signed bearing when cls == "pole"
    pole_dist_mm: float = float("nan")    # range to the pole SURFACE
    p_pole: float = float("nan")          # class posteriors; sonar/sim only
    p_none: float = float("nan")
    pole_az_sigma_deg: float = float("nan")
    pole_dist_sigma_mm: float = float("nan")
    slices_mm: Dict[str, float] = field(
        default_factory=lambda: {n: float("nan") for n in SLICE_NAMES})
    slice_sigma_mm: Dict[str, float] = field(
        default_factory=lambda: {n: float("nan") for n in SLICE_NAMES})


def slice_profile(profile, cone_half_deg) -> Dict[str, float]:
    """Reduce a wall-depth profile to the three slice minima the inverse reports."""
    prof = np.asarray(profile, dtype=float)
    n = len(prof)
    edges = np.linspace(-cone_half_deg, cone_half_deg, 4)
    bearings = np.linspace(-cone_half_deg, cone_half_deg, n)
    out = {}
    for name, lo, hi in zip(SLICE_NAMES, edges[:-1], edges[1:]):
        m = (bearings >= lo) & (bearings <= hi)
        vals = prof[m][np.isfinite(prof[m])]
        out[name] = float(vals.min()) if vals.size else float("nan")
    return out


def true_local_feature(x, y, yaw, geom, cone_half_deg, max_range_mm=None):
    """Geometric truth at a pose, before any sensor error.

    Returns (cls, range_mm, pole_az_deg, slices_mm) where cls is 0 wall, 1 pole,
    2 none. This is the supervision signal the inverse was trained against, so
    it is also what the simulator must corrupt.
    """
    walls, poles = geom["walls"], geom["poles"]
    cls, pole_az, near = nearest_reflector_in_cone(
        walls, poles, geom["pole_radius_mm"], x, y, yaw, cone_half_deg)
    prof = compute_profile(walls, x, y, yaw,
                           opening_angle=2.0 * cone_half_deg,
                           profile_steps=GEOM_PROFILE_STEPS,
                           profile_method="ray_center")
    slices = slice_profile(prof, cone_half_deg)
    if not np.isfinite(cls):
        return 2, float("nan"), float("nan"), slices
    if max_range_mm is not None and np.isfinite(near) and near > max_range_mm:
        return 2, float(near), float("nan"), slices
    return int(cls), float(near), float(pole_az), slices


def feature_from_geometry(x, y, yaw, geom, cone_half_deg,
                          max_range_mm=None) -> LocalFeature:
    """Vision/sim producer: the noiseless local feature at a pose.

    When `max_range_mm` is set, abstain (-> "empty") if the nearest in-cone
    reflector lies beyond it, mirroring the inverse's `none` class.
    """
    cls, near, pole_az, slices = true_local_feature(
        x, y, yaw, geom, cone_half_deg, max_range_mm)
    if cls == 2:
        return LocalFeature(cls="empty")
    if cls == 1:
        return LocalFeature(cls="pole", pole_az_deg=pole_az, pole_dist_mm=near)
    return LocalFeature(cls="wall", slices_mm=slices)
