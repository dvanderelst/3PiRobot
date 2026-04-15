"""
Demonstrator.py
===============
Computes the demonstrator steering command for a given robot pose (x, y, yaw).

The demonstrator policy blends two components:
  - Gradient direction   : corrects toward the ridge (cross-ridge)
  - Tangential direction : travels along the ridge (along-ridge)

The blend weight is the normalised potential value V ∈ [0, 1]:
    target = (1 - V) * grad_unit + V * tangential_unit

V ≈ 0  → mostly gradient  (strong correction when far from ridge)
V ≈ 1  → mostly tangential (smooth cruising when on the ridge)

Of the two possible tangential directions (±90° from gradient), the one
closest to the current yaw is chosen so the robot continues in its current
direction of travel.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class Demonstrator:
    """
    Demonstrator policy backed by a precomputed potential field.

    Parameters
    ----------
    xs_grid : (C,) array
        X coordinates of the grid (mm), strictly increasing.
    ys_grid : (R,) array
        Y coordinates of the grid (mm), strictly increasing.
    potential : (R, C) array
        Normalised potential field in [0, 1]. NaN outside valid area.
    grad_x : (R, C) array
        X-component of the potential gradient. NaN outside valid area.
    grad_y : (R, C) array
        Y-component of the potential gradient. NaN outside valid area.
    """

    def __init__(self,
                 xs_grid: np.ndarray,
                 ys_grid: np.ndarray,
                 potential: np.ndarray,
                 grad_x: np.ndarray,
                 grad_y: np.ndarray):
        opts = dict(method="linear", bounds_error=False, fill_value=np.nan)
        points = (ys_grid, xs_grid)   # RegularGridInterpolator expects (row, col) = (y, x)
        self._V  = RegularGridInterpolator(points, potential, **opts)
        self._gx = RegularGridInterpolator(points, grad_x,   **opts)
        self._gy = RegularGridInterpolator(points, grad_y,   **opts)

    def get_potential(self, x: float, y: float) -> float:
        """Return the normalised potential V ∈ [0, 1] at (x, y). NaN outside valid area."""
        return float(self._V(np.array([[y, x]]))[0])

    def get_delta_angle(self, x: float, y: float, yaw: float) -> float:
        """
        Compute the demonstrator steering angle for pose (x, y, yaw).

        Parameters
        ----------
        x, y : float
            Robot position in mm.
        yaw : float
            Current heading in radians.

        Returns
        -------
        delta_angle : float
            Signed turn angle in radians (positive = left), wrapped to [-π, π].
            Returns 0.0 if position is outside the valid area.
        """
        pt = np.array([[y, x]])   # (y, x) order for RegularGridInterpolator
        V  = float(self._V(pt)[0])
        gx = float(self._gx(pt)[0])
        gy = float(self._gy(pt)[0])

        if np.isnan(V) or np.isnan(gx) or np.isnan(gy):
            return 0.0

        mag = np.hypot(gx, gy)

        if mag < 1e-9:
            # On the ridge: gradient vanishes, continue in current direction
            return 0.0

        # Normalised gradient (cross-ridge component)
        gx_u, gy_u = gx / mag, gy / mag

        # Two tangential candidates (±90°)
        tx1, ty1 =  gy_u, -gx_u   # clockwise rotation
        tx2, ty2 = -gy_u,  gx_u   # anticlockwise rotation

        # Pick the tangential direction closest to current yaw
        heading_t1 = np.arctan2(ty1, tx1)
        heading_t2 = np.arctan2(ty2, tx2)
        d1 = abs(_wrap_angle(heading_t1 - yaw))
        d2 = abs(_wrap_angle(heading_t2 - yaw))
        tx, ty = (tx1, ty1) if d1 <= d2 else (tx2, ty2)

        # Blend: V=0 → pure gradient, V=1 → pure tangential
        bx = (1.0 - V) * gx_u + V * tx
        by = (1.0 - V) * gy_u + V * ty

        target_heading = np.arctan2(by, bx)
        return _wrap_angle(target_heading - yaw)
