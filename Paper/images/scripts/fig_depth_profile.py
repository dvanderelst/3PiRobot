"""Exploration (Methods): how a wall in front maps to the depth profile.

Same +/-35 deg cone as fig_fov, with an angled wall across it. Each sector
(left / center / right) returns the nearest wall distance within it -- the
three outputs of the depth-profile head. Mirrors compute_profile (ray sampling
+ per-sector minimum). A small bar readout shows the resulting profile.

-> Paper/images/image_resources/fig_depth_profile.svg
"""

import numpy as np

import style
from paths import IMAGE_RESOURCES

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

from fig_fov import draw_robot, R, RBODY, SECTORS  # noqa: E402

COLOR = {n: c for n, _t1, _t2, c in SECTORS}
# Simple slanted panel wall (two segments, one bend), like the hand sketch.
# The nearest wall point steps outward from the left sector to the right, so the
# three depths are well separated (near -> far) without a busy zigzag.
WALL = np.array([[2.4, 1.8], [3.7, 0.1], [4.3, -2.4]])


def ray_hit(theta_deg):
    """Distance from the robot to the nearest wall segment along bearing theta."""
    u = np.array([np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))])
    best_t, best_pt = np.inf, None
    for p, q in zip(WALL[:-1], WALL[1:]):
        d = q - p
        M = np.array([[u[0], -d[0]], [u[1], -d[1]]])
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        t, s = np.linalg.solve(M, p)
        if t > 0 and 0.0 <= s <= 1.0 and t < best_t:
            best_t, best_pt = t, t * u
    return best_t, best_pt


def sector_depth(t1, t2):
    best_t, best_pt = np.inf, None
    for th in np.linspace(t1, t2, 160):
        t, pt = ray_hit(th)
        if t < best_t:
            best_t, best_pt = t, pt
    return best_t, best_pt


def main():
    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    for _name, t1, t2, c in SECTORS:                       # faint cone
        ax.add_patch(Wedge((0, 0), R, t1, t2, width=R - RBODY,
                           fc=c, alpha=0.16, ec="none", zorder=1))
    ax.plot(WALL[:, 0], WALL[:, 1], color="#5b3a1e", lw=5,
            solid_capstyle="round", zorder=3)
    ax.text(WALL[0, 0] - 0.1, WALL[0, 1] + 0.25, "wall", fontsize=8, ha="center")

    depths = {}
    lab = {"left": r"$d_\mathrm{left}$", "center": r"$d_\mathrm{center}$", "right": r"$d_\mathrm{right}$"}
    for name, t1, t2, c in SECTORS:
        d, pt = sector_depth(t1, t2)
        depths[name] = d
        if pt is None:
            continue
        ax.plot([0, pt[0]], [0, pt[1]], color=c, lw=2.4, zorder=4)
        ax.scatter([pt[0]], [pt[1]], s=24, color=c, ec="k", lw=0.4, zorder=5)
        phi = np.arctan2(pt[1], pt[0])                     # offset label perpendicular-outward
        po = phi + (-1 if name == "right" else 1) * np.pi / 2
        m = pt * 0.5 + 0.5 * np.array([np.cos(po), np.sin(po)])
        ax.annotate(lab[name], m, fontsize=9, ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="w", ec="none", alpha=0.7))

    draw_robot(ax)

    # 3-bar readout of the profile (bottom-left)
    inset = ax.inset_axes([0.0, 0.02, 0.27, 0.34])
    order = ["left", "center", "right"]
    inset.bar(range(3), [depths[n] for n in order],
              color=[COLOR[n] for n in order], ec="k", lw=0.4)
    inset.set_xticks(range(3))
    inset.set_xticklabels(["L", "C", "R"], fontsize=7)
    inset.set_yticks([])
    inset.set_title("depth profile", fontsize=7)
    inset.tick_params(length=0)

    ax.set_title(r"Wall $\rightarrow$ depth profile", fontsize=9)
    ax.set_xlim(-1.8, R + 1.2)
    ax.set_ylim(-3.6, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")

    out = IMAGE_RESOURCES / "fig_depth_profile.svg"
    fig.savefig(out, bbox_inches="tight", transparent=True)
    print(f"[depth] wrote {out}")
    fig.savefig(IMAGE_RESOURCES / "fig_depth_profile.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
