"""Figure asset: the robot's simulated visual field of view (top-down).

The vision/sim producer reads the nearest reflector inside a forward cone of
half-angle CONE_HALF_DEG = 35 deg (70 deg total) about the heading, and splits
the wall depth profile into three equal sectors -- left / center / right, each
~23.3 deg. This is the *same* +/-35 deg cone the sonar uses; the vision proxy
is deliberately restricted to it so it supplies the same local feature.

Schematic, not to scale. -> Paper/images/image_resources/fig_fov.svg
"""

import numpy as np

import style
from paths import IMAGE_RESOURCES

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge, Circle, Rectangle, Arc  # noqa: E402

HALF = 35.0                    # CONE_HALF_DEG
THIRD = 2 * HALF / 3           # sector width (deg)
R = 5.0                        # cone length (arbitrary units)
RBODY = 1.0                    # robot radius

# sectors (name, theta1, theta2, colour); +azimuth = left = +y (CCW)
SECTORS = [
    ("right", -HALF, -HALF + THIRD, "#8e6fb0"),
    ("center", -HALF + THIRD, HALF - THIRD, "#3a7ca5"),
    ("left", HALF - THIRD, HALF, "#4c9f70"),
]


def draw_robot(ax):
    for sgn in (1, -1):                                   # wheels (top / bottom)
        ax.add_patch(Rectangle((-0.55, sgn * RBODY - 0.11), 1.1, 0.22,
                               fc="#4d4d4d", ec="none", zorder=3))
    ax.add_patch(Circle((0, 0), RBODY, fc="#e9e9e9", ec="k", lw=1.2, zorder=4))
    # three transducer faces near the front (x, y, outward splay deg); ears +/-15, emitter straight
    L = 0.15
    for fx, fy, splay in [(0.60, 0.45, 15.0), (0.72, 0.0, 0.0), (0.60, -0.45, -15.0)]:
        a = np.radians(90.0 + splay)          # face perpendicular to the outward-pointing axis
        dx, dy = L * np.cos(a), L * np.sin(a)
        ax.plot([fx - dx, fx + dx], [fy - dy, fy + dy], color="k", lw=4,
                solid_capstyle="round", zorder=5)
    # heading arrow just past the front edge
    ax.annotate("", xy=(1.95, 0), xytext=(1.05, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6), zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    for _name, t1, t2, c in SECTORS:                      # FOV cone in 3 sectors
        ax.add_patch(Wedge((0, 0), R, t1, t2, width=R - RBODY,
                           fc=c, alpha=0.45, ec="none", zorder=1))
    for s in (-1, 1):                                     # cone outline at +/-35
        a = np.radians(s * HALF)
        ax.plot([RBODY * np.cos(a), R * np.cos(a)], [RBODY * np.sin(a), R * np.sin(a)],
                color="0.3", lw=1.0, zorder=2)

    # range horizon + boresight
    ax.add_patch(Arc((0, 0), 2 * R, 2 * R, angle=0, theta1=-HALF, theta2=HALF,
                     ls="--", color="0.4", lw=1.0, zorder=2))
    ax.text(R * np.cos(np.radians(HALF)) + 0.15, R * np.sin(np.radians(HALF)),
            "max range\n($\\approx$1 m)", fontsize=7, va="center")
    ax.plot([RBODY, R], [0, 0], ls=":", color="0.4", lw=1.0, zorder=2)

    # half-angle annotation
    ra = 2.7
    ax.add_patch(Arc((0, 0), 2 * ra, 2 * ra, angle=0, theta1=0, theta2=HALF,
                     color="k", lw=1.0, zorder=6))
    am = np.radians(HALF / 2)
    ax.text((ra + 0.25) * np.cos(am), (ra + 0.25) * np.sin(am), "$35^{\\circ}$",
            fontsize=8, va="center")

    # sector labels at the far end of each wedge
    for name, t1, t2, _c in SECTORS:
        am = np.radians((t1 + t2) / 2)
        ax.text(R * 0.82 * np.cos(am), R * 0.82 * np.sin(am), name,
                ha="center", va="center", fontsize=8, zorder=6)

    draw_robot(ax)
    ax.text(R * 0.45, R * np.sin(np.radians(HALF)) + 0.55,
            "Simulated visual field of view: $70^{\\circ}$ ($\\pm 35^{\\circ}$)",
            ha="center", fontsize=9)

    ax.set_xlim(-1.8, R + 1.6)
    ax.set_ylim(-R * np.sin(np.radians(HALF)) - 0.7, R * np.sin(np.radians(HALF)) + 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    out = IMAGE_RESOURCES / "fig_fov.svg"
    fig.savefig(out, bbox_inches="tight", transparent=True)
    print(f"[fov] wrote {out}")
    fig.savefig(IMAGE_RESOURCES / "fig_fov.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
