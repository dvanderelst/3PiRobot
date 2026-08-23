"""What a single perception looks like: the wall profile the model recovers.

Everything else in the inverse-model Results is aggregate. This figure shows
individual echoes: the arena around the robot, the three sectors of its field,
the true nearest wall distance in each, and what the model reported with its
own uncertainty.

Examples are chosen by a rule rather than by eye. Within each range band we
take the echo whose mean absolute sector error is the MEDIAN for that band, so
each panel is a typical case for its distance, not a flattering one. The rule
is stated in the caption for the same reason.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_perception_examples.py
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json
import sys

from paths import CONTROL, SONAR_MODEL, SCRIPTS

sys.path.insert(0, str(CONTROL))
os.chdir(str(CONTROL))

import numpy as np  # noqa: E402
import SCRIPT_TrainInverseModel as T  # noqa: E402
from Library.SonarModel import InverseModel  # noqa: E402
from Library.AcquisitionSessionLoader import (_load_walls_for_session,  # noqa: E402
                                              _load_features_for_session)
from pathlib import Path  # noqa: E402
import style  # noqa: E402

NAME = "fig_perception_examples"
BANDS = [(200, 500), (500, 750), (750, 1000), (1000, 1400), (1400, 2000),
         (2000, 3300)]
# Common scale across panels, deliberately: the wedges growing with range is
# itself part of what the figure shows. Half-width is set by the cone reaching
# the top of the panel, tan(35 deg) * 2600.
VIEW_X = 1850
VIEW_Y = 2600
C_TRUE = "#333333"
C_WALL = "#3E6B8A"      # arena geometry, distinct from the grey cone edges
C_PRED = "#C44E52"
SECTOR_FILL = "#DCC9A8"


def arc(ax, radius, lo_deg, hi_deg, **kw):
    """A sector arc as a polyline.

    Drawn this way rather than as a thin Wedge: a Wedge of width 1 has an
    inner and an outer arc a millimetre apart, which overlap on screen, and a
    dashed linestyle then runs along both with different phase. The gaps of one
    fill the dashes of the other, so the same dashed arc renders solid at some
    radii and dashed at others, which looks like it means something. It does
    not.
    """
    a = np.deg2rad(np.linspace(90 + lo_deg, 90 + hi_deg, 60))
    ax.plot(radius * np.cos(a), radius * np.sin(a), **kw)


def sector_edges(half):
    """The three thirds of the field, in SLICE_NAMES order.

    That order is ("right", "center", "left"), i.e. column 0 of the targets is
    the MOST NEGATIVE azimuth, which is the robot's physical right. Azimuth is
    +ccw from forward, so with the panel drawn forward-up, negative azimuth is
    screen-right. Returning these in the other order silently mirrors the
    figure: the right sector's distance gets drawn on the left, and the arcs
    then float in open space instead of landing on the wall.
    """
    step = 2 * half / 3.0
    return [(-half, -half + step),          # right  (most negative azimuth)
            (-half + step, half - step),    # centre
            (half - step, half)]            # left


def main():
    style.setup()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Circle

    sonar, slice_t, classes, az, near, quads, sess, bc, poses = \
        T.load_and_filter(with_poses=True)

    pred_wall = np.full_like(slice_t, np.nan, dtype=float)
    pred_std = np.full_like(slice_t, np.nan, dtype=float)
    for q in T.CV_QUADRANTS:
        sel = quads == q
        inv = InverseModel.load(model_dir=str(SONAR_MODEL), fold=f"q{q}")
        p = T.predict(inv.model, sonar[sel], (inv._s_mean, inv._s_std),
                      (inv._t_mean, inv._t_std), inv.device)
        pred_wall[sel] = p["wall_pred_mean"]
        pred_std[sel] = p["wall_pred_std"]

    err = np.nanmean(np.abs(pred_wall - slice_t), axis=1)
    is_wall = (classes == 0) & np.isfinite(near) & np.isfinite(err)

    # the median-error example of each band
    picks = []
    for lo, hi in BANDS:
        m = is_wall & (near >= lo) & (near < hi)
        if m.sum() < 10:
            continue
        idx = np.where(m)[0]
        picks.append(int(idx[np.argsort(err[idx])[len(idx) // 2]]))

    geo = {}
    for s_name in sorted(set(sess)):
        d = Path(T.ACQUISITIONS_ROOT) / s_name
        f = _load_features_for_session(d)
        geo[s_name] = (_load_walls_for_session(d), f["poles"],
                       float(f.get("pole_radius_mm", 12.5)))

    half = T.CONE_HALF_DEG
    edges = sector_edges(half)
    fig, axes = plt.subplots(2, 3, figsize=(style.WIDTH_2COL, 4.5))
    numbers = []

    for ax, i in zip(axes.ravel(), picks):
        walls, poles, pr = geo[sess[i]]
        x, y, yaw = poses[i]
        # rotate into the robot's frame so every panel faces the same way:
        # the field of view points up, which makes the sectors comparable.
        th = np.deg2rad(yaw)
        R = np.array([[np.cos(-th + np.pi / 2), -np.sin(-th + np.pi / 2)],
                      [np.sin(-th + np.pi / 2), np.cos(-th + np.pi / 2)]])
        W = (R @ (walls[:, :2] - [x, y]).T).T
        ax.scatter(W[:, 0], W[:, 1], s=.9, color=C_WALL, edgecolor="none",
                   zorder=1)
        if len(poles):
            P = (R @ (np.asarray(poles)[:, :2] - [x, y]).T).T
            for px, py in P:
                ax.add_patch(Circle((px, py), pr + 22, fc="#8172B2",
                                    ec="none", zorder=3))

        for k, (lo, hi) in enumerate(edges):
            # +90 because the panel is drawn with the robot's forward axis up
            w0, w1 = 90 + lo, 90 + hi
            t, pm, ps = slice_t[i, k], pred_wall[i, k], pred_std[i, k]
            if np.isfinite(t):
                ax.add_patch(Wedge((0, 0), t, w0, w1, fc=SECTOR_FILL,
                                   ec="none", alpha=.85, zorder=0))
                arc(ax, t, lo, hi, color=C_TRUE, lw=1.2, zorder=4)
            if np.isfinite(pm):
                if np.isfinite(ps):
                    ax.add_patch(Wedge((0, 0), pm + ps, w0, w1, width=2 * ps,
                                       fc=C_PRED, ec="none", alpha=.22,
                                       zorder=2))
                arc(ax, pm, lo, hi, color=C_PRED, lw=1.3, ls=(0, (3.5, 2)),
                    zorder=5)

        for a_deg in (90 - half, 90 + half):
            r = np.deg2rad(a_deg)
            ax.plot([0, VIEW_Y * np.cos(r)], [0, VIEW_Y * np.sin(r)],
                    color="0.45", lw=.7, ls=(0, (4, 2)), zorder=4)
        ax.plot([0], [0], marker="^", ms=5, color="k", zorder=6)
        ax.set_xlim(-VIEW_X, VIEW_X); ax.set_ylim(-330, VIEW_Y)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        # Per-sector errors, not the mean: the spread across the field is what
        # this figure exists to show, and a single mean hides it. In the first
        # example the left sector is out by 145 mm while the mean reads 79.
        # Order is SLICE_NAMES, right / centre / left, matching the panel from
        # left to right when read R/C/L against the drawing's mirror.
        per_sector = np.abs(pred_wall[i] - slice_t[i])
        # Two lines: on one line these collide across columns. The right /
        # centre / left key lives in the legend instead of being repeated six
        # times.
        ax.set_title("nearest wall %d mm\nsector errors %s mm"
                     % (round(near[i]),
                        " / ".join("%d" % round(v) for v in per_sector)),
                     fontsize=6.2, pad=2, linespacing=1.35)
        numbers.append(dict(index=i, session=str(sess[i]),
                            near_mm=float(near[i]), mean_err_mm=float(err[i]),
                            per_sector_err=[float(v) for v in
                                            np.abs(pred_wall[i] - slice_t[i])],
                            true=[float(v) for v in slice_t[i]],
                            pred=[float(v) for v in pred_wall[i]],
                            sigma=[float(v) for v in pred_std[i]]))

    fig.subplots_adjust(hspace=.20, wspace=.04, top=.88, bottom=.01,
                        left=.02, right=.98)
    fig.legend(handles=[
        plt.Line2D([], [], color=C_WALL, lw=0, marker="o", ms=3,
                   label="Arena geometry"),
        plt.Line2D([], [], color=C_TRUE, lw=1.2, label="True nearest wall"),
        plt.Line2D([], [], color=C_PRED, lw=1.2, ls="--",
                   label="Model, with $\\pm\\sigma$"),
        plt.Line2D([], [], color="none",
                   label="errors listed right / centre / left")],
        loc="upper center", ncol=4, frameon=False, fontsize=6.5,
        handlelength=1.6, columnspacing=1.3)
    style.save(fig, NAME)
    with open(SCRIPTS / f"{NAME}_numbers.json", "w") as f:
        json.dump(numbers, f, indent=1)
    print(f"[fig] {len(picks)} examples: "
          + ", ".join("%d mm (err %d)" % (round(n['near_mm']),
                                          round(n['mean_err_mm']))
                      for n in numbers))


if __name__ == "__main__":
    main()
