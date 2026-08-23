"""What the model believes about a pole, echo by echo.

The companion to fig_perception_examples, for the pole channels rather than the
wall profile. Each panel draws the true pole and, over it, the region the model
reported: predicted azimuth with its $\\pm\\sigma$ as an angular wedge, crossed
with predicted range and its $\\pm\\sigma$ as a radial band. Where they cross is
what the model thinks it is looking at.

Three things are meant to be visible at once:

  1. Close in the patch sits on the pole, which is the regime Experiment 1's
     approach runs in.
  2. Past a metre the patch stops moving outward while the pole keeps
     receding, because the pole-range head is trained only within 1 m. The
     saturation in the aggregate figure becomes a spatial fact.
  3. What the classifier actually said, annotated per panel. Out at 2 m it
     mostly says WALL, so the estimate drawn there is one no controller would
     ever act on -- the pole channels are gated by a class call the model
     cannot make at that range. Stated in the caption, because otherwise the
     figure looks like it is showing a failure the system commits, when in
     fact the system never gets that far.

Examples are chosen by rule: within each range band, the echo whose azimuth
error is the median for that band.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_pole_examples.py
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

NAME = "fig_pole_examples"
BANDS = [(200, 500), (500, 750), (750, 1000), (1000, 1400), (1400, 2000),
         (2000, 3300)]
VIEW_X, VIEW_Y = 1850, 2600
C_WALL = "#3E6B8A"
C_POLE = "#8172B2"
C_PRED = "#C44E52"


def _panel_letter(ax, letter):
    ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def main():
    style.setup()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Circle

    sonar, slice_t, classes, az, near, quads, sess, bc, poses = \
        T.load_and_filter(with_poses=True)

    n = len(classes)
    p_az = np.full(n, np.nan); s_az = np.full(n, np.nan)
    p_r = np.full(n, np.nan); s_r = np.full(n, np.nan)
    cls = np.full(n, -1); p_pole = np.full(n, np.nan)
    for q in T.CV_QUADRANTS:
        sel = quads == q
        inv = InverseModel.load(model_dir=str(SONAR_MODEL), fold=f"q{q}")
        p = T.predict(inv.model, sonar[sel], (inv._s_mean, inv._s_std),
                      (inv._t_mean, inv._t_std), inv.device)
        p_az[sel] = p["pole_pred_az_deg"]; s_az[sel] = p["pole_pred_az_std"]
        p_r[sel] = p["pole_pred_dist_mm"]; s_r[sel] = p["pole_pred_dist_std"]
        cls[sel] = p["cls_pred"]; p_pole[sel] = p["cls_probs"][:, 1]

    pole = (classes == 1) & np.isfinite(near) & np.isfinite(p_az)
    az_err = np.abs(p_az - az)

    picks = []
    for lo, hi in BANDS:
        m = pole & (near >= lo) & (near < hi)
        if m.sum() < 10:
            continue
        idx = np.where(m)[0]
        picks.append(int(idx[np.argsort(az_err[idx])[len(idx) // 2]]))

    geo = {}
    for s_name in sorted(set(sess)):
        d = Path(T.ACQUISITIONS_ROOT) / s_name
        f = _load_features_for_session(d)
        geo[s_name] = (_load_walls_for_session(d), f["poles"],
                       float(f.get("pole_radius_mm", 12.5)))

    half = T.CONE_HALF_DEG
    fig, axes = plt.subplots(2, 3, figsize=(style.WIDTH_2COL, 4.6))
    numbers = []

    for ax, i, letter in zip(axes.ravel(), picks, "ABCDEF"):
        walls, poles, pr = geo[sess[i]]
        x, y, yaw = poses[i]
        th = np.deg2rad(yaw)
        rot = -th + np.pi / 2
        R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        W = (R @ (walls[:, :2] - [x, y]).T).T
        ax.scatter(W[:, 0], W[:, 1], s=.9, color=C_WALL, edgecolor="none",
                   zorder=1)

        # the model's belief: predicted bearing +-sigma crossed with predicted
        # range +-sigma. Azimuth is +ccw from forward and the panel is drawn
        # forward-up, so the wedge is centred on 90 + predicted azimuth.
        a0 = 90 + p_az[i] - s_az[i]
        a1 = 90 + p_az[i] + s_az[i]
        r0 = max(p_r[i] - s_r[i], 0.0)
        r1 = p_r[i] + s_r[i]
        ax.add_patch(Wedge((0, 0), r1, a0, a1, width=r1 - r0, fc=C_PRED,
                           ec="none", alpha=.30, zorder=3))
        ang = np.deg2rad(90 + p_az[i])
        ax.plot([0, r1 * np.cos(ang)], [0, r1 * np.sin(ang)], color=C_PRED,
                lw=1.0, ls=(0, (3.5, 2)), zorder=4)
        ax.plot([p_r[i] * np.cos(ang)], [p_r[i] * np.sin(ang)], marker="x",
                ms=5, color=C_PRED, mew=1.4, zorder=5)

        # every pole in the arena, with the labelled one filled
        if len(poles):
            P = (R @ (np.asarray(poles)[:, :2] - [x, y]).T).T
            tx = (near[i] + pr) * np.cos(np.deg2rad(90 + az[i]))
            ty = (near[i] + pr) * np.sin(np.deg2rad(90 + az[i]))
            for px, py in P:
                target = np.hypot(px - tx, py - ty) < 80
                ax.add_patch(Circle((px, py), pr + 26,
                                    fc=C_POLE if target else "none",
                                    ec=C_POLE, lw=1.0, zorder=6))

        for a_deg in (90 - half, 90 + half):
            r = np.deg2rad(a_deg)
            ax.plot([0, VIEW_Y * np.cos(r)], [0, VIEW_Y * np.sin(r)],
                    color="0.45", lw=.7, ls=(0, (4, 2)), zorder=2)
        ax.plot([0], [0], marker="^", ms=5, color="k", zorder=7)
        ax.set_xlim(-VIEW_X, VIEW_X); ax.set_ylim(-330, VIEW_Y)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        called = T.CLASS_NAMES[int(cls[i])]
        ax.set_title("pole at %d mm, %+d$^{\\circ}$\nmodel: %d mm, %+d$^{\\circ}$"
                     "  (says %s, p=%.2f)"
                     % (round(near[i]), round(az[i]), round(p_r[i]),
                        round(p_az[i]), called, p_pole[i]),
                     fontsize=6.0, pad=2, linespacing=1.35)
        _panel_letter(ax, letter)
        numbers.append(dict(index=i, session=str(sess[i]),
                            true_range=float(near[i]), true_az=float(az[i]),
                            pred_range=float(p_r[i]), pred_az=float(p_az[i]),
                            sigma_range=float(s_r[i]), sigma_az=float(s_az[i]),
                            called=called, p_pole=float(p_pole[i])))

    fig.subplots_adjust(hspace=.24, wspace=.04, top=.86, bottom=.01,
                        left=.02, right=.98)
    fig.legend(handles=[
        plt.Line2D([], [], color=C_WALL, lw=0, marker="o", ms=3,
                   label="Arena geometry"),
        plt.Line2D([], [], color=C_POLE, lw=0, marker="o", ms=5, mfc="none",
                   label="Poles (filled: the labelled one)"),
        plt.Line2D([], [], color=C_PRED, lw=1.2, ls="--", marker="x",
                   label="Model's estimate, shaded $\\pm\\sigma$")],
        loc="upper center", ncol=3, frameon=False, fontsize=6.5)
    style.save(fig, NAME)
    with open(SCRIPTS / f"{NAME}_numbers.json", "w") as f:
        json.dump(numbers, f, indent=1)
    for d in numbers:
        print("[fig] true %4.0f mm %+5.1f deg | model %4.0f mm %+5.1f deg | "
              "says %-5s p_pole %.2f"
              % (d["true_range"], d["true_az"], d["pred_range"], d["pred_az"],
                 d["called"], d["p_pole"]))


if __name__ == "__main__":
    main()
