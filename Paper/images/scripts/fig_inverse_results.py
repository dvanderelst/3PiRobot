"""Inverse-model results figure: what "recovers the features, but noisily" means.

Recomputes the deployed model's predictions on the held-out 15% (the same set
Table~\\ref{tab:inverse-results} reports) and shows four panels on a 2x2 grid:
  (A) classification confusion (row-normalised recall),
  (B) pole-azimuth true vs predicted, coloured by predicted sigma,
  (C) pole-range true vs predicted, coloured by predicted sigma,
  (D) wall-depth true vs predicted, the three slices in one panel by colour.
The slice legend sits inside (D). Held-out only, so the figure and the table
describe the same data. Panel C is drawn only when the deployed checkpoint
carries the pole-range head (feature_params wall3_pole_dist); on an older
checkpoint without it the panel is dropped and the other three are unchanged.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_inverse_results.py
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys

from paths import CONTROL, SONAR_MODEL

sys.path.insert(0, str(CONTROL))
os.chdir(str(CONTROL))  # so the trainer's relative data paths resolve

import numpy as np  # noqa: E402
import SCRIPT_TrainInverseModel as T  # noqa: E402
from Library.SonarModel import InverseModel  # noqa: E402
import style  # noqa: E402

NAME = "fig_inverse_results"
SLICE_COLORS = {"left": "#4C72B0", "center": "#55A868", "right": "#C44E52"}
ALPHA = 0.6


def _panel_letter(ax, letter):
    ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


def main():
    style.setup()
    import matplotlib.pyplot as plt

    # Held-out data + deployed-model predictions (matches run_deploy exactly).
    sonar, slice_t, classes, pole_az_deg, pole_dist_mm, quads, sess, bins, poses = \
        T.load_and_filter(with_poses=True)
    is_val = T.spatial_holdout_mask(poses, sess, T.HOLDOUT_FRAC, T.HOLDOUT_SEED)
    inv = InverseModel.load(model_dir=str(SONAR_MODEL), fold="deploy")
    pred = T.predict(inv.model, sonar[is_val],
                     (inv._s_mean, inv._s_std), (inv._t_mean, inv._t_std), inv.device)
    c = classes[is_val]
    has_range = pred.get("pole_pred_dist_mm") is not None

    # Panel E uses ALL echoes, not the held-out 15%. Recall has to be resolved
    # against true range, and 322 held-out pings split five ways gives bins too
    # thin to read. Stated in the caption, since it differs from A-D.
    pred_all = T.predict(inv.model, sonar,
                         (inv._s_mean, inv._s_std), (inv._t_mean, inv._t_std),
                         inv.device)

    # Four equal plot columns, each with its own narrow colourbar slot. Letting
    # fig.colorbar(ax=...) steal width from the parent axes made the four panels
    # different sizes -- B and C have colourbars, A and D do not -- so nothing
    # lined up either across a row or down a column. The slots for A and D stay
    # empty, which keeps all four plot axes identical and lets E span exactly
    # the width of the block above it.
    fig = plt.figure(figsize=(style.WIDTH_2COL, 7.6))
    # col 2 is a spacer: without it the left colourbar (C) sits hard against the
    # right panel's y-axis label.
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.95],
                          width_ratios=[1, 0.05, 0.48, 1, 0.05],
                          hspace=.55, wspace=.18)
    axc = fig.add_subplot(gs[0, 0]); axp = fig.add_subplot(gs[0, 3])
    axr = fig.add_subplot(gs[1, 0]); axw = fig.add_subplot(gs[1, 3])
    axe = fig.add_subplot(gs[2, :])
    cax_p = fig.add_subplot(gs[0, 4])          # pole azimuth (B)
    cax_r = fig.add_subplot(gs[1, 1])          # pole range   (C)

    # (A) confusion, row-normalised to recall
    names = [n.capitalize() for n in T.CLASS_NAMES]
    n = len(names)
    M = np.zeros((n, n))
    for ti in range(n):
        sel = c == ti
        if sel.any():
            for pi in range(n):
                M[ti, pi] = np.mean(pred["cls_pred"][sel] == pi)
    axc.imshow(M, cmap="Greens", vmin=0, vmax=1)
    axc.set_xticks(range(n)); axc.set_xticklabels(names)
    axc.set_yticks(range(n)); axc.set_yticklabels(names)
    axc.set_xlabel("Predicted"); axc.set_ylabel("True")
    for ti in range(n):
        for pi in range(n):
            axc.text(pi, ti, f"{M[ti, pi] * 100:.0f}", ha="center", va="center",
                     color="white" if M[ti, pi] > 0.5 else "black", fontsize=8)
    axc.set_title("Classification (recall %)")
    _panel_letter(axc, "A")

    # (B) pole azimuth true vs pred, coloured by predicted sigma
    pm = c == 1
    lim = T.CONE_HALF_DEG
    ticks_p = [-30, -15, 0, 15, 30]
    axp.plot([-lim, lim], [-lim, lim], "--", color="0.4", lw=0.8, zorder=0)
    sc = axp.scatter(pole_az_deg[is_val][pm], pred["pole_pred_az_deg"][pm],
                     c=pred["pole_pred_az_std"][pm], cmap="viridis", s=14,
                     alpha=ALPHA, edgecolor="none")
    axp.set_xlim(-lim, lim); axp.set_ylim(-lim, lim); axp.set_aspect("equal")
    axp.set_xticks(ticks_p); axp.set_yticks(ticks_p)
    axp.set_xlabel("True azimuth (deg)"); axp.set_ylabel("Predicted (deg)")
    axp.set_title("Pole azimuth")
    cb = fig.colorbar(sc, cax=cax_p)
    cb.set_label(r"Predicted $\sigma$ (deg)")
    _panel_letter(axp, "B")

    # (C) pole range true vs pred, same treatment as azimuth. Range is a monaural
    # time-of-flight cue, so this is the head that should look best.
    if has_range:
        lim_r = T.MAX_RANGE_MM
        ticks_r = [0, 250, 500, 750, 1000]
        axr.plot([0, lim_r], [0, lim_r], "--", color="0.4", lw=0.8, zorder=0)
        scr = axr.scatter(pole_dist_mm[is_val][pm], pred["pole_pred_dist_mm"][pm],
                          c=pred["pole_pred_dist_std"][pm], cmap="viridis", s=14,
                          alpha=ALPHA, edgecolor="none")
        axr.set_xlim(0, lim_r); axr.set_ylim(0, lim_r); axr.set_aspect("equal")
        axr.set_xticks(ticks_r); axr.set_yticks(ticks_r)
        axr.set_xlabel("True range (mm)"); axr.set_ylabel("Predicted (mm)")
        axr.set_title("Pole range")
        cbr = fig.colorbar(scr, cax=cax_r)
        cax_r.yaxis.set_label_position("left")
        cax_r.yaxis.set_ticks_position("left")
        cbr.set_label(r"Predicted $\sigma$ (mm)")
        _panel_letter(axr, "C")
    else:
        axr.set_visible(False)

    # (D) wall depth true vs pred, three slices in one panel by colour
    wm = c == 0
    mx = 0.0
    handles = []
    for i, sname in enumerate(T.SLICE_NAMES):
        tt = slice_t[is_val][wm, i]
        pp = pred["wall_pred_mean"][wm, i]
        v = ~np.isnan(tt)
        if v.any():
            h = axw.scatter(tt[v], pp[v], s=14, edgecolor="none", alpha=ALPHA,
                            color=SLICE_COLORS.get(sname, "k"), label=sname.capitalize())
            handles.append(h)
            mx = max(mx, float(np.nanmax(tt[v])), float(np.nanmax(pp[v])))
    lim_w = 1.05 * mx
    ticks_w = list(range(0, int(lim_w) + 1, 1000))
    axw.plot([0, lim_w], [0, lim_w], "--", color="0.4", lw=0.8, zorder=0)
    axw.set_xlim(0, lim_w); axw.set_ylim(0, lim_w); axw.set_aspect("equal")
    axw.set_xticks(ticks_w); axw.set_yticks(ticks_w)
    axw.set_xlabel("True distance (mm)"); axw.set_ylabel("Predicted (mm)")
    axw.set_title("Wall depth")
    _panel_letter(axw, "D" if has_range else "C")

    # slice legend inside the wall panel, in the empty lower-right corner
    axw.legend(handles=handles, loc="lower right", frameon=False,
               title="Wall slice", fontsize=7, title_fontsize=7)

    # (E) recall vs true range -- the structure the confusion matrix cannot show:
    # the classes are not uniformly recoverable, and pole recall is a hump rather
    # than a threshold, i.e. the sensor has a preferred operating range.
    # 150 mm bins below 500 where poles are sparse (only 22 echoes under 300 mm
    # in total), 100 mm from 500 through 1200 where the counts support it and
    # where the interesting behaviour is. Coarser bins hid it: at 200 mm width
    # pole recall read as a smooth decline, when in fact it holds above 66% to
    # 900 mm and then collapses to 25% in the last 100 mm before the 1 m class
    # boundary. Wall does the same (95% -> 73%), and `none` is worst just past
    # the boundary and recovers with distance -- the model is least reliable
    # exactly where the class definition flips.
    edges = np.array([200., 350., 500., 600., 700., 800., 900., 1000.])
    out_edges = np.array([1000., 1100., 1200., 1400., 1700., 2000.])
    cls_all = np.asarray(pred_all["cls_pred"])
    wall_near = np.nanmin(np.where(np.isfinite(slice_t), slice_t, np.nan), axis=1)
    true_rng = np.where(classes == 0, wall_near, pole_dist_mm)
    colours = {"wall": "#377eb8", "pole": "#c05cff", "none": "#7a7a7a"}
    plotted_x = []
    for ci, nm in enumerate(T.CLASS_NAMES):
        e = out_edges if nm == "none" else edges
        xs, ys = [], []
        for lo, hi in zip(e[:-1], e[1:]):
            m = (classes == ci) & (true_rng >= lo) & (true_rng < hi)
            if m.sum() < 15:
                continue
            xs.append(float(np.mean(true_rng[m])))
            ys.append(100.0 * (cls_all[m] == ci).mean())
        if xs:
            axe.plot(xs, ys, "o-", color=colours[nm], label=nm.capitalize(), ms=4)
            plotted_x += xs
    axe.axvline(1000, color="k", ls=":", lw=.8)
    axe.annotate("1 m class boundary", xy=(1000, 96), xytext=(1000, 96),
                 fontsize=6, ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                           alpha=.85))
    axe.set_xlabel("True range to the nearest reflector (mm)")
    axe.set_ylabel("Correctly classified (%)")
    axe.set_ylim(0, 100)
    axe.set_title("Classification against range (all echoes)", pad=6)
    # Fit the axis to the data. The last `none` bin's mean sits near 1500 mm, so
    # extending to the nominal 2000 mm bin edge left a quarter of the panel empty.
    lo_x, hi_x = min(plotted_x), max(plotted_x)
    axe.set_xlim(lo_x - 80, hi_x + 80)
    axe.legend(frameon=False, fontsize=7, ncol=3, loc="lower left")
    _panel_letter(axe, "E" if has_range else "D")

    style.save(fig, NAME)
    print(f"[fig] held-out: {len(c)} pings, {int(wm.sum())} wall, {int(pm.sum())} pole"
          f"{'' if has_range else '  (no range head in this checkpoint)'}")


if __name__ == "__main__":
    main()
