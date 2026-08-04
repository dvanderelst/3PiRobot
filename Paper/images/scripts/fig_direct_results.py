"""Figure: Experiment 1 trajectories, sonar against vision.

Four panels on a 2x2 grid: columns are the modality (sonar, vision), rows are
the pole placement (P1, P2). Each panel overlays the five runs from the five
start poses, one color per start, so the same start can be traced across
conditions. The dashed ring is the 400 mm perceived range at which the approach
stops.

Trajectories are colored by START rather than by perceived class: with five
runs per panel, per-step class coloring turns into confetti, and the perception
story is carried by the separate sampling panel instead.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_direct_results.py
"""

import csv

import numpy as np

import style
from exp1_stats import (HORIZON_MM, POLES, SOURCES, STARTS, load_geometry,
                        run_dir, true_class)

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

STOP_MM = 400.0
# Discrete qualitative palette (ColorBrewer Dark2): five categorical starts, so
# a categorical map rather than a sampled continuous one. None of the five is
# the pole red or the wall grey.
START_COLORS = plt.get_cmap("Dark2").colors[:len(STARTS)]
WALL_C = "#8a8378"
POLE_C = "#d62728"
MISS_C = "#4d4d4d"


def load_path(pole, start, source, geom):
    """Path, initial heading, and the poses at which a pole was seen or missed.

    A miss is a pose where a pole was genuinely the nearest reflector in the
    forward cone and within the horizon, yet the modality reported something
    else. Vision has none by construction, since it reads the same geometry the
    referee uses; the misses are therefore a sonar-only overlay.
    """
    d = run_dir(pole, start, source)
    with open(d / "trajectory.tsv") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["x_mm"]]
    xy = np.array([[float(r["x_mm"]), float(r["y_mm"])] for r in rows])
    yaw0 = float(rows[0]["yaw_deg"])

    saw_pole, missed_pole = [], []
    walls, poles, prad = geom
    for r in rows:
        perceived = (r["feat_cls"] or "empty") == "pole"
        truth, _ = true_class(walls, poles, prad, float(r["x_mm"]),
                              float(r["y_mm"]), float(r["yaw_deg"]), HORIZON_MM)
        saw_pole.append(perceived)
        missed_pole.append(truth == "pole" and not perceived)
    return xy, yaw0, np.array(saw_pole), np.array(missed_pole)


def draw_panel(ax, pole, source, xlim, ylim):
    geom = load_geometry(run_dir(pole, STARTS[0], source))
    walls, poles, prad = geom
    ax.scatter(walls[:, 0], walls[:, 1], s=0.5, c=WALL_C, alpha=0.55, lw=0,
               zorder=1)
    for px, py in poles:
        ax.add_patch(plt.Circle((px, py), STOP_MM, fill=False, ec=POLE_C,
                                ls=(0, (3, 2)), lw=0.7, alpha=0.8, zorder=2))
        ax.add_patch(plt.Circle((px, py), max(prad, 45.0), fc=POLE_C, ec="k",
                                lw=0.4, zorder=6))

    for color, start in zip(START_COLORS, STARTS):
        xy, yaw0, saw_pole, missed_pole = load_path(pole, start, source, geom)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=0.9, alpha=0.95, zorder=3,
                solid_joinstyle="round")
        # Pole present within the horizon but not reported.
        ax.plot(xy[missed_pole, 0], xy[missed_pole, 1], marker="o", ms=4.4,
                mfc="none", mec=MISS_C, mew=0.8, ls="", zorder=4)
        # Poses at which the modality reported a pole. Under sonar these sit in
        # a shell around the pole, under vision they spread over the arena,
        # which is the sampling difference the perception paragraph reports.
        ax.plot(xy[saw_pole, 0], xy[saw_pole, 1], marker="o", ms=2.6,
                color=color, mec="k", mew=0.3, ls="", zorder=4)
        ax.plot(*xy[0], marker=(3, 0, yaw0 - 90.0), ms=5.5, color=color,
                mec="k", mew=0.4, ls="", zorder=5)
        # Arrival gets a distinct shape: the final pose is itself a detection,
        # so a larger round dot would not be readable against the shell.
        ax.plot(*xy[-1], marker="*", ms=8.5, color=color, mec="k", mew=0.4,
                ls="", zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def bounds(margin=200.0):
    xs, ys = [], []
    for pole in POLES:
        walls, _, _ = load_geometry(run_dir(pole, STARTS[0], SOURCES[0]))
        xs.append(walls[:, 0])
        ys.append(walls[:, 1])
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    return ((xs.min() - margin, xs.max() + margin),
            (ys.min() - margin, ys.max() + margin))


def main():
    xlim, ylim = bounds()
    fig, axes = plt.subplots(2, 2, figsize=(style.WIDTH_2COL, 6.4))

    for row, pole in enumerate(POLES):
        for col, source in enumerate(SOURCES):
            ax = axes[row, col]
            draw_panel(ax, pole, source, xlim, ylim)
            ax.text(0.035, 0.965, "ABCD"[row * 2 + col], transform=ax.transAxes,
                    va="top", ha="left", fontweight="bold", fontsize=8,
                    color="w", bbox=dict(boxstyle="round,pad=0.12", fc="k",
                                         ec="none", alpha=0.6))
            if row == 0:
                ax.set_title(source.capitalize(), fontsize=9, pad=4)
            if col == 0:
                ax.set_ylabel(f"Pole {pole[-1]}", fontsize=9, labelpad=4)

    frac = 1000.0 / (xlim[1] - xlim[0])
    axes[0, 0].plot([0.06, 0.06 + frac], [0.045, 0.045],
                    transform=axes[0, 0].transAxes, color="k", lw=2)
    axes[0, 0].text(0.06, 0.075, "1 m", transform=axes[0, 0].transAxes,
                    fontsize=7)

    handles = [Line2D([], [], color=c, lw=2, label=f"Start {s}")
               for c, s in zip(START_COLORS, STARTS)]
    handles += [
        Line2D([], [], marker="^", ls="", mfc="w", mec="k", ms=6,
               label="Start pose"),
        Line2D([], [], marker="o", ls="", mfc="k", mec="k", ms=3.6,
               label="Pole perceived"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=MISS_C, mew=0.8,
               ms=4.4, label="Pole present, missed"),
        Line2D([], [], marker="*", ls="", mfc="w", mec="k", ms=8,
               label="Arrival"),
        Line2D([], [], marker="o", ls="", mfc=POLE_C, mec="k", ms=6,
               label="Pole"),
        Line2D([], [], color=POLE_C, ls=(0, (3, 2)), lw=1,
               label=f"{STOP_MM:.0f} mm stop range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=8)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.09,
                        wspace=0.04, hspace=0.04)
    style.save(fig, "fig_direct_results")
    fig.savefig(style.IMAGES / "fig_direct_results.png", dpi=150)


if __name__ == "__main__":
    main()
