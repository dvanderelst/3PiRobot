"""The sensory-deprivation control: what the route does when the input is false.

Two panels, one per arena, each drawn in the arena's own frame:

  A  Arena 1 (Path04): one intact run, then three shuffled-input runs
  B  Arena 2 (Path07): the same

In a shuffled-input run every feature the inverse model reports is replaced by a
measurement drawn at random from a pool of 500 real predictions taken on the
same path by the same policy, so the marginal statistics of the input are
realistic and only its relation to where the robot actually is has been cut.

The intact run is drawn as a single trajectory rather than both, because at
this scale the two are indistinguishable and the panel is about the contrast
with the shuffled-input runs, not about baseline variation (Figs 8A and 9A
carry that). Each ends where it hit something: the marker is the last
logged pose, one step before the contact, since a run ends when a drive is
blocked or the operator stops it at a pole.

A pole contact leaves no `crashes.tsv` row -- the crash logger fires on a
blocked drive and a 25 mm dowel does not block the wheels -- so what each run
ended against is read from the final pose against the arena geometry, not from
that file.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_sensory_clamp.py
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import csv
import glob
import json
import sys

from paths import CONTROL, POLICY_RUNS, SCRIPTS

sys.path.insert(0, str(CONTROL))
os.chdir(str(CONTROL))

import numpy as np  # noqa: E402
import style  # noqa: E402

NAME = "fig_sensory_clamp"
STEP_MM = 150.0          # fixed forward step, so steps convert to lap fractions
PANELS = [
    ("A  Arena 1", "Path04", "default_Path04_run01",
     [f"default_Path04_run0{i}_shuffle" for i in (1, 2, 3)]),
    ("B  Arena 2", "Path07", "default_Path07_run01",
     [f"default_Path07_run0{i}_shuffle" for i in (1, 2, 3)]),
]

C_WALL = "#3E6B8A"
C_POLE = "#8172B2"
C_PATH = "0.45"
# The intact run is a reference, not a fourth condition. Drawn grey and behind
# the dashed path so the only saturated trajectories in a panel are the
# shuffled-input ones; its width against the dashed line is the baseline scatter, which is what
# makes the clamped medians legible as distances.
C_INTACT = "0.62"
# Three shuffled-input runs per panel. `CLAMP_SENSING` in SCRIPT_RunPolicy.py has
# a second variant, "const", which was flown twice and retired; only "shuffle"
# is reported, so the paper never says "clamp". Colours are distinct from the
# pole purple and ordered light to dark, so the three read as one set rather
# than as three unrelated conditions.
C_CLAMP = ["#4C72B0", "#55A868", "#CCA000"]


def run_path(run):
    """Runs are filed under PolicyRuns/Paths/; a fresh one sits in the root."""
    for base in (POLICY_RUNS / "Paths", POLICY_RUNS):
        if (base / run).is_dir():
            return base / run
    raise FileNotFoundError(f"run {run} in neither PolicyRuns/Paths nor PolicyRuns")


def load_geometry(run):
    f = glob.glob(f"{run_path(run)}/env_*/arena_features.npz")[0]
    d = np.load(f)
    x, y, k = d["x_mm"], d["y_mm"], d["kind"]
    return (np.stack([x[k == 0], y[k == 0]], 1),
            np.stack([x[k == 1], y[k == 1]], 1))


def load_path(arena):
    wp = json.load(open(f"TargetArenas/{arena}/target_path.json"))["waypoints"]
    W = np.array([[p["x_mm"], p["y_mm"]] for p in wp], float)
    C = np.vstack([W, W[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    t = np.arange(0, s[-1], 25.0)
    return np.stack([np.interp(t, s, C[:, 0]), np.interp(t, s, C[:, 1])], 1), s[-1]


def load_traj(run):
    rows = list(csv.DictReader(open(f"{run_path(run)}/step_metrics.tsv"),
                               delimiter="\t"))
    return np.array([[float(r["x_mm"]), float(r["y_mm"])] for r in rows])


def measures(xy, P, loop_mm):
    """Steps, laps and median distance from the path, as the text quotes them."""
    d = np.linalg.norm(xy[:, None, :] - P[None, :, :], axis=2).min(1)
    return dict(steps=len(xy), laps=len(xy) / (loop_mm / STEP_MM),
                median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                max=float(d.max()))


def main():
    style.setup()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_2COL, 4.7))
    numbers = {}

    for ax, (title, arena, intact, clamped) in zip(axes, PANELS):
        walls, poles = load_geometry(intact)
        P, loop_mm = load_path(arena)
        numbers[arena] = dict(loop_mm=loop_mm, steps_per_lap=loop_mm / STEP_MM,
                              runs={})

        ax.scatter(walls[:, 0], walls[:, 1], s=.6, color=C_WALL,
                   edgecolor="none", zorder=1)
        for p_xy in poles:
            ax.add_patch(Circle(tuple(p_xy), 60, fc=C_POLE, ec=C_POLE, lw=1.0,
                                zorder=5))
        xy = load_traj(intact)
        numbers[arena]["runs"][intact] = measures(xy, P, loop_mm)
        ax.plot(xy[:, 0], xy[:, 1], color=C_INTACT, lw=.5, alpha=.9, zorder=2)
        ax.plot(P[:, 0], P[:, 1], color=C_PATH, lw=1.0, ls=(0, (5, 3)), zorder=3)

        for colour, run in zip(C_CLAMP, clamped):
            xy = load_traj(run)
            numbers[arena]["runs"][run] = measures(xy, P, loop_mm)
            ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=1.1, alpha=.95, zorder=4)
            ax.plot(*xy[0], marker="o", ms=3.2, color=colour, mec="k", mew=.3,
                    ls="", zorder=6)
            ax.plot(*xy[-1], marker="X", ms=6.0, color=colour, mec="k", mew=.4,
                    ls="", zorder=6)

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8, pad=3, loc="left")

    # One scale bar, on the left panel, matching the other arena figures.
    x0, x1 = axes[0].get_xlim()
    frac = 1000.0 / (x1 - x0)
    axes[0].plot([0.06, 0.06 + frac], [0.04, 0.04], transform=axes[0].transAxes,
                 color="k", lw=2)
    axes[0].text(0.06, 0.065, "1 m", transform=axes[0].transAxes, fontsize=7)

    handles = [
        Line2D([], [], color=C_PATH, ls=(0, (5, 3)), lw=1.2, label="Trained path"),
        Line2D([], [], color=C_INTACT, lw=1.2, label="Intact run"),
    ] + [Line2D([], [], color=c, lw=1.2, label=f"Shuffled input {i}")
         for i, c in enumerate(C_CLAMP, 1)] + [
        Line2D([], [], marker="o", ls="", mfc="w", mec="k", ms=4, label="Release"),
        Line2D([], [], marker="X", ls="", mfc="w", mec="k", ms=6, label="Collision"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=7.5)
    fig.subplots_adjust(left=.01, right=.99, top=.96, bottom=.13, wspace=.03)

    style.save(fig, NAME)
    fig.savefig(style.IMAGES / f"{NAME}.png", dpi=150)
    with open(SCRIPTS / f"{NAME}_numbers.json", "w") as fh:
        json.dump(numbers, fh, indent=1)
    for arena, d in numbers.items():
        print(f"{arena}: lap {d['loop_mm']:.0f} mm, {d['steps_per_lap']:.1f} steps/lap")
        for run, m in d["runs"].items():
            print(f"   {run:<34} steps={m['steps']:>3} laps={m['laps']:4.2f} "
                  f"median={m['median']:6.1f} p90={m['p90']:5.0f} max={m['max']:5.0f}")


if __name__ == "__main__":
    main()
