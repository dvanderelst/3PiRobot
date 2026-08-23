"""Arena 1: the trained route, and what each manipulation did to it.

Four panels, all drawn in the arena's own frame:

  A  the two unmanipulated runs, which set the floor
  B  the pole removed
  C  the pole and one block removed
  D  two blocks displaced

The arena geometry comes from the digitised layout stored with the runs. Note
that this layout is the TRAINED one and was not re-digitised for the
manipulated runs, so the removals and the displacement are drawn from the
recorded measurements rather than from a fresh digitisation: removed objects
are crossed out where they stood, and displaced ones carry an arrow from their
trained position to the measured new one.

Which objects those are was established from the arena photographs stored with
each run (`env_*/arena.png`): the block removed with the pole is the central
one, and the two displaced east are the central and the lower-central block.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_arena1_runs.py
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

NAME = "fig_arena1_runs"
ARENA = "Path04"
RUNS = {"baseline": "default_Path04_run01",
        "replicate": "default_Path04_run02",
        "pole_out": "default_Path04_run01_Pole_removed",
        "both_out": "default_Path04_run01_B1_Pole_removed",
        "moved": "default_Path04_run1_B1_moved"}
# Measured displacements of the two blocks, from the 2026-08-17 session.
MOVED = {"B1": (327.0, -44.0), "B2": (308.0, 17.0)}

C_WALL = "#3E6B8A"
C_POLE = "#8172B2"
C_PATH = "0.45"
C_TRAJ = "#C44E52"
C_TRAJ2 = "#DD9B7A"
C_GONE = "#B0B0B0"


def load_arena():
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    f = glob.glob(f"{POLICY_RUNS}/Paths/{RUNS['baseline']}/env_*/arena_features.npz")[0]
    d = np.load(f, allow_pickle=True)
    x, y, k = d["x_mm"], d["y_mm"], d["kind"]
    W = np.stack([x[k == 0], y[k == 0]], 1)
    pole = np.stack([x[k == 1], y[k == 1]], 1)
    t = cKDTree(W)
    pr = t.query_pairs(60, output_type="ndarray")
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(W), len(W)))
    _, lab = connected_components(g, directed=False)
    sizes = {c: int((lab == c).sum()) for c in set(lab.tolist())}
    walls = np.vstack([W[lab == c] for c in sizes if sizes[c] > 1000])
    blocks = {c: W[lab == c] for c in sizes if 100 < sizes[c] <= 1000}
    # name them by position: B1 is the central block (the one removed with the
    # pole), B2 the one below it; both were identified from the run photos.
    cents = {c: B.mean(0) for c, B in blocks.items()}
    order = sorted(cents, key=lambda c: -cents[c][1])
    named = {}
    for c in order:
        cx, cy = cents[c]
        if abs(cx) < 400 and -1200 < cy < 0:
            named["B1"] = c
        elif abs(cx + 400) < 400 and cy < -1800:
            named["B2"] = c
    return walls, blocks, named, pole


def load_path():
    wp = json.load(open(f"TargetArenas/{ARENA}/target_path.json"))["waypoints"]
    W = np.array([[p["x_mm"], p["y_mm"]] for p in wp], float)
    seg = np.diff(np.vstack([W, W[:1]]), axis=0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
    t = np.arange(0, s[-1], 25.0)
    return np.stack([np.interp(t, s, np.concatenate([W[:, 0], W[:1, 0]])),
                     np.interp(t, s, np.concatenate([W[:, 1], W[:1, 1]]))], 1)


def load_traj(run):
    rows = list(csv.DictReader(
        open(f"{POLICY_RUNS}/Paths/{run}/step_metrics.tsv"), delimiter="\t"))
    return np.array([[float(r["x_mm"]), float(r["y_mm"])] for r in rows])


def main():
    style.setup()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    walls, blocks, named, pole = load_arena()
    P = load_path()

    # Four maps on top, and below them the error against position around the
    # loop. That lower panel is what separates the two kinds of effect:
    # displacing the blocks lifts one stretch and leaves the rest alone, while
    # removing the pole lifts the whole loop.
    fig = plt.figure(figsize=(style.WIDTH_2COL, 4.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, .62], hspace=.28, wspace=.04,
                          left=.055, right=.99, top=.95, bottom=.10)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axp = fig.add_subplot(gs[1, :])
    titles = ["A  Trained arena, run twice",
              "B  Pole removed",
              "C  Pole and one block removed",
              "D  Two blocks displaced"]

    for ax, title in zip(axes, titles):
        ax.plot(P[:, 0], P[:, 1], color=C_PATH, lw=1.0, ls=(0, (5, 3)), zorder=2)
        ax.scatter(walls[:, 0], walls[:, 1], s=.6, color=C_WALL,
                   edgecolor="none", zorder=1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=6.5, pad=3, loc="left")

    def draw_blocks(ax, hide=(), shift=None):
        for name, c in named.items():
            B = blocks[c]
            if name in hide:
                ax.scatter(B[:, 0], B[:, 1], s=.6, color=C_GONE,
                           edgecolor="none", zorder=1)
                cen = B.mean(0)
                ax.plot(*cen, marker="x", ms=6, mew=1.6, color="k", zorder=6)
                continue
            if shift and name in shift:
                dx, dy = shift[name]
                ax.scatter(B[:, 0], B[:, 1], s=.6, color=C_GONE,
                           edgecolor="none", zorder=1)
                ax.scatter(B[:, 0] + dx, B[:, 1] + dy, s=.6, color=C_WALL,
                           edgecolor="none", zorder=3)
                cen = B.mean(0)
                ax.annotate("", xy=(cen[0] + dx, cen[1] + dy), xytext=tuple(cen),
                            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0,
                                            shrinkA=2, shrinkB=2), zorder=7)
            else:
                ax.scatter(B[:, 0], B[:, 1], s=.6, color=C_WALL,
                           edgecolor="none", zorder=1)
        for c, B in blocks.items():
            if c not in named.values():
                ax.scatter(B[:, 0], B[:, 1], s=.6, color=C_WALL,
                           edgecolor="none", zorder=1)

    def draw_pole(ax, gone=False):
        ax.add_patch(Circle(tuple(pole[0]), 55, fc="none" if gone else C_POLE,
                            ec=C_GONE if gone else C_POLE, lw=1.0, zorder=5))
        if gone:
            ax.plot(*pole[0], marker="x", ms=6, mew=1.6, color="k", zorder=6)

    T = {k: load_traj(v) for k, v in RUNS.items()}

    draw_blocks(axes[0]); draw_pole(axes[0])
    axes[0].plot(T["baseline"][:, 0], T["baseline"][:, 1], color=C_TRAJ, lw=.7,
                 alpha=.85, zorder=4, label="Run 1")
    axes[0].plot(T["replicate"][:, 0], T["replicate"][:, 1], color=C_TRAJ2,
                 lw=.7, alpha=.85, zorder=4, label="Run 2")
    axes[0].legend(frameon=False, fontsize=5.5, loc="lower left")

    draw_blocks(axes[1]); draw_pole(axes[1], gone=True)
    axes[1].plot(T["pole_out"][:, 0], T["pole_out"][:, 1], color=C_TRAJ, lw=.7,
                 alpha=.85, zorder=4)

    draw_blocks(axes[2], hide=("B1",)); draw_pole(axes[2], gone=True)
    axes[2].plot(T["both_out"][:, 0], T["both_out"][:, 1], color=C_TRAJ, lw=.7,
                 alpha=.85, zorder=4)

    # Mark the stretch of the loop the displaced blocks border, which is where
    # the route shift is measured. Without it the effect -- about 120 mm on a
    # 8.8 m loop -- is easy to miss at this scale.
    lo, hi = int(.54 * len(P)), int(.75 * len(P))
    axes[3].plot(P[lo:hi, 0], P[lo:hi, 1], color="#7A5C3E", lw=2.0, alpha=.45,
                 zorder=3, solid_capstyle="round")
    draw_blocks(axes[3], shift=MOVED); draw_pole(axes[3])
    axes[3].plot(T["moved"][:, 0], T["moved"][:, 1], color=C_TRAJ, lw=.7,
                 alpha=.85, zorder=4)

    # ---- profile around the loop -------------------------------------------
    # Absolute error, with the two unmanipulated runs drawn as a BAND rather
    # than a line. A ratio to baseline was tried and rejected: it reads well
    # for the displacement, but the baseline error itself varies around the
    # loop, so dividing by it moves the maxima and hides the result that
    # matters -- the rise sits at the object that was manipulated, and that
    # alignment only holds in millimetres.
    #
    # Estimated in a SLIDING window rather than in bins. Hard bins of 550 mm
    # were about as wide as the effects themselves, so a peak could be smeared
    # across a bin or split between two. The window is 500 mm of path,
    # evaluated every 100 mm: the same number of steps per estimate, five times
    # the resolution.
    WIN_MM, STEP_MM = 500.0, 100.0
    loop_mm = 25.0 * len(P)

    def frac_dist(run):
        xy = T[run]
        d = np.linalg.norm(xy[:, None, :] - P[None, :, :], axis=2)
        return (d.argmin(1) / len(P)) * loop_mm, d.min(1)

    FD = {k: frac_dist(k) for k in RUNS}
    centres = np.arange(0, loop_mm, STEP_MM)

    def prof(run):
        pos, dist = FD[run]
        out = []
        for c in centres:
            off = (pos - c + loop_mm / 2) % loop_mm - loop_mm / 2   # wrap
            m = np.abs(off) <= WIN_MM / 2
            out.append(np.median(dist[m]) if m.sum() > 5 else np.nan)
        return np.array(out)

    x = 100 * centres / loop_mm
    b1, b2 = prof("baseline"), prof("replicate")
    axp.fill_between(x, np.minimum(b1, b2), np.maximum(b1, b2), color="0.55",
                     alpha=.30, lw=0, zorder=1, label="Trained arena (both runs)")
    for run, colour, lab in (("pole_out", "#4C72B0", "Pole removed"),
                             ("both_out", "#8172B2", "Pole and block removed"),
                             ("moved", C_TRAJ, "Two blocks displaced")):
        axp.plot(x, prof(run), "-", color=colour, lw=1.2, label=lab, zorder=3)

    for pos, txt in ((26, "pole"), (62, "block"), (70, "block")):
        axp.plot([pos], [0], marker="^", ms=4, color="k", clip_on=False, zorder=5)
        axp.text(pos, -30, txt, fontsize=5, ha="center")
    axp.set_xlabel("Position around the loop (%)", labelpad=9)
    axp.set_ylabel("Distance from\npath (mm)")
    axp.set_xlim(0, 100); axp.set_ylim(0, None)
    axp.legend(frameon=False, fontsize=5.5, ncol=4, loc="upper center",
               columnspacing=1.2)

    style.save(fig, NAME)
    print("[fig] blocks named:", {k: int(v) for k, v in named.items()},
          "| pole at", pole[0].round().tolist())


if __name__ == "__main__":
    main()
