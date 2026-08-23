"""Arena 2: the trained route, and what each manipulation did to it.

Four panels, all drawn in the arena's own frame:

  A  the two unmanipulated runs, which set the floor
  B  the two northern poles removed
  C  the same two poles displaced, run twice
  D  an extra wall placed across the southern part of the loop

The arena geometry comes from the digitised layout stored with the runs. Note
that this layout is the TRAINED one and was not re-digitised for the
manipulated runs, so the removals and the displacement are drawn from the
recorded measurements rather than from a fresh digitisation: removed objects
are crossed out where they stood, and displaced ones carry an arrow from their
trained position to the measured new one.

Which objects those are was established from the arena photographs stored with
each run (`env_*/arena.png`): the manipulated poles are the two northern ones,
which are the same two in the removal and displacement conditions, and the
displaced wall is the southern boundary section, moved inward.

Pole and wall displacements are marked but not drawn to their new positions:
unlike Arena 1's blocks, they were not measured object by object, so the figure
shows which objects changed and the text gives what was measured of the route.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_arena2_runs.py
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

NAME = "fig_arena2_runs"
ARENA = "Path07"
RUNS = {"baseline": "default_Path07_run01",
        "replicate": "default_Path07_run02",
        "poles_out": "default_Path07_run01_removed_poles",
        "poles_moved": "default_Path07_run01_moved_poles",
        "poles_moved2": "default_Path07_run02_moved_poles",
        "wall": "default_Path07_run01_added_walls"}
# The two poles manipulated in both conditions, identified from the photos,
# and the loop positions the text refers to.
MANIP_POLES = [(110.0, 736.0), (-1020.0, 404.0)]
WALL_STRETCH = (37, 43)      # % of the loop the displaced wall section borders

# Measured displacements, from the blob inventory of each run's own arena
# photograph (Performance notes 2026-08-20 and 08-21). The two displacement
# runs moved the same two poles in different directions and by different
# amounts, which is why both are drawn.
#   run 1: P0 ~215 mm SSW, P1 ~545 mm N
#   run 2: P0 ~800 mm ESE, P1 ~400 mm SW
# A third pole, P2, was also displaced (~225 mm SW) in the first displacement
# run, and stood at that same displaced position through the removal run. So
# the removal condition is two poles removed and one displaced, and the first
# displacement condition is three poles displaced. Both are drawn as such.
P2 = (1236.0, -128.0)
P2_MOVE = (-159.0, -159.0)
POLE_MOVES = {
    "poles_moved":  {(110.0, 736.0): (-82.0, -199.0),
                     (-1020.0, 404.0): (0.0, 545.0),
                     P2: P2_MOVE},
    "poles_moved2": {(110.0, 736.0): (739.0, -306.0),
                     (-1020.0, 404.0): (-283.0, -283.0)},
    "poles_out":    {P2: P2_MOVE},
}
# The added wall, read off Dieter's annotation of that run's own photograph
# (arena_shark_annotated_moved_wall_drawn.png) in the rectified map frame that
# meta.json defines: 5 mm/px over x in [-2500, 3000], y in [-3500, 2000].
# Fit y = 0.0588 x - 2607 over x in [-470, 1995], rms 40 mm, which agrees with
# the independent fit recorded on 2026-08-21 (y = 0.051 x - 2626) to ~25 mm.
WALL_FIT = dict(slope=0.0588, intercept=-2607.0, x0=-470.0, x1=1995.0)

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
              "B  Two poles removed, one moved",
              "C  Poles displaced, run twice",
              "D  Wall added across the path"]
    for ax, title in zip(axes, titles):
        ax.plot(P[:, 0], P[:, 1], color=C_PATH, lw=1.0, ls=(0, (5, 3)), zorder=2)
        ax.scatter(walls[:, 0], walls[:, 1], s=.6, color=C_WALL,
                   edgecolor="none", zorder=1)
        for c, B in blocks.items():
            ax.scatter(B[:, 0], B[:, 1], s=.6, color=C_WALL, edgecolor="none",
                       zorder=1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=6.0, pad=3, loc="left")

    manip = np.array(MANIP_POLES)

    def draw_poles(ax, mark=None, vacated=()):
        for p_xy in pole:
            if any(np.linalg.norm(np.array(v) - p_xy) < 60 for v in vacated):
                ax.add_patch(Circle(tuple(p_xy), 60, fc="none", ec=C_GONE,
                                    lw=1.0, ls=(0, (2, 2)), zorder=5))
                continue
            is_manip = np.min(np.linalg.norm(manip - p_xy, axis=1)) < 60
            if is_manip and mark:
                ax.add_patch(Circle(tuple(p_xy), 60, fc="none", ec=C_GONE,
                                    lw=1.0, zorder=5))
                if mark == "x":
                    ax.plot(*p_xy, marker="x", ms=6, mew=1.6, color="k", zorder=6)
                else:
                    ax.add_patch(Circle(tuple(p_xy), 140, fc="none", ec="k",
                                        lw=1.0, ls=(0, (2, 2)), zorder=6))
            else:
                ax.add_patch(Circle(tuple(p_xy), 60, fc=C_POLE, ec=C_POLE,
                                    lw=1.0, zorder=5))

    T = {k: load_traj(v) for k, v in RUNS.items()}

    draw_poles(axes[0])
    axes[0].plot(T["baseline"][:, 0], T["baseline"][:, 1], color=C_TRAJ, lw=.7,
                 alpha=.85, zorder=4, label="Run 1")
    axes[0].plot(T["replicate"][:, 0], T["replicate"][:, 1], color=C_TRAJ2,
                 lw=.7, alpha=.85, zorder=4, label="Run 2")
    axes[0].legend(frameon=False, fontsize=5.5, loc="lower left")

    draw_poles(axes[1], mark="x", vacated=(P2,))
    for base_xy, (dx, dy) in POLE_MOVES["poles_out"].items():
        axes[1].annotate("", xy=(base_xy[0] + dx, base_xy[1] + dy),
                         xytext=base_xy,
                         arrowprops=dict(arrowstyle="-|>", color=C_TRAJ, lw=1.0,
                                         shrinkA=3, shrinkB=1), zorder=7)
        axes[1].add_patch(Circle((base_xy[0] + dx, base_xy[1] + dy), 60,
                                 fc=C_TRAJ, ec=C_TRAJ, lw=.8, alpha=.8, zorder=6))
    axes[1].plot(T["poles_out"][:, 0], T["poles_out"][:, 1], color=C_TRAJ,
                 lw=.7, alpha=.85, zorder=4)

    draw_poles(axes[2], mark="o", vacated=(P2,))
    for run, colour in (("poles_moved", C_TRAJ), ("poles_moved2", C_TRAJ2)):
        for base_xy, (dx, dy) in POLE_MOVES[run].items():
            axes[2].annotate("", xy=(base_xy[0] + dx, base_xy[1] + dy),
                             xytext=base_xy,
                             arrowprops=dict(arrowstyle="-|>", color=colour,
                                             lw=1.0, shrinkA=3, shrinkB=1),
                             zorder=7)
            axes[2].add_patch(Circle((base_xy[0] + dx, base_xy[1] + dy), 60,
                                     fc=colour, ec=colour, lw=.8, alpha=.8,
                                     zorder=6))
    axes[2].plot(T["poles_moved"][:, 0], T["poles_moved"][:, 1], color=C_TRAJ,
                 lw=.7, alpha=.85, zorder=4, label="Run 1")
    axes[2].plot(T["poles_moved2"][:, 0], T["poles_moved2"][:, 1], color=C_TRAJ2,
                 lw=.7, alpha=.85, zorder=4, label="Run 2")
    axes[2].legend(frameon=False, fontsize=5.5, loc="lower left")

    # The displaced wall. Its new position was not digitised, so the section
    # that moved is marked on the trained boundary and the arrow is schematic;
    # the magnitude is given in the text.
    draw_poles(axes[3])
    # Select the boundary by the stretch of the LOOP it borders rather than by
    # a coordinate heuristic: WALL_STRETCH is where the route deviates and
    # where the profile peaks, so this marks the wall the result is about.
    # All boundary points, not just the largest cluster: this arena's outline
    # breaks into several pieces (2641, 555 and 202 points), so the section the
    # route deviates at sits in one of the smaller ones.
    all_wall = np.vstack([walls] + list(blocks.values()))
    lo, hi = [int(f / 100 * len(P)) for f in WALL_STRETCH]
    near_stretch = np.min(np.linalg.norm(
        all_wall[:, None, :] - P[None, lo:hi, :], axis=2), axis=1)
    south = all_wall[near_stretch < near_stretch.min() + 250]
    # The wall was ADDED, not displaced: comparing this run's arena photograph
    # with the baseline's shows the southern boundary still in place and an
    # extra wall inside it. So no "before" marking and no arrow -- just the
    # wall where it stood, which is a few centimetres outside the trained path.
    # Clip the drawn wall to the arena. The annotation's east end runs past the
    # boundary (to x = 1995, where the arena stops near 1857), so taking its
    # extent literally draws a wall sticking out of the room.
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path as MplPath
    hull_pts = np.vstack([walls] + list(blocks.values()))
    hull = hull_pts[ConvexHull(hull_pts).vertices]
    inside = MplPath(hull)
    xs = np.linspace(WALL_FIT["x0"], WALL_FIT["x1"], 400)
    ys = WALL_FIT["slope"] * xs + WALL_FIT["intercept"]
    keep = inside.contains_points(np.stack([xs, ys], 1), radius=-60)
    if keep.any():
        xs, ys = xs[keep], ys[keep]
    axes[3].plot(xs, ys, "-", color="k", lw=2.2, zorder=6,
                 solid_capstyle="round")
    print("[fig] wall drawn from x %.0f to %.0f (annotation gave %.0f to %.0f)"
          % (xs.min(), xs.max(), WALL_FIT["x0"], WALL_FIT["x1"]))
    axes[3].plot(T["wall"][:, 0], T["wall"][:, 1], color=C_TRAJ, lw=.7,
                 alpha=.85, zorder=4)

    # ---- profile around the loop -------------------------------------------
    # Absolute error with the two unmanipulated runs as a band, estimated in a
    # sliding window of 500 mm of path every 100 mm, exactly as for Arena 1.
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
            off = (pos - c + loop_mm / 2) % loop_mm - loop_mm / 2
            m = np.abs(off) <= WIN_MM / 2
            out.append(np.median(dist[m]) if m.sum() > 5 else np.nan)
        return np.array(out)

    x = 100 * centres / loop_mm
    b1, b2 = prof("baseline"), prof("replicate")
    axp.fill_between(x, np.minimum(b1, b2), np.maximum(b1, b2), color="0.55",
                     alpha=.30, lw=0, zorder=1, label="Trained arena (both runs)")
    for run, colour, lab in (("poles_out", "#4C72B0", "Poles removed"),
                             ("poles_moved", "#55A868", "Poles displaced (both runs)"),
                             ("poles_moved2", "#55A868", None),
                             ("wall", "#8172B2", "Wall displaced")):
        axp.plot(x, prof(run), "-", color=colour, lw=1.2, label=lab, zorder=3)

    axp.set_xlabel("Position around the loop (%)")
    axp.set_ylabel("Distance from\npath (mm)")
    axp.set_xlim(0, 100); axp.set_ylim(0, None)
    axp.legend(frameon=False, fontsize=5.5, ncol=4, loc="upper center",
               columnspacing=1.2)
    axp.text(0.004, 0.97, "E", transform=axp.transAxes, fontsize=6.5,
             fontweight="bold", va="top", ha="left")

    axp.axvspan(*WALL_STRETCH, color="#7A5C3E", alpha=.13, lw=0, zorder=0)
    for pos, txt in ((1, "pole"), (7, "pole"), (40, "wall")):
        axp.axvline(pos, color="0.35", lw=.8, ls=(0, (2, 2)), zorder=2)
        axp.text(pos, axp.get_ylim()[1] * .035, txt, fontsize=5, ha="center",
                 va="bottom", color="0.25",
                 bbox=dict(boxstyle="round,pad=0.12", fc="#F2E9DA", ec="none"))

    style.save(fig, NAME)
    print("[fig] %d poles, %d manipulated; added wall y = %.4f x %+.0f "
          "over x in [%.0f, %.0f]"
          % (len(pole), len(MANIP_POLES), WALL_FIT["slope"],
             WALL_FIT["intercept"], WALL_FIT["x0"], WALL_FIT["x1"]))


if __name__ == "__main__":
    main()
