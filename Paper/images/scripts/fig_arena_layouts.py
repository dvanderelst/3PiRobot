"""Figure: the six arenas used to collect inverse-model training data.

Panel  A    : arena 1 with the acquisition plan -- the positions at which
              the robot ensonified the arena.
Panels B-E  : arenas 2-5 as top-down warps with the digitized geometry
              (wall tops + poles) overlaid -- a cleaned-up annotated image.
Panel  F    : arena 6, the open far-range arena. No interior walls, so sight
              lines reach the boundary; this is the session that supplied the
              beyond-1.7 m echoes the first five arenas could not produce.

Reads the orthorectified top-down (arena.png) and overlays geometry in world
mm using the warp bounds from meta.json (imshow extent), so no projection is
needed. -> Paper/images/fig_arena_layouts.pdf
"""

import glob
import json

import numpy as np

import style
from paths import ACQ_ARENAS

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ARENAS = ["Acquisition01", "Acquisition02", "Acquisition03", "Acquisition04",
          "Acquisition05", "Acquisition06"]
PLAN_ARENA = "Acquisition01"          # shown as the sampling-plan panel
NCOLS = 3                             # 3 -> 2x3 grid; 6 -> single row
WALL_C, POLE_C, POS_C = "#00e0ff", "#ff2d2d", "#ffd400"


def _one(p):
    g = sorted(glob.glob(p))
    if not g:
        raise FileNotFoundError(p)
    return g[0]


def load_arena(name):
    env = _one(str(ACQ_ARENAS / name / "env_*"))
    meta = json.load(open(_one(env + "/meta.json")))
    b = meta["arena_bounds_mm"]
    extent = (b["min_x"], b["max_x"], b["min_y"], b["max_y"])
    img_path = _one(env + "/arena.png")
    img = plt.imread(img_path)
    d = np.load(_one(env + "/arena_features.npz"), allow_pickle=True)
    kind = np.array(d["kind"])
    x, y = np.array(d["x_mm"]), np.array(d["y_mm"])
    walls = (x[kind == 0], y[kind == 0])
    poles = (x[kind == 1], y[kind == 1])
    plan = json.load(open(_one(str(ACQ_ARENAS / name / "plans" / "plan_*.json"))))
    pos = np.array(plan["positions"], dtype=float)
    return dict(img=img, extent=extent, walls=walls, poles=poles, pos=pos)


def common_bbox(arenas, margin=150.0):
    xs = np.concatenate([np.r_[a["walls"][0], a["poles"][0]] for a in arenas])
    ys = np.concatenate([np.r_[a["walls"][1], a["poles"][1]] for a in arenas])
    return (xs.min() - margin, xs.max() + margin, ys.min() - margin, ys.max() + margin)


def draw_base(ax, A, xlim, ylim):
    ax.imshow(A["img"], extent=A["extent"], origin="upper", interpolation="nearest")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    data = {n: load_arena(n) for n in ARENAS}
    bx0, bx1, by0, by1 = common_bbox(list(data.values()))
    xlim, ylim = (bx0, bx1), (by0, by1)

    nrows = int(np.ceil(len(ARENAS) / NCOLS))
    panel_w = style.WIDTH_2COL / NCOLS
    panel_h = panel_w * (by1 - by0) / (bx1 - bx0)
    fig, axes = plt.subplots(nrows, NCOLS,
                             figsize=(style.WIDTH_2COL, nrows * panel_h + 0.55))
    axes = np.atleast_1d(axes).ravel()
    letters = "ABCDEF"

    for ax, name, letter in zip(axes, ARENAS, letters):
        A = data[name]
        draw_base(ax, A, xlim, ylim)
        if name == PLAN_ARENA:
            ax.scatter(A["walls"][0], A["walls"][1], s=0.8, c=WALL_C, alpha=0.35, lw=0)
            ax.scatter(A["pos"][:, 0], A["pos"][:, 1], s=5, c=POS_C,
                       edgecolors="k", linewidths=0.2, zorder=4)
            sub = f"Arena {name[-1]} (plan)"
        else:
            ax.scatter(A["walls"][0], A["walls"][1], s=1.2, c=WALL_C, alpha=0.6, lw=0)
            sub = f"Arena {name[-1]}"
        ax.scatter(A["poles"][0], A["poles"][1], s=20, c=POLE_C,
                   edgecolors="k", linewidths=0.35, zorder=5)
        ax.set_title(sub, fontsize=8)
        ax.text(0.05, 0.96, letter, transform=ax.transAxes, va="top", ha="left",
                fontweight="bold", fontsize=8, color="w",
                bbox=dict(boxstyle="round,pad=0.12", fc="k", ec="none", alpha=0.6))

    for ax in axes[len(ARENAS):]:
        ax.axis("off")

    # 1 m scale bar in panel A
    frac = 1000.0 / (bx1 - bx0)
    axes[0].plot([0.07, 0.07 + frac], [0.06, 0.06], transform=axes[0].transAxes,
                 color="k", lw=2)
    axes[0].text(0.07, 0.10, "1 m", transform=axes[0].transAxes, fontsize=7)

    # shared legend below the row
    handles = [
        Line2D([], [], color=WALL_C, lw=3, label="Wall"),
        Line2D([], [], marker="o", ls="", mfc=POLE_C, mec="k", ms=6, label="Pole"),
        Line2D([], [], marker="o", ls="", mfc=POS_C, mec="k", ms=5, label="Ping position"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=8)

    bottom = 0.14 if nrows == 1 else 0.07
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=bottom,
                        wspace=0.06, hspace=0.18)
    style.save(fig, "fig_arena_layouts")
    fig.savefig(style.IMAGES / "fig_arena_layouts.png", dpi=150)  # for quick inspection


if __name__ == "__main__":
    main()
