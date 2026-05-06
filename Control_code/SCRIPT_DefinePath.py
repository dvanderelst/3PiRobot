#!/usr/bin/env python3
"""
SCRIPT_DefinePath.py

Interactive tool to draw three things per arena:
  1. A closed-loop target path (red).
  2. A start box (blue) — robot release region used by training.
  3. A start arrow (blue) — release heading inside the box.

Three-phase picker driven by key presses:
  Phase 1 (path)   : left=add waypoint, right=undo, Enter=next, r=reset, Esc=abandon
  Phase 2 (box)    : left=corner (×2),  right=clear,  Enter=next, r=reset, Esc=abandon
  Phase 3 (arrow)  : left=base then tip, right=clear, Enter=save, r=reset, Esc=abandon

Re-running on an arena with an existing target_path.json prompts:
  [o]verwrite path / [b]ox-only / [s]kip

In box-only mode, the existing waypoints are loaded and shown as context;
the picker starts at phase 2 so you can update box/arrow without redrawing
the loop.

Outputs (TargetArenas/<arena>/):
  target_path.json   ordered waypoints + start_box + start_arrow
  target_path.png    walls + distance field + path + box + arrow
"""

import json
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.spatial import cKDTree

from Library import Settings as _settings
_settings.data_folder = "TargetArenas"

from Library.EnvironmentSimulator import EnvironmentSimulator


# ── Settings ──────────────────────────────────────────────────────────────────
ARENAS:             List[str] = ["hard", "easy", "loop1", "loop2"]
ARENAS_ROOT:        str       = "TargetArenas"
GRID_RESOLUTION_MM: float     = 10.0
COLORMAP:           str       = "viridis"
PADDING_MM:         float     = 100.0


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_distance_field(walls: np.ndarray, resolution: float, padding: float):
    """Return (xs, ys, dist) — grid covering wall bbox + padding, dist to nearest wall."""
    x_min, y_min = walls.min(axis=0) - padding
    x_max, y_max = walls.max(axis=0) + padding
    xs = np.arange(x_min, x_max + resolution, resolution)
    ys = np.arange(y_min, y_max + resolution, resolution)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    tree = cKDTree(walls)
    dists, _ = tree.query(pts, k=1)
    return xs, ys, dists.reshape(gx.shape)


def _render_background(ax, walls: np.ndarray, xs, ys, dist, alpha_field=0.85):
    im = ax.imshow(
        dist,
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower", cmap=COLORMAP, alpha=alpha_field, zorder=1,
    )
    ax.scatter(walls[:, 0], walls[:, 1], s=1.0, c="black", linewidths=0, zorder=2)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")
    return im


def _normalise_box(c0: Tuple[float, float], c1: Tuple[float, float]
                   ) -> Tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) regardless of click order."""
    x_min = min(c0[0], c1[0]); x_max = max(c0[0], c1[0])
    y_min = min(c0[1], c1[1]); y_max = max(c0[1], c1[1])
    return x_min, y_min, x_max, y_max


# ══════════════════════════════════════════════════════════════════════════════
# Three-phase interactive picker
# ══════════════════════════════════════════════════════════════════════════════

def collect_path_and_starts(
    arena: str,
    walls: np.ndarray,
    existing_waypoints: Optional[List[Tuple[float, float]]] = None,
) -> Optional[dict]:
    """Interactive picker for waypoints + box + arrow.

    If existing_waypoints is provided, they are pre-loaded and the picker
    starts at phase 2 (box).

    Returns dict {'waypoints', 'box_corners', 'arrow_pts'} or None on Esc.
    """
    print("  Computing distance field …")
    xs, ys, dist = compute_distance_field(walls, GRID_RESOLUTION_MM, PADDING_MM)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = _render_background(ax, walls, xs, ys, dist)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Distance to nearest wall (mm)")

    state = {
        "phase":     2 if existing_waypoints else 1,
        "abandoned": False,
        "waypoints": list(existing_waypoints) if existing_waypoints else [],
        "box":       [],   # list of (x, y) corners; len 0..2
        "arrow":     [],   # list of (x, y); len 0..2
        "hover":     None,
    }

    # ── Drawing artists (reused by redraw) ─────────────────────────────────────
    line_path,   = ax.plot([], [], "-", color="red", linewidth=2.5, zorder=10)
    line_open,   = ax.plot([], [], "-", color="red", linewidth=1.5,
                           linestyle=":", alpha=0.6, zorder=10)
    pts_path,    = ax.plot([], [], "o", color="red", markersize=7,
                           markeredgecolor="black", zorder=11)

    box_artist = Rectangle((0, 0), 0, 0, edgecolor="blue", facecolor="blue",
                           alpha=0.25, linewidth=2.0, zorder=12)
    box_artist.set_visible(False)
    ax.add_patch(box_artist)

    arrow_artist = FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", color="blue",
                                   mutation_scale=25, linewidth=2.5, zorder=13)
    arrow_artist.set_visible(False)
    ax.add_patch(arrow_artist)

    # ── Title / instructions per phase ─────────────────────────────────────────
    def update_title():
        n_w  = len(state["waypoints"])
        phase = state["phase"]
        if phase == 1:
            top = (f"Phase 1/3 — waypoints ({n_w})  |  "
                   "left=add  right=undo  r=reset  Enter=next  Esc=abandon")
        elif phase == 2:
            top = (f"Phase 2/3 — start box ({len(state['box'])}/2 corners)  |  "
                   "left=corner  right=clear  r=reset  Enter=next  Esc=abandon")
        else:
            top = (f"Phase 3/3 — start arrow ({len(state['arrow'])}/2 pts)  |  "
                   "left=base then tip  right=clear  Enter=save  Esc=abandon")
        fig.suptitle(f"{arena}   |   {top}", fontsize=10)
        ax.set_title("draw path → release box → release direction", fontsize=9)

    def redraw():
        # Path
        wps = state["waypoints"]
        if not wps:
            pts_path.set_data([], [])
            line_path.set_data([], [])
            line_open.set_data([], [])
        else:
            xs_, ys_ = zip(*wps)
            pts_path.set_data(xs_, ys_)
            if len(wps) >= 2:
                line_path.set_data(xs_, ys_)
                line_open.set_data([xs_[-1], xs_[0]], [ys_[-1], ys_[0]])
            else:
                line_path.set_data([], [])
                line_open.set_data([], [])

        # Box
        bc = state["box"]
        if len(bc) == 2:
            xm, ym, xM, yM = _normalise_box(bc[0], bc[1])
            box_artist.set_xy((xm, ym))
            box_artist.set_width(xM - xm)
            box_artist.set_height(yM - ym)
            box_artist.set_visible(True)
        elif len(bc) == 1 and state["phase"] == 2 and state["hover"] is not None:
            xm, ym, xM, yM = _normalise_box(bc[0], state["hover"])
            box_artist.set_xy((xm, ym))
            box_artist.set_width(xM - xm)
            box_artist.set_height(yM - ym)
            box_artist.set_visible(True)
        else:
            box_artist.set_visible(False)

        # Arrow
        ap = state["arrow"]
        if len(ap) == 2:
            arrow_artist.set_positions(ap[0], ap[1])
            arrow_artist.set_visible(True)
        elif len(ap) == 1 and state["phase"] == 3 and state["hover"] is not None:
            arrow_artist.set_positions(ap[0], state["hover"])
            arrow_artist.set_visible(True)
        else:
            arrow_artist.set_visible(False)

        update_title()
        fig.canvas.draw_idle()

    # ── Event handlers ─────────────────────────────────────────────────────────
    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        xy = (float(event.xdata), float(event.ydata))
        phase = state["phase"]
        if phase == 1:
            if event.button == 1:
                state["waypoints"].append(xy)
            elif event.button == 3 and state["waypoints"]:
                state["waypoints"].pop()
        elif phase == 2:
            if event.button == 1:
                if len(state["box"]) < 2:
                    state["box"].append(xy)
            elif event.button == 3:
                state["box"] = []
        elif phase == 3:
            if event.button == 1:
                if len(state["arrow"]) < 2:
                    state["arrow"].append(xy)
            elif event.button == 3:
                state["arrow"] = []
        redraw()

    def on_motion(event):
        if event.inaxes != ax or event.xdata is None:
            state["hover"] = None
            return
        state["hover"] = (float(event.xdata), float(event.ydata))
        if state["phase"] in (2, 3):
            redraw()

    def on_key(event):
        if event.key == "escape":
            state["abandoned"] = True
            plt.close(fig)
            return
        if event.key in ("enter", "n"):
            phase = state["phase"]
            if phase == 1:
                if len(state["waypoints"]) < 3:
                    print(f"  need ≥ 3 waypoints, have {len(state['waypoints'])}")
                    return
                state["phase"] = 2
            elif phase == 2:
                if len(state["box"]) != 2:
                    print(f"  need 2 corners for box, have {len(state['box'])}")
                    return
                state["phase"] = 3
            else:
                if len(state["arrow"]) != 2:
                    print(f"  need base + tip for arrow, have {len(state['arrow'])}")
                    return
                plt.close(fig)
                return
            redraw()
        elif event.key == "r":
            phase = state["phase"]
            if phase == 1:
                state["waypoints"] = []
            elif phase == 2:
                state["box"] = []
            else:
                state["arrow"] = []
            redraw()

    redraw()
    cid_c = fig.canvas.mpl_connect("button_press_event",  on_click)
    cid_k = fig.canvas.mpl_connect("key_press_event",     on_key)
    cid_m = fig.canvas.mpl_connect("motion_notify_event", on_motion)
    plt.show()
    fig.canvas.mpl_disconnect(cid_c)
    fig.canvas.mpl_disconnect(cid_k)
    fig.canvas.mpl_disconnect(cid_m)

    if state["abandoned"]:
        return None
    return {
        "waypoints": state["waypoints"],
        "box":       state["box"],
        "arrow":     state["arrow"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Static viz
# ══════════════════════════════════════════════════════════════════════════════

def save_path_viz(
    arena: str,
    walls: np.ndarray,
    points: List[Tuple[float, float]],
    box: dict,
    arrow: dict,
    out_path: str,
) -> None:
    xs, ys, dist = compute_distance_field(walls, GRID_RESOLUTION_MM, PADDING_MM)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = _render_background(ax, walls, xs, ys, dist, alpha_field=0.6)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label="Distance to nearest wall (mm)")

    if points:
        xs_, ys_ = zip(*points)
        closed_x = list(xs_) + [xs_[0]]
        closed_y = list(ys_) + [ys_[0]]
        ax.plot(closed_x, closed_y, "-", color="red", linewidth=2.5, zorder=4,
                label="target path")
        ax.plot(xs_, ys_, "o", color="red", markersize=6,
                markeredgecolor="black", zorder=5)
        ax.plot(xs_[0], ys_[0], "s", color="yellow", markersize=10,
                markeredgecolor="black", zorder=6, label="waypoint 0")

        for i in range(len(points)):
            a = points[i]
            b = points[(i + 1) % len(points)]
            mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
            dx, dy = b[0] - a[0], b[1] - a[1]
            n = np.hypot(dx, dy)
            if n < 1e-6:
                continue
            ux, uy = dx / n, dy / n
            head_len = min(60.0, 0.3 * n)
            ax.annotate(
                "",
                xy=(mx + ux * head_len * 0.5, my + uy * head_len * 0.5),
                xytext=(mx - ux * head_len * 0.5, my - uy * head_len * 0.5),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                zorder=5,
            )

    if box:
        rect = Rectangle(
            (box["x_min_mm"], box["y_min_mm"]),
            box["x_max_mm"] - box["x_min_mm"],
            box["y_max_mm"] - box["y_min_mm"],
            edgecolor="blue", facecolor="blue", alpha=0.25, linewidth=2.0,
            zorder=7, label="release box",
        )
        ax.add_patch(rect)

    if arrow:
        ax.add_patch(FancyArrowPatch(
            (arrow["base_x_mm"], arrow["base_y_mm"]),
            (arrow["tip_x_mm"],  arrow["tip_y_mm"]),
            arrowstyle="->", color="blue", mutation_scale=25, linewidth=2.5, zorder=8,
        ))

    ax.set_title(
        f"{arena}: target path  ({len(points)} waypoints)  "
        f"+ release box & arrow",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _load_walls(arena: str) -> np.ndarray:
    sim = EnvironmentSimulator(arena)
    walls = sim.arena.walls
    if len(walls) == 0:
        raise ValueError(f"No walls found for arena '{arena}'")
    return walls


def _prompt_existing(json_path: str) -> Tuple[str, Optional[List[Tuple[float, float]]]]:
    """Ask what to do with an existing target_path.json. Returns (mode, waypoints)
    where mode ∈ {'overwrite','box_only','skip'}."""
    print(f"  '{json_path}' already exists.")
    resp = input("  [o]verwrite path / [b]ox-only / [s]kip ? ").strip().lower()
    if resp.startswith("s"):
        return "skip", None
    if resp.startswith("b"):
        with open(json_path) as f:
            data = json.load(f)
        wps = [(float(w["x_mm"]), float(w["y_mm"])) for w in data["waypoints"]]
        return "box_only", wps
    if resp.startswith("o"):
        return "overwrite", None
    print("  unrecognised — skipping.")
    return "skip", None


def main() -> None:
    for arena in ARENAS:
        print(f"\n=== {arena} ===")
        arena_dir = os.path.join(ARENAS_ROOT, arena)
        if not os.path.isdir(arena_dir):
            print(f"  arena directory '{arena_dir}' not found — skipping.")
            continue
        json_path = os.path.join(arena_dir, "target_path.json")
        png_path  = os.path.join(arena_dir, "target_path.png")

        existing_waypoints: Optional[List[Tuple[float, float]]] = None
        if os.path.exists(json_path):
            mode, existing_waypoints = _prompt_existing(json_path)
            if mode == "skip":
                continue

        walls = _load_walls(arena)
        result = collect_path_and_starts(arena, walls, existing_waypoints)
        if result is None:
            print("  abandoned (Esc).")
            continue

        waypoints = result["waypoints"]
        if len(waypoints) < 3:
            print(f"  skipped: only {len(waypoints)} waypoints (need ≥ 3).")
            continue

        c0, c1 = result["box"]
        x_min, y_min, x_max, y_max = _normalise_box(c0, c1)
        box = {
            "x_min_mm": float(x_min), "y_min_mm": float(y_min),
            "x_max_mm": float(x_max), "y_max_mm": float(y_max),
        }
        ap = result["arrow"]
        arrow = {
            "base_x_mm": float(ap[0][0]), "base_y_mm": float(ap[0][1]),
            "tip_x_mm":  float(ap[1][0]), "tip_y_mm":  float(ap[1][1]),
        }

        save_data = {
            "arena":       arena,
            "n_waypoints": len(waypoints),
            "waypoints":   [{"x_mm": float(x), "y_mm": float(y)} for x, y in waypoints],
            "start_box":   box,
            "start_arrow": arrow,
        }
        with open(json_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Saved {json_path}  ({len(waypoints)} waypoints + box + arrow)")

        save_path_viz(arena, walls, waypoints, box, arrow, png_path)
        print(f"  Saved {png_path}")


if __name__ == "__main__":
    main()
