#!/usr/bin/env python3
"""
SCRIPT_BuildArenaGeometry.py

Build a per-env arena-features artifact (walls + poles) from per-camera
annotated images.

Motivation
----------
The stitched arena image (arena.png) averages overlapping pixels from the two
cameras, which produces visible ghosting for tall obstacles that aren't on the
floor plane. Annotating that stitched image bakes the ghost into the training
geometry. Annotating each camera's individual top-down warp separately, and
back-projecting the annotations through the appropriate height plane using each
camera's calibration, avoids the averaging artifact and correctly recovers the
base coordinates of vertical features even when the base is occluded in the
image.

Annotation convention
---------------------
On arena_{cam}_annotated.png:
  - **Green polylines** trace the wall tops. Each green pixel is back-projected
    through z = WALL_HEIGHT_MM to recover (X, Y) at that height; because walls
    are vertical, that equals the base (X, Y).
  - **Blue dabs** mark the top of each cardboard pole. Each connected blue blob
    is reduced to one centroid (pixel space) and back-projected through
    z = POLE_HEIGHT_MM to recover the pole's (X, Y) at the top, which for a
    vertical pole equals its base (X, Y). Centroids visible from both cameras
    are merged across cameras (cluster radius POLE_MERGE_RADIUS_MM).

Input per env folder
--------------------
    arena_shark_annotated.png    green polylines + blue dabs on arena_shark_mask.png
    arena_tiger_annotated.png    green polylines + blue dabs on arena_tiger_mask.png

At least one must exist. Both are optional. Either the green or blue layer
can be absent on a given camera.

Output per env folder
---------------------
    arena_features.npz    keys:
        x_mm           (N,) float32   feature x-coordinates in world mm
        y_mm           (N,) float32   feature y-coordinates in world mm
        kind           (N,) uint8     0 = wall point, 1 = pole centre
        source_camera  (N,) uint8     0 = shark, 1 = tiger, 255 = merged (poles only)
        pole_radius_mm scalar float32 physical pole radius (carried for downstream
                                      clearance and labelling)
    arena_features_plot.png    diagnostic overlay: per-camera wall points and
                               merged pole centres on the stitched arena image.
    calibration/pose_{cam}.npz, pose_{cam}.json
        snapshotted camera calibration (K, R, t, H_img2mm) — copied in on first
        build so each env folder is a self-contained geometry record.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import json
import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from Library.DataProcessor import read_wall_mask, read_pole_mask, mask2coordinates


# ══════════════════════════════════════════════════════════════════════════════
# Settings  ← change these before running
# ══════════════════════════════════════════════════════════════════════════════

# Parent folders to walk. Each entry can be:
#   - a parent containing arena folders, e.g. AcquisitionArenas/<layout>/env_*/
#   - an arena folder containing env_* subfolders directly, e.g. TargetArenas/easy/
#   - an env_* folder itself
# Missing entries are skipped with a warning so the two parents below can be
# populated independently.
ROOTS: List[str] = [
    "AcquisitionArenas",   # arenas used for vision-guided sonar data collection
    "TargetArenas",        # arenas used for policy training/deployment
]

# Wall height (mm) — used to back-project annotated wall tops to their (X, Y)
# location. Walls are vertical, so top (X, Y) = base (X, Y).
WALL_HEIGHT_MM: float = 295.0

# Pole height (mm) — height of the cardboard poles whose tops are marked with
# blue dabs. Same back-projection logic as walls but at this height.
POLE_HEIGHT_MM: float = 610.0

# Pole radius (mm) — physical radius of the cardboard pole. Carried into the
# arena_features.npz output so downstream consumers (planner clearance,
# nearest-reflector labelling) don't have to hard-code it.
POLE_RADIUS_MM: float = 25.0

# Cross-camera merge radius (mm) — pole centroids from different cameras whose
# back-projected (X, Y) are within this distance are merged into a single pole
# at their mean position. 2 × POLE_RADIUS_MM is a comfortable margin for
# typical calibration residuals.
POLE_MERGE_RADIUS_MM: float = 50.0

# Source of per-camera calibration (pose_{cam}.npz, pose_{cam}.json). Copied
# into each env folder's calibration/ subdir on first build. Subsequent builds
# reuse the snapshotted copy, so recalibration in PyLorex doesn't retroactively
# change a built env.
PYLOREX_CALIBRATION_DIR: str = "../../PyLorex/PyLorex/Calibration/Results"

# Manual offset applied to shark's world coordinates to align them with tiger's
# world frame (= unified frame = arena grid). Mirrors Settings.shark2tiger_delta_{x,y}
# in PyLorex/LorexLib/Settings.py.
SHARK2TIGER_DELTA_X_MM: float = 0.0
SHARK2TIGER_DELTA_Y_MM: float = -1840.0


# ══════════════════════════════════════════════════════════════════════════════

CAMERA_CODES: Dict[str, int] = {"shark": 0, "tiger": 1}
CAMERA_COLOURS: Dict[str, str] = {"shark": "#e41a1c", "tiger": "#377eb8"}


# ─── Calibration I/O ──────────────────────────────────────────────────────────

def ensure_calibration_snapshot(env_dir: Path, source_dir: Path) -> Path:
    """Ensure pose_{cam}.{npz,json} files are present inside env_dir/calibration/.

    Copies from source_dir on first build. Returns the env's calibration dir.
    """
    dest = env_dir / "calibration"
    dest.mkdir(exist_ok=True)
    for camera in CAMERA_CODES:
        for ext in ("npz", "json"):
            src = source_dir / f"pose_{camera}.{ext}"
            dst = dest / f"pose_{camera}.{ext}"
            if dst.exists():
                continue
            if not src.exists():
                raise FileNotFoundError(
                    f"Calibration source missing: {src}. Check PYLOREX_CALIBRATION_DIR."
                )
            shutil.copy2(src, dst)
            print(f"    snapshotted {dst.relative_to(env_dir)}")
    return dest


def load_camera_calibration(calibration_dir: Path, camera: str) -> Dict[str, np.ndarray]:
    """Return {R, t, C_world} from the snapshotted pose_{cam}.npz.

    We avoid using K here on purpose: the stored K_scaled is not internally
    consistent with the stored H_undistorted (verified empirically — they
    disagree by ~300–600 mm at z=0), so a K-based back-projection through
    z = wall_height would drift in the same way. Instead we rely on two
    robust quantities: the arena-grid labels (which encode the correct z=0
    world mm by construction) and the camera centre C = -Rᵀ·t. Those two
    points define the 3D ray; intersecting it with z = wall_height gives the
    true wall base (X, Y).
    """
    npz_path = calibration_dir / f"pose_{camera}.npz"
    data = np.load(str(npz_path), allow_pickle=False)
    R = np.asarray(data["R_pnp"], dtype=np.float64)
    t = np.asarray(data["t_pnp"], dtype=np.float64).reshape(3, 1)
    C_world = (-R.T @ t).flatten()
    return {"R": R, "t": t, "C_world": C_world}


# ─── Geometry helpers ────────────────────────────────────────────────────────

def _pixels_to_world_at_height(
    cols: np.ndarray,
    rows: np.ndarray,
    calibration: Dict[str, np.ndarray],
    meta: dict,
    offset_xy: Optional[Tuple[float, float]],
    height_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project annotated arena pixels to world (X, Y) at z = height_mm.

    Pipeline:
      1. Each arena pixel → (X_0, Y_0, 0) world mm via the arena-grid affine
         (same as mask2coordinates — correct by construction in the unified
         /tiger frame; shark's offset is already baked into the grid).
      2. Camera centre C_world (from R, t) is translated into the same unified
         frame by adding the shark offset when applicable.
      3. Parametric ray P(λ) = C + λ(A − C) with A = (X_0, Y_0, 0); solve for
         the λ that gives P.z = height_mm and evaluate (P.x, P.y).
    """
    bounds = meta["arena_bounds_mm"]
    min_x = float(bounds["min_x"])
    max_y = float(bounds["max_y"])
    mm_per_px = float(meta["map_mm_per_px"])
    X0 = min_x + cols * mm_per_px + 0.5 * mm_per_px
    Y0 = max_y - rows * mm_per_px + 0.5 * mm_per_px

    C = calibration["C_world"].copy()
    if offset_xy is not None:
        C[0] += offset_xy[0]
        C[1] += offset_xy[1]

    if C[2] <= 0:
        raise ValueError(
            f"Camera Z ({C[2]:.1f} mm) must be above the floor for back-projection."
        )
    # P(λ) = C + λ (A − C). P.z = C_z + λ (0 − C_z) = height_mm
    #   ⇒ λ = 1 − height_mm / C_z
    lam = 1.0 - height_mm / C[2]
    Xh = C[0] + lam * (X0 - C[0])
    Yh = C[1] + lam * (Y0 - C[1])
    return Xh.astype(np.float32), Yh.astype(np.float32)


def backproject_walls(
    annotated_path: Path,
    calibration: Dict[str, np.ndarray],
    meta: dict,
    offset_xy: Optional[Tuple[float, float]],
    wall_height_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract green pixels from *annotated_path* and return their (X, Y) in world mm."""
    wall_mask = read_wall_mask(annotated_path)
    rows, cols = np.nonzero(wall_mask)
    if rows.size == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    return _pixels_to_world_at_height(cols, rows, calibration, meta, offset_xy, wall_height_mm)


def backproject_poles(
    annotated_path: Path,
    calibration: Dict[str, np.ndarray],
    meta: dict,
    offset_xy: Optional[Tuple[float, float]],
    pole_height_mm: float,
    min_blob_area_px: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find blue blobs on *annotated_path*, take centroid per blob, back-project
    each centroid through z = pole_height_mm. Returns (X_mm, Y_mm) per pole as
    seen by this camera (cross-camera merge happens upstream).

    Blobs smaller than `min_blob_area_px` are dropped as paint speckle.
    """
    pole_mask = read_pole_mask(annotated_path)
    if not pole_mask.any():
        return np.empty(0, np.float32), np.empty(0, np.float32)

    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        pole_mask.astype(np.uint8), connectivity=8
    )
    # Label 0 is background — skip it.
    if n_labels <= 1:
        return np.empty(0, np.float32), np.empty(0, np.float32)

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= min_blob_area_px
    if not keep.any():
        return np.empty(0, np.float32), np.empty(0, np.float32)

    cx = centroids[1:, 0][keep]  # column (float)
    cy = centroids[1:, 1][keep]  # row (float)
    return _pixels_to_world_at_height(cx, cy, calibration, meta, offset_xy, pole_height_mm)


def merge_pole_centroids(
    xs: np.ndarray, ys: np.ndarray, merge_radius_mm: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy single-pass clustering: any centroid within *merge_radius_mm* of
    an existing cluster centre joins that cluster (centre updated to the mean
    of its members). Returns (x_merged, y_merged).

    With only a handful of poles per arena this is O(N·K) and trivially fast.
    """
    if xs.size == 0:
        return xs, ys
    clusters: List[List[Tuple[float, float]]] = []
    for x, y in zip(xs.astype(np.float64), ys.astype(np.float64)):
        joined = False
        for c in clusters:
            cx = float(np.mean([p[0] for p in c]))
            cy = float(np.mean([p[1] for p in c]))
            if (x - cx) ** 2 + (y - cy) ** 2 <= merge_radius_mm ** 2:
                c.append((x, y))
                joined = True
                break
        if not joined:
            clusters.append([(x, y)])
    out_x = np.array([np.mean([p[0] for p in c]) for c in clusters], dtype=np.float32)
    out_y = np.array([np.mean([p[1] for p in c]) for c in clusters], dtype=np.float32)
    return out_x, out_y


# ─── Build per env ───────────────────────────────────────────────────────────

def find_env_dirs(root: Path) -> List[Path]:
    """Return env_* folders under *root*. Three accepted shapes:

    - *root* is itself an env_* folder → return [root].
    - *root* contains env_* subfolders directly (an arena folder, e.g.
      TargetArenas/easy/) → return those.
    - *root* contains arena-folder children which themselves contain env_*
      grandchildren (a parent like AcquisitionArenas/) → recurse one level
      and return env_* across all arena subfolders.
    """
    if root.name.startswith("env_"):
        return [root]
    direct = sorted(p for p in root.iterdir()
                    if p.is_dir() and p.name.startswith("env_"))
    if direct:
        return direct
    nested: List[Path] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        nested.extend(sorted(
            p for p in sub.iterdir()
            if p.is_dir() and p.name.startswith("env_")
        ))
    return nested


def load_meta(env_dir: Path) -> dict:
    with open(env_dir / "meta.json", "r") as f:
        return json.load(f)


def annotated_path(env_dir: Path, camera: str) -> Path:
    return env_dir / f"arena_{camera}_annotated.png"


def camera_offset(camera: str) -> Optional[Tuple[float, float]]:
    if camera == "shark":
        return (SHARK2TIGER_DELTA_X_MM, SHARK2TIGER_DELTA_Y_MM)
    return None


def build_features_for_env(env_dir: Path, pylorex_calib_dir: Path) -> Optional[Path]:
    """Build arena_features.npz in *env_dir* from per-camera annotated images.

    Returns the output path on success, None if no annotated image was found.
    """
    meta = load_meta(env_dir)
    calibration_dir = ensure_calibration_snapshot(env_dir, pylorex_calib_dir)

    wall_xs_parts: List[np.ndarray] = []
    wall_ys_parts: List[np.ndarray] = []
    wall_src_parts: List[np.ndarray] = []
    pole_xs_per_cam: List[np.ndarray] = []
    pole_ys_per_cam: List[np.ndarray] = []
    found_cameras: List[str] = []

    for camera, code in CAMERA_CODES.items():
        ann_path = annotated_path(env_dir, camera)
        if not ann_path.exists():
            continue
        calibration = load_camera_calibration(calibration_dir, camera)
        offset = camera_offset(camera)

        wx, wy = backproject_walls(ann_path, calibration, meta, offset, WALL_HEIGHT_MM)
        if wx.size:
            wall_xs_parts.append(wx)
            wall_ys_parts.append(wy)
            wall_src_parts.append(np.full(wx.shape, code, dtype=np.uint8))
            print(f"    {camera}: {wx.size:,} wall points")
        else:
            print(f"    {camera}: no green wall pixels")

        px, py = backproject_poles(ann_path, calibration, meta, offset, POLE_HEIGHT_MM)
        if px.size:
            pole_xs_per_cam.append(px)
            pole_ys_per_cam.append(py)
            print(f"    {camera}: {px.size} pole blob(s)")
        else:
            print(f"    {camera}: no blue pole blobs")

        found_cameras.append(camera)

    if not wall_xs_parts and not pole_xs_per_cam:
        print(f"    no usable annotations in {env_dir.name}")
        return None

    # Walls: concatenate as-is (per-camera duplicates are expected in the
    # overlap region and downstream consumers tolerate the point cloud).
    if wall_xs_parts:
        wall_x = np.concatenate(wall_xs_parts)
        wall_y = np.concatenate(wall_ys_parts)
        wall_src = np.concatenate(wall_src_parts)
    else:
        wall_x = np.empty(0, np.float32)
        wall_y = np.empty(0, np.float32)
        wall_src = np.empty(0, np.uint8)

    # Poles: merge across cameras.
    if pole_xs_per_cam:
        all_px = np.concatenate(pole_xs_per_cam)
        all_py = np.concatenate(pole_ys_per_cam)
        merged_px, merged_py = merge_pole_centroids(all_px, all_py, POLE_MERGE_RADIUS_MM)
        pole_src = np.full(merged_px.shape, 255, dtype=np.uint8)
        print(f"    merged poles across cameras: {all_px.size} detections → "
              f"{merged_px.size} unique poles")
    else:
        merged_px = np.empty(0, np.float32)
        merged_py = np.empty(0, np.float32)
        pole_src = np.empty(0, np.uint8)

    x_all = np.concatenate([wall_x, merged_px])
    y_all = np.concatenate([wall_y, merged_py])
    kind_all = np.concatenate([
        np.zeros(wall_x.shape, dtype=np.uint8),
        np.ones(merged_px.shape, dtype=np.uint8),
    ])
    src_all = np.concatenate([wall_src, pole_src])

    out_path = env_dir / "arena_features.npz"
    np.savez(
        out_path,
        x_mm=x_all,
        y_mm=y_all,
        kind=kind_all,
        source_camera=src_all,
        pole_radius_mm=np.float32(POLE_RADIUS_MM),
    )
    print(f"    saved {out_path.name}  "
          f"({wall_x.size:,} wall pts, {merged_px.size} poles, "
          f"cameras={','.join(found_cameras)})")

    plot_path = env_dir / "arena_features_plot.png"
    plot_features(
        env_dir, meta,
        wall_xs_parts, wall_ys_parts,
        merged_px, merged_py,
        found_cameras, plot_path,
    )
    print(f"    saved {plot_path.name}")

    return out_path


# ─── Diagnostic plot ─────────────────────────────────────────────────────────

def _load_legacy_walls(env_dir: Path, meta: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (x_mm, y_mm) from arena_annotated.png if it exists, else None."""
    legacy = env_dir / "arena_annotated.png"
    if not legacy.exists():
        return None
    mask = read_wall_mask(legacy)
    x, y = mask2coordinates(mask, meta)
    if x.size == 0:
        return None
    return x, y


def plot_features(
    env_dir: Path,
    meta: dict,
    wall_xs_parts: List[np.ndarray],
    wall_ys_parts: List[np.ndarray],
    pole_xs: np.ndarray,
    pole_ys: np.ndarray,
    cameras: List[str],
    out_path: Path,
) -> None:
    """Save a diagnostic overlay of the extracted features (walls + poles, plus
    the legacy arena_annotated.png pass when available for comparison)."""
    bounds = meta["arena_bounds_mm"]
    min_x, max_x = float(bounds["min_x"]), float(bounds["max_x"])
    min_y, max_y = float(bounds["min_y"]), float(bounds["max_y"])

    arena_img_path = env_dir / "arena.png"
    arena_img = cv2.imread(str(arena_img_path)) if arena_img_path.exists() else None

    legacy = _load_legacy_walls(env_dir, meta)

    has_legacy = legacy is not None
    fig, axes = plt.subplots(
        1, 2 if has_legacy else 1,
        figsize=(14 if has_legacy else 7, 7),
        squeeze=False,
    )
    panels = axes[0]

    def draw_background(ax):
        if arena_img is not None:
            ax.imshow(
                cv2.cvtColor(arena_img, cv2.COLOR_BGR2RGB),
                extent=(min_x, max_x, min_y, max_y),
                origin="upper",
                alpha=0.55,
                zorder=0,
            )
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    # Panel 1: new per-camera geometry + merged poles
    ax_new = panels[0]
    draw_background(ax_new)
    for xs, ys, cam in zip(wall_xs_parts, wall_ys_parts, cameras):
        ax_new.scatter(
            xs, ys, s=1.5, c=CAMERA_COLOURS.get(cam, "k"),
            label=f"{cam} walls (n={xs.size:,})", alpha=0.8, zorder=2,
        )
    pole_handle = None
    if pole_xs.size:
        for px, py in zip(pole_xs, pole_ys):
            circ = plt.Circle(
                (px, py), POLE_RADIUS_MM,
                facecolor="#984ea3", edgecolor="black", linewidth=0.6,
                alpha=0.85, zorder=3,
            )
            ax_new.add_patch(circ)
        pole_handle = mpatches.Patch(
            facecolor="#984ea3", edgecolor="black", linewidth=0.6,
            label=f"poles (n={pole_xs.size}, r={POLE_RADIUS_MM:.0f} mm)",
        )
    ax_new.set_title(
        f"Walls @ z={WALL_HEIGHT_MM:.0f} mm + poles @ z={POLE_HEIGHT_MM:.0f} mm\n"
        f"{env_dir.name}"
    )
    handles, labels = ax_new.get_legend_handles_labels()
    if pole_handle is not None:
        handles.append(pole_handle)
        labels.append(pole_handle.get_label())
    ax_new.legend(handles, labels, loc="upper right", markerscale=4, fontsize=8)

    # Panel 2: legacy comparison
    if has_legacy:
        lx, ly = legacy
        ax_old = panels[1]
        draw_background(ax_old)
        ax_old.scatter(lx, ly, s=1.5, c="#4daf4a",
                       label=f"legacy (n={lx.size:,})", alpha=0.8, zorder=2)
        ax_old.set_title("Legacy arena_annotated.png\n(floor-plane homography)")
        ax_old.legend(loc="upper right", markerscale=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def process(root: Path, pylorex_calib_dir: Path) -> None:
    env_dirs = find_env_dirs(root)
    if not env_dirs:
        print(f"  no env_* subfolders under {root}; skipping")
        return
    print(f"Processing {len(env_dirs)} env folder(s) under {root}")
    for env_dir in env_dirs:
        print(f"  {env_dir.relative_to(root.parent) if root.parent in env_dir.parents else env_dir}")
        build_features_for_env(env_dir, pylorex_calib_dir)


def _resolve(path_str: str) -> Path:
    """Resolve *path_str* relative to the script's directory when it's relative."""
    p = Path(path_str)
    if not p.is_absolute():
        p = (Path(__file__).resolve().parent / p)
    return p.resolve()


if __name__ == "__main__":
    pylorex_calib_dir = _resolve(PYLOREX_CALIBRATION_DIR)
    if not pylorex_calib_dir.is_dir():
        raise SystemExit(
            f"PYLOREX_CALIBRATION_DIR not found: {pylorex_calib_dir}\n"
            f"(resolved from '{PYLOREX_CALIBRATION_DIR}')"
        )
    for path_str in ROOTS:
        path = _resolve(path_str)
        if not path.is_dir():
            print(f"Skipping {path_str}: not found at {path}")
            continue
        process(path, pylorex_calib_dir)
