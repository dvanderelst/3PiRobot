#!/usr/bin/env python3
"""
SCRIPT_BuildArenaGeometry.py

Build a per-env wall geometry artifact from per-camera annotated images.

Motivation
----------
The stitched arena image (arena.png) averages overlapping pixels from the two
cameras, which produces visible ghosting for tall obstacles that aren't on the
floor plane. Annotating that stitched image bakes the ghost into the training
geometry. Annotating each camera's individual top-down warp separately, and
back-projecting the annotations through the z = WALL_HEIGHT_MM plane using each
camera's calibration, avoids the averaging artifact and correctly recovers the
wall base coordinates even when the base is occluded in the image.

Annotation convention
---------------------
Draw green polylines along the **wall tops** on arena_{cam}_mask.png and save as
arena_{cam}_annotated.png. The builder back-projects each green pixel through
z = WALL_HEIGHT_MM to recover the (X, Y) of the wall at that height; because
walls are vertical, that equals the base (X, Y).

Input per env folder
--------------------
    arena_shark_annotated.png    green polylines drawn on arena_shark_mask.png
    arena_tiger_annotated.png    green polylines drawn on arena_tiger_mask.png

At least one must exist. Both are optional — if only one camera's annotation
is available, the output contains only that camera's walls.

Output per env folder
---------------------
    arena_walls.npz    keys:
        x_mm           (N,) float32   wall x-coordinates in world mm
        y_mm           (N,) float32   wall y-coordinates in world mm
        source_camera  (N,) uint8     0 = shark, 1 = tiger
    arena_walls_plot.png   diagnostic overlay: new per-camera points on the
                           stitched arena image, with legacy annotation for
                           visual comparison when available.
    calibration/pose_{cam}.npz, pose_{cam}.json
        snapshotted camera calibration (K, R, t, H_img2mm) — copied in on first
        build so each env folder is a self-contained geometry record.

Downstream (DataProcessor.load_arena_masks) prefers arena_walls.npz over the
legacy arena_annotated.png path when present.
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
import matplotlib.pyplot as plt
import numpy as np

from Library.DataProcessor import read_wall_mask, mask2coordinates


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

def backproject_annotation(
    annotated_path: Path,
    calibration: Dict[str, np.ndarray],
    meta: dict,
    offset_xy: Optional[Tuple[float, float]],
    wall_height_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract green pixels from *annotated_path* and return their (X, Y) in world mm.

    Pipeline:
      1. Each green arena pixel → (X_0, Y_0, 0) world mm via the arena-grid
         affine (same as mask2coordinates — this is correct by construction in
         the unified/tiger frame; shark's offset is already baked into the grid).
      2. Camera centre C_world (from R, t) is translated into the same unified
         frame by adding the shark offset when applicable.
      3. Parametric ray P(λ) = C + λ(A − C) with A = (X_0, Y_0, 0); solve for
         the λ that gives P.z = wall_height_mm and evaluate (P.x, P.y).
    """
    wall_mask = read_wall_mask(annotated_path)
    rows, cols = np.nonzero(wall_mask)
    if rows.size == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)

    # Step 1: arena pixel → (X_0, Y_0) at z=0 in the unified/arena frame.
    # Matches mask2coordinates exactly (including the +0.5 pixel-centre offset).
    bounds = meta["arena_bounds_mm"]
    min_x = float(bounds["min_x"])
    max_y = float(bounds["max_y"])
    mm_per_px = float(meta["map_mm_per_px"])
    X0 = min_x + cols * mm_per_px + 0.5 * mm_per_px
    Y0 = max_y - rows * mm_per_px + 0.5 * mm_per_px

    # Step 2: camera centre in the unified frame.
    C = calibration["C_world"].copy()  # in the camera's own world frame
    if offset_xy is not None:
        C[0] += offset_xy[0]
        C[1] += offset_xy[1]

    # Step 3: ray interpolation to z = wall_height_mm.
    # P(λ) = C + λ (A − C).  P.z = C_z + λ (0 − C_z) = wall_height_mm
    #   ⇒ λ = 1 − wall_height_mm / C_z
    if C[2] <= 0:
        raise ValueError(
            f"Camera Z ({C[2]:.1f} mm) must be above the floor for back-projection."
        )
    lam = 1.0 - wall_height_mm / C[2]
    Xh = C[0] + lam * (X0 - C[0])
    Yh = C[1] + lam * (Y0 - C[1])

    return Xh.astype(np.float32), Yh.astype(np.float32)


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


def build_walls_for_env(env_dir: Path, pylorex_calib_dir: Path) -> Optional[Path]:
    """Build arena_walls.npz in *env_dir* from per-camera annotated images.

    Returns the output path on success, None if no annotated image was found.
    """
    meta = load_meta(env_dir)
    calibration_dir = ensure_calibration_snapshot(env_dir, pylorex_calib_dir)

    xs_parts: List[np.ndarray] = []
    ys_parts: List[np.ndarray] = []
    src_parts: List[np.ndarray] = []
    found_cameras: List[str] = []

    for camera, code in CAMERA_CODES.items():
        ann_path = annotated_path(env_dir, camera)
        if not ann_path.exists():
            continue
        calibration = load_camera_calibration(calibration_dir, camera)
        x_mm, y_mm = backproject_annotation(
            ann_path, calibration, meta, camera_offset(camera), WALL_HEIGHT_MM
        )
        if x_mm.size == 0:
            print(f"    {camera}: {ann_path.name} found but no green pixels")
            continue
        xs_parts.append(x_mm)
        ys_parts.append(y_mm)
        src_parts.append(np.full(x_mm.shape, code, dtype=np.uint8))
        found_cameras.append(camera)
        print(f"    {camera}: {x_mm.size:,} wall points")

    if not xs_parts:
        print(f"    no annotated per-camera images found in {env_dir.name}")
        return None

    x_all = np.concatenate(xs_parts)
    y_all = np.concatenate(ys_parts)
    src_all = np.concatenate(src_parts)

    out_path = env_dir / "arena_walls.npz"
    np.savez(out_path, x_mm=x_all, y_mm=y_all, source_camera=src_all)
    print(f"    saved {out_path.name}  "
          f"({x_all.size:,} points, cameras={','.join(found_cameras)})")

    plot_path = env_dir / "arena_walls_plot.png"
    plot_walls(env_dir, meta, xs_parts, ys_parts, found_cameras, plot_path)
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


def plot_walls(
    env_dir: Path,
    meta: dict,
    xs_parts: List[np.ndarray],
    ys_parts: List[np.ndarray],
    cameras: List[str],
    out_path: Path,
) -> None:
    """Save a diagnostic overlay of the new geometry (and legacy, if present)."""
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

    # Panel 1: new per-camera geometry
    ax_new = panels[0]
    draw_background(ax_new)
    for xs, ys, cam in zip(xs_parts, ys_parts, cameras):
        ax_new.scatter(
            xs, ys, s=1.5, c=CAMERA_COLOURS.get(cam, "k"),
            label=f"{cam} (n={xs.size:,})", alpha=0.8, zorder=2,
        )
    ax_new.set_title(f"Per-camera walls (tops @ z={WALL_HEIGHT_MM:.0f} mm)\n{env_dir.name}")
    ax_new.legend(loc="upper right", markerscale=4, fontsize=8)

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
        build_walls_for_env(env_dir, pylorex_calib_dir)


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
