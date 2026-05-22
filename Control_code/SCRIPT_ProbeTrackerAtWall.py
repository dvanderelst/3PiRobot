"""
SCRIPT_ProbeTrackerAtWall.py

Diagnostic for the tracker-vs-physical-wall geometry check.

Continuously polls the tracker every POLL_INTERVAL_S, prints the per-camera
detection of the robot's ArUco marker, and renders a top-down plot that
overlays the marker position on the planner's wall geometry from
arena_features.npz. Plot is written to TempOutput/probe.png on every cycle.

How to use:
  1. Set ARENA_NAME to the arena whose geometry you want to compare against.
  2. Run this script.
  3. Park the robot manually with one of its edges flush against a wall.
  4. Read the printed (x, y) and open TempOutput/probe.png — the planner
     walls are small green dots, the robot is a blue circle (radius =
     robot body), and the yaw arrow shows the marker's forward axis. The
     gap between the robot circle and the green wall dots tells you the
     wall-geometry error directly.
  5. Ctrl-C to stop.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from LorexLib import ServerClient
from LorexLib import Settings as LorexSettings

from Library import Settings


ROBOT_NUMBER     = 1
ARENA_NAME       = "Acquisition03"
POLL_INTERVAL_S  = 1.0
ROBOT_RADIUS_MM  = 48.0  # Pololu 3pi+ 2040 (~96 mm diameter)
PLOT_PATH        = Path(__file__).resolve().parent / "TempOutput" / "probe.png"


def _latest_arena_features(arena_name: str) -> Path:
    """Return the most-recent arena_features.npz under AcquisitionArenas/<arena>/env_*."""
    root = Path(__file__).resolve().parent / "AcquisitionArenas" / arena_name
    candidates = sorted(root.glob("env_*/arena_features.npz"))
    if not candidates:
        raise SystemExit(f"No arena_features.npz under {root}")
    return candidates[-1]


def _camera_centres():
    """Camera centres in board (world) frame from the PyLorex calibration npz files.

    C_world = -R.T @ t, mirroring SCRIPT_BuildArenaGeometry.load_camera_calibration.
    Shark is shifted to the unified tiger frame via shark2tiger_delta.
    """
    base = Path("/home/dieter/Dropbox/PythonRepos/PyLorex/PyLorex/Calibration/Results")
    centres = {}
    for cam in ("tiger", "shark"):
        p = base / f"pose_{cam}.npz"
        if not p.is_file():
            continue
        d = np.load(p, allow_pickle=False)
        R = np.asarray(d["R_pnp"], dtype=float)
        t = np.asarray(d["t_pnp"], dtype=float).reshape(3)
        C = -R.T @ t
        if cam == "shark":
            C = C + np.array([LorexSettings.shark2tiger_delta_x,
                              LorexSettings.shark2tiger_delta_y, 0.0])
        centres[cam] = C
    return centres


def render(plot_path: Path, walls_x, walls_y, poles_x, poles_y, pole_radius_mm,
           cam_centres, detections):
    """Render the static arena geometry + current robot detection(s) to plot_path.

    detections: list of dicts with keys camera, x, y, yaw_deg.
    """
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(walls_x, walls_y, s=1, c="#2ca02c", alpha=0.6, label="planner walls")
    for px, py in zip(poles_x, poles_y):
        ax.add_patch(plt.Circle((px, py), pole_radius_mm, color="#9467bd",
                                fill=False, lw=1.5))
    for cam, C in cam_centres.items():
        ax.plot(C[0], C[1], "x", color="black", markersize=10, mew=2)
        ax.annotate(cam, (C[0], C[1]), xytext=(8, 4),
                    textcoords="offset points", fontsize=8, color="black")
    # Robot detections
    colors = {"tiger": "#1f77b4", "shark": "#d62728"}
    for det in detections:
        col = colors.get(det["camera"], "#1f77b4")
        ax.add_patch(plt.Circle((det["x"], det["y"]), ROBOT_RADIUS_MM,
                                color=col, fill=False, lw=2))
        ax.plot(det["x"], det["y"], "+", color=col, markersize=8, mew=2)
        if det["yaw_deg"] is not None:
            ang = np.radians(det["yaw_deg"])
            dx = ROBOT_RADIUS_MM * np.cos(ang)
            dy = ROBOT_RADIUS_MM * np.sin(ang)
            ax.arrow(det["x"], det["y"], dx, dy, width=4, color=col,
                     length_includes_head=True)
        ax.annotate(f"{det['camera']}: ({det['x']:.0f}, {det['y']:.0f}) "
                    f"yaw={det['yaw_deg']:+.1f}°" if det["yaw_deg"] is not None
                    else f"{det['camera']}: ({det['x']:.0f}, {det['y']:.0f})",
                    (det["x"], det["y"]),
                    xytext=(12, -12), textcoords="offset points",
                    fontsize=9, color=col)
    ax.set_xlim(LorexSettings.environment_arena_min_x_mm,
                LorexSettings.environment_arena_max_x_mm)
    ax.set_ylim(LorexSettings.environment_arena_min_y_mm,
                LorexSettings.environment_arena_max_y_mm)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Probe vs planner walls — {ARENA_NAME}  "
                 f"({time.strftime('%H:%M:%S')})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)


def main():
    config = Settings.get_client_config(ROBOT_NUMBER - 1)
    aruco_id = config.aruco_id
    print(f"Probing tracker for {config.robot_name} (aruco_id={aruco_id}).")
    print(f"Arena: {ARENA_NAME}.")
    print(f"Polling every {POLL_INTERVAL_S:.1f} s. Ctrl-C to stop.")

    features_path = _latest_arena_features(ARENA_NAME)
    print(f"Loading walls from {features_path.name}.")
    feat = np.load(features_path)
    kind = feat["kind"]
    fx = feat["x_mm"]; fy = feat["y_mm"]
    walls_x, walls_y = fx[kind == 0], fy[kind == 0]
    poles_x, poles_y = fx[kind == 1], fy[kind == 1]
    pole_radius_mm = float(feat["pole_radius_mm"]) if "pole_radius_mm" in feat.files else 12.5

    cam_centres = _camera_centres()
    print(f"Plot will be saved to: {PLOT_PATH}\n")

    client = ServerClient.TelemetryClient()

    header = f"{'t(s)':>6}  {'cam':>6}  {'x_mm':>9}  {'y_mm':>9}  {'yaw_deg':>8}  {'h_mm':>7}"
    print(header)
    print("-" * len(header))

    t0 = time.time()
    try:
        while True:
            snapshots = client.get_raw_trackers()
            detections_for_plot = []
            found = False
            for snap in snapshots:
                for det in snap.detections:
                    if int(det.data.get('id', -1)) != aruco_id:
                        continue
                    found = True
                    x, y = det.data['floor_xy_mm']
                    yaw = det.data.get('yaw_deg')
                    h   = det.data.get('height_mm')

                    if snap.camera == 'shark':
                        x += LorexSettings.shark2tiger_delta_x
                        y += LorexSettings.shark2tiger_delta_y

                    yaw_str = f"{yaw:+8.2f}" if yaw is not None else "     N/A"
                    h_str   = f"{h:7.1f}"   if h   is not None else "    N/A"
                    print(f"{time.time() - t0:6.1f}  {snap.camera:>6}  "
                          f"{x:9.1f}  {y:9.1f}  {yaw_str}  {h_str}")
                    detections_for_plot.append({
                        "camera": snap.camera, "x": float(x), "y": float(y),
                        "yaw_deg": float(yaw) if yaw is not None else None,
                    })
            if not found:
                print(f"{time.time() - t0:6.1f}  ------ marker id={aruco_id} not visible "
                      f"to any camera ------")
            render(PLOT_PATH, walls_x, walls_y, poles_x, poles_y, pole_radius_mm,
                   cam_centres, detections_for_plot)
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
