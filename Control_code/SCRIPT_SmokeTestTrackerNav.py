"""
Smoke test for Library.TrackerNav.

Reads the robot's starting pose from the overhead tracker, defines a small
square of waypoints relative to that starting pose, and drives the robot
through them with TrackerNav.go_to_poses. Logs the per-iteration tracker
trace (pose at the start of each TrackerNav iteration is logged through
TrackerNav's own _log).

This is a hardware test: it expects the robot is on the floor, the tracker
sees its marker, and there is empty space around it. Edit SQUARE_SIDE_MM
before running if the available space is small.
"""

import time

from Library import Client
from Library import LorexTracker
from Library import PauseControl
from Library.TrackerNav import TrackerNav


robot_number = 1
square_side_mm = 300.0   # length of one side of the test square
return_to_start = True   # drive back to (and align with) the starting pose
do_plot = True           # ping + plot the sonar at each waypoint after arrival
save_plots = True        # if True, write smoke_waypoint_<i>.png instead of showing a window
selection_mode = 'first'


def main():
    control = PauseControl.PauseControl()
    client = Client.Client(robot_number=robot_number)
    tracker = LorexTracker.LorexTracker()
    nav = TrackerNav(client, tracker, robot_number,
                     pos_tol_mm=40.0,
                     yaw_tol_deg=4.0,
                     align_yaw_thresh_deg=15.0,
                     max_step_distance_m=0.15,
                     max_iterations=30,
                     verbose=True)

    pose = nav.read_pose()
    if pose is None:
        raise RuntimeError("Tracker did not see the robot at startup. "
                           "Check the marker and camera coverage.")
    x0, y0, yaw0 = pose
    print(f"Starting pose: x={x0:.0f}mm, y={y0:.0f}mm, yaw={yaw0:+.1f}°")

    # Warm-up pings (matches SCRIPT_DataAcquisition).
    for _ in range(5):
        client.acquire('ping')
        time.sleep(0.5)

    s = square_side_mm
    waypoints = [
        (x0 + s, y0,     yaw0),
        (x0 + s, y0 + s, yaw0),
        (x0,     y0 + s, yaw0),
    ]
    if return_to_start:
        waypoints.append((x0, y0, yaw0))

    print(f"Driving through {len(waypoints)} waypoints (square side = {s:.0f} mm).")
    for i, wp in enumerate(waypoints, 1):
        print(f"  {i}: ({wp[0]:.0f}, {wp[1]:.0f}, {wp[2]:+.1f}°)")

    control.wait_if_paused()
    t_start = time.time()
    results = []
    for i, wp in enumerate(waypoints, 1):
        x, y, yaw = wp
        print(f"\n=== waypoint {i}/{len(waypoints)}: ({x:.0f}, {y:.0f}, {yaw:+.1f}°) ===")
        res = nav.go_to_pose(x, y, yaw)
        results.append(res)
        print(f"  -> {res.reason} (pos_err={res.final_pos_err_mm:.1f}mm, "
              f"yaw_err={res.final_yaw_err_deg:+.1f}°)")

        # Ping at whatever pose we ended up at — even on nav failure. Confirms
        # the ping plumbing works and gives sonar at the visited pose.
        plot_arg = (f"smoke_waypoint_{i}.png" if save_plots else True) if do_plot else False
        print(f"  pinging...")
        sonar_package = client.read_and_process(do_ping=True, plot=plot_arg,
                                                selection_mode=selection_mode)
        if sonar_package is None:
            print(f"  WARNING: ping at waypoint {i} returned no data")
        else:
            echo_located = sonar_package.get('echo_located', False)
            dist = sonar_package.get('corrected_distance',
                                     sonar_package.get('raw_distance'))
            iid = sonar_package.get('corrected_iid')
            dist_str = f"{dist:.2f} m" if isinstance(dist, (int, float)) else "N/A"
            iid_str = f"{iid:+.2f} dB" if isinstance(iid, (int, float)) else "N/A"
            print(f"  ping ok: echo_located={echo_located}, "
                  f"dist={dist_str}, iid={iid_str}")

        if not res.success:
            print("  stopping waypoint sequence after nav failure")
            break
    elapsed = time.time() - t_start

    print(f"\nFinished in {elapsed:.1f} s. Per-waypoint results:")
    for i, res in enumerate(results, 1):
        status = "OK " if res.success else "FAIL"
        print(f"  {i}: [{status}] {res.reason}  "
              f"pos_err={res.final_pos_err_mm:.1f}mm  "
              f"yaw_err={res.final_yaw_err_deg:+.1f}°  "
              f"iters={res.iterations}")


if __name__ == "__main__":
    main()
