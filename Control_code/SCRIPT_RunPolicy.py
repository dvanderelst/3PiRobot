"""
SCRIPT_RunPolicy.py — drive the robot with a trained HistoryNNPolicy.

Mirrors SCRIPT_DataAcquisition.py structure, replacing the heuristic controller
with PolicyController.  Sonar data and position are saved identically to
SCRIPT_DataAcquisition.py so downstream analysis scripts work unchanged.

Ping ordering (DataAcquisition-style):
    ping → compute rotate1 (from last_iid) + rotate2 (from current ping)
         → execute rotate1 → execute rotate2 → execute drive
"""

import time

import numpy as np
from Library import Dialog
from Library import Client
from Library import DataStorage
from Library import LorexTracker
from Library import PauseControl
from Library import PushOver
from Library.PolicyController import PolicyController, load_policy
from Library.EchoProcessor import EchoProcessor
from LorexLib.Environment import capture_environment_layout


# ── Configuration ────────────────────────────────────────────────────────────

CONDITION       = "memory05"              # sub-folder under Policy/ that holds the JSON
ROBOT_ID        = 1
SESSION         = "sessionB05_policy"     # data session folder name
MAX_STEPS       = 200
FIXED_DRIVE_MM  = 100.0
wait_for_confirmation = False

# Dry-run flags (mirror SCRIPT_DataAcquisition.py convention)
do_rotation     = True
do_translation  = True

POLICY_DIR          = "Policy"               # root folder containing CONDITION sub-folder
ECHO_PROCESSOR_DIR  = "EchoProcessor"        # folder containing echoprocessor_artifacts.pth

# ── Setup ────────────────────────────────────────────────────────────────────

policy_path = f"{POLICY_DIR}/{CONDITION}/best_policy.json"
policy      = load_policy(policy_path)
ctrl        = PolicyController(policy, fixed_drive_mm=FIXED_DRIVE_MM)
ep          = EchoProcessor.load(ECHO_PROCESSOR_DIR)

control = PauseControl.PauseControl()
client  = Client.Client(robot_number=ROBOT_ID)
tracker = LorexTracker.LorexTracker()
writer  = DataStorage.DataWriter(SESSION, autoclear=True, verbose=False)
writer.add_file("Library/PolicyController.py")
writer.add_file("SCRIPT_RunPolicy.py")
snapshot = capture_environment_layout(save_root=f"Data/{SESSION}")

# Warm up sonar (flush stale buffers)
for _ in range(5):
    client.acquire("ping")
    time.sleep(0.5)

# ── Main loop ────────────────────────────────────────────────────────────────

ctrl.reset()
PushOver.send(f"Policy run started: {SESSION} ({CONDITION})")

for step in range(MAX_STEPS):
    control.wait_if_paused()

    # --- Rotate1: orient head based on last step's IID (before pinging) ---
    rotate1 = ctrl.compute_rotate1()
    if do_rotation:
        client.step(angle=rotate1)
        time.sleep(0.15)

    # --- Sonar ping at post-rotate1 orientation ---
    sonar_package = client.read_and_process(do_ping=True, plot=True)
    position      = tracker.get_position(ROBOT_ID)

    if sonar_package is None:
        print(f"Warning: No sonar data at step {step}, skipping.")
        writer.save_data(
            sonar_package=None,
            position=position,
            motion={"rotate1": rotate1, "rotate2": 0.0, "drive_mm": 0.0},
        )
        continue

    sonar_package["robot_number"] = ROBOT_ID
    sonar_lr       = np.asarray(sonar_package["sonar_data"], dtype=np.float32)[:, [1, 2]]
    dist_axis_mm   = np.asarray(sonar_package["corrected_distance_axis"], dtype=np.float32) * 1000.0
    ep_result      = ep.predict(sonar_lr, dist_axis_mm)
    iid_db         = float(ep_result["iid_db"][0])
    dist_mm        = float(ep_result["distance_mm"][0])

    # --- Rotate2: body turn based on current ping ---
    rotate2 = ctrl.compute_rotate2(iid_db, dist_mm)

    rob_x       = position["x"]
    rob_y       = position["y"]
    rob_yaw_deg = position["yaw_deg"]
    if None not in (rob_x, rob_y, rob_yaw_deg):
        pos_str = f"({rob_x:.3f}, {rob_y:.3f}, {rob_yaw_deg:.1f}°)"
    else:
        pos_str = "N/A"

    print(
        f"Step {step:3d}: IID={iid_db:+6.2f} dB  dist={dist_mm:6.0f} mm  "
        f"rot1={rotate1:+6.1f}°  rot2={rotate2:+6.1f}°  net={rotate1+rotate2:+6.1f}°  "
        f"drive={FIXED_DRIVE_MM:.0f} mm  pos={pos_str}"
    )

    # --- Execute rotate2 and drive ---
    if do_rotation:
        client.step(angle=rotate2)
        time.sleep(0.15)

    if do_translation:
        client.step(distance=FIXED_DRIVE_MM / 1000.0)   # Client expects metres
        time.sleep(0.15)

    # --- Update controller history ---
    ctrl.update(rotate1, rotate2, iid_db, dist_mm)

    # --- Save data ---
    writer.save_data(
        sonar_package=sonar_package,
        position=position,
        motion={"rotate1": rotate1, "rotate2": rotate2, "drive_mm": FIXED_DRIVE_MM},
    )

    if step % 100 == 0 and step > 0:
        PushOver.send(f"Policy run progress: {step}/{MAX_STEPS} steps ({SESSION})")

    if wait_for_confirmation:
        response = Dialog.ask_yes_no("Continue", min_size=(400, 200))
        if response[0] == 'No': break
    else:
        time.sleep(0.25)

PushOver.send(f"Policy run completed: {MAX_STEPS} steps for session {SESSION}.")
