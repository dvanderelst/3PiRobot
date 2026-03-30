#!/usr/bin/env python3
"""
SCRIPT_RunPolicy2.py

Deploy a policy trained by SCRIPT_TrainPolicy2.py on the real robot.

Two-phase step sequence (from rationale.md):
  Phase 1 — look:
    1. Decide rotate1 from policy (IID symmetry wrapper applied)
    2. Rotate body by rotate1
    3. Sonar ping → corrected_iid, corrected_distance
  Phase 2 — move:
    4. Decide rotate2 from policy (IID symmetry wrapper applied)
    5. Rotate body by rotate2
    6. Drive forward fixed_drive_mm

IID symmetry wrapper: if physical IID < 0 (wall on left), pass abs(IID) to
the network and negate the output rotation.  History stores canonical values.
"""

import collections
import json
import time

import numpy as np

from Library import Client
from Library import CodeLogger
from Library import DataStorage
from Library import Dialog
from Library import LorexTracker
from Library import PauseControl
from Library import PushOver
from LorexLib.Environment import capture_environment_layout
from SCRIPT_TrainPolicy2 import Config, MLPPolicy, build_input


# ══════════════════════════════════════════════════════════════════════════════
# Settings — edit these
# ══════════════════════════════════════════════════════════════════════════════

CONDITION    = "run4"                        # sub-folder under Policy/
POLICY_FILE  = "best_policy.json"           # filename inside that folder
SESSION      = "sessionP01"
ROBOT_ID     = 1
MAX_STEPS    = 150

# Dry-run flags (set False to disable movement for debugging)
do_rotation    = True
do_translation = True

wait_for_confirmation = False

POLICY_DIR = "Policy"


# ══════════════════════════════════════════════════════════════════════════════
# Load policy
# ══════════════════════════════════════════════════════════════════════════════

def load_policy(path: str):
    with open(path) as f:
        data = json.load(f)
    cfg = Config(
        history_len     = data["history_len"],
        hidden_sizes    = tuple(data["hidden_sizes"]),
        max_rotate1_deg = data["max_rotate1_deg"],
        max_rotate2_deg = data["max_rotate2_deg"],
        fixed_drive_mm  = data["fixed_drive_mm"],
    )
    policy = MLPPolicy(cfg)
    policy.set_genome(np.array(data["genome"], dtype=np.float32))
    return policy, cfg


# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════

policy_path = f"{POLICY_DIR}/{CONDITION}/{POLICY_FILE}"
policy, cfg = load_policy(policy_path)
print(f"Loaded policy: {policy_path}")
print(f"  history_len={cfg.history_len}  hidden={cfg.hidden_sizes}")
print(f"  max_rotate1={cfg.max_rotate1_deg}°  max_rotate2={cfg.max_rotate2_deg}°")
print(f"  fixed_drive={cfg.fixed_drive_mm} mm")

control = PauseControl.PauseControl()
client  = Client.Client(robot_number=ROBOT_ID)
tracker = LorexTracker.LorexTracker()
writer  = DataStorage.DataWriter(SESSION, autoclear=True, verbose=False)
writer.add_file("SCRIPT_RunPolicy.py")
snapshot = capture_environment_layout(save_root=f"Data/{SESSION}")
CodeLogger.log_code(f"Data/{SESSION}", [".", "Library"], label=SESSION)

# Warm up sonar
for _ in range(5):
    client.acquire("ping")
    time.sleep(0.5)

# History buffer — zero-initialised, consistent with training
history           = collections.deque(
    [(0.0, 0.0, 0.0, 0.0)] * cfg.history_len, maxlen=cfg.history_len
)
last_physical_iid = 0.0   # no prior measurement on first step

# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

PushOver.send(f"Policy run started: {SESSION} ({CONDITION})")

for step in range(MAX_STEPS):
    control.wait_if_paused()

    # ── Phase 1: decide and execute rotate1 ───────────────────────────────────
    inp1 = build_input(history, 0.0, 0.0, 0.0, cfg)
    rotate1_canonical = policy.forward(inp1, cfg.max_rotate1_deg)
    flip1   = last_physical_iid < 0.0
    rotate1 = -rotate1_canonical if flip1 else rotate1_canonical

    if do_rotation:
        client.step(angle=rotate1)
        time.sleep(0.5)

    # ── Sonar measurement at post-rotate1 orientation ─────────────────────────
    sonar_package = client.read_and_process(do_ping=True, plot=True)
    position      = tracker.get_position(ROBOT_ID)

    if sonar_package is None:
        print(f"Step {step:4d}: no sonar data — skipping step")
        writer.save_data(
            sonar_package=None,
            position=position,
            motion={"rotate1": rotate1, "rotate2": 0.0,
                    "net_rotation": rotate1, "drive_mm": 0.0},
        )
        continue

    sonar_package["robot_number"] = ROBOT_ID
    physical_iid = float(sonar_package["corrected_iid"])
    dist_mm      = min(float(sonar_package["corrected_distance"]) * 1000.0,
                       cfg.max_dist_mm)

    # ── Phase 2: decide and execute rotate2 ───────────────────────────────────
    flip2         = physical_iid < 0.0
    canonical_iid = abs(physical_iid)
    inp2 = build_input(history, dist_mm, canonical_iid, rotate1_canonical, cfg)
    rotate2_canonical = policy.forward(inp2, cfg.max_rotate2_deg)
    rotate2 = -rotate2_canonical if flip2 else rotate2_canonical

    if do_rotation:
        client.step(angle=rotate2)
        time.sleep(0.5)

    # ── Drive forward ─────────────────────────────────────────────────────────
    if do_translation:
        client.step(distance=cfg.fixed_drive_mm / 1000.0)
        time.sleep(0.15)

    # ── Update history (canonical frame) ─────────────────────────────────────
    history.append((dist_mm, canonical_iid, rotate1_canonical, rotate2_canonical))
    last_physical_iid = physical_iid

    # ── Log ───────────────────────────────────────────────────────────────────
    net_rotation = rotate1 + rotate2
    rob_x        = position["x"]
    rob_y        = position["y"]
    rob_yaw_deg  = position["yaw_deg"]
    pos_str = f"({rob_x:.3f}, {rob_y:.3f}, {rob_yaw_deg:.1f}°)" \
              if None not in (rob_x, rob_y, rob_yaw_deg) else "N/A"

    print(
        f"Step {step:4d}: IID={physical_iid:+6.2f} dB  dist={dist_mm:6.0f} mm  "
        f"r1={rotate1:+6.1f}°  r2={rotate2:+6.1f}°  net={net_rotation:+6.1f}°  "
        f"pos={pos_str}"
    )

    writer.save_data(
        sonar_package=sonar_package,
        position=position,
        motion={
            "rotate1":      rotate1,
            "rotate2":      rotate2,
            "net_rotation": net_rotation,
            "drive_mm":     cfg.fixed_drive_mm,
        },
    )

    if step % 100 == 0 and step > 0:
        PushOver.send(f"{SESSION}: {step}/{MAX_STEPS} steps")

    if wait_for_confirmation:
        response = Dialog.ask_yes_no("Continue", min_size=(400, 200))
        if response[0] == "No":
            break
    else:
        time.sleep(0.25)

PushOver.send(f"Policy run completed: {SESSION}, {step + 1} steps.")
