import time
import torch

from Library import Client
from Library import DataStorage
from Library import LorexTracker
from Library import PauseControl
from Library import PushOver
from Library import Settings
from Library.SonarToCurvature import SonarToCurvature
from Library.SteeringConfigClass import SteeringConfig
from LorexLib.Environment import capture_environment_layout

# ============================================
# CONFIGURATION
# ============================================
robot_number = 1
session = 'testing'

selection_mode = 'first'
do_plot = True
training_output_dir = 'Training'
num_steps = 250
inter_step_sleep_s = 0.5
enable_motion = True
step_distance_m = 0.1
linear_speed = 0.1
rotation_speed = 90
max_step_angle_deg = 90.0
use_delta_pose = False
autoclear = True
pending_delta_pose = (0.0, 0.0, 0.0)
debug_pose_shift = True
circle_radius_hysteresis_mm = 150.0

# Curvature pipeline settings.
window_size = Settings.occupancy_config.window_size
presence_threshold_override = None  # None -> use threshold from training params.
allow_partial_window_startup = True  # If False, wait for full window before estimating curvature.


# ============================================
# SETUP OBJECTS (same flat style as SCRIPT_DataAcquisition)
# ============================================
control = PauseControl.PauseControl()
client = Client.Client(robot_number=robot_number)
tracker = LorexTracker.LorexTracker()
writer = DataStorage.DataWriter(session, autoclear=autoclear, verbose=False)
writer.add_file('Library/Settings.py')
writer.add_file('Library/SteeringConfigClass.py')
snapshot = capture_environment_layout(save_root=f'Data/{session}')

# ============================================
# LOAD TRAINING ARTIFACTS + STEERING CONFIG
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading model artifacts from: {training_output_dir}')
steering_config = SteeringConfig(
    window_size=window_size,
    presence_threshold_override=presence_threshold_override,
    circle_radius_hysteresis_mm=circle_radius_hysteresis_mm,
)
estimator = SonarToCurvature(
    training_output_dir=training_output_dir,
    steering_config=steering_config,
    presence_threshold_override=presence_threshold_override,
    prediction_batch_size=64,
    allow_partial_window_startup=allow_partial_window_startup,
    device=device,
)

print(
    f'Loaded model. profile_opening_angle={estimator.profile_opening_angle_deg:.0f} deg, '
    f'profile_steps={estimator.profile_steps}, presence_threshold={estimator.presence_threshold:.3f}, '
    f'window_size={estimator.steering_config.window_size}'
)


# ============================================
# CLIENT WARMUP + SONAR COLLECTION LOOP (robot stays in place)
# ============================================
for _ in range(5):
    client.acquire('ping')
    time.sleep(0.5)

PushOver.send(f'UseCurvature started: {session}')
control.wait_if_paused()

for step in range(num_steps):
    sonar_package = client.read_and_process(do_ping=True, plot=do_plot, selection_mode=selection_mode)
    position = tracker.get_position(robot_number) if not use_delta_pose else None
    print(position)
    pose_valid = True
    if position is None:
        pose_valid = False
    elif isinstance(position, dict):
        pose_valid = all(position.get(k) is not None for k in ('x', 'y', 'yaw_deg'))
    if sonar_package is None:
        print(f'[{step + 1}/{num_steps}] No sonar data received.')
        writer.save_data(sonar_package=None, position=position, motion={'distance': 0.0, 'rotation': 0.0})
    else:
        if use_delta_pose or not pose_valid:
            delta_pose = pending_delta_pose
            position = None
        else:
            delta_pose = None
        estimate = estimator.update_from_package(
            sonar_package=sonar_package,
            pose=position,
            delta_pose=delta_pose,
            plot=do_plot,
            debug_pose_shift=debug_pose_shift,
        )
        signed_curvature = estimate['signed_curvature_inv_mm']
        if estimate['ready']:
            planner = estimate['planner']
            print(
                f"[{step + 1}/{num_steps}] "
                f"curvature={signed_curvature:+.6f} 1/mm | "
                f"side={planner['chosen_side']} | "
                f"R_left={planner['left_radius_mm']} mm, R_right={planner['right_radius_mm']} mm"
            )
            if enable_motion:
                angle_deg = client.step_curvature(
                    curvature_inv_mm=signed_curvature,
                    step_distance_m=step_distance_m,
                    linear_speed=linear_speed,
                    rotation_speed=rotation_speed,
                    max_angle_deg=max_step_angle_deg,
                )
                pending_delta_pose = (step_distance_m * 1000.0, 0.0, float(angle_deg))
            else:
                pending_delta_pose = (0.0, 0.0, 0.0)
        else:
            print(
                f"[{step + 1}/{num_steps}] "
                f"curvature unavailable ({estimate['window_samples_used']}/{estimate['window_size_config']} samples)."
            )
            pending_delta_pose = (0.0, 0.0, 0.0)

        writer.save_data(
            sonar_package=sonar_package,
            position=position,
            motion={'distance': 0.0, 'rotation': 0.0},
            steering={
                'signed_curvature_inv_mm': signed_curvature,
                'window_samples_used': int(estimate['window_samples_used']),
                'window_size_config': int(estimate['window_size_config']),
                'partial_window_startup': bool(allow_partial_window_startup),
                'step_index': int(step),
            },
        )

    if inter_step_sleep_s > 0:
        time.sleep(inter_step_sleep_s)

print('Loop complete. Next step: convert rolling window -> predicted profiles -> occupancy -> curvature.')
PushOver.send(f'UseCurvature loop complete: {session}')
