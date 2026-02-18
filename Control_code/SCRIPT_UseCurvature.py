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


# ============================================
# CONFIGURATION
# ============================================
robot_number = 1
session = 'sessionB05'

selection_mode = 'first'
do_plot = True
training_output_dir = 'Training'

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
writer = DataStorage.DataWriter(session, autoclear=False, verbose=False)
writer.add_file('Library/Settings.py')
writer.add_file('Library/SteeringConfigClass.py')


# ============================================
# LOAD TRAINING ARTIFACTS + STEERING CONFIG
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading model artifacts from: {training_output_dir}')
steering_config = SteeringConfig(
    window_size=window_size,
    presence_threshold_override=presence_threshold_override,
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
# CLIENT WARMUP + FIRST SONAR COLLECTION (stub end)
# ============================================
for _ in range(5):
    client.acquire('ping')
    time.sleep(0.5)

PushOver.send(f'UseCurvature started: {session}')
control.wait_if_paused()

sonar_package = client.read_and_process(do_ping=True, plot=do_plot, selection_mode=selection_mode)
position = tracker.get_position(robot_number)

if sonar_package is None:
    print('No sonar data received.')
    writer.save_data(sonar_package=None, position=position, motion={'distance': 0.0, 'rotation': 0.0})
else:
    estimate = estimator.update_from_package(sonar_package=sonar_package, pose=position)
    signed_curvature = estimate['signed_curvature_inv_mm']
    if estimate['ready']:
        planner = estimate['planner']
        print(
            f"curvature={signed_curvature:+.6f} 1/mm | "
            f"side={planner['chosen_side']} | "
            f"R_left={planner['left_radius_mm']} mm, R_right={planner['right_radius_mm']} mm"
        )
    else:
        print(f"curvature unavailable ({estimate['window_samples_used']}/{estimate['window_size_config']} samples).")

    # Persist one sample so this stub run is traceable in Data/session.
    writer.save_data(
        sonar_package=sonar_package,
        position=position,
        motion={'distance': 0.0, 'rotation': 0.0},
        steering={
            'signed_curvature_inv_mm': signed_curvature,
            'window_samples_used': int(estimate['window_samples_used']),
            'window_size_config': int(estimate['window_size_config']),
            'partial_window_startup': bool(allow_partial_window_startup),
        },
    )

print('Stub complete. Next step: convert rolling window -> predicted profiles -> occupancy -> curvature.')
PushOver.send(f'UseCurvature stub complete: {session}')
