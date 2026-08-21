from dataclasses import dataclass, field
calibration_folder = 'Library/RobotCalibration'
calibration_plot_folder = 'Library/RobotCalibration/Plots'
# Fallback root for DataStorage.DataWriter. Every script that writes session
# data sets this explicitly before constructing a writer (AcquisitionSessions,
# TargetArenas, PolicyRuns), so this value should never be used. It is named to
# fail loudly rather than plausibly: a folder appearing under this name means a
# script forgot its override. The old default was 'SonarSessions', which held
# nothing but stray Settings.copy files long after the sonar training data moved
# to AcquisitionSessions/.
data_folder = 'UNSET_SessionRoot'

controller_verbosity = 2  # 0=errors, 1=warnings, 2=info, 3=debug
client_verbosity = 3  # 0=errors, 1=warnings, 2=info, 3=debug
tracker_verbosity = 2  # 0=errors, 1=warnings, 2=info, 3=debug


###########################################
# Client Configurations
###########################################
@dataclass
class ClientConfig:
    robot_name: str
    ip: str
    aruco_id: int = 0
    # Acquisition settings
    sample_rate: int = 10000
    samples: int = 200
    emitter_channel: int = 0
    left_channel: int = 2
    right_channel: int = 1
    # Processing parameters
    baseline_extent_m: int = 0.75       # in raw distance, meters
    baseline_shift_right_m: int = 0.05   # in raw distance, meters
    baseline_shift_up_a: int = 5000   # in amplitude units
    integration_window_m: int = 0.17       # in raw distance, meters
    # Rotation calibration table: maps a desired rotation (deg) to the rotation
    # actually obtained by the robot for that command. Both columns are in the
    # CCW-positive convention (positive = left turn), matching tracker yaw,
    # world +X, and the firmware after the May-2026 CW→CCW sign flip in
    # motors.py. Re-measure with SCRIPT_CalibrateRobot.py (Phase 1) when
    # rotation behaviour changes (battery degradation, motor wear, firmware
    # changes, wheel swap, etc.).
    rotation_desired:           list  = field(default_factory=lambda: [-40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40])
    rotation_obtained:          list  = field(default_factory=lambda: [-39.14, -29.89, -20.33, -11.96, -4.64, 0.0, 4.54, 11.77, 18.98, 28.27, 40.39])
    drive_yaw_curl_deg_per_mm:  float = -0.0512
    drive_distance_scale:       float = 0.9624

client1 = ClientConfig(robot_name="Robot01", ip="192.168.0.101", aruco_id=1)
client2 = ClientConfig(robot_name="Robot02", ip="192.168.0.102", aruco_id=2)
client3 = ClientConfig(robot_name="Robot03", ip="192.168.0.103", aruco_id=3)
client_list = [client1, client2, client3]
def get_client_config(index): return client_list[index]
