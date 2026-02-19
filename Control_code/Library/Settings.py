from dataclasses import dataclass

calibration_folder = 'Library/RobotCalibration'
calibration_plot_folder = 'Library/RobotCalibration/Plots'
data_folder = 'Data'

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

client1 = ClientConfig(robot_name="Robot01", ip="192.168.0.101", aruco_id=0)
client2 = ClientConfig(robot_name="Robot02", ip="192.168.0.102", aruco_id=1)
client3 = ClientConfig(robot_name="Robot03", ip="192.168.0.103", aruco_id=2)
client_list = [client1, client2, client3]
def get_client_config(index): return client_list[index]


###########################################
# Occupancy / Curvature Defaults
###########################################
# Central defaults for occupancy integration + curvature planning.
# Update these values here; scripts/modules should read from this single source.
#
# Used by:
# - Library/SteeringConfigClass.py (SteeringConfig defaults)
# - SCRIPT_ComputeOccupancy.py
# - SCRIPT_AssessOccupancy.py
#
# Notes:
# - Script-level overrides can still be passed explicitly when needed.
# - Run summaries persist effective values used for reproducibility.
@dataclass
class OccupancyConfig:
    # Sliding window size for integration over recent samples.
    window_size: int = 5
    # Robot-frame map extent: x,y in [-extent_mm, +extent_mm].
    extent_mm: float = 2500.0
    # Occupancy grid resolution.
    grid_mm: float = 20.0
    # Anisotropic segment evidence spread (perpendicular / parallel).
    sigma_perp_mm: float = 40.0
    sigma_para_mm: float = 120.0
    # Optional post-integration smoothing.
    apply_heatmap_smoothing: bool = False


@dataclass
class CurvatureConfig:
    # Occupancy threshold used to mark blocked cells.
    occ_block_threshold: float = 0.10
    # Robot footprint inflation.
    robot_radius_mm: float = 80.0
    safety_margin_mm: float = 300.0
    # Circle-candidate search.
    circle_radius_min_mm: float = 10.0
    circle_radius_max_mm: float = 2500.0
    circle_radius_step_mm: float = 50.0
    circle_arc_samples: int = 220
    circle_horizon_x_mm: float = 1800.0
    circle_radius_tie_mm: float = 100.0
    circle_radius_hysteresis_mm: float = 150.0


occupancy_config = OccupancyConfig()
curvature_config = CurvatureConfig()
