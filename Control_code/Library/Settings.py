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

