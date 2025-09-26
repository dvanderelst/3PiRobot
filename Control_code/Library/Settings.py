from dataclasses import dataclass

calibration_folder = 'Library/robot_calibration'
calibration_plot_folder = 'Library/robot_calibration/plots'
controller_verbosity = 2  # 0=errors, 1=warnings, 2=info, 3=debug
client_verbosity = 3  # 0=errors, 1=warnings, 2=info, 3=debug

###########################################
# Client Configurations
###########################################
@dataclass
class ClientConfig:
    robot_name: str
    ip: str
    # Acquisition settings
    sample_rate: int = 10000
    samples: int = 100
    emitter_channel: int = 0
    left_channel: int = 2
    right_channel: int = 1
    # Processing parameters
    baseline_extent_m: int = 0.3       # in raw distance, meters
    baseline_shift_right_m: int = 0.15   # in raw distance, meters
    baseline_shift_up_a: int = 5000   # in amplitude units
    integration_window_m: int = 0.2       # in raw distance, meters

client1 = ClientConfig(robot_name="Robot01", ip="192.168.0.101")
client2 = ClientConfig(robot_name="Robot02", ip="192.168.0.102")
client3 = ClientConfig(robot_name="Robot03", ip="192.168.0.103")
client_list = [client1, client2, client3]
def get_client_config(index): return client_list[index]

