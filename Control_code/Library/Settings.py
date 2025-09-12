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
    samples: int = 1000
    emitter_channel: int = 0
    left_channel: int = 2
    right_channel: int = 1
    # Processing parameters
    baseline_extent: int = 40
    baseline_shift_right: int = 1
    baseline_shift_up: int = 2500
    integration_window: int = 5

client1 = ClientConfig(robot_name="Robot01", ip="192.168.0.101")
client2 = ClientConfig(robot_name="Robot02", ip="192.168.0.102")
client_list = [client1, client2]
def get_client_config(index): return client_list[index]

