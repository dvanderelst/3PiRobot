from dataclasses import dataclass

# Client configuration for a robotic system
# These set the default parameters for the clients

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

def get_config(index):
    return client_list[index]
