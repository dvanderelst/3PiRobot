
from Library import Client
import numpy as np

client = Client.Client(robot_number=1, ip='192.168.1.13')
client.change_free_ping_interval(0)
data, distance_axis, timing_info = client.ping(plot=True)
client.close()

print(np.max(data, axis=0))
