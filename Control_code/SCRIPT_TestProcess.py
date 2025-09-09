from Library import Client
import time

client = Client.Client(robot_number=1, ip='192.168.1.13')
client.change_free_ping_period(100)

for x in range(10):
    data, distance_axis, timing_info = client.ping(plot=True)
    print(timing_info)
    time.sleep(1)

client.close()
