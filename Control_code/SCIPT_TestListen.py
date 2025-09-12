import time

from Library import Client
client1 = Client.Client(robot_number=1, ip='192.168.1.13')
client2 = Client.Client(robot_number=2, ip='192.168.1.19')


for i in range(10):
    time.sleep(1)
    client2.change_free_ping_period(110)
    data, distance_axis, timing_info = client1.ping(plot=True)

client1.close()
client2.close()