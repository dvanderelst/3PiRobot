import time
from Library import Dialog
from Library import Client
from matplotlib import pyplot as plt

client1 = Client.Client(robot_number=1, ip='192.168.1.13')
client2 = Client.Client(robot_number=2, ip='192.168.1.19')
client2.change_free_ping_period(100)
time.sleep(1)
print('Go.....')
data, distance_axis, timing_info = client1.listen(plot=True)
print(timing_info)

client1.close()
client2.close()