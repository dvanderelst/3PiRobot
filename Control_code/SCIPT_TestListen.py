import time
import random

from Library import Client
client1 = Client.Client(robot_number=1, ip='192.168.1.13')
client2 = Client.Client(robot_number=2, ip='192.168.1.19')

client2.change_free_ping_period(110)
for i in range(10):
    sleep = random.uniform(1, 2)
    time.sleep(sleep)
    # Don't reset free ping period here!
    data, distance_axis, timing_info = client1.listen(plot=True)

client1.close()
client2.close()