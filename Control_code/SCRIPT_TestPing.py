import time
import random

from Library import Client
client1 = Client.Client(robot_number=1, ip='192.168.200.38')
client1.configuration.samples = 500
acquire_id = client1.acquire(action='ping')
time.sleep(3)
package = client1.read_buffers(plot=True)
print(package.keys())
client1.close()
#client2.close()