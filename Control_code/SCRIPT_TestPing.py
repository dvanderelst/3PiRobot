import time
import random

from Library import Client


client1 = Client.Client(robot_number=1)
client2 = Client.Client(robot_number=2)
client1.configuration.samples = 500
client2.configuration.samples = 500


#client2.acquire(action='ping')
time.sleep(1)

acquire_id1 = client1.acquire(action='ping')
acquire_id2 = client2.acquire(action='listen')

package1 = client1.read_buffers(plot=True)
package2 = client2.read_buffers(plot=True)

client1.close()
client2.close()

Client.print_robot_timing(package1, acquire_id1)
Client.print_robot_timing(package2, acquire_id2)