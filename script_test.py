import time
import Client
import random
from matplotlib import pyplot as plt


client = Client.Client('192.168.200.38')
client.name = 'Robot01'

while True:
    print('------------------')
    data, timing = client.ping(10000, 300, True)
    # value = random.random()
    # if value > 0.8: client.set_motors(0, 0)
    time.sleep(1)

# while True:
#     client.set_motors(0, 0)
#10000
#     time.sleep(1)
# #
# #plt.figure()
# #plt.plot(distances, x)
# #plt.show()
# #time.sleep(0.25)
#
# #x, distances = client.ping(10000, 400, True)