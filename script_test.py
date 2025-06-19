import time

import Client
from matplotlib import pyplot as plt

client = Client.Client('192.168.200.38')
client.name = 'Robot01'
while True:
    x, distances = client.ping(50000, 200, True)
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