import time

import Client
from matplotlib import pyplot as plt

client = Client.Client('192.168.200.38')

# while True:
#     client.set_motors(0, 0)
#     x, distances = client.ping(10000, 200)
#     time.sleep(0.25)
# #
# #plt.figure()
# #plt.plot(distances, x)
# #plt.show()
#     time.sleep(0.25)

x, distances = client.ping(10000, 200)
plt.figure()
plt.plot(distances, x)
plt.show()