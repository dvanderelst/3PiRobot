import Client
from matplotlib import pyplot as plt

client = Client.Client('192.168.1.17')
#client.set_motors(0, 0)
x = client.ping(10000, 300)
print(x)

plt.figure()
plt.plot(x)
plt.show()