import time
import Client
import random
from matplotlib import pyplot as plt


client = Client.Client('192.168.200.38')
client.name = 'Robot01'

while True:
    print('------------------')
    data, timing = client.ping(10000, 150, True)
    time.sleep(1)