import time
import easygui
from Library import Client
from matplotlib import pyplot as plt
from Library import Process

client = Client.Client(robot_number=1, ip='192.168.1.13')
client.change_free_ping_interval(0)
data = client.ping()


