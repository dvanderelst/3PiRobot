import time

from Library import Client
from Library import Process
from Library import FileOperations
from matplotlib import pyplot as plt
#client = Client.Client(robot_number=1, ip='192.168.200.38')
client = Client.Client(robot_number=1)

for x in range(1):
    client.configuration.sample_rate = 5000
    client.configuration.samples = 50
    sonar_package = client.ping_process(plot=True)
    client.configuration.sample_rate = 10000
    client.configuration.samples = 100
    time.sleep(1)
    sonar_package = client.ping_process(plot=True)
    # calibration = FileOperations.load_calibration(client.configuration.robot_name)
    # raw_result = Process.locate_echo(client, sonar_package, calibration)
    # Process.plot_locate_echo(raw_result)
    time.sleep(1)
client.close()
