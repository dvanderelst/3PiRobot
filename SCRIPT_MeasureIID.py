import time
import easygui
from scipy.interpolate import (interp1d)
import Client

client = Client.Client('192.168.200.38')
client.name = 'Robot01'
client.verbose = True

start_angle = -180
end_angle = 180
step_angle = 30

nr_of_steps = int((end_angle - start_angle) / step_angle)

for i in range(nr_of_steps):
    angle = start_angle + i * step_angle
    print(f"Rotating to {angle} degrees")
    client.step(angle=step_angle)
    time.sleep(1)  # Wait for the robot to reach the position
    figure_file = 'plots/plot_' + str(i) + '.pdf'
    result = client.ping_process(rate=20000, samples=100, plot=True, save_plot=figure_file)
    time.sleep(0.5)




