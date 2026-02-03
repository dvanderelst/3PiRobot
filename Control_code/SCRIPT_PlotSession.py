#%%
"""
This script calculates the correspondence between the layout of the environment and the
distance/side of closest obstacle as inferred from the the sonar data.
"""
import numpy as np
from Library import Utils
from Library import DataProcessor
from matplotlib import pyplot as plt
from Library import Colors
import pandas as pd

session = "session06
az_steps = 21
max_extent = 120

plot_environment_interval = 20
load_arena = False
#%%
colors = Colors.ColorMap()
processor = DataProcessor.DataProcessor(session, load_arena=load_arena)
collated = processor.collate_data(az_min=-max_extent, az_max=max_extent, az_steps=az_steps)
n_steps = collated['n']
#%%

rob_xs = collated['rob_x']
rob_ys = collated['rob_y']
rob_yaw_degs = collated['rob_yaw_deg']

plt.figure()
plt.plot(rob_xs, rob_ys, color='black', alpha=0.5)
for selected_index in range(n_steps):
    rob_x = rob_xs[selected_index]
    rob_y = rob_ys[selected_index]
    rob_yaw_deg = rob_yaw_degs[selected_index]
    Utils.plot_robot_positions(x=rob_x, y=rob_y, yaws_deg=rob_yaw_deg)
    if selected_index%plot_environment_interval==0 and load_arena:
        profiles = collated['profiles']
        centers = collated['centers']
        profile = profiles[selected_index, :]
        plot_color = colors.get_next_color()
        wall_x, wall_y = DataProcessor.robot2world(centers, profile, rob_x, rob_y, rob_yaw_deg)
        plt.scatter(wall_x, wall_y, color=plot_color, marker='.', s=50)
        plt.scatter(rob_x, rob_y, color=plot_color, marker='.', s=50)
        plt.text(x=rob_x, y=rob_y, s=str(selected_index), color=plot_color)
plt.show()
