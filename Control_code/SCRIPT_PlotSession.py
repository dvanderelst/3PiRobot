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

session = "sessionB01"
az_steps = 21
max_extent = 120

plot_environment_interval = 20
load_arena = False
#%%
colors = Colors.ColorMap()
processor = DataProcessor.DataProcessor(session)
n_steps = processor.n
rob_x = processor.rob_x
rob_y = processor.rob_y
rob_yaw_deg = processor.rob_yaw_deg
#%%

plt.figure()
plt.plot(rob_x, rob_y, color='black', alpha=0.5)
for selected_index in range(n_steps):
    x = rob_x[selected_index]
    y = rob_y[selected_index]
    yaw = rob_yaw_deg[selected_index]
    Utils.plot_robot_positions(x=x, y=y, yaws_deg=yaw)
plt.show()
