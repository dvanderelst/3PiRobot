from Library import Utils
from Library import DataProcessor
from matplotlib import pyplot as plt


session = "sessionB05"

processor = DataProcessor.DataProcessor(session)
profiles = processor.load_profiles(opening_angle=90)
n_steps = processor.n
rob_x = processor.rob_x
rob_y = processor.rob_y
rob_yaw_deg = processor.rob_yaw_deg
wall_x = processor.wall_x
wall_y = processor.wall_y
#%%

plt.figure()
plt.plot(rob_x, rob_y, color='black', alpha=0.5)
plt.scatter(wall_x, wall_y, color='green', s=10, alpha=0.5)
for selected_index in range(n_steps):
    x = rob_x[selected_index]
    y = rob_y[selected_index]
    yaw = rob_yaw_deg[selected_index]
    Utils.plot_robot_positions(x=x, y=y, yaws_deg=yaw)
    if selected_index%5==0: plt.text(x=x, y=y, s=str(selected_index), color='red', fontsize=7)
plt.show()
