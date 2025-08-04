import time
import numpy as np
from matplotlib import pyplot as plt
import easygui
import pickle
import inspect
import sys

from Library import Client
from Library import FileOperations
from Library import Utils

# Initialize client
client = Client.Client(0)
note = 'a note goes here'

# Sweep parameters
start_angle = 40
end_angle = -40
step_angle = 10
cutoff_index = 50 # values beyond this index will not be used for IID calculation
real_distance1 = 0.30 #in meters
real_distance2 = 0.40 #in meters

# Compute number of steps
nr_of_steps = int(abs(end_angle - start_angle) / step_angle) + 1
if start_angle > end_angle: step_angle = step_angle * -1  # Ensure we sweep in the correct direction

robot_name = client.configuration.robot_name

# Ask user to place object at distance 1
easygui.msgbox(f"Place the object at {real_distance1} meters and press OK.")
# Do measurements at zero angle and distance 1
zero_measurements1 = []
for n in range(5):
    print(f"Measuring at zero angle, distance {real_distance1}, iteration {n+1}")
    result = client.ping_process(cutoff_index=cutoff_index)
    zero_measurements1.append(result)
    time.sleep(1)
client.ping_process(cutoff_index=cutoff_index, plot=True)

# Ask user to place object at distance 2
easygui.msgbox(f"Place the object at {real_distance2} meters and press OK.")
# Do measurements at zero angle and distance 2
zero_measurements2 = []
for n in range(5):
    print(f"Measuring at zero angle, distance {real_distance2}, iteration {n+1}")
    result = client.ping_process(cutoff_index=cutoff_index)
    zero_measurements2.append(result)
    time.sleep(1)
client.ping_process(cutoff_index=cutoff_index, plot=True)

easygui.msgbox(f"Press ok to start the sweep from {start_angle} to {end_angle} degrees.")

# Rotate to start angle
print(f"Rotating to {start_angle} degrees")
client.step(angle=start_angle)  # Relative movement
time.sleep(2)

# Prepare storage
sweep_results = []
sweep_angles = []

# Sweep loop
for i in range(nr_of_steps):
    current_angle = start_angle + i * step_angle
    sweep_angles.append(current_angle)
    print(f"Measuring at {current_angle} degrees, {real_distance2} meters")

    # Run ping and save plot
    figure_file = f'iid_{i}_{current_angle}.png'
    figure_file = FileOperations.get_function_plot_path(robot_name, figure_file)
    result = client.ping_process(plot=figure_file, close_after=True, cutoff_index=cutoff_index)

    # Store results
    sweep_results.append(result)

    # Step only if not at the end
    if i < nr_of_steps - 1:
        client.step(angle=step_angle)
        time.sleep(1.5)

client.step(angle=-1 * step_angle*(nr_of_steps-1)/2)
client.close()


#%% Process
plot_filename = FileOperations.get_function_plot_path(robot_name, 'iid')
spatial_filename = FileOperations.get_spatial_function_path(robot_name)

raw_distances1 = np.array([result['raw_distance'] for result in zero_measurements1])
raw_distances2 = np.array([result['raw_distance'] for result in zero_measurements2])

zeros1 = np.array([result['iid'] for result in zero_measurements1])
zeros2 = np.array([result['iid'] for result in zero_measurements2])

sweep_data = np.array([result['data'] for result in sweep_results])
sweep_onsets = np.array([result['onset'] for result in sweep_results])
sweep_iids = np.array([result['iid'] for result in sweep_results])
sweep_angles = np.array(sweep_angles)

left_sweep_data = sweep_data[:, :, 1]
right_sweep_data = sweep_data[:, :, 2]
sweep_differences = left_sweep_data - right_sweep_data

coefficient, intercept = Utils.fit_linear_calibration(real_distance1, raw_distances1, real_distance2, raw_distances2)
raw_distances_interpolated = np.linspace(0, 0.5, 100)
fitted_real_distances = coefficient * raw_distances_interpolated + intercept

plt.figure(figsize=(15, 6))

plt.subplot(231)
plt.imshow(left_sweep_data, aspect='auto')
plt.colorbar()
plt.yticks(ticks=np.arange(nr_of_steps), labels=sweep_angles)
plt.plot(sweep_onsets, np.arange(nr_of_steps), marker='o', linestyle='', color='red', alpha=0.5)
plt.title('Left Data')

plt.subplot(232)
plt.imshow(right_sweep_data, aspect='auto')
plt.colorbar()
plt.yticks(ticks=np.arange(nr_of_steps), labels=sweep_angles)
plt.plot(sweep_onsets, np.arange(nr_of_steps), marker='o', linestyle='', color='red', alpha=0.5)
plt.title('Right Data')

plt.subplot(233)
plt.imshow(sweep_differences, aspect='auto')
plt.colorbar()
plt.yticks(ticks=np.arange(nr_of_steps), labels=sweep_angles)
plt.plot(sweep_onsets, np.arange(nr_of_steps), marker='o', linestyle='', color='red', alpha=0.5)
plt.title('Data Differences')

plt.subplot(234)
plt.plot(sweep_angles, sweep_iids, marker='o', linestyle='-', color='red')
plt.plot([0] * 5, zeros1, marker='o', color='blue', alpha=0.5)
plt.plot([0] * 5, zeros2, marker='o', color='green', alpha=0.5)
plt.grid()
plt.xlabel('Angle [degrees]')
plt.ylabel('IID')
plt.title('IID vs Angle')

plt.subplot(235)
plt.plot(raw_distances1, [real_distance1] * 5, marker='o', color='blue', alpha=0.5)
plt.plot(raw_distances2, [real_distance2] * 5, marker='o', color='green', alpha=0.5)
plt.plot(raw_distances_interpolated, fitted_real_distances, color='black', linestyle='--')
plt.xlabel('Raw Distance [m]')
plt.ylabel('Real Distance [m]')
plt.title('Real Distance vs Raw Distance')
plt.grid(True)
plt.tight_layout()

plt.savefig(plot_filename)
plt.show()

all_zeros = np.concatenate((zeros1, zeros2))
reference_iid = np.mean(all_zeros)

spatial_data = {
    'robot_name': robot_name,
    'angles': sweep_angles,
    'iids': sweep_iids,
    'zero_measurements': all_zeros,
    'client_configuration': client.configuration,
    'reference_iid': reference_iid,
    'coefficient': coefficient,
    'intercept': intercept,
    'note': note
}

with open(spatial_filename, 'wb') as f: pickle.dump(spatial_data, f)
print('IID collection completed successfully.')