import time
import numpy as np
from matplotlib import pyplot as plt

import pickle
import inspect
import sys

from Library import Client
from Library import FileOperations

# Initialize client
client = Client.Client(0)
note = 'a note goes here'

# Sweep parameters
start_angle = 60
end_angle = -60
step_angle = 20
cutoff_index = 50 # values beyond this index will not be used for IID calculation

# Compute number of steps
nr_of_steps = int(abs(end_angle - start_angle) / step_angle) + 1

if start_angle > end_angle: step_angle = step_angle * -1  # Ensure we sweep in the correct direction

# Do measurements at zero angle
zero_measurements = []
for n in range(5):
    print(f"Measuring at zero angle, iteration {n+1}")
    result = client.ping_process(cutoff_index=cutoff_index)
    zero_measurements.append(result)
    time.sleep(1)
zero_measurements = np.array(zero_measurements)

result = client.ping_process(cutoff_index=cutoff_index, plot=True)

# Rotate to start angle
print(f"Rotating to {start_angle} degrees")
client.step(angle=start_angle)  # Relative movement
time.sleep(2)

# Prepare storage
all_results = []
all_angles = []
robot_name = client.configuration.robot_name

# Sweep loop
for i in range(nr_of_steps):
    current_angle = start_angle + i * step_angle
    all_angles.append(current_angle)
    print(f"Measuring at {current_angle} degrees")

    # Run ping and save plot
    figure_file = f'iid_{i}_{current_angle}.png'
    figure_file = FileOperations.get_function_plot_path(robot_name, figure_file)
    result = client.ping_process(plot=figure_file, close_after=True, cutoff_index=cutoff_index)

    # Store results
    all_results.append(result)

    # Step only if not at the end
    if i < nr_of_steps - 1:
        client.step(angle=step_angle)
        time.sleep(1.5)

client.step(angle=-1 * step_angle*(nr_of_steps-1)/2)
client.close()
#%% Process
plot_filename = FileOperations.get_function_plot_path(robot_name, 'iid')
iid_filename = FileOperations.get_iid_function_path(robot_name)


all_data = [result['data'] for result in all_results]
all_onsets = [result['onset'] for result in all_results]
all_offsets = [result['offset'] for result in all_results]
all_iids = [result['iid'] for result in all_results]
zero_iids = [result['iid'] for result in zero_measurements]
zero_iid = float(np.mean(zero_iids))

all_data = np.array(all_data)

iid_data = {
    'robot_name': robot_name,
    'angles': all_angles,
    'iids': all_iids,
    'zero_measurements': zero_measurements,
    'client_configuration': client.configuration,
    'zero_iid': zero_iid,
    'note': note
}

#%% Plotting and saving

left_data = all_data[:, :, 1]
right_data = all_data[:, :, 2]

left_data_muted = left_data * 1.0
right_data_muted = right_data * 1.0
left_data_muted[:, 0: 20] = np.nan  # Muting the first 20 samples
right_data_muted[:, 0: 20] = np.nan  # Muting the first 20 samples

data_differences = left_data - right_data

plt.figure(figsize=(15, 6))

plt.subplot(231)
plt.imshow(left_data, aspect='auto')
plt.colorbar()
plt.yticks(ticks=np.arange(len(all_angles)), labels=all_angles)
plt.plot(all_onsets, np.arange(len(all_angles)), marker='o', linestyle='', color='red', alpha=0.5)
plt.title('Left Data')

plt.subplot(232)
plt.imshow(right_data, aspect='auto')
plt.colorbar()
plt.yticks(ticks=np.arange(len(all_angles)), labels=all_angles)
plt.plot(all_onsets, np.arange(len(all_angles)), marker='o', linestyle='', color='red', alpha=0.5)
plt.title('Right Data')

plt.subplot(233)
plt.imshow(data_differences, aspect='auto')
plt.colorbar()
plt.yticks(ticks=np.arange(len(all_angles)), labels=all_angles)
plt.plot(all_onsets, np.arange(len(all_angles)), marker='o', linestyle='', color='red', alpha=0.5)
plt.title('Data Differences')

plt.subplot(234)
plt.imshow(left_data_muted, aspect='auto')
plt.colorbar()
plt.yticks(ticks=np.arange(len(all_angles)), labels=all_angles)
plt.plot(all_onsets, np.arange(len(all_angles)), marker='o', linestyle='', color='red', alpha=0.5)
plt.title('Left Data (Emission muted)')

plt.subplot(235)
plt.imshow(right_data_muted, aspect='auto')
plt.colorbar()
plt.yticks(ticks=np.arange(len(all_angles)), labels=all_angles)
plt.plot(all_onsets, np.arange(len(all_angles)), marker='o', linestyle='', color='red', alpha=0.5)
plt.title('Right Data (Emission muted)')

plt.subplot(236)
plt.plot(all_angles, all_iids, marker='o', linestyle='-', color='red')
for zero_iid in zero_iids: plt.plot(0, zero_iid, marker='o', color='blue', alpha=0.5)
plt.grid()
plt.xlabel('Angle [degrees]')
plt.ylabel('IID')
plt.title('IID vs Angle')
plt.tight_layout()

with open(iid_filename, 'wb') as f: pickle.dump(iid_data, f)
plt.savefig(plot_filename)
plt.show()

print('IID collection completed successfully.')


