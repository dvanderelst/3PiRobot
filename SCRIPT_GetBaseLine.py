import pickle
import time
import sys
import inspect
import numpy as np
from Library import Client
from Library import FileOperations
import matplotlib.pyplot as plt

# ─── Baseline collection Settings ────
client_nr = 0
repeats = 10
note = 'a note goes here'
# ─────────────────────────────────────

client = Client.Client(client_nr)
robot_name = client.configuration.robot_name

data_collected = []
for i in range(repeats):
    print(f"Ping {i + 1}/{repeats}...")
    data, distance_axis, timing_info = client.ping()
    data_collected.append(data)
    time.sleep(0.1)
data_collected = np.array(data_collected)

left_measurements = data_collected[:, :, 1]
right_measurements = data_collected[:, :, 2]

left_measurements = np.transpose(left_measurements)
right_measurements = np.transpose(right_measurements)

baseline_left = np.mean(left_measurements, axis=1)
baseline_right = np.mean(right_measurements, axis=1)

overall_max = np.max(data) + 500
overall_min = np.min(data) - 500

plt.figure()
plt.subplot(211)
plt.plot(left_measurements, color='blue', alpha=0.5)
plt.plot(baseline_left, color='black')
plt.ylim(overall_min, overall_max)
plt.ylabel('Amplitude')
plt.xlabel('Index')
plt.grid()
plt.title(f'{robot_name}: Left')

plt.subplot(212)
plt.plot(distance_axis, right_measurements, color='red', alpha=0.5)
plt.plot(distance_axis, baseline_right, color='black')
plt.ylim(overall_min, overall_max)
plt.xlabel('Distance')
plt.ylabel('Amplitude')
plt.grid()
plt.title(f'{robot_name}: Right')

plt.tight_layout()

script_text = inspect.getsource(sys.modules[__name__])

baseline_data = {
    'robot_name': robot_name,
    'left_measurements': left_measurements,
    'right_measurements': right_measurements,
    'baseline_left': baseline_left,
    'baseline_right': baseline_right,
    'distance_axis': distance_axis,
    'client_configuration': client.configuration,
    'script_text': script_text,
    'note': note
}

baseline_filename = FileOperations.get_baseline_function_path(robot_name)
plot_filename = FileOperations.get_function_plot_path(robot_name, 'baseline')

with open(baseline_filename, 'wb') as f: pickle.dump(baseline_data, f)
plt.savefig(plot_filename)

client.close()
plt.show()

print("Baseline collection completed successfully.")
