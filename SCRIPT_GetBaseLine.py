import pickle
import time
import numpy as np
from Library import Client
import matplotlib.pyplot as plt

# ─── Baseline collection Settings ────
client_nr = 0
repeats = 10
# ─────────────────────────────────────

client = Client.Client(client_nr)
robot_name = client.configuration.name

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
plt.title(f'{robot_name}: Left')

plt.subplot(212)
plt.plot(distance_axis, right_measurements, color='red', alpha=0.5)
plt.plot(distance_axis, baseline_right, color='black')
plt.ylim(overall_min, overall_max)
plt.xlabel('Distance')
plt.ylabel('Amplitude')
plt.title(f'{robot_name}: Right')

plt.tight_layout()

baseline_data = {
    'left_measurements': left_measurements,
    'right_measurements': right_measurements,
    'baseline_left': baseline_left,
    'baseline_right': baseline_right,
    'distance_axis': distance_axis,
    'client_configuration': client.configuration
}

baseline_filename = f'baselines/baseline_{robot_name}.pck'
with open(baseline_filename, 'wb') as f: pickle.dump(baseline_data, f)

plot_filename = baseline_filename.replace('.pck', '.png')
plt.savefig(plot_filename)
print(f"Baseline data saved to {baseline_filename}")

client.close()
plt.show()

print("Baseline collection completed successfully.")
