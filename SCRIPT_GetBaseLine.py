import Client
import pickle
import Utils
import numpy as np
import matplotlib.pyplot as plt
import Process
import time

client_ip = '192.168.200.38'
sample_rate = 15000
samples = 75
verbose = True
max_index = 40
right_shift = 3
up_shift = 1000
repeats = 10

client = Client.Client(client_ip)
client.verbose = verbose

data_collected = []
for i in range(repeats):
    print(f"Ping {i + 1}/{repeats}...")
    data, distance_axis, timing_info = client.ping(sample_rate, samples)
    data = Process.preprocess(data)
    data_collected.append(data)
    time.sleep(0.1)
data_collected = np.array(data_collected)

left = data_collected[:, :, 0]
right = data_collected[:, :, 1]

left = np.transpose(left)
right = np.transpose(right)

average_left = np.mean(left, axis=1)
average_right = np.mean(right, axis=1)

overall_max = np.max(data) + up_shift + 500

threshold_left = Utils.trace_back(average_left, max_index=max_index, right=right_shift, up=up_shift)
threshold_right = Utils.trace_back(average_right, max_index=max_index, right=right_shift, up=up_shift)

distance_axis = Utils.get_distance_axis(sample_rate, samples)

plt.figure()
plt.subplot(211)
plt.plot(distance_axis, left, color='black', alpha=0.5)
plt.plot(distance_axis, average_left, color='red')
plt.plot(distance_axis, threshold_left, color='blue', linestyle='--')
plt.ylim(-100, overall_max)
plt.ylabel('Amplitude')
plt.title('Left')

plt.subplot(212)
plt.plot(distance_axis, right, color='black', alpha=0.5)
plt.plot(distance_axis, average_right, color='red')
plt.plot(distance_axis, threshold_right, color='blue', linestyle='--')
plt.ylim(-100, overall_max)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Right')

plt.tight_layout()

# Save the data to a pck file as a dictionary

baseline_data = {
    'left': left,
    'right': right,
    'average_left': average_left,
    'average_right': average_right,
    'threshold_left': threshold_left,
    'threshold_right': threshold_right,
    'distance_axis': distance_axis,
    'client_ip': client_ip,
    'sample_rate': sample_rate,
    'samples': samples,
    'max_index': max_index,
    'right_shift': right_shift,
    'up_shift': up_shift
}

baseline_filename = f'baselines/baseline_{client_ip.replace(".", "_")}.pck'
with open(baseline_filename, 'wb') as f: pickle.dump(baseline_data, f)

plot_filename = baseline_filename.replace('.pck', '.png')
plt.savefig(plot_filename)
print(f"Baseline data saved to {baseline_filename}")

client.close()
plt.show()

print("Baseline collection completed successfully.")