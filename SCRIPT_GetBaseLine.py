import Client
import pickle
import Utils
import numpy as np
import matplotlib.pyplot as plt
import Process
import time

client_ip = '192.168.200.38'
sample_rate = 20000
samples = 100
verbose = True
max_index = 25
right_shift = 2
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

baseline_left = np.mean(left, axis=1)
baseline_right = np.mean(right, axis=1)

overall_max = np.max(data) + up_shift + 500

distance_axis = Utils.get_distance_axis(sample_rate, samples)

plt.figure()
plt.subplot(211)
plt.plot(left, color='black', alpha=0.5)
plt.plot(baseline_left, color='red')
plt.ylim(-100, overall_max)
plt.ylabel('Amplitude')
plt.xlabel('Index')
plt.title('Left')

plt.subplot(212)
plt.plot(distance_axis, right, color='black', alpha=0.5)
plt.plot(distance_axis, baseline_right, color='red')
plt.ylim(-100, overall_max)
plt.xlabel('Distance')
plt.ylabel('Amplitude')
plt.title('Right')

plt.tight_layout()

# Save the data to a pck file as a dictionary

baseline_data = {
    'left': left,
    'right': right,
    'baseline_left': baseline_left,
    'baseline_right': baseline_right,
    'distance_axis': distance_axis,
    'client_ip': client_ip,
    'sample_rate': sample_rate,
    'samples': samples,
    'max_index': max_index,
}

baseline_filename = f'baselines/baseline_{client_ip.replace(".", "_")}.pck'
with open(baseline_filename, 'wb') as f: pickle.dump(baseline_data, f)

plot_filename = baseline_filename.replace('.pck', '.png')
plt.savefig(plot_filename)
print(f"Baseline data saved to {baseline_filename}")

client.close()
plt.show()

print("Baseline collection completed successfully.")