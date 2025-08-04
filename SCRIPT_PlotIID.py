import pickle
import numpy as np
from Library import Process
from Library import ClientList
from matplotlib import pyplot as plt

start_index = 30
end_index = 40

client_nr = 0

client_configuration = ClientList.get_config(client_nr)
robot_name = client_configuration.robot_name

iid_function_file = f'iid_functions/iid_function_{robot_name}.pck'
with open(iid_function_file, 'rb') as f: iid_data = pickle.load(f)
print(iid_data.keys())
all_results = iid_data['all_results']
all_data = iid_data['all_data']
all_thresholded = iid_data['all_thresholded']
all_onsets = [result['onset'] for result in all_results]
all_offsets = [result['offset'] for result in all_results]
all_iids = [result['iid'] for result in all_results]
angles = iid_data['angles']

plt.figure()
plt.plot(angles, all_iids, marker='o', linestyle='-', color='blue')
plt.grid()
plt.show()

left_data = all_data[:, :, 1]
right_data = all_data[:, :, 2]

data_differences = left_data - right_data

plt.close('all')
plt.figure(figsize=(15, 3))
plt.subplot(131)
plt.imshow(left_data, aspect='auto')
plt.colorbar()
plt.yticks(ticks=np.arange(len(angles)), labels=angles)
plt.plot(all_onsets, np.arange(len(angles)), marker='o', linestyle='', color='red', alpha=0.5)
plt.title('Left Data')

plt.subplot(132)
plt.imshow(right_data, aspect='auto')
plt.colorbar()
plt.yticks(ticks=np.arange(len(angles)), labels=angles)
plt.plot(all_onsets, np.arange(len(angles)), marker='o', linestyle='', color='red', alpha=0.5)
plt.title('Right Data')

plt.subplot(133)
plt.imshow(data_differences, aspect='auto')
plt.colorbar()
plt.yticks(ticks=np.arange(len(angles)), labels=angles)
plt.plot(all_onsets, np.arange(len(angles)), marker='o', linestyle='', color='red', alpha=0.5)
plt.title('Data Differences')

plt.tight_layout()
plt.show()

plt.figure()
plt.subplot(121)
plt.plot(angles, all_onsets, marker='o', linestyle='-', color='blue')
plt.grid()
plt.subplot(122)
plt.plot(angles, all_iids, marker='o', linestyle='-', color='red')
plt.grid()
plt.show()

#%%
#
# test = np.sum(difference, axis=1)
# plt.figure()
# plt.plot(angles+5, test)
# plt.grid()
# plt.title('Summed Difference')
# plt.show()
#
# left_sum = np.sum(left, axis=1)
# right_sum = np.sum(right, axis=1)
# left_sum = left_sum / np.max(left_sum)
# right_sum = right_sum / np.max(right_sum)
#
# left_sum = 20 * np.log10(left_sum)
# left_sum[left_sum < -35] = -35
#
# right_sum = 20 * np.log10(right_sum)
# right_sum[right_sum < -35] = -35
#
# plt.figure()
# plt.plot(angles+5, left_sum, label='Left')
# plt.plot(angles+5, right_sum, label='Right')
# plt.grid()
# plt.title('Summed Signals')
# plt.legend()
# plt.show()