import Client
import Process
import time
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

client = Client.Client('192.168.200.38')
client.name = 'Robot01'
client.verbose = True
client.ping_process(rate=15000, samples=75, plot=True)


#
# data = client.collect_baseline(15000, 75)
# data = np.array(data)
#
# left = data[:,:, 0]
# right = data[:,:, 1]
#
# left = np.transpose(left)
# right = np.transpose(right)
#
# average_left = np.mean(left, axis=1)
# average_right = np.mean(right, axis=1)
#
# overall_max = np.max(data)
#
# threshold_left = trace_back(average_left, max_index=40, right=3, up= 1000)
# threshold_right = trace_back(average_right, max_index=40, right=3, up=1000)
#
# plt.figure()
# plt.subplot(211)
# plt.plot(left, color='black', alpha=0.5)
# plt.plot(average_left, color='red')
# plt.plot(threshold_left, color='blue', linestyle='--')
# plt.ylim(-100, overall_max)
# plt.ylabel('Amplitude')
#
# plt.subplot(212)
# plt.plot(right, color='black', alpha=0.5)
# plt.plot(average_right, color='red')
# plt.plot(threshold_right, color='blue', linestyle='--')
# plt.ylim(-100, overall_max)
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.tight_layout()
# plt.show()
#
#
#
#
# # rotation_function = interp1d([0, 0.2,1], [30, 20, 10], kind='linear', fill_value='extrapolate')
# #
# # #client.set_kinematics(0.1, 0.0)
# # while True:
# #     results = client.ping_process(15000, 75, True)
# #     distance = results['distance']
# #     iid = results['iid']
# #     print('Distance:', distance, 'IID:', iid)
# #
# #     rotation_sign = 1
# #     if iid > 0: rotation_sign = -1
# #
# #     rotational_velocity = rotation_function(distance) * rotation_sign
# #     print(rotational_velocity)
# #     #client.set_kinematics(0.075, rotational_velocity)
# #     time.sleep(0.25)
# #
# #
# # client.stop()
#
#
# # for i in range(3):
# #     print('------------------')
# #     data, distance_axis, timing = client.ping(10000, 150, False)
# #     results = Process.process(data, distance_axis, plot=True)
# #     distance = results['onset_distance']
# #     integrals = results['log_integrals']
# #     print('distance:', distance)
# #     print('integrals:', integrals)
# #
# #
#
#
