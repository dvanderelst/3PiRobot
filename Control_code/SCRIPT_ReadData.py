from matplotlib import pyplot as plt
from Library import DataProcessor


processor = DataProcessor.DataProcessor('session5')
processor.plot_trajectory()
processor.plot_mask()
n = processor.n
for index in range(n):
    print(index)
    sonar_data = processor.get_sonar_data_at(index)
    centers, profile = processor.environment_scan_at(index, -45, 45, 10)

processor.relative_environment_at(index, plot=True)

plt.figure()
plt.subplot(2,1,1)
plt.plot(sonar_data)
plt.title(f'Sonar Data at Step {index}')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(centers, profile, marker='o')
plt.title(f'Environment Scan Profile at Step {index}')
plt.xlabel('Angle (degrees)')
plt.ylabel('Distance (mm)')
plt.grid(True)
plt.tight_layout()
plt.show()



# processor.plot_trajectory()
#
# index = 15
#
# sonar_data = processor.get_sonar_data_at(index)
# rel_x, rel_y = processor.relative_environment(index, plot=True)
# centers, profile = processor.relative_environment_scan(index, -45, 45, 10)
# plt.figure()
# plt.plot(centers, profile, marker='o')
# plt.title('Relative Environment Scan Profile at Step 5')
# plt.xlabel('Angle (degrees)')
# plt.ylabel('Distance (mm)')
# plt.grid(True)
# plt.show()
#
# plt.figure()
# plt.plot(sonar_data)
# plt.title('Sonar Data at Step 15')
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()

