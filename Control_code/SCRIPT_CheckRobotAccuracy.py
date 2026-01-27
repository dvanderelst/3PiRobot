#%%
"""
This script calculates the correspondence between the layout of the environment and the
distance/side of closest obstacle as inferred from the the sonar data.
"""
import numpy as np
from Library import Utils
from Library import DataProcessor
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sn

az_steps = 19
max_extent = 60
chip_steps = 8

processor1 = DataProcessor.DataProcessor("session6")
processor2 = DataProcessor.DataProcessor("session7")
processor3 = DataProcessor.DataProcessor("session8")

collated1 = processor1.collate_data(az_min=-max_extent, az_max=max_extent, az_steps=az_steps)
collated2 = processor2.collate_data(az_min=-max_extent, az_max=max_extent, az_steps=az_steps)
collated3 = processor3.collate_data(az_min=-max_extent, az_max=max_extent, az_steps=az_steps)

collated_results = [collated1, collated2, collated3]
all_centers = collated1['centers']
all_centers = np.round(all_centers)
profiles = DataProcessor.collect(collated_results, 'profiles')

sonar_iid = DataProcessor.collect(collated_results, 'corrected_iid')
sonar_iid_sign = np.sign(sonar_iid)
sonar_distance = DataProcessor.collect(collated_results, 'corrected_distance')

rob_x = DataProcessor.collect(collated_results, 'rob_x')
rob_y = DataProcessor.collect(collated_results, 'rob_y')
rob_yaw_deg = DataProcessor.collect(collated_results, 'rob_yaw_deg')
wall_x = processor1.wall_x
wall_y = processor1.wall_y

#%%
all_results = []
for chip_n in range(chip_steps):
    print(chip_n)
    chipped_profiles = Utils.chip(profiles, chip_n)
    chipped_centers = Utils.chip(all_centers, chip_n)
    chipped_extent = np.max(chipped_centers)

    indices = Utils.get_extrema_positions(chipped_profiles, 'min')
    closest_visual_direction = chipped_centers[indices]
    closest_visual_side = np.sign(closest_visual_direction) * -1
    closest_visual_side_average = Utils.get_side_average(chipped_profiles)
    closest_visual_distance = Utils.get_extrema_values(chipped_profiles, 'min')


    result = {}
    result['closest_visual_direction'] = closest_visual_direction
    result['closest_visual_distance'] = closest_visual_distance
    result['closest_visual_side_average'] = closest_visual_side_average
    result['closest_visual_side'] = closest_visual_side
    result['sonar_iid'] = sonar_iid
    result['sonar_iid_sign'] = sonar_iid_sign
    result['sonar_distance'] = sonar_distance
    result['rob_x'] = rob_x
    result['rob_y'] = rob_y
    result['rob_yaw_deg'] = rob_yaw_deg

    result = pd.DataFrame(result)
    result['chipped_extent'] = chipped_extent
    result['chip_n'] = chip_n

    all_results.append(result)

all_results = pd.concat(all_results, axis=0)
all_results['side_matches'] = all_results['closest_visual_side'] == all_results['sonar_iid_sign']
all_results['side_matches_average'] = all_results['closest_visual_side_average'] == all_results['sonar_iid_sign']

print(all_results.columns)

#%%
filtered = all_results.query('closest_visual_side!=0')

grp = filtered.groupby('chipped_extent')
mn1 = grp.agg({'side_matches': 'mean'})
mn1 = mn1.reset_index()

mn2 = grp.agg({'side_matches_average': 'mean'})
mn2 = mn2.reset_index()

plt.figure()
plt.plot(mn1['chipped_extent'], mn1['side_matches'], label='Closest visual direction match')
plt.plot(mn2['chipped_extent'], mn2['side_matches_average'], label='Closest visual direction match, average')
plt.xlabel('Extent (deg)')
plt.legend()
plt.show()

#%%
rhos = []
extents = []
plt.figure(figsize=(12, 8))
for chip_n in range(chip_steps):
    selected = filtered.query('chip_n == @chip_n & sonar_distance < 1.5')
    x = selected['closest_visual_distance'] / 1000
    y = selected['sonar_distance']
    extent = selected.chipped_extent.values[0]
    rho = np.corrcoef(x, y)[0, 1]
    rho = np.around(rho, decimals=2)
    rhos.append(rho)
    extents.append(extent)
    plt.subplot(3, 3, chip_n + 1)
    plt.scatter(x, y, alpha=0.25)
    plt.title(f'Extent: {extent} Deg. Corr: {rho}')
    plt.xlabel('Closest visual distance')
    plt.ylabel('Sonar distance')

plt.subplot(3, 3, chip_steps + 1)
plt.plot(extents, rhos)
plt.xlabel('Extent')
plt.ylabel('Correlation')
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(12, 8))
for chip_n in range(chip_steps):
    selected = filtered.query('chip_n == @chip_n & sonar_distance < 1.5')
    x = selected['closest_visual_distance'] / 1000
    y = selected['sonar_distance']
    error = x-y
    indices = np.where(error > 0.5)[0]
    rbx = selected.rob_x.values[indices]
    rby = selected.rob_y.values[indices]
    rbc = error.values[indices]
    #Xi, Yi, Ci = Utils.interpolate_scattered_xyc(rbx, rby, rbc, nx=150, ny=150)
    #extent = [rbx.min(), rbx.max(), rby.min(), rby.max()]

    plt.subplot(3, 3, chip_n + 1)
    plt.scatter(wall_x, wall_y, alpha=0.25)
    plt.scatter(rbx, rby, c=rbc, cmap='jet')

    #plt.imshow(Ci,extent=extent, origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label='c')
    plt.axis('equal')


plt.tight_layout()
plt.show()