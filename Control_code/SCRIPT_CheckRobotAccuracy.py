#%%
"""
This script calculates the correspondence between the layout of the environment and the
distance/side of closest obstacle as inferred from the sonar data.
"""
import numpy as np
from Library import Utils
from Library import DataProcessor
from matplotlib import pyplot as plt
import pandas as pd


az_steps = 121
max_extent = 45
extents = [20, 40, 60, 80, 100]

sessions = [ 'session03', 'session04', 'session06', 'session07']
collection = DataProcessor.DataCollection(sessions, az_min=-max_extent, az_max=max_extent, az_steps=az_steps)
wall_x, wall_y = collection.get_walls(processor_index=0)
centers = collection.get_centers()
#%%

full_profiles = collection.get_field('profiles') / 1000 #to meters
sonar_distance = collection.get_field('corrected_distance')
sonar_iid = collection.get_field('corrected_iid')
sonar_data = collection.get_field('sonar_data')
sonar_iid_sign = np.sign(sonar_iid)


all_results = []
for extent in extents:
    center_indices = np.where(np.abs(centers) <= extent)[0]
    constrained_profiles = full_profiles[:, center_indices]
    constrained_centers = centers[center_indices]

    indices = Utils.get_extrema_positions(constrained_profiles, 'min')
    closest_visual_direction = constrained_centers[indices]
    closest_visual_side = np.sign(closest_visual_direction) * -1
    closest_visual_distance = Utils.get_extrema_values(constrained_profiles, 'min')

    extent_results = {}
    extent_results['closest_visual_direction'] = closest_visual_direction
    extent_results['closest_visual_distance'] = closest_visual_distance
    extent_results['closest_visual_side'] = closest_visual_side
    extent_results['sonar_distance'] = sonar_distance
    extent_results['sonar_iid'] = sonar_iid
    extent_results['sonar_iid_sign'] = sonar_iid_sign
    extent_results = pd.DataFrame(extent_results)
    extent_results['extent'] = extent
    all_results.append(extent_results)

all_results = pd.concat(all_results, axis=0)
all_results['side_matches'] = all_results['closest_visual_side'] == all_results['sonar_iid_sign']

#%%
plt.figure(figsize=(12, 8))
too_fars = []
too_closes = []
for i, extent in enumerate(extents):
    selected_results = all_results.query('extent == @extent')
    closest_visual_distance = selected_results['closest_visual_distance']
    sonar_distance = selected_results['sonar_distance']

    sonar_too_far = (sonar_distance - closest_visual_distance) > 1
    sonar_too_close = (sonar_distance - closest_visual_distance) < -1

    too_far = np.mean(sonar_too_far)
    too_close = np.mean(sonar_too_close)
    too_fars.append(too_far)
    too_closes.append(too_close)


    plt.subplot(3, 2, i + 1)
    plt.scatter(closest_visual_distance, sonar_distance)
    plt.plot([0, 1.5], [0, 1.5])
    plt.plot([0, 1.5], [-0.1, 1.4])
    plt.xlabel('closest visual distance')
    plt.ylabel('sonar distance')
    plt.title(str(extent))

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(extents, too_fars, label='Sonar Too Far')
plt.plot(extents, too_closes, label='Sonar Too Close')
plt.legend()
plt.show()


#%% Example sonar data
indices = range(1000,1100, 10)
b, m = Utils.profiledist2sonarsample([0.6, 1.95], [13, 48])
for index in indices:
    plt.figure(figsize=(12, 8))
    selected_sonar = sonar_data[index, :]
    selected_sonar = selected_sonar.reshape(-1, 2)
    selected_profile = full_profiles[index, :]
    samples = b + selected_profile * m
    max_sample = int(max(samples)) + 10
    truncated = selected_sonar * 1
    truncated[max_sample:,:] = 5000

    plt.subplot(3,1,1)
    plt.plot(selected_sonar)
    plt.subplot(3,1,2)
    plt.plot(selected_profile)
    plt.ylim([0, 3.5])
    plt.subplot(3,1,3)
    plt.plot(truncated)
    plt.plot()
    plt.show()


