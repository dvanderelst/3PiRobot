import Utils
import numpy as np

def run_3d(n_points_3d, diameter_3d, height_3d, runs=1000):
    mean_3d_distances = []
    for i in range(runs):
        if i%100==0: print(i)
        points = Utils.points_in_cylinder(n_points_3d, diameter_3d, height_3d)
        distances = Utils.nearest_distances(points)
        mean_distance = float(np.mean(distances))
        mean_3d_distances.append(mean_distance)
    mean_3d = np.mean(mean_3d_distances)
    return mean_3d

def run_2d(n_points_2d, diameter_2d, runs=1000):
    mean_2d_distances = []
    for i in range(runs):
        if i%100==0: print(i)
        points = Utils.points_in_circle(n_points_2d, diameter_2d)
        distances = Utils.nearest_distances(points)
        mean_distance = float(np.mean(distances))
        mean_2d_distances.append(mean_distance)
    mean_2d = np.mean(mean_2d_distances)
    return mean_2d