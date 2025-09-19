import numpy as np


def nearest_distances(points):
    # Pairwise squared distances (n x n matrix)
    diffs = points[:, None, :] - points[None, :, :]
    dist_sq = np.sum(diffs**2, axis=-1)

    # Fill diagonal with inf to ignore self-distance
    np.fill_diagonal(dist_sq, np.inf)

    # Nearest neighbor distances
    nearest = np.min(np.sqrt(dist_sq), axis=1)
    return nearest

def points_in_circle(n_points, diameter):
    radius = diameter / 2
    r = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y))
    return points

def points_in_cylinder(n_points, diameter, height=1):
    # Sample random values
    z = np.random.uniform(0, height, n_points)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    r = (diameter / 2) * np.sqrt(np.random.uniform(0, 1, n_points))
    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Stack into (n_points, 3) array
    points = np.column_stack((x, y, z))
    return points

