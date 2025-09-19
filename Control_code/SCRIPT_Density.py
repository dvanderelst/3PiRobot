import numpy as np

diameter = 5
height = 1
n_points = 25


def nearest_distances(points):
    """
    For each point in the array, compute the distance to its nearest neighbor.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, d) with n points in d dimensions.

    Returns
    -------
    np.ndarray
        1D array of length n, nearest distance for each point.
    """
    # Pairwise squared distances (n x n matrix)
    diffs = points[:, None, :] - points[None, :, :]
    dist_sq = np.sum(diffs**2, axis=-1)

    # Fill diagonal with inf to ignore self-distance
    np.fill_diagonal(dist_sq, np.inf)

    # Nearest neighbor distances
    nearest = np.min(np.sqrt(dist_sq), axis=1)
    return nearest

def point_in_cylinder(n_points, diameter, height=1):
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

print(points.shape)   # (25, 3)
print(points[:5])     # show first 5 points
