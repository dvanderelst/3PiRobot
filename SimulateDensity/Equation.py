import math

def bats_3d_to_2d(diameter_3d, height_3d, count_3d, diameter_2d):
    """
    Map a 3D bat count to a 2D bat count such that the expected
    nearest-neighbor spacing is preserved between 3D and 2D.

    Parameters
    ----------
    diameter_3d : float
        Diameter of the 3D cylinder (same units as height).
    height_3d : float
        Height of the 3D cylinder.
    count_3d : int
        Number of bats in the 3D cylinder.
    diameter_2d : float
        Diameter of the 2D disk.

    Returns
    -------
    float
        Equivalent bat count in 2D.
    """
    # Constant from Poisson nearest-neighbor formulas
    K = 0.814672

    # Volume of cylinder
    radius_3d = diameter_3d / 2
    volume_3d = math.pi * radius_3d**2 * height_3d

    # 3D density
    lambda_3d = count_3d / volume_3d

    # Matching 2D density
    lambda_2d = K * (lambda_3d ** (2/3))

    # Area of 2D disk
    radius_2d = diameter_2d / 2
    area_2d = math.pi * radius_2d**2

    # Equivalent 2D count
    count_2d = lambda_2d * area_2d

    return count_2d


# Example usage
if __name__ == "__main__":
    diameter_3d = 5.0
    height_3d = 1.0
    count_3d = 25
    diameter_2d = 1.7
    N2 = bats_3d_to_2d(diameter_3d, height_3d, count_3d, diameter_2d)
    print(f"Equivalent 2D bat count: {N2:.2f}")
