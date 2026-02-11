from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import numpy as np


class Parameters:
    def __init__(self):
        self.rotation_distances = np.array([0.3, 0.75, 1])
        self.rotation_magnitudes = np.array([30, 20, 10])
        fill_values = (30, 0)
        self.rotation_function = interp1d(self.rotation_distances, self.rotation_magnitudes, bounds_error=False, fill_value=fill_values)

        self.translation_distances = np.array([0.3, 0.75, 1])
        self.translation_magnitudes = np.array([0.05, 0.1, 0.2])
        fill_values = (0.05, 0.2)
        self.translation_function = interp1d(self.translation_distances, self.translation_magnitudes, bounds_error=False, fill_value=fill_values)


    def rotation_magnitude(self, distance):
        if distance < 0.25:
            return 180
        return self.rotation_function(distance)

    def translation_magnitude(self, distance):
        return self.translation_function(distance)

    def plot(self, save_path=None):
        interpolated = np.linspace(0, 2, 100)
        rotation_i = self.rotation_function(interpolated)
        distance_i = self.translation_function(interpolated)

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(interpolated, rotation_i)
        plt.scatter(self.rotation_distances, self.rotation_magnitudes,color='red')
        plt.title('Rotation')
        plt.xlabel('Distance (m)')
        plt.ylabel('Magnitude (Deg)')
        plt.subplot(2,1,2)
        plt.plot(interpolated, distance_i)
        plt.scatter(self.translation_distances, self.translation_magnitudes,color='red')
        plt.title('Translation')
        plt.xlabel('Distance (m)')
        plt.ylabel('Magnitude (m)')
        plt.tight_layout()

        if save_path is not None:
            destination = save_path + "/control_parameters.png"
            plt.savefig(destination)

        plt.show()


if __name__ == "__main__":
    p = Parameters()
    p.plot()
    p.rotation_magnitude(1000)
    p.rotation_magnitude(0)
    p.translation_magnitude(1000)
    p.translation_magnitude(0)
