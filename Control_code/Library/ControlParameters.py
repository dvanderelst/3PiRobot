from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


class Parameters:
    def __init__(self):
        self.rotation_distances = [1, 0.75, 0.3]
        self.rotation_magnitudes = [10, 20, 30]
        fill_values = (min(self.rotation_magnitudes), max(self.rotation_magnitudes))
        self.rotation_function = interp1d(self.rotation_distances, self.rotation_magnitudes, bounds_error=False, fill_value=fill_values)

        self.translation_distances = [1, 0.75, 0.3]
        self.translation_magnitudes = [0.2, 0.1, 0.05]
        fill_values = (min(self.translation_magnitudes), max(self.translation_magnitudes))
        self.translation_function = interp1d(self.translation_distances, self.translation_magnitudes, bounds_error=False, fill_value=fill_values)


    def rotation_magnitude(self, distance):
        if distance < 0.25:
            return 180
        return self.rotation_function(distance)

    def translation_magnitude(self, distance):
        return self.translation_function(distance)

    def plot(self):
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(self.rotation_distances, self.rotation_magnitudes)
        plt.title('Rotation')
        plt.xlabel('Distance (m)')
        plt.ylabel('Magnitude (Deg)')
        plt.subplot(1,2,2)
        plt.plot(self.translation_distances, self.translation_magnitudes)
        plt.title('Translation')
        plt.xlabel('Distance (m)')
        plt.ylabel('Magnitude (m)')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    p = Parameters()
    p.plot()
    p.rotation_magnitude(1000)
    p.rotation_magnitude(0)
    p.translation_magnitude(1000)
    p.translation_magnitude(0)
