import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

threshold_points = [[0, 25000], [0.25, 25000], [0.5, 1000], [1, 1000]]
integration_window = 7


def process(data, distance_axis, plot=False):
    data = data - np.min(data, axis=0)  # Normalize data to start from zero
    data = data[:, 1:]  # Remove the first column (Emitter)
    thresholds = compute_thresholds(distance_axis, kind='linear')
    onset, integrals = threshold_and_integrate_global(data, thresholds, distance_axis, plot)
    integrals = np.array(integrals)
    results = {}
    results['thresholds'] = thresholds
    results['data'] = data
    results['distance_axis'] = distance_axis
    results['onset_index'] = int(onset)
    results['onset_distance'] = distance_axis[onset]
    results['integrals'] = integrals
    results['log_integrals'] = np.log10(integrals + 1e-6)  # Avoid log(0)
    return results


def compute_thresholds(distances, kind='linear'):
    x_pts, y_pts = zip(*threshold_points)
    threshold_fn = interp1d(x_pts, y_pts, kind=kind, fill_value='extrapolate')
    return threshold_fn(distances)


def threshold_and_integrate_global(data, thresholds, distances=None, plot=False):
    crossing_mask = data > thresholds[:, None]
    crossing_indices = np.where(np.any(crossing_mask, axis=1))[0]

    if len(crossing_indices) == 0:
        if plot:
            print("No threshold crossing found.")
        return len(thresholds), [0.0] * data.shape[1]

    onset = crossing_indices[0]
    end = min(onset + integration_window, data.shape[0])
    integrals = [float(np.sum(data[onset:end, i])) for i in range(data.shape[1])]

    if plot:
        if distances is None:
            distances = np.arange(data.shape[0])

        plt.figure(figsize=(8, 4))
        for i in range(data.shape[1]):
            plt.plot(distances, data[:, i], label=f'Channel {i}')
        plt.plot(distances, thresholds, 'k--', label='Threshold')
        plt.axvspan(distances[onset], distances[end - 1], color='gray', alpha=0.3, label='Integration window')
        plt.axvline(distances[onset], color='red', linestyle='--', label='Onset')
        plt.xlabel('Distance' if distances is not None else 'Index')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Threshold Crossing and Integration')
        plt.tight_layout()
        plt.show()

    return onset, integrals
