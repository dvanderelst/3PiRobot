import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import Utils

# Processing parameters
integration_window = 5
left_channel = 1
right_channel = 0

def preprocess(data):
    data = data[:, 1:]
    data = data - np.min(data, axis=0)
    data = data
    data = data[:, [left_channel, right_channel]]
    return data

# def process(data, baseline_data, plot=False):
#     data = preprocess(data)
#     onset, integrals = threshold_and_integrate_global(data, thresholds, distance_axis, plot)
#     integrals = np.array(integrals)
#     log_integrals = 20 * np.log10(integrals + 1e-6)
#     iid = log_integrals[1] - log_integrals[0]  # right - left, positive means right is louder
#     results = {}
#     results['thresholds'] = thresholds
#     results['data'] = data
#     results['distance_axis'] = distance_axis
#     results['index'] = int(onset)
#     results['distance'] = distance_axis[onset]
#     results['integrals'] = integrals
#     results['log_integrals'] = log_integrals
#     results['iid'] = iid
#     return results

def process(data, baseline_data, plot=False):
    distance_axis = baseline_data['distance_axis']
    threshold_left = baseline_data['threshold_left']
    threshold_right = baseline_data['threshold_right']
    data = preprocess(data)
    yrange = [int(np.min(data)) - 500, int(np.max(data)) + 2000]

    plt.figure()
    plt.subplot(211)
    Utils.sonar_plot(data[:, 0], distance_axis, yrange=yrange)
    plt.plot(distance_axis, threshold_left, color='red', linestyle='--')

    plt.subplot(212)
    Utils.sonar_plot(data[:, 1], distance_axis, yrange=yrange)
    plt.plot(distance_axis, threshold_right, color='red', linestyle='--')

    plt.tight_layout()
    plt.show()


    # crossing_mask = data > thresholds[:, None]
    # crossing_indices = np.where(np.any(crossing_mask, axis=1))[0]
    #
    # crossed = False
    # if len(crossing_indices) == 0:
    #     onset = len(thresholds) - 1
    #     integrals = [0.0] * data.shape[1]
    # else:
    #     crossed = True
    #     onset = crossing_indices[0]
    #     end = min(onset + integration_window, data.shape[0])
    #     integrals = [float(np.sum(data[onset:end, i])) for i in range(data.shape[1])]
    #
    # if plot:
    #     if distances is None:
    #         distances = np.arange(data.shape[0])
    #
    #     plt.figure(figsize=(8, 4))
    #     channel_labels = ['Left', 'Right']
    #     for i in range(data.shape[1]):
    #         plt.plot(distances, data[:, i], label=channel_labels[i])
    #     plt.plot(distances, thresholds, 'k--', label='Threshold')
    #
    #     if crossed:
    #         plt.axvspan(distances[onset], distances[end - 1], color='gray', alpha=0.3, label='Integration window')
    #         plt.axvline(distances[onset], color='red', linestyle='--', label='Onset')
    #
    #     # ── Custom Grid Lines and Ticks ──
    #     ax = plt.gca()
    #
    #     # Safe max limits for ticks
    #     x_max = float(distances[-1])
    #     y_max = np.nanmax(data)
    #     if not np.isfinite(y_max): y_max = 1.0
    #     y_max = min(y_max, 50000)
    #
    #     # Horizontal axis = distance (x)
    #     ax.set_xticks(np.arange(0, x_max + 0.05, 0.05), minor=True)
    #     ax.set_xticks(np.arange(0, x_max + 0.25, 0.25))
    #     ax.set_xlim(left=0, right=x_max)  # clamp to data range
    #
    #     # Vertical axis = signal values (y)
    #     ax.set_yticks(np.arange(0, y_max + 500, 500), minor=True)
    #     ax.set_yticks(np.arange(0, y_max + 5000, 5000))
    #     ax.set_ylim(bottom=0, top=y_max)
    #
    #     ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    #     ax.grid(which='major', color='darkgray', linestyle='--', linewidth=0.8)
    #
    #     plt.xlabel('Distance [m]')
    #     plt.ylabel('Value [Arbitrary]')
    #     plt.legend()
    #     plt.title('Threshold Crossing and Integration')
    #     plt.tight_layout()
    #     plt.show()
    #
    # return onset, integrals
