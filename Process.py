import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import Utils

# Processing parameters
integration_window = 10
left_channel = 1
right_channel = 0
# Baseline parameters
max_index = 40
right_shift = 1
up_shift = 5000

def preprocess(data):
    data = data[:, 1:]
    data = data - np.min(data, axis=0)
    data = data
    data = data[:, [left_channel, right_channel]]
    return data

def process(data, baseline_data, fixed_onset=None, plot=False):
    distance_axis = baseline_data['distance_axis']
    baseline_left = baseline_data['baseline_left']
    baseline_right = baseline_data['baseline_right']

    threshold_left = Utils.trace_back(baseline_left, max_index=max_index, right=right_shift, up=up_shift)
    threshold_right = Utils.trace_back(baseline_right, max_index=max_index, right=right_shift, up=up_shift)
    threshold_max = np.maximum(threshold_left, threshold_right)

    data = preprocess(data)
    yrange = [int(np.min(data)) - 500, int(np.max(data)) + 2000]

    left = data[:, 0]
    right = data[:, 1]
    thresholded_left = left - threshold_max
    thresholded_right = right - threshold_max
    thresholded_left[thresholded_left < 0] = 0
    thresholded_right[thresholded_right < 0] = 0

    crossing_mask = np.maximum(thresholded_left, thresholded_right)
    crossing_indices = np.where(crossing_mask > 0)[0]

    crossed = False

    if fixed_onset is not None:
        # Override: use fixed onset regardless of crossings
        onset = fixed_onset
        offset = min(onset + integration_window, data.shape[0])
        left_integral = float(np.sum(left[onset:offset]))
        right_integral = float(np.sum(right[onset:offset]))
        integrals = np.array([left_integral, right_integral])

    elif len(crossing_indices) > 0:
        # Use first threshold crossing
        onset = int(crossing_indices[0])
        offset = min(onset + integration_window, data.shape[0])
        left_integral = float(np.sum(left[onset:offset]))
        right_integral = float(np.sum(right[onset:offset]))
        integrals = np.array([left_integral, right_integral])
        crossed = True
    else:
        # Fallback if no crossing and no fixed onset
        onset = len(threshold_left) - 1
        integrals = np.array([0.0, 0.0])

    log_integrals = 20 * np.log10(integrals + 1e-6)
    iid = float(log_integrals[1] - log_integrals[0])  # right - left, positive means right is louder
    results = {}
    results['thresholds'] = np.array([threshold_left, threshold_right])
    results['data'] = data
    results['distance_axis'] = distance_axis
    results['index'] = int(onset)
    results['distance'] = float(distance_axis[onset])
    results['integrals'] = integrals
    results['log_integrals'] = log_integrals
    results['iid'] = iid

    onset_color = 'red'
    if fixed_onset: onset_color = 'black'

    if plot:
        plt.figure()
        plt.subplot(311)
        Utils.sonar_plot(data[:, 0], distance_axis, yrange=yrange)
        plt.plot(distance_axis, threshold_left, color='red', linestyle='--')

        if crossed:
            plt.axvspan(distance_axis[onset], distance_axis[offset - 1], color='gray', alpha=0.3, label='Integration window')
            plt.axvline(distance_axis[onset], color='red', linestyle='--', label='Onset')

        plt.subplot(312)
        Utils.sonar_plot(data[:, 1], distance_axis, yrange=yrange)
        plt.plot(distance_axis, threshold_right, color='red', linestyle='--')

        if crossed:
            plt.axvspan(distance_axis[onset], distance_axis[offset - 1], color='gray', alpha=0.3, label='Integration window')
            plt.axvline(distance_axis[onset], color=onset_color, linestyle='--', label='Onset')

        plt.subplot(313)
        plt.plot(distance_axis, thresholded_left, label='Left', color='blue')
        plt.plot(distance_axis, thresholded_right, label='Right', color='red')
        if crossed:
            plt.axvspan(distance_axis[onset], distance_axis[offset - 1], color='gray', alpha=0.3, label='Integration window')
            plt.axvline(distance_axis[onset], color=onset_color, linestyle='--', label='Onset')
        plt.legend()
        plt.ylim(0, 20000)
        plt.title('Thresholded Signals')
        plt.gca().set_facecolor('#f5f5dc')

        plt.tight_layout()
        plt.show()
        return results

