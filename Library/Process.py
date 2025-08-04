import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Library import Utils

def shift_right(arr, n):
    if n <= 0: return arr.copy()
    shifted = np.empty_like(arr)
    shifted[:n] = arr[0]
    shifted[n:] = arr[:-n]
    return shifted

def baseline_adjust(average, max_index=None, right=0, up=0):
    if max_index is not None: average[max_index:] = np.min(average)
    samples = len(average)
    flipped = np.flip(average)
    threshold = np.zeros(samples)
    current_max = 0
    for i in range(samples):
        if flipped[i] > current_max: current_max = flipped[i]
        threshold[i] = current_max
    threshold = np.flip(threshold)
    threshold = shift_right(threshold, right)
    threshold += up
    return threshold

def process_sonar_data(data, baseline_data, client_configuration):
    # assumes order of data is: [emitter, left, right]
    distance_axis = baseline_data['distance_axis']
    baseline_left = baseline_data['baseline_left']
    baseline_right = baseline_data['baseline_right']

    fixed_onset = client_configuration.fixed_onset
    integration_window = client_configuration.integration_window

    baseline_shift_right = client_configuration.baseline_shift_right
    baseline_shift_up = client_configuration.baseline_shift_up
    baseline_extent = client_configuration.baseline_extent

    threshold_left = baseline_adjust(baseline_left, baseline_extent, baseline_shift_right, baseline_shift_up)
    threshold_right = baseline_adjust(baseline_right, baseline_extent, baseline_shift_right, baseline_shift_up)
    threshold = np.maximum(threshold_left, threshold_right)

    left = data[:, 1]
    right = data[:, 2]

    thresholded_left = left * 1.0
    thresholded_right = right * 1.0
    thresholded_left[thresholded_left < threshold] = 0
    thresholded_right[thresholded_right < threshold] = 0

    crossing_mask = np.maximum(thresholded_left, thresholded_right)
    crossing_indices = np.where(crossing_mask > 0)[0]

    crossed = False

    if fixed_onset > 0:
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
        offset = onset + integration_window
        integrals = np.array([0.0, 0.0])

    log_integrals = 20 * np.log10(integrals + 1e-6)
    iid = float(log_integrals[1] - log_integrals[0])  # right - left, positive means right is louder
    if not crossed: iid = 0

    results = {}

    results['data'] = data
    results['onset'] = int(onset)
    results['offset'] = int(offset)
    results['crossed'] = crossed

    results['thresholds'] = np.array([threshold_left, threshold_right])
    results['threshold'] = threshold
    results['thresholded'] = np.array([thresholded_left, thresholded_right])
    results['distance_axis'] = distance_axis

    results['raw_distance'] = float(distance_axis[onset])
    results['integrals'] = integrals
    results['log_integrals'] = log_integrals
    results['iid'] = iid
    return results


def plot_processing(results, client_configuration, file_name=None,close_after=False):
    # assumes order of data is: [emitter, left, right]
    onset_color = 'red'
    fixed_onset = client_configuration.fixed_onset
    if fixed_onset >0: onset_color = 'black'

    data = results['data']
    onset = results['onset']
    offset = results['offset']
    crossed = results['crossed']
    distance_axis = results['distance_axis']

    threshold = results['threshold']
    thresholded = results['thresholded']
    #thresholded_left = thresholded[0]
    #thresholded_right = thresholded[1]

    sample_rate = client_configuration.sample_rate

    sonar_data = data[:,1:2]
    range_min = np.min(sonar_data) - 500
    print('range_min', range_min)
    range_max = max(np.max(sonar_data), np.max(threshold)) + 500

    yrange = [range_min, range_max]

    plt.figure(figsize=[8, 8])
    onset_color = 'green'
    if fixed_onset > 0: onset_color = 'black'

    plt.subplot(311)
    Utils.sonar_plot(data[:, 1], sample_rate=sample_rate, yrange=yrange, color='blue')
    plt.plot(distance_axis, threshold, color='black', linestyle='--')

    if crossed:
        plt.axvspan(distance_axis[onset], distance_axis[offset - 1], color='gray', alpha=0.3, label='Integration window')
        plt.axvline(distance_axis[onset], color=onset_color, linestyle='--', label='Onset')

    plt.subplot(312)
    Utils.sonar_plot(data[:, 2], sample_rate=sample_rate, yrange=yrange, color='red')
    plt.plot(distance_axis, threshold, color='black', linestyle='--')

    if crossed:
        plt.axvspan(distance_axis[onset], distance_axis[offset - 1], color='gray', alpha=0.3, label='Integration window')
        plt.axvline(distance_axis[onset], color=onset_color, linestyle='--', label='Onset')

    plt.subplot(313)
    #plt.plot(distance_axis, thresholded_left, label='Left', color='blue')
    #plt.plot(distance_axis, thresholded_right, label='Right', color='red')
    Utils.sonar_plot(data[:, 1], sample_rate=sample_rate, yrange=yrange, color='blue')
    Utils.sonar_plot(data[:, 2], sample_rate=sample_rate, yrange=yrange, color='red')

    if crossed:
        plt.axvspan(distance_axis[onset], distance_axis[offset - 1], color='gray', alpha=0.3, label='Integration window')
        plt.axvline(distance_axis[onset], color=onset_color, linestyle='--', label='Onset')
        plt.legend()
    plt.title('Both Signals')
    plt.gca().set_facecolor('#f5f5dc')
    plt.xlabel('Distance [m]')
    plt.tight_layout()
    if file_name is not None: plt.savefig(file_name, dpi=300)
    if close_after:
        plt.close()
    else:
        plt.show()
