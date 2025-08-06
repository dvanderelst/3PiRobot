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

def process_sonar_data(data, baseline_data, client_configuration, selection='first'):
    # assumes the order of data is: [emitter, left, right]
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

    # Threshold the channels, setting samples below the single threshold to 0
    left_channel = data[:, 1]
    right_channel = data[:, 2]

    thresholded_left = left_channel * 1.0
    thresholded_right = right_channel * 1.0
    thresholded_left[thresholded_left < threshold] = 0
    thresholded_right[thresholded_right < threshold] = 0
    # This is the maximum of the two channels, after they were thresholded
    thresholded = np.maximum(thresholded_left, thresholded_right)

    crossed = False
    onset = data.shape[0] - integration_window
    offset = data.shape[0]

    # Use fixed onset regardless of selection
    if fixed_onset > 0:
        onset = fixed_onset
        offset = min(onset + integration_window, data.shape[0])

    elif selection == 'first':
        crossing_indices = np.where(thresholded > 0)[0]
        if len(crossing_indices) > 0:
            onset = int(crossing_indices[0])
            offset = min(onset + integration_window, data.shape[0])
            crossed = True

    elif selection == 'max':
        max_idx = np.argmax(thresholded)
        crossed = thresholded[max_idx] > threshold[max_idx]
        if crossed:
            half_window = integration_window // 2
            onset = max(0, max_idx - half_window)
            offset = min(onset + integration_window, data.shape[0])
            onset = offset - integration_window  # re-adjust if cut short at end

    # Integrate both channels over shared window
    left_integral = float(np.sum(left_channel[onset:offset]))
    right_integral = float(np.sum(right_channel[onset:offset]))
    first_integrals = np.array([left_integral, right_integral])

    log_integrals = 20 * np.log10(first_integrals + 1e-6)
    iid = float(log_integrals[1] - log_integrals[0])
    if not crossed: iid = 0

    results = {
        'data': data,
        'onset': int(onset),
        'offset': int(offset),
        'crossed': crossed,
        'selection': selection,  # ← added this line
        'thresholds': np.array([threshold_left, threshold_right]),
        'threshold': threshold,
        'thresholded': np.array([thresholded_left, thresholded_right]),
        'distance_axis': distance_axis,
        'raw_distance': float(distance_axis[min(onset, len(distance_axis) - 1)]),
        'integrals': first_integrals,
        'log_integrals': log_integrals,
        'iid': iid
    }

    return results


def plot_processing(results, client_configuration, file_name=None, close_after=False):
    import matplotlib.pyplot as plt

    # Get config values
    fixed_onset = client_configuration.fixed_onset
    sample_rate = client_configuration.sample_rate

    # Get processing results
    data = results['data']
    onset = results['onset']
    offset = results['offset']
    crossed = results['crossed']
    distance_axis = results['distance_axis']
    threshold = results['threshold']
    threshold_left, threshold_right = results['thresholds']
    thresholded = results['thresholded']
    selection = results.get('selection', 'first')

    # Color for onset marker
    onset_color = 'black' if fixed_onset > 0 else 'green'

    # Set y-axis range
    sonar_data = data[:, 1:3]  # both left and right
    range_min = np.min(sonar_data) - 500
    range_max = max(np.max(sonar_data), np.max(threshold)) + 500
    yrange = [range_min, range_max]

    def draw_integration_window(ax):
        ax.axvspan(distance_axis[onset], distance_axis[offset - 1], color='gray', alpha=0.3, label='Integration window')
        ax.axvline(distance_axis[onset], color=onset_color, linestyle='--', label='Onset')

    plt.figure(figsize=[10, 9])
    plt.suptitle(f'Sonar Signal Processing (Selection: {selection})')

    # --- Left channel ---
    ax1 = plt.subplot(311)
    Utils.sonar_plot(data[:, 1], sample_rate=sample_rate, yrange=yrange, color='blue')
    ax1.plot(distance_axis, threshold_left, color='black', linestyle='--', label='Threshold (Left)')
    if crossed:
        draw_integration_window(ax1)
        ax1.legend()
    ax1.set_ylabel("Amplitude (Left)")

    # --- Right channel ---
    ax2 = plt.subplot(312)
    Utils.sonar_plot(data[:, 2], sample_rate=sample_rate, yrange=yrange, color='red')
    ax2.plot(distance_axis, threshold_right, color='black', linestyle='--', label='Threshold (Right)')
    if crossed:
        draw_integration_window(ax2)
        ax2.legend()
    ax2.set_ylabel("Amplitude (Right)")

    # --- Both channels together ---
    ax3 = plt.subplot(313)
    Utils.sonar_plot(data[:, 1], sample_rate=sample_rate, yrange=yrange, color='blue')
    Utils.sonar_plot(data[:, 2], sample_rate=sample_rate, yrange=yrange, color='red')
    if crossed: draw_integration_window(ax3)
    ax3.set_facecolor('#f5f5dc')
    ax3.set_xlabel('Distance [m]')
    ax3.set_ylabel("Amplitude")
    ax3.set_title('Both Signals')
    ax3.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # make room for suptitle

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    if close_after:
        plt.close()
    else:
        plt.show()
