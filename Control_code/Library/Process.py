import numpy as np
from matplotlib import pyplot as plt
from Library import Utils
from Library.Utils import draw_integration_box



def shift2right(arr, n):
    if n <= 0: return arr.copy()
    shifted = np.empty_like(arr)
    shifted[:n] = arr[0]
    shifted[n:] = arr[:-n]
    return shifted

def baseline2threshold(baseline, extent=None, right=0, up=0):
    if extent is not None: baseline[extent:] = np.min(baseline)
    samples = len(baseline)
    flipped = np.flip(baseline)
    threshold = np.zeros(samples)
    current_max = 0
    for i in range(samples):
        if flipped[i] > current_max: current_max = flipped[i]
        threshold[i] = current_max
    threshold = np.flip(threshold)
    threshold = shift2right(threshold, right)
    threshold += up
    return threshold

def locate_echo(client, data, calibration, selection_mode='first'):
    # This function processes sonar data using the baseline
    # in the calibration dictionary, and integrates the signal
    # to find the echo onset, based on the selection mode.
    # selection_mode can be 'first', 'max' or an integer index.

    configuration = client.configuration
    integration_window = configuration.integration_window
    shift_right = configuration.baseline_shift_right
    shift_up = configuration.baseline_shift_up
    extent = configuration.baseline_extent

    left_baseline = calibration['left_baseline']
    right_baseline = calibration['right_baseline']

    left_threshold = baseline2threshold(left_baseline, extent, shift_right, shift_up)
    right_threshold = baseline2threshold(right_baseline, extent, shift_right, shift_up)
    threshold = np.maximum(left_threshold, right_threshold)

    # Threshold the channels, setting samples below the single threshold to 0
    left_channel = data[:, 1]
    right_channel = data[:, 2]

    left_thresholded = left_channel * 1.0
    right_thresholded = right_channel * 1.0
    left_thresholded[left_thresholded < threshold] = 0
    right_thresholded[right_thresholded < threshold] = 0
    # This is the maximum of the two channels, after they were thresholded
    thresholded = np.maximum(left_thresholded, right_thresholded)
    amount_above_threshold = np.maximum(left_thresholded - threshold, right_thresholded - threshold)

    crossed = False
    onset = data.shape[0] - integration_window
    offset = data.shape[0]

    # If the selection mode is an integer, use it directly
    if isinstance(selection_mode, int):
        onset = selection_mode
        offset = min(onset + integration_window, data.shape[0])
    # If the selection mode is 'first', find the first crossing
    elif selection_mode == 'first':
        crossing_indices = np.where(thresholded > 0)[0]
        if len(crossing_indices) > 0:
            onset = int(crossing_indices[0])
            offset = min(onset + integration_window, data.shape[0])
            crossed = True
    # If the selection mode is 'max', find the crossing extending furthest above threshold
    elif selection_mode == 'max':
        max_idx = np.argmax(amount_above_threshold)
        crossed = thresholded[max_idx] > threshold[max_idx]
        if crossed:
            half_window = integration_window // 2
            onset = max(0, max_idx - half_window)
            offset = min(onset + integration_window, data.shape[0])
            onset = offset - integration_window  # re-adjust if cut short at end

    # Integrate both channels over shared window
    left_integral = float(np.sum(left_channel[onset:offset]))
    right_integral = float(np.sum(right_channel[onset:offset]))

    integrals = np.array([left_integral, right_integral])
    log_integrals = 20 * np.log10(integrals + 1e-6)
    iid = float(log_integrals[1] - log_integrals[0])
    if not crossed: iid = 0

    raw_distance_axis = calibration['raw_distance_axis']
    raw_distance = raw_distance_axis[onset]

    raw_results = {
        'data': data,
        'onset': int(onset),
        'offset': int(offset),
        'crossed': crossed,
        'selection_mode': selection_mode,
        'thresholds': np.array([left_threshold, right_threshold]),
        'threshold': threshold,
        'thresholded': np.array([left_thresholded, right_thresholded]),
        'raw_distance_axis': raw_distance_axis,
        'integrals': integrals,
        'log_integrals': log_integrals,
        'raw_distance': raw_distance,
        'raw_iid': iid,
        'client_configuration': configuration
    }
    return raw_results



def plot_locate_echo(raw_results, file_name=None, close_after=False, calibration=None):
    # Get config values
    configuration = raw_results['client_configuration']
    sample_rate = configuration.sample_rate

    # If a configuration is provided, we can try to correct the distance axis
    if calibration is None: calibration = {}
    distance_coefficient = calibration.get('distance_coefficient', None)
    distance_intercept = calibration.get('distance_intercept', None)

    # Get processing results
    data = raw_results['data']
    onset = raw_results['onset']
    offset = raw_results['offset']
    crossed = raw_results['crossed']
    threshold = raw_results['threshold']
    selection_mode = raw_results['selection_mode']

    distance_axis = raw_results['raw_distance_axis']
    use_corrected_distance_axis = False
    xlabel = 'Raw Distance (not corrected) [m]'
    if distance_coefficient is not None:
        use_corrected_distance_axis = True
        distance_axis = distance_intercept + distance_coefficient * distance_axis
        xlabel = 'Corrected Distance [m]'

    # Color for onset marker
    fixed_onset = isinstance(selection_mode, int)
    onset_color = 'black' if fixed_onset else 'green'

    # Set y-axis range
    sonar_data = data[:, 1:3]  # both left and right
    range_min = np.min(sonar_data) - 500
    range_max = max(np.max(sonar_data), np.max(threshold)) + 500
    yrange = [range_min, range_max]

    plt.figure(figsize=(12, 3))
    ax1 = plt.subplot(111)
    Utils.sonar_plot(data[:, 1], sample_rate=sample_rate, yrange=yrange, color='blue', label='Left', distance_axis=distance_axis)
    Utils.sonar_plot(data[:, 2], sample_rate=sample_rate, yrange=yrange, color='red', label='Right', distance_axis=distance_axis)
    ax1.plot(distance_axis, threshold, color='black', linestyle='--', label='Threshold')

    min_in_window = np.min(data[onset:offset, :])
    max_in_window = np.max(data[onset:offset, :])
    window_start = distance_axis[onset]
    window_end = distance_axis[offset - 1]
    box_extent = [window_start, window_end, min_in_window, max_in_window]
    if crossed: draw_integration_box(ax1, box_extent, color='gray', alpha=0.3, onset_color=onset_color)

    ax1.set_facecolor('#f5f5dc')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f'Plot Locate Echo\nselection mode: {selection_mode}')
    ax1.legend(loc='upper right')

    plt.tight_layout()
    if file_name is not None: plt.savefig(file_name, dpi=300)
    if close_after: plt.close()
    else: plt.show()


def create_messages(results):
    raw_distance = results.get('raw_distance', 'None')
    raw_iid = results.get('raw_iid', 'None')
    corrected_distance = results.get('corrected_distance', 'None')
    corrected_iid = results.get('corrected_iid', 'None')
    side_code = results.get('side_code', 'None')

    if raw_distance is not None: raw_distance = f"{raw_distance:.2f}"
    if raw_iid is not None: raw_iid = f"{raw_iid:.2f}"
    if corrected_distance is not None: corrected_distance = f"{corrected_distance:.2f}"
    if corrected_iid is not None: corrected_iid = f"{corrected_iid:.2f}"

    raw_message = f"Rdist: {raw_distance} m, Riid: {raw_iid}"
    corrected_message = f"Cdist: {corrected_distance} m, Ciid: {corrected_iid}, Side: {side_code}"
    full_message = f"{raw_message} | {corrected_message}"
    messages = {'raw_message': raw_message, 'corrected_message': corrected_message, 'full_message': full_message}
    return messages
