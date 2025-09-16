import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from Library import Utils

def monotonize_threshold(baseline):
    flipped = np.flip(baseline)
    cumulative_max = np.maximum.accumulate(flipped)
    threshold = np.flip(cumulative_max)
    return threshold

def get_threshold_function(client, calibration):
    configuration = client.configuration
    shift_right_m = configuration.baseline_shift_right_m
    shift_up_a = configuration.baseline_shift_up_a
    extent_m = configuration.baseline_extent_m

    calibration_distance_axis = calibration['raw_distance_axis']
    left_baseline = calibration['left_baseline']
    right_baseline = calibration['right_baseline']
    baseline = np.maximum(left_baseline, right_baseline)

    if extent_m is not None:
        extent_samples = Utils.find_closest_value_index(calibration_distance_axis, extent_m)
        extent_samples = int(np.clip(extent_samples, 0, len(baseline)))
        baseline[extent_samples:] = np.min(baseline)

    calibration_distance_axis = calibration_distance_axis + shift_right_m
    raw_threshold = monotonize_threshold(baseline)
    raw_threshold = raw_threshold + shift_up_a
    fill_value =(raw_threshold[0], raw_threshold[-1])
    function = interp1d(calibration_distance_axis, raw_threshold, bounds_error=False,fill_value=fill_value)
    return function

def locate_echo(client, sonar_package, calibration, selection_mode='first'):
    configuration = client.configuration
    integration_window = int(configuration.integration_window_samples)

    sonar_data = sonar_package['sonar_data']
    raw_distance_axis = sonar_package['raw_distance_axis']

    threshold_function = get_threshold_function(client, calibration)
    threshold = threshold_function(raw_distance_axis)

    # Channels - This order has been set in Client.py already
    left_channel = sonar_data[:, 1]
    right_channel = sonar_data[:, 2]

    # Thresholding
    left_thresholded = left_channel.astype(float)
    right_thresholded = right_channel.astype(float)
    left_thresholded[left_thresholded < threshold] = 0
    right_thresholded[right_thresholded < threshold] = 0

    thresholded = np.maximum(left_thresholded, right_thresholded)
    amount_above_threshold = np.maximum(left_thresholded - threshold, right_thresholded - threshold)

    crossed = False
    onset = max(0, sonar_data.shape[0] - integration_window)  # NEW: guard against negative
    offset = sonar_data.shape[0]

    if isinstance(selection_mode, int):
        onset = int(selection_mode)                 # NEW: clamp
        offset = min(onset + integration_window, sonar_data.shape[0])

    elif selection_mode == 'first':
        crossing_indices = np.where(thresholded > 0)[0]
        if len(crossing_indices) > 0:
            onset = int(crossing_indices[0])
            offset = min(onset + integration_window, sonar_data.shape[0])
            crossed = True

    elif selection_mode == 'max':
        max_idx = int(np.argmax(amount_above_threshold))
        crossed = thresholded[max_idx] > threshold[max_idx]
        if crossed:
            half_window = integration_window // 2
            onset = max(0, max_idx - half_window)
            offset = min(onset + integration_window, sonar_data.shape[0])
            onset = offset - integration_window  # keep exact length

    # Integrate both channels over shared window
    left_integral = float(np.sum(left_channel[onset:offset]))
    right_integral = float(np.sum(right_channel[onset:offset]))

    integrals = np.array([left_integral, right_integral])
    log_integrals = 20 * np.log10(integrals + 1e-6)
    iid = float(log_integrals[1] - log_integrals[0])
    if not crossed: iid = 0
    raw_distance = float(raw_distance_axis[onset])

    raw_results = {
        'onset': int(onset),
        'offset': int(offset),
        'crossed': crossed,
        'selection_mode': selection_mode,
        'threshold': threshold, #<- evaluated on same axis as data
        'integrals': integrals,
        'log_integrals': log_integrals,
        'raw_distance': raw_distance,
        'raw_iid': iid,
        'client_configuration': configuration,
        'threshold_function': threshold_function,
        'calibration': calibration
    }
    sonar_package.update(raw_results)
    return sonar_package

def apply_correction(sonar_package, calibration=None, eps_db=0.5):
    calibration = calibration or {}
    distance_intercept = calibration.get('distance_intercept', None)
    distance_coefficient = calibration.get('distance_coefficient', None)
    zero_iids = calibration.get('zero_iids', None)

    # Try to apply distance correction
    corrected_distance = sonar_package['raw_distance']
    corrected_distance_axis = sonar_package['raw_distance_axis']
    distance_correction_applied = False

    if distance_intercept is not None:
        corrected_distance = distance_intercept + distance_coefficient * corrected_distance
        corrected_distance_axis = distance_intercept + distance_coefficient * corrected_distance_axis
        distance_correction_applied = True
    # Try to apply IID correction
    iid_corrected = sonar_package['raw_iid']
    iid_correction_applied = False
    # --- IID calibration ---
    if zero_iids is not None:
        mean_zero = float(np.mean(zero_iids))
        iid_corrected = iid_corrected - mean_zero
        iid_correction_applied = True

    if iid_corrected < -eps_db: side_code = 'L'
    elif iid_corrected > eps_db: side_code = 'R'
    else: side_code = 'C'

    corrections = {}
    corrections['corrected_distance'] = corrected_distance
    corrections['corrected_distance_axis'] = corrected_distance_axis
    corrections['distance_correction_applied'] = distance_correction_applied
    corrections['corrected_iid'] = iid_corrected
    corrections['iid_correction_applied'] = iid_correction_applied
    corrections['side_code'] = side_code
    sonar_package.update(corrections)
    return sonar_package


def plot_locate_echo(sonar_package, file_name=None, close_after=False, calibration=None):
    # Try to apply correction
    corrected_results = apply_correction(sonar_package, calibration)
    threshold = sonar_package['threshold']
    onset = sonar_package['onset']
    offset = sonar_package['offset']
    crossed = sonar_package['crossed']

    distance_axis_label = 'Raw Distance [m]'
    distance_axis = sonar_package['raw_distance_axis']
    distance_correction_applied = corrected_results['distance_correction_applied']
    if distance_correction_applied:
        distance_axis = corrected_results['corrected_distance_axis']
        distance_axis_label = 'Corrected Distance [m]'

    onset_distance = distance_axis[onset]
    offset_distance = distance_axis[offset - 1]

    mx_sonar_data = np.max(sonar_package['sonar_data'][:])
    mx_threshold = np.max(threshold)
    range_max = max(mx_sonar_data, mx_threshold) + 500
    range_min = np.min(sonar_package['sonar_data'][:]) - 500
    yrange = [range_min, range_max]

    Utils.sonar_plot(sonar_package, title='Locate Echo Results', distance_axis=distance_axis, yrange=yrange)
    # After plotting the sonar data, let's plot some other stuff on top
    ax = plt.gca()
    ax.plot(distance_axis, threshold, color='black', linestyle='--', label='Threshold')
    if crossed: ax.axvspan(onset_distance, offset_distance, color='gray', alpha=0.3, label='Integration Window')
    ax.set_xlabel(distance_axis_label)
    ax.set_facecolor('#f5f5dc')
    ax.legend(loc='upper right')
    plt.tight_layout()
    if file_name is not None: plt.savefig(file_name, bbox_inches='tight')
    if not close_after: plt.show()
    if close_after: plt.close()
