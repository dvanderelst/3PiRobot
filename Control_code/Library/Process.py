import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from Library import Utils

def monotonize_threshold(baseline):
    flipped = np.flip(baseline)
    cumulative_max = np.maximum.accumulate(flipped)
    threshold = np.flip(cumulative_max)
    return threshold

def get_threshold_function(sonar_package):
    configuration = sonar_package['configuration']
    calibration = sonar_package['calibration']

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

def locate_echo(sonar_package, selection_mode='first'):
    location_results = {}
    echo_located = sonar_package.get('echo_located', False)
    if echo_located: return sonar_package  # already applied
    configuration = sonar_package['configuration'] # from Client
    calibration = sonar_package['calibration'] # from Client
    integration_window_m = configuration.integration_window_m
    effective_fs_hz = sonar_package['effective_fs_hz']
    integration_window_samples = Utils.distance2samples(effective_fs_hz, integration_window_m)

    sonar_data = sonar_package['sonar_data']
    raw_distance_axis = sonar_package['raw_distance_axis']

    threshold_function = get_threshold_function(sonar_package)
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
    onset = max(0, sonar_data.shape[0] - integration_window_samples)
    offset = sonar_data.shape[0]

    if isinstance(selection_mode, int):
        onset = int(selection_mode)                 # NEW: clamp
        offset = min(onset + integration_window_samples, sonar_data.shape[0])

    elif selection_mode == 'first':
        crossing_indices = np.where(thresholded > 0)[0]
        if len(crossing_indices) > 0:
            onset = int(crossing_indices[0])
            offset = min(onset + integration_window_samples, sonar_data.shape[0])
            crossed = True

    elif selection_mode == 'max':
        max_idx = int(np.argmax(amount_above_threshold))
        crossed = thresholded[max_idx] > threshold[max_idx]
        if crossed:
            half_window = integration_window_samples // 2
            onset = max(0, max_idx - half_window)
            offset = min(onset + integration_window_samples, sonar_data.shape[0])
            onset = offset - integration_window_samples  # keep exact length

    # Integrate both channels over shared window
    left_integral = float(np.sum(left_channel[onset:offset]))
    right_integral = float(np.sum(right_channel[onset:offset]))

    integrals = np.array([left_integral, right_integral])
    log_integrals = 20 * np.log10(integrals + 1e-6)
    iid = float(log_integrals[1] - log_integrals[0])
    if not crossed: iid = 0
    raw_distance = float(raw_distance_axis[onset])

    location_results['integration_window_samples'] = integration_window_samples
    location_results['echo_located'] = True
    location_results['onset'] = int(onset)
    location_results['offset'] = int(offset)
    location_results['crossed'] = crossed
    location_results['selection_mode'] = selection_mode
    location_results['threshold'] = threshold  # <- evaluated on same axis as data
    location_results['integrals'] = integrals
    location_results['log_integrals'] = log_integrals
    location_results['raw_distance'] = raw_distance
    location_results['raw_iid'] = iid
    location_results['client_configuration'] = configuration
    location_results['threshold_function'] = threshold_function
    location_results['calibration'] = calibration

    sonar_package.update(location_results)
    return sonar_package

def apply_correction(sonar_package, eps_db=0.5):
    corrections = {}
    correction_applied = sonar_package.get('correction_applied', False)
    if correction_applied: return sonar_package  # already applied
    calibration = sonar_package.get('calibration', {})
    distance_intercept = calibration.get('distance_intercept', None)
    distance_coefficient = calibration.get('distance_coefficient', None)
    zero_iids = calibration.get('zero_iids', None)

    # Try to apply distance correction
    raw_distance = sonar_package['raw_distance']
    raw_distance_axis = sonar_package['raw_distance_axis']
    if distance_intercept is not None:
        corrected_distance = distance_intercept + distance_coefficient * raw_distance
        corrected_distance_axis = distance_intercept + distance_coefficient * raw_distance_axis
        corrections['corrected_distance'] = corrected_distance
        corrections['corrected_distance_axis'] = corrected_distance_axis
        corrections['distance_correction_applied'] = True

    # Try to apply IID correction
    iid_corrected = sonar_package['raw_iid']
    if zero_iids is not None:
        mean_zero = np.mean(zero_iids)
        iid_corrected = iid_corrected - mean_zero
        if iid_corrected < -eps_db: side_code = 'L'
        elif iid_corrected > eps_db: side_code = 'R'
        else: side_code = 'C'
        corrections['corrected_iid'] = iid_corrected
        corrections['side_code'] = side_code
        corrections['iid_correction_applied'] = True

    sonar_package.update(corrections)
    return sonar_package


def plot_sonar_data(distance_axis, sonar_package, plot_settings=None):
    meta_result = {} #To store stuff we might want to return
    if plot_settings is None: plot_settings = {}
    sonar_data = sonar_package['sonar_data']
    x_min = float(distance_axis[0])
    x_max = float(distance_axis[-1])
    y_min = float(np.nanmin(sonar_data)) - 1000.0
    y_max = float(np.nanmax(sonar_data)) + 1000.0
    if 'x_min' in plot_settings: x_min = float(plot_settings['x_min'])
    if 'x_max' in plot_settings: x_max = float(plot_settings['x_max'])
    if 'y_min' in plot_settings: y_min = float(plot_settings['y_min'])
    if 'y_max' in plot_settings: y_max = float(plot_settings['y_max'])

    ax = plt.gca()
    ax.plot(distance_axis, sonar_data[:, 0], color='grey', marker='.', label='Emitter')
    ax.plot(distance_axis, sonar_data[:, 1], color='blue',  marker='.', label='Left Channel')
    ax.plot(distance_axis, sonar_data[:, 2], color='red',   marker='.', label='Right Channel')
    # ---- Adaptive ticks for main x and y axes ----
    x_ticks, _ = Utils.make_ticks(x_min, x_max, steps=[0.025, 0.05, 0.1, 0.2, 0.5, 1], preferred=9)
    y_ticks, _ = Utils.make_ticks(y_min, y_max, steps=[500, 1000, 2000, 5000], preferred=8)
    if x_ticks.size: ax.set_xticks(x_ticks)
    if y_ticks.size: ax.set_yticks(y_ticks)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    # ---- Adaptive top axis using indices ----
    n = len(distance_axis)
    # Use indices 0..n-1 as the "range" and pick adaptive ticks
    index_ticks, _ = Utils.make_ticks(0, n-1, steps=[10, 20, 50, 100, 200], preferred=10)
    ax_top = ax.secondary_xaxis('top')
    # Convert index ticks into positions on the distance axis
    pos_ticks = distance_axis[index_ticks.astype(int)]
    ax_top.set_ticks(pos_ticks)
    ax_top.set_xticklabels([str(int(i)) for i in index_ticks])
    ax_top.set_xlabel("Index")
    ax.grid(True, which='both', axis='both')

    meta_result['x_min'] = x_min
    meta_result['x_max'] = x_max
    meta_result['y_min'] = y_min
    meta_result['y_max'] = y_max
    return meta_result


def plot_sonar_package(sonar_package, file_name=None, close_after=False):
    echo_located = sonar_package.get('echo_located', False)
    distance_correction_applied = sonar_package.get('distance_correction_applied', False)
    iid_correction_applied = sonar_package.get('iid_correction_applied', False)
    title = 'Raw Sonar Data'
    plot_settings = None
    sonar_data = sonar_package['sonar_data']
    distance_axis = sonar_package['raw_distance_axis']
    xlab = 'Raw Distance [m]'
    max_sonar_data = np.max(sonar_data)
    if distance_correction_applied: distance_axis = sonar_package['corrected_distance_axis']
    if echo_located or distance_correction_applied:
        title = 'Locate Echo Results'
        onset = sonar_package['onset']
        offset = sonar_package['offset']
        crossed = sonar_package['crossed']
        threshold = sonar_package['threshold']
        onset_distance = distance_axis[onset]
        offset_distance = distance_axis[offset - 1]
        y_min = np.min(sonar_data) - 500
        y_max = np.max(threshold)
        y_max = max(y_max, max_sonar_data) + 500
        plot_settings = {'y_min': y_min, 'y_max': y_max}

    if distance_correction_applied: xlab = 'Corrected Distance [m]'

    plt.figure()
    plot_sonar_output = plot_sonar_data(distance_axis, sonar_package, plot_settings)
    plt.title(title)
    plt.tight_layout()

    if echo_located and crossed:
        plt.plot(distance_axis, threshold, color='black', linestyle='--', label='Threshold')
        plt.axvspan(onset_distance, offset_distance, color='gray', alpha=0.3, label='Integration Window')
        plt.ylim(y_min, y_max)

    if iid_correction_applied:
        iid = sonar_package['corrected_iid']
        side_code = sonar_package['side_code']
        plt.suptitle(f'Corrected IID: {iid:.2f} dB ({side_code})', y=0.98, fontsize=10)

    plt.xlabel(xlab)
    plt.ylabel('Amplitude [a.u.]')
    plt.legend(loc='upper right')
    plt.tight_layout()

    ax = plt.gca()
    ax.set_facecolor('#f5f5dc')

    if file_name is not None: plt.savefig(file_name, bbox_inches='tight')
    if not close_after: plt.show()
    if close_after: plt.close()







