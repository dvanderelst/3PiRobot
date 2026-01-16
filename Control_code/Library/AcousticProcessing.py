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

def apply_correction(sonar_package, eps_db=0):
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
    meta_result = {}  # To store stuff we might want to return
    if plot_settings is None:
        plot_settings = {}

    sonar_data = sonar_package['sonar_data']
    effective_sample_rate = float(sonar_package['effective_fs_hz'])

    # ---- Axis ranges (with optional overrides) ----
    x_min = float(distance_axis[0])
    x_max = float(distance_axis[-1])
    y_min = float(np.nanmin(sonar_data)) - 1000.0
    y_max = float(np.nanmax(sonar_data)) + 1000.0
    if 'x_min' in plot_settings: x_min = float(plot_settings['x_min'])
    if 'x_max' in plot_settings: x_max = float(plot_settings['x_max'])
    if 'y_min' in plot_settings: y_min = float(plot_settings['y_min'])
    if 'y_max' in plot_settings: y_max = float(plot_settings['y_max'])

    # ---- Main axis ----
    ax = plt.gca()
    ax.plot(distance_axis, sonar_data[:, 0], color='grey', marker='.', label='Emitter')
    ax.plot(distance_axis, sonar_data[:, 1], color='blue',  marker='.', label='Left Channel')
    ax.plot(distance_axis, sonar_data[:, 2], color='red',   marker='.', label='Right Channel')

    # Adaptive ticks for main x and y axes
    x_ticks, _ = Utils.make_ticks(x_min, x_max, steps=[0.025, 0.05, 0.1, 0.2, 0.5, 1], preferred=9)
    y_ticks, _ = Utils.make_ticks(y_min, y_max, steps=[500, 1000, 2000, 5000], preferred=8)
    if x_ticks.size: ax.set_xticks(x_ticks)
    if y_ticks.size: ax.set_yticks(y_ticks)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Raw Distance [m]')
    ax.grid(True, which='both', axis='both')

    # ---- Helpers for mapping distance <-> index <-> time ----
    n = len(distance_axis)
    idx = np.arange(n)

    # Ensure monotonic for safe interpolation
    if not np.all(np.diff(distance_axis) >= 0):
        order = np.argsort(distance_axis)
        d_sorted = distance_axis[order]
        i_sorted = idx[order]
    else:
        d_sorted = distance_axis
        i_sorted = idx

    # distance -> index (continuous), and inverse
    def idx_of_dist(x):
        return np.interp(x, d_sorted, i_sorted, left=i_sorted[0], right=i_sorted[-1])

    def dist_of_idx(i):
        return np.interp(i, i_sorted, d_sorted, left=d_sorted[0], right=d_sorted[-1])

    # distance -> time (s), and inverse (time -> distance)
    def time_of_dist(x):
        return idx_of_dist(x) / effective_sample_rate

    def dist_of_time(t):
        return dist_of_idx(t * effective_sample_rate)

    # ---- Top axis #1: Sample index (secondary_xaxis) ----
    ax_samples = ax.secondary_xaxis('top', functions=(idx_of_dist, dist_of_idx))
    ax_samples.set_xlabel("Index")

    # Nice integer ticks for sample index
    index_ticks, _ = Utils.make_ticks(0, n - 1, steps=[10, 20, 50, 100, 200], preferred=10)
    if index_ticks.size:
        # Set ticks in "index space"; secondary axis handles mapping to distance positions
        ax_samples.set_xticks(index_ticks)
        ax_samples.set_xticklabels([str(int(i)) for i in index_ticks])

        # ---- Top axis #2: Time in milliseconds ----
    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.spines['top'].set_position(('axes', 1.15))
    ax_time.set_xlabel("Time [ms]")

    # Total duration in ms
    total_time_ms = (n - 1) / effective_sample_rate * 1000.0 if n > 1 else 0.0
    time_steps = [1, 5, 10]

    t_ticks, _ = Utils.make_ticks(0.0, total_time_ms, steps=time_steps, preferred=8)
    if t_ticks.size:
        # Convert ms ticks → distance positions
        pos_ticks = dist_of_time(t_ticks / 1000.0)  # back to seconds for mapping
        ax_time.set_xticks(pos_ticks)
        ax_time.set_xticklabels([f"{int(t)}" for t in t_ticks])  # integers in ms

    # ---- Legend (optional) ----
    if plot_settings.get('show_legend', False): ax.legend(loc='best')

    plt.tight_layout()

    # ---- Return some metadata ----
    meta_result['x_min'] = x_min
    meta_result['x_max'] = x_max
    meta_result['y_min'] = y_min
    meta_result['y_max'] = y_max
    meta_result['n_samples'] = n
    meta_result['total_time_s'] = (n - 1) / effective_sample_rate if n > 1 else 0.0
    meta_result['axes'] = { 'main': ax, 'samples_top': ax_samples, 'time_top': ax_time}
    return meta_result



def plot_sonar_package(sonar_package, file_name=None, close_after=False):
    echo_located = sonar_package.get('echo_located', False)
    distance_correction_applied = sonar_package.get('distance_correction_applied', False)
    iid_correction_applied = sonar_package.get('iid_correction_applied', False)
    configuration = sonar_package['configuration']
    robot_name = configuration.robot_name
    title = 'Raw Sonar Data'
    plot_settings = None
    sonar_data = sonar_package['sonar_data']
    distance_axis = sonar_package['raw_distance_axis']
    xlab = 'Raw Distance [m]'
    max_sonar_data = np.max(sonar_data)
    crossed = False
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

    if distance_correction_applied:
        title = 'Corrected Results'
        xlab = 'Corrected Distance [m]'

    plt.figure(figsize=(10, 5))
    plot_sonar_output = plot_sonar_data(distance_axis, sonar_package, plot_settings)
    main_axis = plot_sonar_output['axes']['main']
    # samples_top_axis = plot_sonar_output['axes']['samples_top']
    # time_top_axis = plot_sonar_output['axes']['time_top']

    main_axis.set_title(title)
    main_axis.plot(distance_axis, threshold, color='black', linestyle='--', label='Threshold')
    if echo_located and crossed:
        main_axis.axvspan(onset_distance, offset_distance, color='gray', alpha=0.3, label='Integration Window')
        main_axis.set_ylim(y_min, y_max)

    processed_msg = f'[{robot_name}]'
    if iid_correction_applied:
        iid = sonar_package['corrected_iid']
        side_code = sonar_package['side_code']
        processed_msg = f'[{robot_name}] Corrected IID: {iid:.2f} dB ({side_code})'

    # Show the robot name in the bottom left corner of the plot
    # Use relative coordinates (0 to 1)
    transform = plt.gcf().transFigure
    plt.text(0.01, 0.01, processed_msg, transform=transform, fontsize=15, verticalalignment='bottom', horizontalalignment='left', color='gray', alpha=1)

    main_axis.set_xlabel(xlab)
    main_axis.set_ylabel('Amplitude [a.u.]')

    main_axis.set_facecolor('#f5f5dc')
    main_axis.legend(loc='upper right')

    plt.tight_layout()

    if file_name is not None: plt.savefig(file_name, bbox_inches='tight')
    if not close_after: plt.show()
    if close_after: plt.close()







