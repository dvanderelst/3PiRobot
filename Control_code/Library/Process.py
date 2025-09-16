import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from Library import Utils
from Library.Utils import draw_integration_box

def shift2right(arr, n):
    if n <= 0: return arr.copy()
    shifted = np.empty_like(arr)
    shifted[:n] = arr[0]
    shifted[n:] = arr[:-n]
    return shifted

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
        'sonar_package': sonar_package,
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
    return raw_results


def plot_locate_echo(raw_results, file_name=None, close_after=False, calibration=None):
    configuration = raw_results['client_configuration']
    sonar_package = raw_results['sonar_package']
    threshold = raw_results['threshold']
    onset = raw_results['onset']
    offset = raw_results['offset']
    crossed = raw_results['crossed']

    if calibration is None: calibration = {}
    distance_intercept = calibration.get('distance_intercept', None)  # b
    distance_coefficient = calibration.get('distance_coefficient', None)  # a

    distance_axis = sonar_package['raw_distance_axis']
    distance_axis_label = 'Raw Distance [m]'


    if distance_intercept is not None:
        distance_axis = distance_intercept + distance_coefficient * distance_axis
        distance_axis_label = 'Corrected Distance [m]'

    onset_distance = distance_axis[onset]
    offset_distance = distance_axis[offset]

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
    plt.show()

    # --- Results from locate_echo ---

    # #onset             = raw_results['onset']
    # #offset            = raw_results['offset']
    # #crossed           = raw_results['crossed']
    # #threshold_raw_arr = raw_results['threshold']                # evaluated on RAW axis originally
    # #selection_mode    = raw_results['selection_mode']
    # #threshold_fn      = raw_results.get('threshold_function')   # callable on RAW axis (if present)
    # #raw_axis          = raw_results['raw_distance_axis']        # RAW axis (meters, uncorrected)
    #
    # # --- Choose plotting axis & re-evaluate threshold if needed ---
    # # distance_axis = raw_axis.copy()
    # # threshold_for_plot = threshold_raw_arr  # default: raw axis
    #
    # xlabel = 'Raw Distance [m]'
    # use_corrected = (distance_coefficient is not None) and (distance_intercept is not None)
    #
    # if use_corrected:
    #     # Build corrected axis for plotting
    #     corrected_axis = distance_intercept + distance_coefficient * raw_axis
    #     xlabel = 'Corrected Distance [m]'
    #
    #     if threshold_fn is not None:
    #         # Map corrected x back to RAW x for evaluation: x_raw = (x_corr - b) / a
    #         # Guard against a == 0 just in case (fall back to raw axis)
    #         a = distance_coefficient
    #         b = distance_intercept
    #         if a != 0:
    #             x_raw_for_corr = (corrected_axis - b) / a
    #             threshold_for_plot = threshold_fn(x_raw_for_corr)
    #             distance_axis = corrected_axis
    #         else:
    #             # Degenerate fit: keep raw axis/threshold
    #             distance_axis = raw_axis
    #             threshold_for_plot = threshold_raw_arr
    #     else:
    #         # No function available to re-evaluate; keep raw threshold and raw axis
    #         # (Alternatively, you could still switch the axis and accept slight misalignment.)
    #         distance_axis = raw_axis
    #         threshold_for_plot = threshold_raw_arr
    #         xlabel = 'Raw Distance [m]'
    #
    # # --- Plot styling / ranges ---
    # fixed_onset = isinstance(selection_mode, int)
    # onset_color = 'black' if fixed_onset else 'green'
    #
    # sonar_data = data[:, 1:3]  # left & right
    # range_min = np.min(sonar_data) - 500
    # range_max = max(np.max(sonar_data), np.max(threshold_for_plot)) + 500
    # yrange = [range_min, range_max]
    #
    # # --- Figure ---
    # plt.figure(figsize=(12, 3))
    # ax1 = plt.subplot(111)
    #
    # # Left/right traces on the chosen axis
    # Utils.sonar_plot(received_data)
    #
    #
    # # Threshold (aligned with axis in use)
    # ax1.plot(distance_axis, threshold_for_plot, color='black', linestyle='--', label='Threshold')
    # # Integration window shading
    # min_in_window = np.min(data[onset:offset, :])
    # max_in_window = np.max(data[onset:offset, :])
    # window_start = distance_axis[onset]
    # window_end   = distance_axis[offset - 1]
    # if crossed: draw_integration_box(ax1, [window_start, window_end, min_in_window, max_in_window], color='gray', alpha=0.3, onset_color=onset_color)
    #
    # # Cosmetics

    # ax1.set_xlabel(xlabel)
    # ax1.set_ylabel("Amplitude")
    # ax1.set_title(f'Plot Locate Echo\nselection mode: {selection_mode}')
    # ax1.legend(loc='upper right')
    #

    # if file_name is not None: plt.savefig(file_name, dpi=300)
    # if close_after: plt.close()
    # else: plt.show()



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
