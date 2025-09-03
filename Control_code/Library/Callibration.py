import time
import numpy as np
from matplotlib import pyplot as plt
from Library import FileOperations
from Library import Logging
from Library import Process
from Library import Utils



def angles2steps(angles):
    angles = np.array(angles)
    steps = np.diff(angles)
    steps = np.insert(steps, 0, angles[0])
    steps = list(steps)
    return steps


def get_baseline_data(client, nr_repeats=10):
    data_collected = []
    raw_distance_axis = None
    for i in range(nr_repeats):
        message = f"Baseline data: {i + 1}/{nr_repeats}..."
        Logging.print_message('Baseline', message, category='INFO')
        data, raw_distance_axis, timing_info = client.ping()
        data_collected.append(data)
        time.sleep(0.25)
    data_collected = np.array(data_collected)

    left_measurements = data_collected[:, :, 1]
    right_measurements = data_collected[:, :, 2]

    left_measurements = np.transpose(left_measurements)
    right_measurements = np.transpose(right_measurements)

    left_mean = np.mean(left_measurements, axis=1)
    right_mean = np.mean(right_measurements, axis=1)

    overall_max = np.max(data_collected) + 500
    overall_min = np.min(data_collected) - 500
    robot_name = client.configuration.robot_name
    baseline_extent_samples = client.configuration.baseline_extent
    baseline_extent_distance = raw_distance_axis[baseline_extent_samples]

    plt.figure()
    plt.subplot(211)
    plt.plot(left_measurements, color='blue', alpha=0.5)
    plt.plot(left_mean, color='black')
    plt.ylim(overall_min, overall_max)
    plt.ylabel('Amplitude')
    plt.xlabel('Index')
    plt.grid()
    plt.title(f'{robot_name}: Left')
    plt.axvspan(0, baseline_extent_samples, color='grey', alpha=0.25, label='Baseline Extent')
    plt.subplot(212)
    plt.plot(raw_distance_axis, right_measurements, color='red', alpha=0.5)
    plt.plot(raw_distance_axis, right_mean, color='black')
    plt.ylim(overall_min, overall_max)
    plt.xlabel('Uncorrected Distance')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.title(f'{robot_name}: Right')
    plt.axvspan(0, baseline_extent_distance, color='grey', alpha=0.25, label='Baseline Extent')
    plt.legend(loc='lower left')
    plt.tight_layout()

    plot_filename = FileOperations.get_calibration_plot(robot_name, 'baseline_data')
    plt.savefig(plot_filename)
    plt.show()

    data = {}
    data['left_measurements'] = left_measurements
    data['right_measurements'] = right_measurements
    data['left_mean'] = left_mean
    data['right_mean'] = right_mean
    data['raw_distance_axis'] = raw_distance_axis
    data['client_configuration'] = client.configuration
    data['robot_name'] = robot_name
    return data

def get_distance_data(client, calibration, real_distance, nr_repeats=10):
    raw_results = []
    robot_name = client.configuration.robot_name
    close_after = True
    for i in range(nr_repeats):
        if i == nr_repeats - 1: close_after = False
        plot_basename = f'dist_{real_distance:.2f}_repeat_{i + 1}.png'
        file_name = FileOperations.get_calibration_plot(robot_name, plot_basename)
        message = f"Distance data: {i + 1}/{nr_repeats}..."
        Logging.print_message('Baseline', message, category='INFO')
        data, _, _ = client.ping()
        raw_result = Process.locate_echo(client, data, calibration, selection_mode='first')
        Process.plot_locate_echo(raw_result, file_name=file_name, close_after=close_after)
        raw_results.append(raw_result)
        raw_results.append(raw_result)
        time.sleep(0.25)
    raw_distances = np.array([result['raw_distance'] for result in raw_results])
    zeros = np.array([result['raw_iid'] for result in raw_results])
    results = {}
    results['raw_distances'] = raw_distances
    results['raw_iid'] = zeros
    return results


def distance_fit(robot_name, distance1, distance2, raw_distances1, raw_distances2):
    distance_coefficient, distance_intercept = Utils.fit_linear_calibration(distance1, raw_distances1, distance2, raw_distances2)
    max_raw_distance = max(np.max(raw_distances1), np.max(raw_distances2))
    raw_distances_interpolated = np.linspace(0, max_raw_distance * 1.25, 100)
    fitted_real_distances = distance_coefficient * raw_distances_interpolated + distance_intercept

    plt.figure()
    plt.plot(raw_distances1, [distance1] * len(raw_distances1), 'o', label=f'{distance1} m measurements')
    plt.plot(raw_distances2, [distance2] * len(raw_distances2), 'o', label=f'{distance2} m measurements')
    plt.plot(raw_distances_interpolated, fitted_real_distances, '-', label='Fitted line')
    plt.xlabel('Raw Distance (m)')
    plt.ylabel('Real Distance (m)')
    plt.title('Distance Calibration Fit')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plot_filename = FileOperations.get_calibration_plot(robot_name, 'distance_fit')
    plt.savefig(plot_filename)
    plt.show()

    distance_fit_results = {}
    distance_fit_results['distance_coefficient'] = distance_coefficient
    distance_fit_results['distance_intercept'] = distance_intercept
    distance_fit_results['raw_distances_interpolated'] = raw_distances_interpolated
    distance_fit_results['fitted_real_distances'] = fitted_real_distances
    return distance_fit_results

#
def get_sweep_data(client, calibration, sweep_angles):
    nr_of_steps = len(sweep_angles)
    sweep_steps = angles2steps(sweep_angles)
    print(sweep_steps)

    all_sweep_data = []
    for index, step in enumerate(sweep_steps):
        current_angle = sweep_angles[index]
        message = f"Measuring at {current_angle} degrees, step {index + 1}/{nr_of_steps}"
        Logging.print_message('Sweep', message, category='INFO')
        client.step(angle=int(step))
        time.sleep(1.5)
        data, _, _ = client.ping()
        raw_result = Process.locate_echo(client, data, calibration, selection_mode='first')
        all_sweep_data.append(raw_result)

    sweep_data = np.array([result['data'] for result in all_sweep_data])
    sweep_onsets = np.array([result['onset'] for result in all_sweep_data])
    sweep_iids = np.array([result['raw_iid'] for result in all_sweep_data])

    results = {}
    results['all_data'] = all_sweep_data
    results['sweep_angles'] = sweep_angles
    results['sweep_data'] = sweep_data
    results['sweep_onsets'] = sweep_onsets
    results['sweep_iids'] = sweep_iids
    return results


def plot_sweep_data(robot_name, sweep_results, calibration=None):
    if calibration is None: calibration = {}

    zero_iids = calibration.get('zero_iids', [])
    zeros = [0] * len(zero_iids)

    sweep_data =  sweep_results['sweep_data']
    sweep_onsets = sweep_results['sweep_onsets']
    sweep_iids = sweep_results['sweep_iids']
    sweep_angles = sweep_results['sweep_angles']
    nr_of_steps = len(sweep_angles)

    left_sweep_data = sweep_data[:, :, 1]
    right_sweep_data = sweep_data[:, :, 2]
    sweep_differences = left_sweep_data - right_sweep_data

    plot_filename = FileOperations.get_calibration_plot(robot_name, 'sweep_results')

    plt.figure(figsize=(15, 6))
    plt.subplot(221)
    plt.imshow(left_sweep_data, aspect='auto')
    plt.colorbar()
    plt.yticks(ticks=np.arange(nr_of_steps), labels=sweep_angles)
    plt.plot(sweep_onsets, np.arange(nr_of_steps), marker='o', linestyle='', color='red', alpha=0.5)
    plt.title('Left Data')

    plt.subplot(222)
    plt.imshow(right_sweep_data, aspect='auto')
    plt.colorbar()
    plt.yticks(ticks=np.arange(nr_of_steps), labels=sweep_angles)
    plt.plot(sweep_onsets, np.arange(nr_of_steps), marker='o', linestyle='', color='red', alpha=0.5)
    plt.title('Right Data')

    plt.subplot(223)
    plt.imshow(sweep_differences, aspect='auto')
    plt.colorbar()
    plt.yticks(ticks=np.arange(nr_of_steps), labels=sweep_angles)
    plt.plot(sweep_onsets, np.arange(nr_of_steps), marker='o', linestyle='', color='red', alpha=0.5)
    plt.title('Data Differences')

    plt.subplot(224)
    plt.plot(sweep_angles, sweep_iids, marker='o', linestyle='-', color='red', label='Sweep IIDs', alpha=0.5)
    plt.plot(zeros, zero_iids, marker='o', linestyle='', color='blue', label='IIDs from distance calibration', alpha=0.5)
    plt.grid()
    plt.xlabel('Angle [degrees]')
    plt.ylabel('Raw IIDs')
    plt.title('Raw IID vs Angle')
    plt.legend(loc='lower left')
    plt.savefig(plot_filename)
    plt.show()
