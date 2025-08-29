import time
import pickle
import numpy as np
from matplotlib import pyplot as plt
from Library import FileOperations
from Library import Logging
import easygui

def angles2steps(angles):
    angles = np.array(angles)
    steps = np.diff(angles)
    steps = np.insert(steps, 0, angles[0])
    steps = list(steps)
    return steps


def baseline_data(client, nr_repeats=10, show_plot=True, note=''):
    data_collected = []
    data = None
    distance_axis = None
    for i in range(nr_repeats):
        message = f"Ping {i + 1}/{nr_repeats}..."
        Logging.print_message('Baseline', message, category='INFO')
        data, distance_axis, timing_info = client.ping()
        data_collected.append(data)
        time.sleep(0.25)
    data_collected = np.array(data_collected)

    left_measurements = data_collected[:, :, 1]
    right_measurements = data_collected[:, :, 2]

    left_measurements = np.transpose(left_measurements)
    right_measurements = np.transpose(right_measurements)

    baseline_left = np.mean(left_measurements, axis=1)
    baseline_right = np.mean(right_measurements, axis=1)

    overall_max = np.max(data) + 500
    overall_min = np.min(data) - 500

    robot_name = client.configuration.robot_name

    baseline_extent_samples = client.configuration.baseline_extent
    baseline_extent_distance = distance_axis[baseline_extent_samples]


    plt.figure()
    plt.subplot(211)
    plt.plot(left_measurements, color='blue', alpha=0.5)
    plt.plot(baseline_left, color='black')
    plt.ylim(overall_min, overall_max)
    plt.ylabel('Amplitude')
    plt.xlabel('Index')
    plt.grid()
    plt.title(f'{robot_name}: Left')
    plt.axvspan(0, baseline_extent_samples, color='grey', alpha=0.25, label='Baseline Extent')

    plt.subplot(212)
    plt.plot(distance_axis, right_measurements, color='red', alpha=0.5)
    plt.plot(distance_axis, baseline_right, color='black')
    plt.ylim(overall_min, overall_max)
    plt.xlabel('Uncorrected Distance')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.title(f'{robot_name}: Right')
    plt.axvspan(0, baseline_extent_distance, color='grey', alpha=0.25, label='Baseline Extent')

    plt.legend(loc='lower left')
    plt.tight_layout()
    if show_plot: plt.show()

    baseline_data = {
        'robot_name': robot_name,
        'left_measurements': left_measurements,
        'right_measurements': right_measurements,
        'baseline_left': baseline_left,
        'baseline_right': baseline_right,
        'distance_axis': distance_axis,
        'client_configuration': client.configuration,
        'note': note
    }

    plot_filename = FileOperations.get_function_plot_path(robot_name, 'baseline')
    plt.savefig(plot_filename)
    return baseline_data

def distance_data(client, real_distance, nr_repeats=5, cutoff_index=None):
    easygui.msgbox(f"Place the object at {real_distance} meters and press OK.")
    # Do measurements at zero angle and distance 1
    zero_measurements = []
    for n in range(nr_repeats):
        message = f"Measuring at zero angle, distance {real_distance}, iteration {n + 1}"
        Logging.print_message('Distance data', message, category='INFO')
        result = client.ping_process(cutoff_index=cutoff_index)
        zero_measurements.append(result)
        time.sleep(0.25)
    return zero_measurements


def sweep_data(client, sweep_angles):
    start_angle = sweep_angles[0]
    end_angle = sweep_angles[-1]
    nr_of_steps = len(sweep_angles)
    sweep_steps = angles2steps(sweep_angles)

    easygui.msgbox(f"Press ok to start the sweep from {start_angle} to {end_angle} degrees.")
    Logging.print_message('Sweep', f'Starting sweep with {nr_of_steps} steps', category='INFO')

    # Prepare storage
    sweep_results = []
    sweep_angles = []

    # Sweep loop
    for index, step in enumerate(sweep_steps):
        c
        message = f"Measuring at {current_angle} degrees, step {i + 1}/{nr_of_steps}"
        Logging.print_message('Sweep', message, category='INFO')

        # Run ping and save plot
        figure_file = f'iid_{i}_{current_angle}.png'
        robot_name = client.configuration.robot_name
        figure_file = FileOperations.get_function_plot_path(robot_name, figure_file)
        result = client.ping_process(plot=figure_file, close_after=True, cutoff_index=None)

        # Store results
        sweep_results.append(result)

        # Step only if not at the end
        if i < nr_of_steps - 1:
            client.step(angle=steps[i + 1])
            time.sleep(1.5)

    client.step(angle=-1 * sum(steps) / 2)
    return sweep_results, sweep_angles