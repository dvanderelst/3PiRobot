from Library import Client
from Library import FileOperations
from Library import Callibration
from Library import Process
import easygui
import numpy as np

# ─── Baseline collection Settings ────
robot_nr = 1
repeats = 2
real_distance1 = 0.3 # meters
real_distance2 = 0.5 # meters
angles = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
delete_calibration = True
collect_baseline = True
collect_distance_calibration = True
collect_sweep_data = True
# ─────────────────────────────────────

client = Client.Client(robot_nr, ip='192.168.1.13')
client.change_free_ping_period(0) #To ensure no free pings are done during calibration

robot_name = client.configuration.robot_name

if delete_calibration: FileOperations.delete_calibration(robot_name)
calibration = FileOperations.load_calibration(robot_name)
if calibration == {}: message = 'No calibration file found.\nStarting new calibration'
else: message = 'Calibration file found.\nUpdating calibration.'
easygui.msgbox(message)


if collect_baseline:
    user_message = "Starting baseline collection.\nPlease ensure no objects are in front of the robot and press OK."
    easygui.msgbox(user_message)
    baseline_data = Callibration.get_baseline_data(client, nr_repeats=repeats)
    calibration['baseline_present'] = True
    calibration['raw_distance_axis'] = baseline_data['raw_distance_axis']
    calibration['left_baseline'] = baseline_data['left_mean']
    calibration['right_baseline'] = baseline_data['right_mean']
    data, _, _ = client.ping()
    raw_result = Process.locate_echo(client, data, calibration)
    FileOperations.save_calibration(robot_name, calibration)


if collect_distance_calibration:
    easygui.msgbox(f"Place the object at {real_distance1} meters and press OK.")
    distance_data1 = Callibration.get_distance_data(client, calibration, real_distance1, nr_repeats=repeats)

    easygui.msgbox(f"Place the object at {real_distance2} meters and press OK.")
    distance_data2 = Callibration.get_distance_data(client, calibration, real_distance2, nr_repeats=repeats)

    raw_distances1 = distance_data1['raw_distances']
    raw_distances2 = distance_data2['raw_distances']
    raw_iid1 = distance_data1['raw_iid']
    raw_iid2 = distance_data2['raw_iid']
    zero_iids = np.concatenate((raw_iid1, raw_iid2))

    distance_fit_results = Callibration.distance_fit(robot_name, real_distance1, real_distance2, raw_distances1, raw_distances2)
    calibration['distance_present'] = True
    calibration['iid_present'] = True
    calibration['distance_coefficient'] = distance_fit_results['distance_coefficient']
    calibration['distance_intercept'] = distance_fit_results['distance_intercept']
    calibration['zero_iids'] = zero_iids
    FileOperations.save_calibration(robot_name, calibration)

# Sweep data collection does not contribute to calibration but
# is useful for visualizing the robot's sonar transfer function
if collect_sweep_data:
    easygui.msgbox(f"Press ok to start the sweep.")
    sweep_results = Callibration.get_sweep_data(client, calibration, angles)
    Callibration.plot_sweep_data(robot_name, sweep_results, calibration)
    # We save the sweep data as well because it might be useful for later plotting
    calibration['sweep_data'] =  sweep_results['sweep_data']
    calibration['sweep_onsets'] = sweep_results['sweep_onsets']
    calibration['sweep_iids'] = sweep_results['sweep_iids']
    calibration['sweep_angles'] = sweep_results['sweep_angles']
    FileOperations.save_calibration(robot_name, calibration)

print(calibration['distance_intercept'])
print(calibration['distance_coefficient'])

Callibration.print_calibration_content(calibration)