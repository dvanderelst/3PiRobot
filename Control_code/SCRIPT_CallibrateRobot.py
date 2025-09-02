from Library import Client
from Library import FileOperations
from Library import Callibration
from Library import Process
from os import path
import  easygui
import plots


# ─── Baseline collection Settings ────
robot_nr = 1
repeats = 3
real_distance1 = 0.3 # meters
real_distance2 = 0.5 # meters
angles = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
collect_baseline = False
collect_distance_calibration = False
collect_sweep_data = True
# ─────────────────────────────────────

client = Client.Client(robot_nr, ip='192.168.1.13')
client.change_free_ping_interval(0) #To ensure no free pings are done during callibration

robot_name = client.configuration.robot_name

calibration = FileOperations.load_calibration(robot_name)
if calibration == {}: message = 'No calibration file found.\nStarting new calibration'
else: message = 'Calibration file found.\nUpdating calibration.'
easygui.msgbox(message)


if collect_baseline:
    easygui.msgbox("Starting baseline collection.\nPlease ensure no objects are in front of the robot and press OK.")
    baseline_data = Callibration.get_baseline_data(client, nr_repeats=repeats)
    calibration['baseline_present'] = True
    calibration['raw_distance_axis'] = baseline_data['raw_distance_axis']
    calibration['left_baseline'] = baseline_data['left_mean']
    calibration['right_baseline'] = baseline_data['right_mean']
    data, _, _ = client.ping()
    raw_result = Process.process_sonar_data(client, data, calibration)
    FileOperations.save_calibration(robot_name, calibration)

if collect_distance_calibration:
    easygui.msgbox(f"Place the object at {real_distance1} meters and press OK.")
    distance_data1 = Callibration.get_distance_data(client, calibration, real_distance1, nr_repeats=repeats)

    easygui.msgbox(f"Place the object at {real_distance2} meters and press OK.")
    distance_data2 = Callibration.get_distance_data(client, calibration, real_distance2, nr_repeats=repeats)

    raw_distances1 = distance_data1['raw_distances']
    raw_distances2 = distance_data2['raw_distances']
    distance_fit_results = Callibration.distance_fit(robot_name, real_distance1, real_distance2, raw_distances1, raw_distances2)
    calibration['distance_coefficient'] = distance_fit_results['distance_coefficient']
    calibration['distance_intercept'] = distance_fit_results['distance_intercept']
    FileOperations.save_calibration(robot_name, calibration)


if collect_sweep_data:
    easygui.msgbox(f"Press ok to start the sweep.")
    sweep_results = Callibration.get_sweep_data(client, calibration, angles)
    Callibration.plot_sweep_data(robot_name, sweep_results)

    #todo: test the plot sweep data function
    #todo: doubble check to logic of this new implementation