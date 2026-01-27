import time
from Library import Dialog
from Library import Client
from Library import Utils
from Library import LorexTracker
from Library import DataStorage
from Library import DataProcessor
from Library import PauseControl
from LorexLib.Environment import capture_environment_layout


robot_number = 1
session = 'session8'
wait_for_confirmation = False
do_rotation = True
do_translation = True
do_plot = True
selection_mode = 'first'
max_steps = 500

control = PauseControl.PauseControl()
client = Client.Client(robot_number=robot_number)
tracker = LorexTracker.LorexTracker()
writer = DataStorage.DataWriter(session, autoclear=True, verbose=False)
writer.add_file('Library/Settings.py')
snapshot = capture_environment_layout(save_root=f'Data/{session}')

rotation = 0 # to avoid pycharm complaining
for step in range(max_steps):
    control.wait_if_paused()
    client.acquire('ping')
    Utils.sleep_ms(0, 50)
    sonar_package = client.read_and_process(do_ping=False, plot=do_plot, selection_mode=selection_mode)
    iid = sonar_package['corrected_iid']
    corrected_distance = sonar_package['corrected_distance']
    side_code = sonar_package['side_code']
    position  = tracker.get_position(robot_number)
    rob_x = position['x']
    rob_y = position['y']
    rob_yaw_deg = position['yaw_deg']

    print(f"Step {step}: IID {iid}, Distance {corrected_distance:.2f} m, Side {side_code}, Position X:{rob_x}, Y:{rob_y}, Yaw:{rob_yaw_deg}")



    step_distance = 0.20
    rotation_magnitude = 10

    if do_rotation:
        if corrected_distance < 0.5: rotation_magnitude = 20
        if corrected_distance < 0.3: rotation_magnitude = 30
        if side_code == 'L': rotation = rotation_magnitude * 1
        if side_code == 'R': rotation = rotation_magnitude * - 1
        client.step(angle=rotation)
    if do_translation:
        if corrected_distance < 0.5: step_distance = 0.10
        if corrected_distance < 0.3: step_distance = 0.05
        client.step(distance=step_distance)

    if wait_for_confirmation:
        response = Dialog.ask_yes_no("Continue", min_size=(400, 200))
        if response[0] == 'No': break
    else:
        time.sleep(0.5)


    motion = {'distance':step_distance, 'rotation':rotation}
    writer.save_data(sonar_package=sonar_package, position=position, motion=motion)



