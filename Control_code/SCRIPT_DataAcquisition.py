import time
from Library import Dialog
from Library import Client
from Library import Utils
from Library import LorexTracker
from Library import DataStorage
from LorexLib.Environment import capture_environment_layout
robot_number = 1
session = 'session5'
wait_for_confirmation = False
do_rotation = True
do_translation = True
do_plot = True
selection_mode = 'first'
max_steps = 150

client = Client.Client(robot_number=robot_number)
tracker = LorexTracker.LorexTracker()
writer = DataStorage.DataWriter(session, autoclear=True, verbose=False)
writer.add_file('Library/Settings.py')
capture_environment_layout(save_root=f'Data/{session}')

rotation = 0 # to avoid pycharm complaining
for step in range(max_steps):
    client.acquire('ping')
    Utils.sleep_ms(0, 50)
    sonar_package = client.read_and_process(do_ping=False, plot=do_plot, selection_mode=selection_mode)
    iid = sonar_package['corrected_iid']
    corrected_distance = sonar_package['corrected_distance']
    side_code = sonar_package['side_code']
    position  = tracker.get_position(robot_number)

    step_distance = 0.20
    rotation_magnitude = 10

    if do_rotation:
        if corrected_distance < 0.8: rotation_magnitude = 30
        if corrected_distance < 0.4: rotation_magnitude = 50
        if side_code == 'L': rotation = rotation_magnitude * 1
        if side_code == 'R': rotation = rotation_magnitude * - 1
        client.step(angle=rotation)
    if do_translation:
        if corrected_distance < 0.8: step_distance = 0.10
        if corrected_distance < 0.2: step_distance = 0.05
        client.step(distance=step_distance)

    if wait_for_confirmation:
        response = Dialog.ask_yes_no("Continue", min_size=(400, 200))
        if response[0] == 'No': break
    else:
        time.sleep(1)


    motion = {'distance':step_distance, 'rotation':rotation}
    writer.save_data(sonar_package=sonar_package, position=position, motion=motion)



