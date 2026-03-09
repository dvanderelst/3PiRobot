import time
import random
from Library import Dialog
from Library import Client
from Library import ControlParameters
from Library import LorexTracker
from Library import DataStorage
from Library import PauseControl
from Library import PushOver
from LorexLib.Environment import capture_environment_layout


robot_number = 1
session = 'sessionB05'
wait_for_confirmation = False
do_rotation = True
do_translation = True
do_plot = True
selection_mode = 'first'
max_steps = 500
turn_probability = 1/25
turn_distance = 0.20

control = PauseControl.PauseControl()
client = Client.Client(robot_number=robot_number)
tracker = LorexTracker.LorexTracker()
writer = DataStorage.DataWriter(session, autoclear=True, verbose=False)
writer.add_file('Library/Settings.py')
writer.add_file('Library/ControlParameters.py')
snapshot = capture_environment_layout(save_root=f'Data/{session}')
parameters = ControlParameters.Parameters()
parameters.plot(save_path=writer.files_folder())
parameters.plot()


for x in range(5):
    client.acquire('ping')
    time.sleep(0.5)

rotation = 0 # to avoid pycharm complaining
PushOver.send('Data acquisition started: {session}')
for step in range(max_steps):
    control.wait_if_paused()
    #client.acquire('ping')
    #time.sleep(0.25)
    sonar_package = client.read_and_process(do_ping=True, plot=do_plot, selection_mode=selection_mode)
    if sonar_package is None:
        print(f"Warning: No data received at step {step}, skipping this iteration...")
        position = tracker.get_position(robot_number)
        writer.save_data(sonar_package=None, position=position, motion={'distance': 0, 'rotation': 0})
        continue
    sonar_package['robot_number'] = robot_number
    corrected_iid = sonar_package['corrected_iid']
    corrected_distance = sonar_package['corrected_distance']
    side_code = sonar_package['side_code']
    position  = tracker.get_position(robot_number)
    rob_x = position['x']
    rob_y = position['y']
    rob_yaw_deg = position['yaw_deg']

    print(f"Step {step}: IID {corrected_iid}, Distance {corrected_distance:.2f} m, Side {side_code}, Position X:{rob_x}, Y:{rob_y}, Yaw:{rob_yaw_deg}")

    step_distance = parameters.translation_magnitude(corrected_distance)
    rotation_magnitude = parameters.rotation_magnitude(corrected_distance)

    step_distance = float(step_distance)
    rotation_magnitude = float(rotation_magnitude)

    random_number = random.random()
    do_half_turn = (random_number < turn_probability) or (corrected_distance < turn_distance)
    if do_half_turn: rotation_magnitude = 180

    if do_rotation:
        if side_code == 'L': rotation = rotation_magnitude * 1
        if side_code == 'R': rotation = rotation_magnitude * - 1
        client.step(angle=rotation)
        time.sleep(0.15)
    if do_translation:
        if corrected_distance < 0.75: step_distance = 0.10
        if corrected_distance < 0.3: step_distance = 0.05
        client.step(distance=step_distance)
        time.sleep(0.15)

    if wait_for_confirmation:
        response = Dialog.ask_yes_no("Continue", min_size=(400, 200))
        if response[0] == 'No': break
    else:
        time.sleep(0.25)


    motion = {'distance':step_distance, 'rotation':rotation}
    writer.save_data(sonar_package=sonar_package, position=position, motion=motion)
    if step % 100 == 0: PushOver.send(f"Data acquisition progress: {step}/{max_steps} steps completed.")

PushOver.send(f"Data acquisition completed: {max_steps} steps completed for session {session}.")
