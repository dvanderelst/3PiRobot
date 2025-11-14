import time
from matplotlib import pyplot as plt
from Library import Dialog
from Library import Client
from Library import Utils
from Library import LorexTracker

robot_number = 1



client = Client.Client(robot_number=robot_number)
tracker = LorexTracker.LorexTracker()

wait_for_confirmation = False
do_rotation = True
do_translation = True
do_plot = True
selection_mode = 'first'

x_pos = []
y_pos = []
yaw_pos = []

for step in range(50):
    client.acquire('ping')
    Utils.sleep_ms(0, 50)
    sonar_package = client.read_and_process(do_ping=False, plot=do_plot, selection_mode=selection_mode)
    iid = sonar_package['corrected_iid']
    corrected_distance = sonar_package['corrected_distance']
    side_code = sonar_package['side_code']\

    x, y, yaw = tracker.get_position(robot_number)
    x_pos.append(x)
    y_pos.append(y)
    yaw_pos.append(yaw)

    if do_rotation:
        magnitude = 10
        if corrected_distance < 0.8: magnitude = 20
        if corrected_distance < 0.4: magnitude = 50
        if side_code == 'L': client.step(angle=magnitude)
        if side_code == 'R': client.step(angle=-magnitude)
    if do_translation:
        client.step(distance=0.1)

    if wait_for_confirmation:
        response = Dialog.ask_yes_no("Continue", min_size=(400, 200))
        if response[0] == 'No': break
    else:
        time.sleep(1)

#%%
plt.figure()
plt.scatter(x_pos, y_pos, c=yaw_pos, marker='o', cmap='hot')
plt.title('Robot Path')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.axis('equal')
plt.grid()
plt.show()
# while True:
#     plt.close('all')
#     results = []
#     random.shuffle(clients)
#     for client in clients:
#         client.acquire('ping')
#         Utils.sleep_ms(0, 50)
#
#     sonar_packages = []
#     index = 1
#     for client in clients:
#         current_robot_name = client.configuration.robot_name
#         if current_robot_name == 'Robot01': plot = True
#         else: plot = False
#         sonar_package = client.read_and_process(do_ping=False, plot=plot, selection_mode=selection_mode)
#         sonar_packages.append(sonar_package)
#         iid = sonar_package['corrected_iid']
#         corrected_distance = sonar_package['corrected_distance']
#         side_code = sonar_package['side_code']
#         index = index + 1
#
#         if do_rotation:
#             magnitude = 10
#             if corrected_distance < 0.8: magnitude = 20
#             if corrected_distance < 0.4: magnitude = 50
#             if side_code == 'L': client.step(angle=magnitude)
#             if side_code == 'R': client.step(angle=-magnitude)
#             time.sleep(1)
#         if do_translation:
#             client.step(distance=0.1)
#
#     if wait_for_confirmation:
#         response = Dialog.ask_yes_no("Continue",min_size=(400, 200))
#         if response[0] == 'No': break
#     else: time.sleep(1)
