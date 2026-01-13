import time
from Library import Dialog
from Library import Client
from Library import Utils
from matplotlib import pyplot as plt
import random


client1 = Client.Client(robot_number=1)
client2 = Client.Client(robot_number=2)
client3 = Client.Client(robot_number=3)

clients = [client1,client2,client3]

wait_for_confirmation = False
do_rotation = True
do_translation = True
selection_mode = 'first'


while True:
    plt.close('all')
    results = []
    random.shuffle(clients)
    for client in clients:
        client.acquire('ping')
        Utils.sleep_ms(0, 50)

    sonar_packages = []
    index = 1
    for client in clients:
        current_robot_name = client.configuration.robot_name
        if current_robot_name == 'Robot01': plot = True
        else: plot = False
        sonar_package = client.read_and_process(do_ping=False, plot=plot, selection_mode=selection_mode)
        sonar_packages.append(sonar_package)
        iid = sonar_package['corrected_iid']
        corrected_distance = sonar_package['corrected_distance']
        side_code = sonar_package['side_code']
        index = index + 1

        if do_rotation:
            magnitude = 10
            if corrected_distance < 0.8: magnitude = 20
            if corrected_distance < 0.4: magnitude = 50
            if side_code == 'L': client.step(angle=magnitude)
            if side_code == 'R': client.step(angle=-magnitude)
            time.sleep(1)
        if do_translation:
            client.step(distance=0.1)

    if wait_for_confirmation:
        response = Dialog.ask_yes_no("Continue",min_size=(400, 200))
        if response[0] == 'No': break
    else: time.sleep(1)
