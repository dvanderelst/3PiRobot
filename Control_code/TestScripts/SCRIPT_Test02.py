import time
from Library import Dialog
from Library import Client
from Library import Utils
from matplotlib import pyplot as plt
import random


client1 = Client.Client(robot_number=1)
client2 = Client.Client(robot_number=2)
client3 = Client.Client(robot_number=3)

clients = [client1, client2, client3]

wait_for_confirmation = False
do_rotation = True
do_translation = True
selection_mode = 'first'

counter = 0
while True:
    plt.close('all')
    results = []
    random.shuffle(clients)
    for client in clients:
        client.acquire('ping')
        Utils.sleep_ms(0, 50)

    sonar_packages = []

    for client in clients:
        current_robot_name = client.configuration.robot_name
        plot = False
        #if counter % 20 == 0: plot = True

        sonar_package = client.read_and_process(do_ping=False, plot=plot, selection_mode=selection_mode)
        sonar_packages.append(sonar_package)
        iid = sonar_package['corrected_iid']
        corrected_distance = sonar_package['corrected_distance']
        side_code = sonar_package['side_code']

        magnitude = 5
        if corrected_distance < 0.6: magnitude = 20
        if corrected_distance < 0.4: magnitude = 50
        if side_code == 'L': client.set_kinematics(linear_speed=0.1, rotation_speed=magnitude)
        if side_code == 'R': client.set_kinematics(linear_speed=0.1, rotation_speed=-magnitude)

    time.sleep(1)
    counter = counter + 1

