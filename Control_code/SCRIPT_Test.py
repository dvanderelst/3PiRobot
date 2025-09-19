import time
from Library import Dialog
from Library import Client
from matplotlib import pyplot as plt
import random

#client = Client.Client(robot_number=2)
client1 = Client.Client(robot_number=1)
client2 = Client.Client(robot_number=2)
#client2.change_free_ping_period(110)

clients = [client1, client2]

wait_for_confirmation = False
do_rotation = True
do_translation = True
selection_mode = 'first'

while True:
    plt.close('all')
    results = []
    # Echolocation stage
    for client in clients:
        result = client.ping_process(plot=False, selection_mode=selection_mode)
        results.append(result)

    for index, client in enumerate(clients):
        result = results[index]
        current_client = clients[index]
        iid = result['corrected_iid']
        corrected_distance = result['corrected_distance']
        side_code = result['side_code']

        if do_rotation:
            magnitude = int(random.randrange(30, 40))
            if side_code == 'L': client.step(angle=magnitude)
            if side_code == 'R': client.step(angle=-magnitude)
            time.sleep(1)
        if do_translation:
            client.step(distance=0.1)

    if wait_for_confirmation:
        response = Dialog.ask_yes_no("Continue",min_size=(400, 200))
        if response[0] == 'No': break
    else: time.sleep(1)

