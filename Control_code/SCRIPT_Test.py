import time
from Library import Dialog
from Library import Client
from matplotlib import pyplot as plt


#client = Client.Client(robot_number=2)
client = Client.Client(robot_number=1, ip='192.168.1.13')
client.change_free_ping_period(120)

wait_for_confirmation = False
do_rotation = True
do_translation = False
selection_mode = 'max'
while True:
    plt.close('all')
    result = client.ping_process(plot=True, selection_mode=selection_mode)
    iid = result['corrected_iid']
    corrected_distance = result['corrected_distance']
    side_code = result['side_code']

    if do_rotation:
        if side_code == 'L': client.step(angle=20)
        if side_code == 'R': client.step(angle=-20)
        time.sleep(1)
    if do_translation:
        client.step(distance=0.1)
    if wait_for_confirmation:
        response = Dialog.ask_yes_no("Continue",min_size=(400, 200))
        if response[0] == 'No': break
    else: time.sleep(1)

