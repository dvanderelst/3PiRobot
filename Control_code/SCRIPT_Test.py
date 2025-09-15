import time
from Library import Dialog
from Library import Client
from matplotlib import pyplot as plt


#client = Client.Client(robot_number=2)
client1 = Client.Client(robot_number=1)
client2 = Client.Client(robot_number=2)
client2.change_free_ping_period(120)

wait_for_confirmation = True
do_rotation = True
do_translation = False
selection_mode = 'max'

while True:
    plt.close('all')
    result = client1.ping_process(plot=True, selection_mode=selection_mode)
    iid = result['corrected_iid']
    corrected_distance = result['corrected_distance']
    side_code = result['side_code']

    if do_rotation:
        if side_code == 'L': client1.step(angle=20)
        if side_code == 'R': client1.step(angle=-20)
        time.sleep(1)
    if do_translation:
        client1.step(distance=0.1)
    if wait_for_confirmation:
        response = Dialog.ask_yes_no("Continue",min_size=(400, 200))
        if response[0] == 'No': break
    else: time.sleep(1)

