import time
from Library import Dialog
from Library import Client
from Library import Utils
from Library import LorexTracker

robot_number = 1

client = Client.Client(robot_number=robot_number)
tracker = LorexTracker.LorexTracker()

wait_for_confirmation = False
do_rotation = False
do_translation = False
do_plot = True
selection_mode = 'first'

for step in range(1):
    client.acquire('ping')
    Utils.sleep_ms(0, 50)
    sonar_package = client.read_and_process(do_ping=False, plot=do_plot, selection_mode=selection_mode)
    iid = sonar_package['corrected_iid']
    corrected_distance = sonar_package['corrected_distance']
    side_code = sonar_package['side_code']
    position  = tracker.get_position(robot_number)





