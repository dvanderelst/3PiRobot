import time
from Library import Dialog
from Library import Client
from Library import ControlParameters
from Library import LorexTracker
from Library import DataStorage
from Library import PauseControl
from LorexLib.Environment import capture_environment_layout

robot_number = 1
do_plot = True
selection_mode = 'first'

client = Client.Client(robot_number=robot_number)

while True:
    sonar_package = client.read_and_process(do_ping=True, plot=do_plot, selection_mode=selection_mode)
    client.step(distance=0.1)
    time.sleep(0.01)

#client.set_kinematics(linear_speed=0.1)
#time.sleep(5)