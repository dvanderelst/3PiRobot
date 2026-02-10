import time
from Library import Client

robot_number = 1
do_plot = True
selection_mode = 'first'

client = Client.Client(robot_number=robot_number)

while True:
    sonar_package = client.read_and_process(
        do_ping=True,
        plot=do_plot,
        selection_mode=selection_mode
    )
    if sonar_package is None:
        print("No sonar data received; retrying...")
        time.sleep(0.5)
        continue
    client.step(distance=0.1, wait_for_completion=True)
    time.sleep(0.05)

#client.set_kinematics(linear_speed=0.1)
#time.sleep(5)
