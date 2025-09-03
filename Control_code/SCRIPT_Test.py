import time
import easygui
from Library import Client
from matplotlib import pyplot as plt

client = Client.Client(robot_number=1, ip='192.168.1.13')
client.change_free_ping_interval(0)
result = client.ping_process(plot=True)
message = result['full_message']
print(message)

# wait_for_confirmation = False
#
# while True:
#     plt.close('all')
#     result = client.ping_process(plot=True, selection_mode='first')
#     iid = result['corrected_iid']
#     corrected_distance = result['corrected_distance']
#     side_code = result['side_code']
#
#     if side_code == 'L': client.step(angle=20)
#     if side_code == 'R': client.step(angle=-20)
#     time.sleep(1)
#     client.step(distance=0.1)
#     if wait_for_confirmation:
#         response = easygui.ynbox()
#         if not response: break

