import time
import easygui
from scipy.interpolate import (interp1d)
import Client

client = Client.Client('192.168.200.38')
client.name = 'Robot01'
client.verbose = True
client.change_settings('wheel_base_mm', 80)
# rotation_function = interp1d([0, 0.2, 0.5], [30, 20, 0], kind='linear', fill_value=(30, 0), bounds_error=False)
#
# while True:
#     result = client.ping_process(rate=20000, samples=100, plot=True)
#     iid = result['iid']
#     distance = result['distance']
#     rotation_sign = 1
#     if iid > 0: rotation_sign = -1
#     rotational_velocity = rotation_function(distance) * rotation_sign * 3
#     print(distance, iid, rotational_velocity)
#     #client.set_kinematics(0, rotational_velocity)
#     time.sleep(0.5)
#     client.stop()
#     response = easygui.ynbox()
#     if not response: break
#
