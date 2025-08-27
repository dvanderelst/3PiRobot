import time
import easygui
from Library import Client
from matplotlib import pyplot as plt

client = Client.Client(0)

while True:
    plt.close('all')
    result = client.ping_process(plot=True, selection_mode='first')
    #iid = result['iid']
    #side_code = result['side_code']
    #corrected_distance = result['corrected_distance']
    #if side_code == 'L': client.step(angle=20)
    #if side_code == 'R': client.step(angle=-20)
    #time.sleep(0.5)
    #client.step(distance=0.1)
    response = easygui.ynbox()
    if not response: break

