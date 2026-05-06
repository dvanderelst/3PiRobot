from Library import Client
import time
# ─── Baseline collection Settings ────
robot_nr = 1
ip=None
angle = 10
distance = 0
# ─────────────────────────────────────



client = Client.Client(robot_nr, ip=ip)
client.change_free_ping_period(0)

for x in range(5):
    client.acquire('ping')
    time.sleep(0.5)

client.step(angle=angle, distance=distance)

client.read_and_process(plot=True)