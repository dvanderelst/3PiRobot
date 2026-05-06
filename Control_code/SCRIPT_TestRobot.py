from Library import Client
import time
# ─── Baseline collection Settings ────
robot_nr = 1
ip=None
angle = 10
distance = 0
# ─────────────────────────────────────

client = Client.Client(robot_nr, ip=ip)
client.change_free_ping_period(0) #To ensure no free pings are done during calibration
#override the sample rate and samples to ensure consistency in calibration
#client.configuration.sample_rate = 10000
#client.configuration.samples = 200

robot_name = client.configuration.robot_name
start = time.time()
try:
    client.step(angle=int(angle), distance=distance, linear_speed=0.1)
except Exception as e:
    print(f"Error during step command: {e}")

end = time.time()
print(end - start)
client.read_and_process(plot=True)