from Library import Client

# ─── Baseline collection Settings ────
robot_nr = 1
ip='192.168.1.15'
angle = 15
# ─────────────────────────────────────

client = Client.Client(robot_nr, ip=ip)
client.change_free_ping_period(0) #To ensure no free pings are done during calibration
#override the sample rate and samples to ensure consistency in calibration
#client.configuration.sample_rate = 10000
#client.configuration.samples = 200

robot_name = client.configuration.robot_name
client.step(angle=int(angle))
client.read_and_process(plot=True)