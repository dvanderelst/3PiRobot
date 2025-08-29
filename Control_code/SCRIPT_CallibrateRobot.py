from Library import Client
from Library import Callibration
from Library import Logging


# ─── Baseline collection Settings ────
robot_nr = 2
repeats = 10
note = 'a note goes here'
# ─────────────────────────────────────

client = Client.Client(robot_nr)

cutoff_index = 80 # values beyond this index will not be used for IID calculation
distance1 = 0.3
distance2 = 0.5
angles = [-40, -30, -20, -10, 0, 10, 20, 30, 40]


angle_steps = Callibration.angles2steps(angles)
baseline_data = Callibration.baseline_data(client, nr_repeats=repeats, show_plot=True, note=note)
distance_data1 = Callibration.distance_data(client, real_distance=distance1, cutoff_index=cutoff_index)
distance_data2 = Callibration.distance_data(client, real_distance=distance2, cutoff_index=cutoff_index)

