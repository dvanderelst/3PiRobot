#
# Keep this stub as it shows how to read in data. This code can be reused in other scripts.
# This code (and tracing the origin of the read in data) should also give any AI agent
# an idea of what the data looks like (format and semantics).
#
# Data summary (from DataReader/DataProcessor):
# - Each session is a folder under Data/ (e.g., session03). It contains dataNNNNN.dill files.
# - Each file payload has top-level keys: sonar_package, position, motion.
#   - sonar_package includes corrected_iid and corrected_distance (float), plus raw traces and calibration.
#   - position includes x, y, yaw_deg (robot pose in mm/deg).
#   - motion includes distance and rotation (odometry step info).
#
# DataCollection helpers:
# - dc.load_views(...) returns "views": uint8 array (N, H, W, 3) of conical local views
#   cropped from the arena image in the robot frame.
# - dc.load_profiles(...) returns "profiles": float32 array (N, steps) of min wall distance
#   per azimuth bin in the robot frame (computed from arena_annotated.png + meta.json).
# - dc.get_field('sonar_package', 'corrected_iid') returns target IID as float32 (N,).
# - All arrays are aligned by sample index after concatenation across sessions.


from Library import DataProcessor
from matplotlib import pyplot as plt
import numpy as np

sessions = ['session03', 'session04', 'session06', 'session07']

dc = DataProcessor.DataCollection(sessions)
dc.load_sonar(flatten=True)
#dc.load_views(radius_mm=4000, opening_angle=180, output_size=(256, 256))
#dc.load_profiles(opening_angle=180, steps=21)

#views = dc.views
#profiles = dc.profiles
sonar = dc.sonar
sonar_distance = dc.get_field('sonar_package', 'corrected_distance')
sonar_iid = dc.get_field('sonar_package', 'corrected_iid')

plt.figure()
plt.scatter(sonar_distance, sonar_iid)
plt.show()

print(len(sonar_iid))
print(len(np.unique(sonar_iid)))

plt.figure()
plt.hist(sonar_iid, bins=50)
plt.show()

plt.figure()
plt.hist(sonar_distance, bins=50)
plt.show()