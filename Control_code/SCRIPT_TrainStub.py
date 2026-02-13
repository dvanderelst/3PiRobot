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
# - dc.load_sonar(flatten=False) returns "sonar": float32 array (N, samples, 2) containing
#   the raw sonar envelopes from left and right ears (200 samples per ear, 2 ears).
#   Shape: (N, 200, 2) where N=number of samples, 200=time samples, 2=left/right ears.
# - dc.get_field('sonar_package', 'corrected_iid') returns target IID as float32 (N,).
# - All arrays are aligned by sample index after concatenation across sessions.
#
# Data insights from running this script:
# - Sonar data shape: (2500, 200, 2) for 5 sessions × 500 samples each
# - Profile data shape: (2500, 21) for 21 azimuth bins across 180° opening angle
# - Sonar contains time-intensity profiles (envelopes) from left and right ears
# - Profiles contain minimum wall distance per azimuth bin in robot frame
# - Both datasets are perfectly aligned by sample index


from Library import DataProcessor
from matplotlib import pyplot as plt
import numpy as np

sessions = ['sessionB01', 'sessionB02', 'sessionB03', 'sessionB04', 'sessionB05']

dc = DataProcessor.DataCollection(sessions)
dc.load_sonar(flatten=False)

dc.load_profiles(opening_angle=180, steps=21)
#dc.load_views(radius_mm=4000, opening_angle=180, output_size=(256, 256))

#views = dc.views
#profiles = dc.profiles
sonar = dc.sonar
sonar_distance = dc.get_field('sonar_package', 'corrected_distance')
sonar_iid = dc.get_field('sonar_package', 'corrected_iid')

print(len(sonar_iid))
print(len(np.unique(sonar_iid)))

plt.figure()
plt.hist(sonar_iid, bins=50)
plt.title('Histogram of sonar IID')
plt.show()

plt.figure()
plt.hist(sonar_distance, bins=50)
plt.title('Histogram of sonar distances')
plt.show()