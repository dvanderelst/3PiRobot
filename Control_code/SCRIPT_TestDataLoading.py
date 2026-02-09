from Library import DataProcessor
import numpy as np
sessions = ['session03', 'session04', 'session06']

dc = DataProcessor.DataCollection(sessions)
dc.load_views(radius_mm=4000, opening_angle=180, output_size=(256, 256))
dc.load_profiles(opening_angle=180, steps=19)

views = dc.views
profiles = dc.profiles
centers = dc.profile_centers

sonar_distance = dc.get_field('sonar_package', 'corrected_distance')
sonar_iid = dc.get_field('sonar_package', 'corrected_iid')
rotation = dc.get_field('motion', 'rotation')

for session in sessions:
    dp = DataProcessor.DataProcessor(session)
    dp.load_views(radius_mm=4000, opening_angle=180, output_size=(256, 256))
    dp.load_profiles(opening_angle=180, steps=19)

    n = dp.n
    indices = range(0, 50)
    dp.plot_all_sonar(view=True, profile=True, indices=indices)
