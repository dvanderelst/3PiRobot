from Library import DataProcessor
import numpy as np
sessions = ['session03', 'session04', 'session06']

dc = DataProcessor.DataCollection(sessions)
dc.load_views(radius_mm=4000, opening_angle=180, output_size=(256, 256))
dc.load_profiles(opening_angle=180, steps=10)

views = dc.views
profiles = dc.profiles
sonar_distance = dc.get_field('sonar_package', 'corrected_distance')
sonar_iid = dc.get_field('sonar_package', 'corrected_iid')
