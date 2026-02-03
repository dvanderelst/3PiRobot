from Library import DataProcessor
# Parameters
az_extent = 90
az_steps = 19  # Number of azimuth steps in profiles
sessions = ['session03', 'session04']



dp = DataProcessor.DataProcessor('session03')
#dp.plot_trajectory(show=True)
dp.load_sonar()
dp.load_profiles(-45, 45, 19)
dp.load_views(radius_mm=2500, output_size=(256, 256), plot_indices=range(50))
#dp.plot_all_sonar(indices=range(10), view=True, profile=True)
#dp.load_views(radius_mm=1500, opening_deg=45, output_size=(256, 256))

# cl = DataProcessor.DataCollection(sessions, cache_dir='cache')
# cl.load_views(show_example=True)
# cl.load_sonar()
# cl.sonar