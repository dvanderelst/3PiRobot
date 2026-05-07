from Library import LorexTracker
from Library import DataStorage
from LorexLib.Environment import capture_environment_layout

DATA_FOLDER = 'TempOutput'

session = 'Acquisition04'
tracker = LorexTracker.LorexTracker()
writer = DataStorage.DataWriter(session, autoclear=True, verbose=False)
writer.add_file('Library/Settings.py')
snapshot = capture_environment_layout(save_root=f'{DATA_FOLDER}/{session}')