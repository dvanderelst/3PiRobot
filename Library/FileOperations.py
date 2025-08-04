import os
import shutil
from Library import Settings

def get_spatial_function_path(robot_name):
    return os.path.join(Settings.spatial_functions_folder, f'spatial_function_{robot_name}.pck')

def get_baseline_function_path(robot_name):
    return os.path.join(Settings.baseline_functions_folder, f'baseline_function_{robot_name}.pck')

def get_function_plot_path(robot_name, plot_name):
    return os.path.join(Settings.plot_folder, f'{robot_name}_{plot_name}.png')

def create_folder(folder_path, clear_if_exists=True):
    """
    Create a new folder at the specified path. If it already exists,
    optionally clear its contents.

    Parameters:
        folder_path (str): The path of the folder to create.
        clear_if_exists (bool): If True and the folder exists, its contents will be deleted.
    """
    if os.path.exists(folder_path):
        if clear_if_exists:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        else:
            pass  # Folder exists and we don't clear it
    else:
        os.makedirs(folder_path)
