import os
import pickle
import shutil
from os import path
from Library import Settings


def get_calibration_file(robot_name):
    return os.path.join(Settings.calibration_folder, f'calibration_{robot_name}.pck')

def get_calibration_plot(robot_name, plot_name):
    return os.path.join(Settings.calibration_plot_folder, f'{robot_name}_{plot_name}.png')

def load_calibration(robot_name):
    filename = get_calibration_file(robot_name)
    file_exists = path.isfile(filename)
    if not file_exists: return {}
    with open(filename, 'rb') as f: calibration = pickle.load(f)
    return calibration

def delete_calibration(robot_name):
    filename = get_calibration_file(robot_name)
    file_exists = path.isfile(filename)
    if file_exists: os.remove(filename)

def save_calibration(robot_name, calibration):
    filename = get_calibration_file(robot_name)
    with open(filename, 'wb') as f: pickle.dump(calibration, f)

def create_folder(folder_path, clear_if_exists=True):
    if os.path.exists(folder_path):
        if clear_if_exists:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        else: pass  # Folder exists and we don't clear it
    else: os.makedirs(folder_path)
