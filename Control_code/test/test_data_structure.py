#!/usr/bin/env python3

"""
Test script to examine the data structure of the robot data files.
"""

import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

import dill
import numpy as np
from Library.DataStorage import DataReader

def examine_data_structure():
    # Create a data reader for session03
    data_reader = DataReader('session03')
    
    # Get all filenames
    filenames = data_reader.get_all_filenames()
    print(f"Found {len(filenames)} data files in session03")
    
    # Load the first file
    first_file = filenames[0]
    print(f"\nExamining first file: {first_file}")
    
    data = data_reader.load_data(first_file)
    
    print("\nTop-level keys in data:")
    for key in data.keys():
        print(f"  - {key}")
    
    # Examine sonar_package
    if 'sonar_package' in data:
        print("\nSonar package structure:")
        sonar_package = data['sonar_package']
        for key in sonar_package.keys():
            value = sonar_package[key]
            if isinstance(value, np.ndarray):
                print(f"  - {key}: numpy array with shape {value.shape}")
            else:
                print(f"  - {key}: {type(value).__name__}")
    
    # Examine position
    if 'position' in data:
        print("\nPosition structure:")
        position = data['position']
        for key in position.keys():
            value = position[key]
            print(f"  - {key}: {value}")
    
    # Examine motion
    if 'motion' in data:
        print("\nMotion structure:")
        motion = data['motion']
        for key in motion.keys():
            value = motion[key]
            print(f"  - {key}: {value}")
    
    # Get some statistics
    print(f"\nData overview:")
    print(f"  Total files: {len(filenames)}")
    
    # Try to get field data
    try:
        sonar_data = data_reader.get_field('sonar_package', 'sonar_data')
        print(f"  Sonar data samples: {len(sonar_data)}")
        if len(sonar_data) > 0:
            print(f"  Sonar data shape: {sonar_data[0].shape}")
    except Exception as e:
        print(f"  Error getting sonar data: {e}")
    
    try:
        positions = data_reader.get_field('position', 'x')
        print(f"  Position samples: {len(positions)}")
    except Exception as e:
        print(f"  Error getting positions: {e}")

if __name__ == "__main__":
    examine_data_structure()