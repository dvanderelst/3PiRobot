#!/usr/bin/env python3

"""
Debug data loading issue.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from Library import DataProcessor

def debug_data_loading():
    """Debug data loading."""
    print("Debugging data loading...")
    
    try:
        dc = DataProcessor.DataCollection(['session07'])
        session_proc = None
        
        for p in dc.processors:
            if 'session07' in p.session:
                session_proc = p
                break
        
        if session_proc is None:
            print("❌ Could not find session07 processor")
            return
        
        print(f"Found session processor: {session_proc.session}")
        
        # Load sonar data
        print("Loading sonar data...")
        sonar_data = session_proc.load_sonar(flatten=True)
        print(f"Sonar data type: {type(sonar_data)}")
        print(f"Sonar data: {sonar_data}")
        
        # Load profiles
        print("Loading profiles...")
        session_proc.load_profiles(opening_angle=45, steps=9)
        profiles = session_proc.profiles
        print(f"Profiles type: {type(profiles)}")
        print(f"Profiles shape: {profiles.shape if profiles is not None else 'None'}")
        
        print("✓ Data loading debug completed")
        
    except Exception as e:
        print(f"❌ Data loading debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_loading()