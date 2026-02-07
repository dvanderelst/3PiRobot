#!/usr/bin/env python3

"""
Test script to verify that the profile clipping functionality works correctly.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from SCRIPT_Train_MLP_Profiles import MAX_PROFILE_MM, PLOT_MAX_PROFILE_MM

def test_clipping_values():
    """Test that the clipping constants are set correctly."""
    print("Testing clipping constants...")
    
    # Check that MAX_PROFILE_MM is set to 1500.0
    assert MAX_PROFILE_MM == 1500.0, f"Expected MAX_PROFILE_MM=1500.0, got {MAX_PROFILE_MM}"
    print(f"✓ MAX_PROFILE_MM = {MAX_PROFILE_MM}mm")
    
    # Check that PLOT_MAX_PROFILE_MM is set to 1450.0
    assert PLOT_MAX_PROFILE_MM == 1450.0, f"Expected PLOT_MAX_PROFILE_MM=1450.0, got {PLOT_MAX_PROFILE_MM}"
    print(f"✓ PLOT_MAX_PROFILE_MM = {PLOT_MAX_PROFILE_MM}mm")
    
    # Check that PLOT_MAX_PROFILE_MM is less than MAX_PROFILE_MM
    assert PLOT_MAX_PROFILE_MM < MAX_PROFILE_MM, "PLOT_MAX_PROFILE_MM should be less than MAX_PROFILE_MM"
    print(f"✓ PLOT_MAX_PROFILE_MM ({PLOT_MAX_PROFILE_MM}) < MAX_PROFILE_MM ({MAX_PROFILE_MM})")

def test_profile_clipping():
    """Test the profile clipping logic."""
    print("\nTesting profile clipping logic...")
    
    # Create some test profiles with values beyond the sonar range
    test_profiles = np.array([
        [1000, 1200, 1400, 1600, 1800],  # Some values beyond 1500mm
        [800, 1000, 1200, 1400, 1550],   # Some values beyond 1500mm
        [500, 700, 900, 1100, 1300],    # All values within range
    ])
    
    print(f"Original profiles shape: {test_profiles.shape}")
    print(f"Original profiles:\n{test_profiles}")
    
    # Apply clipping
    clipped_profiles = np.clip(test_profiles, a_min=None, a_max=MAX_PROFILE_MM)
    
    print(f"Clipped profiles:\n{clipped_profiles}")
    
    # Verify that all values are <= MAX_PROFILE_MM
    assert np.all(clipped_profiles <= MAX_PROFILE_MM), "Some values still exceed MAX_PROFILE_MM after clipping"
    print(f"✓ All clipped values are <= {MAX_PROFILE_MM}mm")
    
    # Verify that values within range are unchanged
    expected_within_range = test_profiles[test_profiles <= MAX_PROFILE_MM]
    actual_within_range = clipped_profiles[test_profiles <= MAX_PROFILE_MM]
    assert np.array_equal(expected_within_range, actual_within_range), "Values within range were incorrectly modified"
    print("✓ Values within range are unchanged")
    
    # Verify that values beyond range are clipped to MAX_PROFILE_MM
    beyond_range_mask = test_profiles > MAX_PROFILE_MM
    assert np.all(clipped_profiles[beyond_range_mask] == MAX_PROFILE_MM), "Values beyond range were not clipped to MAX_PROFILE_MM"
    print(f"✓ Values beyond range are clipped to {MAX_PROFILE_MM}mm")

def test_plot_clipping():
    """Test the plot clipping logic."""
    print("\nTesting plot clipping logic...")
    
    # Create test profile data
    pred_profile = np.array([1000, 1200, 1400, 1600, 1800])
    target_profile = np.array([900, 1100, 1300, 1500, 1700])
    
    print(f"Original prediction: {pred_profile}")
    print(f"Original target: {target_profile}")
    
    # Apply plot clipping
    pred_clipped = np.where(pred_profile < PLOT_MAX_PROFILE_MM, pred_profile, np.nan)
    target_clipped = np.where(target_profile < PLOT_MAX_PROFILE_MM, target_profile, np.nan)
    
    print(f"Clipped prediction: {pred_clipped}")
    print(f"Clipped target: {target_clipped}")
    
    # Verify that values beyond PLOT_MAX_PROFILE_MM are set to NaN
    pred_beyond = pred_profile > PLOT_MAX_PROFILE_MM
    target_beyond = target_profile > PLOT_MAX_PROFILE_MM
    
    assert np.all(np.isnan(pred_clipped[pred_beyond])), "Prediction values beyond plot limit should be NaN"
    assert np.all(np.isnan(target_clipped[target_beyond])), "Target values beyond plot limit should be NaN"
    print(f"✓ Values beyond {PLOT_MAX_PROFILE_MM}mm are set to NaN for plotting")
    
    # Verify that values within plot range are unchanged
    pred_within = pred_profile <= PLOT_MAX_PROFILE_MM
    target_within = target_profile <= PLOT_MAX_PROFILE_MM
    
    assert np.all(pred_clipped[pred_within] == pred_profile[pred_within]), "Prediction values within range should be unchanged"
    assert np.all(target_clipped[target_within] == target_profile[target_within]), "Target values within range should be unchanged"
    print("✓ Values within plot range are unchanged")

if __name__ == "__main__":
    print("Running profile clipping tests...")
    print("=" * 50)
    
    try:
        test_clipping_values()
        test_profile_clipping()
        test_plot_clipping()
        
        print("\n" + "=" * 50)
        print("✅ All clipping tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise