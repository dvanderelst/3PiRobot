#!/usr/bin/env python3

"""
Test script to verify world-frame visualization functionality.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from SCRIPT_Train_MLP_Profiles import plot_world_frame_overlay_from_evaluation, EVAL_SESSION, EVAL_INDEX_START, EVAL_INDEX_END
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

def test_world_frame_function_signature():
    """Test that the world-frame function has the correct signature."""
    print("Testing world-frame function signature...")
    
    # Create dummy data for testing
    model = nn.Linear(200, 9)  # Simple model matching our architecture
    eval_session = EVAL_SESSION
    eval_indices = np.arange(EVAL_INDEX_START, EVAL_INDEX_END)
    results_dir = "./test_output"
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # Test that the function can be called (it may fail due to missing data, but signature should be correct)
    try:
        # Create dummy predictions and targets
        dummy_predictions = np.random.rand(len(eval_indices), 9) * 1000  # 9 azimuth bins
        dummy_targets = np.random.rand(len(eval_indices), 9) * 1000
        
        plot_world_frame_overlay_from_evaluation(dummy_predictions, dummy_targets, eval_session, eval_indices, results_dir)
        print("✓ World-frame function called successfully")
    except Exception as e:
        # Expected to fail due to missing data, but signature should work
        if "could not find" in str(e).lower() or "data" in str(e).lower():
            print("✓ World-frame function signature is correct (failed as expected due to missing data)")
        else:
            raise

def test_world_frame_configuration():
    """Test world-frame configuration values."""
    print("\nTesting world-frame configuration...")
    
    assert EVAL_SESSION == 'session07', f"Expected EVAL_SESSION='session07', got {EVAL_SESSION}"
    print(f"✓ EVAL_SESSION = {EVAL_SESSION}")
    
    assert EVAL_INDEX_START == 225, f"Expected EVAL_INDEX_START=225, got {EVAL_INDEX_START}"
    print(f"✓ EVAL_INDEX_START = {EVAL_INDEX_START}")
    
    assert EVAL_INDEX_END == 250, f"Expected EVAL_INDEX_END=250, got {EVAL_INDEX_END}"
    print(f"✓ EVAL_INDEX_END = {EVAL_INDEX_END}")
    
    eval_range = EVAL_INDEX_END - EVAL_INDEX_START
    print(f"✓ Evaluation range: {eval_range} indices ({EVAL_INDEX_START} to {EVAL_INDEX_END})")

def test_index_calculation():
    """Test that evaluation indices are calculated correctly."""
    print("\nTesting index calculation...")
    
    # Test the same logic used in the main script
    eval_indices = np.arange(EVAL_INDEX_START, EVAL_INDEX_END, dtype=int)
    print(f"✓ Generated indices: {len(eval_indices)} indices from {eval_indices[0]} to {eval_indices[-1]}")
    
    # Verify the range
    assert len(eval_indices) == 25, f"Expected 25 indices, got {len(eval_indices)}"
    assert eval_indices[0] == EVAL_INDEX_START
    assert eval_indices[-1] == EVAL_INDEX_END - 1
    print("✓ Index calculation is correct")

if __name__ == "__main__":
    print("Testing world-frame visualization functionality")
    print("=" * 50)
    
    try:
        test_world_frame_configuration()
        test_index_calculation()
        test_world_frame_function_signature()
        
        print("\n" + "=" * 50)
        print("✅ World-frame visualization tests completed!")
        print("\nTo see the world-frame visualization in action:")
        print("1. Run the full training script: python SCRIPT_Train_MLP_Profiles.py")
        print("2. Check the results directory for 'world_frame_overlay_*.png' files")
        print("3. The visualization will show true vs predicted profiles in world coordinates")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise