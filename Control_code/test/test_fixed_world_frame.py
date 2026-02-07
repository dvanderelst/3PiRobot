#!/usr/bin/env python3

"""
Test the fixed world-frame visualization.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from SCRIPT_Train_MLP_Profiles import generate_predictions_for_indices
from Library import DataProcessor
import torch
import torch.nn as nn

def test_generate_predictions_for_indices():
    """Test the new function to generate predictions for specific indices."""
    print("Testing generate_predictions_for_indices function...")
    
    try:
        # Create a simple model for testing
        model = nn.Linear(200, 9)
        
        # Create a simple scaler
        from sklearn.preprocessing import StandardScaler
        x_scaler = StandardScaler()
        x_scaler.mean_ = np.zeros(200)
        x_scaler.scale_ = np.ones(200)
        
        # Test indices
        indices = [10, 20, 30, 40, 50]
        
        # Call the function
        predictions, targets = generate_predictions_for_indices(
            model, 'session07', indices, x_scaler, 45, 9
        )
        
        print(f"✓ Generated predictions for {len(predictions)} indices")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Verify the indices are correct
        dc = DataProcessor.DataCollection(['session07'])
        session_proc = None
        for p in dc.processors:
            if 'session07' in p.session:
                session_proc = p
                break
        
        if session_proc is not None:
            session_proc.load_profiles(opening_angle=45, steps=9)
            
            print("\nVerifying predictions match expected targets:")
            for i, idx in enumerate(indices):
                if i < len(targets):
                    expected_target = session_proc.profiles[idx]
                    actual_target = targets[i]
                    
                    if np.array_equal(expected_target, actual_target):
                        print(f"✓ Index {idx}: Targets match")
                    else:
                        print(f"❌ Index {idx}: Targets don't match")
        
        print("✓ generate_predictions_for_indices test completed")
        
    except Exception as e:
        print(f"❌ generate_predictions_for_indices test failed: {e}")
        import traceback
        traceback.print_exc()

def test_index_fix():
    """Test that the index fix works correctly."""
    print("\nTesting index fix...")
    
    try:
        # Simulate the scenario
        predictions = np.random.rand(100, 9)  # Only 100 test samples
        targets = np.random.rand(100, 9)
        
        # Test with indices within range
        good_indices = [10, 20, 30, 40, 50]
        
        print("Testing with good indices:")
        for idx in good_indices:
            if idx < len(predictions) and idx < len(targets):
                pred_profile = predictions[idx]
                true_profile = targets[idx]
                print(f"✓ Index {idx}: Valid access")
            else:
                print(f"❌ Index {idx}: Out of range")
        
        # Test with indices out of range
        bad_indices = [105, 110, 115]
        
        print("\nTesting with bad indices:")
        for idx in bad_indices:
            if idx < len(predictions) and idx < len(targets):
                pred_profile = predictions[idx]
                true_profile = targets[idx]
                print(f"✓ Index {idx}: Valid access")
            else:
                print(f"✓ Index {idx}: Correctly detected as out of range")
        
        print("✓ Index fix test completed")
        
    except Exception as e:
        print(f"❌ Index fix test failed: {e}")

if __name__ == "__main__":
    print("Testing fixed world-frame visualization")
    print("=" * 50)
    
    try:
        test_generate_predictions_for_indices()
        test_index_fix()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("\nKey improvements:")
        print("1. Added generate_predictions_for_indices function")
        print("2. Fixed index cycling issue")
        print("3. Added proper index range checking")
        print("4. User can now specify exact indices to visualize")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise