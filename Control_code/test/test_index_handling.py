#!/usr/bin/env python3

"""
Test index handling in world-frame visualization.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from Library import DataProcessor

def test_index_mismatch():
    """Test for potential index mismatch issues."""
    print("Testing index handling...")
    
    try:
        dc = DataProcessor.DataCollection(['session07'])
        eval_proc = None
        
        for p in dc.processors:
            if 'session07' in p.session:
                eval_proc = p
                break
        
        if eval_proc is None:
            print("❌ Could not find session07 processor")
            return
        
        # Load profiles
        eval_proc.load_profiles(opening_angle=45, steps=9)
        
        print(f"Total profiles: {len(eval_proc.profiles)}")
        print(f"Total robot positions: {len(eval_proc.rob_x)}")
        
        # Simulate the scenario from the main script
        # The main script uses test data, which might have fewer samples than the full session
        
        # Let's say we have 100 test samples but 500 total profiles
        num_test_samples = 100
        predictions = eval_proc.profiles[:num_test_samples]  # Simulate predictions
        targets = eval_proc.profiles[:num_test_samples]      # Simulate targets
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Test the index selection logic
        WORLD_FRAME_PLOT_INDICES = None
        WORLD_FRAME_NUM_EXAMPLES = 15
        
        if WORLD_FRAME_PLOT_INDICES is not None:
            example_indices = np.array(WORLD_FRAME_PLOT_INDICES)
        else:
            example_indices = np.arange(min(WORLD_FRAME_NUM_EXAMPLES, 5))  # Default: first 5
        
        print(f"Example indices: {example_indices}")
        
        # Now test the problematic part: using idx % len(predictions)
        print("\nTesting index cycling:")
        for k in range(len(example_indices)):
            idx = example_indices[k]
            
            # This is what the script does:
            pred_idx = idx % len(predictions)
            target_idx = idx % len(targets)
            
            print(f"Example {k}: idx={idx}, pred_idx={pred_idx}, target_idx={target_idx}")
            
            # Check if we're accessing the right robot position
            rob_x = eval_proc.rob_x[idx]
            rob_y = eval_proc.rob_y[idx]
            
            # But we're using different indices for predictions/targets!
            pred_profile = predictions[pred_idx]
            target_profile = targets[target_idx]
            
            print(f"  Robot position: ({rob_x:.1f}, {rob_y:.1f})")
            print(f"  Prediction from index {pred_idx}: min={pred_profile.min():.1f}, max={pred_profile.max():.1f}")
            print(f"  Target from index {target_idx}: min={target_profile.min():.1f}, max={target_profile.max():.1f}")
            
            # This is the issue: we're using robot position from idx
            # but profiles from pred_idx and target_idx
            # If idx != pred_idx, we're mixing data from different positions!
            
            if idx != pred_idx:
                print(f"  ⚠️  MISMATCH: Using robot position {idx} with profile {pred_idx}")
            
        print("\n✓ Index handling test completed")
        
    except Exception as e:
        print(f"❌ Index handling test failed: {e}")

def test_correct_index_handling():
    """Test the correct way to handle indices."""
    print("\nTesting correct index handling...")
    
    try:
        dc = DataProcessor.DataCollection(['session07'])
        eval_proc = None
        
        for p in dc.processors:
            if 'session07' in p.session:
                eval_proc = p
                break
        
        if eval_proc is None:
            print("❌ Could not find session07 processor")
            return
        
        # Load profiles
        eval_proc.load_profiles(opening_angle=45, steps=9)
        
        # Simulate test data
        num_test_samples = 100
        predictions = eval_proc.profiles[:num_test_samples]
        targets = eval_proc.profiles[:num_test_samples]
        
        # The correct approach: use the same indices for robot positions and profiles
        example_indices = np.arange(5)  # First 5 examples
        
        print("Correct approach - using consistent indices:")
        for k in range(len(example_indices)):
            idx = example_indices[k]
            
            # Use the same index for robot position and profiles
            rob_x = eval_proc.rob_x[idx]
            rob_y = eval_proc.rob_y[idx]
            
            # But we need to make sure idx is within the range of predictions/targets
            if idx < len(predictions) and idx < len(targets):
                pred_profile = predictions[idx]
                target_profile = targets[idx]
                
                print(f"Example {k}: idx={idx}")
                print(f"  Robot position: ({rob_x:.1f}, {rob_y:.1f})")
                print(f"  Prediction: min={pred_profile.min():.1f}, max={pred_profile.max():.1f}")
                print(f"  Target: min={target_profile.min():.1f}, max={target_profile.max():.1f}")
                print(f"  ✓ Consistent indices")
            else:
                print(f"Example {k}: idx={idx} is out of range for predictions/targets")
        
        print("\n✓ Correct index handling test completed")
        
    except Exception as e:
        print(f"❌ Correct index handling test failed: {e}")

if __name__ == "__main__":
    print("Testing index handling in world-frame visualization")
    print("=" * 60)
    
    try:
        test_index_mismatch()
        test_correct_index_handling()
        
        print("\n" + "=" * 60)
        print("✅ Index handling tests completed!")
        print("\nKey findings:")
        print("1. The current script uses idx % len(predictions) which can cause index mismatch")
        print("2. This means robot positions and profiles might come from different samples")
        print("3. The correct approach is to use consistent indices")
        print("4. Need to ensure that the indices used are within the range of test data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise