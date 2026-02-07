#!/usr/bin/env python3

"""
Test the index cycling issue more thoroughly.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from Library import DataProcessor

def test_index_cycling_with_large_indices():
    """Test index cycling with larger indices."""
    print("Testing index cycling with larger indices...")
    
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
        
        # Simulate test data with only 100 samples
        num_test_samples = 100
        predictions = eval_proc.profiles[:num_test_samples]
        targets = eval_proc.profiles[:num_test_samples]
        
        print(f"Total profiles: {len(eval_proc.profiles)}")
        print(f"Test samples: {len(predictions)}")
        
        # Test with specific indices that would cause cycling
        WORLD_FRAME_PLOT_INDICES = [105, 110, 115, 120, 125]  # Larger than test set
        example_indices = np.array(WORLD_FRAME_PLOT_INDICES)
        
        print(f"Example indices: {example_indices}")
        
        print("\nTesting index cycling with large indices:")
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
            
            print(f"  Robot position {idx}: ({rob_x:.1f}, {rob_y:.1f})")
            print(f"  Prediction from index {pred_idx}: min={pred_profile.min():.1f}, max={pred_profile.max():.1f}")
            print(f"  Target from index {target_idx}: min={target_profile.min():.1f}, max={target_profile.max():.1f}")
            
            # This is the issue: we're using robot position from idx
            # but profiles from pred_idx and target_idx
            # If idx != pred_idx, we're mixing data from different positions!
            
            if idx != pred_idx:
                print(f"  ⚠️  MISMATCH: Using robot position {idx} with profile {pred_idx}")
                print(f"  Robot position {pred_idx}: ({eval_proc.rob_x[pred_idx]:.1f}, {eval_proc.rob_y[pred_idx]:.1f})")
        
        print("\n✓ Index cycling test completed")
        
    except Exception as e:
        print(f"❌ Index cycling test failed: {e}")

def test_solution():
    """Test a solution to the index cycling issue."""
    print("\nTesting solution to index cycling issue...")
    
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
        
        # Solution: Only use indices that are within the range of test data
        WORLD_FRAME_PLOT_INDICES = [105, 110, 115, 120, 125]
        example_indices = np.array(WORLD_FRAME_PLOT_INDICES)
        
        # Filter to only use indices that are within the test data range
        valid_indices = [idx for idx in example_indices if idx < len(predictions) and idx < len(targets)]
        
        print(f"Original indices: {example_indices}")
        print(f"Valid indices: {valid_indices}")
        
        if len(valid_indices) == 0:
            print("No valid indices found - using first few instead")
            valid_indices = list(range(min(5, len(predictions))))
        
        print(f"\nUsing indices: {valid_indices}")
        
        for k in range(len(valid_indices)):
            idx = valid_indices[k]
            
            # Use consistent indices
            rob_x = eval_proc.rob_x[idx]
            rob_y = eval_proc.rob_y[idx]
            pred_profile = predictions[idx]
            target_profile = targets[idx]
            
            print(f"Example {k}: idx={idx}")
            print(f"  Robot position: ({rob_x:.1f}, {rob_y:.1f})")
            print(f"  Prediction: min={pred_profile.min():.1f}, max={pred_profile.max():.1f}")
            print(f"  Target: min={target_profile.min():.1f}, max={target_profile.max():.1f}")
            print(f"  ✓ Consistent indices")
        
        print("\n✓ Solution test completed")
        
    except Exception as e:
        print(f"❌ Solution test failed: {e}")

if __name__ == "__main__":
    print("Testing index cycling issue")
    print("=" * 40)
    
    try:
        test_index_cycling_with_large_indices()
        test_solution()
        
        print("\n" + "=" * 40)
        print("✅ Index cycling tests completed!")
        print("\nKey findings:")
        print("1. The current script uses idx % len(predictions) which causes index cycling")
        print("2. This can lead to mismatched robot positions and profiles")
        print("3. The solution is to only use indices within the test data range")
        print("4. Or provide specific indices that are known to be valid")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise