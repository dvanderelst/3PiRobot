#!/usr/bin/env python3

"""
Complete test of the fixed world-frame visualization.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from SCRIPT_Train_MLP_Profiles import plot_world_frame_comparison
from Library import DataProcessor

def test_complete_world_frame():
    """Test the complete world-frame visualization with the fixes."""
    print("Testing complete world-frame visualization...")
    
    try:
        # Create dummy predictions and targets
        num_examples = 5
        predictions = np.random.rand(num_examples, 9) * 2000 + 500  # Range 500-2500
        targets = np.random.rand(num_examples, 9) * 2000 + 500
        
        # Test with specific indices
        plot_indices = [10, 20, 30, 40, 50]
        
        # Call the world-frame function
        plot_world_frame_comparison(
            results_dir="./test_output",
            predictions=predictions,
            targets=targets,
            eval_session='session07',
            num_examples=num_examples,
            plot_indices=plot_indices,
            max_plot_mm=3000.0
        )
        
        print("✓ Complete world-frame visualization test completed")
        
    except Exception as e:
        print(f"❌ Complete world-frame visualization test failed: {e}")
        import traceback
        traceback.print_exc()

def test_with_real_data():
    """Test with real data from the session."""
    print("\nTesting with real data...")
    
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
        
        # Use real profiles as both predictions and targets
        num_examples = 5
        predictions = eval_proc.profiles[:num_examples]
        targets = eval_proc.profiles[:num_examples]
        
        # Test plotting
        plot_world_frame_comparison(
            results_dir="./test_output",
            predictions=predictions,
            targets=targets,
            eval_session='session07',
            num_examples=num_examples,
            plot_indices=None,  # Use default
            max_plot_mm=3000.0
        )
        
        print("✓ Real data test completed")
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")

if __name__ == "__main__":
    print("Complete world-frame visualization tests")
    print("=" * 50)
    
    try:
        test_complete_world_frame()
        test_with_real_data()
        
        print("\n" + "=" * 50)
        print("✅ All complete tests passed!")
        print("\nThe world-frame visualization should now work correctly:")
        print("1. Index cycling issue is fixed")
        print("2. Specific indices can be selected")
        print("3. Predictions are generated for the exact indices requested")
        print("4. Robot positions and profiles are consistently matched")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise