#!/usr/bin/env python3

"""
Test script to identify and debug world-frame visualization issues.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from Library import DataProcessor
import matplotlib.pyplot as plt

def test_robot2world_transformation():
    """Test the robot2world transformation with known values."""
    print("Testing robot2world transformation...")
    
    # Test case 1: Simple forward direction (0° azimuth)
    az_deg = 0
    dist = 1000  # 1 meter
    rob_x, rob_y, rob_yaw_deg = 0, 0, 0  # Robot at origin, facing right (0°)
    
    x_world, y_world = DataProcessor.robot2world(az_deg, dist, rob_x, rob_y, rob_yaw_deg)
    
    print(f"Test 1 - Forward (0° azimuth, 0° yaw):")
    print(f"  Expected: x={dist}, y=0")
    print(f"  Got: x={x_world:.1f}, y={y_world:.1f}")
    
    # Test case 2: 90° azimuth (up/left in robot frame)
    az_deg = 90
    dist = 1000
    
    x_world, y_world = DataProcessor.robot2world(az_deg, dist, rob_x, rob_y, rob_yaw_deg)
    
    print(f"Test 2 - Up/Left (90° azimuth, 0° yaw):")
    print(f"  Expected: x=0, y={dist}")
    print(f"  Got: x={x_world:.1f}, y={y_world:.1f}")
    
    # Test case 3: Robot facing 90° (up/north), forward direction
    az_deg = 0
    dist = 1000
    rob_yaw_deg = 90
    
    x_world, y_world = DataProcessor.robot2world(az_deg, dist, rob_x, rob_y, rob_yaw_deg)
    
    print(f"Test 3 - Forward with 90° yaw:")
    print(f"  Expected: x=0, y={dist}")
    print(f"  Got: x={x_world:.1f}, y={y_world:.1f}")
    
    print("✓ Transformation tests completed")

def test_profile_clipping():
    """Test profile clipping behavior."""
    print("\nTesting profile clipping...")
    
    # Create a sample profile with some extreme values
    profile = np.array([500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
    
    print(f"Original profile: {profile}")
    
    # Test clipping at 1450
    clipped_profile = np.where(profile > 1450.0, np.nan, profile)
    print(f"Clipped at 1450: {clipped_profile}")
    
    # Test clipping at 3000
    clipped_profile = np.where(profile > 3000.0, np.nan, profile)
    print(f"Clipped at 3000: {clipped_profile}")
    
    # Test with extreme values
    extreme_profile = np.array([500, 600, 700, 800, 900, 1000, 1100, 1200, 4000])
    clipped_profile = np.where(extreme_profile > 3000.0, np.nan, extreme_profile)
    print(f"Extreme profile clipped at 3000: {clipped_profile}")
    
    print("✓ Clipping tests completed")

def test_index_selection():
    """Test index selection logic."""
    print("\nTesting index selection...")
    
    # Test the logic from the main script
    WORLD_FRAME_PLOT_INDICES = None
    WORLD_FRAME_NUM_EXAMPLES = 15
    
    if WORLD_FRAME_PLOT_INDICES is not None:
        example_indices = np.array(WORLD_FRAME_PLOT_INDICES)
    else:
        example_indices = np.arange(min(WORLD_FRAME_NUM_EXAMPLES, 5))  # Default: first 5
    
    print(f"Example indices: {example_indices}")
    print(f"Number of indices: {len(example_indices)}")
    
    # Test with specific indices
    WORLD_FRAME_PLOT_INDICES = [10, 20, 30, 40, 50]
    example_indices = np.array(WORLD_FRAME_PLOT_INDICES)
    print(f"Specific indices: {example_indices}")
    
    print("✓ Index selection tests completed")

def test_data_loading():
    """Test data loading from session."""
    print("\nTesting data loading...")
    
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
        
        print(f"Loaded {len(eval_proc.profiles)} profiles")
        print(f"Profile shape: {eval_proc.profiles.shape}")
        
        # Check robot positions
        if hasattr(eval_proc, 'rob_x') and eval_proc.rob_x is not None:
            print(f"Robot positions: {len(eval_proc.rob_x)} positions")
            print(f"X range: {eval_proc.rob_x.min():.1f} to {eval_proc.rob_x.max():.1f} mm")
            print(f"Y range: {eval_proc.rob_y.min():.1f} to {eval_proc.rob_y.max():.1f} mm")
        else:
            print("❌ No robot position data available")
        
        # Check a few sample profiles
        print("\nSample profiles (first 5):")
        for i in range(5):
            profile = eval_proc.profiles[i]
            print(f"Profile {i}: min={profile.min():.1f}, max={profile.max():.1f}, mean={profile.mean():.1f}")
        
        print("✓ Data loading tests completed")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")

def test_world_frame_plotting():
    """Test the actual world-frame plotting."""
    print("\nTesting world-frame plotting...")
    
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
        
        # Create dummy predictions and targets
        num_examples = 5
        predictions = np.random.rand(num_examples, 9) * 1000
        targets = np.random.rand(num_examples, 9) * 1000
        
        # Test plotting
        plt.figure(figsize=(12, 10))
        cmap = plt.get_cmap('tab10')
        
        profiles_plotted = 0
        example_indices = np.arange(num_examples)
        
        for k in range(len(example_indices)):
            idx = example_indices[k]
            color = cmap(k % cmap.N)
            az_deg = np.linspace(-22.5, 22.5, 9)
            
            rob_x = eval_proc.rob_x[idx]
            rob_y = eval_proc.rob_y[idx]
            rob_yaw = eval_proc.rob_yaw_deg[idx]
            
            # Get profiles
            pred_profile = predictions[k]
            true_profile = targets[k]
            
            # Apply clipping
            pred_profile = np.where(pred_profile > 3000.0, np.nan, pred_profile)
            true_profile = np.where(true_profile > 3000.0, np.nan, true_profile)
            
            # Check if profiles have valid data
            if np.all(np.isnan(pred_profile)) or np.all(np.isnan(true_profile)):
                print(f"Profile {k}: All NaN after clipping - SKIPPING")
                continue
            
            # Transform to world coordinates
            try:
                x_pred, y_pred = DataProcessor.robot2world(az_deg, pred_profile, rob_x, rob_y, rob_yaw)
                x_true, y_true = DataProcessor.robot2world(az_deg, true_profile, rob_x, rob_y, rob_yaw)
                
                # Check if transformation produced valid results
                if np.all(np.isnan(x_true)) or np.all(np.isnan(y_true)):
                    print(f"Profile {k}: Transformation produced all NaN - SKIPPING")
                    continue
                
                plt.plot(x_true, y_true, color=color, linewidth=2, linestyle='-', alpha=0.8)
                plt.plot(x_pred, y_pred, color=color, linewidth=2, linestyle='--', alpha=0.8)
                plt.plot(rob_x, rob_y, 'ko', markersize=6, alpha=0.8)
                profiles_plotted += 1
                
            except Exception as e:
                print(f"Profile {k}: Transformation failed: {e}")
                continue
        
        plt.axis('equal')
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title("World-frame test plotting")
        plt.grid(True, alpha=0.3)
        
        plt.savefig('test_world_frame_plotting.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ World-frame plotting test completed. Plotted {profiles_plotted}/{num_examples} profiles")
        
    except Exception as e:
        print(f"❌ World-frame plotting failed: {e}")

if __name__ == "__main__":
    print("Testing world-frame visualization issues")
    print("=" * 50)
    
    try:
        test_robot2world_transformation()
        test_profile_clipping()
        test_index_selection()
        test_data_loading()
        test_world_frame_plotting()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise