#!/usr/bin/env python3

"""
Test creating an environment outline by plotting multiple profiles.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from Library import DataProcessor
import matplotlib.pyplot as plt

def test_environment_outline():
    """Test creating environment outline by plotting multiple profiles."""
    print("Testing environment outline creation...")
    
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
        print(f"Robot positions: {len(eval_proc.rob_x)}")
        
        # Create plot showing multiple profiles to form environment outline
        plt.figure(figsize=(14, 12))
        
        # Plot all robot positions
        plt.plot(eval_proc.rob_x, eval_proc.rob_y, 'k.', markersize=3, alpha=0.3, label='Robot path')
        
        # Plot profiles from multiple positions to see environment outline
        num_profiles_to_plot = 20
        step = len(eval_proc.profiles) // num_profiles_to_plot
        indices = range(0, len(eval_proc.profiles), step)
        
        print(f"Plotting {len(indices)} profiles to form environment outline...")
        
        cmap = plt.get_cmap('viridis')
        
        for i, idx in enumerate(indices):
            if idx >= len(eval_proc.profiles):
                break
                
            rob_x = eval_proc.rob_x[idx]
            rob_y = eval_proc.rob_y[idx]
            rob_yaw = eval_proc.rob_yaw_deg[idx]
            profile = eval_proc.profiles[idx]
            
            # Create azimuth angles (-22.5° to +22.5° for 45° opening angle)
            az_deg = np.linspace(-22.5, 22.5, 9)
            
            # Transform profile to world coordinates
            x_world, y_world = DataProcessor.robot2world(az_deg, profile, rob_x, rob_y, rob_yaw)
            
            # Plot profile
            color = cmap(i / len(indices))
            plt.plot(x_world, y_world, color=color, linewidth=1, alpha=0.7)
            
            # Mark robot position
            plt.plot(rob_x, rob_y, 'ko', markersize=4, alpha=0.8)
        
        plt.axis('equal')
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title(f"Environment outline from {len(indices)} profiles")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('test_environment_outline.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Environment outline test completed")
        print("  Saved visualization to test_environment_outline.png")
        
    except Exception as e:
        print(f"❌ Environment outline test failed: {e}")
        import traceback
        traceback.print_exc()

def test_profile_overlay():
    """Test overlaying multiple profiles to see if they form environment outline."""
    print("\nTesting profile overlay...")
    
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
        
        # Create plot showing profile overlay
        plt.figure(figsize=(14, 12))
        
        # Plot robot trajectory
        plt.plot(eval_proc.rob_x, eval_proc.rob_y, 'r-', linewidth=2, alpha=0.5, label='Robot trajectory')
        
        # Plot profiles from consecutive positions to see overlap
        start_idx = 100
        num_consecutive = 10
        
        print(f"Plotting {num_consecutive} consecutive profiles starting from index {start_idx}...")
        
        for i in range(num_consecutive):
            idx = start_idx + i
            if idx >= len(eval_proc.profiles):
                break
                
            rob_x = eval_proc.rob_x[idx]
            rob_y = eval_proc.rob_y[idx]
            rob_yaw = eval_proc.rob_yaw_deg[idx]
            profile = eval_proc.profiles[idx]
            
            # Create azimuth angles
            az_deg = np.linspace(-22.5, 22.5, 9)
            
            # Transform profile to world coordinates
            x_world, y_world = DataProcessor.robot2world(az_deg, profile, rob_x, rob_y, rob_yaw)
            
            # Plot profile
            plt.plot(x_world, y_world, 'b-', linewidth=1, alpha=0.3)
            
            # Mark robot position
            plt.plot(rob_x, rob_y, 'bo', markersize=6)
            plt.text(rob_x, rob_y, str(idx), fontsize=8)
        
        plt.axis('equal')
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title(f"Profile overlay - consecutive positions {start_idx} to {start_idx + num_consecutive - 1}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('test_profile_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Profile overlay test completed")
        print("  Saved visualization to test_profile_overlay.png")
        
    except Exception as e:
        print(f"❌ Profile overlay test failed: {e}")

def test_dense_profile_plotting():
    """Test plotting many profiles densely to see environment outline."""
    print("\nTesting dense profile plotting...")
    
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
        
        # Create plot with many profiles
        plt.figure(figsize=(14, 12))
        
        # Plot robot trajectory
        plt.plot(eval_proc.rob_x, eval_proc.rob_y, 'k-', linewidth=1, alpha=0.3, label='Robot trajectory')
        
        # Plot many profiles to see if they form environment outline
        num_profiles = min(100, len(eval_proc.profiles))  # Plot first 100 profiles
        step = max(1, len(eval_proc.profiles) // num_profiles)
        
        print(f"Plotting {num_profiles} profiles densely...")
        
        for i in range(num_profiles):
            idx = i * step
            if idx >= len(eval_proc.profiles):
                break
                
            rob_x = eval_proc.rob_x[idx]
            rob_y = eval_proc.rob_y[idx]
            rob_yaw = eval_proc.rob_yaw_deg[idx]
            profile = eval_proc.profiles[idx]
            
            # Create azimuth angles
            az_deg = np.linspace(-22.5, 22.5, 9)
            
            # Transform profile to world coordinates
            x_world, y_world = DataProcessor.robot2world(az_deg, profile, rob_x, rob_y, rob_yaw)
            
            # Plot profile
            plt.plot(x_world, y_world, 'blue', linewidth=0.5, alpha=0.1)
        
        plt.axis('equal')
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title(f"Dense profile plotting - {num_profiles} profiles")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('test_dense_profiles.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Dense profile plotting test completed")
        print("  Saved visualization to test_dense_profiles.png")
        
    except Exception as e:
        print(f"❌ Dense profile plotting test failed: {e}")

if __name__ == "__main__":
    print("Testing environment outline creation")
    print("=" * 50)
    
    try:
        test_environment_outline()
        test_profile_overlay()
        test_dense_profile_plotting()
        
        print("\n" + "=" * 50)
        print("✅ All environment outline tests completed!")
        print("\nKey insights:")
        print("1. Multiple profiles from different positions should form environment outline")
        print("2. The coordinate transformations are working correctly")
        print("3. Check the generated plots to see if profiles overlap correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise