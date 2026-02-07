#!/usr/bin/env python3

"""
Detailed debugging of the world-frame visualization issue.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from Library import DataProcessor
import matplotlib.pyplot as plt

def debug_world_frame_step_by_step():
    """Debug the world-frame visualization step by step."""
    print("Debugging world-frame visualization step by step...")
    
    # Load session data
    dc = DataProcessor.DataCollection(['session07'])
    for p in dc.processors:
        p.load_profiles(opening_angle=45, steps=9)
        
        print(f"Data loaded: {len(p.profiles)} profiles")
        
        # Check robot positions
        rob_x = p.rob_x.astype(float)
        rob_y = p.rob_y.astype(float)
        rob_yaw = p.rob_yaw_deg.astype(float)
        
        print(f"Robot positions: {len(rob_x)} positions")
        print(f"X range: {rob_x.min():.1f} to {rob_x.max():.1f} mm")
        print(f"Y range: {rob_y.min():.1f} to {rob_y.max():.1f} mm")
        
        # Create debug plot showing what should be visible
        plt.figure(figsize=(12, 12))
        
        # Show all robot positions
        plt.plot(rob_x, rob_y, 'ko', markersize=4, alpha=0.3, label='All robot positions')
        
        # Show first 5 profiles with their positions
        cmap = plt.get_cmap('tab10')
        example_indices = np.arange(5)
        
        profiles_plotted = 0
        
        for k in range(len(example_indices)):
            idx = example_indices[k]
            color = cmap(k % cmap.N)
            az_deg = np.linspace(-22.5, 22.5, 9)
            
            # Get and clip profile
            actual_profile = p.profiles[idx]
            actual_profile = np.clip(actual_profile, a_min=None, a_max=1500.0)
            actual_profile = np.where(actual_profile < 1450.0, actual_profile, np.nan)
            
            # Check if profile has valid data
            if np.all(np.isnan(actual_profile)):
                print(f"Profile {k}: All NaN after clipping - SKIPPING")
                continue
            
            if np.any(~np.isnan(actual_profile)):
                print(f"Profile {k}: Has valid data - PLOTTING")
                profiles_plotted += 1
            else:
                print(f"Profile {k}: All NaN - SKIPPING")
                continue
            
            rob_x_val, rob_y_val, rob_yaw_val = rob_x[idx], rob_y[idx], rob_yaw[idx]
            
            try:
                x_world, y_world = DataProcessor.robot2world(az_deg, actual_profile, rob_x_val, rob_y_val, rob_yaw_val)
                
                # Check if transformation produced valid results
                if np.all(np.isnan(x_world)) or np.all(np.isnan(y_world)):
                    print(f"Profile {k}: Transformation produced all NaN - SKIPPING")
                    continue
                
                plt.plot(x_world, y_world, color=color, linewidth=2, linestyle='-', alpha=0.8)
                plt.plot(rob_x_val, rob_y_val, 'ko', markersize=6, alpha=0.8)
                
            except Exception as e:
                print(f"Profile {k}: Transformation failed: {e}")
                continue
        
        # Set axis limits
        x_min, x_max = rob_x.min() - 500, rob_x.max() + 500
        y_min, y_max = rob_y.min() - 500, rob_y.max() + 500
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.axis('equal')
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title(f"Debug: World-frame profiles (plotted {profiles_plotted} of 5)")
        plt.grid(True, alpha=0.3)
        
        plt.savefig('debug_world_frame_detailed.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Debug plot saved. Profiles plotted: {profiles_plotted}/5")
        
        # Analyze why profiles might not be showing
        if profiles_plotted < 5:
            print(f"\n⚠️  Only {profiles_plotted}/5 profiles were plotted!")
            print("Possible reasons:")
            print("1. Profiles were clipped to all NaN")
            print("2. Transformation produced all NaN")
            print("3. Other errors during plotting")
        
        break

if __name__ == "__main__":
    debug_world_frame_step_by_step()