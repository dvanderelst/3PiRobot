#!/usr/bin/env python3

"""
Diagnostic test to understand world-frame visualization issues.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from Library import DataProcessor
import matplotlib.pyplot as plt

def analyze_world_frame_data():
    """Analyze the world-frame data to understand potential issues."""
    print("Analyzing world-frame data...")
    
    # Load session data
    dc = DataProcessor.DataCollection(['session07'])
    for p in dc.processors:
        p.load_profiles(opening_angle=45, steps=9)
        
        # Load sonar data too
        p.load_sonar(flatten=True)
        
        # Get some sample data
        print(f"Session data analysis:")
        print(f"  Total samples: {len(p.sonar_data)}")
        print(f"  Profile shape: {p.profiles.shape}")
        print(f"  Robot positions: {len(p.rob_x)} samples")
        
        # Check a few samples
        for i in range(3):
            print(f"\nSample {i}:")
            print(f"  Robot position: ({p.rob_x[i]:.1f}, {p.rob_y[i]:.1f}) mm")
            print(f"  Robot yaw: {p.rob_yaw_deg[i]:.1f}°")
            print(f"  Profile distances: {p.profiles[i]}")
            
            # Check if distances are reasonable
            if np.any(p.profiles[i] > 2000):  # More than 2m
                print(f"  ⚠️  Some distances > 2000mm (sonar range limit)")
            if np.any(p.profiles[i] < 100):  # Less than 10cm
                print(f"  ⚠️  Some distances < 100mm (very close)")
        
        # Check coordinate ranges
        print(f"\nCoordinate ranges:")
        print(f"  X range: {p.rob_x.min():.1f} to {p.rob_x.max():.1f} mm")
        print(f"  Y range: {p.rob_y.min():.1f} to {p.rob_y.max():.1f} mm")
        print(f"  Profile distance range: {p.profiles.min():.1f} to {p.profiles.max():.1f} mm")
        
        # Test the robot2world transformation
        print(f"\nTesting robot2world transformation:")
        for i in range(1):
            az_deg = np.linspace(-22.5, 22.5, 9)  # 45° opening angle
            profile = p.profiles[i]
            rob_x, rob_y, rob_yaw = p.rob_x[i], p.rob_y[i], p.rob_yaw_deg[i]
            
            print(f"  Input: az={az_deg}, dist={profile}, pos=({rob_x:.1f},{rob_y:.1f}), yaw={rob_yaw:.1f}")
            
            try:
                x_world, y_world = DataProcessor.robot2world(az_deg, profile, rob_x, rob_y, rob_yaw)
                print(f"  Output: x_world={x_world}, y_world={y_world}")
                print(f"  World coords range: x={x_world.min():.1f}-{x_world.max():.1f}, y={y_world.min():.1f}-{y_world.max():.1f}")
                
                # Check if transformation makes sense
                expected_radius = np.mean(profile)
                actual_radius = np.mean(np.sqrt(x_world**2 + y_world**2))
                print(f"  Expected radius: {expected_radius:.1f}mm, Actual: {actual_radius:.1f}mm")
                
            except Exception as e:
                print(f"  ❌ Transformation failed: {e}")
        
        break

def test_simple_visualization():
    """Create a simple test visualization to debug issues."""
    print("\nCreating simple test visualization...")
    
    # Load session data
    dc = DataProcessor.DataCollection(['session07'])
    for p in dc.processors:
        p.load_profiles(opening_angle=45, steps=9)
        
        # Load sonar data for this test too
        p.load_sonar(flatten=True)
        
        # Create simple visualization of first sample
        plt.figure(figsize=(8, 8))
        
        # Plot robot position
        rob_x, rob_y = p.rob_x[0], p.rob_y[0]
        plt.plot(rob_x, rob_y, 'ro', markersize=10, label='Robot')
        
        # Plot profile in world coordinates
        az_deg = np.linspace(-22.5, 22.5, 9)
        profile = p.profiles[0]
        rob_yaw = p.rob_yaw_deg[0]
        
        try:
            x_world, y_world = DataProcessor.robot2world(az_deg, profile, rob_x, rob_y, rob_yaw)
            plt.plot(x_world, y_world, 'b-', linewidth=2, label='Profile')
            plt.plot([rob_x, x_world[0]], [rob_y, y_world[0]], 'b--', alpha=0.5)
            
            # Add some context
            plt.axis('equal')
            plt.grid(True)
            plt.xlabel('X [mm]')
            plt.ylabel('Y [mm]')
            plt.title('Simple World-Frame Test')
            plt.legend()
            
            plt.savefig('test_simple_world_frame.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✓ Simple visualization saved to test_simple_world_frame.png")
            
        except Exception as e:
            print(f"❌ Simple visualization failed: {e}")
        
        break

if __name__ == "__main__":
    print("Running world-frame diagnostics...")
    print("=" * 50)
    
    try:
        analyze_world_frame_data()
        test_simple_visualization()
        
        print("\n" + "=" * 50)
        print("✅ Diagnostics completed!")
        
    except Exception as e:
        print(f"\n❌ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()