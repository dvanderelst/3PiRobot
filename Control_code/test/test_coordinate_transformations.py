#!/usr/bin/env python3

"""
Test coordinate transformations to verify they work correctly.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from Library import DataProcessor
import matplotlib.pyplot as plt

def test_robot2world_basic():
    """Test basic robot2world transformation."""
    print("Testing basic robot2world transformation...")
    
    # Test case 1: Robot at origin, facing right (0° yaw), forward direction (0° azimuth)
    az_deg = 0
    dist = 1000  # 1 meter
    rob_x, rob_y, rob_yaw_deg = 0, 0, 0
    
    x_world, y_world = DataProcessor.robot2world(az_deg, dist, rob_x, rob_y, rob_yaw_deg)
    
    print(f"Test 1 - Forward (0° azimuth, 0° yaw):")
    print(f"  Expected: x={dist}, y=0")
    print(f"  Got: x={x_world:.1f}, y={y_world:.1f}")
    print(f"  Match: {np.isclose(x_world, dist) and np.isclose(y_world, 0)}")
    
    # Test case 2: Robot at origin, facing right (0° yaw), 90° azimuth (up/left)
    az_deg = 90
    dist = 1000
    
    x_world, y_world = DataProcessor.robot2world(az_deg, dist, rob_x, rob_y, rob_yaw_deg)
    
    print(f"Test 2 - Up/Left (90° azimuth, 0° yaw):")
    print(f"  Expected: x=0, y={dist}")
    print(f"  Got: x={x_world:.1f}, y={y_world:.1f}")
    print(f"  Match: {np.isclose(x_world, 0) and np.isclose(y_world, dist)}")
    
    # Test case 3: Robot at origin, facing up (90° yaw), forward direction (0° azimuth)
    az_deg = 0
    dist = 1000
    rob_yaw_deg = 90
    
    x_world, y_world = DataProcessor.robot2world(az_deg, dist, rob_x, rob_y, rob_yaw_deg)
    
    print(f"Test 3 - Forward with 90° yaw:")
    print(f"  Expected: x=0, y={dist}")
    print(f"  Got: x={x_world:.1f}, y={y_world:.1f}")
    print(f"  Match: {np.isclose(x_world, 0) and np.isclose(y_world, dist)}")
    
    print("✓ Basic robot2world tests completed")

def test_world2robot_basic():
    """Test basic world2robot transformation."""
    print("\nTesting basic world2robot transformation...")
    
    # Test case 1: Robot at origin, facing right (0° yaw), point at (1000, 0)
    x_coords = 1000
    y_coords = 0
    rob_x, rob_y, rob_yaw_deg = 0, 0, 0
    
    x_rel, y_rel = DataProcessor.world2robot(x_coords, y_coords, rob_x, rob_y, rob_yaw_deg)
    
    print(f"Test 1 - Point (1000, 0) with 0° yaw:")
    print(f"  Expected: x_rel={x_coords}, y_rel=0")
    print(f"  Got: x_rel={x_rel:.1f}, y_rel={y_rel:.1f}")
    print(f"  Match: {np.isclose(x_rel, x_coords) and np.isclose(y_rel, 0)}")
    
    # Test case 2: Robot at origin, facing right (0° yaw), point at (0, 1000)
    x_coords = 0
    y_coords = 1000
    
    x_rel, y_rel = DataProcessor.world2robot(x_coords, y_coords, rob_x, rob_y, rob_yaw_deg)
    
    print(f"Test 2 - Point (0, 1000) with 0° yaw:")
    print(f"  Expected: x_rel=0, y_rel={y_coords}")
    print(f"  Got: x_rel={x_rel:.1f}, y_rel={y_rel:.1f}")
    print(f"  Match: {np.isclose(x_rel, 0) and np.isclose(y_rel, y_coords)}")
    
    print("✓ Basic world2robot tests completed")

def test_round_trip():
    """Test that robot2world and world2robot are inverses."""
    print("\nTesting round-trip transformations...")
    
    # Test with robot at origin, facing right
    rob_x, rob_y, rob_yaw_deg = 0, 0, 0
    
    # Test points at various azimuths
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
    distances = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    
    for az, dist in zip(azimuths, distances):
        # Convert to world coordinates
        x_world, y_world = DataProcessor.robot2world(az, dist, rob_x, rob_y, rob_yaw_deg)
        
        # Convert back to robot coordinates
        x_rel, y_rel = DataProcessor.world2robot(x_world, y_world, rob_x, rob_y, rob_yaw_deg)
        
        # Calculate expected robot coordinates
        az_rad = np.deg2rad(az)
        expected_x_rel = dist * np.cos(az_rad)
        expected_y_rel = dist * np.sin(az_rad)
        
        print(f"Azimuth {az}°:")
        print(f"  Original: x_rel={expected_x_rel:.1f}, y_rel={expected_y_rel:.1f}")
        print(f"  Round-trip: x_rel={x_rel:.1f}, y_rel={y_rel:.1f}")
        print(f"  Match: {np.isclose(x_rel, expected_x_rel) and np.isclose(y_rel, expected_y_rel)}")
    
    print("✓ Round-trip tests completed")

def test_with_real_robot_positions():
    """Test transformations with real robot positions from session."""
    print("\nTesting with real robot positions...")
    
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
        
        # Load profiles to get robot positions
        eval_proc.load_profiles(opening_angle=45, steps=9)
        
        # Test a few robot positions
        test_indices = [0, 50, 100, 150, 200]
        
        for idx in test_indices:
            rob_x = eval_proc.rob_x[idx]
            rob_y = eval_proc.rob_y[idx]
            rob_yaw = eval_proc.rob_yaw_deg[idx]
            
            print(f"\nRobot position {idx}:")
            print(f"  Position: ({rob_x:.1f}, {rob_y:.1f})")
            print(f"  Yaw: {rob_yaw:.1f}°")
            
            # Test forward direction
            az_deg = 0
            dist = 500
            
            x_world, y_world = DataProcessor.robot2world(az_deg, dist, rob_x, rob_y, rob_yaw)
            
            print(f"  Forward point: ({x_world:.1f}, {y_world:.1f})")
            
            # Test round-trip
            x_rel, y_rel = DataProcessor.world2robot(x_world, y_world, rob_x, rob_y, rob_yaw)
            
            expected_x_rel = dist * np.cos(np.deg2rad(az_deg))
            expected_y_rel = dist * np.sin(np.deg2rad(az_deg))
            
            print(f"  Round-trip match: {np.isclose(x_rel, expected_x_rel) and np.isclose(y_rel, expected_y_rel)}")
        
        print("✓ Real robot position tests completed")
        
    except Exception as e:
        print(f"❌ Real robot position tests failed: {e}")

def test_profile_transformation():
    """Test transformation of complete profiles."""
    print("\nTesting complete profile transformations...")
    
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
        
        # Test first profile
        idx = 0
        rob_x = eval_proc.rob_x[idx]
        rob_y = eval_proc.rob_y[idx]
        rob_yaw = eval_proc.rob_yaw_deg[idx]
        profile = eval_proc.profiles[idx]
        
        print(f"Testing profile {idx}:")
        print(f"  Robot position: ({rob_x:.1f}, {rob_y:.1f})")
        print(f"  Robot yaw: {rob_yaw:.1f}°")
        print(f"  Profile: {profile}")
        
        # Create azimuth angles
        az_deg = np.linspace(-22.5, 22.5, 9)
        
        # Transform profile to world coordinates
        x_world, y_world = DataProcessor.robot2world(az_deg, profile, rob_x, rob_y, rob_yaw)
        
        print(f"  World coordinates shape: {x_world.shape}")
        print(f"  X range: {x_world.min():.1f} to {x_world.max():.1f}")
        print(f"  Y range: {y_world.min():.1f} to {y_world.max():.1f}")
        
        # Test round-trip for each point
        all_match = True
        for i in range(len(az_deg)):
            x_rel, y_rel = DataProcessor.world2robot(x_world[i], y_world[i], rob_x, rob_y, rob_yaw)
            
            expected_x_rel = profile[i] * np.cos(np.deg2rad(az_deg[i]))
            expected_y_rel = profile[i] * np.sin(np.deg2rad(az_deg[i]))
            
            if not (np.isclose(x_rel, expected_x_rel) and np.isclose(y_rel, expected_y_rel)):
                all_match = False
                break
        
        print(f"  Round-trip match: {all_match}")
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Plot robot position
        plt.plot(rob_x, rob_y, 'ro', markersize=10, label='Robot position')
        
        # Plot profile
        plt.plot(x_world, y_world, 'b-', linewidth=2, label='Profile')
        
        # Plot individual points
        for i in range(len(az_deg)):
            plt.plot(x_world[i], y_world[i], 'bo', markersize=6)
            plt.text(x_world[i], y_world[i], f'{az_deg[i]:.0f}°', fontsize=8)
        
        plt.axis('equal')
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title(f"Profile {idx} transformation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('test_profile_transformation.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  Saved visualization to test_profile_transformation.png")
        
        print("✓ Profile transformation tests completed")
        
    except Exception as e:
        print(f"❌ Profile transformation tests failed: {e}")

if __name__ == "__main__":
    print("Testing coordinate transformations")
    print("=" * 50)
    
    try:
        test_robot2world_basic()
        test_world2robot_basic()
        test_round_trip()
        test_with_real_robot_positions()
        test_profile_transformation()
        
        print("\n" + "=" * 50)
        print("✅ All coordinate transformation tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise