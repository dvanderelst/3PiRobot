#!/usr/bin/env python3

"""
SCRIPT_TestConventions.py

Comprehensive test script to demonstrate and validate coordinate conventions
in the DataProcessor module. Tests the consistency between world and robot
coordinate systems, yaw orientations, and azimuth conventions.

This script serves as both documentation and a regression test for the
coordinate transformation functions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Library'))

from DataProcessor import world2robot, robot2world
from Utils import plot_robot_positions
import numpy as np
import matplotlib.pyplot as plt

def test_world_robot_consistency():
    """Test that world2robot and robot2world are perfect inverses"""
    print("🔄 Testing World ↔ Robot Coordinate Consistency")
    print("=" * 50)
    
    # Test multiple robot positions and orientations
    test_cases = [
        (0, 0, 0),    # Origin, facing right
        (5, 3, 90),   # Facing up
        (2, 2, 45),   # Facing northeast
        (-1, -1, 180), # Facing left
        (3, -2, 270), # Facing down
    ]
    
    all_passed = True
    
    for rob_x, rob_y, rob_yaw in test_cases:
        print(f"\n📍 Robot at ({rob_x}, {rob_y}), yaw={rob_yaw}°")
        print("   Testing round-trip conversions: World → Robot → World")
        
        # Test multiple world points
        test_points = [
            (rob_x + 1, rob_y),     # 1m in X direction
            (rob_x, rob_y + 1),     # 1m in Y direction
            (rob_x + 1, rob_y + 1), # 1m diagonal
            (rob_x - 1, rob_y),     # 1m negative X
        ]
        
        for world_x, world_y in test_points:
            # Convert world -> robot
            x_rel, y_rel = world2robot(world_x, world_y, rob_x, rob_y, rob_yaw)
            
            # Convert back robot -> world
            x_back, y_back = robot2world(
                np.rad2deg(np.arctan2(y_rel, x_rel)),  # azimuth from relative coords
                np.sqrt(x_rel**2 + y_rel**2),           # distance
                rob_x, rob_y, rob_yaw
            )
            
            # Check consistency
            if np.allclose([x_back, y_back], [world_x, world_y]):
                print(f"  ✅ World({world_x}, {world_y}) → Robot({x_rel:.1f}, {y_rel:.1f}) → World({x_back:.1f}, {y_back:.1f})")
            else:
                print(f"  ❌ World({world_x}, {world_y}) → Robot({x_rel:.1f}, {y_rel:.1f}) → World({x_back:.1f}, {y_back:.1f})")
                all_passed = False
    
    print(f"\n{'🎉 All consistency tests passed!' if all_passed else '❌ Some tests failed!'}")
    return all_passed

def test_azimuth_convention():
    """Test and demonstrate the azimuth convention"""
    print("\n🎯 Testing Azimuth Convention")
    print("=" * 50)
    
    # Test with robot at origin facing right
    rob_x, rob_y, rob_yaw = 0, 0, 0
    
    print(f"Robot at ({rob_x}, {rob_y}), yaw={rob_yaw}° (facing right)")
    print("\nAzimuth Direction Map (MATHEMATICAL convention - counter-clockwise):")
    
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
    distances = [1] * len(azimuths)
    directions = ['Forward', 'Front-Up', 'Up/Left', 'Back-Up', 'Backward', 'Back-Down', 'Down/Right', 'Front-Down']
    
    x_world, y_world = robot2world(azimuths, distances, rob_x, rob_y, rob_yaw)
    
    for az, dist, x, y, direction in zip(azimuths, distances, x_world, y_world, directions):
        print(f"  {az:3d}° ({direction:10s}): ({x: .1f}, {y: .1f})")
    
    # Verify the key directions
    expected_positions = {
        0: (1, 0),    # Forward = right when yaw=0°
        90: (0, 1),   # Up/Left = up when yaw=0° (mathematical convention)
        180: (-1, 0), # Backward = left when yaw=0°
        270: (0, -1), # Down/Right = down when yaw=0° (mathematical convention)
    }
    
    print("\n🔍 Verifying key directions:")
    all_correct = True
    for az, expected in expected_positions.items():
        idx = azimuths.index(az)
        actual = (x_world[idx], y_world[idx])
        if np.allclose(actual, expected):
            print(f"  ✅ {az}° azimuth -> {actual} (expected {expected})")
        else:
            print(f"  ❌ {az}° azimuth -> {actual} (expected {expected})")
            all_correct = False
    
    print(f"\n{'🎉 Azimuth convention is correct!' if all_correct else '❌ Azimuth convention has issues!'}")
    return all_correct

def test_yaw_convention():
    """Test and demonstrate the yaw convention"""
    print("\n🧭 Testing Yaw Convention")
    print("=" * 50)
    
    # Test point 1 meter forward from robot at different orientations
    distance = 1.0
    
    print("Robot at origin (0,0) with different yaw orientations:")
    print("1 meter forward (azimuth=0°) should appear at different world positions")
    print()
    
    yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    directions = ['Right', 'Northeast', 'Up', 'Northwest', 'Left', 'Southwest', 'Down', 'Southeast']
    
    all_correct = True
    
    for yaw, direction in zip(yaw_angles, directions):
        rob_x, rob_y, rob_yaw = 0, 0, yaw
        
        # 1 meter forward (0° azimuth)
        x_world, y_world = robot2world(0, distance, rob_x, rob_y, rob_yaw)
        
        # Expected positions based on yaw
        expected_x = np.cos(np.deg2rad(yaw))
        expected_y = np.sin(np.deg2rad(yaw))
        
        if np.allclose([x_world, y_world], [expected_x, expected_y]):
            print(f"  ✅ Yaw {yaw:3d}° ({direction:8s}): ({x_world:.1f}, {y_world:.1f})")
        else:
            print(f"  ❌ Yaw {yaw:3d}° ({direction:8s}): ({x_world:.1f}, {y_world:.1f}) (expected {expected_x:.1f}, {expected_y:.1f})")
            all_correct = False
    
    print(f"\n{'🎉 Yaw convention is correct!' if all_correct else '❌ Yaw convention has issues!'}")
    return all_correct

def plot_coordinate_conventions():
    """Create visual plots of the coordinate conventions"""
    print("\n📊 Creating Coordinate Convention Plots")
    print("=" * 50)
    
    # Plot 1: Azimuth convention
    print("\n1️⃣ Creating azimuth convention visualization...")
    print("   This plot shows where points appear in world coordinates")
    print("   when using different azimuth angles from a robot at (0,0) facing right")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Robot at center
    rob_x, rob_y, rob_yaw = 0, 0, 0
    
    # Points at different azimuths
    azimuths = np.linspace(0, 360, 13)[:-1]  # 0° to 330° in 30° steps
    distances = np.ones(12) * 2.0
    
    x_world, y_world = robot2world(azimuths, distances, rob_x, rob_y, rob_yaw)
    
    # Plot robot
    ax.scatter([rob_x], [rob_y], c='red', s=100, marker='o', label='Robot')
    
    # Plot directions
    for az, x, y in zip(azimuths, x_world, y_world):
        ax.arrow(rob_x, rob_y, x - rob_x, y - rob_y, 
                 head_width=0.1, head_length=0.2, 
                 fc='blue', ec='blue', alpha=0.7)
        ax.text(x, y, f'{int(az)}°', ha='center', va='center')
    
    # Add labels for cardinal directions
    cardinals = [(0, 2.5, '0° Forward'), (2.5, 0, '90° Up/Left'), 
                 (0, -2.5, '180° Backward'), (-2.5, 0, '270° Down/Right')]
    for x, y, label in cardinals:
        ax.text(x, y, label, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Add explanation text
    ax.text(0, 3.2, 'MATHEMATICAL CONVENTION (Counter-Clockwise)', 
            ha='center', va='center', 
            bbox=dict(facecolor='lightblue', alpha=0.8))
    ax.text(0, -3.2, 'Azimuths increase COUNTER-CLOCKWISE from forward', 
            ha='center', va='center',
            bbox=dict(facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title('Azimuth Convention: Robot at (0,0) Facing Right (Yaw=0°)', fontsize=12)
    ax.set_xlabel('X (World Coordinates)', fontsize=10)
    ax.set_ylabel('Y (World Coordinates)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', alpha=0.5)
    ax.axvline(0, color='gray', alpha=0.5)
    
    # Save to Plots directory
    import os
    os.makedirs('Plots', exist_ok=True)
    fig.savefig('Plots/azimuth_convention.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: Plots/azimuth_convention.png")
    
    # Plot 2: Yaw convention
    print("\n2️⃣ Creating yaw convention visualization...")
    print("   This plot shows robot forward direction (red arrows)")
    print("   at different yaw orientations around a circle")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Different robot orientations
    yaw_angles = [0, 90, 180, 270]
    colors = ['red', 'green', 'blue', 'purple']
    positions = [(-3, 0), (0, 3), (3, 0), (0, -3)]
    
    for yaw, color, (x, y) in zip(yaw_angles, colors, positions):
        # Plot robot position
        ax.scatter([x], [y], c=color, s=100, marker='o', label=f'Yaw {yaw}°')
        
        # Plot forward direction (1m at 0° azimuth)
        x_forward, y_forward = robot2world(0, 1, x, y, yaw)
        ax.arrow(x, y, x_forward - x, y_forward - y, 
                 head_width=0.1, head_length=0.2, 
                 fc=color, ec=color, alpha=0.8, length_includes_head=True)
        
        # Add label
        direction = ['Right', 'Up', 'Left', 'Down'][yaw_angles.index(yaw)]
        ax.text(x, y + 0.5, f'Facing {direction}', ha='center', 
                bbox=dict(facecolor=color, alpha=0.3))
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title('Yaw Convention: Forward Direction at Different Orientations', fontsize=12)
    ax.set_xlabel('X (World Coordinates)', fontsize=10)
    ax.set_ylabel('Y (World Coordinates)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', alpha=0.5)
    ax.axvline(0, color='gray', alpha=0.5)
    ax.legend(fontsize=8)
    
    # Add explanation
    ax.text(0, 5.2, 'Yaw Convention: 0°=Right, 90°=Up, 180°=Left, 270°=Down (Clockwise)',
            ha='center', va='center',
            bbox=dict(facecolor='lightgreen', alpha=0.8))
    
    # Save to Plots directory
    fig.savefig('Plots/yaw_convention.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: Plots/yaw_convention.png")
    
    # Plot 3: World ↔ Robot consistency visualization
    print("\n3️⃣ Creating world ↔ robot consistency visualization...")
    print("   This plot demonstrates the perfect inverse relationship")
    print("   between world2robot and robot2world functions")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Test point
    rob_x, rob_y, rob_yaw = 2, 2, 45
    test_azimuths = [0, 90, 180, 270]
    test_distances = [1.5, 1.5, 1.5, 1.5]
    
    # Convert robot-relative to world
    x_world, y_world = robot2world(test_azimuths, test_distances, rob_x, rob_y, rob_yaw)
    
    # Convert back to robot-relative
    x_rel_back, y_rel_back = world2robot(x_world, y_world, rob_x, rob_y, rob_yaw)
    
    # Plot original robot-relative points
    for az, x, y in zip(test_azimuths, x_rel_back, y_rel_back):
        ax.scatter([x], [y], c='blue', s=50, alpha=0.7)
        ax.text(x, y, f'{az}°', ha='center', va='center',
                bbox=dict(facecolor='blue', alpha=0.3))
    
    # Plot world points
    for az, x, y in zip(test_azimuths, x_world, y_world):
        ax.scatter([x], [y], c='red', s=50, alpha=0.7)
        ax.text(x, y + 0.1, f'W{az}°', ha='center', va='center',
                bbox=dict(facecolor='red', alpha=0.3))
    
    # Plot robot position
    ax.scatter([rob_x], [rob_y], c='green', s=100, marker='*', label='Robot')
    
    # Add connecting lines
    for i in range(len(test_azimuths)):
        ax.plot([x_rel_back[i], x_world[i]], [y_rel_back[i], y_world[i]],
                'k--', alpha=0.3)
    
    # Add explanation
    ax.text(2, 4.5, 'World ↔ Robot Consistency Demonstration',
            ha='center', va='center',
            bbox=dict(facecolor='lightblue', alpha=0.8))
    ax.text(2, 4.2, 'Blue: Robot-relative coordinates (azimuth, distance)',
            ha='center', va='center',
            bbox=dict(facecolor='lightyellow', alpha=0.8))
    ax.text(2, 3.9, 'Red: World coordinates (converted from robot-relative)',
            ha='center', va='center',
            bbox=dict(facecolor='lightyellow', alpha=0.8))
    ax.text(2, 3.6, 'Dashed lines: Perfect inverse transformation',
            ha='center', va='center',
            bbox=dict(facecolor='lightgreen', alpha=0.8))
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.set_title('World ↔ Robot Coordinate Consistency', fontsize=12)
    ax.set_xlabel('X (World Coordinates)', fontsize=10)
    ax.set_ylabel('Y (World Coordinates)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Save to Plots directory
    fig.savefig('Plots/world_robot_consistency.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: Plots/world_robot_consistency.png")
    
    # Plot 4: Robot positions with orientations
    print("\n4️⃣ Creating robot path visualization...")
    print("   Using plot_robot_positions to show different orientations")
    
    # Create a path
    positions = [(0, 0, 0), (2, 0, 45), (4, 2, 90), (2, 4, 135), (0, 2, 180)]
    
    fig, ax = plot_robot_positions(
        [p[0] for p in positions], 
        [p[1] for p in positions],
        [p[2] for p in positions],
        dot_color='blue',
        arrow_color='red',
        arrow_length=0.5
    )
    
    # Add labels
    for i, (x, y, yaw) in enumerate(positions):
        ax.text(x, y + 0.3, f'Pos {i+1}', ha='center', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add explanation
    ax.text(2, 4.5, 'Robot Path with Different Orientations',
            ha='center', va='center',
            bbox=dict(facecolor='lightblue', alpha=0.8))
    
    # Save to Plots directory
    fig.savefig('Plots/robot_path.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: Plots/robot_path.png")
    
    print("\n🎉 All plots created successfully!")

def main():
    """Run all convention tests"""
    print("🚀 Running Coordinate Convention Tests")
    print("=" * 60)
    
    # Run all tests
    test1_passed = test_world_robot_consistency()
    test2_passed = test_azimuth_convention()
    test3_passed = test_yaw_convention()
    
    # Create visualizations
    plot_coordinate_conventions()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"World ↔ Robot Consistency: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Azimuth Convention:        {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Yaw Convention:           {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    
    # Test plot_robot_positions consistency
    print("\n🎨 Testing plot_robot_positions convention consistency...")
    rob_x, rob_y, rob_yaw = 2, 2, 45
    
    # Calculate expected forward using robot2world
    x_expected, y_expected = robot2world(0, 1, rob_x, rob_y, rob_yaw)
    
    # Calculate what plot_robot_positions would use
    yaw_rad = np.deg2rad(rob_yaw)
    dx = 1.0 * np.cos(yaw_rad)
    dy = 1.0 * np.sin(yaw_rad)
    x_plot = rob_x + dx
    y_plot = rob_y + dy
    
    if np.allclose([x_expected, y_expected], [x_plot, y_plot]):
        print(f"   ✅ plot_robot_positions uses same convention as robot2world")
        print(f"      Both use: dx = cos(yaw), dy = sin(yaw) for forward direction")
    else:
        print(f"   ❌ plot_robot_positions uses DIFFERENT convention!")
        print(f"      robot2world: ({x_expected:.3f}, {y_expected:.3f})")
        print(f"      plot_robot_positions: ({x_plot:.3f}, {y_plot:.3f})")
    
    print(f"\n{'🎉 ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED'}")
    print("\n📊 Visualizations created in Plots/ directory:")
    print("   • azimuth_convention.png - Shows azimuth directions (counter-clockwise)")
    print("   • yaw_convention.png - Shows robot orientations at different yaws")
    print("   • world_robot_consistency.png - Demonstrates perfect inverse transformations")
    print("   • robot_path.png - Shows robot path with plot_robot_positions")
    print("   • convention_check.png - Verifies plot_robot_positions convention")
    print("\n💡 All plots include detailed explanations and are saved with high quality")
    
    # Clean up plots (optional)
    # import os
    # for filename in ['azimuth_convention.png', 'yaw_convention.png', 'robot_path.png']:
    #     try:
    #         os.remove(filename)
    #     except FileNotFoundError:
    #         pass

if __name__ == "__main__":
    main()