#!/usr/bin/env python3

"""
Compare trajectory plot orientation with conical view orientation for the same robot.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

def compare_trajectory_vs_conical():
    """Compare orientations for the same robot position."""
    
    print("🚀 Comparing Trajectory vs Conical View Orientations")
    print("=" * 60)
    
    processor = DataProcessor('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code/Data/session03')
    processor.load_arena_metadata()
    arena_img = processor.load_arena_image()
    
    # Test the first robot position
    test_index = 0
    
    rob_x = processor.rob_x[test_index]
    rob_y = processor.rob_y[test_index]
    rob_yaw = processor.rob_yaw_deg[test_index]
    
    print(f"Analyzing robot position {test_index}:")
    print(f"  Position: ({rob_x:.1f}, {rob_y:.1f}) mm")
    print(f"  Yaw: {rob_yaw:.1f}°")
    
    # Convert to pixel coordinates
    center_x, center_y = processor.world_to_pixel(rob_x, rob_y)
    
    # Get next position to calculate movement direction
    if test_index + 1 < len(processor.rob_x):
        next_x = processor.rob_x[test_index + 1]
        next_y = processor.rob_y[test_index + 1]
        movement_angle = np.degrees(np.arctan2(next_y - rob_y, next_x - rob_x)) % 360
        print(f"  Movement direction: {movement_angle:.1f}°")
    
    plt.figure(figsize=(16, 12))
    plt.imshow(arena_img)
    
    # Draw robot position
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.text(center_x, center_y - 20, f'Robot {test_index}', ha='center', color='white',
             bbox=dict(facecolor='black', alpha=0.7))
    
    # Draw TRAJECTORY PLOT orientation (this should match movement)
    arrow_len = 80
    u_trajectory = np.cos(np.deg2rad(rob_yaw)) * arrow_len
    v_trajectory = np.sin(np.deg2rad(rob_yaw)) * arrow_len
    plt.arrow(center_x, center_y, u_trajectory, v_trajectory,
              head_width=10, head_length=15, fc='green', ec='green', width=2)
    plt.text(center_x + u_trajectory, center_y + v_trajectory, 'Trajectory orientation',
             color='green', bbox=dict(facecolor='black', alpha=0.7))
    
    # Draw CONICAL VIEW orientation (current implementation)
    corrected_yaw = (rob_yaw + 180) % 360
    u_conical = np.cos(np.deg2rad(corrected_yaw)) * arrow_len
    v_conical = np.sin(np.deg2rad(corrected_yaw)) * arrow_len
    plt.arrow(center_x, center_y, u_conical, v_conical,
              head_width=10, head_length=15, fc='blue', ec='blue', width=2)
    plt.text(center_x + u_conical, center_y + v_conical, 'Conical view orientation',
             color='blue', bbox=dict(facecolor='black', alpha=0.7))
    
    # Draw MOVEMENT direction
    if test_index + 1 < len(processor.rob_x):
        next_px, next_py = processor.world_to_pixel(next_x, next_y)
        plt.arrow(center_x, center_y, next_px - center_x, next_py - center_y,
                  head_width=10, head_length=15, fc='yellow', ec='yellow', width=2, alpha=0.7)
        plt.text(next_px, next_py, 'Actual movement',
                 color='yellow', bbox=dict(facecolor='black', alpha=0.7))
    
    # Draw conical view extraction area
    radius_px = 60
    opening_angle_deg = 60
    half_angle = np.deg2rad(opening_angle_deg / 2)
    cone_theta = np.deg2rad(corrected_yaw) + np.linspace(-half_angle, half_angle, 100)
    x_cone = center_x + radius_px * np.cos(cone_theta)
    y_cone = center_y + radius_px * np.sin(cone_theta)
    plt.plot(x_cone, y_cone, 'blue', linewidth=2, alpha=0.6, linestyle='--')
    plt.plot([center_x, x_cone[0]], [center_y, y_cone[0]], 'blue', linewidth=2, alpha=0.6, linestyle='--')
    plt.plot([center_x, x_cone[-1]], [center_y, y_cone[-1]], 'blue', linewidth=2, alpha=0.6, linestyle='--')
    
    plt.title(f'Comparison: Trajectory vs Conical View vs Movement\nRobot {test_index}: ({rob_x:.1f}, {rob_y:.1f}) mm, Yaw: {rob_yaw:.1f}°')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Analysis:")
    print(f"  - Green arrow: Trajectory plot orientation (yaw)")
    print(f"  - Blue arrow: Conical view orientation (yaw + 180°)")
    print(f"  - Yellow arrow: Actual movement direction")
    print(f"  - Blue dashed cone: Conical view extraction area")
    print(f"\nThe conical view should point in the direction the robot is facing/moving.")
    print(f"If these don't align, the orientation needs to be corrected.")

if __name__ == "__main__":
    compare_trajectory_vs_conical()