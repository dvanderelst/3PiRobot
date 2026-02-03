#!/usr/bin/env python3

"""
Compare robot orientation in trajectory plot vs conical view.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

def compare_orientations():
    """Compare trajectory plot orientation with conical view orientation."""
    
    print("🚀 Comparing Trajectory vs Conical View Orientations")
    print("=" * 60)
    
    processor = DataProcessor('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code/Data/session03')
    processor.load_arena_metadata()
    arena_img = processor.load_arena_image()
    
    # Test a specific robot position
    test_index = 10  # Change this to test different positions
    
    rob_x = processor.rob_x[test_index]
    rob_y = processor.rob_y[test_index]
    rob_yaw = processor.rob_yaw_deg[test_index]
    
    print(f"Testing robot position {test_index}:")
    print(f"  Position: ({rob_x:.1f}, {rob_y:.1f}) mm")
    print(f"  Yaw: {rob_yaw:.1f}°")
    
    # Convert to pixel coordinates
    center_x, center_y = processor.world_to_pixel(rob_x, rob_y)
    
    plt.figure(figsize=(16, 12))
    
    # Show arena image
    plt.imshow(arena_img)
    
    # Draw robot position
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.text(center_x, center_y - 15, f'Robot {test_index}', ha='center', color='white',
             bbox=dict(facecolor='black', alpha=0.7))
    
    # Draw orientation using TRAJECTORY PLOT convention (this should be correct)
    arrow_len = 60
    u = np.cos(np.deg2rad(rob_yaw)) * arrow_len
    v = np.sin(np.deg2rad(rob_yaw)) * arrow_len
    
    plt.quiver(center_x, center_y, u, v, 
               angles='xy', scale_units='xy', scale=1, width=0.005, color='yellow')
    plt.text(center_x + u, center_y + v, 'Trajectory orientation', 
             color='yellow', bbox=dict(facecolor='black', alpha=0.7))
    
    # Draw conical view orientation (current implementation)
    radius_px = 50
    opening_angle_deg = 60
    
    # Current conical view orientation
    half_angle = np.deg2rad(opening_angle_deg / 2)
    cone_theta = np.deg2rad(rob_yaw) + np.linspace(-half_angle, half_angle, 50)
    x_cone = center_x + radius_px * np.cos(cone_theta)
    y_cone = center_y + radius_px * np.sin(cone_theta)
    
    plt.plot(x_cone, y_cone, 'blue', linewidth=3, alpha=0.8)
    plt.plot([center_x, x_cone[0]], [center_y, y_cone[0]], 'blue', linewidth=3, alpha=0.8)
    plt.plot([center_x, x_cone[-1]], [center_y, y_cone[-1]], 'blue', linewidth=3, alpha=0.8)
    
    # Label the middle of the cone
    mid_idx = len(cone_theta) // 2
    mid_x, mid_y = x_cone[mid_idx], y_cone[mid_idx]
    plt.text(mid_x, mid_y, 'Current conical view', 
             color='blue', bbox=dict(facecolor='white', alpha=0.7))
    
    # Draw what a 180° rotated conical view would look like
    cone_theta_180 = np.deg2rad(rob_yaw + 180) + np.linspace(-half_angle, half_angle, 50)
    x_cone_180 = center_x + radius_px * np.cos(cone_theta_180)
    y_cone_180 = center_y + radius_px * np.sin(cone_theta_180)
    
    plt.plot(x_cone_180, y_cone_180, 'red', linewidth=2, alpha=0.6, linestyle='--')
    plt.plot([center_x, x_cone_180[0]], [center_y, y_cone_180[0]], 'red', linewidth=2, alpha=0.6, linestyle='--')
    plt.plot([center_x, x_cone_180[-1]], [center_y, y_cone_180[-1]], 'red', linewidth=2, alpha=0.6, linestyle='--')
    
    # Label the middle of the 180° rotated cone
    mid_x_180, mid_y_180 = x_cone_180[mid_idx], y_cone_180[mid_idx]
    plt.text(mid_x_180, mid_y_180, '180° rotated conical view', 
             color='red', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(f'Comparison: Trajectory Orientation vs Conical View Orientation\nRobot {test_index} at ({rob_x:.1f}, {rob_y:.1f}) mm, Yaw: {rob_yaw:.1f}°')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"\n👀 Examining the visualization:")
    print(f"  - Yellow arrow: Trajectory plot orientation (should be correct)")
    print(f"  - Blue cone: Current conical view orientation")
    print(f"  - Red dashed cone: 180° rotated conical view")
    print(f"\nIf the blue cone doesn't align with the yellow arrow,")
    print(f"then the conical view orientation needs to be corrected.")
    print(f"The red dashed cone shows what a 180° rotation would look like.")

if __name__ == "__main__":
    compare_orientations()