#!/usr/bin/env python3

"""
Debug script to visualize conical view orientation options.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

def debug_conical_orientation():
    """Debug conical view orientation by showing different options."""
    
    print("🚀 Debugging Conical View Orientation")
    print("=" * 50)
    
    processor = DataProcessor('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code/Data/session03')
    processor.load_arena_metadata()
    arena_img = processor.load_arena_image()
    
    # Test a few different robot positions
    test_indices = [10, 50, 100]
    
    for idx in test_indices:
        print(f"\n📍 Testing robot position {idx}:")
        
        rob_x = processor.rob_x[idx]
        rob_y = processor.rob_y[idx]
        rob_yaw = processor.rob_yaw_deg[idx]
        
        print(f"   Position: ({rob_x:.1f}, {rob_y:.1f}) mm")
        print(f"   Yaw: {rob_yaw:.1f}°")
        
        # Convert to pixel coordinates
        center_x, center_y = processor.world_to_pixel(rob_x, rob_y)
        
        # Test different orientation options
        orientations = [
            (rob_yaw, "Original (yaw)"),
            ((rob_yaw + 90) % 360, "Yaw + 90°"),
            ((rob_yaw + 180) % 360, "Yaw + 180°"),
            ((rob_yaw - 90) % 360, "Yaw - 90°")
        ]
        
        plt.figure(figsize=(16, 12))
        plt.imshow(arena_img)
        plt.scatter(center_x, center_y, c='red', s=100, marker='x')
        plt.title(f'Robot {idx}: Testing Different Conical Orientations')
        
        # Draw robot orientation arrow
        arrow_length = 50
        arrow_x = center_x + arrow_length * np.cos(np.deg2rad(rob_yaw))
        arrow_y = center_y + arrow_length * np.sin(np.deg2rad(rob_yaw))
        plt.arrow(center_x, center_y, arrow_x - center_x, arrow_y - center_y,
                  head_width=8, head_length=12, fc='yellow', ec='yellow', width=2)
        plt.text(center_x, center_y - 10, f'Yaw: {rob_yaw:.1f}°', ha='center', color='white',
                 bbox=dict(facecolor='black', alpha=0.7))
        
        # Draw different cone options
        radius_px = 40
        opening_angle_deg = 60
        
        for i, (orient_deg, label) in enumerate(orientations):
            # Calculate cone boundaries
            half_angle = np.deg2rad(opening_angle_deg / 2)
            orient_rad = np.deg2rad(orient_deg)
            
            cone_theta = np.linspace(orient_rad - half_angle, orient_rad + half_angle, 50)
            x_cone = center_x + radius_px * np.cos(cone_theta)
            y_cone = center_y + radius_px * np.sin(cone_theta)
            
            # Draw cone outline
            colors = ['blue', 'green', 'red', 'cyan']
            plt.plot(x_cone, y_cone, color=colors[i], linewidth=2, alpha=0.8)
            plt.plot([center_x, x_cone[0]], [center_y, y_cone[0]], color=colors[i], linewidth=2, alpha=0.8)
            plt.plot([center_x, x_cone[-1]], [center_y, y_cone[-1]], color=colors[i], linewidth=2, alpha=0.8)
            
            # Add label
            label_x = center_x + (radius_px + 20) * np.cos(orient_rad)
            label_y = center_y + (radius_px + 20) * np.sin(orient_rad)
            plt.text(label_x, label_y, label, color=colors[i],
                     bbox=dict(facecolor='white', alpha=0.7))
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"   👀 Please examine the visualization to determine the correct orientation")
        print(f"   Options: 1=Original (blue), 2=+90° (green), 3=+180° (red), 4=-90° (cyan)")
        print(f"   Close the figure window to continue...")

if __name__ == "__main__":
    debug_conical_orientation()