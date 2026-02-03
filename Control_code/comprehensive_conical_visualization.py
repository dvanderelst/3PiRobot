#!/usr/bin/env python3

"""
Comprehensive visualization of conical view extraction process.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

def comprehensive_conical_view_visualization():
    """Show exactly how conical views are extracted from the arena."""
    
    print("🚀 Comprehensive Conical View Extraction Visualization")
    print("=" * 70)
    
    processor = DataProcessor('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code/Data/session03')
    processor.load_arena_metadata()
    arena_img = processor.load_arena_image()
    
    # Test a specific robot position
    test_index = 10  # Change this to test different positions
    
    rob_x = processor.rob_x[test_index]
    rob_y = processor.rob_y[test_index]
    rob_yaw = processor.rob_yaw_deg[test_index]
    
    print(f"Analyzing robot position {test_index}:")
    print(f"  World position: ({rob_x:.1f}, {rob_y:.1f}) mm")
    print(f"  Yaw orientation: {rob_yaw:.1f}°")
    
    # Convert to pixel coordinates
    center_x, center_y = processor.world_to_pixel(rob_x, rob_y)
    print(f"  Pixel coordinates: ({center_x}, {center_y})")
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 12))
    
    # Subplot 1: Full arena with robot position and conical view extraction area
    plt.subplot(1, 3, 1)
    plt.imshow(arena_img)
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.text(center_x, center_y - 20, f'Robot {test_index}', ha='center', color='white',
             bbox=dict(facecolor='black', alpha=0.7))
    
    # Draw robot orientation arrow (like in trajectory plot)
    arrow_len = 80
    u = np.cos(np.deg2rad(rob_yaw)) * arrow_len
    v = np.sin(np.deg2rad(rob_yaw)) * arrow_len
    plt.arrow(center_x, center_y, u, v, 
              head_width=10, head_length=15, fc='yellow', ec='yellow', width=2)
    plt.text(center_x + u, center_y + v, 'Robot orientation (yaw)', 
             color='yellow', bbox=dict(facecolor='black', alpha=0.7))
    
    # Draw the conical view extraction area
    radius_px = 60
    opening_angle_deg = 60
    
    # Show the cone boundaries
    half_angle = np.deg2rad(opening_angle_deg / 2)
    cone_theta = np.deg2rad((rob_yaw + 180) % 360) + np.linspace(-half_angle, half_angle, 100)
    x_cone = center_x + radius_px * np.cos(cone_theta)
    y_cone = center_y + radius_px * np.sin(cone_theta)
    
    plt.plot(x_cone, y_cone, 'blue', linewidth=3, alpha=0.8)
    plt.plot([center_x, x_cone[0]], [center_y, y_cone[0]], 'blue', linewidth=2, alpha=0.8)
    plt.plot([center_x, x_cone[-1]], [center_y, y_cone[-1]], 'blue', linewidth=2, alpha=0.8)
    
    # Label the cone
    mid_idx = len(cone_theta) // 2
    mid_x, mid_y = x_cone[mid_idx], y_cone[mid_idx]
    plt.text(mid_x, mid_y, 'Conical view extraction area', 
             color='blue', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(f'Arena with Conical View Extraction\nRobot {test_index}: ({rob_x:.1f}, {rob_y:.1f}) mm, Yaw: {rob_yaw:.1f}°')
    plt.axis('off')
    
    # Subplot 2: Extracted conical view (what gets saved)
    plt.subplot(1, 3, 2)
    
    # Extract the conical view using the current implementation
    conical_view = processor.extract_conical_view(
        rob_x, rob_y, rob_yaw,
        radius_mm=1000,  # 1 meter radius
        opening_angle_deg=60,
        output_size=(128, 128),
        visualize=False
    )
    
    plt.imshow(conical_view)
    plt.title(f'Extracted Conical View\n1000mm radius, 60° opening')
    plt.axis('off')
    
    # Subplot 3: Close-up of extraction area with transparency overlay
    plt.subplot(1, 3, 3)
    
    # Show a close-up of the arena around the robot
    close_up_size = 150  # pixels
    x_min = max(0, center_x - close_up_size)
    x_max = min(arena_img.shape[1], center_x + close_up_size)
    y_min = max(0, center_y - close_up_size)
    y_max = min(arena_img.shape[0], center_y + close_up_size)
    
    arena_close_up = arena_img[y_min:y_max, x_min:x_max]
    plt.imshow(arena_close_up)
    
    # Overlay the conical view extraction area with transparency
    mask = np.zeros_like(arena_close_up)
    
    # Create mask for the conical region
    y, x = np.ogrid[:arena_close_up.shape[0], :arena_close_up.shape[1]]
    x_rel = x - (center_x - x_min)
    y_rel = y - (center_y - y_min)
    
    distance = np.sqrt(x_rel**2 + y_rel**2)
    angles = np.arctan2(y_rel, x_rel)
    angles = np.mod(angles, 2 * np.pi)
    
    half_angle = np.deg2rad(opening_angle_deg / 2)
    orient_rad = np.deg2rad((rob_yaw + 180) % 360)
    
    in_cone = np.logical_or(
        (angles >= orient_rad - half_angle) & (angles <= orient_rad + half_angle),
        (orient_rad - half_angle < 0) & (angles >= orient_rad - half_angle + 2 * np.pi)
    )
    in_cone = in_cone & (distance <= radius_px)
    
    # Create semi-transparent overlay
    overlay = np.zeros((*arena_close_up.shape[:2], 4), dtype=np.uint8)
    overlay[..., :3] = 255  # White
    overlay[..., 3] = 100  # Semi-transparent
    overlay[~in_cone] = 0  # Make non-cone area transparent
    
    plt.imshow(overlay)
    
    # Draw center and orientation
    plt.scatter(center_x - x_min, center_y - y_min, c='red', s=50, marker='x')
    
    # Draw orientation arrow
    local_u = (center_x + u - x_min) - (center_x - x_min)
    local_v = (center_y + v - y_min) - (center_y - y_min)
    plt.arrow(center_x - x_min, center_y - y_min, local_u, local_v,
              head_width=5, head_length=8, fc='yellow', ec='yellow')
    
    plt.title(f'Close-up: Conical Region Overlay\nWhite area = extracted view')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Analysis:")
    print(f"  1. Left: Full arena showing where the conical view is extracted from")
    print(f"  2. Middle: The actual extracted conical view (what gets saved)")
    print(f"  3. Right: Close-up showing exactly which pixels are extracted")
    print(f"\n👀 What to check:")
    print(f"  - Does the blue cone in the left plot point in the expected direction?")
    print(f"  - Does the extracted view (middle) show the area ahead of the robot?")
    print(f"  - Does the white overlay (right) match the expected forward direction?")
    print(f"\nIf the conical view doesn't show what's ahead of the robot,")
    print(f"then the orientation needs to be adjusted.")

if __name__ == "__main__":
    comprehensive_conical_view_visualization()