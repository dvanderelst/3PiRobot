#!/usr/bin/env python3
"""
Debug the conical mask creation for index 14.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

def debug_mask_creation():
    """Debug the conical mask creation."""
    
    print("🔍 Debugging Conical Mask Creation for Index 14")
    print("=" * 60)
    
    # Initialize data processor
    processor = DataProcessor('session03')
    processor._load_meta_only()
    
    # Load arena image
    arena_img = processor.load_arena_image()
    processor.arena_image_shape = arena_img.shape
    
    # Test index 14 specifically
    test_index = 14
    
    rob_x = processor.rob_x[test_index]
    rob_y = processor.rob_y[test_index]
    rob_yaw = processor.rob_yaw_deg[test_index]
    
    print(f"Index {test_index}:")
    print(f"  Position: ({rob_x:.1f}, {rob_y:.1f}) mm")
    print(f"  Yaw: {rob_yaw:.1f}°")
    
    # Convert to pixel coordinates
    center_x, center_y = processor.world_to_pixel(rob_x, rob_y)
    print(f"  Pixel coordinates: ({center_x}, {center_y})")
    
    # Parameters
    radius_mm = 2500
    opening_deg = 90
    
    # Convert radius to pixels
    mm_per_px = float(processor.meta["map_mm_per_px"])
    radius_px = int(round(radius_mm / mm_per_px))
    
    print(f"  Radius: {radius_mm}mm = {radius_px}px")
    
    # Create conical mask
    conical_mask = processor.create_conical_mask(
        center_x, center_y, radius_px, opening_deg, rob_yaw
    )
    
    # Analyze the mask
    total_mask_pixels = np.sum(conical_mask)
    print(f"  Conical mask has {total_mask_pixels} pixels")
    
    # Check mask symmetry
    # Count pixels in left vs right halves
    left_half = np.sum(conical_mask[:, :center_x])
    right_half = np.sum(conical_mask[:, center_x:])
    print(f"  Left half pixels: {left_half}")
    print(f"  Right half pixels: {right_half}")
    print(f"  Ratio: {left_half/right_half:.2f}")
    
    # Check mask in different quadrants
    top_left = np.sum(conical_mask[:center_y, :center_x])
    top_right = np.sum(conical_mask[:center_y, center_x:])
    bottom_left = np.sum(conical_mask[center_y:, :center_x])
    bottom_right = np.sum(conical_mask[center_y:, center_x:])
    
    print(f"  Quadrant analysis:")
    print(f"    Top-left: {top_left}")
    print(f"    Top-right: {top_right}")
    print(f"    Bottom-left: {bottom_left}")
    print(f"    Bottom-right: {bottom_right}")
    
    # Visualize the mask
    plt.figure(figsize=(15, 8))
    
    # 1. Original arena
    plt.subplot(1, 2, 1)
    plt.imshow(arena_img)
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    
    # Draw cone outline
    half_opening_rad = np.deg2rad(opening_deg / 2)
    orientation_rad = np.deg2rad(rob_yaw)
    lower_bound = orientation_rad - half_opening_rad
    upper_bound = orientation_rad + half_opening_rad
    cone_theta = np.linspace(lower_bound, upper_bound, 100)
    x_cone = center_x + radius_px * np.cos(cone_theta)
    y_cone = center_y - radius_px * np.sin(cone_theta)
    plt.plot(x_cone, y_cone, 'r-', linewidth=2)
    
    plt.title('Original Arena with Cone Outline')
    plt.axis('off')
    
    # 2. Conical mask overlay
    plt.subplot(1, 2, 2)
    plt.imshow(arena_img)
    
    # Create mask display
    mask_display = np.zeros((*arena_img.shape[:2], 3), dtype=np.uint8)
    mask_display[conical_mask] = [255, 0, 0]  # Red for cone region
    plt.imshow(mask_display, alpha=0.5)
    
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.title(f'Conical Mask Overlay\nTotal pixels: {total_mask_pixels}')
    plt.axis('off')
    
    plt.suptitle(f'Index {test_index} - Mask Analysis')
    plt.tight_layout()
    plt.show()
    
    # Check if the mask matches the outline
    print(f"\nMask vs Outline Analysis:")
    print(f"  The red outline shows the theoretical cone boundaries")
    print(f"  The red transparent overlay shows the actual mask pixels")
    print(f"  If they don't match, there's an issue with mask creation")

if __name__ == "__main__":
    debug_mask_creation()