#!/usr/bin/env python3
"""
Debug script to check if the conical mask is working correctly.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

def debug_conical_mask():
    """Debug the conical mask creation."""
    
    print("🔍 Debugging Conical Mask Creation")
    print("=" * 50)
    
    # Initialize data processor
    processor = DataProcessor('session03')
    processor._load_meta_only()
    
    # Load arena image
    arena_img = processor.load_arena_image()
    processor.arena_image_shape = arena_img.shape
    
    # Test with a specific position
    test_index = 10
    
    rob_x = processor.rob_x[test_index]
    rob_y = processor.rob_y[test_index]
    rob_yaw = processor.rob_yaw_deg[test_index]
    
    print(f"Testing index {test_index}:")
    print(f"  Position: ({rob_x:.1f}, {rob_y:.1f}) mm")
    print(f"  Yaw: {rob_yaw:.1f}°")
    
    # Convert to pixel coordinates
    center_x, center_y = processor.world_to_pixel(rob_x, rob_y)
    print(f"  Pixel coordinates: ({center_x}, {center_y})")
    
    # Parameters
    radius_mm = 1500
    opening_deg = 90
    
    # Convert radius to pixels
    mm_per_px = float(processor.meta["map_mm_per_px"])
    radius_px = int(round(radius_mm / mm_per_px))
    
    # Create conical mask
    print("Creating conical mask...")
    conical_mask = processor.create_conical_mask(
        center_x, center_y, radius_px, opening_deg, rob_yaw
    )
    
    # Check mask statistics
    total_pixels = np.sum(conical_mask)
    print(f"  Conical mask has {total_pixels} pixels")
    
    # Check if the center pixel is in the mask
    center_in_mask = conical_mask[center_y, center_x]
    print(f"  Center pixel in mask: {center_in_mask}")
    
    # Extract conical region
    conical_region = np.zeros_like(arena_img)
    conical_region[conical_mask] = arena_img[conical_mask]
    
    # Check center pixel in conical region
    center_pixel = conical_region[center_y, center_x]
    print(f"  Center pixel in conical region: {center_pixel}")
    print(f"  Center is non-black: {np.any(center_pixel > 10)}")
    
    # Visualize
    plt.figure(figsize=(15, 8))
    
    # 1. Original arena
    plt.subplot(1, 3, 1)
    plt.imshow(arena_img)
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.title('Original Arena')
    plt.axis('off')
    
    # 2. Conical mask
    plt.subplot(1, 3, 2)
    mask_display = np.zeros_like(arena_img)
    mask_display[conical_mask] = [255, 0, 0]  # Red for cone region
    plt.imshow(arena_img)
    plt.imshow(mask_display, alpha=0.5)
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.title('Conical Mask Overlay')
    plt.axis('off')
    
    # 3. Conical region
    plt.subplot(1, 3, 3)
    plt.imshow(conical_region)
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.title('Conical Region')
    plt.axis('off')
    
    plt.suptitle('Conical Mask Debug')
    plt.tight_layout()
    plt.show()
    
    # Check a larger area around the center
    print("Checking 10x10 area around center:")
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            y = center_y + dy
            x = center_x + dx
            if 0 <= y < conical_region.shape[0] and 0 <= x < conical_region.shape[1]:
                pixel = conical_region[y, x]
                in_mask = conical_mask[y, x]
                if np.any(pixel > 10):
                    print(f"  ({dx:2d}, {dy:2d}): {pixel} - in_mask: {in_mask}")
    
    return center_in_mask and np.any(center_pixel > 10)

if __name__ == "__main__":
    debug_conical_mask()