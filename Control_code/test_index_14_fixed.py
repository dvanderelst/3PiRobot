#!/usr/bin/env python3
"""
Test the fixed approach for index 14.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

def test_index_14_fixed():
    """Test the fixed approach for index 14."""
    
    print("🔍 Testing Index 14 with Fixed Approach")
    print("=" * 50)
    
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
    
    # Extract conical view using the fixed method
    view = processor.extract_conical_view(
        rob_x, rob_y, rob_yaw,
        radius_mm=2500,
        opening_angle_deg=90,
        output_size=(256, 256),
        visualize=True
    )
    
    # Analyze the result
    gray_view = np.mean(view, axis=2)
    non_black_pixels = np.sum(gray_view > 10)
    
    print(f"Final view analysis:")
    print(f"  Non-black pixels: {non_black_pixels}")
    
    # Check center
    center_x, center_y = view.shape[1] // 2, view.shape[0] // 2
    center_pixel = view[center_y, center_x]
    has_center_content = np.any(center_pixel > 10)
    
    print(f"  Center pixel: {center_pixel}")
    print(f"  Has content at center: {has_center_content}")
    
    # Show the view
    plt.figure(figsize=(8, 8))
    plt.imshow(view)
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.title(f'Index {test_index} - Fixed Approach\nNon-black pixels: {non_black_pixels}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_index_14_fixed()