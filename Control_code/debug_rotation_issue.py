#!/usr/bin/env python3
"""
Debug script to understand the rotation and centering issue.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt
import cv2

def debug_rotation_and_centering():
    """Debug the rotation and centering process step by step."""
    
    print("🔍 Debugging Rotation and Centering")
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
    radius_mm = 2500
    opening_deg = 90
    output_size = (256, 256)
    
    # Convert radius to pixels
    mm_per_px = float(processor.meta["map_mm_per_px"])
    radius_px = int(round(radius_mm / mm_per_px))
    
    # Create conical mask
    print("Creating conical mask...")
    conical_mask = processor.create_conical_mask(
        center_x, center_y, radius_px, opening_deg, rob_yaw
    )
    
    # Check if center is in mask
    center_in_mask = conical_mask[center_y, center_x]
    print(f"  Center pixel in mask: {center_in_mask}")
    
    # Extract conical region
    conical_region = np.zeros_like(arena_img)
    conical_region[conical_mask] = arena_img[conical_mask]
    
    # Check center pixel in conical region
    center_pixel = conical_region[center_y, center_x]
    print(f"  Center pixel in conical region: {center_pixel}")
    
    # Create padding and extract region
    padding = radius_px // 2
    min_x = max(0, center_x - radius_px - padding)
    max_x = min(arena_img.shape[1], center_x + radius_px + padding)
    min_y = max(0, center_y - radius_px - padding)
    max_y = min(arena_img.shape[0], center_y + radius_px + padding)
    
    region_with_padding = conical_region[min_y:max_y, min_x:max_x]
    region_center_x = center_x - min_x
    region_center_y = center_y - min_y
    
    print(f"  Region with padding shape: {region_with_padding.shape}")
    print(f"  Region center: ({region_center_x}, {region_center_y})")
    
    # Check center in extracted region
    region_center_pixel = region_with_padding[region_center_y, region_center_x]
    print(f"  Region center pixel: {region_center_pixel}")
    
    # Rotation
    rotation_angle = 90 - rob_yaw
    print(f"  Rotation angle: {rotation_angle}°")
    
    rotation_matrix = cv2.getRotationMatrix2D(
        (region_center_x, region_center_y),
        rotation_angle,
        1.0
    )
    
    rotated_region = cv2.warpAffine(
        region_with_padding,
        rotation_matrix,
        (region_with_padding.shape[1], region_with_padding.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # Check if center is still at rotation center after rotation
    rotated_center_pixel = rotated_region[region_center_y, region_center_x]
    print(f"  After rotation, center pixel: {rotated_center_pixel}")
    
    # Centering
    target_center_x = rotated_region.shape[1] // 2
    target_center_y = rotated_region.shape[0] // 2
    
    translation_x = target_center_x - region_center_x
    translation_y = target_center_y - region_center_y
    
    print(f"  Translation: ({translation_x}, {translation_y})")
    
    translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    centered_region = cv2.warpAffine(
        rotated_region,
        translation_matrix,
        (rotated_region.shape[1], rotated_region.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # Check final center
    final_center_pixel = centered_region[target_center_y, target_center_x]
    print(f"  Final center pixel: {final_center_pixel}")
    
    # Visualize each step
    plt.figure(figsize=(20, 12))
    
    # 1. Original arena with cone
    plt.subplot(2, 3, 1)
    plt.imshow(arena_img)
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.title(f'Original Arena\nRobot at ({center_x}, {center_y})')
    plt.axis('off')
    
    # 2. Conical region
    plt.subplot(2, 3, 2)
    plt.imshow(conical_region)
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.title(f'Conical Region\nCenter pixel: {center_pixel}')
    plt.axis('off')
    
    # 3. Region with padding
    plt.subplot(2, 3, 3)
    plt.imshow(region_with_padding)
    plt.scatter(region_center_x, region_center_y, c='red', s=100, marker='x')
    plt.title(f'Region with Padding\nRegion center: ({region_center_x}, {region_center_y})')
    plt.axis('off')
    
    # 4. After rotation
    plt.subplot(2, 3, 4)
    plt.imshow(rotated_region)
    plt.scatter(region_center_x, region_center_y, c='red', s=100, marker='x')
    plt.scatter(target_center_x, target_center_y, c='green', s=50, marker='o')
    plt.title(f'After Rotation\nRotation center: ({region_center_x}, {region_center_y})')
    plt.axis('off')
    
    # 5. After centering
    plt.subplot(2, 3, 5)
    plt.imshow(centered_region)
    plt.scatter(target_center_x, target_center_y, c='red', s=100, marker='x')
    plt.title(f'After Centering\nFinal center: ({target_center_x}, {target_center_y})')
    plt.axis('off')
    
    # 6. Final cropped view
    crop_size = min(centered_region.shape[:2])
    start_x = (centered_region.shape[1] - crop_size) // 2
    start_y = (centered_region.shape[0] - crop_size) // 2
    cropped_region = centered_region[start_y:start_y+crop_size, start_x:start_x+crop_size]
    final_view = cv2.resize(cropped_region, output_size, interpolation=cv2.INTER_AREA)
    
    plt.subplot(2, 3, 6)
    plt.imshow(final_view)
    final_center_x, final_center_y = final_view.shape[1] // 2, final_view.shape[0] // 2
    plt.scatter(final_center_x, final_center_y, c='red', s=100, marker='x')
    final_center_pixel = final_view[final_center_y, final_center_x]
    plt.title(f'Final View\nCenter pixel: {final_center_pixel}')
    plt.axis('off')
    
    plt.suptitle('Rotation and Centering Debug')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    debug_rotation_and_centering()