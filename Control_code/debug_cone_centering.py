#!/usr/bin/env python3
"""
Debug script to understand what's happening in the cone centering process.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt
import cv2

def debug_cone_extraction():
    """Debug the cone extraction process step by step."""
    
    print("🔍 Debugging Cone Extraction Process")
    print("=" * 50)
    
    # Initialize data processor
    processor = DataProcessor('session03')
    processor._load_meta_only()
    
    # Load arena image
    arena_img = processor.load_arena_image()
    processor.arena_image_shape = arena_img.shape
    print(f"Arena image shape: {arena_img.shape}")
    
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
    output_size = (128, 128)
    
    # Convert radius to pixels
    mm_per_px = float(processor.meta["map_mm_per_px"])
    radius_px = int(round(radius_mm / mm_per_px))
    print(f"  Radius: {radius_mm}mm = {radius_px}px")
    
    # Create conical mask
    print("Creating conical mask...")
    conical_mask = processor.create_conical_mask(
        center_x, center_y, radius_px, opening_deg, rob_yaw
    )
    
    # Extract conical region
    conical_region = np.zeros_like(arena_img)
    conical_region[conical_mask] = arena_img[conical_mask]
    
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
    
    print(f"  After rotation, region center should still be at: ({region_center_x}, {region_center_y})")
    
    # Check where the cone tip is in the rotated region
    # The cone tip should be at (region_center_x, region_center_y)
    tip_pixel = rotated_region[region_center_y, region_center_x]
    print(f"  Cone tip pixel at rotation center: {tip_pixel}")
    print(f"  Tip is non-black: {np.any(tip_pixel > 10)}")
    
    # NEW CENTERING LOGIC (from our fix)
    cone_tip_x = region_center_x
    cone_tip_y = region_center_y
    
    target_center_x = rotated_region.shape[1] // 2
    target_center_y = rotated_region.shape[0] // 2
    
    translation_x = target_center_x - cone_tip_x
    translation_y = target_center_y - cone_tip_y
    
    print(f"  Target center: ({target_center_x}, {target_center_y})")
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
    
    # Check if cone tip is now at center
    final_center_pixel = centered_region[target_center_y, target_center_x]
    print(f"  Final center pixel: {final_center_pixel}")
    print(f"  Final center is non-black: {np.any(final_center_pixel > 10)}")
    
    # Visualize the process
    plt.figure(figsize=(18, 12))
    
    # 1. Original arena with cone overlay
    plt.subplot(2, 3, 1)
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
    plt.title('Original Arena with Cone')
    plt.axis('off')
    
    # 2. Conical region extracted
    plt.subplot(2, 3, 2)
    plt.imshow(conical_region)
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.title('Conical Region')
    plt.axis('off')
    
    # 3. Region with padding
    plt.subplot(2, 3, 3)
    plt.imshow(region_with_padding)
    plt.scatter(region_center_x, region_center_y, c='red', s=100, marker='x')
    plt.title('Region with Padding')
    plt.axis('off')
    
    # 4. After rotation
    plt.subplot(2, 3, 4)
    plt.imshow(rotated_region)
    plt.scatter(region_center_x, region_center_y, c='red', s=100, marker='x')
    plt.scatter(target_center_x, target_center_y, c='green', s=50, marker='o')
    plt.title('After Rotation')
    plt.axis('off')
    
    # 5. After centering
    plt.subplot(2, 3, 5)
    plt.imshow(centered_region)
    plt.scatter(target_center_x, target_center_y, c='green', s=100, marker='x')
    plt.title('After Centering')
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
    plt.scatter(final_center_x, final_center_y, c='green', s=100, marker='x')
    plt.title('Final View')
    plt.axis('off')
    
    plt.suptitle('Cone Extraction Debug Visualization')
    plt.tight_layout()
    plt.show()
    
    return np.any(final_center_pixel > 10)

if __name__ == "__main__":
    debug_cone_extraction()