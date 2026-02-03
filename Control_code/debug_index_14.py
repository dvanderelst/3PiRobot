#!/usr/bin/env python3
"""
Debug script specifically for index 14 to understand the missing pixels issue.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt
import cv2

def debug_index_14():
    """Debug the specific issue with index 14."""
    
    print("🔍 Debugging Index 14 Issue")
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
    
    print(f"  Radius: {radius_mm}mm = {radius_px}px")
    print(f"  Arena dimensions: {arena_img.shape[1]} x {arena_img.shape[0]} pixels")
    
    # Check if cone would extend beyond arena boundaries
    extends_left = center_x - radius_px < 0
    extends_right = center_x + radius_px >= arena_img.shape[1]
    extends_top = center_y - radius_px < 0
    extends_bottom = center_y + radius_px >= arena_img.shape[0]
    
    print(f"  Extends beyond boundaries:")
    print(f"    Left: {extends_left}, Right: {extends_right}")
    print(f"    Top: {extends_top}, Bottom: {extends_bottom}")
    
    # Create conical mask
    conical_mask = processor.create_conical_mask(
        center_x, center_y, radius_px, opening_deg, rob_yaw
    )
    
    # Count pixels in mask
    total_mask_pixels = np.sum(conical_mask)
    print(f"  Conical mask has {total_mask_pixels} pixels")
    
    # Extract conical region
    conical_region = np.zeros_like(arena_img)
    conical_region[conical_mask] = arena_img[conical_mask]
    
    # Count non-black pixels in conical region
    conical_region_gray = np.mean(conical_region, axis=2)
    non_black_pixels = np.sum(conical_region_gray > 10)
    print(f"  Conical region has {non_black_pixels} non-black pixels")
    
    # Create padding and extract region
    padding = radius_px // 2
    min_x = max(0, center_x - radius_px - padding)
    max_x = min(arena_img.shape[1], center_x + radius_px + padding)
    min_y = max(0, center_y - radius_px - padding)
    max_y = min(arena_img.shape[0], center_y + radius_px + padding)
    
    print(f"  Extraction region: [{min_y}:{max_y}, {min_x}:{max_x}]")
    print(f"  Region size: {max_x - min_x} x {max_y - min_y}")
    
    region_with_padding = conical_region[min_y:max_y, min_x:max_x]
    region_center_x = center_x - min_x
    region_center_y = center_y - min_y
    
    # Count non-black pixels in extracted region
    region_gray = np.mean(region_with_padding, axis=2)
    region_non_black = np.sum(region_gray > 10)
    print(f"  Extracted region has {region_non_black} non-black pixels")
    
    # Rotation
    rotation_angle = 90 - rob_yaw
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
    
    # Count non-black pixels after rotation
    rotated_gray = np.mean(rotated_region, axis=2)
    rotated_non_black = np.sum(rotated_gray > 10)
    print(f"  After rotation: {rotated_non_black} non-black pixels")
    
    # Centering
    target_center_x = rotated_region.shape[1] // 2
    target_center_y = rotated_region.shape[0] // 2
    
    translation_x = target_center_x - region_center_x
    translation_y = target_center_y - region_center_y
    
    translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    centered_region = cv2.warpAffine(
        rotated_region,
        translation_matrix,
        (rotated_region.shape[1], rotated_region.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # Count non-black pixels after centering
    centered_gray = np.mean(centered_region, axis=2)
    centered_non_black = np.sum(centered_gray > 10)
    print(f"  After centering: {centered_non_black} non-black pixels")
    
    # Cropping
    crop_size = min(centered_region.shape[:2])
    start_x = (centered_region.shape[1] - crop_size) // 2
    start_y = (centered_region.shape[0] - crop_size) // 2
    cropped_region = centered_region[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    print(f"  Crop region: [{start_y}:{start_y+crop_size}, {start_x}:{start_x+crop_size}]")
    print(f"  Crop size: {crop_size} x {crop_size}")
    
    # Count non-black pixels after cropping
    cropped_gray = np.mean(cropped_region, axis=2)
    cropped_non_black = np.sum(cropped_gray > 10)
    print(f"  After cropping: {cropped_non_black} non-black pixels")
    
    # Final resize
    final_view = cv2.resize(cropped_region, output_size, interpolation=cv2.INTER_AREA)
    
    # Count non-black pixels in final view
    final_gray = np.mean(final_view, axis=2)
    final_non_black = np.sum(final_gray > 10)
    print(f"  Final view: {final_non_black} non-black pixels")
    
    # Calculate pixel loss
    pixel_loss = total_mask_pixels - final_non_black
    pixel_loss_percent = (pixel_loss / total_mask_pixels * 100) if total_mask_pixels > 0 else 0
    print(f"\nPixel Loss Analysis:")
    print(f"  Total pixels lost: {pixel_loss} ({pixel_loss_percent:.1f}%)")
    
    # Visualize the process
    plt.figure(figsize=(20, 12))
    
    # 1. Original arena with cone overlay
    plt.subplot(2, 3, 1)
    plt.imshow(arena_img)
    
    # Draw cone outline
    half_opening_rad = np.deg2rad(opening_deg / 2)
    orientation_rad = np.deg2rad(rob_yaw)
    lower_bound = orientation_rad - half_opening_rad
    upper_bound = orientation_rad + half_opening_rad
    cone_theta = np.linspace(lower_bound, upper_bound, 100)
    x_cone = center_x + radius_px * np.cos(cone_theta)
    y_cone = center_y - radius_px * np.sin(cone_theta)
    plt.plot(x_cone, y_cone, 'r-', linewidth=2)
    
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.title(f'Original Arena\nTotal mask pixels: {total_mask_pixels}')
    plt.axis('off')
    
    # 2. Conical region
    plt.subplot(2, 3, 2)
    plt.imshow(conical_region)
    plt.scatter(center_x, center_y, c='red', s=100, marker='x')
    plt.title(f'Conical Region\nNon-black pixels: {non_black_pixels}')
    plt.axis('off')
    
    # 3. Region with padding
    plt.subplot(2, 3, 3)
    plt.imshow(region_with_padding)
    plt.scatter(region_center_x, region_center_y, c='red', s=100, marker='x')
    plt.title(f'Region with Padding\nNon-black pixels: {region_non_black}')
    plt.axis('off')
    
    # 4. After rotation
    plt.subplot(2, 3, 4)
    plt.imshow(rotated_region)
    plt.scatter(region_center_x, region_center_y, c='red', s=100, marker='x')
    plt.title(f'After Rotation\nNon-black pixels: {rotated_non_black}')
    plt.axis('off')
    
    # 5. After centering
    plt.subplot(2, 3, 5)
    plt.imshow(centered_region)
    plt.scatter(target_center_x, target_center_y, c='red', s=100, marker='x')
    plt.title(f'After Centering\nNon-black pixels: {centered_non_black}')
    plt.axis('off')
    
    # 6. Final view
    plt.subplot(2, 3, 6)
    plt.imshow(final_view)
    final_center_x, final_center_y = final_view.shape[1] // 2, final_view.shape[0] // 2
    plt.scatter(final_center_x, final_center_y, c='red', s=100, marker='x')
    plt.title(f'Final View\nNon-black pixels: {final_non_black}\nLost: {pixel_loss} ({pixel_loss_percent:.1f}%)')
    plt.axis('off')
    
    plt.suptitle(f'Index {test_index} Pixel Loss Analysis')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    debug_index_14()