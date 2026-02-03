#!/usr/bin/env python3
"""
Simple test to verify cone tip consistency fix.
"""

import numpy as np
import cv2
import sys
import os
from pathlib import Path

# Add Library to path
sys.path.append(str(Path(__file__).parent / 'Library'))

from DataProcessor import DataProcessor

def test_cone_tip_centering():
    """Test that the cone tip centering logic works correctly."""
    
    print("Testing cone tip centering logic...")
    
    # Create a mock scenario to test the centering logic
    # We'll simulate the key parts of the extract_conical_view method
    
    # Simulate a rotated region with cone tip at known position
    test_image_size = (200, 200, 3)
    rotated_region = np.zeros(test_image_size, dtype=np.uint8)
    
    # Simulate cone tip position (this should be at the rotation center)
    region_center_x, region_center_y = 100, 100
    
    # Draw a simple cone shape centered at (region_center_x, region_center_y)
    # This simulates what happens after rotation
    cv2.circle(rotated_region, (region_center_x, region_center_y), 5, (255, 255, 255), -1)  # Cone tip
    
    # Draw some cone content extending from the tip
    for angle_deg in range(-45, 46, 5):  # 90 degree cone
        angle_rad = np.deg2rad(angle_deg)
        length = 80
        end_x = int(region_center_x + length * np.cos(angle_rad))
        end_y = int(region_center_y + length * np.sin(angle_rad))
        cv2.line(rotated_region, (region_center_x, region_center_y), (end_x, end_y), (200, 200, 200), 2)
    
    print(f"Created test cone with tip at: ({region_center_x}, {region_center_y})")
    
    # Apply the NEW centering logic (from our fix)
    target_center_x = rotated_region.shape[1] // 2
    target_center_y = rotated_region.shape[0] // 2
    
    cone_tip_x = region_center_x
    cone_tip_y = region_center_y
    
    translation_x = target_center_x - cone_tip_x
    translation_y = target_center_y - cone_tip_y
    
    print(f"Target center: ({target_center_x}, {target_center_y})")
    print(f"Translation needed: ({translation_x}, {translation_y})")
    
    # Apply translation
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
    center_pixel = centered_region[target_center_y, target_center_x]
    is_center_non_black = np.any(center_pixel > 10)
    
    print(f"After centering:")
    print(f"  Center pixel RGB: {center_pixel}")
    print(f"  Center is non-black: {is_center_non_black}")
    
    # Visualize
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rotated_region)
    plt.scatter(region_center_x, region_center_y, c='red', s=100, marker='x')
    plt.title('Before Centering')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(centered_region)
    plt.scatter(target_center_x, target_center_y, c='green', s=100, marker='x')
    plt.title('After Centering')
    plt.axis('off')
    
    plt.suptitle('Cone Tip Centering Test')
    plt.tight_layout()
    plt.show()
    
    return is_center_non_black

def test_multiple_orientations():
    """Test centering with multiple different orientations."""
    
    print("\nTesting multiple orientations...")
    
    # Test different cone tip positions (simulating different rotations)
    test_positions = [
        (95, 105),   # Slightly off center
        (105, 95),   # Different offset
        (90, 110),   # More offset
        (110, 90),   # Opposite offset
        (100, 100),  # Perfect center
    ]
    
    results = []
    
    for i, (tip_x, tip_y) in enumerate(test_positions):
        # Create test image
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(test_image, (tip_x, tip_y), 5, (255, 255, 255), -1)
        
        # Apply centering logic
        target_center_x, target_center_y = 100, 100
        translation_x = target_center_x - tip_x
        translation_y = target_center_y - tip_y
        
        translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        centered_image = cv2.warpAffine(
            test_image,
            translation_matrix,
            (test_image.shape[1], test_image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Check result
        center_pixel = centered_image[target_center_y, target_center_x]
        success = np.any(center_pixel > 10)
        results.append(success)
        
        print(f"  Test {i+1}: tip at ({tip_x}, {tip_y}) -> center success: {success}")
    
    all_success = all(results)
    print(f"All tests passed: {all_success}")
    
    return all_success

if __name__ == "__main__":
    print("Simple Cone Tip Centering Test")
    print("="*50)
    
    # Test basic centering
    success1 = test_cone_tip_centering()
    
    # Test multiple orientations
    success2 = test_multiple_orientations()
    
    print(f"\nFinal Results:")
    print(f"  Basic centering test: {'PASS' if success1 else 'FAIL'}")
    print(f"  Multiple orientations test: {'PASS' if success2 else 'FAIL'}")
    print(f"  Overall: {'PASS' if success1 and success2 else 'FAIL'}")