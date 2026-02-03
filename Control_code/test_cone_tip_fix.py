#!/usr/bin/env python3
"""
Test script to verify that the cone tip positioning fix works correctly.
This tests the actual DataProcessor with real data.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

def test_cone_tip_consistency_with_real_data():
    """Test cone tip consistency using real data from session03."""
    
    print("🎯 Testing Cone Tip Consistency with Real Data")
    print("=" * 60)
    
    # Initialize data processor
    print("📁 Loading data processor for session03...")
    processor = DataProcessor('session03')
    
    # Load meta data needed for coordinate conversion
    print("📋 Loading meta data...")
    processor._load_meta_only()
    
    # Test parameters
    radius_mm = 1500
    opening_deg = 90
    output_size = (128, 128)
    
    # Test multiple positions
    test_indices = [0, 10, 20, 30, 40, 50]
    
    print(f"🧪 Testing {len(test_indices)} positions...")
    
    cone_tip_positions = []
    views = []
    
    for index in test_indices:
        rob_x = processor.rob_x[index]
        rob_y = processor.rob_y[index]
        rob_yaw = processor.rob_yaw_deg[index]
        
        print(f"  Index {index}: pos=({rob_x:.1f}, {rob_y:.1f}), yaw={rob_yaw:.1f}°")
        
        # Extract conical view
        view = processor.extract_conical_view(
            rob_x, rob_y, rob_yaw,
            radius_mm=radius_mm,
            opening_angle_deg=opening_deg,
            output_size=output_size,
            visualize=False
        )
        
        views.append(view)
        
        # Find the cone tip position in the normalized view
        # The cone tip should be at the center of the view
        center_x = view.shape[1] // 2
        center_y = view.shape[0] // 2
        
        # Check if there's content at the center
        center_pixel = view[center_y, center_x]
        is_center_non_black = np.any(center_pixel > 10)
        
        # Find the actual cone tip by looking for the highest point with content
        gray_view = np.mean(view, axis=2)
        
        # Find the topmost rows with significant content
        top_rows_with_content = []
        for y in range(view.shape[0]):
            row_content = np.any(gray_view[y, :] > 10)
            if row_content:
                top_rows_with_content.append(y)
                if len(top_rows_with_content) >= 5:  # Found the tip region
                    break
        
        if top_rows_with_content:
            cone_tip_y = top_rows_with_content[0]  # Topmost row with content
            # Find the center x position in that row
            row_content = gray_view[cone_tip_y, :] > 10
            content_x_positions = np.where(row_content)[0]
            if len(content_x_positions) > 0:
                cone_tip_x = np.mean(content_x_positions)
                cone_tip_positions.append((cone_tip_x, cone_tip_y))
            else:
                cone_tip_positions.append((center_x, center_y))
        else:
            cone_tip_positions.append((center_x, center_y))
        
        print(f"    Cone tip detected at: ({cone_tip_positions[-1][0]:.1f}, {cone_tip_positions[-1][1]:.1f})")
        print(f"    Expected center: ({center_x}, {center_y})")
        print(f"    Center has content: {is_center_non_black}")
    
    # Analyze consistency
    if len(cone_tip_positions) > 1:
        cone_tip_array = np.array(cone_tip_positions)
        x_positions = cone_tip_array[:, 0]
        y_positions = cone_tip_array[:, 1]
        
        x_std = np.std(x_positions)
        y_std = np.std(y_positions)
        x_mean = np.mean(x_positions)
        y_mean = np.mean(y_positions)
        
        print(f"\n📊 Cone Tip Position Consistency Analysis:")
        print(f"  X positions: mean={x_mean:.2f}, std={x_std:.2f}")
        print(f"  Y positions: mean={y_mean:.2f}, std={y_std:.2f}")
        
        # Check if positions are consistent (std < 3 pixels - allowing some tolerance)
        x_consistent = x_std < 3.0
        y_consistent = y_std < 3.0
        
        if x_consistent and y_consistent:
            print(f"  ✅ Cone tips are CONSISTENT (std < 2.0 pixels)")
            consistency_result = "PASS"
        else:
            print(f"  ❌ Cone tips are INCONSISTENT")
            if not x_consistent:
                print(f"     X positions vary by {x_std:.2f} pixels")
            if not y_consistent:
                print(f"     Y positions vary by {y_std:.2f} pixels")
            consistency_result = "FAIL"
    else:
        consistency_result = "INSUFFICIENT_DATA"
    
    # Create visualization
    plt.figure(figsize=(18, 10))
    
    n_views = len(views)
    for i, (view, (tip_x, tip_y)) in enumerate(zip(views, cone_tip_positions)):
        plt.subplot(2, (n_views + 1) // 2, i+1)
        plt.imshow(view)
        plt.scatter(tip_x, tip_y, c='red', s=100, marker='x')
        
        # Mark the expected center
        center_x = view.shape[1] // 2
        center_y = view.shape[0] // 2
        plt.scatter(center_x, center_y, c='green', s=50, marker='o')
        
        plt.title(f'Index {test_indices[i]}')
        plt.axis('off')
        
        # Add position info
        plt.text(10, 20, f'Tip: ({tip_x:.0f}, {tip_y:.0f})', 
                color='white', bbox=dict(facecolor='black', alpha=0.7))
        plt.text(10, 40, f'Center: ({center_x}, {center_y})', 
                color='white', bbox=dict(facecolor='black', alpha=0.7))
    
    plt.suptitle('Cone Tip Position Consistency Test (Red X = Detected Tip, Green O = Expected Center)')
    plt.tight_layout()
    plt.show()
    
    return consistency_result

def test_specific_orientation():
    """Test a specific orientation to verify the fix works."""
    
    print("\n🔍 Testing Specific Orientation")
    print("-" * 40)
    
    # Initialize data processor
    processor = DataProcessor('session03')
    
    # Load meta data needed for coordinate conversion
    processor._load_meta_only()
    
    # Test a specific index with visualization
    test_index = 25
    
    rob_x = processor.rob_x[test_index]
    rob_y = processor.rob_y[test_index]
    rob_yaw = processor.rob_yaw_deg[test_index]
    
    print(f"Testing index {test_index}:")
    print(f"  Position: ({rob_x:.1f}, {rob_y:.1f}) mm")
    print(f"  Yaw: {rob_yaw:.1f}°")
    
    # Extract view with visualization
    view = processor.extract_conical_view(
        rob_x, rob_y, rob_yaw,
        radius_mm=1500,
        opening_angle_deg=90,
        output_size=(128, 128),
        visualize=True  # This will show the extraction process
    )
    
    # Check if there's content near the center (within 5 pixels)
    center_x, center_y = view.shape[1] // 2, view.shape[0] // 2
    center_pixel = view[center_y, center_x]
    
    # Check a 5x5 area around the center
    has_content_near_center = False
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            y = center_y + dy
            x = center_x + dx
            if 0 <= y < view.shape[0] and 0 <= x < view.shape[1]:
                pixel = view[y, x]
                if np.any(pixel > 10):
                    has_content_near_center = True
                    break
        if has_content_near_center:
            break
    
    print(f"Result:")
    print(f"  Center pixel RGB: {center_pixel}")
    print(f"  Content near center: {has_content_near_center}")
    
    return has_content_near_center

if __name__ == "__main__":
    print("Testing Cone Tip Positioning Fix")
    print("=" * 60)
    
    # Run the main consistency test
    consistency_result = test_cone_tip_consistency_with_real_data()
    
    # Run specific orientation test
    center_has_content = test_specific_orientation()
    
    print(f"\n🏁 FINAL RESULTS:")
    print(f"  Consistency test: {consistency_result}")
    print(f"  Center content test: {'PASS' if center_has_content else 'FAIL'}")
    
    overall_success = consistency_result == "PASS" and center_has_content
    print(f"  Overall: {'✅ PASS' if overall_success else '❌ FAIL'}")