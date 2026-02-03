#!/usr/bin/env python3
"""
Test script to verify that cone tip positions are consistent across different views.
This script extracts conical views from multiple robot positions and checks that
the cone tip (robot position) is always at the same normalized position.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the Library directory to Python path
sys.path.append(str(Path(__file__).parent / 'Library'))

from DataProcessor import DataProcessor
from DataStorage import DataReader

def test_cone_tip_consistency():
    """Test that cone tips are consistently positioned in normalized views."""
    
    # Use a sample session for testing
    test_session = 'Data/session03'
    
    # Convert to absolute path
    test_session = str(Path(__file__).parent / test_session)
    
    if not Path(test_session).exists():
        print(f"Test session not found at {test_session}")
        print("Available sessions:")
        data_dir = Path(__file__).parent / 'Data'
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.startswith('session'):
                print(f"  {item.name}")
        return
    
    print(f"Testing cone tip consistency with session: {test_session}")
    
    # Initialize data processor
    data_reader = DataReader(test_session)
    processor = DataProcessor(data_reader)
    
    # Load a few sample views for testing
    test_indices = [0, 10, 20, 30, 40]  # Test different positions
    
    # Parameters for conical views
    radius_mm = 1500
    opening_deg = 90
    output_size = (128, 128)
    
    print(f"Extracting conical views for indices: {test_indices}")
    
    # Extract views and collect cone tip positions
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
            opening_deg=opening_deg,
            output_size=output_size,
            visualize=False
        )
        
        views.append(view)
        
        # Find the cone tip in the normalized view
        # The cone tip should be at the center, but let's verify
        center_x = view.shape[1] // 2
        center_y = view.shape[0] // 2
        
        # Check if there's content at the center
        center_pixel = view[center_y, center_x]
        is_center_non_black = np.any(center_pixel > 10)
        
        # Find the actual cone tip by looking for the "tip" pattern
        # The cone tip should be the point where the cone content starts
        # We'll look for the highest point with significant content
        gray_view = np.mean(view, axis=2)
        
        # Find the topmost row with significant content (this should be the cone tip)
        top_rows_with_content = []
        for y in range(view.shape[0]):
            row_content = np.any(gray_view[y, :] > 10)
            if row_content:
                top_rows_with_content.append(y)
                if len(top_rows_with_content) >= 3:  # Found the tip region
                    break
        
        if top_rows_with_content:
            cone_tip_y = top_rows_with_content[0]  # Topmost row with content
            # Find the center x position in that row
            row_content = gray_view[cone_tip_y, :] > 10
            content_x_positions = np.where(row_content)[0]
            if len(content_x_positions) > 0:
                cone_tip_x = np.mean(content_x_positions)
                cone_tip_positions.append((cone_tip_x, cone_tip_y))
                print(f"    Cone tip at: ({cone_tip_x:.1f}, {cone_tip_y:.1f})")
            else:
                cone_tip_positions.append((center_x, center_y))
                print(f"    Cone tip at center: ({center_x}, {center_y})")
        else:
            cone_tip_positions.append((center_x, center_y))
            print(f"    Cone tip at center: ({center_x}, {center_y})")
    
    # Analyze consistency
    if len(cone_tip_positions) > 1:
        cone_tip_array = np.array(cone_tip_positions)
        x_positions = cone_tip_array[:, 0]
        y_positions = cone_tip_array[:, 1]
        
        x_std = np.std(x_positions)
        y_std = np.std(y_positions)
        x_mean = np.mean(x_positions)
        y_mean = np.mean(y_positions)
        
        print(f"\nCone Tip Position Consistency Analysis:")
        print(f"  X positions: mean={x_mean:.2f}, std={x_std:.2f}")
        print(f"  Y positions: mean={y_mean:.2f}, std={y_std:.2f}")
        
        # Check if positions are consistent (std < 1 pixel)
        x_consistent = x_std < 1.0
        y_consistent = y_std < 1.0
        
        if x_consistent and y_consistent:
            print(f"  ✅ Cone tips are CONSISTENT (std < 1.0 pixel)")
        else:
            print(f"  ❌ Cone tips are INCONSISTENT")
            if not x_consistent:
                print(f"     X positions vary by {x_std:.2f} pixels")
            if not y_consistent:
                print(f"     Y positions vary by {y_std:.2f} pixels")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Plot each view with cone tip marked
    n_views = len(views)
    for i, (view, (tip_x, tip_y)) in enumerate(zip(views, cone_tip_positions)):
        plt.subplot(1, n_views, i+1)
        plt.imshow(view)
        plt.scatter(tip_x, tip_y, c='red', s=100, marker='x')
        plt.title(f'View {test_indices[i]}')
        plt.axis('off')
        
        # Add text showing position
        plt.text(10, 20, f'Tip: ({tip_x:.0f}, {tip_y:.0f})', 
                color='white', bbox=dict(facecolor='black', alpha=0.7))
    
    plt.suptitle('Cone Tip Position Consistency Test')
    plt.tight_layout()
    plt.show()
    
    return cone_tip_positions

def test_rotation_precision():
    """Test the precision of the rotation and centering logic."""
    
    print("\n" + "="*60)
    print("Testing Rotation and Centering Precision")
    print("="*60)
    
    # Create a simple test case
    test_session = 'Data/session03'
    
    # Convert to absolute path
    test_session = str(Path(__file__).parent / test_session)
    
    if not Path(test_session).exists():
        print(f"Test session not found at {test_session}")
        return
    
    # Initialize data processor
    data_reader = DataReader(test_session)
    processor = DataProcessor(data_reader)
    
    # Test with a specific index
    test_index = 10
    
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
        opening_deg=90,
        output_size=(128, 128),
        visualize=True
    )
    
    # Check the center pixel
    center_x, center_y = view.shape[1] // 2, view.shape[0] // 2
    center_pixel = view[center_y, center_x]
    
    print(f"Center pixel RGB: {center_pixel}")
    print(f"Center is non-black: {np.any(center_pixel > 10)}")

if __name__ == "__main__":
    print("Testing Cone Tip Consistency in Conical Views")
    print("="*60)
    
    # Run the main test
    cone_tip_positions = test_cone_tip_consistency()
    
    # Run rotation precision test
    test_rotation_precision()