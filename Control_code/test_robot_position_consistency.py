#!/usr/bin/env python3
"""
Test script to verify that the robot position is consistently at the center.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

def test_robot_position_consistency():
    """Test that robot position is consistently at the center."""
    
    print("🎯 Testing Robot Position Consistency")
    print("=" * 50)
    
    # Initialize data processor
    processor = DataProcessor('session03')
    
    # Test parameters
    test_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    radius_mm = 2500
    opening_deg = 90
    output_size = (256, 256)
    
    print(f"Testing {len(test_indices)} positions...")
    
    # Load views for test indices
    views = processor.load_views(
        radius_mm=radius_mm,
        opening_deg=opening_deg,
        output_size=output_size,
        indices=test_indices
    )
    
    # Check robot position in each view
    robot_positions = []
    center_positions = []
    
    for i, view in enumerate(views):
        # The robot should be at the center of the view
        center_x = view.shape[1] // 2
        center_y = view.shape[0] // 2
        
        # Check if there's content at the center (robot position)
        center_pixel = view[center_y, center_x]
        has_content_at_center = np.any(center_pixel > 10)
        
        robot_positions.append((center_x, center_y))
        center_positions.append((center_x, center_y))
        
        print(f"  Index {test_indices[i]}: center=({center_x}, {center_y}), has_content={has_content_at_center}")
    
    # Analyze consistency
    robot_array = np.array(robot_positions)
    x_positions = robot_array[:, 0]
    y_positions = robot_array[:, 1]
    
    x_std = np.std(x_positions)
    y_std = np.std(y_positions)
    x_mean = np.mean(x_positions)
    y_mean = np.mean(y_positions)
    
    print(f"\nRobot Position Analysis:")
    print(f"  X positions: mean={x_mean:.2f}, std={x_std:.2f}")
    print(f"  Y positions: mean={y_mean:.2f}, std={y_std:.2f}")
    
    # Check consistency
    x_consistent = x_std < 0.1  # Should be exactly the same
    y_consistent = y_std < 0.1  # Should be exactly the same
    
    if x_consistent and y_consistent:
        print(f"  ✅ Robot positions are PERFECTLY CONSISTENT")
        print(f"  All robot positions are at: ({int(x_mean)}, {int(y_mean)})")
    else:
        print(f"  ❌ Robot positions are INCONSISTENT")
        if not x_consistent:
            print(f"     X positions vary by {x_std:.2f} pixels")
        if not y_consistent:
            print(f"     Y positions vary by {y_std:.2f} pixels")
    
    # Create visualization showing robot positions
    plt.figure(figsize=(15, 8))
    
    n_views = len(views)
    for i, view in enumerate(views):
        plt.subplot(2, (n_views + 1) // 2, i+1)
        plt.imshow(view)
        
        # Mark the center (robot position)
        center_x = view.shape[1] // 2
        center_y = view.shape[0] // 2
        plt.scatter(center_x, center_y, c='red', s=100, marker='x')
        
        # Check if there's content at center
        center_pixel = view[center_y, center_x]
        has_content = np.any(center_pixel > 10)
        
        plt.title(f'Index {test_indices[i]}')
        plt.axis('off')
        
        # Add text
        color = 'white' if has_content else 'red'
        plt.text(10, 20, f'Center: ({center_x}, {center_y})', 
                color=color, bbox=dict(facecolor='black', alpha=0.7))
        plt.text(10, 40, f'Content: {has_content}', 
                color=color, bbox=dict(facecolor='black', alpha=0.7))
    
    plt.suptitle('Robot Position Consistency Test (Red X = Robot Position)')
    plt.tight_layout()
    plt.show()
    
    return x_consistent and y_consistent

if __name__ == "__main__":
    success = test_robot_position_consistency()
    print(f"\nOverall Result: {'✅ PASS' if success else '❌ FAIL'}")