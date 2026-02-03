#!/usr/bin/env python3
"""
Test script to check the load_views visualization issue.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

def test_load_views_visualization():
    """Test the load_views method with visualization."""
    
    print("🎯 Testing load_views visualization")
    print("=" * 50)
    
    # Initialize data processor
    processor = DataProcessor('session03')
    
    # Load views with visualization for first 5 indices
    print("Loading views with visualization for indices 0-4...")
    
    try:
        views = processor.load_views(
            radius_mm=2500,
            opening_deg=90,
            output_size=(256, 256),
            plot_indices=range(5),  # This will show visualizations
            indices=range(5)  # Only process these indices
        )
        
        # Note: load_views with plot_indices might not return views, so check
        if views is not None:
            print(f"Successfully loaded {len(views)} views")
            print(f"View shape: {views[0].shape}")
        else:
            print("Views not returned (expected when using plot_indices)")
            # Load views again without plot_indices to get the actual views
            views = processor.load_views(
                radius_mm=2500,
                opening_deg=90,
                output_size=(256, 256),
                indices=range(5)
            )
            print(f"Successfully loaded {len(views)} views")
            print(f"View shape: {views[0].shape}")
        
        # Analyze the cone tip positions in the final views
        cone_tip_positions = []
        
        for i, view in enumerate(views):
            # Find cone tip by looking for highest point with content
            gray_view = np.mean(view, axis=2)
            
            # Find topmost rows with content
            for y in range(view.shape[0]):
                row_content = np.any(gray_view[y, :] > 10)
                if row_content:
                    # Find center x position in this row
                    row_content_mask = gray_view[y, :] > 10
                    content_x_positions = np.where(row_content_mask)[0]
                    if len(content_x_positions) > 0:
                        cone_tip_x = np.mean(content_x_positions)
                        cone_tip_positions.append((cone_tip_x, y))
                        print(f"  View {i}: cone tip at ({cone_tip_x:.1f}, {y})")
                    break
        
        # Analyze consistency
        if len(cone_tip_positions) > 1:
            cone_tip_array = np.array(cone_tip_positions)
            x_positions = cone_tip_array[:, 0]
            y_positions = cone_tip_array[:, 1]
            
            x_std = np.std(x_positions)
            y_std = np.std(y_positions)
            x_mean = np.mean(x_positions)
            y_mean = np.mean(y_positions)
            
            print(f"\nCone Tip Position Analysis:")
            print(f"  X positions: mean={x_mean:.2f}, std={x_std:.2f}")
            print(f"  Y positions: mean={y_mean:.2f}, std={y_std:.2f}")
            
            # Check consistency
            x_consistent = x_std < 3.0
            y_consistent = y_std < 3.0
            
            if x_consistent and y_consistent:
                print(f"  ✅ Cone tips are CONSISTENT")
            else:
                print(f"  ❌ Cone tips are INCONSISTENT")
        
        return cone_tip_positions
        
    except Exception as e:
        print(f"Error in load_views: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    test_load_views_visualization()