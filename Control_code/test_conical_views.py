#!/usr/bin/env python3

"""
Test script for conical view extraction functionality.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor, DataCollection
import numpy as np

def test_conical_view_extraction():
    """Test the conical view extraction functionality."""
    
    print("🚀 Testing Conical View Extraction")
    print("=" * 50)
    
    # Create data processor for session03
    print("📁 Loading data processor for session03...")
    processor = DataProcessor('session03')
    
    # Test basic functionality
    print("🧪 Testing basic methods...")
    
    # Test world to pixel conversion
    test_x, test_y = 0, 0
    px, py = processor.world_to_pixel(test_x, test_y)
    print(f"   World (0, 0) → Pixel ({px}, {py})")
    
    # Test pixel to world conversion
    wx, wy = processor.pixel_to_world(px, py)
    print(f"   Pixel ({px}, {py}) → World ({wx:.1f}, {wy:.1f})")
    
    # Test arena image loading
    arena_img = processor.load_arena_image()
    print(f"   Arena image shape: {arena_img.shape}")
    
    # Test single conical view extraction
    print("🎯 Testing single conical view extraction...")
    
    # Get first robot position
    rob_x = processor.rob_x[0]
    rob_y = processor.rob_y[0]
    rob_yaw = processor.rob_yaw_deg[0]
    
    print(f"   Robot position: ({rob_x:.1f}, {rob_y:.1f}) mm")
    print(f"   Robot yaw: {rob_yaw:.1f}°")
    
    # Extract conical view
    conical_view = processor.extract_conical_view(
        rob_x, rob_y, rob_yaw,
        radius_mm=1000,  # 1 meter radius
        opening_angle_deg=60,  # 60 degree opening
        output_size=(64, 64),  # Small output for testing
        visualize=True  # Show visualization
    )
    
    print(f"   Conical view shape: {conical_view.shape}")
    print(f"   Conical view dtype: {conical_view.dtype}")
    
    # Test collate_data with conical views
    print("🔄 Testing collate_data with conical views...")
    
    collated_data = processor.collate_data(
        az_min=-45,
        az_max=45,
        az_steps=20,
        extract_conical_views=True,
        conical_radius_mm=1000,
        conical_opening_deg=60,
        conical_output_size=(64, 64)
    )
    
    print("✅ Collation completed successfully!")
    print(f"   Available fields: {list(collated_data.keys())}")
    
    if 'conical_views' in collated_data:
        conical_views = collated_data['conical_views']
        print(f"   Conical views shape: {conical_views.shape}")
        print(f"   Number of views: {len(conical_views)}")
        print(f"   View dimensions: {conical_views.shape[1:]} (H, W, C)")
    
    # Test DataCollection with conical views
    print("📚 Testing DataCollection with conical views...")
    
    try:
        collection = DataCollection(['session03'], az_steps=20, extract_conical_views=True)
        
        # Get conical views
        all_conical_views = collection.get_conical_views()
        print(f"   Total conical views: {len(all_conical_views)}")
        print(f"   Shape: {all_conical_views.shape}")
        
        # Get other data for comparison
        sonar_data = collection.get_field('sonar_data')
        profiles = collection.get_field('profiles')
        
        print(f"   Sonar data shape: {sonar_data.shape}")
        print(f"   Profiles shape: {profiles.shape}")
        print(f"   Conical views shape: {all_conical_views.shape}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error in DataCollection: {e}")
        
    print("🎉 Conical view extraction test completed!")

if __name__ == "__main__":
    test_conical_view_extraction()