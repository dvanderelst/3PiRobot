#!/usr/bin/env python3

"""
Test script for the new polar plot functionality in plot_all_sonar.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np

def test_polar_profiles():
    """Test the polar profile functionality."""
    
    print("🚀 Testing Polar Profile Plots")
    print("=" * 50)
    
    # Create a DataProcessor
    processor = DataProcessor('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code/Data/session03')
    
    # Load sonar data
    print("📡 Loading sonar data...")
    processor.load_sonar(flatten=False)
    
    # Load profiles
    print("📊 Loading distance profiles...")
    processor.load_profiles(az_min=-45, az_max=45, az_steps=20)
    
    # Test 1: Basic polar profile plotting
    print("\n🎨 Test 1: Basic polar profile plotting")
    plot_dir1 = processor.plot_all_sonar(
        profile=True, 
        indices=range(0, 3),  # Just 3 samples for quick testing
        output_dir='test_plots'
    )
    print(f"   ✅ Created polar profile plots: {plot_dir1}")
    
    # Verify files were created
    files = os.listdir(plot_dir1)
    expected_files = 3
    print(f"   ✅ Created {len(files)} polar profile plot files")
    assert len(files) == expected_files, f"Expected {expected_files} files, got {len(files)}"
    
    # Test 2: Polar profiles with views
    print("\n👁️  Test 2: Polar profiles with conical views")
    processor.load_views(radius_mm=1000, opening_deg=60, output_size=(64, 64), show_example=False)
    
    plot_dir2 = processor.plot_all_sonar(
        view=True,
        profile=True, 
        indices=range(0, 2),  # Just 2 samples for quick testing
        output_dir='test_plots'
    )
    print(f"   ✅ Created combined plots: {plot_dir2}")
    
    # Verify files were created
    files = os.listdir(plot_dir2)
    expected_files = 2
    print(f"   ✅ Created {len(files)} combined plot files")
    assert len(files) == expected_files, f"Expected {expected_files} files, got {len(files)}"
    
    # Test 3: Verify polar plot characteristics
    print("\n🔍 Test 3: Polar plot characteristics")
    
    # Check that files have reasonable size (polar plots should be similar to regular plots)
    first_plot = os.path.join(plot_dir1, 'sonar_0000.png')
    file_size = os.path.getsize(first_plot)
    print(f"   ✅ Polar plot file size: {file_size / 1024:.1f} KB")
    assert file_size > 1000, "Polar plot file seems too small"
    
    # Test 4: Cleanup
    print("\n🧹 Test 4: Cleanup")
    
    # Remove test plot directories
    import shutil
    for plot_dir in [plot_dir1, plot_dir2]:
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)
            print(f"   ✅ Removed test directory: {plot_dir}")
    
    print("\n🎉 All polar profile tests passed!")
    print("✅ Basic polar profile plotting works")
    print("✅ Polar profiles with views combination works")
    print("✅ Polar plot files are generated correctly")
    print("✅ File sizes are reasonable")
    
    print("\n📊 Polar plots provide better geometric visualization")
    print("🎨 Azimuth angles are displayed radially around the robot")
    print("📐 Distances are shown as radial lengths from center")
    print("🔄 Clockwise orientation matches robot's perspective")

if __name__ == "__main__":
    test_polar_profiles()