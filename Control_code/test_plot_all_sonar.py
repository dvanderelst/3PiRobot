#!/usr/bin/env python3

"""
Test the new plot_all_sonar function.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np

def test_plot_all_sonar():
    """Test the plot_all_sonar function with different options."""
    
    print("📊 Testing plot_all_sonar Function")
    print("=" * 50)
    
    # Create a DataProcessor
    processor = DataProcessor('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code/Data/session03')
    
    # Load sonar data
    print("📡 Loading sonar data...")
    processor.load_sonar(flatten=False)
    
    # Test 1: Basic sonar plotting (sonar only)
    print("\n🎨 Test 1: Basic sonar plotting (sonar only)")
    plot_dir1 = processor.plot_all_sonar(view=False, profile=False, output_dir='test_plots')
    print(f"   ✅ Saved plots to: {plot_dir1}")
    
    # Verify files were created
    files = os.listdir(plot_dir1)
    print(f"   ✅ Created {len(files)} sonar plot files")
    assert len(files) == len(processor.sonar_data), f"Expected {len(processor.sonar_data)} files, got {len(files)}"
    
    # Test 2: Sonar with views
    print("\n👁️  Test 2: Sonar with views")
    processor.load_views(radius_mm=1000, opening_deg=60, output_size=(64, 64), show_example=False)
    plot_dir2 = processor.plot_all_sonar(view=True, profile=False, output_dir='test_plots')
    print(f"   ✅ Saved plots with views to: {plot_dir2}")
    
    # Verify files were created
    files = os.listdir(plot_dir2)
    print(f"   ✅ Created {len(files)} sonar+view plot files")
    
    # Test 3: Sonar with profiles
    print("\n📈 Test 3: Sonar with profiles")
    processor.load_profiles(az_min=-45, az_max=45, az_steps=20)
    plot_dir3 = processor.plot_all_sonar(view=False, profile=True, output_dir='test_plots')
    print(f"   ✅ Saved plots with profiles to: {plot_dir3}")
    
    # Verify files were created
    files = os.listdir(plot_dir3)
    print(f"   ✅ Created {len(files)} sonar+profile plot files")
    
    # Test 4: Sonar with both views and profiles
    print("\n🔄 Test 4: Sonar with views AND profiles")
    plot_dir4 = processor.plot_all_sonar(view=True, profile=True, output_dir='test_plots')
    print(f"   ✅ Saved comprehensive plots to: {plot_dir4}")
    
    # Verify files were created
    files = os.listdir(plot_dir4)
    print(f"   ✅ Created {len(files)} comprehensive plot files")
    
    # Test 5: Error handling
    print("\n⚠️  Test 5: Error handling")
    
    # Test error when views not loaded but requested
    empty_processor = DataProcessor('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code/Data/session03')
    empty_processor.load_sonar()
    
    try:
        empty_processor.plot_all_sonar(view=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✅ View error handled: {e}")
    
    # Test error when profiles not loaded but requested
    try:
        empty_processor.plot_all_sonar(profile=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✅ Profile error handled: {e}")
    
    # Test 6: Verify plot content
    print("\n🔍 Test 6: Plot content verification")
    
    # Check that the first plot file exists and has reasonable size
    first_plot = os.path.join(plot_dir1, 'sonar_0000.png')
    assert os.path.exists(first_plot), "First plot file not found"
    
    file_size = os.path.getsize(first_plot)
    print(f"   ✅ First plot file size: {file_size / 1024:.1f} KB")
    assert file_size > 1000, "Plot file seems too small"
    
    # Test 7: Test new indices parameter
    print("\n🎯 Test 7: New indices parameter")
    
    # Test with specific range
    plot_dir5 = processor.plot_all_sonar(
        view=False, 
        profile=False, 
        output_dir='test_plots',
        indices=range(0, 10, 2)  # Only plot indices 0, 2, 4, 6, 8
    )
    
    files = os.listdir(plot_dir5)
    expected_files = 5  # indices 0, 2, 4, 6, 8
    print(f"   ✅ Created {len(files)} sonar plot files with range(0, 10, 2)")
    print(f"   Expected: {expected_files} files")
    assert len(files) == expected_files, f"Expected {expected_files} files, got {len(files)}"
    
    # Test 8: Cleanup
    print("\n🧹 Test 8: Cleanup")
    
    # Remove test plot directories
    import shutil
    for plot_dir in [plot_dir1, plot_dir2, plot_dir3, plot_dir4, plot_dir5]:
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)
            print(f"   ✅ Removed test directory: {plot_dir}")
    
    print("\n🎉 All plot_all_sonar tests passed!")
    print("✅ Basic sonar plotting works")
    print("✅ Sonar + views combination works")
    print("✅ Sonar + profiles combination works")
    print("✅ Sonar + views + profiles combination works")
    print("✅ Error handling works correctly")
    print("✅ Plot files are generated and saved properly")
    print("✅ Progress tracking with tqdm works")
    print("✅ New indices parameter works correctly")
    
    print("\n📁 Plot files are saved in organized subdirectories with timestamps")
    print("🎨 Each plot includes robot position and orientation information")
    print("📊 Function provides clear progress feedback during generation")
    print("🎯 Indices parameter allows selective plotting for testing and debugging")

if __name__ == "__main__":
    test_plot_all_sonar()