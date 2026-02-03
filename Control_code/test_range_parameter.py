#!/usr/bin/env python3

"""
Test script for the new range parameter in plot_all_sonar.
"""

import sys
import os
sys.path.insert(0, '.')
os.chdir('.')

from Library.DataProcessor import DataProcessor
import numpy as np

def test_range_parameter():
    """Test the range parameter functionality."""
    
    print("🚀 Testing Range Parameter in plot_all_sonar")
    print("=" * 50)
    
    # Create a DataProcessor
    processor = DataProcessor('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code/Data/session03')
    
    # Load sonar data
    print("📡 Loading sonar data...")
    processor.load_sonar(flatten=False)
    print(f"   Total sonar measurements: {len(processor.sonar_data)}")
    
    # Test 1: Range with step (e.g., range(0, 10, 2))
    print("\n🎨 Test 1: Range with step - range(0, 10, 2)")
    plot_dir1 = processor.plot_all_sonar(
        view=False, 
        profile=False, 
        output_dir='test_plots',
        indices=range(0, 10, 2)
    )
    print(f"   ✅ Saved plots to: {plot_dir1}")
    
    # Verify files were created
    files = os.listdir(plot_dir1)
    expected_files = 5  # indices 0, 2, 4, 6, 8
    print(f"   ✅ Created {len(files)} sonar plot files")
    print(f"   Expected: {expected_files} files (indices 0, 2, 4, 6, 8)")
    assert len(files) == expected_files, f"Expected {expected_files} files, got {len(files)}"
    
    # Test 2: Specific list of indices
    print("\n📋 Test 2: Specific indices - [0, 5, 10, 15]")
    plot_dir2 = processor.plot_all_sonar(
        view=False, 
        profile=False, 
        output_dir='test_plots',
        indices=[0, 5, 10, 15]
    )
    print(f"   ✅ Saved plots to: {plot_dir2}")
    
    # Verify files were created
    files = os.listdir(plot_dir2)
    expected_files = 4  # indices 0, 5, 10, 15
    print(f"   ✅ Created {len(files)} sonar plot files")
    print(f"   Expected: {expected_files} files (indices 0, 5, 10, 15)")
    assert len(files) == expected_files, f"Expected {expected_files} files, got {len(files)}"
    
    # Test 3: None (default behavior - all measurements)
    print("\n🔄 Test 3: None (default - all measurements)")
    plot_dir3 = processor.plot_all_sonar(
        view=False, 
        profile=False, 
        output_dir='test_plots',
        indices=None
    )
    print(f"   ✅ Saved plots to: {plot_dir3}")
    
    # Verify files were created
    files = os.listdir(plot_dir3)
    expected_files = len(processor.sonar_data)
    print(f"   ✅ Created {len(files)} sonar plot files")
    print(f"   Expected: {expected_files} files (all measurements)")
    assert len(files) == expected_files, f"Expected {expected_files} files, got {len(files)}"
    
    # Test 4: Cleanup
    print("\n🧹 Test 4: Cleanup")
    
    # Remove test plot directories
    import shutil
    for plot_dir in [plot_dir1, plot_dir2, plot_dir3]:
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)
            print(f"   ✅ Removed test directory: {plot_dir}")
    
    print("\n🎉 All range parameter tests passed!")
    print("✅ range(0, 10, 2) works correctly")
    print("✅ Specific index list works correctly")
    print("✅ None (default) works correctly")
    print("✅ Correct number of files created for each case")
    
    print("\n📊 Range parameter provides flexible control over which sonar measurements to plot")
    print("🎨 Useful for testing, debugging, and selective visualization")

if __name__ == "__main__":
    test_range_parameter()