#!/usr/bin/env python3

"""
Comprehensive test script for data loading and processing.
This script tests the DataCollection and DataProcessor classes.
"""

import sys
import os
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

import numpy as np
import matplotlib.pyplot as plt
from Library import DataProcessor

def test_data_loading():
    """Test basic data loading functionality."""
    print("🧪 Testing data loading...")
    
    # Test with a single session
    sessions = ['session03']
    dc = DataProcessor.DataCollection(sessions)
    
    print(f"✅ Created DataCollection with {len(dc.processors)} processors")
    print(f"   Total samples: {dc._cache_metadata['total_samples']}")
    
    # Test sonar loading
    print("\n📡 Testing sonar loading...")
    sonar = dc.load_sonar(flatten=False)
    print(f"   Sonar shape: {sonar.shape}")
    print(f"   Sonar dtype: {sonar.dtype}")
    
    # Test flattened sonar
    sonar_flat = dc.load_sonar(flatten=True)
    print(f"   Flattened sonar shape: {sonar_flat.shape}")
    
    # Test field access
    print("\n📊 Testing field access...")
    sonar_distance = dc.get_field('sonar_package', 'corrected_distance')
    sonar_iid = dc.get_field('sonar_package', 'corrected_iid')
    
    print(f"   Distance shape: {sonar_distance.shape}")
    print(f"   IID shape: {sonar_iid.shape}")
    print(f"   Distance range: [{np.min(sonar_distance):.3f}, {np.max(sonar_distance):.3f}]")
    print(f"   IID range: [{np.min(sonar_iid):.3f}, {np.max(sonar_iid):.3f}]")
    
    # Test position data
    rob_x = dc.rob_x
    rob_y = dc.rob_y
    rob_yaw = dc.rob_yaw_deg
    
    print(f"\n📍 Position data:")
    print(f"   X range: [{np.min(rob_x):.1f}, {np.max(rob_x):.1f}]")
    print(f"   Y range: [{np.min(rob_y):.1f}, {np.max(rob_y):.1f}]")
    print(f"   Yaw range: [{np.min(rob_yaw):.1f}, {np.max(rob_yaw):.1f}]")
    
    return True

def test_views_loading():
    """Test conical views loading."""
    print("\n🎯 Testing conical views loading...")
    
    sessions = ['session03']
    dc = DataProcessor.DataCollection(sessions)
    
    # Test views with small size for speed
    views = dc.load_views(radius_mm=1500, opening_angle=90, output_size=(64, 64), show_example=False)
    
    print(f"   Views shape: {views.shape}")
    print(f"   Views dtype: {views.dtype}")
    print(f"   Memory usage: {views.nbytes / (1024*1024):.2f} MB")
    
    # Check that views are not all black
    non_black_pixels = np.any(views > 10, axis=(1, 2, 3))
    valid_views = np.sum(non_black_pixels)
    print(f"   Valid (non-black) views: {valid_views}/{len(views)}")
    
    return True

def test_profiles_loading():
    """Test distance profiles loading."""
    print("\n📊 Testing distance profiles loading...")
    
    sessions = ['session03']
    dc = DataProcessor.DataCollection(sessions)
    
    profiles, centers = dc.load_profiles(opening_angle=90, steps=20)
    
    print(f"   Profiles shape: {profiles.shape}")
    print(f"   Centers shape: {centers.shape}")
    print(f"   Profile range: [{np.nanmin(profiles):.1f}, {np.nanmax(profiles):.1f}]")
    
    # Check for NaN values
    nan_count = np.isnan(profiles).sum()
    print(f"   NaN values: {nan_count}/{profiles.size}")
    
    return True

def test_multi_session():
    """Test multi-session data collection."""
    print("\n📁 Testing multi-session data collection...")
    
    sessions = ['session03', 'session04']
    dc = DataProcessor.DataCollection(sessions)
    
    print(f"   Processors: {len(dc.processors)}")
    print(f"   Total samples: {dc._cache_metadata['total_samples']}")
    
    # Load sonar data
    sonar = dc.load_sonar()
    print(f"   Combined sonar shape: {sonar.shape}")
    
    # Test field access across sessions
    distances = dc.get_field('sonar_package', 'corrected_distance')
    print(f"   Combined distances: {len(distances)}")
    
    return True

def test_data_visualization():
    """Test basic data visualization."""
    print("\n📈 Testing data visualization...")
    
    sessions = ['session03']
    dc = DataProcessor.DataCollection(sessions)
    
    # Load data
    sonar_distance = dc.get_field('sonar_package', 'corrected_distance')
    sonar_iid = dc.get_field('sonar_package', 'corrected_iid')
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(sonar_distance, sonar_iid, alpha=0.5)
    plt.xlabel('Distance (m)')
    plt.ylabel('IID (dB)')
    plt.title('Sonar Distance vs IID')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('test/test_scatter_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved scatter plot to test/test_scatter_plot.png")
    
    # Create histogram
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(sonar_distance, bins=50, alpha=0.7)
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(sonar_iid, bins=50, alpha=0.7)
    plt.xlabel('IID (dB)')
    plt.ylabel('Frequency')
    plt.title('IID Distribution')
    
    plt.tight_layout()
    plt.savefig('test/test_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved histograms to test/test_histograms.png")
    
    return True

def test_cache_functionality():
    """Test cache functionality."""
    print("\n💾 Testing cache functionality...")
    
    sessions = ['session03']
    
    # Test with cache enabled
    dc1 = DataProcessor.DataCollection(sessions, cache_dir='./test_cache')
    sonar1 = dc1.load_sonar()
    
    # Test loading from cache
    dc2 = DataProcessor.DataCollection(sessions, cache_dir='./test_cache')
    sonar2 = dc2.load_sonar()
    
    # Verify data is identical
    data_identical = np.array_equal(sonar1, sonar2)
    print(f"   Cache data identical: {data_identical}")
    
    # Clean up
    dc2.clear_cache()
    
    return data_identical

def run_all_tests():
    """Run all tests."""
    print("🚀 Running comprehensive data loading tests...\n")
    
    tests = [
        test_data_loading,
        test_views_loading,
        test_profiles_loading,
        test_multi_session,
        test_data_visualization,
        test_cache_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result is True:
                print(f"✅ {test.__name__} PASSED\n")
                passed += 1
            else:
                print(f"❌ {test.__name__} FAILED (returned {result})\n")
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED with exception: {e}\n")
            failed += 1
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)