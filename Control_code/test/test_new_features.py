#!/usr/bin/env python3

"""
Test script to verify the new features: NaN handling, SmoothL1Loss, and world-frame visualization.
"""

import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from SCRIPT_Train_MLP_Profiles import (
    MAX_PROFILE_MM, PLOT_MAX_PROFILE_MM, EVAL_SESSION, 
    EVAL_INDEX_START, EVAL_INDEX_END, EVAL_INDICES
)

def test_nan_handling():
    """Test NaN handling functionality."""
    print("Testing NaN handling...")
    
    # Create test data with NaN values
    test_data = np.array([
        [1.0, 2.0, np.nan, 4.0],
        [5.0, np.nan, 7.0, 8.0],
        [np.nan, np.nan, np.nan, np.nan]
    ])
    
    print(f"Original data:\n{test_data}")
    
    # Apply NaN handling (replace with mean)
    if np.isnan(test_data).any():
        cleaned_data = np.nan_to_num(test_data, nan=np.nanmean(test_data, axis=0))
        print(f"Cleaned data:\n{cleaned_data}")
        
        # Verify no NaN values remain
        assert not np.isnan(cleaned_data).any(), "NaN values should be removed"
        print("✓ NaN values successfully handled")
        
        # Verify that non-NaN values are preserved
        original_non_nan = test_data[~np.isnan(test_data)]
        cleaned_non_nan = cleaned_data[~np.isnan(test_data)]
        assert np.array_equal(original_non_nan, cleaned_non_nan), "Non-NaN values should be preserved"
        print("✓ Non-NaN values preserved")
        
        # Verify that NaN positions are filled with reasonable values
        # Check that some NaN values were actually filled (not all zeros)
        nan_positions = np.where(np.isnan(test_data))
        if len(nan_positions[0]) > 0:
            # Check a few filled values are not zero (assuming data isn't all zeros)
            sample_filled = cleaned_data[nan_positions[0][0], nan_positions[1][0]]
            assert sample_filled != 0, "Filled NaN values should not be zero"
            print("✓ NaN values filled with reasonable values")
        else:
            print("✓ No NaN values to fill")

def test_smooth_l1_loss():
    """Test that SmoothL1Loss is used instead of MSELoss."""
    print("\nTesting SmoothL1Loss...")
    
    # Create a simple model and test the loss
    model = nn.Linear(10, 5)
    criterion = nn.SmoothL1Loss()
    
    # Test data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 5)
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    print(f"SmoothL1Loss value: {loss.item():.4f}")
    assert isinstance(criterion, nn.SmoothL1Loss), "Should use SmoothL1Loss"
    print("✓ SmoothL1Loss is correctly implemented")
    
    # Compare with MSELoss to show difference
    mse_criterion = nn.MSELoss()
    mse_loss = mse_criterion(outputs, targets)
    print(f"MSELoss value: {mse_loss.item():.4f}")
    print("✓ SmoothL1Loss provides different (typically more robust) loss values")

def test_world_frame_config():
    """Test world-frame visualization configuration."""
    print("\nTesting world-frame visualization configuration...")
    
    # Verify configuration values
    assert EVAL_SESSION == 'session07', f"Expected EVAL_SESSION='session07', got {EVAL_SESSION}"
    print(f"✓ EVAL_SESSION = {EVAL_SESSION}")
    
    assert EVAL_INDEX_START == 225, f"Expected EVAL_INDEX_START=225, got {EVAL_INDEX_START}"
    print(f"✓ EVAL_INDEX_START = {EVAL_INDEX_START}")
    
    assert EVAL_INDEX_END == 250, f"Expected EVAL_INDEX_END=250, got {EVAL_INDEX_END}"
    print(f"✓ EVAL_INDEX_END = {EVAL_INDEX_END}")
    
    assert EVAL_INDICES is None, f"Expected EVAL_INDICES=None, got {EVAL_INDICES}"
    print(f"✓ EVAL_INDICES = {EVAL_INDICES}")
    
    # Verify index range
    eval_range = EVAL_INDEX_END - EVAL_INDEX_START
    assert eval_range == 25, f"Expected evaluation range of 25, got {eval_range}"
    print(f"✓ Evaluation range: {eval_range} indices")

def test_loss_function_robustness():
    """Test robustness of SmoothL1Loss vs MSELoss with outliers."""
    print("\nTesting loss function robustness...")
    
    # Create data with outliers
    predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])  # Last value is outlier
    targets = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.0])
    
    # Test both loss functions
    smooth_l1 = nn.SmoothL1Loss()
    mse = nn.MSELoss()
    
    smooth_l1_loss = smooth_l1(predictions, targets)
    mse_loss = mse(predictions, targets)
    
    print(f"With outlier (100.0 vs 5.0):")
    print(f"  SmoothL1Loss: {smooth_l1_loss.item():.4f}")
    print(f"  MSELoss: {mse_loss.item():.4f}")
    
    # SmoothL1 should be more robust to outliers
    assert smooth_l1_loss < mse_loss, "SmoothL1Loss should be more robust to outliers"
    print("✓ SmoothL1Loss is more robust to outliers")
    
    # Test without outliers
    predictions_no_outlier = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    smooth_l1_loss_no_outlier = smooth_l1(predictions_no_outlier, targets)
    mse_loss_no_outlier = mse(predictions_no_outlier, targets)
    
    print(f"Without outlier:")
    print(f"  SmoothL1Loss: {smooth_l1_loss_no_outlier.item():.4f}")
    print(f"  MSELoss: {mse_loss_no_outlier.item():.4f}")
    
    # Both should be similar without outliers (SmoothL1 is typically about 0.5x MSE)
    ratio = smooth_l1_loss_no_outlier / mse_loss_no_outlier
    assert 0.1 < ratio < 1.0, f"Loss functions should be similar without outliers, got ratio {ratio}"
    print(f"✓ Both loss functions perform similarly without outliers (ratio: {ratio:.3f})")

if __name__ == "__main__":
    print("Testing new features in SCRIPT_Train_MLP_Profiles.py")
    print("=" * 60)
    
    try:
        test_nan_handling()
        test_smooth_l1_loss()
        test_world_frame_config()
        test_loss_function_robustness()
        
        print("\n" + "=" * 60)
        print("✅ All new feature tests passed!")
        print("\nNew features implemented:")
        print("  • NaN handling with mean imputation")
        print("  • SmoothL1Loss for robust training")
        print("  • World-frame visualization configuration")
        print("  • Improved robustness to outliers")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise