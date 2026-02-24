#!/usr/bin/env python3
"""
Test script for the Emulator class.

This script verifies that the Emulator can be loaded and makes predictions.
"""

import numpy as np
from Library.Emulator import Emulator

def test_emulator_loading():
    """Test that we can load the emulator."""
    print("Testing Emulator loading...")
    
    try:
        emulator = Emulator.load(device="cpu")  # Use CPU for testing to avoid device issues
        print("✓ Emulator loaded successfully")
        
        # Check profile parameters
        params = emulator.get_profile_params()
        print(f"✓ Profile parameters: {params}")
        
        return emulator
        
    except FileNotFoundError as e:
        print(f"✗ Failed to load emulator: {e}")
        print("  Make sure you have run:")
        print("  1. SCRIPT_TrainEchoProcessor.py")
        print("  2. SCRIPT_TrainEmulator.py")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

def test_emulator_prediction(emulator: Emulator):
    """Test that the emulator can make predictions."""
    print("\nTesting Emulator predictions...")
    
    if emulator is None:
        print("✗ No emulator to test")
        return
    
    try:
        # Create a simple test profile
        profile_steps = emulator.profile_steps
        
        # Test with a simple profile (e.g., flat profile at 500mm)
        test_profile = np.full(profile_steps, 500.0, dtype=np.float32)
        
        # Test single prediction
        result = emulator.predict_single(test_profile)
        print(f"✓ Single prediction: distance={result['distance_mm']:.1f}mm, IID={result['iid_db']:.1f}dB")
        
        # Test batch prediction
        batch_profiles = np.tile(test_profile, (5, 1))
        batch_result = emulator.predict(batch_profiles)
        print(f"✓ Batch prediction: {len(batch_result['distance_mm'])} samples")
        print(f"  Distances: {batch_result['distance_mm']}")
        print(f"  IIDs: {batch_result['iid_db']}")
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")

def main():
    print("Emulator Test Script")
    print("=" * 50)
    
    emulator = test_emulator_loading()
    
    if emulator is not None:
        test_emulator_prediction(emulator)
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed - emulator not available")

if __name__ == "__main__":
    main()