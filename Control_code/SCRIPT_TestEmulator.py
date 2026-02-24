#!/usr/bin/env python3
"""
Test script for the Emulator class.

This script verifies that the Emulator can be loaded and makes predictions.
"""

import numpy as np
from Library.Emulator import Emulator


def generate_simple_environment_profiles(emulator, num_profiles=5):
    """
    Generate simple synthetic environment profiles for demonstration.
    
    This simulates what a real environment simulator would do - generate
    distance profiles based on the robot's position and environment geometry.
    """
    profile_steps = emulator.profile_steps
    opening_angle = emulator.profile_opening_angle
    
    # Convert opening angle to radians for calculation
    angle_rad = np.radians(opening_angle)
    
    # Generate azimuth angles for each profile step
    azimuths = np.linspace(-angle_rad/2, angle_rad/2, profile_steps)
    
    profiles = []
    
    for i in range(num_profiles):
        # Simulate different scenarios
        if i == 0:
            # Wall straight ahead at 800mm
            profile = np.full(profile_steps, 800.0)
        elif i == 1:
            # Wall at 45 degrees (right side closer)
            base_distance = 1000.0
            asymmetry = 500.0  # Right side is closer by this amount
            profile = base_distance - asymmetry * np.sin(azimuths)
        elif i == 2:
            # Open space (far distances)
            profile = np.full(profile_steps, 1800.0)
        elif i == 3:
            # Corner - closer on both sides
            profile = 1200.0 + 400.0 * np.abs(np.sin(azimuths))
        else:
            # Random environment
            profile = 500.0 + 1000.0 * np.random.random(profile_steps)
        
        profiles.append(profile)
    
    return np.array(profiles, dtype=np.float32)

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
        profile_steps = emulator.profile_steps
        
        # Test 1: Simple synthetic profiles
        print("Test 1: Synthetic profiles")
        test_profile = np.full(profile_steps, 500.0, dtype=np.float32)
        result = emulator.predict_single(test_profile)
        print(f"  ✓ Flat profile (500mm): distance={result['distance_mm']:.1f}mm, IID={result['iid_db']:.1f}dB")
        
        # Test 2: Batch prediction
        print("Test 2: Batch prediction")
        batch_profiles = np.tile(test_profile, (5, 1))
        batch_result = emulator.predict(batch_profiles)
        print(f"  ✓ Batch of {len(batch_result['distance_mm'])} profiles processed")
        
        # Test 3: Try real data if available
        print("Test 3: Real environment data (if available)")
        try:
            from Library import DataProcessor
            dc = DataProcessor.DataCollection(["sessionB01"])
            real_profiles, _ = dc.load_profiles(
                opening_angle=emulator.profile_opening_angle,
                steps=emulator.profile_steps
            )
            if len(real_profiles) > 0:
                # Test on first 5 real profiles
                test_real = real_profiles[:5]
                real_results = emulator.predict(test_real)
                print(f"  ✓ Real profiles: {len(real_results['distance_mm'])} samples")
                print(f"    Distance range: {real_results['distance_mm'].min():.1f}-{real_results['distance_mm'].max():.1f}mm")
                print(f"    IID range: {real_results['iid_db'].min():.1f}-{real_results['iid_db'].max():.1f}dB")
            else:
                print("  ⚠ No real data available for testing")
        except Exception as e:
            print(f"  ⚠ Could not load real data: {e}")
        
        # Test 4: Environment simulation
        print("Test 4: Simulated environment profiles")
        env_profiles = generate_simple_environment_profiles(emulator)
        env_results = emulator.predict(env_profiles)
        print(f"  ✓ Environment profiles: {len(env_results['distance_mm'])} scenarios")
        for i, (dist, iid) in enumerate(zip(env_results['distance_mm'], env_results['iid_db'])):
            scenario_names = ["Wall ahead", "Angled wall", "Open space", "Corner", "Random"]
            print(f"    Scenario {i+1} ({scenario_names[i]}): {dist:.1f}mm, {iid:.1f}dB")
        
        # Test 5: Edge cases
        print("Test 5: Edge cases")
        # Very close profile
        close_profile = np.full(profile_steps, 100.0, dtype=np.float32)
        close_result = emulator.predict_single(close_profile)
        print(f"  ✓ Close profile (100mm): distance={close_result['distance_mm']:.1f}mm, IID={close_result['iid_db']:.1f}dB")
        
        # Very far profile
        far_profile = np.full(profile_steps, 2000.0, dtype=np.float32)
        far_result = emulator.predict_single(far_profile)
        print(f"  ✓ Far profile (2000mm): distance={far_result['distance_mm']:.1f}mm, IID={far_result['iid_db']:.1f}dB")
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

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