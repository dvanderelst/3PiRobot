#!/usr/bin/env python3

"""
Test script to verify the MLP model can be loaded and used for predictions.
"""

import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

import numpy as np
import torch
import json
from Library import DataProcessor

def test_model_loading():
    """Test that the trained model can be loaded and used."""
    print("🧪 Testing MLP model loading and inference...")
    
    # Find the latest results directory
    import os
    import glob
    
    # Look for results directories
    results_dirs = glob.glob('mlp_profiles_results_*')
    if not results_dirs:
        print("❌ No results directories found. Please run SCRIPT_Train_MLP_Profiles.py first.")
        return False
    
    # Use the most recent one
    results_dir = max(results_dirs)
    print(f"   Using results from: {results_dir}")
    
    # Load configuration
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"   Model architecture: {config['input_size']} → {' → '.join(map(str, config['hidden_sizes']))} → {config['output_size']}")
    
    # Recreate the model architecture
    def create_mlp_model(input_size, hidden_sizes, output_size, dropout_rate=0.2):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(torch.nn.Linear(prev_size, output_size))
        return torch.nn.Sequential(*layers)
    
    model = create_mlp_model(
        config['input_size'],
        config['hidden_sizes'],
        config['output_size']
    )
    
    # Load the trained weights
    model_path = os.path.join(results_dir, 'mlp_profiles_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with some sample data
    print(f"\n📊 Testing model with sample data...")
    
    # Load a small sample from the same sessions
    dc = DataProcessor.DataCollection(config['sessions'][:1])  # Just use first session for quick test
    sonar_data = dc.load_sonar(flatten=True)
    
    # Take a few samples
    test_samples = sonar_data[:5]
    print(f"   Test samples shape: {test_samples.shape}")
    
    # Standardize using the same scaler parameters (we'll approximate)
    # In a real scenario, we would save and load the scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    test_samples_scaled = scaler.fit_transform(test_samples)
    
    # Convert to tensor
    test_tensor = torch.FloatTensor(test_samples_scaled)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(test_tensor)
    
    predictions_np = predictions.numpy()
    print(f"   Predictions shape: {predictions_np.shape}")
    print(f"   Prediction range: [{np.min(predictions_np):.1f}, {np.max(predictions_np):.1f}]")
    
    # Check that predictions are reasonable (positive distances)
    negative_preds = np.sum(predictions_np < 0)
    if negative_preds > 0:
        print(f"   ⚠️  Warning: {negative_preds} negative predictions found")
    else:
        print(f"   ✅ All predictions are positive")
    
    # Load evaluation metrics
    eval_path = os.path.join(results_dir, 'evaluation_metrics.json')
    with open(eval_path, 'r') as f:
        evaluation = json.load(f)
    
    print(f"\n📈 Model performance metrics:")
    print(f"   Test MSE: {evaluation['test_loss']:.2f}")
    print(f"   Mean Absolute Error: {evaluation['mean_absolute_error']:.1f}mm")
    print(f"   Std Absolute Error: {evaluation['std_absolute_error']:.1f}mm")
    
    # Check if azimuth diagnostics are available
    if 'az_bin_correlations' in evaluation:
        print(f"\n📊 Per-Azimuth-Bin Diagnostics:")
        print(f"   Mean Correlation: {evaluation['mean_correlation']:.3f}")
        print(f"   Mean R² Score: {evaluation['mean_r2_score']:.3f}")
        print(f"   Best Azimuth: {evaluation['best_azimuth_deg']:.1f}° (corr: {evaluation['best_correlation']:.3f})")
        print(f"   Worst Azimuth: {evaluation['worst_azimuth_deg']:.1f}° (corr: {evaluation['worst_correlation']:.3f})")
        
        # Check that all azimuth bins have reasonable metrics
        min_corr = min(evaluation['az_bin_correlations'])
        max_corr = max(evaluation['az_bin_correlations'])
        
        if min_corr > 0.3 and max_corr > 0.7:
            print(f"   ✅ Correlation range looks reasonable")
        else:
            print(f"   ⚠️  Correlation range might need attention")
    else:
        print(f"   ⚠️  No azimuth diagnostics found (older model)")
    
    return True

def test_data_consistency():
    """Test that we can reproduce the data loading process."""
    print(f"\n🔄 Testing data consistency...")
    
    sessions = ['session03', 'session04', 'session06', 'session07']
    opening_angle = 90
    profile_steps = 20
    
    # Load data
    dc = DataProcessor.DataCollection(sessions)
    sonar_data = dc.load_sonar(flatten=True)
    profiles, centers = dc.load_profiles(opening_angle=opening_angle, steps=profile_steps)
    
    print(f"   Sonar data shape: {sonar_data.shape}")
    print(f"   Profiles shape: {profiles.shape}")
    print(f"   Data types: sonar={sonar_data.dtype}, profiles={profiles.dtype}")
    
    # Check for NaN values
    sonar_nans = np.isnan(sonar_data).sum()
    profile_nans = np.isnan(profiles).sum()
    
    print(f"   Sonar NaN values: {sonar_nans}")
    print(f"   Profile NaN values: {profile_nans}")
    
    if sonar_nans == 0 and profile_nans == 0:
        print(f"   ✅ No NaN values in data")
        return True
    else:
        print(f"   ❌ Found NaN values in data")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing MLP Model and Data Pipeline")
    print("=" * 50)
    
    tests = [
        test_model_loading,
        test_data_consistency
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
                print(f"❌ {test.__name__} FAILED\n")
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
    success = main()
    exit(0 if success else 1)