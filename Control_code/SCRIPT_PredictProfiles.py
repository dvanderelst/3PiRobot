#
# SCRIPT_PredictProfiles: Predict profiles with optional 2D heatmap visualization
#
# This script predicts profiles and optionally creates a 2D heatmap showing the spatial
# distribution of predicted profile points for the selected indices, displayed over the
# full spatial range of all robot positions. Only robot positions for the selected
# indices are shown. This provides complete spatial context while maintaining focus on
# the selected data. The heatmap provides a clean visualization without contour lines.
#
# Usage: Set the session_to_predict variable to the session you want to analyze
#
# Input: Sonar data (N, 2, 200) from specified session
# Output: Predicted profile data (N, profile_steps) + 2D heatmap visualization
#
# Dependencies: Requires a trained model in the specified output directory
#

# ============================================
# CONFIGURATION SETTINGS
# ============================================

# Prediction Configuration
session_to_predict = 'sessionB05'  # Session to predict profiles for
output_dir = 'Training'            # Directory containing trained model

# Visualization Configuration
plot_indices = [300, 305]  # Set to None to disable individual profile plotting

# 2D Heatmap Configuration
create_heatmap = True       # Set to False to disable heatmap visualization
heatmap_resolution = 100    # Heatmap grid resolution (pixels per dimension)
heatmap_colormap = 'hot'  # Color scheme ('viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'jet')
heatmap_alpha = 0.7         # Transparency (0.0-1.0)
# IMPORTS
# ============================================

heatmap_smoothing = 2   # Gaussian smoothing sigma (0 = no smoothing)
heatmap_border_expansion = 0.3  # Border expansion factor (0.0 = no expansion, 0.2 = 20% padding)

# IMPORTS
# ============================================
# Distance Filter Configuration
filter_max_distance = 1500   # Max distance from robot to include (mm), None = no filtering

# ============================================
# IMPORTS
# ========================================================================================
# IMPORTS
# ========================================================================================
# IMPORTS
# ============================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import joblib
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from Library import DataProcessor
from Library import Utils
from Library.DataProcessor import robot2world

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

class SonarToProfileCNN(nn.Module):
    """CNN model matching the training architecture."""
    def __init__(self, input_shape, output_shape, training_params=None):
        super(SonarToProfileCNN, self).__init__()
        
        # Input: (batch_size, channels=2, time_samples=200)
        # Use parameters from training if provided, otherwise use defaults
        if training_params is None:
            training_params = {
                'conv_filters': [32, 64, 128],
                'kernel_size': 5,
                'pool_size': 2,
                'dropout_rate': 0.3
            }
        
        conv_filters = training_params['conv_filters']  # number of filters in each conv layer
        kernel_size = training_params['kernel_size']    # convolution kernel size
        pool_size = training_params['pool_size']        # max pooling size
        dropout_rate = training_params['dropout_rate']   # dropout rate for regularization
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=conv_filters[0], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate the size after conv layers
        self._calculate_flattened_size(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_shape)
        )
    
    def _calculate_flattened_size(self, input_shape):
        """Calculate the flattened size after convolutional layers."""
        with torch.no_grad():
            # Create a dummy input tensor
            dummy_input = torch.zeros(1, *input_shape)
            
            # Pass through conv layers
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            
            # Flatten and get size
            self.flattened_size = x.reshape(x.size(0), -1).shape[1]
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

def load_training_parameters():
    """Load training parameters from saved metadata."""
    params_path = f'{output_dir}/training_params.json'
    
    if os.path.exists(params_path):
        import json
        with open(params_path, 'r') as f:
            params = json.load(f)
        print(f"✅ Loaded training parameters from: {params_path}")
        return params
    else:
        print("⚠️  Training parameters file not found. Using default values.")
        print(f"   Expected file: {params_path}")
        print("   Make sure to run training script to generate this file.")
        # Return default values that should match typical training
        return {
            'profile_opening_angle': 30,
            'profile_steps': 11,
            'conv_filters': [32, 64, 128],
            'kernel_size': 5,
            'pool_size': 2,
            'dropout_rate': 0.3
        }

def load_scalers_from_training():
    """Load scalers from training metadata."""
    x_scaler_path = f'{output_dir}/x_scaler.joblib'
    y_scaler_path = f'{output_dir}/y_scaler.joblib'
    
    if os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path):
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)
        print(f"✅ Loaded scalers from: {output_dir}")
        return x_scaler, y_scaler
    else:
        print("❌ Scaler files not found. Please run training first.")
        print(f"   Expected files: {x_scaler_path}, {y_scaler_path}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Make sure {output_dir}/ directory exists and contains scaler files.")
        print(f"   Run training script first: python SCRIPT_Sonar2Profiles.py")
        raise FileNotFoundError("Scaler files not found. Please train the model first.")

def load_and_preprocess_data(profile_opening_angle, profile_steps):
    """Load and preprocess sonar data for prediction."""
    print(f"📡 Loading data for session: {session_to_predict}")
    
    # Load data for the specified session
    dc = DataProcessor.DataCollection([session_to_predict])
    sonar = dc.load_sonar(flatten=False)
    profiles, _ = dc.load_profiles(opening_angle=profile_opening_angle, steps=profile_steps)
    
    print(f"📊 Sonar data shape: {sonar.shape}")
    print(f"📊 Profile data shape: {profiles.shape}")
    
    # Remove NaN values
    nan_mask = ~np.isnan(profiles).any(axis=1)
    sonar = sonar[nan_mask]
    profiles = profiles[nan_mask]
    
    print(f"📊 Clean data shapes - Sonar: {sonar.shape}, Profiles: {profiles.shape}")
    
    return sonar, profiles

def preprocess_data(sonar, profiles, x_scaler, y_scaler):
    """Preprocess data using scalers from training."""
    print("🧹 Preprocessing data...")
    
    # Normalize profile data (target)
    y_scaled = y_scaler.transform(profiles)
    
    # Normalize sonar data (input)
    # Reshape to (N*200, 2) for scaling, then reshape back
    original_shape = sonar.shape
    X_reshaped = sonar.reshape(-1, 2)
    
    X_scaled = x_scaler.transform(X_reshaped).reshape(original_shape)
    
    # Convert to PyTorch tensors and move to device
    # PyTorch expects (N, C, L) format for Conv1D
    X_tensor = torch.FloatTensor(X_scaled).permute(0, 2, 1).to(device)
    y_tensor = torch.FloatTensor(y_scaled).to(device)
    
    # Create dataset and data loader
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    return data_loader, x_scaler, y_scaler

def predict_profiles(model, data_loader, y_scaler):
    """Make predictions using the trained model."""
    print("📊 Making predictions...")
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    y_pred_scaled = np.concatenate(all_predictions, axis=0)
    y_test_scaled = np.concatenate(all_targets, axis=0)
    
    # Inverse transform to get actual values
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_actual = y_scaler.inverse_transform(y_test_scaled)
    
    return y_pred, y_test_actual

def transform_profiles_to_world(profile_distances, profile_azimuths, rob_x, rob_y, rob_yaw_deg):
    """Transform profile distances from robot-relative to world coordinates."""
    # Profile is a 1D array where each element is the distance at a specific azimuth
    # Transform each (azimuth, distance) pair to world coordinates
    x_world, y_world = robot2world(profile_azimuths, profile_distances, rob_x, rob_y, rob_yaw_deg)
    
    return x_world, y_world

def create_2d_heatmap_with_indices(predictions, targets, profile_azimuths, rob_x, rob_y, rob_yaw_deg, profile_opening_angle, profile_steps, indices):
    """Create a clean 2D heatmap from predicted profiles using only specified indices, but covering the full spatial range (no contour lines)."""
    print("📊 Creating 2D heatmap visualization...")
    
    # First, determine the full spatial range by checking all robot positions
    # This gives us the complete coverage area even though we only plot selected indices
    all_rob_x = rob_x
    all_rob_y = rob_y
    
    # Calculate full spatial range based on all robot positions
    # We'll use this to set the heatmap bounds
    full_x_min, full_x_max = np.min(all_rob_x), np.max(all_rob_x)
    full_y_min, full_y_max = np.min(all_rob_y), np.max(all_rob_y)
    
    # Transform only the selected predicted profiles to world coordinates
    all_pred_x = []
    all_pred_y = []
    
    for i in indices:
        try:
            pred_x, pred_y = transform_profiles_to_world(
                predictions[i], profile_azimuths, rob_x[i], rob_y[i], rob_yaw_deg[i]
            )
            if pred_x is not None and pred_y is not None:
                # Apply distance filter if enabled
                if filter_max_distance is not None:
                    # Calculate distance from robot position
                    robot_pos = np.array([rob_x[i], rob_y[i]])
                    profile_points = np.column_stack([pred_x, pred_y])
                    distances = np.linalg.norm(profile_points - robot_pos, axis=1)
                    
                    # Only keep points within max distance
                    close_mask = distances <= filter_max_distance
                    pred_x = np.array(pred_x)[close_mask].tolist()
                    pred_y = np.array(pred_y)[close_mask].tolist()
                    
                    if len(pred_x) == 0:
                        print(f"⚠️  All points filtered out for profile {i} (max distance: {filter_max_distance}mm)")
                        continue
                
                all_pred_x.extend(pred_x)
                all_pred_y.extend(pred_y)
        except Exception as e:
            print(f"⚠️  Error transforming profile {i}: {e}")
    
    if len(all_pred_x) == 0 or len(all_pred_y) == 0:
        print("⚠️  No valid prediction points for heatmap")
        return
    
    # Convert to numpy arrays and clean data
    all_pred_x = np.array(all_pred_x)
    all_pred_y = np.array(all_pred_y)
    
    # Remove NaN and Inf values
    valid_mask = np.isfinite(all_pred_x) & np.isfinite(all_pred_y)
    clean_x = all_pred_x[valid_mask]
    clean_y = all_pred_y[valid_mask]
    
    if clean_x.size == 0 or clean_y.size == 0:
        print("⚠️  No valid finite points after cleaning")
        return
    
    # Create 2D histogram for heatmap
    # Inform about filtering
    if filter_max_distance is not None:
        print(f"📊 Distance filter active: max {filter_max_distance}mm from robot center")
    
    print(f"📊 Creating heatmap with {len(clean_x)} points from {len(indices)} selected profiles...")
    print(f"📊 Full spatial coverage: X [{full_x_min:.1f}, {full_x_max:.1f}] mm, Y [{full_y_min:.1f}, {full_y_max:.1f}] mm")
    print(f"📊 Data points range: X [{np.min(clean_x):.1f}, {np.max(clean_x):.1f}] mm, Y [{np.min(clean_y):.1f}, {np.max(clean_y):.1f}] mm")
    print(f"📊 Robot positions shown: {len(indices)} selected positions (matching profile indices)")
    if heatmap_border_expansion > 0:
        print(f"📊 Heatmap border expansion: {heatmap_border_expansion * 100:.0f}% padding around spatial range")
    
    # Use the full spatial range for heatmap bounds, but only plot the selected data points
    x_min, x_max = full_x_min, full_x_max
    y_min, y_max = full_y_min, full_y_max
    
    # Add configurable padding based on border expansion parameter
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_pad = x_range * heatmap_border_expansion
    y_pad = y_range * heatmap_border_expansion
    
    # Create grid for heatmap
    x_edges = np.linspace(x_min - x_pad, x_max + x_pad, heatmap_resolution)
    y_edges = np.linspace(y_min - y_pad, y_max + y_pad, heatmap_resolution)
    
    # Create 2D histogram
    heatmap, x_edges, y_edges = np.histogram2d(clean_x, clean_y, bins=[x_edges, y_edges])
    heatmap = heatmap.T  # Transpose to match image orientation
    
    # Apply Gaussian smoothing if requested
    if heatmap_smoothing > 0:
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=heatmap_smoothing)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    img = plt.imshow(heatmap, extent=extent, origin='lower', 
                    cmap=heatmap_colormap, alpha=heatmap_alpha)
    
    # Add colorbar
    plt.colorbar(img, label='Point Count')
    
    # Plot robot positions for the selected indices only
    selected_rob_x = [rob_x[i] for i in indices]
    selected_rob_y = [rob_y[i] for i in indices]
    plt.scatter(selected_rob_x, selected_rob_y, color='red', s=30, alpha=0.8, label='Robot Positions (Selected Indices)')
    

    plt.title(f'2D Heatmap: {len(indices)} Selected Profiles over Full Spatial Range ({heatmap_resolution}x{heatmap_resolution} grid, {heatmap_border_expansion*100:.0f}% border)', 
             fontsize=16)
    plt.xlabel('X (mm)', fontsize=14)
    plt.ylabel('Y (mm)', fontsize=14)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """Main prediction pipeline."""
    print("🚀 Starting profile prediction with 2D heatmap visualization")
    print(f"   Session: {session_to_predict}")
    print(f"   Model: {output_dir}/best_model_pytorch.pth")
    
    # Load training parameters to ensure consistency
    training_params = load_training_parameters()
    
    # Extract parameters from training (these are NOT configurable in prediction script)
    profile_opening_angle = training_params['profile_opening_angle']
    profile_steps = training_params['profile_steps']
    
    print(f"📊 Using training parameters:")
    print(f"   Profile opening angle: {profile_opening_angle}°")
    print(f"   Profile steps: {profile_steps}")
    
    # Load and preprocess data
    sonar, profiles = load_and_preprocess_data(profile_opening_angle, profile_steps)
    
    # Load scalers (must exist from training)
    x_scaler, y_scaler = load_scalers_from_training()
    
    # Preprocess data using training scalers
    data_loader, _, _ = preprocess_data(sonar, profiles, x_scaler, y_scaler)
    
    # Build model with same architecture as training
    input_shape = (2, 200)  # (channels, time_samples)
    output_shape = profile_steps  # azimuth bins
    
    model = SonarToProfileCNN(input_shape, output_shape, training_params).to(device)
    print(f"🏗️ Model architecture:")
    print(f"   Input shape: {input_shape}")
    print(f"   Output shape: {output_shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Load trained model
    model_path = f'{output_dir}/best_model_pytorch.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Loaded trained model from: {model_path}")
    else:
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    
    # Make predictions
    predictions, targets = predict_profiles(model, data_loader, y_scaler)
    
    print("✅ Prediction complete!")
    print(f"📊 Predictions shape: {predictions.shape}")
    print(f"📊 Targets shape: {targets.shape}")
    
    # Calculate performance metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    print(f"📊 Performance Metrics:")
    print(f"   MSE: {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    # Plot selected indices if requested
    if plot_indices is not None:
        print(f"📊 Plotting profiles for indices: {plot_indices}")
        
        # Load robot positions for plotting
        dc = DataProcessor.DataProcessor(session_to_predict)
        rob_x = dc.rob_x
        rob_y = dc.rob_y
        rob_yaw_deg = dc.rob_yaw_deg
        
        # Generate azimuth angles for the profile
        az_min = -0.5 * profile_opening_angle
        az_max = 0.5 * profile_opening_angle
        profile_azimuths = np.linspace(az_min, az_max, profile_steps)
        
        # Plot all requested indices on a single comprehensive plot
        if isinstance(plot_indices, list) and len(plot_indices) == 2:
            start_idx, end_idx = plot_indices
            plot_indices_range = range(start_idx, min(end_idx, len(predictions)))
            
            # Create a single plot with all selected indices
            plt.figure(figsize=(16, 12))
            
            # Plot each profile in the selected range
            for idx in plot_indices_range:
                # Transform profiles to world coordinates
                pred_x, pred_y = transform_profiles_to_world(
                    predictions[idx], profile_azimuths, rob_x[idx], rob_y[idx], rob_yaw_deg[idx]
                )
                real_x, real_y = transform_profiles_to_world(
                    targets[idx], profile_azimuths, rob_x[idx], rob_y[idx], rob_yaw_deg[idx]
                )
                
                # Apply distance filter to individual profiles if enabled
                if filter_max_distance is not None:
                    # Filter predicted points
                    robot_pos = np.array([rob_x[idx], rob_y[idx]])
                    pred_points = np.column_stack([pred_x, pred_y])
                    pred_distances = np.linalg.norm(pred_points - robot_pos, axis=1)
                    pred_mask = pred_distances <= filter_max_distance
                    pred_x = np.array(pred_x)[pred_mask].tolist()
                    pred_y = np.array(pred_y)[pred_mask].tolist()
                    
                    # Filter real points
                    real_points = np.column_stack([real_x, real_y])
                    real_distances = np.linalg.norm(real_points - robot_pos, axis=1)
                    real_mask = real_distances <= filter_max_distance
                    real_x = np.array(real_x)[real_mask].tolist()
                    real_y = np.array(real_y)[real_mask].tolist()
                    
                    if len(pred_x) == 0 or len(real_x) == 0:
                        print(f"⚠️  Profile {idx} completely filtered out by distance limit")
                        continue
                
                # Plot robot position for this index
                plt.scatter(rob_x[idx], rob_y[idx], color='black', s=100, label=f'Robot {idx}' if idx == start_idx else '')
                
                # Plot orientation arrow
                dx = 100 * np.cos(np.deg2rad(rob_yaw_deg[idx]))
                dy = 100 * np.sin(np.deg2rad(rob_yaw_deg[idx]))
                plt.arrow(rob_x[idx], rob_y[idx], dx, dy, color='red', width=5, 
                         length_includes_head=True, head_width=20)
                
                # Plot predicted and real profiles
                plt.plot(pred_x, pred_y, 'g-', linewidth=2, alpha=0.7, label='Predicted' if idx == start_idx else '')
                plt.plot(real_x, real_y, 'b--', linewidth=2, alpha=0.7, label='Real' if idx == start_idx else '')
                
                # Add markers for profile points
                plt.scatter(pred_x, pred_y, color='green', s=40, alpha=0.5)
                plt.scatter(real_x, real_y, color='blue', s=40, alpha=0.5)
                
                # Add index label near robot position
                plt.text(rob_x[idx], rob_y[idx], f'{idx}', 
                         fontsize=12, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
            
            plt.title(f'Predicted vs Real Profiles for Indices {start_idx} to {end_idx-1}', fontsize=16)
            plt.xlabel('X (mm)', fontsize=14)
            plt.ylabel('Y (mm)', fontsize=14)
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            # Add legend (only show one entry for each type)
            handles, labels = plt.gca().get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            plt.legend(unique_labels.values(), unique_labels.keys(), fontsize=12)
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️  Invalid plot_indices format. Expected [start, end], got {plot_indices}")
            print("   Example: plot_indices = [0, 5]  # Plot indices 0 through 4")
    
    # Create 2D heatmap visualization
    if create_heatmap:
        # Determine which indices to use for heatmap (same as plotted profiles)
        if plot_indices is not None and isinstance(plot_indices, list) and len(plot_indices) == 2:
            start_idx, end_idx = plot_indices
            heatmap_indices = range(start_idx, min(end_idx, len(predictions)))
            print(f"📊 Creating heatmap using predicted profiles for indices: {start_idx} to {min(end_idx, len(predictions))-1}")
        else:
            # If plot_indices is not properly configured, use all data
            heatmap_indices = range(len(predictions))
            print(f"📊 Creating heatmap using all predicted profiles (0 to {len(predictions)-1})")
        
        # Load robot positions for heatmap
        dc = DataProcessor.DataProcessor(session_to_predict)
        rob_x = dc.rob_x
        rob_y = dc.rob_y
        rob_yaw_deg = dc.rob_yaw_deg
        
        # Generate azimuth angles for the profile
        az_min = -0.5 * profile_opening_angle
        az_max = 0.5 * profile_opening_angle
        profile_azimuths = np.linspace(az_min, az_max, profile_steps)
        
        # Create heatmap using only the selected indices
        create_2d_heatmap_with_indices(predictions, targets, profile_azimuths, rob_x, rob_y, rob_yaw_deg, 
                                      profile_opening_angle, profile_steps, heatmap_indices)
    else:
        print("📊 Skipping heatmap visualization (disabled in configuration)")
    
    return predictions, targets

if __name__ == "__main__":
    predictions, targets = main()