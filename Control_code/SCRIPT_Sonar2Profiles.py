#
# SCRIPT_Sonar2Profiles: Train a CNN to predict profile data from sonar data
#
# This script trains a convolutional neural network to predict the wall distance profiles
# from the raw sonar envelopes (time-intensity profiles) from left and right ears.
#
# Input: sonar data (N, 2, 200) - time-intensity profiles from left and right ears
# Output: profile data (N, 21) - minimum wall distance per azimuth bin
#
# Architecture: 1D CNN that processes the temporal sonar data and predicts spatial profiles
# Implementation: PyTorch with CUDA acceleration
#

# ============================================
# CONFIGURATION SETTINGS
# ============================================

# Data Configuration
sessions = ['sessionB01', 'sessionB02', 'sessionB03', 'sessionB04', 'sessionB05']
hold_out_session = 'sessionB05'  # Set to session name (e.g., 'sessionB05') to hold out for separate testing
random_state = 42

# Profile Parameters
profile_opening_angle = 30  # degrees
profile_steps = 5           # number of azimuth bins

# Training Configuration
test_size = 0.2              # 20% test set
validation_split = 0.2       # 16% validation from training (64-16-20 split)
batch_size = 32
epochs = 100
patience = 10               # early stopping patience

# Plot Configuration
plot_format = 'png'         # 'png', 'svg', or 'both'
plot_dpi = 300              # DPI for PNG plots

# Output Configuration
output_dir = 'Training'     # Directory to store all training outputs (relative to root directory)

# Model Architecture
conv_filters = [32, 64, 128] # number of filters in each conv layer
kernel_size = 5             # convolution kernel size
pool_size = 2               # max pooling size
dropout_rate = 0.3           # dropout rate for regularization
l2_reg = 0.001               # L2 regularization strength
learning_rate = 0.001       # Adam optimizer learning rate

# ============================================
# IMPORTS
# ============================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import os
from datetime import datetime
import joblib

from Library import DataProcessor

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def save_plot(filename):
    """Save current plot to output directory in specified format."""
    if plot_format in ['png', 'both']:
        png_filename = f"{output_dir}/{filename}.png"
        plt.savefig(png_filename, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved PNG plot: {output_dir}/{filename}.png")
    
    if plot_format in ['svg', 'both']:
        svg_filename = f"{output_dir}/{filename}.svg"
        plt.savefig(svg_filename, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved SVG plot: {output_dir}/{filename}.svg")
    
    if plot_format not in ['png', 'svg', 'both']:
        print(f"⚠️  Unknown plot format: {plot_format}. Using 'both' as default.")
        png_filename = f"{output_dir}/{filename}.png"
        svg_filename = f"{output_dir}/{filename}.svg"
        plt.savefig(png_filename, dpi=plot_dpi, bbox_inches='tight', facecolor='white')
        plt.savefig(svg_filename, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved both formats for: {output_dir}/{filename}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

class SonarToProfileCNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(SonarToProfileCNN, self).__init__()
        
        # Input: (batch_size, channels=2, time_samples=200)
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
            self.flattened_size = x.view(x.size(0), -1).shape[1]
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

def load_and_prepare_data():
    """Load and prepare sonar and profile data for training."""
    print("📡 Loading data...")
    
    # Separate training sessions and hold-out session
    if hold_out_session and hold_out_session in sessions:
        train_sessions = [s for s in sessions if s != hold_out_session]
        hold_out_sessions = [hold_out_session]
        print(f"🔍 Holding out session: {hold_out_session}")
        print(f"   Training on sessions: {train_sessions}")
    else:
        train_sessions = sessions
        hold_out_sessions = []
        print("🔍 No hold-out session specified")
    
    # Load training data
    dc_train = DataProcessor.DataCollection(train_sessions)
    sonar_train = dc_train.load_sonar(flatten=False)
    profiles_train, _ = dc_train.load_profiles(opening_angle=profile_opening_angle, steps=profile_steps)
    
    print(f"📊 Training sonar shape: {sonar_train.shape}")
    print(f"📊 Training profiles shape: {profiles_train.shape}")
    
    # Load hold-out data if specified
    if hold_out_sessions:
        dc_holdout = DataProcessor.DataCollection(hold_out_sessions)
        sonar_holdout = dc_holdout.load_sonar(flatten=False)
        profiles_holdout, _ = dc_holdout.load_profiles(opening_angle=profile_opening_angle, steps=profile_steps)
        
        print(f"📊 Hold-out sonar shape: {sonar_holdout.shape}")
        print(f"📊 Hold-out profiles shape: {profiles_holdout.shape}")
    else:
        sonar_holdout = None
        profiles_holdout = None
    
    # Remove NaN values from training data
    nan_mask = ~np.isnan(profiles_train).any(axis=1)
    sonar_train = sonar_train[nan_mask]
    profiles_train = profiles_train[nan_mask]
    
    # Remove NaN values from hold-out data if it exists
    if profiles_holdout is not None:
        nan_mask_holdout = ~np.isnan(profiles_holdout).any(axis=1)
        sonar_holdout = sonar_holdout[nan_mask_holdout]
        profiles_holdout = profiles_holdout[nan_mask_holdout]
    
    print(f"📊 Clean training data shapes - Sonar: {sonar_train.shape}, Profiles: {profiles_train.shape}")
    if profiles_holdout is not None:
        print(f"📊 Clean hold-out data shapes - Sonar: {sonar_holdout.shape}, Profiles: {profiles_holdout.shape}")
    
    return sonar_train, profiles_train, sonar_holdout, profiles_holdout

def preprocess_data(sonar_train, profiles_train, sonar_holdout=None, profiles_holdout=None):
    """Preprocess data for training."""
    print("🧹 Preprocessing data...")
    
    # Split training data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        sonar_train, profiles_train, test_size=test_size, random_state=random_state
    )
    
    # Further split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=random_state
    )
    
    print(f"📊 Train: {X_train.shape[0]} samples")
    print(f"📊 Validation: {X_val.shape[0]} samples")
    print(f"📊 Test: {X_test.shape[0]} samples")
    if sonar_holdout is not None:
        print(f"📊 Hold-out: {sonar_holdout.shape[0]} samples")
    
    # Normalize profile data (target) - fit only on training data
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)
    
    # Normalize sonar data (input) - fit only on training data
    # Reshape to (N*200, 2) for scaling, then reshape back
    original_shape = X_train.shape
    X_train_reshaped = X_train.reshape(-1, 2)
    X_val_reshaped = X_val.reshape(-1, 2)
    X_test_reshaped = X_test.reshape(-1, 2)
    
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = x_scaler.transform(X_val_reshaped).reshape(X_val.shape)
    X_test_scaled = x_scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Convert to PyTorch tensors and move to device
    # PyTorch expects (N, C, L) format for Conv1D
    X_train_tensor = torch.FloatTensor(X_train_scaled).permute(0, 2, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).permute(0, 2, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).permute(0, 2, 1).to(device)
    
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Prepare hold-out data if it exists
    hold_out_loader = None
    if sonar_holdout is not None and profiles_holdout is not None:
        # Apply the same scaling (fit on training data only)
        # Store original shape before reshaping
        original_holdout_shape = sonar_holdout.shape
        print(f"📊 Hold-out data original shape: {original_holdout_shape}")
        
        X_holdout_reshaped = sonar_holdout.reshape(-1, 2)
        print(f"📊 Hold-out data reshaped for scaling: {X_holdout_reshaped.shape}")
        
        y_holdout_scaled = y_scaler.transform(profiles_holdout)
        
        X_holdout_scaled = x_scaler.transform(X_holdout_reshaped).reshape(original_holdout_shape)
        print(f"📊 Hold-out data after scaling: {X_holdout_scaled.shape}")
        print(f"📊 Hold-out profiles after scaling: {y_holdout_scaled.shape}")
        
        # Convert to PyTorch tensors
        X_holdout_tensor = torch.FloatTensor(X_holdout_scaled).permute(0, 2, 1).to(device)
        y_holdout_tensor = torch.FloatTensor(y_holdout_scaled).to(device)
        
        print(f"📊 Hold-out tensor shapes - X: {X_holdout_tensor.shape}, y: {y_holdout_tensor.shape}")
        
        # Validate shapes
        expected_samples = X_holdout_scaled.shape[0]
        expected_x_shape = (expected_samples, 2, 200)  # (N, channels, time_samples)
        expected_y_shape = (expected_samples, profile_steps)  # (N, azimuth_bins)
        
        if X_holdout_tensor.shape != expected_x_shape:
            print(f"⚠️  Warning: Expected X shape {expected_x_shape}, got {X_holdout_tensor.shape}")
        if y_holdout_tensor.shape != expected_y_shape:
            print(f"⚠️  Warning: Expected y shape {expected_y_shape}, got {y_holdout_tensor.shape}")
        
        hold_out_dataset = TensorDataset(X_holdout_tensor, y_holdout_tensor)
        hold_out_loader = DataLoader(hold_out_dataset, batch_size=batch_size, shuffle=False)
        print(f"✅ Prepared hold-out data loader with {len(hold_out_dataset)} samples")
    
    return (
        train_loader, val_loader, test_loader, hold_out_loader,
        x_scaler, y_scaler
    )

def train_model(model, train_loader, val_loader, epochs, patience):
    """Train the model with early stopping."""
    print("🚂 Training model...")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    
    # Early stopping setup
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
        
        train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
        
        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save best model
            model_path = f'{output_dir}/best_model_pytorch.pth'
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved best model to: {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, y_scaler, title="Test"):
    """Evaluate model performance."""
    print(f"📊 Evaluating model on {title} set...")
    
    model.eval()
    criterion = nn.MSELoss()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    y_pred_scaled = np.concatenate(all_predictions, axis=0)
    y_test_scaled = np.concatenate(all_targets, axis=0)
    
    # Inverse transform to get actual values
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_actual = y_scaler.inverse_transform(y_test_scaled)
    
    # Calculate metrics
    mse = np.mean((y_pred - y_test_actual) ** 2)
    mae = np.mean(np.abs(y_pred - y_test_actual))
    
    print(f"📊 {title} MSE: {mse:.4f}")
    print(f"📊 {title} MAE: {mae:.4f}")
    
    # Detailed performance analysis by azimuth bin
    print(f"\n📊 {title} Performance by Azimuth Bin:")
    bin_mse = np.mean((y_pred - y_test_actual) ** 2, axis=0)
    bin_mae = np.mean(np.abs(y_pred - y_test_actual), axis=0)
    bin_corr = np.array([pearsonr(y_pred[:, i], y_test_actual[:, i])[0] for i in range(y_pred.shape[1])])
    
    # Print statistics for each bin
    for i in range(len(bin_mse)):
        print(f"  Bin {i:2d}: MSE={bin_mse[i]:8.2f}, MAE={bin_mae[i]:8.2f}, Corr={bin_corr[i]:.3f}")
    
    print(f"\n📊 {title} Overall Correlation: {np.mean(bin_corr):.3f} ± {np.std(bin_corr):.3f}")
    
    # Plot some predictions
    plt.figure(figsize=(12, 6))
    for i in range(min(5, len(y_test_actual))):
        plt.subplot(5, 1, i+1)
        plt.plot(y_test_actual[i], 'b-', label='True')
        plt.plot(y_pred[i], 'r--', label='Predicted')
        plt.title(f'{title} Sample {i}')
        plt.legend()
    plt.tight_layout()
    plt.suptitle(f'True vs Predicted Profiles ({title})')
    save_plot(f'sample_predictions_{title.lower()}')
    plt.show()
    
    # Correlation plots by azimuth bin
    plt.figure(figsize=(15, 10))
    for i in range(min(12, y_pred.shape[1])):  # Show first 12 bins
        plt.subplot(4, 3, i+1)
        plt.scatter(y_test_actual[:, i], y_pred[:, i], alpha=0.3, s=10)
        plt.plot([min(y_test_actual[:, i]), max(y_test_actual[:, i])], 
                 [min(y_test_actual[:, i]), max(y_test_actual[:, i])], 
                 'r--', alpha=0.5)
        plt.title(f'Az Bin {i}: corr={bin_corr[i]:.3f}')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.suptitle(f'Correlation Analysis by Azimuth Bin ({title})')
    save_plot(f'correlation_scatter_{title.lower()}')
    plt.show()
    
    # Overall correlation heatmap
    plt.figure(figsize=(12, 6))
    corr_matrix = np.corrcoef(y_test_actual.T, y_pred.T)[:y_pred.shape[1], y_pred.shape[1]:]
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", 
                xticklabels=[f"Pred {i}" for i in range(y_pred.shape[1])],
                yticklabels=[f"True {i}" for i in range(y_pred.shape[1])])
    plt.title(f'Correlation Matrix: True vs Predicted Profiles ({title})')
    plt.tight_layout()
    save_plot(f'correlation_heatmap_{title.lower()}')
    plt.show()
    
    return mse, mae, bin_corr

def main():
    """Main training pipeline."""
    print("🚀 Starting PyTorch CNN training for profile prediction from sonar data")
    
    # Load and prepare data
    sonar_train, profiles_train, sonar_holdout, profiles_holdout = load_and_prepare_data()
    
    # Preprocess data
    (train_loader, val_loader, test_loader, hold_out_loader,
     x_scaler, y_scaler) = preprocess_data(sonar_train, profiles_train, sonar_holdout, profiles_holdout)
    
    # Build model
    input_shape = (2, 200)  # (channels, time_samples)
    output_shape = profile_steps  # azimuth bins (adaptable)
    
    model = SonarToProfileCNN(input_shape, output_shape).to(device)
    print(f"🏗️ Model architecture:")
    print(model)
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    start_time = time.time()
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs, patience)
    training_time = time.time() - start_time
    
    print(f"⏱️  Training completed in {training_time:.2f} seconds")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(f'{output_dir}/best_model_pytorch.pth'))
    
    # Evaluate model on test set
    test_mse, test_mae, test_bin_corr = evaluate_model(model, test_loader, y_scaler, "Test")
    
    # Evaluate on hold-out set if it exists
    holdout_mse, holdout_mae, holdout_bin_corr = None, None, None
    if hold_out_loader is not None:
        holdout_mse, holdout_mae, holdout_bin_corr = evaluate_model(model, hold_out_loader, y_scaler, "Hold-out")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(np.sqrt(train_losses), label='Train RMSE')
    plt.plot(np.sqrt(val_losses), label='Validation RMSE')
    plt.title('RMSE over epochs')
    plt.legend()
    plt.tight_layout()
    save_plot('training_history')
    plt.show()
    
    # Summary
    print("✅ Training complete!")
    print(f"📊 Final Test MSE: {test_mse:.4f}")
    print(f"📊 Final Test MAE: {test_mae:.4f}")
    print(f"📊 Test Mean Correlation: {np.mean(test_bin_corr):.3f}")
    
    if holdout_mse is not None:
        print(f"\n🔍 Hold-out Session Performance:")
        print(f"📊 Hold-out MSE: {holdout_mse:.4f}")
        print(f"📊 Hold-out MAE: {holdout_mae:.4f}")
        print(f"📊 Hold-out Mean Correlation: {np.mean(holdout_bin_corr):.3f}")
        
        # Compare performance
        print(f"\n📊 Performance Comparison:")
        print(f"   Test vs Hold-out MSE ratio: {holdout_mse/test_mse:.2f}")
        print(f"   Test vs Hold-out MAE ratio: {holdout_mae/test_mae:.2f}")
        print(f"   Test vs Hold-out Corr ratio: {np.mean(holdout_bin_corr)/np.mean(test_bin_corr):.2f}")
    
    # Save training parameters for use in prediction
    training_params = {
        'profile_opening_angle': profile_opening_angle,
        'profile_steps': profile_steps,
        'conv_filters': conv_filters,
        'kernel_size': kernel_size,
        'pool_size': pool_size,
        'dropout_rate': dropout_rate,
        'input_shape': (2, 200),
        'output_shape': profile_steps
    }
    
    import json
    with open(f'{output_dir}/training_params.json', 'w') as f:
        json.dump(training_params, f, indent=2)
    print(f"💾 Saved training parameters to: {output_dir}/training_params.json")
    
    # Save scalers for use in prediction
    import joblib
    joblib.dump(x_scaler, f'{output_dir}/x_scaler.joblib')
    joblib.dump(y_scaler, f'{output_dir}/y_scaler.joblib')
    print(f"💾 Saved scalers to: {output_dir}/x_scaler.joblib and {output_dir}/y_scaler.joblib")
    
    return model, (x_scaler, y_scaler)

if __name__ == "__main__":
    model, scalers = main()