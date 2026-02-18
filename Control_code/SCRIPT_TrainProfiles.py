#
# SCRIPT_Sonar2Profiles_TwoHeaded: Two-headed CNN for sonar-to-profile prediction
#
# This script trains a two-headed convolutional neural network that:
# 1. Detects wall presence (binary classification per azimuth bin)
# 2. Predicts wall distance (regression for distances 0-2000mm)
#
# Key Features:
# - Explicit handling of "no wall" cases
# - Focused learning on valid distance range
# - Adaptive task balancing to prevent interference
# - Spatial generalization evaluation
#
# Input: sonar data (N, 2, 200) - time-intensity profiles from left and right ears
# Output: presence (N, bins) + distance (N, bins) predictions
#
# Architecture: Two-headed 1D CNN with shared feature extractor
# Implementation: PyTorch with CUDA acceleration
#

# ============================================
# CONFIGURATION SETTINGS
# ============================================

# Data Configuration
sessions = ['sessionB01', 'sessionB02', 'sessionB03', 'sessionB04']

# Spatial Splitting Configuration
train_quadrants = [0, 1, 2, 3]    # Quadrants to train on
test_quadrant = 0             # Quadrant to test on

# Profile Parameters
profile_opening_angle = 60   # degrees
profile_steps = 11            # number of azimuth bins

# Distance Threshold
distance_threshold = 1500.0  # mm, maximum distance for prediction

# Training Configuration
validation_split = 0.3
batch_size = 32
epochs = 100                  # Reduced for testing
patience = 10                 # early stopping patience
seed = 42                    # Reproducibility seed
debug = False                # Print extra debug info

# Plot Configuration
plot_format = 'png'         # 'png', 'svg', or 'both'
plot_dpi = 300              # DPI for PNG plots

# Output Configuration
output_dir = 'Training'  # Directory for two-headed training outputs

# Model Architecture
conv_filters = [32, 64, 128] # number of filters in each conv layer
kernel_size = 5             # convolution kernel size
pool_size = 2               # max pooling size
dropout_rate = 0.3           # dropout rate for regularization
l2_reg = 0.001               # L2 regularization strength
learning_rate = 0.001        # Adam optimizer learning rate

# ============================================
# IMPORTS
# ============================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import os
import joblib
import json

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


def write_training_readme():
    """Write a short artifact guide to the training output folder."""
    readme_path = f"{output_dir}/README.md"
    txt = f"""# Training Output

This folder contains artifacts from `SCRIPT_TrainProfiles.py` (two-headed sonar->profile model).

## Core Artifacts

- `best_model_pytorch.pth`
  Best model checkpoint selected by validation loss (PyTorch `state_dict`).
- `training_params.json`
  Training and model parameters needed for inference.
- `y_scaler.joblib`
  Per-bin distance scaler used to normalize/inverse-transform distance predictions.

## Evaluation Tables

- `presence_confusion_test_set.csv`
  TP/FP/TN/FN and derived metrics per azimuth bin on test set.
- `presence_confusion_training_set.csv`
  TP/FP/TN/FN and derived metrics per azimuth bin on training set.

## Plots

Saved in the configured plot format (`{plot_format}`):

- `training_curves.*`
  Train/validation loss trajectories.
- `bin_scatter_plots_test_set.*`
  Distance prediction scatter plots by azimuth bin (test set).
- `bin_scatter_plots_training_set.*`
  Distance prediction scatter plots by azimuth bin (training set).
- `presence_confusion_test_set.*`
  Presence confusion visualization (test set).
- `presence_confusion_training_set.*`
  Presence confusion visualization (training set).
- `test_samples.*`
  Example per-sample prediction plots.
- `spatial_errors.*`
  Spatial map of test error.

## Notes

- Distances are clipped to `distance_threshold` during target construction.
- Presence threshold used for inference/evaluation is stored in `training_params.json`.
- Output indices and splits are based on the filtered dataset used during training.
"""
    with open(readme_path, 'w') as f:
        f.write(txt)
    print(f"💾 Saved training readme: {readme_path}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

# Reproducibility
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Two-headed model for presence detection and distance prediction
class TwoHeadedSonarCNN(nn.Module):
    def __init__(self, input_shape, profile_steps):
        super(TwoHeadedSonarCNN, self).__init__()
        self.profile_steps = profile_steps
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=conv_filters[0], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate flattened size after shared layers
        self._calculate_flattened_size(input_shape)
        
        # Presence detection head (binary classification)
        self.presence_head = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, profile_steps)  # Logits (use BCEWithLogitsLoss)
        )
        
        # Distance prediction head (regression)
        self.distance_head = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, profile_steps)
            # No activation for regression output
        )
    
    def _calculate_flattened_size(self, input_shape):
        """Calculate the flattened size after convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.shared(dummy_input)
            self.flattened_size = x.view(x.size(0), -1).shape[1]
    
    def forward(self, x):
        # Shared feature extraction
        features = self.shared(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Task-specific predictions
        presence = self.presence_head(features)
        distance = self.distance_head(features)
        
        return presence, distance


class DistanceScaler:
    """Per-bin standardization fit on present-wall bins only."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, y_distance, y_presence):
        n_bins = y_distance.shape[1]
        self.mean_ = np.zeros(n_bins, dtype=np.float32)
        self.scale_ = np.ones(n_bins, dtype=np.float32)
        
        for bin_idx in range(n_bins):
            present_vals = y_distance[y_presence[:, bin_idx] > 0.5, bin_idx]
            if present_vals.size >= 2:
                std = np.std(present_vals)
                self.mean_[bin_idx] = np.mean(present_vals)
                self.scale_[bin_idx] = std if std > 1e-6 else 1.0
            elif present_vals.size == 1:
                self.mean_[bin_idx] = present_vals[0]
                self.scale_[bin_idx] = 1.0
            else:
                self.mean_[bin_idx] = 0.0
                self.scale_[bin_idx] = 1.0
        return self
    
    def transform(self, y):
        return (y - self.mean_) / self.scale_
    
    def inverse_transform(self, y):
        return y * self.scale_ + self.mean_

def load_and_prepare_data():
    """Load and prepare sonar and profile data with spatial quadrant information."""
    print("📡 Loading data with spatial information...")
    
    # Load all sessions
    dc = DataProcessor.DataCollection(sessions)
    sonar_data = dc.load_sonar(flatten=False)
    profiles_data, _ = dc.load_profiles(opening_angle=profile_opening_angle, steps=profile_steps)
    
    # Get spatial quadrant information
    quadrants = dc.quadrants
    rob_x = dc.rob_x
    rob_y = dc.rob_y
    
    print(f"📊 Loaded data:")
    print(f"   Sonar shape: {sonar_data.shape}")
    print(f"   Profiles shape: {profiles_data.shape}")
    print(f"   Quadrants shape: {quadrants.shape}")
    
    # Print quadrant distribution
    for q in range(4):
        count = np.sum(quadrants == q)
        percentage = count / len(quadrants) * 100
        print(f"   Quadrant {q}: {count} samples ({percentage:.1f}%)")
    
    # Remove NaN values
    nan_mask = ~np.isnan(profiles_data).any(axis=1)
    sonar_data = sonar_data[nan_mask]
    profiles_data = profiles_data[nan_mask]
    quadrants = quadrants[nan_mask]
    rob_x = rob_x[nan_mask]
    rob_y = rob_y[nan_mask]
    
    print(f"📊 After NaN removal:")
    print(f"   Sonar shape: {sonar_data.shape}")
    print(f"   Profiles shape: {profiles_data.shape}")
    print(f"   Quadrants shape: {quadrants.shape}")
    
    return sonar_data, profiles_data, quadrants, rob_x, rob_y

def create_spatial_split(sonar_data, profiles_data, quadrants, train_quadrants=None, test_quadrant=None):
    """Create single spatial split based on quadrant information."""
    print("🗺️  Creating single_split spatial split...")
    
    # Single spatial split: train on specified quadrants, test on one quadrant
    train_mask = np.isin(quadrants, train_quadrants)
    test_mask = (quadrants == test_quadrant)
    
    print(f"   Training quadrants: {train_quadrants}")
    print(f"   Test quadrant: {test_quadrant}")
    print(f"   Training samples: {np.sum(train_mask)}")
    print(f"   Test samples: {np.sum(test_mask)}")
    
    n_train = int(np.sum(train_mask))
    n_test = int(np.sum(test_mask))
    if n_train < 3:
        raise ValueError(
            f"Not enough training samples ({n_train}) for train/val split. "
            f"Adjust train_quadrants={train_quadrants}."
        )
    if n_test < 1:
        raise ValueError(
            f"No test samples found for test_quadrant={test_quadrant}. "
            "Adjust split configuration."
        )
    
    n_val = int(np.ceil(validation_split * n_train))
    if n_val < 1 or (n_train - n_val) < 1:
        raise ValueError(
            f"Invalid validation_split={validation_split} for {n_train} training samples. "
            "Need at least one sample in both train and validation."
        )
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        sonar_data[train_mask], profiles_data[train_mask],
        test_size=validation_split, random_state=42
    )
    
    X_test = sonar_data[test_mask]
    y_test = profiles_data[test_mask]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_mask, test_mask

def preprocess_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Preprocess data for two-headed model training."""
    print("🧹 Preprocessing data for two-headed model...")
    
    print(f"📊 Dataset sizes:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Create presence targets (binary: 1 if distance <= threshold, else 0)
    y_train_presence = (y_train <= distance_threshold).astype(float)
    y_val_presence = (y_val <= distance_threshold).astype(float)
    y_test_presence = (y_test <= distance_threshold).astype(float)
    
    # Clip distance targets to threshold
    y_train_distance = np.clip(y_train, 0, distance_threshold)
    y_val_distance = np.clip(y_val, 0, distance_threshold)
    y_test_distance = np.clip(y_test, 0, distance_threshold)
    
    # Normalize distance targets using present bins only.
    y_scaler = DistanceScaler().fit(y_train_distance, y_train_presence)
    y_train_distance_scaled = y_scaler.transform(y_train_distance)
    y_val_distance_scaled = y_scaler.transform(y_val_distance)
    y_test_distance_scaled = y_scaler.transform(y_test_distance)
    
    # Keep tensors on CPU; move each batch to device in training/eval loops.
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1)
    X_val_tensor = torch.FloatTensor(X_val).permute(0, 2, 1)
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 2, 1)
    
    y_train_presence_tensor = torch.FloatTensor(y_train_presence)
    y_val_presence_tensor = torch.FloatTensor(y_val_presence)
    y_test_presence_tensor = torch.FloatTensor(y_test_presence)
    
    y_train_distance_tensor = torch.FloatTensor(y_train_distance_scaled)
    y_val_distance_tensor = torch.FloatTensor(y_val_distance_scaled)
    y_test_distance_tensor = torch.FloatTensor(y_test_distance_scaled)
    
    # Create custom dataset class for two-headed model
    class TwoHeadedDataset(torch.utils.data.Dataset):
        def __init__(self, X, presence, distance):
            self.X = X
            self.presence = presence
            self.distance = distance
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.presence[idx], self.distance[idx]
    
    train_dataset = TwoHeadedDataset(X_train_tensor, y_train_presence_tensor, y_train_distance_tensor)
    val_dataset = TwoHeadedDataset(X_val_tensor, y_val_presence_tensor, y_val_distance_tensor)
    test_dataset = TwoHeadedDataset(X_test_tensor, y_test_presence_tensor, y_test_distance_tensor)
    
    # Create data loaders
    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
    # Return the original data for later visualization
    return train_loader, val_loader, test_loader, y_scaler, {
        'X_train': X_train, 'y_train_presence': y_train_presence, 'y_train_distance': y_train_distance,
        'X_test': X_test, 'y_test_presence': y_test_presence, 'y_test_distance': y_test_distance
    }

def train_model(model, train_loader, val_loader, epochs, patience):
    """Train the two-headed model with early stopping."""
    print("🚂 Training two-headed model...")
    
    # Loss functions
    presence_criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy on logits
    distance_criterion = nn.MSELoss(reduction='none')  # masked MSE for distance
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    
    # Early stopping setup
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Training loop
    train_losses = []
    val_losses = []
    presence_losses = []
    distance_losses = []
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        running_presence_loss = 0.0
        running_distance_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            # Batch is a list: [inputs, presence_targets, distance_targets]
            inputs = batch[0].to(device, non_blocking=True)
            presence_targets = batch[1].to(device, non_blocking=True)
            distance_targets = batch[2].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            presence_logits, distance_pred = model(inputs)
            
            # Calculate individual losses
            presence_loss = presence_criterion(presence_logits, presence_targets)
            distance_loss_raw = distance_criterion(distance_pred, distance_targets)
            present_mask = (presence_targets > 0.5).float()
            valid_count = present_mask.sum()
            if valid_count.item() > 0:
                distance_loss = (distance_loss_raw * present_mask).sum() / valid_count
            else:
                # Keep graph connected even if batch has no present-wall bins.
                distance_loss = distance_pred.sum() * 0.0
            
            # Combined loss (equal weighting for now)
            loss = presence_loss + distance_loss
            
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            running_train_loss += loss.item() * inputs.size(0)
            running_presence_loss += presence_loss.item() * inputs.size(0)
            running_distance_loss += distance_loss.item() * inputs.size(0)
        
        # Calculate epoch losses
        train_loss = running_train_loss / len(train_loader.dataset)
        train_presence_loss = running_presence_loss / len(train_loader.dataset)
        train_distance_loss = running_distance_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        presence_losses.append(train_presence_loss)
        distance_losses.append(train_distance_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} "
              f"(Presence: {train_presence_loss:.6f}, Distance: {train_distance_loss:.6f})")
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        running_val_presence_loss = 0.0
        running_val_distance_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                # Batch is a list: [inputs, presence_targets, distance_targets]
                inputs = batch[0].to(device, non_blocking=True)
                presence_targets = batch[1].to(device, non_blocking=True)
                distance_targets = batch[2].to(device, non_blocking=True)
                
                # Forward pass
                presence_logits, distance_pred = model(inputs)
                
                # Calculate individual losses
                presence_loss = presence_criterion(presence_logits, presence_targets)
                distance_loss_raw = distance_criterion(distance_pred, distance_targets)
                present_mask = (presence_targets > 0.5).float()
                valid_count = present_mask.sum()
                if valid_count.item() > 0:
                    distance_loss = (distance_loss_raw * present_mask).sum() / valid_count
                else:
                    distance_loss = torch.tensor(0.0, device=inputs.device)
                
                # Combined loss (equal weighting)
                loss = presence_loss + distance_loss
                
                # Accumulate losses
                running_val_loss += loss.item() * inputs.size(0)
                running_val_presence_loss += presence_loss.item() * inputs.size(0)
                running_val_distance_loss += distance_loss.item() * inputs.size(0)
        
        # Calculate validation losses
        val_loss = running_val_loss / len(val_loader.dataset)
        val_presence_loss = running_val_presence_loss / len(val_loader.dataset)
        val_distance_loss = running_val_distance_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.6f} "
              f"(Presence: {val_presence_loss:.6f}, Distance: {val_distance_loss:.6f})")
        
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
    
    return train_losses, val_losses, presence_losses, distance_losses

def create_scatter_plots(model, data_loader, y_scaler, dataset_name):
    """Create comprehensive scatter plots for distance predictions by azimuth bin."""
    print(f"📊 Creating scatter plots for {dataset_name}...")
    
    # Get predictions
    model.eval()
    all_presence_preds = []
    all_distance_preds = []
    all_presence_targets = []
    all_distance_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device, non_blocking=True)
            presence_targets = batch[1].to(device, non_blocking=True)
            distance_targets = batch[2].to(device, non_blocking=True)
            presence_logits, distance_pred = model(inputs)
            presence_probs = torch.sigmoid(presence_logits)
            all_presence_preds.append(presence_probs.cpu().numpy())
            all_distance_preds.append(distance_pred.cpu().numpy())
            all_presence_targets.append(presence_targets.cpu().numpy())
            all_distance_targets.append(distance_targets.cpu().numpy())
    
    presence_preds = np.concatenate(all_presence_preds, axis=0)
    distance_preds = np.concatenate(all_distance_preds, axis=0)
    presence_targets = np.concatenate(all_presence_targets, axis=0)
    distance_targets_scaled = np.concatenate(all_distance_targets, axis=0)
    
    # Rescale distance predictions back to original units
    distance_preds_rescaled = y_scaler.inverse_transform(distance_preds)
    distance_targets_rescaled = y_scaler.inverse_transform(distance_targets_scaled)
    
    # Create presence mask
    present_mask = presence_targets > 0.5
    
    plt.figure(figsize=(15, 10))
    for bin_idx in range(min(12, presence_preds.shape[1])):
        plt.subplot(4, 3, bin_idx+1)
        mask = present_mask[:, bin_idx]
        if np.sum(mask) > 0:
            plt.scatter(distance_targets_rescaled[mask, bin_idx], 
                       distance_preds_rescaled[mask, bin_idx], 
                       alpha=0.3, s=10)
            # Plot perfect prediction line
            target_range = [np.min(distance_targets_rescaled[mask, bin_idx]), 
                           np.max(distance_targets_rescaled[mask, bin_idx])]
            plt.plot(target_range, target_range, 'r--', alpha=0.5)
            
            # Calculate correlation and error metrics for this bin
            corr = np.corrcoef(distance_preds_rescaled[mask, bin_idx], 
                             distance_targets_rescaled[mask, bin_idx])[0, 1]
            mae = np.mean(np.abs(distance_preds_rescaled[mask, bin_idx] - distance_targets_rescaled[mask, bin_idx]))
            mse = np.mean((distance_preds_rescaled[mask, bin_idx] - distance_targets_rescaled[mask, bin_idx])**2)
            
            plt.title(f'Bin {bin_idx}: corr={corr:.3f}')
            plt.text(0.05, 0.95, f'MAE={mae:.1f}', transform=plt.gca().transAxes, 
                    verticalalignment='top', fontsize=8)
            plt.text(0.05, 0.88, f'MSE={mse:.1f}', transform=plt.gca().transAxes, 
                    verticalalignment='top', fontsize=8)
        else:
            plt.title(f'Bin {bin_idx}: no data')
        plt.xlabel('True (mm)')
        plt.ylabel('Predicted (mm)')
        plt.grid(True, alpha=0.3)
    plt.suptitle(f'Distance Prediction Scatter Plots by Azimuth Bin - {dataset_name}', fontsize=16)
    plt.tight_layout()
    save_plot(f'bin_scatter_plots_{dataset_name.lower().replace(" ", "_")}')
    plt.show()


def create_presence_confusion_plots(model, data_loader, dataset_name, threshold=0.5):
    """Visualize TP/FP/TN/FN per azimuth bin for presence prediction."""
    print(f"📊 Creating presence confusion plots for {dataset_name}...")
    
    model.eval()
    all_presence_probs = []
    all_presence_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device, non_blocking=True)
            presence_targets = batch[1].to(device, non_blocking=True)
            presence_logits, _ = model(inputs)
            presence_probs = torch.sigmoid(presence_logits)
            
            all_presence_probs.append(presence_probs.cpu().numpy())
            all_presence_targets.append(presence_targets.cpu().numpy())
    
    presence_probs = np.concatenate(all_presence_probs, axis=0)
    presence_targets = np.concatenate(all_presence_targets, axis=0).astype(bool)
    presence_preds = (presence_probs >= threshold)
    
    tp = np.sum(presence_preds & presence_targets, axis=0)
    fp = np.sum(presence_preds & ~presence_targets, axis=0)
    tn = np.sum(~presence_preds & ~presence_targets, axis=0)
    fn = np.sum(~presence_preds & presence_targets, axis=0)
    
    confusion_df = pd.DataFrame({
        'bin': np.arange(presence_targets.shape[1]),
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    })
    
    # Derived metrics per bin
    confusion_df['precision'] = confusion_df['TP'] / np.maximum(confusion_df['TP'] + confusion_df['FP'], 1)
    confusion_df['recall'] = confusion_df['TP'] / np.maximum(confusion_df['TP'] + confusion_df['FN'], 1)
    confusion_df['specificity'] = confusion_df['TN'] / np.maximum(confusion_df['TN'] + confusion_df['FP'], 1)
    confusion_df['f1'] = (2 * confusion_df['precision'] * confusion_df['recall']) / np.maximum(
        confusion_df['precision'] + confusion_df['recall'], 1e-12
    )
    
    csv_name = f"{output_dir}/presence_confusion_{dataset_name.lower().replace(' ', '_')}.csv"
    confusion_df.to_csv(csv_name, index=False)
    print(f"💾 Saved presence confusion table: {csv_name}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    count_matrix = confusion_df[['TP', 'FP', 'TN', 'FN']].values
    sns.heatmap(
        count_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        yticklabels=[f"Bin {i}" for i in confusion_df['bin']],
        xticklabels=['TP', 'FP', 'TN', 'FN']
    )
    plt.title(f'Presence Confusion Counts - {dataset_name}')
    plt.xlabel('Outcome')
    plt.ylabel('Azimuth Bin')
    
    plt.subplot(1, 2, 2)
    row_sums = np.maximum(count_matrix.sum(axis=1, keepdims=True), 1)
    norm_matrix = count_matrix / row_sums
    sns.heatmap(
        norm_matrix,
        annot=True,
        fmt='.2f',
        cmap='Greens',
        vmin=0,
        vmax=1,
        yticklabels=[f"Bin {i}" for i in confusion_df['bin']],
        xticklabels=['TP', 'FP', 'TN', 'FN']
    )
    plt.title(f'Presence Confusion Fractions - {dataset_name}')
    plt.xlabel('Outcome')
    plt.ylabel('Azimuth Bin')
    
    plt.tight_layout()
    save_plot(f'presence_confusion_{dataset_name.lower().replace(" ", "_")}')
    plt.show()


def find_best_presence_threshold(model, val_loader):
    """Find presence threshold on validation set that maximizes micro-F1."""
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device, non_blocking=True)
            presence_targets = batch[1].to(device, non_blocking=True)
            presence_logits, _ = model(inputs)
            presence_probs = torch.sigmoid(presence_logits)
            all_probs.append(presence_probs.cpu().numpy())
            all_targets.append(presence_targets.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0).reshape(-1)
    targets = np.concatenate(all_targets, axis=0).reshape(-1) > 0.5

    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for thr in thresholds:
        preds = probs >= thr
        tp = np.sum(preds & targets)
        fp = np.sum(preds & ~targets)
        fn = np.sum(~preds & targets)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

        if (f1 > best_f1) or (np.isclose(f1, best_f1) and abs(thr - 0.5) < abs(best_threshold - 0.5)):
            best_f1 = f1
            best_threshold = float(thr)
            best_precision = float(precision)
            best_recall = float(recall)

    print(
        f"🎯 Best validation presence threshold: {best_threshold:.2f} "
        f"(F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f})"
    )
    return best_threshold

def main():
    """Main training pipeline for two-headed model."""
    print("🚀 Starting Two-Headed Model Training")
    print("=" * 60)
    model = None
    
    # Load and prepare data
    sonar_data, profiles_data, quadrants, rob_x, rob_y = load_and_prepare_data()
    
    # Create spatial split
    print(f"🎯 Using spatial split: Train on {train_quadrants}, Test on {test_quadrant}")
    
    X_train, X_val, X_test, y_train, y_val, y_test, train_mask, test_mask = create_spatial_split(
        sonar_data, profiles_data, quadrants,
        train_quadrants=train_quadrants, test_quadrant=test_quadrant
    )
        
    # Preprocess data
    train_loader, val_loader, test_loader, y_scaler, original_data = preprocess_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
        
    # Build model
    input_shape = (2, 200)  # (channels, time_samples)
    model = TwoHeadedSonarCNN(input_shape, profile_steps).to(device)
    print(f"🏗️ Model architecture:")
    print(model)
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
    # Train model
    start_time = time.time()
    train_losses, val_losses, presence_losses, distance_losses = train_model(
        model, train_loader, val_loader, epochs, patience
    )
    training_time = time.time() - start_time
    
    print(f"⏱️  Training completed in {training_time:.2f} seconds")
        
    print("\n✅ Two-Headed Model Training Complete!")
    print(f"📊 Final Results:")
    print(f"   Train Loss: {train_losses[-1]:.4f}")
    print(f"   Val Loss: {val_losses[-1]:.4f}")
    print(f"   Presence Loss: {presence_losses[-1]:.4f}")
    print(f"   Distance Loss: {distance_losses[-1]:.4f}")
        
    # Plot training curves
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Total Loss')
    plt.plot(val_losses, label='Val Total Loss')
    plt.plot(presence_losses, label='Train Presence Loss')
    plt.plot(distance_losses, label='Train Distance Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot('training_curves')
    plt.show()
        
    # Load best model for final evaluation
    model.load_state_dict(torch.load(f'{output_dir}/best_model_pytorch.pth', map_location=device))
    
    # Choose presence threshold on validation set and save with training artifacts.
    presence_threshold = find_best_presence_threshold(model, val_loader)
    
    # Save training parameters
    training_params = {
        'train_quadrants': train_quadrants,
        'test_quadrant': test_quadrant,
        'profile_opening_angle': profile_opening_angle,
        'profile_steps': profile_steps,
        'distance_threshold': distance_threshold,
        'presence_threshold': presence_threshold,
        'conv_filters': conv_filters,
        'kernel_size': kernel_size,
        'pool_size': pool_size,
        'dropout_rate': dropout_rate,
        'input_shape': (2, 200),
        'output_shape': profile_steps
    }
    
    with open(f'{output_dir}/training_params.json', 'w') as f:
        json.dump(training_params, f, indent=2)
    print(f"💾 Saved training parameters to: {output_dir}/training_params.json")
    
    # Save scaler
    joblib.dump(y_scaler, f'{output_dir}/y_scaler.joblib')
    print(f"💾 Saved distance scaler to: {output_dir}/y_scaler.joblib")
    write_training_readme()
        
    # Optional debug output
    if debug:
        print(f"Debug - Test distance targets shape: {original_data['y_test_distance'].shape}")
        print(f"Debug - Test distance targets range: [{np.min(original_data['y_test_distance']):.1f}, {np.max(original_data['y_test_distance']):.1f}]")
        
    # Create scatter plots for test set
    create_scatter_plots(
        model, test_loader, y_scaler, "Test Set"
    )
        
    # Create scatter plots for training set
    create_scatter_plots(
        model, train_loader, y_scaler, "Training Set"
    )
    
    # Presence confusion visualizations
    create_presence_confusion_plots(
        model, test_loader, "Test Set", threshold=presence_threshold
    )
    create_presence_confusion_plots(
        model, train_loader, "Training Set", threshold=presence_threshold
    )
        
    # Evaluate on test set for metrics
    model.eval()
    all_presence_preds = []
    all_distance_preds = []
    all_presence_targets = []
    all_distance_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device, non_blocking=True)
            presence_targets = batch[1].to(device, non_blocking=True)
            distance_targets = batch[2].to(device, non_blocking=True)
            
            presence_logits, distance_pred = model(inputs)
            presence_pred = torch.sigmoid(presence_logits)
            
            all_presence_preds.append(presence_pred.cpu().numpy())
            all_distance_preds.append(distance_pred.cpu().numpy())
            all_presence_targets.append(presence_targets.cpu().numpy())
            all_distance_targets.append(distance_targets.cpu().numpy())
    
    # Concatenate all batches
    presence_preds = np.concatenate(all_presence_preds, axis=0)
    distance_preds = np.concatenate(all_distance_preds, axis=0)
    presence_targets = np.concatenate(all_presence_targets, axis=0)
    distance_targets_scaled = np.concatenate(all_distance_targets, axis=0)
    
    # Convert distance outputs back to mm for reporting/plots.
    distance_preds_mm = y_scaler.inverse_transform(distance_preds)
    distance_targets_mm = y_scaler.inverse_transform(distance_targets_scaled)
    
    # Calculate test metrics
    presence_accuracy = (presence_preds > presence_threshold) == (presence_targets > 0.5)
    presence_acc = np.mean(presence_accuracy)
    
    # Only calculate distance metrics for cases where wall is present
    present_mask = presence_targets > 0.5
    if np.any(present_mask):
        distance_mse = np.mean((distance_preds_mm[present_mask] - distance_targets_mm[present_mask])**2)
        distance_mae = np.mean(np.abs(distance_preds_mm[present_mask] - distance_targets_mm[present_mask]))
    else:
        distance_mse = float('nan')
        distance_mae = float('nan')
    
    print(f"\n📊 Test Set Performance:")
    print(f"   Presence Accuracy: {presence_acc:.4f}")
    print(f"   Distance MSE (present only): {distance_mse:.4f}")
    print(f"   Distance MAE (present only): {distance_mae:.4f}")
    
    # Plot some test predictions
    plt.figure(figsize=(12, 8))
    for i in range(min(6, len(presence_preds))):
        ax1 = plt.subplot(3, 2, i+1)
        
        # Show presence prediction
        ax1.plot(presence_targets[i], 'b-', label='True Presence')
        ax1.plot(presence_preds[i], 'r--', label='Pred Presence')
        
        # Plot distance on a separate y-axis in mm.
        ax2 = ax1.twinx()
        if np.any(presence_targets[i] > 0.5):
            ax2.plot(distance_targets_mm[i], 'g-', label='True Distance (mm)')
            ax2.plot(distance_preds_mm[i], 'm--', label='Pred Distance (mm)')
            ax2.set_ylim(0, distance_threshold * 1.05)
        ax2.set_ylabel('Distance (mm)')
        
        ax1.set_title(f'Test Sample {i}')
        ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
        ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
        ax1.legend(ax1_handles + ax2_handles, ax1_labels + ax2_labels, fontsize=8, loc='upper right')
        ax1.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    save_plot('test_samples')
    plt.show()
    
    # Spatial analysis
    sample_errors = []
    for i in range(len(presence_preds)):
        if np.any(present_mask[i]):
            per_bin_error = np.abs(distance_preds_mm[i] - distance_targets_mm[i])
            sample_errors.append(np.mean(per_bin_error[present_mask[i]]))
        else:
            sample_errors.append(0)  # No error if no wall present
    
    # Plot spatial distribution of errors
    plt.figure(figsize=(12, 8))
    test_rob_x = rob_x[test_mask]
    test_rob_y = rob_y[test_mask]
    
    scatter = plt.scatter(test_rob_x, test_rob_y, c=sample_errors, 
                        cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Distance Error (mm)')
    
    # Add quadrant boundaries
    mean_x = np.mean(rob_x)
    mean_y = np.mean(rob_y)
    plt.axvline(mean_x, color='red', linestyle='--', alpha=0.5, label='Mean X')
    plt.axhline(mean_y, color='blue', linestyle='--', alpha=0.5, label='Mean Y')
    
    plt.title('Spatial Distribution of Test Errors')
    plt.xlabel('X position (mm)')
    plt.ylabel('Y position (mm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    save_plot('spatial_errors')
    plt.show()
    
    return model

if __name__ == "__main__":
    model = main()
