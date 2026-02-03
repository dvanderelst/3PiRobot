"""
Train a neural network to predict sonar data from visual distance profiles.
This script explores whether the spatial pattern of distances around the robot
can be used to predict what the sonar system would detect.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from datetime import datetime

from Library import DataProcessor

# Configuration parameters
az_extent = 45  # Total azimuth range in degrees
az_steps = 5  # Number of azimuth steps in profiles
sessions = ['session03', 'session04', 'session06', 'session07']

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.1
HIDDEN_SIZE = 512
NUM_LAYERS = 3
PATIENCE = 50  # Early stopping patience
PCA_COMPONENTS = 50  # Number of PCA components to use for sonar data

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def print_data_info(profiles, sonar_data):
    """Print information about the loaded datasets."""
    print(f"Profiles shape: {profiles.shape}")
    print(f"Sonar data shape: {sonar_data.shape}")
    print(f"Number of samples: {profiles.shape[0]}")
    print(f"Profile input dimension: {profiles.shape[1]}")
    print(f"Sonar output dimension: {sonar_data.shape[1]}")
    print(f"Profile data range: [{np.min(profiles):.1f}, {np.max(profiles):.1f}] mm")
    print(f"Sonar data range: [{np.min(sonar_data):.3f}, {np.max(sonar_data):.3f}]")

class SonarMLP(nn.Module):
    """Multi-layer perceptron for predicting sonar data from profiles."""
    
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=3):
        super(SonarMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create layers
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def create_dataloaders(profiles, sonar_data, batch_size=32, test_size=0.2, val_size=0.1):
    """Create train, validation, and test dataloaders."""
    
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        profiles, sonar_data, test_size=test_size, random_state=42
    )
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=42
    )
    
    # Create Tensor datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, patience=5):
    """Train the model with early stopping."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    
    print(f"Training on {device}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model weights
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, pca=None, sonar_scaler=None):
    """Evaluate the model on test data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    test_loss = 0.0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    
    # Concatenate all predictions and targets
    all_predictions_pca = np.concatenate(all_predictions, axis=0)
    all_targets_pca = np.concatenate(all_targets, axis=0)
    
    # Calculate PCA component errors
    component_errors = np.mean(np.abs(all_predictions_pca - all_targets_pca), axis=0)
    
    print(f"Test Loss (MSE - PCA space): {test_loss:.6f}")
    print(f"Test RMSE (PCA space): {np.sqrt(test_loss):.6f}")
    print(f"Test MAE (PCA space): {np.mean(np.abs(all_predictions_pca - all_targets_pca)):.6f}")
    
    # Reconstruct full sonar data from PCA predictions
    if pca is not None and sonar_scaler is not None:
        # Reconstruct predicted sonar data
        predictions_reconstructed = pca.inverse_transform(all_predictions_pca)
        predictions_reconstructed = sonar_scaler.inverse_transform(predictions_reconstructed)
        
        # Reconstruct target sonar data
        targets_reconstructed = pca.inverse_transform(all_targets_pca)
        targets_reconstructed = sonar_scaler.inverse_transform(targets_reconstructed)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.abs(predictions_reconstructed - targets_reconstructed), axis=0)
        
        print(f"\nReconstruction metrics (original sonar space):")
        print(f"Reconstruction MAE: {np.mean(reconstruction_errors):.6f}")
        print(f"Reconstruction RMSE: {np.sqrt(np.mean(reconstruction_errors**2)):.6f}")
        
        return test_loss, all_predictions_pca, all_targets_pca, component_errors, predictions_reconstructed, targets_reconstructed, reconstruction_errors
    else:
        return test_loss, all_predictions_pca, all_targets_pca, component_errors, None, None, None

def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss history."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_component_errors(component_errors):
    """Plot prediction errors by PCA component."""
    plt.figure(figsize=(12, 6))
    components = np.arange(len(component_errors))
    plt.bar(components, component_errors, alpha=0.7)
    plt.xlabel('PCA Component Index')
    plt.ylabel('Mean Absolute Error')
    plt.title('Prediction Error by PCA Component')
    plt.grid(True, axis='y')
    
    # Add explained variance information if available
    if hasattr(plot_component_errors, 'explained_variance_ratio'):
        plt.twinx()
        plt.plot(components, plot_component_errors.explained_variance_ratio * 100, 'r-', alpha=0.5)
        plt.ylabel('Explained Variance (%)', color='r')
    
    plt.show()

def plot_sample_predictions(predictions, targets, num_samples=3):
    """Plot predictions vs targets for sample sonar signals."""
    indices = np.random.choice(len(predictions), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.figure(figsize=(12, 6))
        plt.plot(targets[idx], 'b-', label='Ground Truth')
        plt.plot(predictions[idx], 'r--', label='Prediction')
        plt.xlabel('Sonar Channel')
        plt.ylabel('Signal Amplitude')
        plt.title(f'Sample {i+1} - Sonar Signal Prediction')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_error_distribution_by_channel(predictions, targets):
    """Plot the distribution of errors for each sonar channel."""
    errors = predictions - targets
    
    # Calculate error statistics per channel
    channel_errors_mean = np.mean(errors, axis=0)
    channel_errors_std = np.std(errors, axis=0)
    channel_errors_mae = np.mean(np.abs(errors), axis=0)
    
    # Plot error distribution statistics
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Mean error by channel
    plt.subplot(3, 1, 1)
    plt.plot(channel_errors_mean, 'b-', label='Mean Error')
    plt.fill_between(np.arange(len(channel_errors_mean)), 
                     channel_errors_mean - channel_errors_std, 
                     channel_errors_mean + channel_errors_std, 
                     alpha=0.3, color='blue', label='±1 Std Dev')
    plt.axhline(y=0, color='r', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Sonar Channel')
    plt.ylabel('Error')
    plt.title('Mean Error ± Standard Deviation by Channel')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: MAE by channel
    plt.subplot(3, 1, 2)
    plt.plot(channel_errors_mae, 'g-', label='MAE')
    plt.xlabel('Sonar Channel')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error by Channel')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Error distribution histograms for selected channels
    plt.subplot(3, 1, 3)
    
    # Select a few representative channels
    selected_channels = [0, 50, 100, 150, -1]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    labels = [f'Channel {ch}' for ch in selected_channels]
    
    max_error = np.max(np.abs(errors))
    bins = np.linspace(-max_error, max_error, 30)
    
    for i, ch_idx in enumerate(selected_channels):
        plt.hist(errors[:, ch_idx], bins=bins, alpha=0.6, color=colors[i], 
                 label=labels[i], density=True)
    
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.title('Error Distribution for Selected Channels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed error analysis
    print(f"\n{'='*60}")
    print("ERROR DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Overall statistics
    all_errors = errors.flatten()
    overall_mae = np.mean(np.abs(all_errors))
    overall_rmse = np.sqrt(np.mean(all_errors**2))
    
    print(f"Overall Statistics:")
    print(f"  MAE: {overall_mae:.6f}")
    print(f"  RMSE: {overall_rmse:.6f}")
    print(f"  Mean Error: {np.mean(all_errors):.6f}")
    print(f"  Std Error: {np.std(all_errors):.6f}")
    
    # Per-channel statistics
    print(f"\nPer-Channel Statistics:")
    print(f"  Best channel (lowest MAE): {np.argmin(channel_errors_mae)} ({np.min(channel_errors_mae):.6f})")
    print(f"  Worst channel (highest MAE): {np.argmax(channel_errors_mae)} ({np.max(channel_errors_mae):.6f})")
    print(f"  Mean MAE across channels: {np.mean(channel_errors_mae):.6f}")
    print(f"  Std MAE across channels: {np.std(channel_errors_mae):.6f}")
    
    # Error distribution characteristics
    error_percentiles = np.percentile(np.abs(all_errors), [25, 50, 75, 90, 95])
    print(f"\nError Distribution Percentiles:")
    print(f"  25th: {error_percentiles[0]:.6f}")
    print(f"  50th (Median): {error_percentiles[1]:.6f}")
    print(f"  75th: {error_percentiles[2]:.6f}")
    print(f"  90th: {error_percentiles[3]:.6f}")
    print(f"  95th: {error_percentiles[4]:.6f}")
    
    print("="*60)

def save_model(model, filename, pca=None, profile_scaler=None, sonar_scaler=None):
    """Save the trained model with preprocessing components."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'output_size': model.output_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'pca_components': PCA_COMPONENTS
    }
    
    # Save PCA and scalers if provided
    if pca is not None:
        save_dict['pca_mean'] = pca.mean_
        save_dict['pca_components'] = pca.components_
        save_dict['pca_explained_variance'] = pca.explained_variance_
        save_dict['pca_explained_variance_ratio'] = pca.explained_variance_ratio_
    
    if profile_scaler is not None:
        save_dict['profile_scaler_mean'] = profile_scaler.mean_
        save_dict['profile_scaler_scale'] = profile_scaler.scale_
    
    if sonar_scaler is not None:
        save_dict['sonar_scaler_mean'] = sonar_scaler.mean_
        save_dict['sonar_scaler_scale'] = sonar_scaler.scale_
    
    torch.save(save_dict, filename)
    print(f"Model and preprocessing components saved to {filename}")

def main():
    print("\n" + "="*60)
    print("SONAR PREDICTION FROM PROFILES")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    collection = DataProcessor.DataCollection(sessions, az_min=-az_extent, az_max=az_extent, az_steps=az_steps)
    profiles = collection.get_field('profiles')  # Shape: (n_samples, az_steps)
    sonar_data = collection.get_field('sonar_data')  # Shape: (n_samples, 200)
    
    print_data_info(profiles, sonar_data)
    
    # Data preprocessing - standardization and PCA
    print("\nPreprocessing data...")
    
    # Standardize profiles
    profile_scaler = StandardScaler()
    profiles_scaled = profile_scaler.fit_transform(profiles)
    
    # Standardize sonar data
    sonar_scaler = StandardScaler()
    sonar_data_scaled = sonar_scaler.fit_transform(sonar_data)
    
    # Apply PCA to sonar data
    print(f"Applying PCA to sonar data (keeping {PCA_COMPONENTS} components)...")
    pca = PCA(n_components=PCA_COMPONENTS)
    sonar_pca = pca.fit_transform(sonar_data_scaled)
    
    # Print PCA information
    print(f"PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_):.3f}")
    print(f"PCA components shape: {sonar_pca.shape}")
    
    # Create dataloaders with PCA components
    train_loader, val_loader, test_loader = create_dataloaders(
        profiles_scaled, sonar_pca, 
        batch_size=BATCH_SIZE, 
        test_size=TEST_SIZE, 
        val_size=VAL_SIZE
    )
    
    # Create model
    print(f"\nCreating MLP model...")
    model = SonarMLP(
        input_size=profiles.shape[1],
        output_size=PCA_COMPONENTS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    )
    
    print(f"Model architecture:")
    print(f"  Input size: {profiles.shape[1]} (profile dimensions)")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Hidden layers: {NUM_LAYERS}")
    print(f"  Output size: {PCA_COMPONENTS} (PCA components)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    start_time = time.time()
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, 
        learning_rate=LEARNING_RATE,
        patience=PATIENCE
    )
    training_time = time.time() - start_time
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, predictions_pca, targets_pca, component_errors, predictions_reconstructed, targets_reconstructed, reconstruction_errors = evaluate_model(
        trained_model, test_loader, pca=pca, sonar_scaler=sonar_scaler
    )
    
    # Plot results
    print("\nPlotting results...")
    plot_training_history(train_losses, val_losses)
    plot_component_errors(component_errors)
    if predictions_reconstructed is not None:
        plot_error_distribution_by_channel(predictions_reconstructed, targets_reconstructed)
        plot_sample_predictions(predictions_reconstructed, targets_reconstructed, num_samples=7)
    
    # Save model with preprocessing components
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"sonar_mlp_pca_{timestamp}.pt"
    save_model(trained_model, model_filename, pca=pca, profile_scaler=profile_scaler, sonar_scaler=sonar_scaler)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Training time: {training_time:.1f} seconds")
    print(f"Final test MSE (PCA space): {test_loss:.6f}")
    print(f"Final test RMSE (PCA space): {np.sqrt(test_loss):.6f}")
    print(f"Final test MAE (PCA space): {np.mean(np.abs(predictions_pca - targets_pca)):.6f}")
    print(f"Mean component error: {np.mean(component_errors):.6f}")
    
    if reconstruction_errors is not None:
        print(f"\nReconstruction metrics (original sonar space):")
        print(f"Reconstruction MAE: {np.mean(reconstruction_errors):.6f}")
        print(f"Reconstruction RMSE: {np.sqrt(np.mean(reconstruction_errors**2)):.6f}")
        print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    print(f"Model saved to: {model_filename}")
    print("="*60)

if __name__ == "__main__":
    main()