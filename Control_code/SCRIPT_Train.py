"""
Train a neural network to predict visual distance profiles from sonar data.
This script explores whether the spatial pattern of distances around the robot
can be predicted from what the sonar system detects.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from datetime import datetime

from Library import DataProcessor

# Configuration parameters
az_extent = 30  # Total azimuth range in degrees
az_steps = 61  # Number of azimuth steps in profiles
sessions = ['session03', 'session04', 'session06', 'session07']

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 250
TEST_SIZE = 0.2
VAL_SIZE = 0.1
HIDDEN_SIZE = 256
NUM_LAYERS = 3
PATIENCE = 50  # Early stopping patience

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def print_data_info(sonar_data, profiles):
    """Print information about the loaded datasets."""
    print(f"Sonar data shape: {sonar_data.shape}")
    print(f"Profiles shape: {profiles.shape}")
    print(f"Number of samples: {sonar_data.shape[0]}")
    print(f"Sonar input dimension: {sonar_data.shape[1]}")
    print(f"Profile output dimension: {profiles.shape[1]}")
    print(f"Sonar data range: [{np.min(sonar_data):.3f}, {np.max(sonar_data):.3f}]")
    print(f"Profile data range: [{np.min(profiles):.3f}, {np.max(profiles):.3f}]")

class ProfileMLP(nn.Module):
    """Multi-layer perceptron for predicting profiles from sonar data."""
    
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=3):
        super(ProfileMLP, self).__init__()
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

def create_dataloaders(sonar_data, profiles, batch_size=32, test_size=0.2, val_size=0.1):
    """Create train, validation, and test dataloaders."""
    
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sonar_data, profiles, test_size=test_size, random_state=42
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

def evaluate_model(model, test_loader, profile_scaler=None):
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
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate per-azimuth errors in standardized space
    azimuth_errors_scaled = np.mean(np.abs(all_predictions - all_targets), axis=0)
    
    # Convert to original scale if scaler is provided
    if profile_scaler is not None:
        # Calculate errors in original units
        predictions_original = profile_scaler.inverse_transform(all_predictions)
        targets_original = profile_scaler.inverse_transform(all_targets)
        azimuth_errors_original = np.mean(np.abs(predictions_original - targets_original), axis=0)
        
        mae_original = np.mean(np.abs(predictions_original - targets_original))
        rmse_original = np.sqrt(np.mean((predictions_original - targets_original)**2))
        
        print(f"Test Loss (MSE - standardized): {test_loss:.6f}")
        print(f"Test RMSE (standardized): {np.sqrt(test_loss):.6f}")
        print(f"Test MAE (standardized): {np.mean(np.abs(all_predictions - all_targets)):.6f}")
        print(f"\nTest MAE (original units): {mae_original:.1f} mm ({mae_original/1000:.3f} m)")
        print(f"Test RMSE (original units): {rmse_original:.1f} mm ({rmse_original/1000:.3f} m)")
        
        return test_loss, all_predictions, all_targets, azimuth_errors_original, predictions_original, targets_original
    else:
        print(f"Test Loss (MSE): {test_loss:.6f}")
        print(f"Test RMSE: {np.sqrt(test_loss):.6f}")
        print(f"Test MAE: {np.mean(np.abs(all_predictions - all_targets)):.6f}")
        
        return test_loss, all_predictions, all_targets, azimuth_errors_scaled, None, None

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

def plot_azimuth_errors(azimuth_errors, centers):
    """Plot prediction errors by azimuth angle."""
    plt.figure(figsize=(12, 6))
    plt.plot(centers, azimuth_errors)
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Mean Absolute Error (mm)')
    plt.title('Prediction Error by Azimuth Angle')
    plt.grid(True)
    mean_error_mm = np.mean(azimuth_errors)
    mean_error_m = mean_error_mm / 1000
    plt.axhline(y=mean_error_mm, color='r', linestyle='--', 
                label=f'Mean Error: {mean_error_mm:.1f}mm ({mean_error_m:.3f}m)')
    plt.legend()
    plt.show()

def plot_sample_predictions(predictions, targets, centers, num_samples=3):
    """Plot predictions vs targets for sample profiles."""
    indices = np.random.choice(len(predictions), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.figure(figsize=(12, 6))
        plt.plot(centers, targets[idx], 'b-', label='Ground Truth')
        plt.plot(centers, predictions[idx], 'r--', label='Prediction')
        plt.xlabel('Azimuth Angle (degrees)')
        plt.ylabel('Distance (mm)')
        plt.title(f'Sample {i+1} - Profile Prediction')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_error_distribution_by_azimuth(predictions, targets, centers):
    """Plot the distribution of errors for each azimuth direction."""
    errors = predictions - targets
    
    # Calculate error statistics per azimuth
    az_errors_mean = np.mean(errors, axis=0)
    az_errors_std = np.std(errors, axis=0)
    az_errors_mae = np.mean(np.abs(errors), axis=0)
    
    # Plot 1: Error distribution statistics
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Mean error by azimuth
    plt.subplot(3, 1, 1)
    plt.plot(centers, az_errors_mean, 'b-', label='Mean Error')
    plt.fill_between(centers, az_errors_mean - az_errors_std, az_errors_mean + az_errors_std, 
                     alpha=0.3, color='blue', label='±1 Std Dev')
    plt.axhline(y=0, color='r', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Error (mm)')
    plt.title('Mean Error ± Standard Deviation by Azimuth Direction')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: MAE by azimuth
    plt.subplot(3, 1, 2)
    plt.plot(centers, az_errors_mae, 'g-', label='MAE')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('MAE (mm)')
    plt.title('Mean Absolute Error by Azimuth Direction')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Error distribution histograms for selected directions
    plt.subplot(3, 1, 3)
    
    # Select a few representative azimuth directions
    selected_azimuths = [0, len(centers)//4, len(centers)//2, 3*len(centers)//4, -1]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    labels = [f'{int(centers[az]):d}°' for az in selected_azimuths]
    
    max_error = np.max(np.abs(errors))
    bins = np.linspace(-max_error, max_error, 30)
    
    for i, az_idx in enumerate(selected_azimuths):
        plt.hist(errors[:, az_idx], bins=bins, alpha=0.6, color=colors[i], 
                 label=labels[i], density=True)
    
    plt.xlabel('Error (mm)')
    plt.ylabel('Density')
    plt.title('Error Distribution for Selected Azimuth Directions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Error vs Distance scatter with azimuth coloring
    plt.figure(figsize=(12, 8))
    
    # Flatten all data
    all_errors = errors.flatten()
    all_distances = targets.flatten()
    all_azimuths = np.tile(centers, predictions.shape[0])
    
    # Create scatter plot colored by azimuth
    scatter = plt.scatter(all_distances, all_errors, c=all_azimuths, cmap='viridis', 
                         alpha=0.4, s=20)
    plt.colorbar(scatter, label='Azimuth Angle (degrees)')
    
    plt.axhline(y=0, color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Ground Truth Distance (mm)')
    plt.ylabel('Prediction Error (mm)')
    plt.title('Prediction Error vs Distance (Colored by Azimuth)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add overall statistics
    overall_mae = np.mean(np.abs(all_errors))
    overall_rmse = np.sqrt(np.mean(all_errors**2))
    stats_text = f'Overall MAE: {overall_mae:.1f} mm\nOverall RMSE: {overall_rmse:.1f} mm'
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()
    
    # Plot 3: Cumulative error distribution
    plt.figure(figsize=(12, 8))
    
    # Calculate absolute errors and sort them
    abs_errors = np.abs(all_errors)
    sorted_errors = np.sort(abs_errors)
    
    # Calculate cumulative distribution
    cumulative_fraction = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    # Plot cumulative distribution
    plt.plot(sorted_errors, cumulative_fraction, 'b-', linewidth=2, label='Cumulative Errors')
    
    # Add reference lines for common error thresholds
    error_thresholds = [50, 100, 200, 300, 500]  # mm
    colors = ['green', 'orange', 'red', 'purple', 'brown']
    
    for i, threshold in enumerate(error_thresholds):
        # Find the fraction of errors below this threshold
        fraction_below = np.sum(abs_errors <= threshold) / len(abs_errors)
        plt.axvline(x=threshold, color=colors[i], linestyle='--', alpha=0.7)
        plt.axhline(y=fraction_below, color=colors[i], linestyle='--', alpha=0.7)
        plt.text(threshold + 10, fraction_below + 0.02, 
                 f'{int(fraction_below*100)}% < {threshold}mm',
                 color=colors[i], fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.xlabel('Absolute Error (mm)')
    plt.ylabel('Cumulative Fraction of Predictions')
    plt.title('Cumulative Distribution of Absolute Errors')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, np.max(sorted_errors))
    plt.ylim(0, 1.05)
    
    # Add key statistics
    median_error = np.median(abs_errors)
    plt.axvline(median_error, color='black', linestyle=':', label=f'Median: {median_error:.1f}mm')
    
    # Calculate error thresholds for common percentages
    error_50 = np.percentile(abs_errors, 50)
    error_75 = np.percentile(abs_errors, 75)
    error_90 = np.percentile(abs_errors, 90)
    error_95 = np.percentile(abs_errors, 95)
    
    stats_text = f'50th percentile: {error_50:.1f}mm\n'
    stats_text += f'75th percentile: {error_75:.1f}mm\n'
    stats_text += f'90th percentile: {error_90:.1f}mm\n'
    stats_text += f'95th percentile: {error_95:.1f}mm'
    
    plt.text(0.98, 0.05, stats_text, transform=plt.gca().transAxes,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.show()
    
    # Print detailed error analysis
    print(f"\n{'='*60}")
    print("ERROR DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Overall statistics
    print(f"Overall Statistics:")
    print(f"  MAE: {overall_mae:.1f} mm ({overall_mae/1000:.3f} m)")
    print(f"  RMSE: {overall_rmse:.1f} mm ({overall_rmse/1000:.3f} m)")
    print(f"  Mean Error: {np.mean(all_errors):.1f} mm")
    print(f"  Std Error: {np.std(all_errors):.1f} mm")
    
    # Per-azimuth statistics
    print(f"\nPer-Azimuth Statistics:")
    print(f"  Best azimuth (lowest MAE): {centers[np.argmin(az_errors_mae)]:.1f}° ({np.min(az_errors_mae):.1f} mm)")
    print(f"  Worst azimuth (highest MAE): {centers[np.argmax(az_errors_mae)]:.1f}° ({np.max(az_errors_mae):.1f} mm)")
    print(f"  Mean MAE across azimuths: {np.mean(az_errors_mae):.1f} mm")
    print(f"  Std MAE across azimuths: {np.std(az_errors_mae):.1f} mm")
    
    # Error distribution characteristics
    error_percentiles = np.percentile(np.abs(all_errors), [25, 50, 75, 90, 95])
    print(f"\nError Distribution Percentiles:")
    print(f"  25th: {error_percentiles[0]:.1f} mm")
    print(f"  50th (Median): {error_percentiles[1]:.1f} mm")
    print(f"  75th: {error_percentiles[2]:.1f} mm")
    print(f"  90th: {error_percentiles[3]:.1f} mm")
    print(f"  95th: {error_percentiles[4]:.1f} mm")
    
    # Cumulative error analysis
    print(f"\nCumulative Error Analysis:")
    for threshold in [50, 100, 200, 300, 500]:
        fraction_below = np.sum(np.abs(all_errors) <= threshold) / len(all_errors)
        print(f"  {int(fraction_below*100)}% of predictions have error ≤ {threshold}mm")
    
    # Distance-dependent error analysis
    short_dist_errors = all_errors[all_distances < 1000]
    med_dist_errors = all_errors[(all_distances >= 1000) & (all_distances < 2000)]
    long_dist_errors = all_errors[all_distances >= 2000]
    
    print(f"\nDistance-Dependent Errors:")
    print(f"  Short range (<1m): MAE = {np.mean(np.abs(short_dist_errors)):.1f} mm, {len(short_dist_errors)} samples")
    print(f"  Medium range (1-2m): MAE = {np.mean(np.abs(med_dist_errors)):.1f} mm, {len(med_dist_errors)} samples")
    print(f"  Long range (>2m): MAE = {np.mean(np.abs(long_dist_errors)):.1f} mm, {len(long_dist_errors)} samples")
    
    print("="*60)

def save_model(model, filename):
    """Save the trained model."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'output_size': model.output_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers
    }, filename)
    print(f"Model saved to {filename}")

def main():
    print("\n" + "="*60)
    print("PROFILE PREDICTION FROM SONAR DATA")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    collection = DataProcessor.DataCollection(sessions, az_min=-az_extent, az_max=az_extent, az_steps=az_steps)
    profiles = collection.get_field('profiles')  # Shape: (n_samples, az_steps)
    sonar_data = collection.get_field('sonar_data')  # Shape: (n_samples, 200)
    
    # Get centers for plotting
    centers = collection.get_centers()
    
    print_data_info(sonar_data, profiles)
    
    # Data preprocessing - standardization
    print("\nPreprocessing data...")
    
    # Standardize sonar data
    sonar_scaler = StandardScaler()
    sonar_data_scaled = sonar_scaler.fit_transform(sonar_data)
    
    # Standardize profiles
    profile_scaler = StandardScaler()
    profiles_scaled = profile_scaler.fit_transform(profiles)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        sonar_data_scaled, profiles_scaled, 
        batch_size=BATCH_SIZE, 
        test_size=TEST_SIZE, 
        val_size=VAL_SIZE
    )
    
    # Create model
    print(f"\nCreating MLP model...")
    model = ProfileMLP(
        input_size=sonar_data.shape[1],
        output_size=profiles.shape[1],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    )
    
    print(f"Model architecture:")
    print(f"  Input size: {sonar_data.shape[1]}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Hidden layers: {NUM_LAYERS}")
    print(f"  Output size: {profiles.shape[1]}")
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
    test_loss, predictions_scaled, targets_scaled, azimuth_errors, predictions, targets = evaluate_model(
        trained_model, test_loader, profile_scaler=profile_scaler
    )
    
    # Plot results
    print("\nPlotting results...")
    plot_training_history(train_losses, val_losses)
    plot_azimuth_errors(azimuth_errors, centers)
    plot_error_distribution_by_azimuth(predictions, targets, centers)
    plot_sample_predictions(predictions, targets, centers, num_samples=10)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"profile_mlp_{timestamp}.pt"
    save_model(trained_model, model_filename)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Training time: {training_time:.1f} seconds")
    print(f"Final test MSE (standardized): {test_loss:.6f}")
    print(f"Final test RMSE (standardized): {np.sqrt(test_loss):.6f}")
    
    # Calculate final metrics in original units
    final_mae = np.mean(np.abs(predictions - targets))
    final_rmse = np.sqrt(np.mean((predictions - targets)**2))
    final_mean_azimuth_error = np.mean(azimuth_errors)
    
    print(f"Final test MAE (original): {final_mae:.1f} mm ({final_mae/1000:.3f} m)")
    print(f"Final test RMSE (original): {final_rmse:.1f} mm ({final_rmse/1000:.3f} m)")
    print(f"Mean azimuth error: {final_mean_azimuth_error:.1f} mm ({final_mean_azimuth_error/1000:.3f} m)")
    print(f"Model saved to: {model_filename}")
    print("="*60)

if __name__ == "__main__":
    main()
