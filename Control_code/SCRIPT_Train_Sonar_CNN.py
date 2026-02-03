"""
Train a 1D CNN to predict sonar data from high-resolution visual distance profiles.
This script explores whether spatial patterns in distance profiles can better predict
sonar readings using convolutional neural networks.
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
az_steps = 101  # Increased resolution: 101 azimuth steps (was 51)
sessions = ['session07']

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001  # Default for PCA
LEARNING_RATE_RAW = 0.0001  # Reduced for raw sonar prediction
EPOCHS = 300
TEST_SIZE = 0.2
VAL_SIZE = 0.1
HIDDEN_SIZE = 256
PATIENCE = 10  # Early stopping patience
USE_PCA = True  # Set to False to predict raw sonar data directly
PCA_COMPONENTS = 50  # Number of PCA components (only used if USE_PCA=True)
GRADIENT_CLIPPING = 1.0  # Clip gradients to prevent explosion

# CNN-specific parameters
CNN_KERNEL_SIZES = [21, 11, 7, 3]  # Multi-scale kernels
CNN_CHANNELS = [32, 64, 128, 128]  # Increasing channel depth
DROPOUT_RATE = 0.3

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
    print(f"Profile input dimension: {profiles.shape[1]} (azimuth steps)")
    print(f"Sonar output dimension: {sonar_data.shape[1]}")
    print(f"Profile data range: [{np.min(profiles):.1f}, {np.max(profiles):.1f}] mm")
    print(f"Sonar data range: [{np.min(sonar_data):.3f}, {np.max(sonar_data):.3f}]")

class SonarCNN(nn.Module):
    """1D CNN for predicting sonar data from azimuth profiles."""
    
    def __init__(self, input_length=101, output_size=50):
        super(SonarCNN, self).__init__()
        self.input_length = input_length
        self.output_size = output_size
        
        # 1D CNN layers for spatial feature extraction
        cnn_layers = []
        in_channels = 1
        
        for i, (out_channels, kernel_size) in enumerate(zip(CNN_CHANNELS, CNN_KERNEL_SIZES)):
            padding = kernel_size // 2  # Maintain spatial dimension
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=1, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE if i < len(CNN_CHANNELS)-1 else 0)
            ])
            in_channels = out_channels
            
            # Add pooling every other layer to reduce dimensionality
            if i % 2 == 1:
                cnn_layers.append(nn.MaxPool1d(2))
        
        self.cnn_layers = nn.Sequential(*cnn_layers)
        
        # Calculate flattened size after CNN
        self._calculate_flattened_size()
        
        # MLP layers for final prediction with better initialization
        self.mlp_layers = nn.Sequential(
            nn.Linear(self.flattened_size, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE//2),
            nn.BatchNorm1d(HIDDEN_SIZE//2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE//2),
            
            nn.Linear(HIDDEN_SIZE//2, output_size)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization for better stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _calculate_flattened_size(self):
        """Calculate the size after CNN layers."""
        with torch.no_grad():
            test_input = torch.zeros(1, 1, self.input_length)
            test_output = self.cnn_layers(test_input)
            self.flattened_size = test_output.numel()
            print(f"CNN output flattened size: {self.flattened_size}")
    
    def forward(self, x):
        # Add channel dimension for CNN (batch_size, 1, sequence_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x.squeeze(1)  # Remove extra dimension if present
            x = x.unsqueeze(1)
        
        # Debug shape check
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input after reshaping, got {x.dim()}D with shape {x.shape}")
        
        # CNN feature extraction
        cnn_features = self.cnn_layers(x)
        
        # Flatten for MLP
        flattened = cnn_features.view(cnn_features.size(0), -1)
        
        # Final prediction
        output = self.mlp_layers(flattened)
        
        return output

class HybridSonarModel(nn.Module):
    """Hybrid CNN+MLP model for sonar prediction."""
    
    def __init__(self, input_size=101, output_size=50):
        super(HybridSonarModel, self).__init__()
        
        # CNN branch for spatial features
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        # MLP branch for raw features
        self.mlp_branch = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Combined prediction
        self.combined = nn.Sequential(
            nn.Linear(32 + 32, 128),  # CNN features + MLP features
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, output_size)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # CNN branch
        if x.dim() == 2:
            cnn_input = x.unsqueeze(1)
        elif x.dim() == 4:
            cnn_input = x.squeeze(1).unsqueeze(1)
        else:
            cnn_input = x
            
        if cnn_input.dim() != 3:
            raise ValueError(f"Expected 3D input for CNN, got {cnn_input.dim()}D with shape {cnn_input.shape}")
            
        cnn_features = self.cnn_branch(cnn_input).squeeze(-1)
        
        # MLP branch
        mlp_features = self.mlp_branch(x)
        
        # Combine and predict
        combined = torch.cat([cnn_features, mlp_features], dim=1)
        return self.combined(combined)

def create_dataloaders(profiles, sonar_data, batch_size=32, test_size=0.2, val_size=0.1):
    """Create train, validation, and test dataloaders with proper reshaping for CNN."""
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        profiles, sonar_data, test_size=test_size, random_state=42
    )
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=42
    )
    
    # Create Tensor datasets with original shape (will reshape in forward pass)
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
    print(f"Input shape: {X_train.shape} (samples, azimuth_steps)")
    print(f"CNN expects: (batch, 1, {X_train.shape[1]}) after reshaping")
    
    # Store test data for later plotting
    test_data = {'X_test': X_test, 'y_test': y_test}
    
    return train_loader, val_loader, test_loader, test_data

def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, patience=5, use_gradient_clipping=False, clip_value=1.0):
    """Train the model with early stopping and gradient clipping."""
    
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
            
            # Debug shape check
            if epoch == 0 and batch_x.dim() != 2:
                print(f"Warning: Expected 2D input, got {batch_x.dim()}D with shape {batch_x.shape}")
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
            # Check for NaN values
            if torch.isnan(loss).any():
                print(f"⚠️  NaN loss detected at epoch {epoch+1}, batch {len(train_losses)}")
                print(f"   Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                print(f"   Target range: [{batch_y.min().item():.3f}, {batch_y.max().item():.3f}]")
                break
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Check for NaN in overall loss
        if np.isnan(train_loss):
            print(f"⚠️  NaN training loss at epoch {epoch+1}: {train_loss}")
            break
        
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
        
        # Check for NaN in validation loss
        if np.isnan(val_loss):
            print(f"⚠️  NaN validation loss at epoch {epoch+1}: {val_loss}")
            break
        
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

def evaluate_model(model, test_loader, pca=None, sonar_scaler=None, use_pca=False):
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
    
    # Calculate errors
    component_errors = np.mean(np.abs(all_predictions - all_targets), axis=0)
    
    if use_pca:
        print(f"Test Loss (MSE - PCA space): {test_loss:.6f}")
        print(f"Test RMSE (PCA space): {np.sqrt(test_loss):.6f}")
        print(f"Test MAE (PCA space): {np.mean(np.abs(all_predictions - all_targets)):.6f}")
        
        # Reconstruct full sonar data from PCA predictions
        if pca is not None and sonar_scaler is not None:
            # Reconstruct predicted sonar data
            predictions_reconstructed = pca.inverse_transform(all_predictions)
            predictions_reconstructed = sonar_scaler.inverse_transform(predictions_reconstructed)
            
            # Reconstruct target sonar data
            targets_reconstructed = pca.inverse_transform(all_targets)
            targets_reconstructed = sonar_scaler.inverse_transform(targets_reconstructed)
            
            # Calculate reconstruction errors
            reconstruction_errors = np.mean(np.abs(predictions_reconstructed - targets_reconstructed), axis=0)
            
            print(f"\nReconstruction metrics (original sonar space):")
            print(f"Reconstruction MAE: {np.mean(reconstruction_errors):.6f}")
            print(f"Reconstruction RMSE: {np.sqrt(np.mean(reconstruction_errors**2)):.6f}")
            
            return test_loss, all_predictions, all_targets, component_errors, predictions_reconstructed, targets_reconstructed, reconstruction_errors
        else:
            return test_loss, all_predictions, all_targets, component_errors, None, None, None
    else:
        # Raw sonar prediction metrics
        print(f"Test Loss (MSE - Raw sonar space): {test_loss:.6f}")
        print(f"Test RMSE (Raw sonar space): {np.sqrt(test_loss):.6f}")
        print(f"Test MAE (Raw sonar space): {np.mean(np.abs(all_predictions - all_targets)):.6f}")
        
        # Inverse transform to get back to original scale
        if sonar_scaler is not None:
            predictions_original = sonar_scaler.inverse_transform(all_predictions)
            targets_original = sonar_scaler.inverse_transform(all_targets)
            
            original_errors = np.mean(np.abs(predictions_original - targets_original), axis=0)
            print(f"\nOriginal scale metrics:")
            print(f"Original MAE: {np.mean(original_errors):.6f}")
            print(f"Original RMSE: {np.sqrt(np.mean(original_errors**2)):.6f}")
            
            return test_loss, all_predictions, all_targets, component_errors, predictions_original, targets_original, original_errors
        else:
            return test_loss, all_predictions, all_targets, component_errors, all_predictions, all_targets, component_errors

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
    plt.show()

def plot_sample_profiles_with_predictions(profiles, predictions, targets, num_samples=3):
    """Plot sample profiles with their predictions."""
    # Ensure we don't exceed the minimum length
    min_length = min(len(profiles), len(predictions), len(targets))
    if min_length == 0:
        print("No samples available for plotting")
        return
        
    indices = np.random.choice(min_length, min(num_samples, min_length), replace=False)
    
    for i, idx in enumerate(indices):
        plt.figure(figsize=(15, 8))
        
        # Plot profile
        plt.subplot(2, 1, 1)
        plt.plot(profiles[idx].flatten(), 'b-', label='Distance Profile')
        plt.xlabel('Azimuth Step')
        plt.ylabel('Distance (mm)')
        plt.title(f'Sample {i+1} - Distance Profile (Azimuth: ±{az_extent}°)')
        plt.grid(True)
        plt.legend()
        
        # Plot prediction vs target
        plt.subplot(2, 1, 2)
        plt.plot(targets[idx], 'g-', label='Ground Truth Sonar', alpha=0.7)
        plt.plot(predictions[idx], 'r--', label='Predicted Sonar', alpha=0.7)
        plt.xlabel('Sonar Channel')
        plt.ylabel('Signal Amplitude')
        plt.title(f'Sonar Prediction (PCA Space)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def save_model(model, filename, pca=None, profile_scaler=None, sonar_scaler=None, use_pca=False):
    """Save the trained model with preprocessing components."""
    save_dict = {
        'model_type': model.__class__.__name__,
        'model_state_dict': model.state_dict(),
        'input_length': getattr(model, 'input_length', None),
        'output_size': getattr(model, 'output_size', None),
        'use_pca': use_pca,
        'pca_components': PCA_COMPONENTS if use_pca else None,
        'cnn_kernel_sizes': CNN_KERNEL_SIZES,
        'cnn_channels': CNN_CHANNELS
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
    print("SONAR PREDICTION FROM PROFILES - CNN VERSION")
    print("="*60)
    
    # Load data with higher resolution
    print(f"\nLoading data with {az_steps} azimuth steps...")
    collection = DataProcessor.DataCollection(sessions, az_min=-az_extent, az_max=az_extent, az_steps=az_steps)
    profiles = collection.get_field('profiles')  # Shape: (n_samples, az_steps)
    sonar_data = collection.get_field('sonar_data')  # Shape: (n_samples, 200)
    
    # Check if masked sonar data is available
    try:
        masked_sonar_data = collection.get_field('masked_sonar_data')  # Shape: (n_samples, 200)
        use_masked_data = True
        print("Using masked sonar data for training")
    except:
        use_masked_data = False
        print("Masked sonar data not available, using original sonar data")
    
    # Use masked data if available, otherwise use original
    if use_masked_data:
        sonar_data = masked_sonar_data
    
    print_data_info(profiles, sonar_data)
    
    # Data preprocessing - standardization and PCA
    print("\nPreprocessing data...")
    
    # Standardize profiles
    profile_scaler = StandardScaler()
    profiles_scaled = profile_scaler.fit_transform(profiles)
    
    # Standardize sonar data
    sonar_scaler = StandardScaler()
    sonar_data_scaled = sonar_scaler.fit_transform(sonar_data)
    
    # Apply PCA to sonar data (optional)
    if USE_PCA:
        print(f"Applying PCA to sonar data (keeping {PCA_COMPONENTS} components)...")
        pca = PCA(n_components=PCA_COMPONENTS)
        sonar_target = pca.fit_transform(sonar_data_scaled)
        
        # Print PCA information
        print(f"PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_):.3f}")
        print(f"PCA components shape: {sonar_target.shape}")
        print(f"Individual component variances: {pca.explained_variance_ratio_[:10]}")  # Show first 10
    else:
        print("Using raw sonar data (no PCA dimensionality reduction)")
        sonar_target = sonar_data_scaled
        pca = None
        print(f"Target shape: {sonar_target.shape}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, test_data = create_dataloaders(
        profiles_scaled, sonar_target, 
        batch_size=BATCH_SIZE, 
        test_size=TEST_SIZE, 
        val_size=VAL_SIZE
    )
    
    # Create CNN model
    print(f"\nCreating CNN model...")
    output_size = PCA_COMPONENTS if USE_PCA else sonar_target.shape[1]
    model = SonarCNN(
        input_length=profiles.shape[1],
        output_size=output_size
    )
    
    print(f"Model architecture:")
    print(f"  Input: {profiles.shape[1]} azimuth steps")
    print(f"  CNN layers: {len(CNN_KERNEL_SIZES)} with kernels {CNN_KERNEL_SIZES}")
    print(f"  CNN channels: {CNN_CHANNELS}")
    output_desc = f"{PCA_COMPONENTS} PCA components" if USE_PCA else f"{output_size} raw sonar channels"
    print(f"  Output: {output_desc}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model with appropriate settings
    start_time = time.time()
    current_lr = LEARNING_RATE_RAW if not USE_PCA else LEARNING_RATE
    use_clipping = not USE_PCA  # Use gradient clipping for raw data
    
    print(f"Using learning rate: {current_lr}")
    print(f"Using gradient clipping: {use_clipping}")
    
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, 
        learning_rate=current_lr,
        patience=PATIENCE,
        use_gradient_clipping=use_clipping,
        clip_value=GRADIENT_CLIPPING
    )
    training_time = time.time() - start_time
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, predictions, targets, component_errors, predictions_final, targets_final, final_errors = evaluate_model(
        trained_model, test_loader, pca=pca, sonar_scaler=sonar_scaler, use_pca=USE_PCA
    )
    
    # Plot results
    print("\nPlotting results...")
    plot_training_history(train_losses, val_losses)
    plot_component_errors(component_errors)
    
    # Plot sample predictions with their corresponding profiles
    if predictions_final is not None:
        # Use the actual test profiles that correspond to the test predictions
        test_profiles = test_data['X_test']
        
        plot_sample_profiles_with_predictions(
            test_profiles, predictions_final, targets_final, num_samples=10
        )
    
    # Save model with preprocessing components
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "pca" if USE_PCA else "raw"
    model_filename = f"sonar_cnn_{model_type}_{timestamp}.pt"
    save_model(trained_model, model_filename, pca=pca, profile_scaler=profile_scaler, sonar_scaler=sonar_scaler, use_pca=USE_PCA)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Training time: {training_time:.1f} seconds")
    
    if USE_PCA:
        print(f"Final test MSE (PCA space): {test_loss:.6f}")
        print(f"Final test RMSE (PCA space): {np.sqrt(test_loss):.6f}")
        print(f"Final test MAE (PCA space): {np.mean(np.abs(predictions - targets)):.6f}")
        print(f"Mean component error: {np.mean(component_errors):.6f}")
        
        if pca is not None:
            print(f"\nReconstruction metrics (original sonar space):")
            print(f"Reconstruction MAE: {np.mean(final_errors):.6f}")
            print(f"Reconstruction RMSE: {np.sqrt(np.mean(final_errors**2)):.6f}")
            print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
    else:
        print(f"Final test MSE (Raw sonar space): {test_loss:.6f}")
        print(f"Final test RMSE (Raw sonar space): {np.sqrt(test_loss):.6f}")
        print(f"Final test MAE (Raw sonar space): {np.mean(np.abs(predictions - targets)):.6f}")
        print(f"Mean channel error: {np.mean(component_errors):.6f}")
        
        if sonar_scaler is not None:
            print(f"\nOriginal scale metrics:")
            print(f"Original MAE: {np.mean(final_errors):.6f}")
            print(f"Original RMSE: {np.sqrt(np.mean(final_errors**2)):.6f}")
    
    print(f"Model saved to: {model_filename}")
    print("="*60)

if __name__ == "__main__":
    main()