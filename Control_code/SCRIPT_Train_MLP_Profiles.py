#!/usr/bin/env python3

"""
Train an MLP to predict distance profiles from sonar data.

This script:
1. Loads sonar data and distance profiles from multiple sessions
2. Creates and trains an MLP model
3. Evaluates model performance
4. Saves the trained model and training history
"""

import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from Library import DataProcessor

# ============================================
# CONFIGURATION SETTINGS (Easily modifiable)
# ============================================

# Data Configuration
SESSIONS = ['session03', 'session04', 'session06', 'session07']  # Sessions to use
OPENING_ANGLE = 45  # degrees (field of view)
PROFILE_STEPS = 9  # number of azimuth steps in profiles

# Data Split Configuration
TEST_SIZE = 0.2  # Fraction of data for testing
VAL_SIZE = 0.1  # Fraction of training data for validation
RANDOM_STATE = 42  # Random seed for reproducibility

# Model Architecture Configuration
INPUT_SIZE = 200  # Flattened sonar data (100 samples × 2 channels)
HIDDEN_SIZES = [256, 128, 64]  # Hidden layer sizes
OUTPUT_SIZE = PROFILE_STEPS  # Predict distance at each azimuth step
DROPOUT_RATE = 0.2  # Dropout rate for regularization

# Training Configuration
EPOCHS = 200  # Maximum number of epochs
BATCH_SIZE = 32  # Batch size for training
LEARNING_RATE = 0.001  # Initial learning rate
PATIENCE = 15  # Early stopping patience

# ============================================
# END CONFIGURATION SETTINGS
# ============================================

def create_mlp_model(input_size, hidden_sizes, output_size, dropout_rate=0.2):
    """
    Create an MLP model with configurable architecture.
    
    Args:
        input_size: Size of input features
        hidden_sizes: List of hidden layer sizes
        output_size: Size of output layer
        dropout_rate: Dropout rate for regularization
        
    Returns:
        torch.nn.Module: MLP model
    """
    layers = []
    
    # Input layer
    prev_size = input_size
    for i, hidden_size in enumerate(hidden_sizes):
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        prev_size = hidden_size
    
    # Output layer
    layers.append(nn.Linear(prev_size, output_size))
    
    return nn.Sequential(*layers)

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
    """
    Train the MLP model with early stopping.
    
    Args:
        model: MLP model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
        
    Returns:
        dict: Training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"🚀 Training on {device}...")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"   Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    os.remove('best_model.pth')
    
    return history

def evaluate_model(model, test_loader):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        
    Returns:
        dict: Evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    test_loss = 0.0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate per-profile metrics
    profile_errors = np.mean(np.abs(predictions - targets), axis=1)
    mean_absolute_error = np.mean(profile_errors)
    std_absolute_error = np.std(profile_errors)
    
    # Calculate per-azimuth-bin metrics
    az_bin_errors = np.mean(np.abs(predictions - targets), axis=0)
    az_bin_correlations = []
    az_bin_r2_scores = []
    
    for i in range(predictions.shape[1]):
        # Pearson correlation
        corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
        az_bin_correlations.append(corr)
        
        # R² score
        ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        az_bin_r2_scores.append(r2)
    
    return {
        'test_loss': test_loss,
        'mean_absolute_error': mean_absolute_error,
        'std_absolute_error': std_absolute_error,
        'profile_errors': profile_errors,
        'az_bin_errors': az_bin_errors,
        'az_bin_correlations': az_bin_correlations,
        'az_bin_r2_scores': az_bin_r2_scores,
        'predictions': predictions,
        'targets': targets
    }

def plot_training_history(history, results_dir='.'):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        results_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 8))
    
    # Loss plot
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate plot
    plt.subplot(2, 1, 2)
    plt.plot(history['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_prediction_examples(predictions, targets, num_examples=5, results_dir='.'):
    """
    Plot example predictions vs targets.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        num_examples: Number of examples to plot
        results_dir: Directory to save plots
    """
    # Select random examples
    indices = np.random.choice(len(predictions), num_examples, replace=False)
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(indices):
        plt.subplot(num_examples, 1, i + 1)
        
        # Get the profile data
        pred_profile = predictions[idx]
        target_profile = targets[idx]
        
        # Create azimuth angles for plotting
        azimuths = np.linspace(-45, 45, len(pred_profile))  # Assuming 90° opening angle
        
        plt.plot(azimuths, pred_profile, 'r-', label='Predicted')
        plt.plot(azimuths, target_profile, 'b--', label='Target')
        plt.fill_between(azimuths, pred_profile, target_profile, color='gray', alpha=0.3)
        
        plt.xlabel('Azimuth (degrees)')
        plt.ylabel('Distance (mm)')
        plt.title(f'Example {i + 1} - MAE: {np.mean(np.abs(pred_profile - target_profile)):.1f}mm')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if i == 0:
            plt.legend(loc='upper right')
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'prediction_examples.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_azimuth_diagnostics(azimuths, az_bin_errors, az_bin_correlations, az_bin_r2_scores, predictions, targets, results_dir='.'):
    """
    Plot per-azimuth-bin diagnostics with correlation scatter plots.
    
    Args:
        azimuths: Array of azimuth angles
        az_bin_errors: Mean absolute errors per azimuth bin
        az_bin_correlations: Pearson correlations per azimuth bin
        az_bin_r2_scores: R² scores per azimuth bin
        predictions: Model predictions array
        targets: Ground truth targets array
        results_dir: Directory to save plots
    """
    # Create main diagnostics figure
    plt.figure(figsize=(15, 12))
    
    # Error plot
    plt.subplot(4, 1, 1)
    plt.bar(azimuths, az_bin_errors, width=4, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.axhline(np.mean(az_bin_errors), color='red', linestyle='--', label='Mean Error')
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Mean Absolute Error (mm)')
    plt.title('Per-Azimuth-Bin Prediction Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Correlation plot
    plt.subplot(4, 1, 2)
    plt.bar(azimuths, az_bin_correlations, width=4, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Pearson Correlation')
    plt.title('Per-Azimuth-Bin Correlation (Predicted vs Target)')
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    
    # R² score plot
    plt.subplot(4, 1, 3)
    plt.bar(azimuths, az_bin_r2_scores, width=4, alpha=0.7, color='lightcoral', edgecolor='darkred')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('R² Score')
    plt.title('Per-Azimuth-Bin R² Score')
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'azimuth_diagnostics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create correlation scatter plots for best and worst azimuths
    create_azimuth_correlation_plots(azimuths, predictions, targets, az_bin_correlations, results_dir)

def create_azimuth_correlation_plots(azimuths, predictions, targets, correlations, results_dir):
    """
    Create scatter plots showing correlation between predicted and target values
    for ALL azimuth bins.
    
    Args:
        azimuths: Array of azimuth angles
        predictions: Model predictions array
        targets: Ground truth targets array
        correlations: Pearson correlations per azimuth bin
        results_dir: Directory to save plots
    """
    n_bins = len(azimuths)
    
    # Create grid of scatter plots for all azimuths
    plt.figure(figsize=(20, 15))
    
    # Calculate grid size (5 columns x 4 rows for 20 bins)
    cols = 5
    rows = int(np.ceil(n_bins / cols))
    
    for i in range(n_bins):
        plt.subplot(rows, cols, i + 1)
        
        # Scatter plot with color indicating correlation strength
        corr = correlations[i]
        if corr > 0.7:
            color = 'blue'  # High correlation
        elif corr > 0.5:
            color = 'green'  # Medium correlation
        else:
            color = 'orange'  # Low correlation
        
        plt.scatter(targets[:, i], predictions[:, i], alpha=0.4, color=color, s=20)
        
        # Perfect correlation line
        data_range = [min(targets[:, i]), max(targets[:, i])]
        plt.plot(data_range, data_range, 'r--', linewidth=1, alpha=0.7)
        
        plt.xlabel('Target (mm)', fontsize=8)
        plt.ylabel('Predicted (mm)', fontsize=8)
        plt.title(f'{azimuths[i]:.0f}° (corr: {corr:.2f})', fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'all_azimuth_correlation_scatter.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create the best/worst comparison plot
    create_best_worst_correlation_plots(azimuths, predictions, targets, correlations, results_dir)

def create_best_worst_correlation_plots(azimuths, predictions, targets, correlations, results_dir):
    """
    Create comparison scatter plots for best and worst performing azimuth bins.
    
    Args:
        azimuths: Array of azimuth angles
        predictions: Model predictions array
        targets: Ground truth targets array
        correlations: Pearson correlations per azimuth bin
        results_dir: Directory to save plots
    """
    # Find best and worst azimuth indices
    best_idx = np.argmax(correlations)
    worst_idx = np.argmin(correlations)
    best_az = azimuths[best_idx]
    worst_az = azimuths[worst_idx]
    
    plt.figure(figsize=(15, 6))
    
    # Best azimuth correlation plot
    plt.subplot(1, 2, 1)
    plt.scatter(targets[:, best_idx], predictions[:, best_idx], alpha=0.5, color='blue')
    plt.plot([min(targets[:, best_idx]), max(targets[:, best_idx])], 
             [min(targets[:, best_idx]), max(targets[:, best_idx])], 
             'r--', linewidth=2, label='Perfect correlation')
    plt.xlabel('Target Distance (mm)')
    plt.ylabel('Predicted Distance (mm)')
    plt.title(f'Best Azimuth: {best_az:.1f}° (corr: {correlations[best_idx]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Worst azimuth correlation plot
    plt.subplot(1, 2, 2)
    plt.scatter(targets[:, worst_idx], predictions[:, worst_idx], alpha=0.5, color='orange')
    plt.plot([min(targets[:, worst_idx]), max(targets[:, worst_idx])], 
             [min(targets[:, worst_idx]), max(targets[:, worst_idx])], 
             'r--', linewidth=2, label='Perfect correlation')
    plt.xlabel('Target Distance (mm)')
    plt.ylabel('Predicted Distance (mm)')
    plt.title(f'Worst Azimuth: {worst_az:.1f}° (corr: {correlations[worst_idx]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'best_worst_correlation_scatter.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """
    Main training script.
    """
    print("🤖 MLP Training: Sonar → Distance Profiles")
    print("=" * 50)
    
    # Use global configuration settings
    global SESSIONS, OPENING_ANGLE, PROFILE_STEPS, TEST_SIZE, VAL_SIZE, RANDOM_STATE
    global INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE, DROPOUT_RATE
    global EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE
    
    print(f"📊 Loading data from sessions: {SESSIONS}")
    
    # Load data
    dc = DataProcessor.DataCollection(SESSIONS)
    
    # Load sonar data (flattened)
    sonar_data = dc.load_sonar(flatten=True)
    print(f"   Sonar data shape: {sonar_data.shape}")
    
    # Load distance profiles
    profiles, centers = dc.load_profiles(opening_angle=OPENING_ANGLE, steps=PROFILE_STEPS)
    print(f"   Profiles shape: {profiles.shape}")
    print(f"   Centers shape: {centers.shape}")
    
    # Split data into train, validation, and test sets
    print(f"\n🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        sonar_data, profiles, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Validation set: {len(X_val)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Standardize the data
    print(f"\n📏 Standardizing data...")
    
    # Fit scaler on training data only
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    
    # Transform validation and test data
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print(f"\n🤖 Creating MLP model...")
    print(f"   Architecture: {INPUT_SIZE} → {' → '.join(map(str, HIDDEN_SIZES))} → {OUTPUT_SIZE}")
    
    model = create_mlp_model(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE, DROPOUT_RATE)
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print(f"\n🎓 Training model...")
    history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE
    )
    
    # Evaluate model
    print(f"\n📊 Evaluating model...")
    evaluation = evaluate_model(model, test_loader)
    
    print(f"   Test Loss (MSE): {evaluation['test_loss']:.6f}")
    print(f"   Mean Absolute Error: {evaluation['mean_absolute_error']:.1f}mm")
    print(f"   Std Absolute Error: {evaluation['std_absolute_error']:.1f}mm")
    
    # Print per-azimuth-bin diagnostics
    print(f"\n📈 Per-Azimuth-Bin Diagnostics:")
    azimuths = np.linspace(-45, 45, len(evaluation['az_bin_errors']))
    
    print(f"   {'Azimuth':<10} {'MAE (mm)':<12} {'Correlation':<12} {'R² Score':<10}")
    print(f"   {'-'*45}")
    
    for i, (az, err, corr, r2) in enumerate(zip(azimuths, 
                                                  evaluation['az_bin_errors'],
                                                  evaluation['az_bin_correlations'],
                                                  evaluation['az_bin_r2_scores'])):
        print(f"   {az:6.1f}°      {err:8.1f}      {corr:8.3f}       {r2:6.3f}")
    
    # Calculate and print summary statistics
    mean_corr = np.mean(evaluation['az_bin_correlations'])
    mean_r2 = np.mean(evaluation['az_bin_r2_scores'])
    best_az = azimuths[np.argmax(evaluation['az_bin_correlations'])]
    worst_az = azimuths[np.argmin(evaluation['az_bin_correlations'])]
    
    print(f"\n   Summary Statistics:")
    print(f"   Mean Correlation: {mean_corr:.3f}")
    print(f"   Mean R² Score: {mean_r2:.3f}")
    print(f"   Best Azimuth: {best_az:.1f}° (corr: {np.max(evaluation['az_bin_correlations']):.3f})")
    print(f"   Worst Azimuth: {worst_az:.1f}° (corr: {np.min(evaluation['az_bin_correlations']):.3f})")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'mlp_profiles_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot results
    print(f"\n📈 Generating plots...")
    plot_training_history(history, results_dir)
    plot_prediction_examples(evaluation['predictions'], evaluation['targets'], num_examples=5, results_dir=results_dir)
    plot_azimuth_diagnostics(azimuths, evaluation['az_bin_errors'], 
                           evaluation['az_bin_correlations'], evaluation['az_bin_r2_scores'], 
                           evaluation['predictions'], evaluation['targets'], results_dir)
    
    # Save results
    print(f"\n💾 Saving results to: {results_dir}")
    
    # Save model
    model_path = os.path.join(results_dir, 'mlp_profiles_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history_path = os.path.join(results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save evaluation metrics
    evaluation_path = os.path.join(results_dir, 'evaluation_metrics.json')
    with open(evaluation_path, 'w') as f:
        json.dump({
            'test_loss': float(evaluation['test_loss']),
            'mean_absolute_error': float(evaluation['mean_absolute_error']),
            'std_absolute_error': float(evaluation['std_absolute_error']),
            'profile_errors': [float(err) for err in evaluation['profile_errors']],
            'az_bin_errors': [float(err) for err in evaluation['az_bin_errors']],
            'az_bin_correlations': [float(corr) for corr in evaluation['az_bin_correlations']],
            'az_bin_r2_scores': [float(r2) for r2 in evaluation['az_bin_r2_scores']],
            'mean_correlation': float(mean_corr),
            'mean_r2_score': float(mean_r2),
            'best_azimuth_deg': float(best_az),
            'best_correlation': float(np.max(evaluation['az_bin_correlations'])),
            'worst_azimuth_deg': float(worst_az),
            'worst_correlation': float(np.min(evaluation['az_bin_correlations']))
        }, f, indent=2)
    
    # Save configuration
    config = {
        'sessions': SESSIONS,
        'opening_angle': OPENING_ANGLE,
        'profile_steps': PROFILE_STEPS,
        'input_size': INPUT_SIZE,
        'hidden_sizes': HIDDEN_SIZES,
        'output_size': OUTPUT_SIZE,
        'dropout_rate': DROPOUT_RATE,
        'test_size': TEST_SIZE,
        'val_size': VAL_SIZE,
        'random_state': RANDOM_STATE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'patience': PATIENCE,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'timestamp': timestamp
    }
    
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Plots are now saved directly to results directory
    
    print(f"\n🎉 Training complete!")
    print(f"   Results saved to: {results_dir}")
    print(f"   Final Test MSE: {evaluation['test_loss']:.6f}")
    print(f"   Final MAE: {evaluation['mean_absolute_error']:.1f}mm")
    
    return results_dir

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    results_dir = main()