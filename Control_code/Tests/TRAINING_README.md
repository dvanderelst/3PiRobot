# Profile Prediction from Sonar Data - Training Script

## Overview
This script trains a Multi-Layer Perceptron (MLP) to predict visual distance profiles from sonar sensor data. The goal is to explore whether the spatial pattern of distances around the robot can be reconstructed from acoustic sensor readings.

## Key Features

### 1. **Data Processing**
- Loads data from multiple sessions (session03, session04, session06, session07)
- Extracts sonar data (200-dimensional) and visual profiles (19-dimensional by default)
- Standardizes both input and output data for better training

### 2. **MLP Architecture**
- **Input**: 200-dimensional sonar data (concatenated left/right microphone envelopes)
- **Hidden Layers**: 3 layers with 256 units each
- **Activation**: ReLU with BatchNorm and Dropout (0.2)
- **Output**: 19-dimensional profile (distance predictions at different azimuth angles)
- **Total Parameters**: ~189,000

### 3. **Training Process**
- **Batch Size**: 32
- **Learning Rate**: 0.001 with Adam optimizer
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Patience of 10 epochs
- **Learning Rate Scheduling**: Reduce on plateau
- **Data Split**: 72% train, 8% validation, 20% test

### 4. **Evaluation Metrics**
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Per-azimuth angle error analysis

### 5. **Visualization**
- Training history plots (loss curves)
- Azimuth error analysis (error vs. angle)
- Sample prediction comparisons (ground truth vs. predicted)

## Usage

### Basic Training
```bash
python SCRIPT_Train.py
```

### Custom Parameters
Modify the configuration parameters at the top of the script:
- `az_extent`: Total azimuth range in degrees
- `az_steps`: Number of azimuth steps in profiles
- `sessions`: List of sessions to include
- `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, etc.

## Output
- Trained model saved as `profile_mlp_{timestamp}.pt`
- Training history plots
- Error analysis plots
- Sample prediction visualizations
- Comprehensive performance metrics

## Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy
- scikit-learn
- matplotlib
- tqdm

## Model Architecture Details
```
Input (200) → Linear(200→256) → BatchNorm → ReLU → Dropout(0.2) →
Linear(256→256) → BatchNorm → ReLU → Dropout(0.2) →
Linear(256→256) → BatchNorm → ReLU → Dropout(0.2) →
Linear(256→19) → Output (19)
```

## Performance Expectations
- Training time: ~1-5 minutes depending on hardware
- Expected test RMSE: Will depend on data quality and sensor characteristics
- The model should learn to predict general spatial patterns from sonar data

## Notes
- The script automatically handles GPU/CPU detection
- All random seeds are set for reproducibility
- Early stopping prevents overfitting
- Model checkpoints are saved with training metadata