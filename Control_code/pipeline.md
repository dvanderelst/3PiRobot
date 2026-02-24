# EchoProcessor Pipeline Summary

This document describes the two-stage pipeline for processing sonar data and profile information to extract distance and IID (Interaural Intensity Difference) metrics.

## Stage 1: SCRIPT_TrainEchoProcessor.py

**Purpose:** Train a CNN to predict distance from sonar echoes and compute IID from predicted distance.

### Key Components:

1. **DistanceCNN Model**
   - 1D convolutional neural network with 3 conv layers
   - Takes sonar echoes (2 channels) as input
   - Predicts closest distance to profile

2. **IID Computation**
   - Extracts IID from local echo window around predicted distance
   - Configurable window parameters (pre_samples, post_samples, center_offset)
   - Includes window sweep optimization

3. **Training Pipeline**
   - Data split: train (quadrants 0,1,2), test (quadrant 3)
   - Huber loss for robust distance prediction
   - Linear calibration on validation set
   - Early stopping based on validation loss

4. **Evaluation Metrics**
   - Distance metrics: RMSE, MAE, Bias, Pearson/Spearman correlation
   - IID metrics: Correlation with profile asymmetry, sign accuracy
   - Confusion matrices for IID sign prediction

### Outputs:
- `best_model_pytorch.pth` - Trained distance prediction model
- `echoprocessor_artifacts.pth` - Portable inference artifact
- `training_params.json` - Configuration and metrics
- Visualizations: training curves, distance scatter plots, IID correlation plots

## Stage 2: SCRIPT_TrainProfileToEchoProcessor.py

**Purpose:** Train an MLP to emulate EchoProcessor using profile data as input.

### Key Components:

1. **ProfileMLP Model**
   - Multi-layer perceptron with configurable hidden sizes
   - Dual output heads: distance and IID prediction
   - Feature augmentation from profile data

2. **Feature Augmentation**
   - Raw profile bins
   - Minimum value and location
   - Left/right asymmetry
   - Local slope around minimum
   - Weighted center-of-mass

3. **Training Pipeline**
   - Multi-task learning with separate loss weights
   - Huber loss for both distance and IID targets
   - IID sample weighting for balanced learning
   - Output calibration on validation set

4. **Evaluation Metrics**
   - Distance metrics: RMSE, MAE, Bias, Pearson/Spearman
   - IID metrics: RMSE, MAE, Bias, correlation, sign accuracy
   - Sign confusion counts (TP, FN, FP, TN)

### Outputs:
- `best_model_pytorch.pth` - Trained profile-to-EchoProcessor model
- `training_params.json` - Configuration and metrics
- Visualizations: training curves, test scatter plots

## Pipeline Flow

```
Sonar Data → EchoProcessor (CNN) → Distance Prediction → IID Extraction
                                      ↓
                                 Profile Data → ProfileToEchoProcessor (MLP) → (Distance, IID)
```

The two-stage approach provides:
1. Direct sonar-based distance prediction and IID computation
2. Profile-based emulation of the complete sonar system
3. Cross-validation between sonar-derived and profile-derived metrics

## Configuration

Both scripts use similar configuration patterns:
- Session selection and quadrant-based splitting
- Normalization options (sonar/profile and target)
- Training hyperparameters (batch size, epochs, learning rate)
- Regularization (L2, dropout)
- Early stopping with patience

The pipeline is designed for reproducibility with seeded random operations and comprehensive logging of all parameters and metrics.