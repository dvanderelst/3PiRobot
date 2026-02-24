# Emulator Output

Artifacts from `SCRIPT_TrainEmulator.py`.

This emulator serves as a "world model" that predicts what sonar measurements
(distance and IID) would be received from different positions in an environment,
based on profile data.

## Core
- `best_model_pytorch.pth`: Best profile->(distance,iid) emulator model by validation loss.
- `training_params.json`: Configuration + normalization + test metrics.

## Plots
- `training_curves.png`: Training and validation loss curves
- `test_scatter.png`: Predicted vs true distance and IID on test set

## Usage
This emulator can be used to:
1. Simulate robot navigation through environments
2. Generate synthetic training data for policy learning
3. Enable "imagination-based" planning and learning
