# Script Overview

- `SCRIPT_AssessOccupancy.py`: Loads precomputed occupancy tensors and converts both predicted and real-profile occupancy maps to curvature. Produces comparison stats/plots for predicted-vs-real curvature and relations to sonar IID/distance.

- `SCRIPT_CallibrateRobot.py`: Runs robot calibration routines (baseline, distance, sweep) and updates calibration files for a selected robot. Intended for collecting and maintaining sonar calibration data.

- `SCRIPT_CheckRobotAccuracy.py`: Evaluates how sonar-derived nearest-obstacle signals correspond to environment geometry across sessions. Computes summary metrics across different angular extents.

- `SCRIPT_ComputeOccupancy.py`: Computes robot-frame occupancy evidence tensors for sliding windows in a session (predicted profiles and real profiles). Saves a reusable `.npz` package plus summary/optional diagnostic plots.

- `SCRIPT_DataAcquisition.py`: Runs online data collection with robot control + tracking and writes session data files. Captures synchronized sonar, pose, motion, and environment snapshot metadata.

- `SCRIPT_PlotSession.py`: Quick visualizer for one session trajectory, walls, and per-step robot pose orientation. Useful for sanity checks of recorded data.

- `SCRIPT_TakeEnvSnapshot.py`: Captures and stores an environment snapshot/layout for a session folder. Minimal utility script for environment metadata capture.

- `SCRIPT_TrainProfiles.py`: Trains the two-headed sonar-to-profile network (presence + distance) and saves model/scaler/params and evaluation plots/tables into `Training/`.

- `SCRIPT_TrainStub.py`: Reference/stub script documenting how to load and inspect core data modalities with `DataProcessor`. Serves as a template and data-format primer for new scripts.

## Pipeline Note

1. Step 1, file: `SCRIPT_CallibrateRobot.py`  
   Collect/update sonar calibration for the robot before recording sessions.

2. Step 2, file: `SCRIPT_DataAcquisition.py`  
   Record session data (sonar, pose, motion, environment metadata) while the robot explores.

3. Step 3 (optional), file: `SCRIPT_PlotSession.py`  
   Run a quick sanity check of trajectory/walls/poses for a recorded session.

4. Step 4, file: `SCRIPT_TrainProfiles.py`  
   Train the sonar->profile model on selected sessions and save artifacts to `Training/`.

5. Step 5, file: `SCRIPT_ComputeOccupancy.py`  
   Use trained artifacts to compute occupancy tensors (predicted + real-profile based) for a target session.

6. Step 6, file: `SCRIPT_AssessOccupancy.py`  
   Compare occupancy-derived curvatures (predicted vs real) and relate curvature to IID/distance.

`SCRIPT_TakeEnvSnapshot.py` is a standalone utility for manual environment snapshot capture and is usually outside the main pipeline.
