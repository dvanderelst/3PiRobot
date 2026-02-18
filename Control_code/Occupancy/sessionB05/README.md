# Occupancy Output: sessionB05

This folder contains precomputed robot-frame occupancy evidence for sliding windows in one session.

## Files

- `sessionB05_occupancy_series_data.npz`
  Main data package for downstream analysis.
- `sessionB05_occupancy_summary.json`
  Run/config summary (window settings, grid settings, threshold, etc.).
- `plots/` (optional)
  Example figures generated from the saved `.npz` package.

## NPZ Contents

- `heatmaps_pred`:
  Occupancy evidence maps from **predicted** profiles (shape: `[n_windows, H, W]`).
- `heatmaps_real`:
  Occupancy evidence maps from **real** profiles (same shape as above).
- `heatmaps`:
  Backward-compatible alias to `heatmaps_pred`.
- `x_grid`, `y_grid`:
  Robot-frame grid axes in mm used by all heatmaps.
- `chunk_indices`:
  Filtered sample indices used per window (shape: `[n_windows, window_size]`).
- `window_start`, `window_end`:
  Start/end filtered indices for each window.
- `anchor_index`:
  Anchor index for each window (equal to `window_end`).
- `anchor_rob_x`, `anchor_rob_y`, `anchor_rob_yaw_deg`:
  Anchor pose in world coordinates.
- `chunk_rob_x`, `chunk_rob_y`, `chunk_rob_yaw_deg`:
  Pose sequence (world frame) for each window.
- `grid_mm`, `extent_mm`, `sigma_perp_mm`, `sigma_para_mm`:
  Occupancy integration parameters used to generate this package.

## Notes

- Indices are in **filtered sample space** (after invalid rows were removed).
- Both predicted and real occupancy use identical windows/grid/settings for direct comparison.
