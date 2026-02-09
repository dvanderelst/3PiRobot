"""
Train a CNN to predict corrected_iid from overhead conical views.

Notes:
- Uses DataProcessor.DataCollection to load views and target IID values.
- Views are uint8 RGB (N, H, W, 3); normalized to [0, 1].
- Blocked split per session to reduce leakage.
"""

import os
import json
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    raise SystemExit(
        "PyTorch is required for this script. Install with: pip install torch\n"
        f"Import error: {e}"
    )

from Library import DataProcessor
from Library.DataProcessor import find_bounding_box_across_views


# -----------------------------
# Configuration
# -----------------------------
SESSIONS = ['session03', 'session04', 'session06', 'session07']
RADIUS_MM = 2000
OPENING_ANGLE = 220
OUTPUT_SIZE = (512, 512)  # (W, H)

SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
# Filter: keep samples within this distance range (meters)
MIN_DISTANCE_M = 0.4
MAX_DISTANCE_M = 1.45
# Sanity check: shuffle labels (should destroy performance)
SHUFFLE_LABELS = False
USE_BBOX_CROP = True

BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 200
PATIENCE = 15
DROP_OUT = 0.2

MODEL_DIR = os.path.join('cache', 'iid_view_model')
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def blocked_split_indices(n, train_frac, val_frac, test_frac):
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train/val/test fractions must sum to 1")
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)
    return train_idx, val_idx, test_idx


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    set_seed(SEED)

    # Load data
    dc = DataProcessor.DataCollection(SESSIONS)
    dc.load_views(radius_mm=RADIUS_MM, opening_angle=OPENING_ANGLE, output_size=OUTPUT_SIZE, show_example=False)

    # Build per-session arrays (filter first), then apply optional bbox crop
    per_session_X = []
    per_session_y = []

    for p in dc.processors:
        X_s = p.views.astype(np.float32)  # (N, H, W, 3)
        y_s = p.get_field('sonar_package', 'corrected_iid').astype(np.float32)
        d_s = p.get_field('sonar_package', 'corrected_distance').astype(np.float32)

        # Filter by corrected_distance (per session)
        if MAX_DISTANCE_M is not None or MIN_DISTANCE_M is not None:
            keep = np.ones_like(d_s, dtype=bool)
            if MIN_DISTANCE_M is not None:
                keep &= d_s >= MIN_DISTANCE_M
            if MAX_DISTANCE_M is not None:
                keep &= d_s <= MAX_DISTANCE_M
            X_s = X_s[keep]
            y_s = y_s[keep]

        if X_s.ndim != 4 or y_s.ndim != 1 or X_s.shape[0] != y_s.shape[0]:
            raise ValueError(f"Bad shapes after filtering: X {X_s.shape}, y {y_s.shape}")

        per_session_X.append(X_s)
        per_session_y.append(y_s)

    # Optional global bounding-box crop to remove black borders
    if USE_BBOX_CROP:
        all_views = np.concatenate(per_session_X, axis=0)
        bbox = find_bounding_box_across_views(all_views)
        if bbox is None:
            print("⚠️  Bounding-box crop skipped (no non-black pixels found).")
        else:
            x_min, y_min, x_max, y_max = bbox
            print(f"Cropping views to bbox: x[{x_min}:{x_max}] y[{y_min}:{y_max}]")
            for i in range(len(per_session_X)):
                per_session_X[i] = per_session_X[i][:, y_min:y_max, x_min:x_max, :]

    # Apply blocked split per session
    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []
    X_test_list = []
    y_test_list = []

    for X_s, y_s in zip(per_session_X, per_session_y):
        n_s = X_s.shape[0]
        train_idx_s, val_idx_s, test_idx_s = blocked_split_indices(
            n_s, TRAIN_FRAC, VAL_FRAC, TEST_FRAC
        )

        X_train_list.append(X_s[train_idx_s])
        y_train_list.append(y_s[train_idx_s])
        X_val_list.append(X_s[val_idx_s])
        y_val_list.append(y_s[val_idx_s])
        X_test_list.append(X_s[test_idx_s])
        y_test_list.append(y_s[test_idx_s])

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Optional label shuffle (sanity check)
    if SHUFFLE_LABELS:
        rng = np.random.RandomState(SEED)
        y_all = np.concatenate([y_train, y_val, y_test], axis=0)
        rng.shuffle(y_all)
        n_train = y_train.shape[0]
        n_val = y_val.shape[0]
        y_train = y_all[:n_train]
        y_val = y_all[n_train:n_train + n_val]
        y_test = y_all[n_train + n_val:]
        print("⚠️  SHUFFLE_LABELS enabled: y has been randomly permuted.")

    # Normalize inputs: [0,1] then per-channel mean/std (train only)
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    # Compute per-channel mean/std from training data (NHWC)
    train_mean = X_train.mean(axis=(0, 1, 2))
    train_std = X_train.std(axis=(0, 1, 2))
    train_std[train_std == 0] = 1.0

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # Normalize targets using train stats only
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std == 0:
        y_std = 1.0
    yn_train = (y_train - y_mean) / y_std
    yn_val = (y_val - y_mean) / y_std
    yn_test = (y_test - y_mean) / y_std

    # Convert to torch tensors and channel-first
    def make_loader(x_arr, y_arr, shuffle=False):
        x_t = torch.from_numpy(np.transpose(x_arr, (0, 3, 1, 2)))  # NHWC -> NCHW
        y_t = torch.from_numpy(y_arr).unsqueeze(1)
        ds = TensorDataset(x_t, y_t)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(X_train, yn_train, shuffle=True)
    val_loader = make_loader(X_val, yn_val)
    test_loader = make_loader(X_test, yn_test)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(in_channels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val = float('inf')
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping")
                break
        scheduler.step(val_loss)

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation (denormalized metrics)
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.vstack(preds).squeeze(1)
    trues = np.vstack(trues).squeeze(1)

    # Denormalize
    preds_denorm = preds * y_std + y_mean
    trues_denorm = trues * y_std + y_mean

    mse = np.mean((preds_denorm - trues_denorm) ** 2)
    mae = np.mean(np.abs(preds_denorm - trues_denorm))
    print(f"Test MSE (denorm): {mse:.4f}")
    print(f"Test MAE (denorm): {mae:.4f}")

    # Save model + normalization stats
    model_path = os.path.join(MODEL_DIR, 'cnn_views_iid.pt')
    torch.save(model.state_dict(), model_path)

    stats = {
        'sessions': SESSIONS,
        'radius_mm': RADIUS_MM,
        'opening_angle': OPENING_ANGLE,
        'output_size': OUTPUT_SIZE,
        'train_frac': TRAIN_FRAC,
        'val_frac': VAL_FRAC,
        'test_frac': TEST_FRAC,
        'max_distance_m': MAX_DISTANCE_M,
        'min_distance_m': MIN_DISTANCE_M,
        'shuffle_labels': SHUFFLE_LABELS,
        'use_bbox_crop': USE_BBOX_CROP,
        'split_mode': 'blocked_per_session',
        'optimizer': 'AdamW',
        'weight_decay': WEIGHT_DECAY,
        'loss': 'SmoothL1',
        'dropout': DROP_OUT,
        'seed': SEED,
        'train_mean': train_mean.tolist(),
        'train_std': train_std.tolist(),
    }
    with open(os.path.join(MODEL_DIR, 'normalization.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved normalization stats to: {os.path.join(MODEL_DIR, 'normalization.json')}")

    # Extra outputs: text summary + plots
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Matplotlib not available for plots: {e}")
        raise SystemExit(0)

    # Textual summary
    print("")
    print("Prediction summary (test set, denormalized):")
    print(f"  Mean true IID: {np.mean(trues_denorm):.4f}")
    print(f"  Std  true IID: {np.std(trues_denorm):.4f}")
    print(f"  Mean pred IID: {np.mean(preds_denorm):.4f}")
    print(f"  Std  pred IID: {np.std(preds_denorm):.4f}")
    print(f"  Mean error (pred-true): {np.mean(preds_denorm - trues_denorm):.4f}")
    print(f"  Std  error: {np.std(preds_denorm - trues_denorm):.4f}")
    corr = float(np.corrcoef(trues_denorm, preds_denorm)[0, 1])
    print(f"  Pearson r (true vs pred): {corr:.4f}")
    sign_acc = float(np.mean(np.sign(trues_denorm) == np.sign(preds_denorm)))
    print(f"  Sign accuracy (true vs pred): {sign_acc:.4f}")

    # Plots
    plots_dir = os.path.join('Plots', 'iid_from_views')
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Scatter true vs pred (color by sign correctness)
    plt.figure(figsize=(5, 5))
    sign_match = np.sign(trues_denorm) == np.sign(preds_denorm)
    plt.scatter(trues_denorm[sign_match], preds_denorm[sign_match], s=14, alpha=0.6, label='Sign correct')
    plt.scatter(trues_denorm[~sign_match], preds_denorm[~sign_match], s=14, alpha=0.6, label='Sign wrong')
    min_v = float(min(trues_denorm.min(), preds_denorm.min()))
    max_v = float(max(trues_denorm.max(), preds_denorm.max()))
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1)
    plt.xlabel('True corrected_iid')
    plt.ylabel('Predicted corrected_iid')
    plt.title('True vs Predicted IID (Test)')
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(plots_dir, 'scatter_true_vs_pred.png')
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    # 2) Residual histogram
    residuals = preds_denorm - trues_denorm
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=40, alpha=0.8)
    plt.xlabel('Residual (pred - true)')
    plt.ylabel('Count')
    plt.title('Residuals Histogram (Test)')
    plt.tight_layout()
    residuals_path = os.path.join(plots_dir, 'residuals_hist.png')
    plt.savefig(residuals_path, dpi=150)
    plt.close()

    print(f"Saved plots to: {plots_dir}")
