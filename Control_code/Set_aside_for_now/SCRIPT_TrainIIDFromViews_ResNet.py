"""
Train a ResNet18 to predict corrected_iid from overhead conical views.

Notes:
- Uses DataProcessor.DataCollection to load views and target IID values.
- Views are uint8 RGB (N, H, W, 3); normalized to [0, 1] then mean/std.
- Blocked split per session to reduce leakage.
- Supports pretrained or from-scratch ResNet18 via USE_PRETRAINED.
"""

import os
import json
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import models
except Exception as e:
    raise SystemExit(
        "PyTorch + torchvision are required for this script. Install with: pip install torch torchvision\n"
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
PROFILE_STEPS = 51

SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
# Filter: keep samples within this distance range (meters)
MIN_DISTANCE_M = 0.4
MAX_DISTANCE_M = 1.45
# Sanity check: shuffle labels (should destroy performance)
SHUFFLE_LABELS = False
# Crop black borders
USE_BBOX_CROP = True
# Pretrained vs from scratch
USE_PRETRAINED = True
USE_PROFILES = False

BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 200
PATIENCE = 5
DROP_OUT = 0.2
FREEZE_EPOCHS = 10

MODEL_DIR = os.path.join('cache', 'iid_view_resnet')
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


def build_resnet18(use_pretrained: bool):
    if use_pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)
    else:
        model = models.resnet18(weights=None)

    # Replace final layer with identity; add custom head later
    model.fc = nn.Identity()
    return model


class ResNetWithProfiles(nn.Module):
    def __init__(self, use_pretrained: bool, profile_dim: int):
        super().__init__()
        self.backbone = build_resnet18(use_pretrained)
        feat_dim = 512  # resnet18 output
        self.head = nn.Sequential(
            nn.Dropout(DROP_OUT),
            nn.Linear(feat_dim + profile_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(128, 1),
        )

    def forward(self, x_img, x_prof):
        feats = self.backbone(x_img)
        x = torch.cat([feats, x_prof], dim=1)
        return self.head(x)


class ResNetRegressor(nn.Module):
    def __init__(self, use_pretrained: bool):
        super().__init__()
        self.backbone = build_resnet18(use_pretrained)
        feat_dim = 512
        self.head = nn.Sequential(
            nn.Dropout(DROP_OUT),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x_img):
        feats = self.backbone(x_img)
        return self.head(feats)


def set_backbone_trainable(model, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    set_seed(SEED)

    # Load data
    dc = DataProcessor.DataCollection(SESSIONS)
    dc.load_views(radius_mm=RADIUS_MM, opening_angle=OPENING_ANGLE, output_size=OUTPUT_SIZE, show_example=False)
    if USE_PROFILES:
        dc.load_profiles(opening_angle=OPENING_ANGLE, steps=PROFILE_STEPS)

    # Build per-session arrays (filter first), then apply optional bbox crop
    per_session_X = []
    per_session_P = []
    per_session_y = []

    for p in dc.processors:
        X_s = p.views.astype(np.float32)  # (N, H, W, 3)
        P_s = p.profiles.astype(np.float32) if USE_PROFILES else None
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
            if USE_PROFILES:
                P_s = P_s[keep]
            y_s = y_s[keep]

        if X_s.ndim != 4 or y_s.ndim != 1 or X_s.shape[0] != y_s.shape[0]:
            raise ValueError(f"Bad shapes after filtering: X {X_s.shape}, y {y_s.shape}")
        if USE_PROFILES and (P_s.ndim != 2 or P_s.shape[0] != y_s.shape[0]):
            raise ValueError(f"Bad profiles after filtering: P {P_s.shape}, y {y_s.shape}")

        per_session_X.append(X_s)
        if USE_PROFILES:
            per_session_P.append(P_s)
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
    P_train_list = []
    y_train_list = []
    X_val_list = []
    P_val_list = []
    y_val_list = []
    X_test_list = []
    P_test_list = []
    y_test_list = []

    for i in range(len(per_session_X)):
        X_s = per_session_X[i]
        y_s = per_session_y[i]
        P_s = per_session_P[i] if USE_PROFILES else None
        n_s = X_s.shape[0]
        train_idx_s, val_idx_s, test_idx_s = blocked_split_indices(
            n_s, TRAIN_FRAC, VAL_FRAC, TEST_FRAC
        )

        X_train_list.append(X_s[train_idx_s])
        y_train_list.append(y_s[train_idx_s])
        if USE_PROFILES:
            P_train_list.append(P_s[train_idx_s])
        X_val_list.append(X_s[val_idx_s])
        y_val_list.append(y_s[val_idx_s])
        if USE_PROFILES:
            P_val_list.append(P_s[val_idx_s])
        X_test_list.append(X_s[test_idx_s])
        y_test_list.append(y_s[test_idx_s])
        if USE_PROFILES:
            P_test_list.append(P_s[test_idx_s])

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    if USE_PROFILES:
        P_train = np.concatenate(P_train_list, axis=0)
        P_val = np.concatenate(P_val_list, axis=0)
        P_test = np.concatenate(P_test_list, axis=0)

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

    train_mean = X_train.mean(axis=(0, 1, 2))
    train_std = X_train.std(axis=(0, 1, 2))
    train_std[train_std == 0] = 1.0

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    if USE_PROFILES:
        prof_mean = P_train.mean(axis=0)
        prof_std = P_train.std(axis=0)
        prof_std[prof_std == 0] = 1.0
        P_train = (P_train - prof_mean) / prof_std
        P_val = (P_val - prof_mean) / prof_std
        P_test = (P_test - prof_mean) / prof_std

    # Normalize targets using train stats only
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std == 0:
        y_std = 1.0
    yn_train = (y_train - y_mean) / y_std
    yn_val = (y_val - y_mean) / y_std
    yn_test = (y_test - y_mean) / y_std

    # Convert to torch tensors and channel-first
    def make_loader(x_arr, p_arr, y_arr, shuffle=False):
        x_t = torch.from_numpy(np.transpose(x_arr, (0, 3, 1, 2)))  # NHWC -> NCHW
        if USE_PROFILES:
            p_t = torch.from_numpy(p_arr)
            ds = TensorDataset(x_t, p_t, torch.from_numpy(y_arr).unsqueeze(1))
        else:
            ds = TensorDataset(x_t, torch.from_numpy(y_arr).unsqueeze(1))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    if USE_PROFILES:
        train_loader = make_loader(X_train, P_train, yn_train, shuffle=True)
        val_loader = make_loader(X_val, P_val, yn_val)
        test_loader = make_loader(X_test, P_test, yn_test)
    else:
        train_loader = make_loader(X_train, None, yn_train, shuffle=True)
        val_loader = make_loader(X_val, None, yn_val)
        test_loader = make_loader(X_test, None, yn_test)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if USE_PROFILES:
        model = ResNetWithProfiles(USE_PRETRAINED, profile_dim=PROFILE_STEPS).to(device)
    else:
        model = ResNetRegressor(USE_PRETRAINED).to(device)

    # Freeze backbone initially (train head only), then unfreeze
    set_backbone_trainable(model, False)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val = float('inf')
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            print("Unfroze backbone for fine-tuning.")
        model.train()
        train_losses = []
        for batch in train_loader:
            if USE_PROFILES:
                xb, pb, yb = batch
                xb = xb.to(device)
                pb = pb.to(device)
                yb = yb.to(device)
                preds = model(xb, pb)
            else:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
            optimizer.zero_grad()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if USE_PROFILES:
                    xb, pb, yb = batch
                    xb = xb.to(device)
                    pb = pb.to(device)
                    yb = yb.to(device)
                    preds = model(xb, pb)
                else:
                    xb, yb = batch
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
        for batch in test_loader:
            if USE_PROFILES:
                xb, pb, yb = batch
                xb = xb.to(device)
                pb = pb.to(device)
                preds.append(model(xb, pb).cpu().numpy())
                trues.append(yb.cpu().numpy())
            else:
                xb, yb = batch
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
    model_path = os.path.join(MODEL_DIR, 'resnet18_views_iid.pt')
    torch.save(model.state_dict(), model_path)

    stats = {
        'sessions': SESSIONS,
        'radius_mm': RADIUS_MM,
        'opening_angle': OPENING_ANGLE,
        'output_size': OUTPUT_SIZE,
        'profile_steps': PROFILE_STEPS,
        'train_frac': TRAIN_FRAC,
        'val_frac': VAL_FRAC,
        'test_frac': TEST_FRAC,
        'max_distance_m': MAX_DISTANCE_M,
        'min_distance_m': MIN_DISTANCE_M,
        'shuffle_labels': SHUFFLE_LABELS,
        'use_bbox_crop': USE_BBOX_CROP,
        'use_pretrained': USE_PRETRAINED,
        'use_profiles': USE_PROFILES,
        'split_mode': 'blocked_per_session',
        'optimizer': 'AdamW',
        'weight_decay': WEIGHT_DECAY,
        'loss': 'SmoothL1',
        'dropout': DROP_OUT,
        'seed': SEED,
        'train_mean': train_mean.tolist(),
        'train_std': train_std.tolist(),
        'freeze_epochs': FREEZE_EPOCHS,
    }
    if USE_PROFILES:
        stats['profile_mean'] = prof_mean.tolist()
        stats['profile_std'] = prof_std.tolist()
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
    plots_dir = os.path.join('Plots', 'iid_from_views_resnet')
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
