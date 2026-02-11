"""
Train a small neural network to predict distance profiles from sonar data.

Notes:
- Uses DataProcessor.DataCollection to load profiles and flattened sonar traces.
- Profiles are distance-to-wall features in the robot frame.
- Script normalizes inputs/targets and saves model + normalization stats.
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


# -----------------------------
# Configuration
# -----------------------------
TRAIN_SESSIONS = ['session07', 'session04', 'session06']
OPENING_ANGLE = 25
STEPS = 11
FLATTEN_SONAR = False
USE_CNN = True

SEED = 128
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
# Filter: keep samples within this distance range (meters)
MIN_DISTANCE_M = 0
# Profile targets are in mm; cap beyond sonar range.
MAX_PROFILE_MM = 1500.0
# Plotting: hide points at/near max range to avoid range-cap arcs.
PLOT_MAX_PROFILE_MM = 3000.0
# Sanity check: shuffle inputs (should destroy performance)
SHUFFLE_INPUTS = False
# Sanity check: shuffle labels (should destroy performance)
SHUFFLE_LABELS = False

# Optional world-frame overlay from a single session.
EVAL_SESSION = 'session03'
EVAL_INDEX_START = 200
EVAL_INDEX_END = 210  # exclusive
EVAL_INDICES = None  # list/array overrides start/end when provided

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 300
PATIENCE = 30
DROP_OUT = 0.2

MODEL_DIR = os.path.join('cache', 'profiles_from_sonar_model')
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


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32, 256),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    set_seed(SEED)

    # Load data
    if USE_CNN and FLATTEN_SONAR:
        raise ValueError("USE_CNN requires FLATTEN_SONAR = False")
    if (not USE_CNN) and (not FLATTEN_SONAR):
        raise ValueError("MLP requires FLATTEN_SONAR = True")

    dc = DataProcessor.DataCollection(TRAIN_SESSIONS)
    dc.load_profiles(opening_angle=OPENING_ANGLE, steps=STEPS)
    dc.load_sonar(flatten=FLATTEN_SONAR)

    # Build per-session arrays and apply blocked split per session
    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []
    X_test_list = []
    y_test_list = []

    for p in dc.processors:
        X_s = p.sonar_data.astype(np.float32)
        y_s = p.profiles.astype(np.float32)
        d_s = p.get_field('sonar_package', 'corrected_distance').astype(np.float32)

        if MAX_PROFILE_MM is not None:
            y_s = np.clip(y_s, a_min=None, a_max=MAX_PROFILE_MM)

        if MIN_DISTANCE_M is not None:
            keep = d_s >= MIN_DISTANCE_M
            X_s = X_s[keep]
            y_s = y_s[keep]

        if np.isnan(X_s).any():
            X_s = np.nan_to_num(X_s, nan=np.nanmean(X_s, axis=0))
        if np.isnan(y_s).any():
            y_s = np.nan_to_num(y_s, nan=np.nanmean(y_s, axis=0))

        if USE_CNN:
            bad_shape = X_s.ndim != 3 or y_s.ndim != 2 or X_s.shape[0] != y_s.shape[0]
        else:
            bad_shape = X_s.ndim != 2 or y_s.ndim != 2 or X_s.shape[0] != y_s.shape[0]
        if bad_shape:
            raise ValueError(f"Bad shapes after filtering: X {X_s.shape}, y {y_s.shape}")

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

    # Optional input shuffle (sanity check)
    if SHUFFLE_INPUTS:
        rng = np.random.RandomState(SEED)
        idx_train = rng.permutation(X_train.shape[0])
        idx_val = rng.permutation(X_val.shape[0])
        idx_test = rng.permutation(X_test.shape[0])
        X_train = X_train[idx_train]
        X_val = X_val[idx_val]
        X_test = X_test[idx_test]
        print("⚠️  SHUFFLE_INPUTS enabled: X has been randomly permuted per split.")

    # Optional label shuffle (sanity check)
    if SHUFFLE_LABELS:
        rng = np.random.RandomState(SEED + 1)
        idx_train = rng.permutation(y_train.shape[0])
        idx_val = rng.permutation(y_val.shape[0])
        idx_test = rng.permutation(y_test.shape[0])
        y_train = y_train[idx_train]
        y_val = y_val[idx_val]
        y_test = y_test[idx_test]
        print("⚠️  SHUFFLE_LABELS enabled: y has been randomly permuted per split.")

    # Normalize using train stats only
    if USE_CNN:
        x_mean = X_train.mean(axis=(0, 1))
        x_std = X_train.std(axis=(0, 1))
        x_std[x_std == 0] = 1.0
        Xn_train = (X_train - x_mean) / x_std
        Xn_val = (X_val - x_mean) / x_std
        Xn_test = (X_test - x_mean) / x_std
    else:
        x_mean = X_train.mean(axis=0)
        x_std = X_train.std(axis=0)
        x_std[x_std == 0] = 1.0
        Xn_train = (X_train - x_mean) / x_std
        Xn_val = (X_val - x_mean) / x_std
        Xn_test = (X_test - x_mean) / x_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std == 0] = 1.0
    yn_train = (y_train - y_mean) / y_std
    yn_val = (y_val - y_mean) / y_std
    yn_test = (y_test - y_mean) / y_std

    def make_loader(x_arr, y_arr, shuffle=False):
        if USE_CNN:
            x_t = torch.from_numpy(np.transpose(x_arr, (0, 2, 1)))  # N, L, C -> N, C, L
        else:
            x_t = torch.from_numpy(x_arr)
        y_t = torch.from_numpy(y_arr)
        ds = TensorDataset(x_t, y_t)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(Xn_train, yn_train, shuffle=True)
    val_loader = make_loader(Xn_val, yn_val)
    test_loader = make_loader(Xn_test, yn_test)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if USE_CNN:
        model = CNN1D(in_ch=X_train.shape[2], out_dim=y_train.shape[1]).to(device)
    else:
        model = MLP(in_dim=X_train.shape[1], out_dim=y_train.shape[1]).to(device)
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
    preds = np.vstack(preds)
    trues = np.vstack(trues)

    # Denormalize
    preds_denorm = preds * y_std + y_mean
    trues_denorm = trues * y_std + y_mean

    mse = float(np.mean((preds_denorm - trues_denorm) ** 2))
    mae = float(np.mean(np.abs(preds_denorm - trues_denorm)))
    print(f"Test MSE (denorm): {mse:.4f}")
    print(f"Test MAE (denorm): {mae:.4f}")

    # Per-bin correlation (Pearson r) and summary
    per_bin_r = []
    for i in range(trues_denorm.shape[1]):
        r = float(np.corrcoef(trues_denorm[:, i], preds_denorm[:, i])[0, 1])
        per_bin_r.append(r)
    per_bin_r = np.array(per_bin_r, dtype=np.float32)
    print("Per-bin Pearson r (true vs pred):")
    for i, r in enumerate(per_bin_r):
        print(f"  bin {i:02d}: {r:.4f}")
    print(f"Mean r: {float(np.mean(per_bin_r)):.4f}")
    print(f"Median r: {float(np.median(per_bin_r)):.4f}")

    # Save model + normalization stats
    model_path = os.path.join(MODEL_DIR, 'mlp_profiles_from_sonar.pt')
    torch.save(model.state_dict(), model_path)

    stats = {
        'x_mean': x_mean.tolist(),
        'x_std': x_std.tolist(),
        'y_mean': y_mean.tolist(),
        'y_std': y_std.tolist(),
        'sessions': TRAIN_SESSIONS,
        'opening_angle': OPENING_ANGLE,
        'steps': STEPS,
        'flatten_sonar': FLATTEN_SONAR,
        'model_type': 'cnn1d' if USE_CNN else 'mlp',
        'x_norm_mode': 'per_channel' if USE_CNN else 'per_feature',
        'max_profile_mm': MAX_PROFILE_MM,
        'train_frac': TRAIN_FRAC,
        'val_frac': VAL_FRAC,
        'test_frac': TEST_FRAC,
        'min_distance_m': MIN_DISTANCE_M,
        'shuffle_inputs': SHUFFLE_INPUTS,
        'shuffle_labels': SHUFFLE_LABELS,
        'split_mode': 'blocked_per_session',
        'optimizer': 'AdamW',
        'weight_decay': WEIGHT_DECAY,
        'loss': 'SmoothL1',
        'dropout': DROP_OUT,
        'seed': SEED,
    }
    with open(os.path.join(MODEL_DIR, 'normalization.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved normalization stats to: {os.path.join(MODEL_DIR, 'normalization.json')}")

    # -----------------------------
    # Extra outputs: plots
    # -----------------------------
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Matplotlib not available for plots: {e}")
        raise SystemExit(0)

    plots_dir = os.path.join('Plots', 'profiles_from_sonar')
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Per-bin scatter: true vs pred for each profile bin (grid)
    n_bins = trues_denorm.shape[1]
    cols = min(6, n_bins)
    rows = int(np.ceil(n_bins / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i in range(n_bins):
        ax = axes[i]
        ax.scatter(trues_denorm[:, i], preds_denorm[:, i], s=8, alpha=0.5)
        min_v = float(min(trues_denorm[:, i].min(), preds_denorm[:, i].min()))
        max_v = float(max(trues_denorm[:, i].max(), preds_denorm[:, i].max()))
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1)
        ax.set_xlabel(f"True bin {i}")
        ax.set_ylabel(f"Pred bin {i}")
    for j in range(n_bins, axes.size):
        axes[j].axis('off')
    fig.suptitle("True vs Predicted Profile Bins (Test)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'scatter_bins.png'), dpi=150)
    plt.close(fig)

    # 1b) Per-bin scatter: individual files
    per_bin_dir = os.path.join(plots_dir, 'per_bin_scatter')
    os.makedirs(per_bin_dir, exist_ok=True)
    for i in range(n_bins):
        plt.figure(figsize=(4, 4))
        plt.scatter(trues_denorm[:, i], preds_denorm[:, i], s=10, alpha=0.6)
        min_v = float(min(trues_denorm[:, i].min(), preds_denorm[:, i].min()))
        max_v = float(max(trues_denorm[:, i].max(), preds_denorm[:, i].max()))
        plt.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1)
        plt.xlabel(f"True bin {i}")
        plt.ylabel(f"Pred bin {i}")
        plt.title(f"Bin {i:02d} Scatter")
        plt.tight_layout()
        plt.savefig(os.path.join(per_bin_dir, f"bin_{i:02d}.png"), dpi=150)
        plt.close()

    # 2) Error histogram across all bins
    residuals = (preds_denorm - trues_denorm).ravel()
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, alpha=0.8)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.title("Residuals Histogram (All Bins, Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_hist.png'), dpi=150)
    plt.close()

    # 3) Per-bin correlation bar plot
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(per_bin_r)), per_bin_r, color='tab:blue', alpha=0.8)
    plt.ylim(-1.0, 1.0)
    plt.xlabel("Profile bin")
    plt.ylabel("Pearson r")
    plt.title("Per-bin Correlation (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'per_bin_corr.png'), dpi=150)
    plt.close()

    # 4) Example profiles: overlay true/pred for a few samples
    n_examples = min(6, trues_denorm.shape[0])
    example_idx = np.linspace(0, trues_denorm.shape[0] - 1, n_examples, dtype=int)
    theta = np.deg2rad(np.linspace(-OPENING_ANGLE / 2, OPENING_ANGLE / 2, STEPS))
    fig, axes = plt.subplots(2, 3, subplot_kw={'projection': 'polar'}, figsize=(12, 6))
    axes = axes.ravel()
    for k, ax in enumerate(axes[:n_examples]):
        idx = int(example_idx[k])
        ax.plot(theta, trues_denorm[idx], label="True")
        ax.plot(theta, preds_denorm[idx], label="Pred")
        ax.set_title(f"Sample {idx}")
    axes[0].legend(loc='upper right', fontsize=8)
    fig.suptitle("Example Profiles (Test)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'example_profiles.png'), dpi=150)
    plt.close(fig)

    print(f"Saved plots to: {plots_dir}")

    # -----------------------------
    # World-frame overlay for a session
    # -----------------------------
    if EVAL_SESSION is not None:
        eval_proc = None
        for p in dc.processors:
            if os.path.basename(p.session) == EVAL_SESSION:
                eval_proc = p
                break

        if eval_proc is None:
            eval_proc = DataProcessor.DataProcessor(EVAL_SESSION)
            eval_proc.load_profiles(opening_angle=OPENING_ANGLE, steps=STEPS)
            eval_proc.load_sonar(flatten=FLATTEN_SONAR)

        if EVAL_INDICES is not None:
            indices = np.array(EVAL_INDICES, dtype=int)
        else:
            indices = np.arange(EVAL_INDEX_START, EVAL_INDEX_END, dtype=int)
        indices = indices[(indices >= 0) & (indices < eval_proc.n)]

        if indices.size == 0:
            print("No valid indices for world-frame overlay.")
        else:
            X_eval = eval_proc.sonar_data[indices].astype(np.float32)
            if USE_CNN:
                Xn_eval = (X_eval - x_mean) / x_std
                Xn_eval = np.transpose(Xn_eval, (0, 2, 1))  # N, L, C -> N, C, L
            else:
                Xn_eval = (X_eval - x_mean) / x_std
            with torch.no_grad():
                preds_eval = model(torch.from_numpy(Xn_eval).to(device)).cpu().numpy()
            preds_eval = preds_eval * y_std + y_mean
            if MAX_PROFILE_MM is not None:
                preds_eval = np.clip(preds_eval, a_min=None, a_max=MAX_PROFILE_MM)

            true_eval = eval_proc.profiles[indices].astype(np.float32)
            if MAX_PROFILE_MM is not None:
                true_eval = np.clip(true_eval, a_min=None, a_max=MAX_PROFILE_MM)
            centers = eval_proc.profile_centers[indices]

            fig = plt.figure(figsize=(8, 8))
            cmap = plt.get_cmap('tab10')
            denom = max(1, len(indices) - 1)
            for k, idx in enumerate(indices):
                color = cmap(k % cmap.N)
                az_deg = centers[k]
                rob_x = float(eval_proc.rob_x[idx])
                rob_y = float(eval_proc.rob_y[idx])
                rob_yaw = float(eval_proc.rob_yaw_deg[idx])

                dist_true = true_eval[k].copy()
                dist_pred = preds_eval[k].copy()
                if PLOT_MAX_PROFILE_MM is not None:
                    dist_true = np.where(dist_true < PLOT_MAX_PROFILE_MM, dist_true, np.nan)
                    dist_pred = np.where(dist_pred < PLOT_MAX_PROFILE_MM, dist_pred, np.nan)

                x_t, y_t = DataProcessor.robot2world(az_deg, dist_true, rob_x, rob_y, rob_yaw)
                x_p, y_p = DataProcessor.robot2world(az_deg, dist_pred, rob_x, rob_y, rob_yaw)

                plt.plot(x_t, y_t, color=color, linewidth=1.5, linestyle='-')
                plt.plot(x_p, y_p, color=color, linewidth=1.0, linestyle='--')

            plt.plot(eval_proc.rob_x[indices], eval_proc.rob_y[indices],
                     'k.', markersize=2, alpha=0.6)

            from matplotlib.lines import Line2D
            legend_elems = [
                Line2D([0], [0], color='k', linestyle='-', label='True profile'),
                Line2D([0], [0], color='k', linestyle='--', label='Pred profile'),
                Line2D([0], [0], color='k', marker='.', linestyle='None', label='Robot pose'),
            ]
            plt.legend(handles=legend_elems, loc='best')
            plt.axis('equal')
            plt.xlabel("X [mm]")
            plt.ylabel("Y [mm]")
            plt.title(
                f"World-frame profiles: {EVAL_SESSION} "
                f"indices {int(indices[0])}..{int(indices[-1])}"
            )
            plt.tight_layout()

            out_path = os.path.join(
                plots_dir,
                f"world_profiles_{EVAL_SESSION}_idx{int(indices[0])}_{int(indices[-1])}.png"
            )
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved world-frame overlay to: {out_path}")
