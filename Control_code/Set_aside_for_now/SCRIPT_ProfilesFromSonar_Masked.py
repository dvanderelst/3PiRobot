"""
Train a small neural network to predict distance profiles from sonar data,
including a per-bin "return present" head to handle out-of-range targets.

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
TRAIN_SESSIONS = ['session06','session03', 'session04']
OPENING_ANGLE = 60
STEPS = 7
FLATTEN_SONAR = False
USE_CNN = True

SEED = 128
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
# Filter: keep samples within this distance range (meters)
MIN_DISTANCE_M = 0
SONAR_RANGE_MM = 1450.0
PRED_RETURN_THRESHOLD = 0.5 ###<<<<<<<<<<<<<IMPORTANT
MASK_LOSS_WEIGHT = 1.0
PLOT_MAX_PROFILE_MM = 1450.0
# Sanity check: shuffle inputs (should destroy performance)
SHUFFLE_INPUTS = False
# Sanity check: shuffle labels (should destroy performance)
SHUFFLE_LABELS = False

# Optional world-frame overlay from a single session.
EVAL_SESSION = 'session07'
EVAL_RANGES = [[10,15], [10, 20], [20, 30]]
PLOT_RANGES_SEPARATE = True
EVAL_CHUNK_SIZE = 10
EVAL_CHUNK_STRIDE = 1
PRED_DENSITY_BIN_MM = 20.0
PRED_DENSITY_SIGMA_MM = 100.0
PRED_DENSITY_BOUNDS = 'auto'  # 'auto' uses predicted points; 'arena' uses arena/walls

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 300
PATIENCE = 30
DROP_OUT = 0.3

MODEL_DIR = os.path.join('cache', 'profiles_from_sonar_model_masked')
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
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head_dist = nn.Linear(128, out_dim)
        self.head_mask = nn.Linear(128, out_dim)

    def forward(self, x):
        z = self.trunk(x)
        return self.head_dist(z), self.head_mask(z)


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
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32, 256),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
        )
        self.head_dist = nn.Linear(256, out_dim)
        self.head_mask = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.conv(x)
        z = self.trunk(x)
        return self.head_dist(z), self.head_mask(z)

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
    dc.load_profiles(opening_angle=OPENING_ANGLE, steps=STEPS, fill_nans=False)
    dc.load_sonar(flatten=FLATTEN_SONAR)

    # Build per-session arrays and apply blocked split per session
    X_train_list = []
    y_train_list = []
    m_train_list = []
    X_val_list = []
    y_val_list = []
    m_val_list = []
    X_test_list = []
    y_test_list = []
    m_test_list = []

    for p in dc.processors:
        X_s = p.sonar_data.astype(np.float32)
        y_s = p.profiles.astype(np.float32)
        d_s = p.get_field('sonar_package', 'corrected_distance').astype(np.float32)
        m_s = (~np.isnan(y_s) & (y_s < SONAR_RANGE_MM)).astype(np.float32)
        y_s = np.where(m_s > 0.5, y_s, SONAR_RANGE_MM)

        if MIN_DISTANCE_M is not None:
            keep = d_s >= MIN_DISTANCE_M
            X_s = X_s[keep]
            y_s = y_s[keep]
            m_s = m_s[keep]

        if np.isnan(X_s).any():
            X_s = np.nan_to_num(X_s, nan=np.nanmean(X_s, axis=0))
        if np.isnan(y_s).any():
            y_s = np.nan_to_num(y_s, nan=np.nanmean(y_s, axis=0))
        if np.isnan(m_s).any():
            m_s = np.nan_to_num(m_s, nan=0.0)

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
        m_train_list.append(m_s[train_idx_s])
        X_val_list.append(X_s[val_idx_s])
        y_val_list.append(y_s[val_idx_s])
        m_val_list.append(m_s[val_idx_s])
        X_test_list.append(X_s[test_idx_s])
        y_test_list.append(y_s[test_idx_s])
        m_test_list.append(m_s[test_idx_s])

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    m_train = np.concatenate(m_train_list, axis=0)
    m_val = np.concatenate(m_val_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    m_test = np.concatenate(m_test_list, axis=0)

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
        m_train = m_train[idx_train]
        m_val = m_val[idx_val]
        m_test = m_test[idx_test]
        print("⚠️  SHUFFLE_LABELS enabled: y/m have been randomly permuted per split.")

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

    def masked_mean_std(y_arr, m_arr):
        means = np.zeros(y_arr.shape[1], dtype=np.float32)
        stds = np.ones(y_arr.shape[1], dtype=np.float32)
        for i in range(y_arr.shape[1]):
            vals = y_arr[m_arr[:, i] > 0.5, i]
            if vals.size > 0:
                means[i] = float(np.mean(vals))
                std = float(np.std(vals))
                stds[i] = std if std > 0 else 1.0
        return means, stds

    y_mean, y_std = masked_mean_std(y_train, m_train)
    yn_train = (y_train - y_mean) / y_std
    yn_val = (y_val - y_mean) / y_std
    yn_test = (y_test - y_mean) / y_std

    def make_loader(x_arr, y_arr, m_arr, shuffle=False):
        if USE_CNN:
            x_t = torch.from_numpy(np.transpose(x_arr, (0, 2, 1)))  # N, L, C -> N, C, L
        else:
            x_t = torch.from_numpy(x_arr)
        y_t = torch.from_numpy(y_arr)
        m_t = torch.from_numpy(m_arr)
        ds = TensorDataset(x_t, y_t, m_t)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(Xn_train, yn_train, m_train, shuffle=True)
    val_loader = make_loader(Xn_val, yn_val, m_val)
    test_loader = make_loader(Xn_test, yn_test, m_test)

    # Model
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if USE_CNN:
        model = CNN1D(in_ch=X_train.shape[2], out_dim=y_train.shape[1]).to(device)
    else:
        model = MLP(in_dim=X_train.shape[1], out_dim=y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion_dist = nn.SmoothL1Loss(reduction='none')
    criterion_mask = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val = float('inf')
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            optimizer.zero_grad()
            preds_dist, preds_mask = model(xb)
            raw_dist = criterion_dist(preds_dist, yb)
            mask_sum = torch.sum(mb)
            if mask_sum.item() > 0:
                dist_loss = torch.sum(raw_dist * mb) / mask_sum
            else:
                dist_loss = torch.tensor(0.0, device=device)
            mask_loss = criterion_mask(preds_mask, mb)
            loss = dist_loss + MASK_LOSS_WEIGHT * mask_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)
                preds_dist, preds_mask = model(xb)
                raw_dist = criterion_dist(preds_dist, yb)
                mask_sum = torch.sum(mb)
                if mask_sum.item() > 0:
                    dist_loss = torch.sum(raw_dist * mb) / mask_sum
                else:
                    dist_loss = torch.tensor(0.0, device=device)
                mask_loss = criterion_mask(preds_mask, mb)
                loss = dist_loss + MASK_LOSS_WEIGHT * mask_loss
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
    preds_dist = []
    preds_mask = []
    trues = []
    trues_mask = []
    with torch.no_grad():
        for xb, yb, mb in test_loader:
            xb = xb.to(device)
            dist_out, mask_out = model(xb)
            preds_dist.append(dist_out.cpu().numpy())
            preds_mask.append(mask_out.cpu().numpy())
            trues.append(yb.cpu().numpy())
            trues_mask.append(mb.cpu().numpy())
    preds = np.vstack(preds_dist)
    preds_mask = np.vstack(preds_mask)
    trues = np.vstack(trues)
    trues_mask = np.vstack(trues_mask)

    # Denormalize
    preds_denorm = preds * y_std + y_mean
    trues_denorm = trues * y_std + y_mean

    mask_flat = trues_mask > 0.5
    if np.any(mask_flat):
        mse = float(np.mean((preds_denorm[mask_flat] - trues_denorm[mask_flat]) ** 2))
        mae = float(np.mean(np.abs(preds_denorm[mask_flat] - trues_denorm[mask_flat])))
    else:
        mse = float('nan')
        mae = float('nan')
    print(f"Test MSE (denorm): {mse:.4f}")
    print(f"Test MAE (denorm): {mae:.4f}")

    # Per-bin correlation (Pearson r) and summary
    per_bin_r = []
    for i in range(trues_denorm.shape[1]):
        valid = mask_flat[:, i]
        if np.sum(valid) >= 2:
            r = float(np.corrcoef(trues_denorm[valid, i], preds_denorm[valid, i])[0, 1])
        else:
            r = float('nan')
        per_bin_r.append(r)
    per_bin_r = np.array(per_bin_r, dtype=np.float32)
    print("Per-bin Pearson r (true vs pred):")
    for i, r in enumerate(per_bin_r):
        print(f"  bin {i:02d}: {r:.4f}")
    print(f"Mean r: {float(np.nanmean(per_bin_r)):.4f}")
    print(f"Median r: {float(np.nanmedian(per_bin_r)):.4f}")

    mask_probs = 1.0 / (1.0 + np.exp(-preds_mask))
    mask_pred = (mask_probs >= PRED_RETURN_THRESHOLD)
    mask_true = trues_mask > 0.5
    mask_acc = float(np.mean(mask_pred == mask_true))
    print(f"Return-present accuracy: {mask_acc:.4f}")

    # Save model + normalization stats
    model_path = os.path.join(MODEL_DIR, 'profiles_from_sonar_masked.pt')
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
        'sonar_range_mm': SONAR_RANGE_MM,
        'pred_return_threshold': PRED_RETURN_THRESHOLD,
        'mask_loss_weight': MASK_LOSS_WEIGHT,
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

    plots_dir = os.path.join('Plots', 'profiles_from_sonar_masked')
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Per-bin scatter: true vs pred for each profile bin (grid)
    n_bins = trues_denorm.shape[1]
    cols = min(6, n_bins)
    rows = int(np.ceil(n_bins / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i in range(n_bins):
        ax = axes[i]
        valid = mask_true[:, i]
        if np.any(valid):
            ax.scatter(trues_denorm[valid, i], preds_denorm[valid, i], s=8, alpha=0.5)
            min_v = float(min(trues_denorm[valid, i].min(), preds_denorm[valid, i].min()))
            max_v = float(max(trues_denorm[valid, i].max(), preds_denorm[valid, i].max()))
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
        valid = mask_true[:, i]
        if np.any(valid):
            plt.scatter(trues_denorm[valid, i], preds_denorm[valid, i], s=10, alpha=0.6)
            min_v = float(min(trues_denorm[valid, i].min(), preds_denorm[valid, i].min()))
            max_v = float(max(trues_denorm[valid, i].max(), preds_denorm[valid, i].max()))
            plt.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1)
        plt.xlabel(f"True bin {i}")
        plt.ylabel(f"Pred bin {i}")
        plt.title(f"Bin {i:02d} Scatter")
        plt.tight_layout()
        plt.savefig(os.path.join(per_bin_dir, f"bin_{i:02d}.png"), dpi=150)
        plt.close()

    # 2) Error histogram across all bins
    residuals = (preds_denorm - trues_denorm)[mask_flat].ravel()
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
        true_mask = mask_true[idx]
        pred_mask = mask_pred[idx]
        dist_true = np.where(true_mask, trues_denorm[idx], np.nan)
        dist_pred = np.where(pred_mask, preds_denorm[idx], np.nan)
        if PLOT_MAX_PROFILE_MM is not None:
            dist_true = np.where(dist_true < PLOT_MAX_PROFILE_MM, dist_true, np.nan)
            dist_pred = np.where(dist_pred < PLOT_MAX_PROFILE_MM, dist_pred, np.nan)
        ax.plot(theta, dist_true, label="True")
        ax.plot(theta, dist_pred, label="Pred")
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

        if EVAL_RANGES is None:
            raise ValueError("EVAL_RANGES must be set to plot world-frame overlays.")

        # Build full-session indices for density plots
        indices_all = np.arange(eval_proc.n, dtype=int)

        # Build specific range indices for overlay plots
        gathered = []
        range_labels = []
        range_index_sets = []
        for r in EVAL_RANGES:
            if r is None or len(r) != 2:
                raise ValueError(f"Invalid range entry: {r}. Expected [start, end).")
            start, end = int(r[0]), int(r[1])
            if end < start:
                start, end = end, start
            if end == start:
                continue
            r_indices = np.arange(start, end, dtype=int)
            gathered.append(r_indices)
            range_labels.append(f"{start}-{end}")
            range_index_sets.append((f"{start}-{end}", r_indices))
        indices = np.concatenate(gathered, axis=0) if gathered else np.array([], dtype=int)
        # Deduplicate while preserving order
        if indices.size:
            seen = set()
            unique = []
            for i in indices.tolist():
                if i not in seen:
                    seen.add(i)
                    unique.append(i)
            indices = np.array(unique, dtype=int)
        indices = indices[(indices >= 0) & (indices < eval_proc.n)]

        # Streaming inference helpers (avoid full-session caching to save RAM)
        centers_all = eval_proc.profile_centers[indices_all]

        def infer_chunk(idxs):
            X_eval = eval_proc.sonar_data[idxs].astype(np.float32)
            if USE_CNN:
                Xn_eval = (X_eval - x_mean) / x_std
                Xn_eval = np.transpose(Xn_eval, (0, 2, 1))  # N, L, C -> N, C, L
            else:
                Xn_eval = (X_eval - x_mean) / x_std
            with torch.no_grad():
                dist_out, mask_out = model(torch.from_numpy(Xn_eval).to(device))
                preds = dist_out.cpu().numpy()
                mask_logits = mask_out.cpu().numpy()
            preds = preds * y_std + y_mean
            mask_probs = 1.0 / (1.0 + np.exp(-mask_logits))
            pred_mask = mask_probs >= PRED_RETURN_THRESHOLD
            return preds, pred_mask

        if indices.size == 0:
            print("No valid indices for world-frame overlay.")
        else:
            idx_to_pos = {int(v): i for i, v in enumerate(indices.tolist())}
            true_eval = eval_proc.profiles[indices].astype(np.float32)
            true_mask_eval = (true_eval < SONAR_RANGE_MM)
            centers = eval_proc.profile_centers[indices]
            preds_eval, pred_mask_eval = infer_chunk(indices)

            def plot_range(index_subset, pos_subset, range_label):
                fig = plt.figure(figsize=(8, 8))
                cmap = plt.get_cmap('tab10')
                for k, (idx, pos) in enumerate(zip(index_subset, pos_subset)):
                    color = cmap(k % cmap.N)
                    az_deg = centers[pos]
                    rob_x = float(eval_proc.rob_x[idx])
                    rob_y = float(eval_proc.rob_y[idx])
                    rob_yaw = float(eval_proc.rob_yaw_deg[idx])

                    dist_true = np.where(true_mask_eval[pos], true_eval[pos], np.nan)
                    dist_pred = np.where(pred_mask_eval[pos], preds_eval[pos], np.nan)
                    if PLOT_MAX_PROFILE_MM is not None:
                        dist_true = np.where(dist_true < PLOT_MAX_PROFILE_MM, dist_true, np.nan)
                        dist_pred = np.where(dist_pred < PLOT_MAX_PROFILE_MM, dist_pred, np.nan)

                    x_t, y_t = DataProcessor.robot2world(az_deg, dist_true, rob_x, rob_y, rob_yaw)
                    x_p, y_p = DataProcessor.robot2world(az_deg, dist_pred, rob_x, rob_y, rob_yaw)

                    plt.plot(x_t, y_t, color=color, linewidth=1.5, linestyle='-')
                    plt.plot(x_p, y_p, color=color, linewidth=1.0, linestyle='--')

                plt.plot(eval_proc.rob_x[index_subset], eval_proc.rob_y[index_subset],
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
                plt.title(f"World-frame profiles: {EVAL_SESSION} ranges {range_label}")
                plt.tight_layout()

                out_path = os.path.join(
                    plots_dir,
                    f"world_profiles_{EVAL_SESSION}_ranges_{range_label}.png"
                )
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"Saved world-frame overlay to: {out_path}")

            if PLOT_RANGES_SEPARATE:
                for label, idxs in range_index_sets:
                    idxs = idxs[(idxs >= 0) & (idxs < eval_proc.n)]
                    if idxs.size == 0:
                        print(f"No valid indices for range {label}")
                        continue
                    pairs = [(int(v), idx_to_pos[int(v)]) for v in idxs if int(v) in idx_to_pos]
                    if len(pairs) == 0:
                        print(f"No cached indices for range {label}")
                        continue
                    idxs_f = np.array([p[0] for p in pairs], dtype=int)
                    pos_f = [p[1] for p in pairs]
                    plot_range(idxs_f, pos_f, label)
            else:
                ranges_label = "_".join(range_labels) if range_labels else "none"
                plot_range(indices, list(range(indices.size)), ranges_label)

            # -----------------------------
            # Density plots: chunked predictions
            # -----------------------------
            def get_density_bounds():
                if hasattr(eval_proc, "meta") and isinstance(eval_proc.meta, dict):
                    arena = eval_proc.meta.get("arena_bounds_mm", None)
                    if arena:
                        return (
                            float(arena["min_x"]),
                            float(arena["max_x"]),
                            float(arena["min_y"]),
                            float(arena["max_y"]),
                        )
                if hasattr(eval_proc, "wall_x") and hasattr(eval_proc, "wall_y"):
                    wall_x = np.asarray(eval_proc.wall_x, dtype=float)
                    wall_y = np.asarray(eval_proc.wall_y, dtype=float)
                    finite = np.isfinite(wall_x) & np.isfinite(wall_y)
                    if np.any(finite):
                        min_x = float(np.nanmin(wall_x[finite]))
                        max_x = float(np.nanmax(wall_x[finite]))
                        min_y = float(np.nanmin(wall_y[finite]))
                        max_y = float(np.nanmax(wall_y[finite]))
                        pad = 100.0
                        return min_x - pad, max_x + pad, min_y - pad, max_y + pad
                # Fallback to robot positions
                min_x = float(np.nanmin(eval_proc.rob_x))
                max_x = float(np.nanmax(eval_proc.rob_x))
                min_y = float(np.nanmin(eval_proc.rob_y))
                max_y = float(np.nanmax(eval_proc.rob_y))
                pad = 200.0
                return min_x - pad, max_x + pad, min_y - pad, max_y + pad

            def smooth_density(H, sigma_mm, bin_mm):
                sigma_px = float(sigma_mm) / float(bin_mm)
                if sigma_px <= 0:
                    return H
                try:
                    import cv2
                    k = int(np.ceil(sigma_px * 6))
                    k = max(3, k | 1)  # odd kernel size
                    return cv2.GaussianBlur(H.astype(np.float32), (k, k), sigmaX=sigma_px, sigmaY=sigma_px)
                except Exception:
                    try:
                        from scipy.ndimage import gaussian_filter
                        return gaussian_filter(H.astype(np.float32), sigma=sigma_px)
                    except Exception:
                        return H

            def compute_bounds_from_preds():
                min_x = float("inf")
                max_x = float("-inf")
                min_y = float("inf")
                max_y = float("-inf")
                found = False
                chunk_starts = np.arange(0, eval_proc.n, EVAL_CHUNK_STRIDE, dtype=int)
                for start in chunk_starts:
                    end = int(min(start + EVAL_CHUNK_SIZE, eval_proc.n))
                    idxs = np.arange(start, end, dtype=int)
                    if idxs.size == 0:
                        continue
                    preds_eval, pred_mask_eval = infer_chunk(idxs)
                    for pos, idx in enumerate(idxs.tolist()):
                        az_deg = centers_all[idx]
                        rob_x = float(eval_proc.rob_x[idx])
                        rob_y = float(eval_proc.rob_y[idx])
                        rob_yaw = float(eval_proc.rob_yaw_deg[idx])
                        dist_pred = np.where(pred_mask_eval[pos], preds_eval[pos], np.nan)
                        if PLOT_MAX_PROFILE_MM is not None:
                            dist_pred = np.where(dist_pred < PLOT_MAX_PROFILE_MM, dist_pred, np.nan)
                        x_p, y_p = DataProcessor.robot2world(az_deg, dist_pred, rob_x, rob_y, rob_yaw)
                        valid = np.isfinite(x_p) & np.isfinite(y_p)
                        if np.any(valid):
                            found = True
                            min_x = min(min_x, float(np.nanmin(x_p[valid])))
                            max_x = max(max_x, float(np.nanmax(x_p[valid])))
                            min_y = min(min_y, float(np.nanmin(y_p[valid])))
                            max_y = max(max_y, float(np.nanmax(y_p[valid])))
                if not found:
                    return None
                pad = 100.0
                return min_x - pad, max_x + pad, min_y - pad, max_y + pad

            if PRED_DENSITY_BOUNDS == 'auto':
                bounds = compute_bounds_from_preds()
                if bounds is None:
                    bounds = get_density_bounds()
            else:
                bounds = get_density_bounds()

            min_x, max_x, min_y, max_y = bounds
            bin_mm = float(PRED_DENSITY_BIN_MM)
            x_edges = np.arange(min_x, max_x + bin_mm, bin_mm)
            y_edges = np.arange(min_y, max_y + bin_mm, bin_mm)

            # Chunk through the entire session (non-overlapping)
            all_indices = np.arange(eval_proc.n, dtype=int)
            chunk_starts = np.arange(0, eval_proc.n, EVAL_CHUNK_STRIDE, dtype=int)
            for chunk_id, start in enumerate(chunk_starts):
                end = int(min(start + EVAL_CHUNK_SIZE, eval_proc.n))
                idxs = all_indices[start:end]
                if idxs.size == 0:
                    continue

                preds_eval, pred_mask_eval = infer_chunk(idxs)
                true_eval = eval_proc.profiles[idxs].astype(np.float32)
                true_mask_eval = (true_eval < SONAR_RANGE_MM)

                pred_points = []
                true_lines = []
                for pos, idx in enumerate(idxs.tolist()):
                    az_deg = centers_all[idx]
                    rob_x = float(eval_proc.rob_x[idx])
                    rob_y = float(eval_proc.rob_y[idx])
                    rob_yaw = float(eval_proc.rob_yaw_deg[idx])

                    dist_true = np.where(true_mask_eval[pos], true_eval[pos], np.nan)
                    dist_pred = np.where(pred_mask_eval[pos], preds_eval[pos], np.nan)
                    if PLOT_MAX_PROFILE_MM is not None:
                        dist_true = np.where(dist_true < PLOT_MAX_PROFILE_MM, dist_true, np.nan)
                        dist_pred = np.where(dist_pred < PLOT_MAX_PROFILE_MM, dist_pred, np.nan)

                    x_t, y_t = DataProcessor.robot2world(az_deg, dist_true, rob_x, rob_y, rob_yaw)
                    x_p, y_p = DataProcessor.robot2world(az_deg, dist_pred, rob_x, rob_y, rob_yaw)
                    true_lines.append((x_t, y_t))

                    valid = np.isfinite(x_p) & np.isfinite(y_p)
                    if np.any(valid):
                        pred_points.append(np.column_stack([x_p[valid], y_p[valid]]))

                if not pred_points:
                    continue

                pred_points = np.vstack(pred_points)
                H, _, _ = np.histogram2d(
                    pred_points[:, 0],
                    pred_points[:, 1],
                    bins=[x_edges, y_edges],
                )
                H_smooth = smooth_density(H, PRED_DENSITY_SIGMA_MM, bin_mm)

                fig = plt.figure(figsize=(8, 8))
                plt.imshow(
                    H_smooth.T,
                    origin='lower',
                    extent=[min_x, max_x, min_y, max_y],
                    cmap='hot',
                    alpha=0.9,
                )
                for x_t, y_t in true_lines:
                    plt.plot(x_t, y_t, color='cyan', linewidth=1.0, alpha=0.8)
                plt.plot(eval_proc.rob_x[idxs], eval_proc.rob_y[idxs],
                         'w.', markersize=3, alpha=0.9)
                plt.axis('equal')
                plt.xlabel("X [mm]")
                plt.ylabel("Y [mm]")
                plt.title(
                    f"Pred density (chunk {start}-{end - 1}, size {EVAL_CHUNK_SIZE})"
                )
                plt.tight_layout()
                out_path = os.path.join(
                    plots_dir,
                    f"world_pred_density_{EVAL_SESSION}_chunk_{start:04d}_{end-1:04d}.png"
                )
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"Saved density plot to: {out_path}")
