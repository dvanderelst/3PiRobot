import json
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Library import DataProcessor
from Library.SonarWallSteering import SteeringConfig


# ============================================
# CONFIGURATION
# ============================================
sessions = ['sessionB01', 'sessionB02', 'sessionB03', 'sessionB04', 'sessionB05']
output_dir = 'TrainingCurvature'

# Data/target generation
profile_opening_angle_deg = 90.0
profile_steps = 15
distance_cutoff_mm = 1500.0
window_size = 7

# Spatial split
train_quadrants = [0, 1, 2]
test_quadrant = 3
validation_split = 0.2
random_seed = 42

# Curvature planner settings used for target generation.
steering_config = SteeringConfig(
    extent_mm=2500.0,
    grid_mm=20.0,
    sigma_perp_mm=40.0,
    sigma_para_mm=120.0,
    apply_heatmap_smoothing=False,
    occ_block_threshold=0.10,
    robot_radius_mm=80.0,
    safety_margin_mm=120.0,
    circle_radius_min_mm=250.0,
    circle_radius_max_mm=2500.0,
    circle_radius_step_mm=50.0,
    circle_arc_samples=220,
    circle_horizon_x_mm=1800.0,
    circle_radius_tie_mm=100.0,
)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
num_epochs = 100
learning_rate = 1e-3
weight_decay = 1e-4
dropout_rate = 0.2
hidden_size = 128
pose_hidden = 24
val_patience = 12
num_workers = 0

# Weighted-loss options (helps when zero-curvature dominates).
use_weighted_loss = True
weight_alpha_abs = 2.0
weight_beta_nonzero = 1.0
nonzero_eps_inv_mm = 2e-4
high_turn_quantile = 0.75  # Top quantile of |true curvature| considered "turn events"
high_turn_quantiles_eval = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

# Cache/compute behavior
force_recompute_curvatures = False
parallel_curvature = True
curvature_workers = None
curvature_backend = 'thread'  # 'thread' or 'process'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def wrap_deg(deg):
    return (deg + 180.0) % 360.0 - 180.0


def safe_corrcoef(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return np.nan
    x = x[valid]
    y = y[valid]
    rx = np.empty_like(x)
    ry = np.empty_like(y)
    rx[np.argsort(x, kind='mergesort')] = np.arange(len(x), dtype=np.float64)
    ry[np.argsort(y, kind='mergesort')] = np.arange(len(y), dtype=np.float64)
    return safe_corrcoef(rx, ry)


class CurvatureDataset(Dataset):
    def __init__(self, x_sonar, x_pose, y, w=None):
        self.x_sonar = torch.from_numpy(np.asarray(x_sonar, dtype=np.float32))
        self.x_pose = torch.from_numpy(np.asarray(x_pose, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        if w is None:
            w = np.ones_like(y, dtype=np.float32)
        self.w = torch.from_numpy(np.asarray(w, dtype=np.float32))

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        return self.x_sonar[idx], self.x_pose[idx], self.y[idx], self.w[idx]


class SonarPoseCurvatureNet(nn.Module):
    def __init__(self, window_n, dropout=0.2, hidden=128, pose_hidden_dim=24):
        super().__init__()
        self.window_n = int(window_n)

        self.sonar_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 2, 200)
            feat = self.sonar_encoder(dummy)
            self.sonar_feat_dim = int(feat.reshape(1, -1).shape[1])

        self.pose_encoder = nn.Sequential(
            nn.Linear(3, pose_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        step_dim = self.sonar_feat_dim + pose_hidden_dim
        self.temporal = nn.GRU(input_size=step_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_sonar, x_pose):
        b, w, _, _ = x_sonar.shape
        x = x_sonar.reshape(b * w, 200, 2).permute(0, 2, 1)
        sf = self.sonar_encoder(x).reshape(b * w, -1)
        pf = self.pose_encoder(x_pose.reshape(b * w, 3))
        step = torch.cat([sf, pf], dim=1).reshape(b, w, -1)
        _, h_n = self.temporal(step)
        h = h_n[-1]
        return self.head(h).squeeze(1)


def load_base_data():
    print('Loading multi-session data...')
    dc = DataProcessor.DataCollection(sessions)
    sonar_data = np.asarray(dc.load_sonar(flatten=False), dtype=np.float32)
    profiles_mm, _ = dc.load_profiles(
        opening_angle=profile_opening_angle_deg,
        steps=profile_steps,
        fill_nans=False,
    )
    profiles_mm = np.asarray(profiles_mm, dtype=np.float32)
    y_curv = np.asarray(dc.load_curvatures(
        distance_cutoff_mm=distance_cutoff_mm,
        force_recompute=force_recompute_curvatures,
        steering_config=steering_config,
        show_progress=True,
        parallel=parallel_curvature,
        num_workers=curvature_workers,
        backend=curvature_backend,
    ), dtype=np.float32)

    quadrants = np.asarray(dc.quadrants, dtype=np.int64)
    rob_x = np.asarray(dc.rob_x, dtype=np.float32)
    rob_y = np.asarray(dc.rob_y, dtype=np.float32)
    rob_yaw = np.asarray(dc.rob_yaw_deg, dtype=np.float32)

    session_ids = []
    for i, p in enumerate(dc.processors):
        session_ids.append(np.full(p.n, i, dtype=np.int32))
    session_ids = np.concatenate(session_ids, axis=0)

    finite_mask = np.isfinite(sonar_data).all(axis=(1, 2))
    finite_mask &= np.isfinite(profiles_mm).all(axis=1)
    finite_mask &= np.isfinite(y_curv)
    finite_mask &= np.isfinite(rob_x) & np.isfinite(rob_y) & np.isfinite(rob_yaw)

    print(f'Raw samples: {len(sonar_data)}')
    print(f'Finite samples: {int(np.sum(finite_mask))}')

    return {
        'sonar': sonar_data,
        'curvature': y_curv,
        'quadrants': quadrants,
        'rob_x': rob_x,
        'rob_y': rob_y,
        'rob_yaw_deg': rob_yaw,
        'session_ids': session_ids,
        'finite_mask': finite_mask,
    }


def build_windowed_dataset(base):
    sonar = base['sonar']
    curv = base['curvature']
    quadrants = base['quadrants']
    rob_x = base['rob_x']
    rob_y = base['rob_y']
    rob_yaw = base['rob_yaw_deg']
    session_ids = base['session_ids']
    finite_mask = base['finite_mask']

    n = len(curv)
    sample_idx = []
    for t in range(window_size - 1, n):
        i0 = t - window_size + 1
        if session_ids[i0] != session_ids[t]:
            continue
        if not np.all(session_ids[i0:t + 1] == session_ids[t]):
            continue
        if not np.all(finite_mask[i0:t + 1]):
            continue
        sample_idx.append(t)

    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    m = len(sample_idx)
    if m == 0:
        raise ValueError('No valid windowed samples found.')

    x_sonar = np.zeros((m, window_size, 200, 2), dtype=np.float32)
    x_pose = np.zeros((m, window_size, 3), dtype=np.float32)
    y = np.zeros(m, dtype=np.float32)
    q = np.zeros(m, dtype=np.int64)

    for k, t in enumerate(sample_idx):
        i0 = t - window_size + 1
        win = np.arange(i0, t + 1)
        x_sonar[k] = sonar[win]
        y[k] = curv[t]
        q[k] = quadrants[t]

        ax = float(rob_x[t])
        ay = float(rob_y[t])
        ayaw = float(rob_yaw[t])
        rx, ry = DataProcessor.world2robot(
            np.asarray(rob_x[win], dtype=np.float32),
            np.asarray(rob_y[win], dtype=np.float32),
            ax, ay, ayaw
        )
        ryaw = wrap_deg(np.asarray(rob_yaw[win], dtype=np.float32) - ayaw)
        x_pose[k, :, 0] = np.asarray(rx, dtype=np.float32)
        x_pose[k, :, 1] = np.asarray(ry, dtype=np.float32)
        x_pose[k, :, 2] = np.asarray(ryaw, dtype=np.float32)

    print(f'Windowed samples: {m}')
    return {'x_sonar': x_sonar, 'x_pose': x_pose, 'y': y, 'quadrants': q, 'anchor_indices': sample_idx}


def split_by_quadrant(data):
    x_sonar = data['x_sonar']
    x_pose = data['x_pose']
    y = data['y']
    q = data['quadrants']

    train_mask = np.isin(q, np.asarray(train_quadrants, dtype=np.int64))
    test_mask = (q == int(test_quadrant))
    x_s_train_full = x_sonar[train_mask]
    x_p_train_full = x_pose[train_mask]
    y_train_full = y[train_mask]
    x_s_test = x_sonar[test_mask]
    x_p_test = x_pose[test_mask]
    y_test = y[test_mask]

    idx = np.arange(len(y_train_full))
    idx_train, idx_val = train_test_split(idx, test_size=float(validation_split), random_state=int(random_seed))

    split = {
        'x_s_train': x_s_train_full[idx_train],
        'x_p_train': x_p_train_full[idx_train],
        'y_train': y_train_full[idx_train],
        'x_s_val': x_s_train_full[idx_val],
        'x_p_val': x_p_train_full[idx_val],
        'y_val': y_train_full[idx_val],
        'x_s_test': x_s_test,
        'x_p_test': x_p_test,
        'y_test': y_test,
    }
    print(f"Split sizes: train={len(split['y_train'])}, val={len(split['y_val'])}, test={len(split['y_test'])}")
    return split


def scale_targets(split):
    mu = float(np.mean(split['y_train']))
    sigma = float(np.std(split['y_train']))
    if sigma < 1e-8:
        sigma = 1.0
    out = {}
    for key, val in split.items():
        if key.startswith('y_'):
            out[key] = ((val - mu) / sigma).astype(np.float32)
        else:
            out[key] = val
    return out, mu, sigma


def build_train_weights(y_train_unscaled):
    y_abs = np.abs(np.asarray(y_train_unscaled, dtype=np.float32))
    p90 = float(np.percentile(y_abs, 90.0)) if y_abs.size > 0 else 1.0
    p90 = max(p90, 1e-6)
    w = np.ones_like(y_abs, dtype=np.float32)
    w += float(weight_alpha_abs) * np.clip(y_abs / p90, 0.0, 2.0)
    w += float(weight_beta_nonzero) * (y_abs > float(nonzero_eps_inv_mm)).astype(np.float32)
    w /= max(float(np.mean(w)), 1e-6)
    return w.astype(np.float32)


def eval_model(model, loader, loss_fn):
    model.eval()
    loss_sum = 0.0
    n = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xs, xp, y, _w in loader:
            xs = xs.to(device, non_blocking=True)
            xp = xp.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yp = model(xs, xp)
            loss = loss_fn(yp, y)
            b = y.shape[0]
            loss_sum += float(loss.item()) * b
            n += b
            y_true.append(y.cpu().numpy())
            y_pred.append(yp.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0) if y_true else np.array([], dtype=np.float32)
    y_pred = np.concatenate(y_pred, axis=0) if y_pred else np.array([], dtype=np.float32)
    return loss_sum / max(n, 1), y_true, y_pred


def train_model(split_scaled, train_weights):
    train_ds = CurvatureDataset(split_scaled['x_s_train'], split_scaled['x_p_train'], split_scaled['y_train'], w=train_weights)
    val_ds = CurvatureDataset(split_scaled['x_s_val'], split_scaled['x_p_val'], split_scaled['y_val'])
    test_ds = CurvatureDataset(split_scaled['x_s_test'], split_scaled['x_p_test'], split_scaled['y_test'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SonarPoseCurvatureNet(window_n=window_size, dropout=dropout_rate, hidden=hidden_size, pose_hidden_dim=pose_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn_eval = nn.HuberLoss(delta=1.0)
    loss_fn_train = nn.HuberLoss(delta=1.0, reduction='none')

    best_state = None
    best_val = np.inf
    no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_sum = 0.0
        train_n = 0
        for xs, xp, y, w in train_loader:
            xs = xs.to(device, non_blocking=True)
            xp = xp.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            yp = model(xs, xp)
            if use_weighted_loss:
                per = loss_fn_train(yp, y)
                loss = torch.sum(per * w) / torch.sum(w)
            else:
                loss = loss_fn_eval(yp, y)
            loss.backward()
            optimizer.step()
            b = y.shape[0]
            train_sum += float(loss.item()) * b
            train_n += b

        train_loss = train_sum / max(train_n, 1)
        val_loss, _, _ = eval_model(model, val_loader, loss_fn_eval)
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        print(f"Epoch {epoch:03d}/{num_epochs}: train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= val_patience:
                print(f"Early stopping at epoch {epoch} (patience={val_patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, y_test_true, y_test_pred = eval_model(model, test_loader, loss_fn_eval)
    print(f'Best val loss: {best_val:.6f}')
    print(f'Test loss: {test_loss:.6f}')
    return model, history, float(best_val), float(test_loss), y_test_true, y_test_pred


def unscale(y_scaled, mu, sigma):
    return y_scaled * sigma + mu


def prediction_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true
    abs_err = np.abs(residual)
    abs_true = np.abs(y_true)
    abs_pred = np.abs(y_pred)
    return {
        'corr': safe_corrcoef(y_true, y_pred),
        'spearman_corr': safe_spearman(y_true, y_pred),
        'mae': float(np.mean(abs_err)) if len(y_true) > 0 else np.nan,
        'rmse': float(np.sqrt(np.mean((residual) ** 2))) if len(y_true) > 0 else np.nan,
        'sign_acc': float(np.mean(np.sign(y_pred) == np.sign(y_true))) if len(y_true) > 0 else np.nan,
        'mae_abs_mag': float(np.mean(np.abs(abs_pred - abs_true))) if len(y_true) > 0 else np.nan,
        'corr_abs': safe_corrcoef(abs_true, abs_pred),
        'spearman_abs': safe_spearman(abs_true, abs_pred),
        'corr_abs_true_vs_abs_err': safe_corrcoef(abs_true, abs_err),
        'spearman_abs_true_vs_abs_err': safe_spearman(abs_true, abs_err),
    }


def control_metrics(y_true, y_pred, turn_quantile=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if turn_quantile is None:
        turn_quantile = float(high_turn_quantile)
    if y_true.size == 0:
        return {
            'turn_quantile': float(turn_quantile),
            'turn_threshold_abs_kappa': np.nan,
            'turn_recall': np.nan,
            'turn_precision': np.nan,
            'turn_f1': np.nan,
            'prediction_jitter_std_diff': np.nan,
            'prediction_jitter_mean_abs_diff': np.nan,
        }

    abs_true = np.abs(y_true)
    abs_pred = np.abs(y_pred)
    thr = float(np.quantile(abs_true, float(turn_quantile)))
    true_turn = abs_true >= thr
    pred_turn = abs_pred >= thr

    tp = float(np.sum(true_turn & pred_turn))
    fp = float(np.sum((~true_turn) & pred_turn))
    fn = float(np.sum(true_turn & (~pred_turn)))
    recall = tp / max(tp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    pred_diff = np.diff(y_pred) if y_pred.size > 1 else np.array([], dtype=np.float64)
    jitter_std = float(np.std(pred_diff)) if pred_diff.size > 0 else np.nan
    jitter_mad = float(np.mean(np.abs(pred_diff))) if pred_diff.size > 0 else np.nan

    return {
        'turn_quantile': float(turn_quantile),
        'turn_threshold_abs_kappa': thr,
        'turn_recall': float(recall),
        'turn_precision': float(precision),
        'turn_f1': float(f1),
        'prediction_jitter_std_diff': jitter_std,
        'prediction_jitter_mean_abs_diff': jitter_mad,
    }


def save_artifacts(model, history, best_val, test_loss, split, y_test_true_scaled, y_test_pred_scaled, mu, sigma):
    ensure_dir(output_dir)
    model_path = os.path.join(output_dir, 'best_curvature_model.pth')
    torch.save(model.state_dict(), model_path)
    scaler_path = os.path.join(output_dir, 'curvature_target_scaler.json')
    with open(scaler_path, 'w') as f:
        json.dump({'mean': float(mu), 'std': float(sigma)}, f, indent=2)

    y_true = unscale(y_test_true_scaled, mu, sigma)
    y_pred = unscale(y_test_pred_scaled, mu, sigma)
    residual = y_pred - y_true
    abs_err = np.abs(residual)
    abs_true = np.abs(y_true)
    model_m = prediction_metrics(y_true, y_pred)
    ctrl_m = control_metrics(y_true, y_pred, turn_quantile=high_turn_quantile)
    ctrl_tradeoff = []
    for q in high_turn_quantiles_eval:
        ctrl_tradeoff.append(control_metrics(y_true, y_pred, turn_quantile=float(q)))
    finite_tradeoff = [m for m in ctrl_tradeoff if np.isfinite(m['turn_f1'])]
    best_turn_op = max(finite_tradeoff, key=lambda m: m['turn_f1']) if finite_tradeoff else None

    y_pred_zero = np.zeros_like(y_true)
    y_pred_train_mean = np.full_like(y_true, float(np.mean(split['y_train'])))
    baseline_zero = prediction_metrics(y_true, y_pred_zero)
    baseline_train_mean = prediction_metrics(y_true, y_pred_train_mean)

    sign_true = np.sign(y_true).astype(np.int8)
    sign_pred = np.sign(y_pred).astype(np.int8)
    sign_vals = [-1, 0, 1]
    sign_conf = np.zeros((3, 3), dtype=np.int64)
    for i, s_t in enumerate(sign_vals):
        for j, s_p in enumerate(sign_vals):
            sign_conf[i, j] = int(np.sum((sign_true == s_t) & (sign_pred == s_p)))

    summary = {
        'sessions': sessions,
        'device': str(device),
        'profile_opening_angle_deg': float(profile_opening_angle_deg),
        'profile_steps': int(profile_steps),
        'distance_cutoff_mm': float(distance_cutoff_mm),
        'window_size': int(window_size),
        'train_quadrants': [int(x) for x in train_quadrants],
        'test_quadrant': int(test_quadrant),
        'validation_split': float(validation_split),
        'random_seed': int(random_seed),
        'model': {
            'type': 'SonarPoseCurvatureNet',
            'dropout_rate': float(dropout_rate),
            'hidden_size': int(hidden_size),
            'pose_hidden': int(pose_hidden),
        },
        'training': {
            'batch_size': int(batch_size),
            'num_epochs': int(num_epochs),
            'learning_rate': float(learning_rate),
            'weight_decay': float(weight_decay),
            'val_patience': int(val_patience),
            'use_weighted_loss': bool(use_weighted_loss),
            'weight_alpha_abs': float(weight_alpha_abs),
            'weight_beta_nonzero': float(weight_beta_nonzero),
            'nonzero_eps_inv_mm': float(nonzero_eps_inv_mm),
        },
        'split_sizes': {
            'train': int(len(split['y_train'])),
            'val': int(len(split['y_val'])),
            'test': int(len(split['y_test'])),
        },
        'metrics': {
            'best_val_loss_scaled': float(best_val),
            'test_loss_scaled': float(test_loss),
            'test_corr': model_m['corr'],
            'test_spearman_corr': model_m['spearman_corr'],
            'test_corr_abs_magnitude': model_m['corr_abs'],
            'test_spearman_abs_magnitude': model_m['spearman_abs'],
            'test_mae_inv_mm': model_m['mae'],
            'test_rmse_inv_mm': model_m['rmse'],
            'test_sign_accuracy': model_m['sign_acc'],
            'test_mae_abs_magnitude': model_m['mae_abs_mag'],
            'test_corr_abs_true_vs_abs_error': model_m['corr_abs_true_vs_abs_err'],
            'test_spearman_abs_true_vs_abs_error': model_m['spearman_abs_true_vs_abs_err'],
            'test_sign_confusion_rows_true_cols_pred': sign_conf.tolist(),
            'control': ctrl_m,
            'control_turn_tradeoff': ctrl_tradeoff,
            'control_best_turn_operating_point': best_turn_op,
        },
        'baselines': {'zero_predictor': baseline_zero, 'train_mean_predictor': baseline_train_mean},
        'paths': {'model': model_path, 'scaler': scaler_path},
    }

    summary_path = os.path.join(output_dir, 'curvature_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    history_path = os.path.join(output_dir, 'curvature_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(7, 4))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss (scaled target)')
    plt.title('Curvature Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'curvature_training_loss.png'), dpi=180, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.7)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], '--', color='gray', lw=1.2)
    plt.xlabel('True Curvature (1/mm)')
    plt.ylabel('Predicted Curvature (1/mm)')
    plt.title('Test: True vs Predicted Curvature')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'curvature_test_scatter.png'), dpi=180, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(residual, bins=50, alpha=0.85, color='tab:blue')
    plt.axvline(0.0, color='black', linestyle='--', linewidth=1.2)
    plt.xlabel('Residual (pred - true) [1/mm]')
    plt.ylabel('Count')
    plt.title('Test Residual Histogram')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'curvature_test_residual_hist.png'), dpi=180, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(abs_true, abs_err, s=10, alpha=0.7)
    plt.xlabel('|True Curvature| (1/mm)')
    plt.ylabel('|Error| (1/mm)')
    plt.title('Test |Error| vs |True Curvature|')
    plt.text(
        0.03, 0.97,
        f"Pearson r = {model_m['corr_abs_true_vs_abs_err']:.3f}\nSpearman rho = {model_m['spearman_abs_true_vs_abs_err']:.3f}",
        transform=plt.gca().transAxes,
        va='top', ha='left', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'curvature_test_abs_error_vs_abs_true.png'), dpi=180, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(sign_conf, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Count')
    labels = ['-1', '0', '+1']
    ax.set_xticks(range(3), labels=labels)
    ax.set_yticks(range(3), labels=labels)
    ax.set_xlabel('Predicted sign')
    ax.set_ylabel('True sign')
    ax.set_title('Test Sign Confusion')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(sign_conf[i, j]), ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'curvature_test_sign_confusion.png'), dpi=180, bbox_inches='tight')
    plt.close()

    print(f'Saved model: {model_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved history: {history_path}')
    print(
        f"Diagnostics | corr={model_m['corr']:.3f}, spearman={model_m['spearman_corr']:.3f}, "
        f"MAE={model_m['mae']:.6f}, RMSE={model_m['rmse']:.6f}, sign_acc={model_m['sign_acc']:.3f}, "
        f"corr(|true|,|err|)={model_m['corr_abs_true_vs_abs_err']:.3f}"
    )
    print(
        f"Control | turn_recall={ctrl_m['turn_recall']:.3f}, turn_precision={ctrl_m['turn_precision']:.3f}, "
        f"turn_f1={ctrl_m['turn_f1']:.3f}, jitter_std={ctrl_m['prediction_jitter_std_diff']:.6f}, "
        f"q={ctrl_m['turn_quantile']:.2f}"
    )
    if best_turn_op is not None:
        print(
            f"Best turn-op | q={best_turn_op['turn_quantile']:.2f}, "
            f"thr={best_turn_op['turn_threshold_abs_kappa']:.6f}, "
            f"recall={best_turn_op['turn_recall']:.3f}, "
            f"precision={best_turn_op['turn_precision']:.3f}, "
            f"f1={best_turn_op['turn_f1']:.3f}"
        )
    for m in ctrl_tradeoff:
        print(
            f"Turn tradeoff | q={m['turn_quantile']:.2f} "
            f"R={m['turn_recall']:.3f} P={m['turn_precision']:.3f} F1={m['turn_f1']:.3f}"
        )
    print(
        f"Baselines | zero_mae={baseline_zero['mae']:.6f}, mean_mae={baseline_train_mean['mae']:.6f}, "
        f"model_mae={model_m['mae']:.6f}"
    )


def main():
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    ensure_dir(output_dir)

    base = load_base_data()
    windowed = build_windowed_dataset(base)
    split = split_by_quadrant(windowed)
    split_scaled, mu, sigma = scale_targets(split)
    train_weights = build_train_weights(split['y_train'])

    model, history, best_val, test_loss, y_test_true_scaled, y_test_pred_scaled = train_model(
        split_scaled=split_scaled,
        train_weights=train_weights,
    )
    save_artifacts(
        model=model,
        history=history,
        best_val=best_val,
        test_loss=test_loss,
        split=split,
        y_test_true_scaled=y_test_true_scaled,
        y_test_pred_scaled=y_test_pred_scaled,
        mu=mu,
        sigma=sigma,
    )


if __name__ == '__main__':
    main()
