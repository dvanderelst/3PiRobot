#!/usr/bin/env python3
"""
SCRIPT_TrainEmulator2.py

Train a CNN emulator that predicts sonar readings (IID and distance) from
geometric profiles of the robot arena.

Architecture:
  - Input: 1D profile of wall distances (normalised by fixed global scale)
  - Conv layers -> AdaptiveAvgPool -> FC -> two regression heads:
      iid_head      : predicted IID in dB       (echo-present samples only)
      distance_head : predicted distance in mm  (echo-present samples only)
  - Echo presence is NOT predicted. Both heads are trained and evaluated on
    echo-present samples only (corrected_distance < max_dist_mm). For
    echo-absent situations the networks extrapolate: a flat far-away profile
    naturally maps to large distance and near-zero IID.

Artifacts saved to output_dir/:
  best_model_pytorch.pth, training_params.json,
  training_curve.png, scatter_plots.png, code_emulator2.zip
"""

import json
import os

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from Library.DataProcessor import DataCollection
from Library import CodeLogger


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Data
    session_paths: List[str] = field(default_factory=lambda: [
        "sessionB01",
        "sessionB02",
        "sessionB03",
        "sessionB04",
        "sessionB05",
    ])
    cache_dir: str = "Cache"

    # Profile
    opening_angle: float = 220.0
    profile_steps: int = 22
    profile_method: str = "min_bin"

    # Maximum distance in mm — two roles:
    #   1. Normalisation scale: profiles divided by this value before z-scoring.
    #   2. Echo presence threshold: samples with corrected_distance >= this value
    #      are treated as no-echo (sentinel from AcousticProcessing) and excluded
    #      from both regression losses.
    max_dist_mm: float = 3000.0

    # CNN architecture
    conv_channels: List[int] = field(default_factory=lambda: [32, 64])
    conv_kernel: int = 3
    fc_hidden: int = 64      # shared backbone FC
    head_hidden: int = 32    # per-head FC before output; 0 = linear head

    # Training
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    # Relative weight of distance MSE vs IID MSE in the combined loss.
    # Both are in normalised (z-scored) units so 1.0 is a reasonable starting point.
    dist_loss_weight: float = 1.0

    # Train / validation split (quadrant indices withheld per session)
    validation_quadrants: Dict[str, List[int]] = field(default_factory=lambda: {
        "sessionB01": [1],
        "sessionB02": [3],
        "sessionB03": [2],
        "sessionB04": [4],
        "sessionB05": [1],
    })

    # Output
    output_dir: str = "Emulator"
    seed: int = 42


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class EmulatorCNN(nn.Module):
    """
    1D CNN emulator: profile -> (iid_head, distance_head).

    Input shape:  (batch, profile_steps)  -- normalised wall distances
    Output dict:  {"iid": (batch, 1), "distance": (batch, 1)}
                  Both outputs are in normalised (z-scored) units.
    """
    def __init__(
        self,
        profile_steps: int,
        conv_channels: List[int],
        conv_kernel: int,
        fc_hidden: int,
        head_hidden: int = 0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, conv_kernel, padding=conv_kernel // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc   = nn.Sequential(
            nn.Linear(conv_channels[-1] * 8, fc_hidden),
            nn.ReLU(),
        )

        def make_head(in_dim: int, hidden: int) -> nn.Module:
            if hidden > 0:
                return nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
                )
            return nn.Linear(in_dim, 1)

        self.iid_head      = make_head(fc_hidden, head_hidden)
        self.distance_head = make_head(fc_hidden, head_hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (batch, profile_steps)  -- add channel dim
        z = self.conv(x.unsqueeze(1))   # (batch, C, L)
        z = self.pool(z).flatten(1)
        z = self.fc(z)
        return {
            "iid":      self.iid_head(z),
            "distance": self.distance_head(z),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(cfg: Config):
    """
    Load profiles, corrected_iid, corrected_distance and echo label from all
    sessions. Returns per-processor list for quadrant-based splitting.

    Returns
    -------
    per_proc : list of dicts, one per processor, each with keys:
        session_name, profiles, iid, distance, crossed, quadrants
    """
    dc = DataCollection(cfg.session_paths, cache_dir=cfg.cache_dir)
    dc.load_profiles(
        opening_angle=cfg.opening_angle,
        steps=cfg.profile_steps,
        profile_method=cfg.profile_method,
    )

    per_proc = []
    for proc in dc.processors:
        session_name = os.path.basename(proc.session)
        n = proc.n

        profiles = proc.profiles  # (n, profile_steps)

        # corrected_distance is in metres in the sonar_package; convert to mm.
        iid_vals  = proc.get_field('sonar_package', 'corrected_iid')
        dist_vals = proc.get_field('sonar_package', 'corrected_distance')

        iid_vals  = np.where(np.isfinite(iid_vals),  iid_vals,  0.0).astype(np.float32)
        dist_vals = np.where(np.isfinite(dist_vals), dist_vals,
                             cfg.max_dist_mm / 1000.0).astype(np.float32)
        dist_vals = dist_vals * 1000.0   # metres → mm

        # Echo presence label: True when distance < max_dist_mm.
        # Used to exclude sentinel no-echo values from regression training.
        crossed = (dist_vals < cfg.max_dist_mm)

        quads = proc.quadrants  # (n,) int array

        per_proc.append({
            "session_name": session_name,
            "profiles":     profiles.astype(np.float32),
            "iid":          iid_vals,
            "distance":     dist_vals,
            "crossed":      crossed,
            "quadrants":    quads,
        })
        print(f"  {session_name}: {n} samples, "
              f"echo_present={crossed.sum()}/{n} ({100*crossed.mean():.1f}%), "
              f"quadrants={sorted(set(quads.tolist()))}")

    return per_proc


def make_split(per_proc: List[dict], validation_quadrants: Dict[str, List[int]]):
    """
    Split per-processor data into train and validation sets.

    Samples whose quadrant is in validation_quadrants[session] go to validation;
    all others go to training. Sessions not listed contribute all data to training.
    """
    def empty():
        return {k: [] for k in ("profiles", "iid", "distance", "crossed")}

    train_parts = empty()
    val_parts   = empty()

    for pd in per_proc:
        name  = pd["session_name"]
        val_q = set(validation_quadrants.get(name, []))
        quads = pd["quadrants"]
        is_val = np.isin(quads, list(val_q)) if val_q else np.zeros(len(quads), dtype=bool)

        for key in ("profiles", "iid", "distance", "crossed"):
            arr = pd[key]
            train_parts[key].append(arr[~is_val])
            val_parts[key].append(arr[is_val])

    def concat(parts):
        return {k: np.concatenate(parts[k], axis=0) for k in parts}

    return concat(train_parts), concat(val_parts)


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_norm_stats(train: dict, max_dist_mm: float) -> dict:
    """
    Compute normalisation statistics from training data (echo-present samples only
    for IID and distance, so sentinel no-echo values don't bias the stats).

    x is divided by max_dist_mm then z-scored.
    IID and distance are z-scored independently.

    Returns dict with lists: x_mean, x_std, iid_mean, iid_std, dist_mean, dist_std.
    """
    x_norm = train["profiles"] / max_dist_mm
    x_mean = x_norm.mean(axis=0)
    x_std  = x_norm.std(axis=0)
    x_std  = np.where(x_std < 1e-8, 1.0, x_std)

    ep = train["crossed"].astype(bool)

    iid_vals  = train["iid"][ep]
    iid_mean  = float(iid_vals.mean()) if len(iid_vals) > 0 else 0.0
    iid_std   = float(iid_vals.std())  if len(iid_vals) > 1 else 1.0
    iid_std   = max(iid_std, 1e-8)

    dist_vals = train["distance"][ep]
    dist_mean = float(dist_vals.mean()) if len(dist_vals) > 0 else 0.0
    dist_std  = float(dist_vals.std())  if len(dist_vals) > 1 else 1.0
    dist_std  = max(dist_std, 1e-8)

    return {
        "x_mean":    x_mean.tolist(),
        "x_std":     x_std.tolist(),
        "iid_mean":  iid_mean,
        "iid_std":   iid_std,
        "dist_mean": dist_mean,
        "dist_std":  dist_std,
    }


def normalise_x(profiles: np.ndarray, max_dist_mm: float,
                x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return ((profiles / max_dist_mm) - x_mean) / x_std


def normalise(vals: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (vals - mean) / std


def denormalise(vals: np.ndarray, mean: float, std: float) -> np.ndarray:
    return vals * std + mean


# ══════════════════════════════════════════════════════════════════════════════
# Dataset / DataLoader
# ══════════════════════════════════════════════════════════════════════════════

class ProfileDataset(torch.utils.data.Dataset):
    def __init__(self, profiles_norm: np.ndarray, iid_norm: np.ndarray,
                 dist_norm: np.ndarray, crossed: np.ndarray):
        self.profiles = torch.as_tensor(profiles_norm, dtype=torch.float32)
        self.iid      = torch.as_tensor(iid_norm,      dtype=torch.float32)
        self.dist     = torch.as_tensor(dist_norm,     dtype=torch.float32)
        self.crossed  = torch.as_tensor(crossed.astype(np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        return self.profiles[idx], self.iid[idx], self.dist[idx], self.crossed[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model: nn.Module, loader: torch.utils.data.DataLoader,
              optimiser: Optional[torch.optim.Optimizer],
              dist_loss_weight: float,
              device: torch.device) -> Tuple[float, float]:
    """
    Run one epoch. If optimiser is None, runs in eval mode (validation).

    Both IID MSE and distance MSE are computed on echo-present samples only.

    Returns (mean_iid_mse, mean_dist_mse) in normalised units.
    """
    training = optimiser is not None
    model.train(training)

    total_iid_mse  = 0.0
    total_dist_mse = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for profiles, iid_norm, dist_norm, crossed in loader:
            profiles = profiles.to(device)
            iid_norm = iid_norm.to(device)
            dist_norm = dist_norm.to(device)
            crossed  = crossed.to(device)

            out      = model(profiles)
            pred_iid  = out["iid"].squeeze(1)       # (batch,)
            pred_dist = out["distance"].squeeze(1)   # (batch,)

            mask = crossed.bool()
            if mask.any():
                iid_mse  = ((pred_iid[mask]  - iid_norm[mask])  ** 2).mean()
                dist_mse = ((pred_dist[mask] - dist_norm[mask]) ** 2).mean()
            else:
                iid_mse  = torch.tensor(0.0, device=device)
                dist_mse = torch.tensor(0.0, device=device)

            loss = iid_mse + dist_loss_weight * dist_mse

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_iid_mse  += iid_mse.item()
            total_dist_mse += dist_mse.item()
            n_batches += 1

    denom = max(n_batches, 1)
    return total_iid_mse / denom, total_dist_mse / denom


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data")
    per_proc = load_dataset(cfg)
    train_data, val_data = make_split(per_proc, cfg.validation_quadrants)
    n_train     = len(train_data['profiles'])
    n_ep_train  = int(train_data['crossed'].sum())
    print(f"  Train: {n_train} samples ({n_ep_train} echo-present, "
          f"{n_train - n_ep_train} no-echo excluded from loss)")
    n_val    = len(val_data['profiles'])
    n_ep_val = int(val_data['crossed'].sum())
    print(f"  Val:   {n_val} samples ({n_ep_val} echo-present)")

    # ── Normalisation stats ───────────────────────────────────────────────────
    print("\n[2/5] Computing normalisation stats")
    stats = compute_norm_stats(train_data, cfg.max_dist_mm)
    x_mean    = np.array(stats["x_mean"], dtype=np.float32)
    x_std     = np.array(stats["x_std"],  dtype=np.float32)
    iid_mean  = stats["iid_mean"]
    iid_std   = stats["iid_std"]
    dist_mean = stats["dist_mean"]
    dist_std  = stats["dist_std"]
    print(f"  IID:      mean={iid_mean:.3f} dB,  std={iid_std:.3f} dB")
    print(f"  Distance: mean={dist_mean:.1f} mm, std={dist_std:.1f} mm")

    # ── Normalise inputs/outputs ──────────────────────────────────────────────
    def prep(split):
        x    = normalise_x(split["profiles"], cfg.max_dist_mm, x_mean, x_std)
        y_i  = normalise(split["iid"],      iid_mean,  iid_std)
        y_d  = normalise(split["distance"], dist_mean, dist_std)
        return (x.astype(np.float32), y_i.astype(np.float32),
                y_d.astype(np.float32), split["crossed"])

    train_x, train_yi, train_yd, train_c = prep(train_data)
    val_x,   val_yi,   val_yd,   val_c   = prep(val_data)

    train_ds = ProfileDataset(train_x, train_yi, train_yd, train_c)
    val_ds   = ProfileDataset(val_x,   val_yi,   val_yd,   val_c)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=False)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[3/5] Building model on {device}")
    model = EmulatorCNN(
        profile_steps=cfg.profile_steps,
        conv_channels=cfg.conv_channels,
        conv_kernel=cfg.conv_kernel,
        fc_hidden=cfg.fc_hidden,
        head_hidden=cfg.head_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for {cfg.epochs} epochs")
    history = {
        "train_iid_mse": [], "train_dist_mse": [],
        "val_iid_mse":   [], "val_dist_mse":   [],
    }
    best_val_loss = float("inf")
    best_epoch    = -1

    model_save_path = os.path.join(cfg.output_dir, "best_model_pytorch.pth")

    for epoch in range(1, cfg.epochs + 1):
        tr_iid, tr_dist = run_epoch(model, train_loader, optimiser,
                                    cfg.dist_loss_weight, device)
        va_iid, va_dist = run_epoch(model, val_loader,   None,
                                    cfg.dist_loss_weight, device)

        history["train_iid_mse"].append(tr_iid)
        history["train_dist_mse"].append(tr_dist)
        history["val_iid_mse"].append(va_iid)
        history["val_dist_mse"].append(va_dist)

        val_loss = va_iid + cfg.dist_loss_weight * va_dist
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({"model_state_dict": model.state_dict()}, model_save_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{cfg.epochs}  "
                  f"train IID={tr_iid:.4f}  dist={tr_dist:.4f}  |  "
                  f"val IID={va_iid:.4f}  dist={va_dist:.4f}"
                  + (" *" if epoch == best_epoch else ""))

    print(f"\n  Best epoch: {best_epoch}  val combined={best_val_loss:.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    print("\n[5/5] Saving artifacts")

    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # training_params.json -- fields read by Emulator.load() are marked
    training_params = {
        # Architecture (read by Emulator.load)
        "conv_channels":         cfg.conv_channels,
        "conv_kernel":           cfg.conv_kernel,
        "fc_hidden":             cfg.fc_hidden,
        "head_hidden":           cfg.head_hidden,
        # Profile params (read by Emulator.load)
        "profile_opening_angle": cfg.opening_angle,
        "profile_steps":         cfg.profile_steps,
        # Normalisation (read by Emulator.load)
        # norm_stats uses legacy keys x_mean/x_std/y_mean/y_std for IID backward compat;
        # distance norm is stored separately.
        "norm_stats": {
            "x_mean": stats["x_mean"],
            "x_std":  stats["x_std"],
            "y_mean": [iid_mean],
            "y_std":  [iid_std],
        },
        "normalize_x":            True,
        "normalize_y":            True,
        "calibration":            None,
        # Distance normalisation
        "dist_norm": {
            "dist_mean": dist_mean,
            "dist_std":  dist_std,
        },
        # Echo threshold (used at inference to gate outputs)
        "no_echo_min_distance_mm": cfg.max_dist_mm,
        # Config
        "dist_loss_weight": cfg.dist_loss_weight,
        "config":           asdict(cfg),
    }
    params_path = os.path.join(cfg.output_dir, "training_params.json")
    with open(params_path, "w") as f:
        json.dump(training_params, f, indent=2)
    print(f"  Saved {params_path}")

    # ── Training curve ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_ax = np.arange(1, cfg.epochs + 1)

    axes[0].plot(epochs_ax, history["train_iid_mse"],  label="train")
    axes[0].plot(epochs_ax, history["val_iid_mse"],    label="val")
    axes[0].axvline(best_epoch, color="k", linestyle="--", alpha=0.5,
                    label=f"best={best_epoch}")
    axes[0].set_title("IID MSE (normalised, echo-present)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs_ax, history["train_dist_mse"], label="train")
    axes[1].plot(epochs_ax, history["val_dist_mse"],   label="val")
    axes[1].axvline(best_epoch, color="k", linestyle="--", alpha=0.5,
                    label=f"best={best_epoch}")
    axes[1].set_title("Distance MSE (normalised, echo-present)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    curve_path = os.path.join(cfg.output_dir, "training_curve.png")
    plt.savefig(curve_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {curve_path}")

    # ── Scatter plots ─────────────────────────────────────────────────────────
    def predict_split(x_norm):
        """Run best model; return pred_iid_db and pred_dist_mm (denormalised)."""
        ds  = torch.utils.data.TensorDataset(
            torch.as_tensor(x_norm, dtype=torch.float32))
        ldr = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
        iid_preds, dist_preds = [], []
        with torch.no_grad():
            for (batch,) in ldr:
                out = model(batch.to(device))
                iid_preds.append(out["iid"].cpu().squeeze(1))
                dist_preds.append(out["distance"].cpu().squeeze(1))
        pred_iid  = denormalise(torch.cat(iid_preds).numpy(),  iid_mean,  iid_std)
        pred_dist = denormalise(torch.cat(dist_preds).numpy(), dist_mean, dist_std)
        return pred_iid, pred_dist

    tr_pred_iid, tr_pred_dist = predict_split(train_x)
    va_pred_iid, va_pred_dist = predict_split(val_x)

    tr_true_iid  = train_data["iid"]
    va_true_iid  = val_data["iid"]
    tr_true_dist = train_data["distance"]
    va_true_dist = val_data["distance"]

    from scipy.stats import pearsonr

    def scatter_panel(ax, true_tr, pred_tr, mask_tr, true_va, pred_va, mask_va,
                      xlabel, ylabel, title):
        if mask_tr.any():
            ax.scatter(true_tr[mask_tr], pred_tr[mask_tr],
                       alpha=0.2, s=5, label=f"train n={mask_tr.sum()}")
        if mask_va.any():
            ax.scatter(true_va[mask_va], pred_va[mask_va],
                       alpha=0.3, s=5, color="orange", label=f"val n={mask_va.sum()}")
        all_true = np.concatenate([true_tr[mask_tr], true_va[mask_va]])
        all_pred = np.concatenate([pred_tr[mask_tr], pred_va[mask_va]])
        if len(all_true) > 0:
            lo = min(all_true.min(), all_pred.min())
            hi = max(all_true.max(), all_pred.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        if mask_tr.sum() > 1:
            r, _ = pearsonr(true_tr[mask_tr], pred_tr[mask_tr])
            rmse = float(np.sqrt(np.mean((true_tr[mask_tr] - pred_tr[mask_tr])**2)))
            ax.annotate(f"train  r={r:.3f}  RMSE={rmse:.2f}",
                        xy=(0.04, 0.93), xycoords="axes fraction",
                        fontsize=8, color="steelblue")
        if mask_va.sum() > 1:
            r, _ = pearsonr(true_va[mask_va], pred_va[mask_va])
            rmse = float(np.sqrt(np.mean((true_va[mask_va] - pred_va[mask_va])**2)))
            ax.annotate(f"val    r={r:.3f}  RMSE={rmse:.2f}",
                        xy=(0.04, 0.86), xycoords="axes fraction",
                        fontsize=8, color="orange")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(markerscale=3)

    tr_ep = train_c.astype(bool)
    va_ep = val_c.astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter_panel(axes[0],
                  tr_true_iid,  tr_pred_iid,  tr_ep,
                  va_true_iid,  va_pred_iid,  va_ep,
                  "True IID (dB)", "Predicted IID (dB)",
                  "IID (echo-present only)")

    scatter_panel(axes[1],
                  tr_true_dist, tr_pred_dist, tr_ep,
                  va_true_dist, va_pred_dist, va_ep,
                  "True distance (mm)", "Predicted distance (mm)",
                  "Distance (echo-present only)")

    plt.tight_layout()
    scatter_path = os.path.join(cfg.output_dir, "scatter_plots.png")
    plt.savefig(scatter_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {scatter_path}")

    # ── IID residuals by distance bin ─────────────────────────────────────────
    all_true_iid  = np.concatenate([tr_true_iid[tr_ep],  va_true_iid[va_ep]])
    all_pred_iid  = np.concatenate([tr_pred_iid[tr_ep],  va_pred_iid[va_ep]])
    all_true_dist = np.concatenate([tr_true_dist[tr_ep], va_true_dist[va_ep]])
    residuals     = all_true_iid - all_pred_iid

    n_bins    = 5
    bin_edges = np.percentile(all_true_dist, np.linspace(0, 100, n_bins + 1))
    bin_edges[0]  -= 1.0
    bin_edges[-1] += 1.0

    fig, axes = plt.subplots(1, n_bins, figsize=(3.5 * n_bins, 4), sharey=True)
    for i, ax in enumerate(axes):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (all_true_dist >= lo) & (all_true_dist < hi)
        resid  = residuals[mask]
        ax.hist(resid, bins=25, color="#4C72B0", alpha=0.8, edgecolor="white", lw=0.4)
        ax.axvline(0, color="k", lw=1, linestyle="--")
        ax.set_title(f"{lo:.0f}–{hi:.0f} mm\nn={mask.sum()}", fontsize=9)
        ax.set_xlabel("IID residual (dB)")
        if i == 0:
            ax.set_ylabel("Count")
        std = float(resid.std()) if len(resid) > 1 else 0.0
        ax.annotate(f"σ = {std:.2f} dB", xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9)

    fig.suptitle("IID residuals (true − predicted) by distance bin  [train + val, echo-present]",
                 fontsize=10)
    plt.tight_layout()
    resid_path = os.path.join(cfg.output_dir, "iid_residuals_by_distance.png")
    plt.savefig(resid_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {resid_path}")

    # ── Code log ──────────────────────────────────────────────────────────────
    CodeLogger.log_code(cfg.output_dir, [".", "Library"], label="emulator2")

    print("\nDone.")
    print(f"  Artifacts in: {cfg.output_dir}/")
    return model, training_params


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = Config()
    train(cfg)
