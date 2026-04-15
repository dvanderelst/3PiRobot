import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


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
        conv_channels: List[int] = [32, 64],
        conv_kernel: int = 5,
        fc_hidden: int = 64,
        head_hidden: int = 32,
    ):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [nn.Conv1d(in_ch, out_ch, conv_kernel, padding=conv_kernel // 2), nn.ReLU()]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc   = nn.Sequential(nn.Linear(conv_channels[-1] * 8, fc_hidden), nn.ReLU())

        def make_head(in_dim: int, hidden: int) -> nn.Module:
            if hidden > 0:
                return nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
                )
            return nn.Linear(in_dim, 1)

        self.iid_head      = make_head(fc_hidden, head_hidden)
        self.distance_head = make_head(fc_hidden, head_hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.conv(x.unsqueeze(1))   # (batch, C, L)
        z = self.pool(z).flatten(1)
        z = self.fc(z)
        return {"iid": self.iid_head(z), "distance": self.distance_head(z)}


class Emulator:
    """
    Environment emulator that predicts IID (dB) and distance (mm) from wall-distance
    profiles using a shared CNN backbone with two regression heads.

    Both heads are trained on echo-present samples only; echo presence is not predicted.
    At inference, out-of-range (no-echo) profiles naturally extrapolate toward large
    distance and near-zero IID.

    Profile parameters (opening_angle, steps) are read from the training artifact so
    that profile generation in EnvironmentSimulator is always consistent with training.
    """

    def __init__(
        self,
        model: nn.Module,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        iid_mean: float,
        iid_std: float,
        dist_mean: float,
        dist_std: float,
        max_dist_mm: float,
        profile_opening_angle: float,
        profile_steps: int,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.model.eval()

        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_std  = np.asarray(x_std,  dtype=np.float32)
        self.iid_mean  = float(iid_mean)
        self.iid_std   = float(iid_std)
        self.dist_mean = float(dist_mean)
        self.dist_std  = float(dist_std)
        self.max_dist_mm = float(max_dist_mm)

        self.profile_opening_angle = float(profile_opening_angle)
        self.profile_steps = int(profile_steps)

    @staticmethod
    def load(emulator_dir: str = "Emulator", device: Optional[str] = None) -> 'Emulator':
        """Load a trained emulator from disk."""
        params_path = os.path.join(emulator_dir, "training_params.json")
        if not os.path.exists(params_path):
            raise FileNotFoundError(
                f"Emulator training params not found at {params_path}. "
                "Please run SCRIPT_TrainEmulator2.py first."
            )

        with open(params_path) as f:
            params = json.load(f)

        profile_opening_angle = params["profile_opening_angle"]
        profile_steps         = int(params["profile_steps"])

        # Architecture
        conv_channels = list(params["conv_channels"])
        conv_kernel   = int(params["conv_kernel"])
        fc_hidden     = int(params["fc_hidden"])
        head_hidden   = int(params.get("head_hidden", 0))

        # Normalisation stats
        norm_stats = params["norm_stats"]
        x_mean   = np.array(norm_stats["x_mean"], dtype=np.float32)
        x_std    = np.array(norm_stats["x_std"],  dtype=np.float32)
        iid_mean = float(norm_stats["y_mean"][0])
        iid_std  = float(norm_stats["y_std"][0])

        dist_norm = params.get("dist_norm", {})
        dist_mean = float(dist_norm.get("dist_mean", 0.0))
        dist_std  = float(dist_norm.get("dist_std",  1.0))

        # max_dist_mm used for input normalisation (profiles divided by this before z-scoring)
        max_dist_mm = float(
            params.get("no_echo_min_distance_mm",
                       params.get("config", {}).get("max_dist_mm", 3000.0))
        )

        # Build and load model
        model = EmulatorCNN(
            profile_steps=profile_steps,
            conv_channels=conv_channels,
            conv_kernel=conv_kernel,
            fc_hidden=fc_hidden,
            head_hidden=head_hidden,
        )
        target_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            os.path.join(emulator_dir, "best_model_pytorch.pth"), map_location="cpu"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(target_device)

        return Emulator(
            model=model,
            x_mean=x_mean,
            x_std=x_std,
            iid_mean=iid_mean,
            iid_std=iid_std,
            dist_mean=dist_mean,
            dist_std=dist_std,
            max_dist_mm=max_dist_mm,
            profile_opening_angle=profile_opening_angle,
            profile_steps=profile_steps,
            device=target_device,
        )

    def _sanitize_profiles(self, profiles: np.ndarray) -> np.ndarray:
        """
        Replace non-finite profile values with conservative finite values.

        For partially valid rows, fill missing bins with the row-wise max distance
        (unknown bins become "far away"). For fully invalid rows, fall back to
        max_dist_mm so the network sees a plausible open-space profile.
        """
        p = np.asarray(profiles, dtype=np.float32).copy()
        finite = np.isfinite(p)
        if finite.all():
            return p

        for i in range(p.shape[0]):
            row_finite = finite[i]
            if np.any(row_finite):
                p[i, ~row_finite] = float(np.max(p[i, row_finite]))
            else:
                p[i, :] = self.max_dist_mm
        return p

    def _normalize_input(self, profiles: np.ndarray) -> torch.Tensor:
        """
        Normalise profiles to match the training pipeline:
          1. Divide by max_dist_mm  (fixed global scale)
          2. Z-score with per-bin training mean and std
        """
        x = profiles / self.max_dist_mm
        x_tensor = torch.as_tensor(x, dtype=torch.float32).to(self.device)
        x_mean = torch.as_tensor(self.x_mean, dtype=torch.float32).to(self.device)
        x_std  = torch.as_tensor(self.x_std,  dtype=torch.float32).to(self.device)
        return (x_tensor - x_mean) / x_std

    def _forward(self, profiles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single forward pass.

        Returns (iid_norm [N,], dist_norm [N,]) — both in normalised (z-scored) units.
        """
        x = self._sanitize_profiles(profiles)
        with torch.no_grad():
            out = self.model(self._normalize_input(x))
            iid_norm  = out["iid"].cpu().numpy().squeeze(1)       # (N,)
            dist_norm = out["distance"].cpu().numpy().squeeze(1)  # (N,)
        return iid_norm, dist_norm

    def predict(self, profiles: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict IID (dB) and distance (mm) from an array of profiles.

        Symmetrised inference: each profile is passed through the network twice —
        once as-is and once horizontally flipped — and the outputs are combined to
        enforce the physical (anti)symmetry constraints regardless of any residual
        asymmetry in the trained weights:

            iid_db      = (iid_normal  − iid_flipped)  / 2   [antisymmetric under flip]
            distance_mm = (dist_normal + dist_flipped) / 2   [symmetric under flip]

        Args:
            profiles: (n_samples, profile_steps) array of wall distances in mm.

        Returns:
            dict with keys:
              'iid_db':      predicted IID in dB       (n_samples,)
              'distance_mm': predicted distance in mm  (n_samples,)
        """
        p = self._sanitize_profiles(np.asarray(profiles, dtype=np.float32))

        iid_n, dist_n = self._forward(p)
        iid_f, dist_f = self._forward(p[:, ::-1].copy())

        iid_db      = (iid_n - iid_f) / 2.0 * self.iid_std  + self.iid_mean
        distance_mm = (dist_n + dist_f) / 2.0 * self.dist_std + self.dist_mean

        return {"iid_db": iid_db, "distance_mm": distance_mm}

    def predict_single(self, profile: np.ndarray) -> Dict[str, float]:
        """
        Predict IID and distance for a single profile.

        Args:
            profile: (profile_steps,) array of wall distances in mm.

        Returns:
            dict with keys 'iid_db' and 'distance_mm'.
        """
        result = self.predict(profile[np.newaxis, :])
        return {
            "iid_db":      float(result["iid_db"][0]),
            "distance_mm": float(result["distance_mm"][0]),
        }

    def get_profile_params(self) -> Dict[str, float]:
        """Return profile parameters needed by EnvironmentSimulator."""
        return {
            "profile_opening_angle": self.profile_opening_angle,
            "profile_steps":         self.profile_steps,
        }
