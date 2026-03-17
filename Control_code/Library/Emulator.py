import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class EmulatorCNN(nn.Module):
    """
    1D CNN for the emulator that predicts IID and echo presence from raw profiles.
    Distance is not regressed here — it is computed geometrically by the caller.
    """
    def __init__(
        self,
        profile_steps: int,
        conv_channels: List[int] = [16, 32, 32],
        conv_kernel: int = 7,
        fc_hidden: int = 64,
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
        self.echo_present_head = nn.Linear(fc_hidden, 1)
        self.iid_head          = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.conv(x.unsqueeze(1))   # (batch, C, L)
        z = self.pool(z).flatten(1)
        z = self.fc(z)
        return {"echo_logit": self.echo_present_head(z), "iid": self.iid_head(z)}


class Emulator:
    """
    Environment emulator that predicts IID and echo presence probability from profiles.

    Distance is computed geometrically (minimum over central 90° of the profile)
    by EnvironmentSimulator.get_sonar_measurement() and is not part of this model.

    The emulator reads profile parameters (opening_angle, steps) from its own
    training artifact.
    """
    
    def __init__(
        self,
        model: nn.Module,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        normalize_x: bool,
        normalize_y: bool,
        calibration: Optional[List[Dict[str, float]]],
        profile_opening_angle: float,
        profile_steps: int,
        device: Optional[str] = None,
        no_echo_min_distance_mm: float = 3500.0,
    ):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()

        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_std = np.asarray(x_std, dtype=np.float32)
        self.y_mean = np.asarray(y_mean, dtype=np.float32)
        self.y_std = np.asarray(y_std, dtype=np.float32)
        self.normalize_x = bool(normalize_x)
        self.normalize_y = bool(normalize_y)
        self.calibration = calibration

        # Profile parameters used for profile generation/validation in simulation.
        self.profile_opening_angle = float(profile_opening_angle)
        self.profile_steps = int(profile_steps)
        self.no_echo_min_distance_mm = float(no_echo_min_distance_mm)

    @staticmethod
    def load(
        emulator_dir: str = "Emulator",
        device: Optional[str] = None
    ) -> 'Emulator':
        """
        Load a trained emulator from disk.

        Args:
            emulator_dir: Directory containing emulator artifacts
            device: Device to load model onto (None for auto-detection)

        Returns:
            Loaded Emulator instance
        """
        # Load emulator artifacts
        emulator_artifact_path = os.path.join(emulator_dir, "training_params.json")
        if not os.path.exists(emulator_artifact_path):
            raise FileNotFoundError(
                f"Emulator training params not found at {emulator_artifact_path}. "
                "Please run SCRIPT_TrainEmulator.py first."
            )
        
        # Load training params JSON
        import json
        with open(emulator_artifact_path, 'r') as f:
            params = json.load(f)

        profile_opening_angle = params.get("profile_opening_angle", None)
        profile_steps = params.get("profile_steps", None)

        if profile_opening_angle is None or profile_steps is None:
            raise ValueError(
                "Emulator training_params.json is missing profile_opening_angle/profile_steps. "
                "Please retrain by running SCRIPT_TrainEmulator.py."
            )
        
        # Load model
        model_path = os.path.join(emulator_dir, "best_model_pytorch.pth")
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Determine target device
        target_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reconstruct model architecture
        conv_channels = list(params["conv_channels"])
        conv_kernel   = int(params["conv_kernel"])
        fc_hidden     = int(params["fc_hidden"])

        model = EmulatorCNN(
            profile_steps=profile_steps,
            conv_channels=conv_channels,
            conv_kernel=conv_kernel,
            fc_hidden=fc_hidden,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load normalization stats
        norm_stats = params["norm_stats"]
        x_mean = np.array(norm_stats["x_mean"], dtype=np.float32)
        x_std = np.array(norm_stats["x_std"], dtype=np.float32)
        y_mean = np.array(norm_stats["y_mean"], dtype=np.float32)
        y_std = np.array(norm_stats["y_std"], dtype=np.float32)
        
        normalize_x = bool(params.get("normalize_x", True))
        normalize_y = bool(params.get("normalize_y", True))
        calibration = params.get("calibration", None)
        no_echo_min_distance_mm = float(params.get("no_echo_min_distance_mm", 3500.0))

        model = model.to(target_device)

        return Emulator(
            model=model,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            normalize_x=normalize_x,
            normalize_y=normalize_y,
            calibration=calibration,
            profile_opening_angle=profile_opening_angle,
            profile_steps=profile_steps,
            device=target_device,
            no_echo_min_distance_mm=no_echo_min_distance_mm,
        )

    def _sanitize_profiles(self, profiles: np.ndarray) -> np.ndarray:
        """
        Replace non-finite profile values with conservative finite values.

        For partially valid rows, fill missing bins with the row-wise max distance
        (unknown bins become "far"). For fully invalid rows, use a fallback derived
        from training-time profile statistics when available.
        """
        p = np.asarray(profiles, dtype=np.float32).copy()
        finite = np.isfinite(p)
        if finite.all():
            return p

        default_fill = 3000.0
        if self.x_mean.size >= self.profile_steps:
            base = self.x_mean[:self.profile_steps]
            if np.isfinite(base).any():
                default_fill = float(np.nanmean(base[np.isfinite(base)]))

        for i in range(p.shape[0]):
            row_finite = finite[i]
            if np.any(row_finite):
                row_fill = float(np.max(p[i, row_finite]))
            else:
                row_fill = default_fill
            p[i, ~row_finite] = row_fill
        return p

    def _normalize_input(self, x: np.ndarray) -> torch.Tensor:
        """Normalize input features."""
        # Convert to tensor and move to device immediately
        x_tensor = torch.as_tensor(x, dtype=torch.float32).to(self.device)
        if self.normalize_x:
            x_mean = torch.as_tensor(self.x_mean, dtype=torch.float32).to(self.device)
            x_std = torch.as_tensor(self.x_std, dtype=torch.float32).to(self.device)
            x_tensor = (x_tensor - x_mean) / x_std
        return x_tensor

    def _denormalize_output(self, y: torch.Tensor) -> np.ndarray:
        """Denormalize model outputs."""
        if self.normalize_y:
            y_mean = torch.as_tensor(self.y_mean, dtype=torch.float32).to(self.device)
            y_std = torch.as_tensor(self.y_std, dtype=torch.float32).to(self.device)
            y = y * y_std + y_mean
        # Ensure y is on CPU before converting to numpy
        return y.cpu().numpy()

    def _apply_calibration(self, y: np.ndarray) -> np.ndarray:
        """Apply output calibration if available."""
        if self.calibration is None:
            return y
        
        y_calibrated = y.copy()
        for i, cal in enumerate(self.calibration):
            if cal is not None:
                slope = float(cal["slope"])
                intercept = float(cal["intercept"])
                y_calibrated[:, i] = slope * y_calibrated[:, i] + intercept
        return y_calibrated

    def _forward(self, profiles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (iid_output [N,1], echo_prob [N])"""
        x = self._sanitize_profiles(profiles)
        # Normalise by per-profile mean (matches training preprocessing).
        x = x / np.clip(np.mean(x, axis=1, keepdims=True), 1e-6, None)
        with torch.no_grad():
            out = self.model(self._normalize_input(x))
            iid = self._apply_calibration(self._denormalize_output(out["iid"]))
            echo_prob = torch.sigmoid(out["echo_logit"]).cpu().numpy().squeeze(1)
        return iid, echo_prob

    def predict(self, profiles: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict echo_present and IID from profile data.

        Symmetrized inference: the profile is passed through the model twice —
        once as-is and once horizontally flipped.  The two predictions are
        combined so that the result is guaranteed to be antisymmetric in IID
        (flipping the profile negates the predicted IID) regardless of any
        residual asymmetry the network may have learned.

        Distance is NOT predicted here — it is computed geometrically by
        EnvironmentSimulator.get_sonar_measurement().

        Args:
            profiles: Array of shape (n_samples, profile_steps) containing distance profiles

        Returns:
            Dictionary with keys:
            - 'echo_present_prob': Predicted echo presence probability (n_samples,)
            - 'iid_db': Predicted IID in decibels (n_samples,)
        """
        p = self._sanitize_profiles(np.asarray(profiles, dtype=np.float32))

        iid_orig, ep_orig = self._forward(p)
        iid_flip, ep_flip = self._forward(p[:, ::-1].copy())

        # echo_present is symmetric under flip
        echo_present_prob = (ep_orig + ep_flip) / 2.0
        # iid is antisymmetric under flip
        iid_db = (iid_orig[:, 0] - iid_flip[:, 0]) / 2.0

        return {
            'echo_present_prob': echo_present_prob,
            'iid_db':            iid_db,
        }

    def predict_single(self, profile: np.ndarray) -> Dict[str, float]:
        """
        Predict echo_present and IID for a single profile.

        Args:
            profile: Single profile array of shape (profile_steps,)

        Returns:
            Dictionary with keys:
            - 'echo_present_prob': Predicted echo presence probability
            - 'iid_db': Predicted IID in decibels
        """
        result = self.predict(profile[np.newaxis, :])
        return {
            'echo_present_prob': float(result['echo_present_prob'][0]),
            'iid_db':            float(result['iid_db'][0]),
        }

    def get_profile_params(self) -> Dict[str, float]:
        """Get the profile parameters used by this emulator."""
        return {
            'profile_opening_angle': self.profile_opening_angle,
            'profile_steps': self.profile_steps
        }
