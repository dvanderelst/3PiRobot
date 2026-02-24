import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class EmulatorMLP(nn.Module):
    """
    MLP model for the emulator that predicts distance and IID from profiles.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_sizes: List[int] = [128, 128, 64],
        head_hidden_size: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim = h
        self.trunk = nn.Sequential(*layers)
        self.distance_head = nn.Sequential(
            nn.Linear(dim, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )
        self.iid_head = nn.Sequential(
            nn.Linear(dim, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.trunk(x)
        d = self.distance_head(z)
        i = self.iid_head(z)
        return {"reg": torch.cat([d, i], dim=1)}


class Emulator:
    """
    Environment emulator that predicts sonar measurements (distance and IID) from profiles.
    
    This class loads a trained emulator model and provides a clean interface for
    simulating what sonar measurements would be received from different positions
    in an environment.
    
    The emulator reads profile parameters (opening_angle, steps) from the EchoProcessor
    artifacts to ensure consistency across the pipeline.
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
        
        # Profile parameters - read from EchoProcessor to ensure consistency
        self.profile_opening_angle = float(profile_opening_angle)
        self.profile_steps = int(profile_steps)

    @staticmethod
    def load(
        emulator_dir: str = "Emulator",
        echo_processor_dir: str = "EchoProcessor",
        device: Optional[str] = None
    ) -> 'Emulator':
        """
        Load a trained emulator from disk.
        
        Args:
            emulator_dir: Directory containing emulator artifacts
            echo_processor_dir: Directory containing EchoProcessor artifacts
            device: Device to load model onto (None for auto-detection)
            
        Returns:
            Loaded Emulator instance
        """
        # First load EchoProcessor artifacts to get profile parameters
        echo_artifact_path = os.path.join(echo_processor_dir, "echoprocessor_artifacts.pth")
        if not os.path.exists(echo_artifact_path):
            raise FileNotFoundError(
                f"EchoProcessor artifacts not found at {echo_artifact_path}. "
                "Please run SCRIPT_TrainEchoProcessor.py first."
            )
        
        echo_payload = torch.load(echo_artifact_path, map_location="cpu")
        profile_opening_angle = echo_payload["profile_opening_angle"]
        profile_steps = echo_payload["profile_steps"]
        
        if profile_opening_angle is None or profile_steps is None:
            raise ValueError(
                "EchoProcessor artifacts are missing profile_opening_angle/profile_steps. "
                "Please retrain SCRIPT_TrainEchoProcessor.py."
            )
        
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
        
        # Load model
        model_path = os.path.join(emulator_dir, "best_model_pytorch.pth")
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Determine target device
        target_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reconstruct model architecture
        input_feature_dim = int(params["input_feature_dim"])
        hidden_sizes = list(params["hidden_sizes"])
        head_hidden_size = int(params["head_hidden_size"])
        dropout = float(params["dropout"])
        
        model = EmulatorMLP(
            in_dim=input_feature_dim,
            hidden_sizes=hidden_sizes,
            head_hidden_size=head_hidden_size,
            dropout=dropout,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load normalization stats
        norm_stats = params["norm_stats"]
        x_mean = np.array(norm_stats["x_mean"], dtype=np.float32)
        x_std = np.array(norm_stats["x_std"], dtype=np.float32)
        y_mean = np.array(norm_stats["y_mean"], dtype=np.float32)
        y_std = np.array(norm_stats["y_std"], dtype=np.float32)
        
        normalize_x = bool(params["normalize_x"])
        normalize_y = bool(params["normalize_y"])
        calibration = params.get("calibration", None)
        
        # Move model to the target device
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
        )

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

    def build_profile_features(self, profiles: np.ndarray) -> np.ndarray:
        """
        Feature augmentation from profiles (matches training preprocessing).
        
        Args:
            profiles: Array of shape (n_samples, profile_steps) containing distance profiles
            
        Returns:
            Augmented features of shape (n_samples, input_feature_dim)
        """
        p = np.asarray(profiles, dtype=np.float32)
        n, steps = p.shape
        
        # Validate profile dimensions
        if steps != self.profile_steps:
            raise ValueError(f"Expected profiles with {self.profile_steps} steps, got {steps}")
        
        half = steps // 2
        idx = np.arange(steps, dtype=np.float32)
        idx_grid = np.broadcast_to(idx[None, :], p.shape)

        min_val = np.min(p, axis=1)
        argmin = np.argmin(p, axis=1).astype(np.float32)
        argmin_norm = argmin / max(steps - 1, 1)
        left_min = np.min(p[:, :half], axis=1)
        right_min = np.min(p[:, half:], axis=1)
        asym = left_min - right_min

        # local slope around minimum bin (simple finite difference)
        argmin_i = argmin.astype(np.int64)
        left_i = np.clip(argmin_i - 1, 0, steps - 1)
        right_i = np.clip(argmin_i + 1, 0, steps - 1)
        local_slope = p[np.arange(n), right_i] - p[np.arange(n), left_i]

        # weighted center-of-mass with inverse distance weights
        w = 1.0 / np.clip(p, 1e-3, None)
        wsum = np.sum(w, axis=1)
        com = np.sum(w * idx_grid, axis=1) / np.clip(wsum, 1e-6, None)
        com_norm = com / max(steps - 1, 1)

        extras = np.stack([min_val, argmin_norm, asym, local_slope, com_norm], axis=1).astype(np.float32)
        return np.concatenate([p, extras], axis=1).astype(np.float32)

    def predict(self, profiles: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict distance and IID from profile data.
        
        Args:
            profiles: Array of shape (n_samples, profile_steps) containing distance profiles
            
        Returns:
            Dictionary with keys:
            - 'distance_mm': Predicted distances in millimeters (n_samples,)
            - 'iid_db': Predicted IID in decibels (n_samples,)
        """
        # Build features (matches training preprocessing)
        x_features = self.build_profile_features(profiles)
        
        # Normalize and convert to tensor
        x_tensor = self._normalize_input(x_features)
        
        # Predict
        with torch.no_grad():
            pred_out = self.model(x_tensor)
            y_pred = pred_out["reg"]
        
        # Denormalize
        y_pred = self._denormalize_output(y_pred)
        
        # Apply calibration
        y_pred = self._apply_calibration(y_pred)
        
        # Split into distance and IID
        distance_mm = y_pred[:, 0]
        iid_db = y_pred[:, 1]
        
        return {
            'distance_mm': distance_mm,
            'iid_db': iid_db
        }

    def predict_single(self, profile: np.ndarray) -> Dict[str, float]:
        """
        Predict distance and IID for a single profile.
        
        Args:
            profile: Single profile array of shape (profile_steps,)
            
        Returns:
            Dictionary with keys:
            - 'distance_mm': Predicted distance in millimeters
            - 'iid_db': Predicted IID in decibels
        """
        result = self.predict(profile[np.newaxis, :])
        return {
            'distance_mm': float(result['distance_mm'][0]),
            'iid_db': float(result['iid_db'][0])
        }

    def get_profile_params(self) -> Dict[str, Union[float, int]]:
        """Get the profile parameters used by this emulator."""
        return {
            'profile_opening_angle': self.profile_opening_angle,
            'profile_steps': self.profile_steps
        }