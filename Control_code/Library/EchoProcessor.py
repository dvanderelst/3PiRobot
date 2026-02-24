import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class EchoProcessorDistanceCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        conv_filters=None,
        kernel_size=5,
        pool_size=2,
        dropout_rate=0.30,
    ):
        super().__init__()
        if conv_filters is None:
            conv_filters = [32, 64, 128]
        self.shared = nn.Sequential(
            nn.Conv1d(2, conv_filters[0], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout_rate),
            nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout_rate),
            nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout_rate),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.shared(dummy)
            flat = out.view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        z = self.shared(x)
        return self.head(z).squeeze(-1)


class EchoProcessor:
    """
    Inference wrapper for the trained EchoProcessor pipeline:
    sonar -> distance (model) -> IID (window around mapped distance sample).
    """

    def __init__(
        self,
        model: nn.Module,
        sonar_mean: np.ndarray,
        sonar_std: np.ndarray,
        y_mean: float,
        y_std: float,
        normalize_sonar: bool,
        normalize_target: bool,
        calibration: Optional[Dict[str, float]],
        iid_window: Dict[str, int],
        iid_eps: float = 1e-9,
        profile_opening_angle: Optional[float] = None,
        profile_steps: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()

        self.sonar_mean = np.asarray(sonar_mean, dtype=np.float32)
        self.sonar_std = np.asarray(sonar_std, dtype=np.float32)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self.normalize_sonar = bool(normalize_sonar)
        self.normalize_target = bool(normalize_target)
        self.calibration = calibration
        self.iid_window = {
            "pre_samples": int(iid_window["pre_samples"]),
            "post_samples": int(iid_window["post_samples"]),
            "center_offset_samples": int(iid_window["center_offset_samples"]),
        }
        self.iid_eps = float(iid_eps)
        self.profile_opening_angle = None if profile_opening_angle is None else float(profile_opening_angle)
        self.profile_steps = None if profile_steps is None else int(profile_steps)

    @staticmethod
    def _ensure_batch(sonar_lr: np.ndarray) -> np.ndarray:
        arr = np.asarray(sonar_lr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3 or arr.shape[-1] != 2:
            raise ValueError("Expected sonar shape (samples,2) or (N,samples,2).")
        return arr

    def predict_distance(self, sonar_lr: np.ndarray) -> np.ndarray:
        arr = self._ensure_batch(sonar_lr)
        x = np.transpose(arr, (0, 2, 1))  # (N,samples,2) -> (N,2,samples)
        x_t = torch.from_numpy(x).to(self.device)
        if self.normalize_sonar:
            m = torch.from_numpy(self.sonar_mean).to(self.device).view(1, -1, 1)
            s = torch.from_numpy(self.sonar_std).to(self.device).view(1, -1, 1)
            x_t = (x_t - m) / s

        with torch.no_grad():
            y = self.model(x_t).detach().cpu().numpy().astype(np.float32)

        if self.normalize_target:
            y = y * self.y_std + self.y_mean
        if self.calibration is not None:
            y = float(self.calibration["slope"]) * y + float(self.calibration["intercept"])
        return y

    @staticmethod
    def _iid_from_distance_window(
        sonar_lr: np.ndarray,
        dist_axis_mm: np.ndarray,
        target_dist_mm: float,
        pre_samples: int,
        post_samples: int,
        center_offset_samples: int,
        eps: float,
    ) -> float:
        axis = np.asarray(dist_axis_mm, dtype=np.float32)
        if axis.ndim != 1:
            raise ValueError("Distance axis must be 1D per sample.")
        if not (np.isfinite(target_dist_mm) and np.any(np.isfinite(axis))):
            return np.nan

        idx = int(np.argmin(np.abs(axis - target_dist_mm)))
        idx = int(np.clip(idx + int(center_offset_samples), 0, len(axis) - 1))
        lo = max(0, idx - int(pre_samples))
        hi = min(len(axis), idx + int(post_samples) + 1)
        if hi <= lo:
            return np.nan

        w = np.asarray(sonar_lr, dtype=np.float32)[lo:hi, :]
        left = w[:, 0]
        right = w[:, 1]
        e_left = float(np.sum(left * left))
        e_right = float(np.sum(right * right))
        return float(10.0 * np.log10((e_left + eps) / (e_right + eps)))

    def compute_iid(
        self,
        sonar_lr: np.ndarray,
        distance_axis_mm: np.ndarray,
        distance_mm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        arr = self._ensure_batch(sonar_lr)
        axis = np.asarray(distance_axis_mm, dtype=np.float32)
        if axis.ndim == 1:
            axis = np.repeat(axis[None, :], arr.shape[0], axis=0)
        if axis.ndim != 2 or axis.shape[0] != arr.shape[0]:
            raise ValueError("Distance axis shape must be (samples,) or (N,samples).")

        if distance_mm is None:
            dist = self.predict_distance(arr)
        else:
            dist = np.asarray(distance_mm, dtype=np.float32)
            if dist.ndim == 0:
                dist = np.full((arr.shape[0],), float(dist), dtype=np.float32)
            if dist.ndim != 1 or dist.shape[0] != arr.shape[0]:
                raise ValueError("distance_mm must be scalar or shape (N,).")

        out = np.array(
            [
                self._iid_from_distance_window(
                    arr[i],
                    axis[i],
                    float(dist[i]),
                    pre_samples=self.iid_window["pre_samples"],
                    post_samples=self.iid_window["post_samples"],
                    center_offset_samples=self.iid_window["center_offset_samples"],
                    eps=self.iid_eps,
                )
                for i in range(arr.shape[0])
            ],
            dtype=np.float32,
        )
        return out

    def predict(self, sonar_lr: np.ndarray, distance_axis_mm: np.ndarray) -> Dict[str, np.ndarray]:
        distance_mm = self.predict_distance(sonar_lr)
        iid_db = self.compute_iid(sonar_lr, distance_axis_mm, distance_mm=distance_mm)
        return {"distance_mm": distance_mm, "iid_db": iid_db}

    @classmethod
    def load(cls, artifact_dir: str, artifact_name: str = "echoprocessor_artifacts.pth", device: Optional[str] = None):
        path = os.path.join(artifact_dir, artifact_name)
        payload = torch.load(path, map_location=(device if device is not None else "cpu"))

        model_cfg = payload["model_config"]
        model = EchoProcessorDistanceCNN(
            input_shape=(2, int(model_cfg["num_sonar_samples"])),
            conv_filters=list(model_cfg["conv_filters"]),
            kernel_size=int(model_cfg["kernel_size"]),
            pool_size=int(model_cfg["pool_size"]),
            dropout_rate=float(model_cfg["dropout_rate"]),
        )
        model.load_state_dict(payload["model_state_dict"])

        norm = payload["norm_stats"]
        return cls(
            model=model,
            sonar_mean=np.asarray(norm["sonar_mean"], dtype=np.float32),
            sonar_std=np.asarray(norm["sonar_std"], dtype=np.float32),
            y_mean=float(norm["y_mean"]),
            y_std=float(norm["y_std"]),
            normalize_sonar=bool(payload["normalize_sonar"]),
            normalize_target=bool(payload["normalize_target"]),
            calibration=payload.get("calibration", None),
            iid_window=payload["iid_window"],
            iid_eps=float(payload.get("iid_eps", 1e-9)),
            profile_opening_angle=payload.get("profile_opening_angle", None),
            profile_steps=payload.get("profile_steps", None),
            device=device,
        )


def save_artifacts(
    artifact_dir: str,
    model_state_dict: Dict[str, Any],
    model_config: Dict[str, Any],
    norm_stats: Dict[str, Any],
    normalize_sonar: bool,
    normalize_target: bool,
    calibration: Optional[Dict[str, float]],
    iid_window: Dict[str, int],
    iid_eps: float,
    profile_opening_angle: Optional[float] = None,
    profile_steps: Optional[int] = None,
    artifact_name: str = "echoprocessor_artifacts.pth",
) -> str:
    payload = {
        "model_state_dict": model_state_dict,
        "model_config": model_config,
        "norm_stats": {
            "sonar_mean": np.asarray(norm_stats["sonar_mean"], dtype=np.float32).tolist(),
            "sonar_std": np.asarray(norm_stats["sonar_std"], dtype=np.float32).tolist(),
            "y_mean": float(norm_stats["y_mean"]),
            "y_std": float(norm_stats["y_std"]),
        },
        "normalize_sonar": bool(normalize_sonar),
        "normalize_target": bool(normalize_target),
        "calibration": calibration,
        "iid_window": {
            "pre_samples": int(iid_window["pre_samples"]),
            "post_samples": int(iid_window["post_samples"]),
            "center_offset_samples": int(iid_window["center_offset_samples"]),
        },
        "iid_eps": float(iid_eps),
        "profile_opening_angle": None if profile_opening_angle is None else float(profile_opening_angle),
        "profile_steps": None if profile_steps is None else int(profile_steps),
    }
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, artifact_name)
    torch.save(payload, path)
    return path
