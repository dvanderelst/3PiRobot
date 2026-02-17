import json
from collections import deque
from dataclasses import dataclass
import os
import sys

import joblib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from Library import DataProcessor


@dataclass
class SteeringConfig:
    window_size: int = 5
    presence_threshold_override: float | None = None

    # Robot-frame walliness grid/integration.
    extent_mm: float = 2500.0
    grid_mm: float = 20.0
    sigma_perp_mm: float = 40.0
    sigma_para_mm: float = 120.0
    apply_heatmap_smoothing: bool = False

    # Circle planner.
    occ_block_threshold: float = 0.10
    robot_radius_mm: float = 80.0
    safety_margin_mm: float = 120.0
    circle_radius_min_mm: float = 250.0
    circle_radius_max_mm: float = 2500.0
    circle_radius_step_mm: float = 50.0
    circle_arc_samples: int = 220
    circle_horizon_x_mm: float = 1800.0
    circle_radius_tie_mm: float = 100.0

    # Optional per-call debug plotting.
    debug_plot_dir: str | None = None
    debug_plot_dpi: int = 180


class TwoHeadedSonarCNN(nn.Module):
    def __init__(self, input_shape, profile_steps, conv_filters, kernel_size, pool_size, dropout_rate):
        super(TwoHeadedSonarCNN, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=conv_filters[0], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate),

            nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate),

            nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size=kernel_size),
            nn.BatchNorm1d(conv_filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate),
        )
        self._calculate_flattened_size(input_shape)

        self.presence_head = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, profile_steps),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, profile_steps),
        )

    def _calculate_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.shared(dummy_input)
            self.flattened_size = x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        features = self.shared(x)
        features = features.view(features.size(0), -1)
        presence_logits = self.presence_head(features)
        distance_scaled = self.distance_head(features)
        return presence_logits, distance_scaled


class DistanceScaler:
    """Compatibility class for loading scaler pickled from training script (__main__.DistanceScaler)."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def transform(self, y):
        return (y - self.mean_) / self.scale_

    def inverse_transform(self, y):
        return y * self.scale_ + self.mean_


class SonarWallSteering:
    """
    End-to-end steering helper:
    sonar sequence + pose sequence -> signed curvature.

    Signed curvature convention:
    - positive: turn left
    - negative: turn right
    - zero: go straight
    Units: 1/mm
    """

    def __init__(self, training_output_dir='Training', config: SteeringConfig | None = None, device: torch.device | None = None):
        self.config = config if config is not None else SteeringConfig()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, self.y_scaler, self.params = self._load_training_artifacts(training_output_dir)
        self.profile_steps = int(self.params['profile_steps'])
        self.profile_opening_angle = float(self.params['profile_opening_angle'])
        self.presence_threshold = (
            float(self.params.get('presence_threshold', 0.5))
            if self.config.presence_threshold_override is None
            else float(self.config.presence_threshold_override)
        )

        self.profile_centers_deg = self._build_profile_centers(self.profile_opening_angle, self.profile_steps)
        self._init_robot_grid()

        self._sonar_buffer = deque(maxlen=int(self.config.window_size))
        self._pose_buffer = deque(maxlen=int(self.config.window_size))
        self._plot_counter = 0

    def reset(self):
        self._sonar_buffer.clear()
        self._pose_buffer.clear()

    def update(
        self,
        sonar_frame,
        rob_x,
        rob_y,
        rob_yaw_deg,
        return_debug=False,
        plot=False,
        plot_path=None,
        plot_title=None,
    ):
        self._sonar_buffer.append(np.asarray(sonar_frame, dtype=np.float32))
        self._pose_buffer.append(np.asarray([rob_x, rob_y, rob_yaw_deg], dtype=np.float32))

        sonar_seq = np.stack(list(self._sonar_buffer), axis=0)
        pose_seq = np.stack(list(self._pose_buffer), axis=0)
        return self.compute_curvature(
            sonar_seq=sonar_seq,
            rob_x_seq=pose_seq[:, 0],
            rob_y_seq=pose_seq[:, 1],
            rob_yaw_deg_seq=pose_seq[:, 2],
            return_debug=return_debug,
            plot=plot,
            plot_path=plot_path,
            plot_title=plot_title,
        )

    def compute_curvature(
        self,
        sonar_seq,
        rob_x_seq,
        rob_y_seq,
        rob_yaw_deg_seq,
        return_debug=False,
        plot=False,
        plot_path=None,
        plot_title=None,
    ):
        sonar_seq = np.asarray(sonar_seq, dtype=np.float32)
        rob_x_seq = np.asarray(rob_x_seq, dtype=np.float32)
        rob_y_seq = np.asarray(rob_y_seq, dtype=np.float32)
        rob_yaw_deg_seq = np.asarray(rob_yaw_deg_seq, dtype=np.float32)

        self._validate_inputs(sonar_seq, rob_x_seq, rob_y_seq, rob_yaw_deg_seq)

        # Keep most recent window only.
        max_n = int(self.config.window_size)
        if sonar_seq.shape[0] > max_n:
            sonar_seq = sonar_seq[-max_n:]
            rob_x_seq = rob_x_seq[-max_n:]
            rob_y_seq = rob_y_seq[-max_n:]
            rob_yaw_deg_seq = rob_yaw_deg_seq[-max_n:]

        presence_probs, presence_bin, distance_mm = self._predict_profiles(sonar_seq)
        evidence_norm = self._build_robot_frame_evidence(
            distance_mm=distance_mm,
            presence_probs=presence_probs,
            presence_bin=presence_bin,
            rob_x_seq=rob_x_seq,
            rob_y_seq=rob_y_seq,
            rob_yaw_deg_seq=rob_yaw_deg_seq,
        )
        planner = self._plan_circles(evidence_norm)
        signed_curvature = self._planner_to_curvature(planner)
        debug = {
            'signed_curvature_inv_mm': float(signed_curvature),
            'chosen_side': planner['chosen_side'],
            'chosen_radius_mm': float(planner['chosen_radius_mm']),
            'left_radius_mm': planner['left_radius_mm'],
            'right_radius_mm': planner['right_radius_mm'],
            'left_evidence_score': planner['left_evidence_score'],
            'right_evidence_score': planner['right_evidence_score'],
            'left_x': planner['left_x'],
            'left_y': planner['left_y'],
            'right_x': planner['right_x'],
            'right_y': planner['right_y'],
            'blocked_mask': planner['blocked_mask'],
            'presence_threshold': float(self.presence_threshold),
            'window_size_used': int(sonar_seq.shape[0]),
            'evidence_map': evidence_norm,
            'x_grid': self.x_grid,
            'y_grid': self.y_grid,
        }

        if plot:
            out_path = self._resolve_plot_path(plot_path)
            self.plot_debug(debug=debug, out_path=out_path, title=plot_title)

        if return_debug:
            return signed_curvature, debug
        return signed_curvature

    def _resolve_plot_path(self, plot_path):
        if plot_path is not None:
            return plot_path
        if self.config.debug_plot_dir is None:
            return None
        os.makedirs(self.config.debug_plot_dir, exist_ok=True)
        out = os.path.join(self.config.debug_plot_dir, f'steering_debug_{self._plot_counter:06d}.png')
        self._plot_counter += 1
        return out

    def plot_debug(self, debug, out_path=None, title=None):
        hm = debug['evidence_map']
        x_grid = debug['x_grid']
        y_grid = debug['y_grid']
        blocked = debug.get('blocked_mask', hm >= self.config.occ_block_threshold)

        left_x = debug.get('left_x', None)
        left_y = debug.get('left_y', None)
        right_x = debug.get('right_x', None)
        right_y = debug.get('right_y', None)

        chosen_side = str(debug.get('chosen_side', 'straight'))
        if chosen_side.startswith('left'):
            path_x, path_y = left_x, left_y
        elif chosen_side.startswith('right'):
            path_x, path_y = right_x, right_y
        else:
            path_x = np.linspace(0.0, self.config.circle_horizon_x_mm, self.config.circle_arc_samples, dtype=np.float32)
            path_y = np.zeros_like(path_x)

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        im = ax.imshow(
            hm,
            origin='lower',
            extent=[float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1])],
            cmap='magma',
            aspect='equal'
        )
        fig.colorbar(im, ax=ax, label='Wall Evidence (norm)')
        ax.scatter([0.0], [0.0], c='cyan', s=45, label='Robot (anchor)')
        ax.arrow(0.0, 0.0, 200.0, 0.0, head_width=70, head_length=90, fc='cyan', ec='cyan')

        xx, yy = np.meshgrid(x_grid, y_grid)
        ax.contourf(xx, yy, blocked.astype(float), levels=[0.5, 1.5], colors=['red'], alpha=0.16)

        if left_x is not None:
            ax.plot(left_x, left_y, '--', color='deepskyblue', lw=1.4, alpha=0.8, label='Left candidate')
        if right_x is not None:
            ax.plot(right_x, right_y, '--', color='springgreen', lw=1.4, alpha=0.8, label='Right candidate')
        if path_x is not None and len(path_x) > 1:
            ax.plot(path_x, path_y, '-', color='white', lw=2.0, label='Chosen circle arc')

        txt = (
            f"side={debug['chosen_side']}\n"
            f"R={debug['chosen_radius_mm']:.0f} mm\n"
            f"curv={debug['signed_curvature_inv_mm']:+.5f} 1/mm\n"
            f"E(L,R)=({debug['left_evidence_score']:.4f},{debug['right_evidence_score']:.4f})"
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=10, bbox=dict(facecolor='black', alpha=0.45, edgecolor='none'))

        ax.set_xlim(0.0, self.config.circle_horizon_x_mm + 200.0)
        ax.set_ylim(float(y_grid[0]), float(y_grid[-1]))
        ax.set_xlabel('Robot X (mm)')
        ax.set_ylabel('Robot Y (mm)')
        ax.set_title(title if title is not None else 'SonarWallSteering Debug')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right')
        plt.tight_layout()

        if out_path is None:
            plt.show(block=False)
            plt.pause(0.001)
            plt.close(fig)
        else:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(out_path, dpi=int(self.config.debug_plot_dpi), bbox_inches='tight')
            plt.close(fig)

    def _load_training_artifacts(self, base_dir):
        params_path = os.path.join(base_dir, 'training_params.json')
        model_path = os.path.join(base_dir, 'best_model_pytorch.pth')
        scaler_path = os.path.join(base_dir, 'y_scaler.joblib')
        missing = [p for p in [params_path, model_path, scaler_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing training artifacts: {missing}")

        with open(params_path, 'r') as f:
            params = json.load(f)

        model = TwoHeadedSonarCNN(
            input_shape=tuple(params.get('input_shape', [2, 200])),
            profile_steps=int(params['profile_steps']),
            conv_filters=params['conv_filters'],
            kernel_size=int(params['kernel_size']),
            pool_size=int(params['pool_size']),
            dropout_rate=float(params['dropout_rate']),
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        # Backward-compatible load: training script serialized scaler as __main__.DistanceScaler.
        main_mod = sys.modules.get('__main__')
        had_distance_scaler = hasattr(main_mod, 'DistanceScaler') if main_mod is not None else False
        old_distance_scaler = getattr(main_mod, 'DistanceScaler', None) if had_distance_scaler else None
        try:
            if main_mod is not None:
                setattr(main_mod, 'DistanceScaler', DistanceScaler)
            y_scaler = joblib.load(scaler_path)
        finally:
            if main_mod is not None:
                if had_distance_scaler:
                    setattr(main_mod, 'DistanceScaler', old_distance_scaler)
                else:
                    try:
                        delattr(main_mod, 'DistanceScaler')
                    except AttributeError:
                        pass
        return model, y_scaler, params

    @staticmethod
    def _build_profile_centers(opening_angle_deg, steps):
        edges = np.linspace(-0.5 * opening_angle_deg, 0.5 * opening_angle_deg, int(steps) + 1, dtype=np.float32)
        return 0.5 * (edges[:-1] + edges[1:])

    def _init_robot_grid(self):
        x_min, x_max = -float(self.config.extent_mm), float(self.config.extent_mm)
        y_min, y_max = -float(self.config.extent_mm), float(self.config.extent_mm)
        self.x_grid = np.arange(x_min, x_max + self.config.grid_mm, self.config.grid_mm, dtype=np.float32)
        self.y_grid = np.arange(y_min, y_max + self.config.grid_mm, self.config.grid_mm, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(self.x_grid, self.y_grid)
        self.grid_x_min = float(x_min)
        self.grid_y_min = float(y_min)

    def _validate_inputs(self, sonar_seq, rob_x_seq, rob_y_seq, rob_yaw_deg_seq):
        if sonar_seq.ndim != 3 or sonar_seq.shape[1:] != (200, 2):
            raise ValueError(f"sonar_seq must have shape (N, 200, 2), got {sonar_seq.shape}")
        n = sonar_seq.shape[0]
        if n < 1:
            raise ValueError("sonar_seq must contain at least one sample")
        if not (len(rob_x_seq) == len(rob_y_seq) == len(rob_yaw_deg_seq) == n):
            raise ValueError("pose sequences must all have same length as sonar_seq")
        if not np.isfinite(sonar_seq).all():
            raise ValueError("sonar_seq contains NaN/inf")
        if not np.isfinite(rob_x_seq).all() or not np.isfinite(rob_y_seq).all() or not np.isfinite(rob_yaw_deg_seq).all():
            raise ValueError("pose sequences contain NaN/inf")

    def _predict_profiles(self, sonar_seq):
        x_tensor = torch.FloatTensor(sonar_seq).permute(0, 2, 1).to(self.device)  # (N,2,200)
        with torch.no_grad():
            presence_logits, distance_scaled = self.model(x_tensor)
            presence_probs = torch.sigmoid(presence_logits).cpu().numpy()
            distance_scaled = distance_scaled.cpu().numpy()

        presence_bin = (presence_probs >= self.presence_threshold).astype(np.uint8)
        distance_mm = self.y_scaler.inverse_transform(distance_scaled)
        distance_mm = np.asarray(distance_mm, dtype=np.float32)
        distance_mm[presence_bin == 0] = np.nan

        distance_threshold = float(self.params.get('distance_threshold', np.inf))
        invalid = (distance_mm < 0.0) | (distance_mm > distance_threshold)
        distance_mm[invalid] = np.nan
        return presence_probs, presence_bin, distance_mm

    @staticmethod
    def _smooth_heatmap(hm):
        k = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        k /= np.sum(k)
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), axis=1, arr=hm)
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), axis=0, arr=out)
        return out

    def _build_robot_frame_evidence(self, distance_mm, presence_probs, presence_bin, rob_x_seq, rob_y_seq, rob_yaw_deg_seq):
        n = distance_mm.shape[0]
        anchor_x = float(rob_x_seq[-1])
        anchor_y = float(rob_y_seq[-1])
        anchor_yaw = float(rob_yaw_deg_seq[-1])
        evidence = np.zeros_like(self.xx, dtype=np.float32)

        for i in range(n):
            az_deg = self.profile_centers_deg
            dist_i = distance_mm[i].copy()
            dist_i[presence_bin[i] == 0] = np.nan
            probs_i = presence_probs[i]

            xw, yw = DataProcessor.robot2world(az_deg, dist_i, float(rob_x_seq[i]), float(rob_y_seq[i]), float(rob_yaw_deg_seq[i]))
            xr, yr = DataProcessor.world2robot(xw, yw, anchor_x, anchor_y, anchor_yaw)

            for j in range(len(az_deg) - 1):
                if not (np.isfinite(xr[j]) and np.isfinite(yr[j]) and np.isfinite(xr[j + 1]) and np.isfinite(yr[j + 1])):
                    continue
                p0 = np.array([xr[j], yr[j]], dtype=float)
                p1 = np.array([xr[j + 1], yr[j + 1]], dtype=float)
                seg_vec = p1 - p0
                seg_len = float(np.linalg.norm(seg_vec))
                if seg_len < 1e-6:
                    continue
                weight = float(0.5 * (probs_i[j] + probs_i[j + 1]))
                if weight <= 0.0:
                    continue
                self._accumulate_segment(evidence, p0, p1, seg_len, weight)

        evidence_norm = evidence / max(n, 1)
        if self.config.apply_heatmap_smoothing:
            evidence_norm = self._smooth_heatmap(evidence_norm)
        max_v = float(np.max(evidence_norm))
        if max_v > 1e-12:
            evidence_norm = evidence_norm / max_v
        return evidence_norm.astype(np.float32)

    def _accumulate_segment(self, evidence, p0, p1, seg_len, weight):
        tangent = (p1 - p0) / seg_len
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        mid = 0.5 * (p0 + p1)
        half_len = 0.5 * seg_len

        local_margin = 3.0 * max(self.config.sigma_perp_mm, self.config.sigma_para_mm)
        bx0 = max(float(self.x_grid[0]), min(p0[0], p1[0]) - local_margin)
        bx1 = min(float(self.x_grid[-1]), max(p0[0], p1[0]) + local_margin)
        by0 = max(float(self.y_grid[0]), min(p0[1], p1[1]) - local_margin)
        by1 = min(float(self.y_grid[-1]), max(p0[1], p1[1]) + local_margin)

        ix0 = int(max(0, np.floor((bx0 - self.grid_x_min) / self.config.grid_mm)))
        ix1 = int(min(len(self.x_grid) - 1, np.ceil((bx1 - self.grid_x_min) / self.config.grid_mm)))
        iy0 = int(max(0, np.floor((by0 - self.grid_y_min) / self.config.grid_mm)))
        iy1 = int(min(len(self.y_grid) - 1, np.ceil((by1 - self.grid_y_min) / self.config.grid_mm)))

        sub_x = self.xx[iy0:iy1 + 1, ix0:ix1 + 1]
        sub_y = self.yy[iy0:iy1 + 1, ix0:ix1 + 1]
        dx = sub_x - mid[0]
        dy = sub_y - mid[1]

        d_para = dx * tangent[0] + dy * tangent[1]
        d_perp = dx * normal[0] + dy * normal[1]
        outside_para = np.maximum(np.abs(d_para) - half_len, 0.0)

        kernel = np.exp(
            -0.5 * (d_perp / self.config.sigma_perp_mm) ** 2
            -0.5 * (outside_para / self.config.sigma_para_mm) ** 2
        )
        evidence[iy0:iy1 + 1, ix0:ix1 + 1] += (weight * kernel).astype(np.float32)

    @staticmethod
    def _dilate_binary(mask, radius_px):
        if radius_px <= 0:
            return mask.copy()
        h, w = mask.shape
        out = np.zeros_like(mask, dtype=bool)
        rr2 = radius_px * radius_px
        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                if dx * dx + dy * dy > rr2:
                    continue
                y_src0 = max(0, -dy)
                y_src1 = min(h, h - dy)
                x_src0 = max(0, -dx)
                x_src1 = min(w, w - dx)
                if y_src1 <= y_src0 or x_src1 <= x_src0:
                    continue
                y_dst0 = y_src0 + dy
                y_dst1 = y_src1 + dy
                x_dst0 = x_src0 + dx
                x_dst1 = x_src1 + dx
                out[y_dst0:y_dst1, x_dst0:x_dst1] |= mask[y_src0:y_src1, x_src0:x_src1]
        return out

    @staticmethod
    def _sample_tangent_circle(radius_mm, side_sign, horizon_x_mm, n_samples):
        if radius_mm <= 1e-6:
            return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
        phi_max = np.arcsin(np.clip(horizon_x_mm / radius_mm, 0.0, 1.0))
        if phi_max <= 1e-6:
            return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
        phi = np.linspace(0.0, phi_max, int(n_samples), dtype=np.float32)
        x = radius_mm * np.sin(phi)
        y = side_sign * radius_mm * (1.0 - np.cos(phi))
        return x.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def _path_collision_free(path_x, path_y, x_grid, y_grid, blocked):
        h, w = blocked.shape
        for x, y in zip(path_x, path_y):
            ix = int(np.argmin(np.abs(x_grid - x)))
            iy = int(np.argmin(np.abs(y_grid - y)))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                return False
            if blocked[iy, ix]:
                return False
        return True

    @staticmethod
    def _path_evidence_score(path_x, path_y, x_grid, y_grid, hm):
        if path_x is None or len(path_x) == 0:
            return np.inf
        vals = []
        h, w = hm.shape
        for x, y in zip(path_x, path_y):
            ix = int(np.argmin(np.abs(x_grid - x)))
            iy = int(np.argmin(np.abs(y_grid - y)))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                continue
            vals.append(float(hm[iy, ix]))
        if len(vals) == 0:
            return np.inf
        return float(np.mean(vals))

    def _plan_circles(self, hm_norm):
        wall_mask = hm_norm >= self.config.occ_block_threshold
        grid_mm = float(np.mean(np.diff(self.x_grid))) if len(self.x_grid) > 1 else self.config.grid_mm
        inflate_px = int(np.ceil((self.config.robot_radius_mm + self.config.safety_margin_mm) / max(grid_mm, 1e-6)))
        blocked = self._dilate_binary(wall_mask, inflate_px)

        radii = np.arange(
            self.config.circle_radius_min_mm,
            self.config.circle_radius_max_mm + self.config.circle_radius_step_mm,
            self.config.circle_radius_step_mm,
            dtype=np.float32,
        )
        result = {}
        for side_name, side_sign in [('left', 1.0), ('right', -1.0)]:
            best_r = None
            best_xy = (None, None)
            touched_obstacle = False
            for r in radii:
                px, py = self._sample_tangent_circle(float(r), side_sign, self.config.circle_horizon_x_mm, self.config.circle_arc_samples)
                ok = self._path_collision_free(px, py, self.x_grid, self.y_grid, blocked)
                if ok:
                    best_r = float(r)
                    best_xy = (px, py)
                else:
                    touched_obstacle = True
                    break
            result[side_name] = {
                'radius': best_r,
                'x': best_xy[0],
                'y': best_xy[1],
                'touched_obstacle': touched_obstacle,
            }

        left_score = self._path_evidence_score(result['left']['x'], result['left']['y'], self.x_grid, self.y_grid, hm_norm)
        right_score = self._path_evidence_score(result['right']['x'], result['right']['y'], self.x_grid, self.y_grid, hm_norm)

        r_left = result['left']['radius'] if result['left']['radius'] is not None else -np.inf
        r_right = result['right']['radius'] if result['right']['radius'] is not None else -np.inf
        left_never_touched = not result['left']['touched_obstacle']
        right_never_touched = not result['right']['touched_obstacle']

        if (r_left < 0) and (r_right < 0):
            chosen_side = 'straight'
            chosen_radius = np.inf
        elif left_never_touched and right_never_touched:
            chosen_side = 'straight_open'
            chosen_radius = np.inf
        else:
            if abs(float(r_left - r_right)) <= float(self.config.circle_radius_tie_mm):
                if left_score <= right_score:
                    chosen_side = 'left'
                    chosen_radius = r_left
                else:
                    chosen_side = 'right'
                    chosen_radius = r_right
            elif r_left > r_right:
                chosen_side = 'left'
                chosen_radius = r_left
            else:
                chosen_side = 'right'
                chosen_radius = r_right

        return {
            'chosen_side': chosen_side,
            'chosen_radius_mm': float(chosen_radius) if np.isfinite(chosen_radius) else float('inf'),
            'left_radius_mm': None if result['left']['radius'] is None else float(result['left']['radius']),
            'right_radius_mm': None if result['right']['radius'] is None else float(result['right']['radius']),
            'left_evidence_score': float(left_score) if np.isfinite(left_score) else float('nan'),
            'right_evidence_score': float(right_score) if np.isfinite(right_score) else float('nan'),
            'left_x': result['left']['x'],
            'left_y': result['left']['y'],
            'right_x': result['right']['x'],
            'right_y': result['right']['y'],
            'blocked_mask': blocked,
        }

    @staticmethod
    def _planner_to_curvature(planner):
        side = str(planner.get('chosen_side', 'straight'))
        radius = float(planner.get('chosen_radius_mm', np.inf))
        if not np.isfinite(radius) or radius <= 1e-6:
            return 0.0
        if side.startswith('left'):
            return 1.0 / radius
        if side.startswith('right'):
            return -1.0 / radius
        return 0.0
