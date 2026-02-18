from collections import deque

import numpy as np
import torch

from Library.CurvatureCalculation import plan_circles_from_heatmap, planner_to_curvature
from Library.OccupancyCalculation import build_robot_frame_evidence, make_robot_frame_grid
from Library.ProfileInference import load_training_artifacts, predict_profiles
from Library.SteeringConfigClass import SteeringConfig


class SonarToCurvature:
    """
    Stateful sonar -> profile -> occupancy -> curvature estimator.

    The model predicts profiles per sonar frame. A rolling window of predicted
    profiles + poses is then integrated into robot-frame occupancy, and a circle
    planner converts occupancy into signed curvature.
    """

    def __init__(
        self,
        training_output_dir='Training',
        steering_config: SteeringConfig | None = None,
        presence_threshold_override: float | None = None,
        prediction_batch_size: int = 64,
        allow_partial_window_startup: bool = True,
        device: torch.device | None = None,
    ):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.steering_config = steering_config if steering_config is not None else SteeringConfig()
        self.prediction_batch_size = int(max(1, prediction_batch_size))
        self.allow_partial_window_startup = bool(allow_partial_window_startup)

        self.model, self.y_scaler, self.training_params = load_training_artifacts(training_output_dir, self.device)
        self.profile_opening_angle_deg = float(self.training_params['profile_opening_angle'])
        self.profile_steps = int(self.training_params['profile_steps'])

        th_default = float(self.training_params.get('presence_decision_threshold', 0.5))
        self.presence_threshold = th_default if presence_threshold_override is None else float(presence_threshold_override)

        self.sonar_window = deque(maxlen=int(self.steering_config.window_size))
        self.pose_window = deque(maxlen=int(self.steering_config.window_size))
        self.x_grid, self.y_grid, self.xx, self.yy = make_robot_frame_grid(
            extent_mm=float(self.steering_config.extent_mm),
            grid_mm=float(self.steering_config.grid_mm),
        )
        self.profile_centers_deg_template = np.linspace(
            -0.5 * self.profile_opening_angle_deg,
            0.5 * self.profile_opening_angle_deg,
            self.profile_steps,
            dtype=np.float32,
        )

    @staticmethod
    def extract_sonar_lr(sonar_package):
        sonar_data = np.asarray(sonar_package['sonar_data'], dtype=np.float32)
        if sonar_data.ndim != 2 or sonar_data.shape[1] < 3:
            raise ValueError(f"Unexpected sonar_data shape: {sonar_data.shape}, expected (samples, >=3)")
        return sonar_data[:, [1, 2]].astype(np.float32)

    @staticmethod
    def _pose_to_xyz(pose):
        if isinstance(pose, dict):
            return float(pose['x']), float(pose['y']), float(pose['yaw_deg'])
        if len(pose) != 3:
            raise ValueError('Pose must be dict with x,y,yaw_deg or tuple/list (x,y,yaw_deg).')
        return float(pose[0]), float(pose[1]), float(pose[2])

    def reset(self):
        self.sonar_window.clear()
        self.pose_window.clear()

    def update(self, sonar_lr, pose):
        self.sonar_window.append(np.asarray(sonar_lr, dtype=np.float32))
        self.pose_window.append(self._pose_to_xyz(pose))

        min_required = 1 if self.allow_partial_window_startup else int(self.steering_config.window_size)
        if len(self.sonar_window) < min_required:
            return {
                'ready': False,
                'signed_curvature_inv_mm': None,
                'window_samples_used': int(len(self.sonar_window)),
                'window_size_config': int(self.steering_config.window_size),
                'reason': 'insufficient_window',
            }

        sonar_batch = np.stack(list(self.sonar_window), axis=0).astype(np.float32)
        n_win = int(sonar_batch.shape[0])
        profile_centers_deg_seq = np.repeat(self.profile_centers_deg_template[None, :], n_win, axis=0)

        presence_probs, presence_bin, distance_mm, _ = predict_profiles(
            model=self.model,
            y_scaler=self.y_scaler,
            sonar_data=sonar_batch,
            batch_size=min(self.prediction_batch_size, n_win),
            threshold=float(self.presence_threshold),
            device=self.device,
        )

        pose_arr = np.asarray(list(self.pose_window), dtype=np.float32)
        rob_x_seq = pose_arr[:, 0]
        rob_y_seq = pose_arr[:, 1]
        rob_yaw_seq = pose_arr[:, 2]

        hm_norm = build_robot_frame_evidence(
            profile_centers_deg_seq=profile_centers_deg_seq,
            distance_mm_seq=distance_mm,
            presence_probs_seq=presence_probs,
            presence_bin_seq=presence_bin,
            rob_x_seq=rob_x_seq,
            rob_y_seq=rob_y_seq,
            rob_yaw_deg_seq=rob_yaw_seq,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            xx=self.xx,
            yy=self.yy,
            grid_mm=float(self.steering_config.grid_mm),
            sigma_perp_mm=float(self.steering_config.sigma_perp_mm),
            sigma_para_mm=float(self.steering_config.sigma_para_mm),
            apply_smoothing=bool(self.steering_config.apply_heatmap_smoothing),
        )
        planner = plan_circles_from_heatmap(
            hm_norm=hm_norm,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            config=self.steering_config,
        )
        signed_curvature = float(planner_to_curvature(planner))
        return {
            'ready': True,
            'signed_curvature_inv_mm': signed_curvature,
            'window_samples_used': int(n_win),
            'window_size_config': int(self.steering_config.window_size),
            'planner': planner,
            'occupancy': hm_norm,
        }

    def update_from_package(self, sonar_package, pose):
        sonar_lr = self.extract_sonar_lr(sonar_package)
        return self.update(sonar_lr=sonar_lr, pose=pose)
