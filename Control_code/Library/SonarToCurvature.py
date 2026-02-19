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
        self._last_pose = None
        self._last_chosen_side = None

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
        self._last_pose = None
        self._last_chosen_side = None

    def update(self, sonar_lr, pose=None, delta_pose=None, plot=False, debug_pose_shift=False):
        self.sonar_window.append(np.asarray(sonar_lr, dtype=np.float32))
        pose_xyz = self._resolve_pose(pose=pose, delta_pose=delta_pose)
        self.pose_window.append(pose_xyz)

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
        if debug_pose_shift:
            from Library import DataProcessor

            anchor_x = float(rob_x_seq[-1])
            anchor_y = float(rob_y_seq[-1])
            anchor_yaw = float(rob_yaw_seq[-1])
            # Pose displacement of the first frame in the anchor robot frame.
            dx = float(rob_x_seq[0] - anchor_x)
            dy = float(rob_y_seq[0] - anchor_y)
            yaw_rad = np.deg2rad(anchor_yaw)
            c, s = np.cos(yaw_rad), np.sin(yaw_rad)
            pose_x_rel = c * dx + s * dy
            pose_y_rel = -s * dx + c * dy

            def _frame_stats(i):
                az_deg = profile_centers_deg_seq[i]
                dist_i = distance_mm[i].copy()
                dist_i[presence_bin[i] == 0] = np.nan
                xw, yw = DataProcessor.robot2world(az_deg, dist_i, float(rob_x_seq[i]), float(rob_y_seq[i]), float(rob_yaw_seq[i]))
                xr, yr = DataProcessor.world2robot(xw, yw, anchor_x, anchor_y, anchor_yaw)
                xr = xr[np.isfinite(xr)]
                if xr.size == 0:
                    return np.nan, np.nan, np.nan
                return float(np.nanmean(xr)), float(np.nanmin(xr)), float(np.nanmax(xr))

            mean0, min0, max0 = _frame_stats(0)
            meanN, minN, maxN = _frame_stats(n_win - 1)
            delta_mean = meanN - mean0 if np.isfinite(mean0) and np.isfinite(meanN) else np.nan
            print(
                f"[PoseShift] win={n_win} pose_first_rel=({pose_x_rel:.1f},{pose_y_rel:.1f}) "
                f"mean_x first={mean0:.1f} last={meanN:.1f} delta={delta_mean:.1f} | "
                f"min/max first=({min0:.1f},{max0:.1f}) last=({minN:.1f},{maxN:.1f})"
            )
        planner = plan_circles_from_heatmap(
            hm_norm=hm_norm,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            config=self.steering_config,
        )
        self._apply_hysteresis(planner)
        signed_curvature = float(planner_to_curvature(planner))
        if plot:
            self._plot_estimate(hm_norm, planner, pose=pose)
        return {
            'ready': True,
            'signed_curvature_inv_mm': signed_curvature,
            'window_samples_used': int(n_win),
            'window_size_config': int(self.steering_config.window_size),
            'planner': planner,
            'occupancy': hm_norm,
        }

    def update_from_package(self, sonar_package, pose=None, delta_pose=None, plot=False, debug_pose_shift=False):
        sonar_lr = self.extract_sonar_lr(sonar_package)
        return self.update(
            sonar_lr=sonar_lr,
            pose=pose,
            delta_pose=delta_pose,
            plot=plot,
            debug_pose_shift=debug_pose_shift,
        )

    def _plot_estimate(self, hm_norm, planner, pose=None):
        import matplotlib.pyplot as plt
        import numpy as np

        extent = [
            float(self.x_grid.min()),
            float(self.x_grid.max()),
            float(self.y_grid.min()),
            float(self.y_grid.max()),
        ]
        plt.figure(figsize=(6, 6))
        plt.imshow(hm_norm, origin='lower', extent=extent, vmin=0.0, vmax=1.0, cmap='magma', aspect='equal')

        # Robot-centric marker: always at origin with heading arrow.
        x0, y0 = 0.0, 0.0
        plt.plot([x0], [y0], marker='o', color='white', markersize=6, zorder=5)
        plt.arrow(
            x0,
            y0,
            250.0,
            0.0,
            width=6.0,
            head_width=25.0,
            head_length=25.0,
            color='white',
            zorder=5,
        )

        # Pose window trail in robot frame (anchor at last pose).
        if len(self.pose_window) > 0:
            pose_arr = np.asarray(list(self.pose_window), dtype=np.float32)
            anchor_x, anchor_y, anchor_yaw = pose_arr[-1]
            dx = pose_arr[:, 0] - anchor_x
            dy = pose_arr[:, 1] - anchor_y
            yaw_rad = np.deg2rad(anchor_yaw)
            c, s = np.cos(yaw_rad), np.sin(yaw_rad)
            x_rel = c * dx + s * dy
            y_rel = -s * dx + c * dy
            plt.plot(x_rel, y_rel, 'o-', color='white', alpha=0.6, markersize=3, linewidth=1.0)

        # Circle planner overlays.
        left_x = planner.get('left_x', None)
        left_y = planner.get('left_y', None)
        right_x = planner.get('right_x', None)
        right_y = planner.get('right_y', None)
        chosen = str(planner.get('chosen_side', 'straight'))

        def _plot_arc(x, y, chosen_side, side_name, color):
            if x is None or y is None or len(x) == 0:
                return
            is_chosen = chosen_side.startswith(side_name)
            ls = '-' if is_chosen else '--'
            lw = 2.5 if is_chosen else 1.5
            plt.plot(x, y, ls=ls, lw=lw, color=color, alpha=0.9)

        _plot_arc(left_x, left_y, chosen, 'left', color='cyan')
        _plot_arc(right_x, right_y, chosen, 'right', color='lime')

        plt.title('Occupancy + Planner')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.colorbar(label='Occupancy')
        plt.tight_layout()
        plt.show()

    def _resolve_pose(self, pose=None, delta_pose=None):
        if delta_pose is not None:
            dx, dy, dyaw = self._pose_to_xyz(delta_pose)
            if self._last_pose is None:
                yaw0 = 0.0
                yaw1 = yaw0 + dyaw
                c, s = np.cos(np.deg2rad(yaw1)), np.sin(np.deg2rad(yaw1))
                pose_xyz = (c * dx - s * dy, s * dx + c * dy, yaw1)
            else:
                x0, y0, yaw0 = self._last_pose
                yaw1 = yaw0 + dyaw
                c, s = np.cos(np.deg2rad(yaw1)), np.sin(np.deg2rad(yaw1))
                pose_xyz = (x0 + (c * dx - s * dy), y0 + (s * dx + c * dy), yaw1)
        elif pose is not None:
            pose_xyz = self._pose_to_xyz(pose)
        else:
            pose_xyz = (0.0, 0.0, 0.0)
        self._last_pose = pose_xyz
        return pose_xyz

    def _apply_hysteresis(self, planner):
        hysteresis_mm = float(getattr(self.steering_config, 'circle_radius_hysteresis_mm', 0.0))
        if hysteresis_mm <= 0.0:
            self._last_chosen_side = planner.get('chosen_side', None)
            return

        r_left = planner.get('left_radius_mm', None)
        r_right = planner.get('right_radius_mm', None)
        if r_left is None or r_right is None:
            self._last_chosen_side = planner.get('chosen_side', None)
            return

        if not np.isfinite(r_left) or not np.isfinite(r_right):
            self._last_chosen_side = planner.get('chosen_side', None)
            return

        if abs(float(r_left - r_right)) > hysteresis_mm:
            self._last_chosen_side = planner.get('chosen_side', None)
            return

        if self._last_chosen_side in ('left', 'right'):
            chosen_side = self._last_chosen_side
            planner['chosen_side'] = chosen_side
            planner['chosen_radius_mm'] = float(r_left if chosen_side == 'left' else r_right)
        self._last_chosen_side = planner.get('chosen_side', None)
