from dataclasses import dataclass

from Library import Settings


@dataclass
class SteeringConfig:
    window_size: int = Settings.occupancy_config.window_size
    presence_threshold_override: float | None = None

    # Robot-frame occupancy grid/integration.
    extent_mm: float = Settings.occupancy_config.extent_mm
    grid_mm: float = Settings.occupancy_config.grid_mm
    sigma_perp_mm: float = Settings.occupancy_config.sigma_perp_mm
    sigma_para_mm: float = Settings.occupancy_config.sigma_para_mm
    apply_heatmap_smoothing: bool = Settings.occupancy_config.apply_heatmap_smoothing

    # Circle planner.
    occ_block_threshold: float = Settings.curvature_config.occ_block_threshold
    robot_radius_mm: float = Settings.curvature_config.robot_radius_mm
    safety_margin_mm: float = Settings.curvature_config.safety_margin_mm
    circle_radius_min_mm: float = Settings.curvature_config.circle_radius_min_mm
    circle_radius_max_mm: float = Settings.curvature_config.circle_radius_max_mm
    circle_radius_step_mm: float = Settings.curvature_config.circle_radius_step_mm
    circle_arc_samples: int = Settings.curvature_config.circle_arc_samples
    circle_horizon_x_mm: float = Settings.curvature_config.circle_horizon_x_mm
    circle_radius_tie_mm: float = Settings.curvature_config.circle_radius_tie_mm
    circle_radius_hysteresis_mm: float = Settings.curvature_config.circle_radius_hysteresis_mm

    # Optional per-call debug plotting.
    debug_plot_dir: str | None = None
    debug_plot_dpi: int = 180
