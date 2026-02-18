from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from Library import DataProcessor, Utils


# ============================================
# CONFIGURATION
# ============================================
session = "sessionB01"

# Session trajectory/profile visualization.
show_trajectory_plot = True
label_step_trajectory_plot = 5

# Sonar-vs-visual consistency check.
run_sonar_visual_check = True
profile_steps = 11
profile_opening_angles_deg = [10, 20, 40, 60, 80, 90]  # full opening angles in degrees
distance_tolerance_m = 1.0  # sonar too far/too close threshold in meters


def plot_session_trajectory(session_name):
    processor = DataProcessor.DataProcessor(session_name)
    _ = processor.load_profiles(opening_angle=max(profile_opening_angles_deg), steps=profile_steps)

    n_steps = processor.n
    rob_x = processor.rob_x
    rob_y = processor.rob_y
    rob_yaw_deg = processor.rob_yaw_deg
    wall_x = processor.wall_x
    wall_y = processor.wall_y

    plt.figure(figsize=(10, 8))
    plt.plot(rob_x, rob_y, color='black', alpha=0.5, label='Trajectory')
    plt.scatter(wall_x, wall_y, color='green', s=10, alpha=0.5, label='Walls')
    # Plot all robot poses in one call to avoid repeated legend entries.
    Utils.plot_robot_positions(x=rob_x, y=rob_y, yaws_deg=rob_yaw_deg)
    for i in range(n_steps):
        if i % max(1, int(label_step_trajectory_plot)) == 0:
            plt.text(x=rob_x[i], y=rob_y[i], s=str(i), color='red', fontsize=7)
    plt.title(f"Session overview: {session_name}")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def run_session_sonar_visual_check(session_name):
    if len(profile_opening_angles_deg) == 0:
        raise ValueError('opening_angles_deg must contain at least one value.')
    max_opening_angle = float(max(profile_opening_angles_deg))
    max_half_angle = 0.5 * max_opening_angle

    dc = DataProcessor.DataCollection([session_name])
    profiles_mm, centers_deg = dc.load_profiles(opening_angle=max_opening_angle, steps=profile_steps, fill_nans=True)
    profiles_m = profiles_mm / 1000.0
    sonar_distance = np.asarray(dc.get_field('sonar_package', 'corrected_distance'), dtype=np.float32)
    sonar_iid = np.asarray(dc.get_field('sonar_package', 'corrected_iid'), dtype=np.float32)
    sonar_iid_sign = np.sign(sonar_iid)

    # Keep finite rows only so visual and sonar arrays stay aligned.
    finite = np.isfinite(profiles_m).all(axis=1)
    finite &= np.isfinite(sonar_distance)
    finite &= np.isfinite(sonar_iid)
    profiles_m = profiles_m[finite]
    sonar_distance = sonar_distance[finite]
    sonar_iid = sonar_iid[finite]
    sonar_iid_sign = sonar_iid_sign[finite]

    all_results = []
    for opening_angle in profile_opening_angles_deg:
        half_angle = 0.5 * float(opening_angle)
        center_indices = np.where(np.abs(centers_deg[0]) <= half_angle)[0]
        constrained_profiles = profiles_m[:, center_indices]
        constrained_centers = centers_deg[0, center_indices]

        min_idx = np.argmin(constrained_profiles, axis=1)
        closest_visual_direction = constrained_centers[min_idx]
        closest_visual_side = np.sign(closest_visual_direction) * -1.0
        closest_visual_distance = constrained_profiles[np.arange(len(constrained_profiles)), min_idx]

        df = pd.DataFrame({
            'opening_angle_deg': float(opening_angle),
            'closest_visual_direction': closest_visual_direction,
            'closest_visual_side': closest_visual_side,
            'closest_visual_distance': closest_visual_distance,
            'sonar_distance': sonar_distance,
            'sonar_iid': sonar_iid,
            'sonar_iid_sign': sonar_iid_sign,
        })
        df['side_matches'] = df['closest_visual_side'] == df['sonar_iid_sign']
        all_results.append(df)

    all_results = pd.concat(all_results, axis=0, ignore_index=True)

    # Scatter plots: sonar distance vs closest visual distance by extent.
    n = len(profile_opening_angles_deg)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(12, 4 * nrows))
    too_fars = []
    too_closes = []
    side_accs = []
    for i, opening_angle in enumerate(profile_opening_angles_deg):
        r = all_results.query('opening_angle_deg == @opening_angle')
        vis_d = r['closest_visual_distance'].values
        son_d = r['sonar_distance'].values

        sonar_too_far = (son_d - vis_d) > float(distance_tolerance_m)
        sonar_too_close = (son_d - vis_d) < -float(distance_tolerance_m)
        too_far = float(np.mean(sonar_too_far))
        too_close = float(np.mean(sonar_too_close))
        side_acc = float(np.mean(r['side_matches'].values))
        too_fars.append(too_far)
        too_closes.append(too_close)
        side_accs.append(side_acc)

        ax = plt.subplot(nrows, ncols, i + 1)
        ax.scatter(vis_d, son_d, s=10, alpha=0.4)
        lim_lo, lim_hi = 0.0, max(np.max(vis_d), np.max(son_d), 1.5)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', alpha=0.8)
        ax.plot([lim_lo, lim_hi], [lim_lo - distance_tolerance_m, lim_hi - distance_tolerance_m], 'r:', alpha=0.6)
        ax.plot([lim_lo, lim_hi], [lim_lo + distance_tolerance_m, lim_hi + distance_tolerance_m], 'r:', alpha=0.6)
        ax.set_title(f'Opening {opening_angle}°')
        ax.set_xlabel('Closest visual distance (m)')
        ax.set_ylabel('Sonar distance (m)')
        ax.grid(True, alpha=0.25)
        ax.text(
            0.03, 0.97,
            f'too far={too_far:.2f}\ntoo close={too_close:.2f}\nside acc={side_acc:.2f}',
            transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    plt.tight_layout()
    plt.show()

    # Aggregate trends vs extent.
    plt.figure(figsize=(10, 5))
    plt.plot(profile_opening_angles_deg, too_fars, 'o-', label='Sonar too far')
    plt.plot(profile_opening_angles_deg, too_closes, 'o-', label='Sonar too close')
    plt.plot(profile_opening_angles_deg, side_accs, 'o-', label='Side match accuracy (IID sign)')
    plt.xlabel('Opening angle (deg)')
    plt.ylabel('Fraction')
    plt.title(f'Sonar vs Visual Consistency: {session_name}')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    print(f"Session check summary for {session_name}:")
    for opening_angle, tf, tc, sa in zip(profile_opening_angles_deg, too_fars, too_closes, side_accs):
        print(f"  opening={opening_angle:>3}° | too_far={tf:.3f} | too_close={tc:.3f} | side_acc={sa:.3f}")


def main():
    if show_trajectory_plot:
        plot_session_trajectory(session)
    if run_sonar_visual_check:
        run_session_sonar_visual_check(session)


if __name__ == '__main__':
    main()
