#%%
"""
This script calculates the correspondence between the layout of the environment and the
distance/side of closest obstacle as inferred from the sonar data.

Produces one figure per session showing, for each opening angle, a scatter plot of
closest visual distance (from profile) vs corrected sonar distance.  Figures are saved
to the Plots/ folder so sessions can be compared side-by-side.
"""
import os
import numpy as np
from Library import Utils
from Library import DataProcessor
from matplotlib import pyplot as plt
import pandas as pd


az_steps = 121
opening_angles = [60, 90, 120, 150, 180]  # full opening angles in degrees
max_opening_angle = max(opening_angles)

sessions = ['sessionB01', 'sessionB02', 'sessionB03', 'sessionB04', 'sessionB05']

os.makedirs('Plots', exist_ok=True)

for session in sessions:
    print(f"\n=== {session} ===")
    collection = DataProcessor.DataCollection([session])
    full_profiles_mm, centers_2d = collection.load_profiles(opening_angle=max_opening_angle, steps=az_steps)
    centers = centers_2d[0]  # 1D azimuth bin centres, shape (az_steps,)

    full_profiles = full_profiles_mm / 1000  # to meters
    sonar_distance = np.asarray(collection.get_field('sonar_package', 'corrected_distance'), dtype=np.float32)
    sonar_iid      = np.asarray(collection.get_field('sonar_package', 'corrected_iid'),      dtype=np.float32)
    sonar_iid_sign = np.sign(sonar_iid)

    all_results = []
    for opening_angle in opening_angles:
        center_indices = np.where(np.abs(centers) <= opening_angle / 2)[0]
        constrained_profiles = full_profiles[:, center_indices]
        constrained_centers  = centers[center_indices]

        indices                = Utils.get_extrema_positions(constrained_profiles, 'min')
        closest_visual_direction = constrained_centers[indices]
        closest_visual_side    = np.sign(closest_visual_direction) * -1
        closest_visual_distance = Utils.get_extrema_values(constrained_profiles, 'min')

        df = pd.DataFrame({
            'closest_visual_direction': closest_visual_direction,
            'closest_visual_distance':  closest_visual_distance,
            'closest_visual_side':      closest_visual_side,
            'sonar_distance':           sonar_distance,
            'sonar_iid':                sonar_iid,
            'sonar_iid_sign':           sonar_iid_sign,
            'opening_angle':            opening_angle,
        })
        df['side_matches'] = df['closest_visual_side'] == df['sonar_iid_sign']
        all_results.append(df)

    all_results = pd.concat(all_results, axis=0, ignore_index=True)

    ncols = len(opening_angles)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    fig.suptitle(session, fontsize=14)

    for ax, opening_angle in zip(axes, opening_angles):
        r = all_results.query('opening_angle == @opening_angle')
        vis_d  = r['closest_visual_distance'].values
        son_d  = r['sonar_distance'].values

        too_far   = np.mean((son_d - vis_d) > 1)
        too_close = np.mean((son_d - vis_d) < -1)
        side_acc  = np.mean(r['side_matches'].values)

        lim = max(np.nanmax(vis_d), np.nanmax(son_d), 1.5)
        ax.scatter(vis_d, son_d, s=6, alpha=0.3)
        ax.plot([0, lim], [0, lim], 'k--', lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel('Closest visual (m)')
        ax.set_ylabel('Sonar distance (m)')
        ax.set_title(f'{opening_angle}°')
        ax.text(0.03, 0.97,
                f'too far:   {too_far:.0%}\ntoo close: {too_close:.0%}\nside acc:  {side_acc:.0%}',
                transform=ax.transAxes, va='top', fontsize=8,
                fontfamily='monospace')
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    save_path = os.path.join('Plots', f'accuracy_{session}.png')
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved → {save_path}")
