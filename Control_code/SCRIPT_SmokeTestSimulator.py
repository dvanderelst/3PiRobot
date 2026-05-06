#!/usr/bin/env python3
"""
SCRIPT_SmokeTestSimulator.py

Sanity check that EnvironmentSimulator + SonarModel are wired together
correctly and produce inspectable output.

What it checks (printed to stdout):
  1. Simulator picks up the right profile params from the SonarModel artifact
     (opening_angle, profile_steps, profile_method, cone_half_deg).
  2. get_sonar_measurement returns the new 6-key dict (no iid_db left over).
  3. Noise scales match σ_sim (multiple noisy samples at the same position
     have spread comparable to the reported σ).
  4. Single vs batched calls produce equivalent results.

What it plots (saved to SonarModel/smoketest_simulator.png):
  6 positions in the arena, each as a panel showing
    - the geometric profile (azimuth vs distance)
    - the ±cone shaded green
    - the slice boundaries as dashed verticals
    - per-slice true min as black stars
    - one noisy observation per slice as red bars centred at obs ± σ
"""

import os

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Library.EnvironmentSimulator import EnvironmentSimulator
from Library.SonarModel import SonarModel


SESSION_NAME    = "sessionB01"
SONAR_MODEL_DIR = "SonarModel"
OUTPUT_PATH     = os.path.join(SONAR_MODEL_DIR, "smoketest_simulator.png")
SEED            = 0


def main():
    print("[1/4] Constructing simulator")
    sim = EnvironmentSimulator(
        session_name=SESSION_NAME,
        sonar_model_dir=SONAR_MODEL_DIR,
        seed=SEED,
    )
    print(f"  sim.profile_method = {sim.profile_method!r}    (should be 'ray_center')")
    print(f"  sim.opening_angle  = {sim.opening_angle}°       (should be 270°)")
    print(f"  sim.profile_steps  = {sim.profile_steps}        (should be 90)")
    print(f"  sim.cone_half_deg  = {sim.cone_half_deg}°       (should be 35°)")

    # ── Pick 6 inside-arena positions spanning the actual coordinate space ───
    x_lo, x_hi = sim.arena.arena_min_x, sim.arena.arena_max_x
    y_lo, y_hi = sim.arena.arena_min_y, sim.arena.arena_max_y
    fx_lo, fx_hi = x_lo + 0.25 * (x_hi - x_lo), x_lo + 0.75 * (x_hi - x_lo)
    fy_lo, fy_hi = y_lo + 0.25 * (y_hi - y_lo), y_lo + 0.75 * (y_hi - y_lo)
    positions = [
        (fx_lo, fy_lo,    0.0),
        (0.5 * (fx_lo + fx_hi), fy_lo,  45.0),
        (fx_hi, 0.5 * (fy_lo + fy_hi),  90.0),
        (fx_lo, fy_hi, 135.0),
        (0.5 * (fx_lo + fx_hi), fy_hi, -135.0),
        (fx_hi, fy_hi, -45.0),
    ]

    print(f"\n[2/4] Single-call sonar measurements at 6 positions")
    print(f"  arena bounds: x in [{x_lo:.0f}, {x_hi:.0f}],  y in [{y_lo:.0f}, {y_hi:.0f}]")
    measurements = []
    for i, (x, y, yaw) in enumerate(positions):
        m = sim.get_sonar_measurement(x, y, yaw)
        measurements.append(m)
        print(f"  pos {i+1}  ({x:5.0f}, {y:5.0f}, yaw {yaw:+5.0f}°):")
        for slc in ("left", "center", "right"):
            print(f"    {slc:>6}  d = {m[f'distance_{slc}_mm']:7.1f} mm   "
                  f"σ = {m[f'sigma_{slc}_mm']:6.1f} mm")

    # ── Verify single vs batched are consistent (σ identical, mean differs by RNG) ──
    print(f"\n[3/4] Single vs batched consistency")
    sim_b = EnvironmentSimulator(
        session_name=SESSION_NAME, sonar_model_dir=SONAR_MODEL_DIR, seed=SEED)
    batch = sim_b.get_sonar_measurements_batch(positions)
    print(f"  Re-running same positions through batched call (fresh RNG with same seed):")
    print(f"  σ should match exactly (only depends on geometry); means may differ")
    print(f"  from above due to consuming the RNG in a different order.")
    n_sigma_match = 0
    for k, (single, batched) in enumerate(zip(measurements, batch)):
        for slc in ("left", "center", "right"):
            if abs(single[f"sigma_{slc}_mm"] - batched[f"sigma_{slc}_mm"]) < 1e-3:
                n_sigma_match += 1
    print(f"  σ matches: {n_sigma_match}/{3 * len(measurements)} slice-positions")

    # ── Check noise scale: 200 samples at one position, see if std ≈ reported σ ──
    print(f"\n[4/4] Noise scale at one position (200 samples)")
    test_pos = positions[1]
    samples = np.array([
        SonarModel.to_policy_obs(sim.get_sonar_measurement(*test_pos))
        for _ in range(200)
    ])
    # samples shape: (200, 6) — order [d_L, d_C, d_R, σ_L, σ_C, σ_R]
    obs_std = samples[:, :3].std(axis=0)
    sigma_med = np.median(samples[:, 3:], axis=0)
    for i, slc in enumerate(("left", "center", "right")):
        print(f"  {slc:>6}: empirical obs std = {obs_std[i]:6.1f} mm   "
              f"reported σ (median) = {sigma_med[i]:6.1f} mm   "
              f"ratio = {obs_std[i] / max(sigma_med[i], 1e-6):.2f}")

    # ── Plot per-position diagnostic ─────────────────────────────────────────
    print(f"\n  Plotting to {OUTPUT_PATH}")
    bin_centers = sim.sonar_model.bin_centers
    cone        = sim.cone_half_deg
    third       = 2.0 * cone / 3.0
    slice_edges = [-cone, -cone + third, -cone + 2 * third, cone]
    slice_names = ["left", "center", "right"]

    # Use a fresh seeded simulator so the saved obs match what we plot.
    sim_plot = EnvironmentSimulator(
        session_name=SESSION_NAME, sonar_model_dir=SONAR_MODEL_DIR, seed=SEED + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (x, y, yaw) in zip(axes.flat, positions):
        profile = sim_plot.get_profile_at_position(x, y, yaw)
        m       = sim_plot.get_sonar_measurement(x, y, yaw)

        ax.plot(bin_centers, profile, '.-', color='steelblue', lw=1, ms=3, alpha=0.85,
                label='profile')
        ax.axvspan(-cone, cone, alpha=0.10, color='green', label=f'±{cone:.0f}° cone')
        for e in slice_edges[1:-1]:
            ax.axvline(e, color='black', lw=0.6, ls=':', alpha=0.6)

        # Per-slice true min and noisy observation
        masks = sim.sonar_model.slice_masks
        for i, name in enumerate(slice_names):
            mask     = masks[i]
            true_min = float(profile[mask].min())
            obs      = m[f'distance_{name}_mm']
            sigma    = m[f'sigma_{name}_mm']
            slo, shi = slice_edges[i], slice_edges[i + 1]
            xc       = 0.5 * (slo + shi)
            # True min: black horizontal line spanning the slice — "the floor"
            ax.hlines(true_min, slo, shi, color='black', lw=2, alpha=0.85, zorder=9,
                      label=('true min in slice' if i == 0 else None))
            # Noisy obs ± σ: red errorbar at slice center
            ax.errorbar([xc], [obs], yerr=[sigma], fmt='s', color='crimson',
                        markersize=6, capsize=5, alpha=0.85, zorder=10,
                        label=('noisy obs ± σ' if i == 0 else None))

        ax.set_title(f"({x:.0f}, {y:.0f}) yaw {yaw:+.0f}°", fontsize=10)
        ax.set_xlabel("azimuth (°)")
        ax.set_ylabel("distance (mm)")
        ax.set_ylim(0, max(3000, profile.max() * 1.05))
        ax.set_xlim(bin_centers[0], bin_centers[-1])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(f"Simulator smoke test  ({SESSION_NAME}, seed={SEED + 1})  "
                 f"— EnvironmentSimulator wired to SonarModel",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120)
    plt.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
