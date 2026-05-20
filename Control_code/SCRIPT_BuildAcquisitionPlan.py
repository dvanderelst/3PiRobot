"""
Build a sampling plan for vision-guided sonar data acquisition in a target arena.

Reads AcquisitionArenas/<ARENA_NAME>/env_*/arena_features.npz (most recent env_*),
which carries both wall points and pole centres + radius. Samples a tour of
feasible (x, y) waypoints subject to:
  - wall clearance:     each waypoint and each segment stays >= CLEARANCE_MM
                        from any wall point
  - pole clearance:     each waypoint and each segment stays
                        >= CLEARANCE_MM + pole_radius_mm from any pole centre
  - step length:        each new waypoint >= MIN_STEP_MM from its predecessor
  - neighbour spacing:  each new waypoint >= MIN_NEIGHBOR_MM from every prior
                        waypoint (spreads coverage)

Writes plan_<timestamp>.json (full spec) and plan_<timestamp>.png (diagnostic
plot) into AcquisitionArenas/<ARENA_NAME>/plans/. Run multiple times and
compare plots until you like the look — the run script consumes a saved plan
file.
"""

import time
from pathlib import Path

from Library.AcquisitionPlanner import (
    AcquisitionPlan,
    build_plan,
    load_arena,
    plot_diagnostics,
    plot_plan,
    reorder_tour,
    save_plan,
    total_path_length_mm,
)


# ── Settings ──────────────────────────────────────────────────────────────────
ARENA_NAME             = "Acquisition04"
TARGET_K               = 250       # number of waypoints (excluding the start)
N_YAWS                 = 5         # measurements per position (72° spacing → cones non-overlapping)
CLEARANCE_MM           = 250.0     # min distance from any wall (point AND segment)
MIN_STEP_MM            = 300.0     # min distance between consecutive waypoints
MIN_NEIGHBOR_MM        = 200.0     # min distance from ANY prior waypoint (spreads coverage)
MAX_ATTEMPTS_PER_STEP  = 200       # safety cap on rejection sampling per step
REORDER_TOUR           = True      # post-hoc nearest-neighbour TSP to cut drive time
SEED                   = None      # int for reproducibility; None = wall-clock


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p.resolve()


def main():
    arena_dir = _resolve(f"AcquisitionArenas/{ARENA_NAME}")
    if not arena_dir.is_dir():
        raise SystemExit(f"Arena dir not found: {arena_dir}")

    arena = load_arena(arena_dir)
    print(f"Loaded arena from {arena_dir}")
    print(f"  env: {arena['env_dir'].name}")
    print(f"  walls: {arena['walls'].shape[0]} points")
    b = arena["bounds"]
    print(f"  bounds: x=[{b['min_x']:.0f}, {b['max_x']:.0f}], "
          f"y=[{b['min_y']:.0f}, {b['max_y']:.0f}] mm")

    seed = SEED if SEED is not None else int(time.time())
    print(f"\nBuilding plan (seed={seed}, K={TARGET_K}, "
          f"n_yaws={N_YAWS}, clearance={CLEARANCE_MM:.0f} mm, "
          f"min_step={MIN_STEP_MM:.0f} mm, "
          f"min_neighbor={MIN_NEIGHBOR_MM:.0f} mm)...")

    plan = build_plan(
        arena=arena,
        target_k=TARGET_K,
        clearance_mm=CLEARANCE_MM,
        min_step_mm=MIN_STEP_MM,
        min_neighbor_mm=MIN_NEIGHBOR_MM,
        max_attempts_per_step=MAX_ATTEMPTS_PER_STEP,
        n_yaws=N_YAWS,
        arena_name=ARENA_NAME,
        arena_dir=str(arena_dir),
        seed=seed,
    )

    n_pos = len(plan.positions)
    n_pings = n_pos * plan.n_yaws
    print(f"\nPlan: {n_pos} positions × {plan.n_yaws} yaws = {n_pings} pings")
    if n_pos < TARGET_K + 1:
        print(f"  (truncated from target {TARGET_K + 1}; check plot for cause)")

    if REORDER_TOUR and n_pos >= 3:
        len_before = total_path_length_mm(plan.positions)
        plan = reorder_tour(plan, arena)
        len_after = total_path_length_mm(plan.positions)
        print(f"  tour length: {len_before/1000:.1f} m -> {len_after/1000:.1f} m "
              f"after NN reorder ({100 * (1 - len_after / len_before):.0f}% reduction)")

    plans_dir = arena_dir / "plans"
    plans_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    plan_path = plans_dir / f"plan_{timestamp}.json"
    plot_path = plans_dir / f"plan_{timestamp}.png"
    diag_path = plans_dir / f"diagnostics_{timestamp}.png"
    save_plan(plan, plan_path)
    plot_plan(plan, arena, plot_path)
    plot_diagnostics(plan, arena, diag_path)
    print(f"\nSaved {plan_path.relative_to(arena_dir.parent.parent)}")
    print(f"Saved {plot_path.relative_to(arena_dir.parent.parent)}")
    print(f"Saved {diag_path.relative_to(arena_dir.parent.parent)}")


if __name__ == "__main__":
    main()
