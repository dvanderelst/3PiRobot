# Vision-Guided Sonar Data Acquisition — Work in Progress

Picking up: write `SCRIPT_VisualDataAcquisition.py` (the runner that consumes a saved plan and pings the real robot through it).

## What we're building

Protocol B from `Control_code/Docs/rationale.md` — the overhead tracker plus an annotated arena map actively steers the robot through deliberately chosen poses, in contrast to the existing sonar-IID-driven `SCRIPT_DataAcquisition.py` which avoids the very close-wall and corner configurations the SonarModel needs the most. Generalised across multiple arena layouts.

## Pipeline (one-time per layout)

1. Snapshot the arena into `AcquisitionArenas/<layout>/env_*/` (already done for `Acquisition01`).
2. Manually annotate `arena_*_mask.png` → `arena_*_annotated.png` (green polylines along wall tops).
3. `SCRIPT_BuildArenaGeometry.py` — walks `AcquisitionArenas/` and `TargetArenas/` (refactored to accept parent folders that contain arena subfolders). Produces `arena_walls.npz` per env.
4. `SCRIPT_BuildAcquisitionPlan.py` — samples a tour of feasible `(x, y)` waypoints; writes `plans/plan_<timestamp>.json` and `.png`. Re-run as needed until the plot looks right.

## Pipeline (per acquisition run — the next thing to write)

5. **`SCRIPT_VisualDataAcquisition.py` (TODO)** — consumes a saved plan and the arena. For each waypoint: drive there, then for each yaw in `yaws_at_position[i]` rotate, settled-poll, ping, save. Output goes to `AcquisitionSessions/<session>/` (folder will be `mkdir -p`'d on first run).

### Runner shape we agreed on

- Loads `plan.json` + arena walls.
- Reads tracker → drives to `plan.positions[0]` via `TrackerNav.go_to_pose(start_xy)`. User places the robot near it; the controller covers the last bit.
- Per waypoint: `nav.go_to_pose(x_i, y_i)` → for each yaw in `yaws_at_position[i]`: `nav.go_to_pose(x_i, y_i, yaw)` (rotation-only at this point) → ping → save.
- Per ping artifact: `sonar_package` (full), executed pose (tracker), planned pose, **ground-truth profile via `ArenaLayout.compute_profile`**, position index, yaw index. Ground-truth profile is the supervisory signal — direct labels for SonarModel training, free of cost since the walls are known.
- `PauseControl.wait_if_paused()` guards each iteration. Pushover notifications every ~100 pings. Nav failures logged; plan continues to the next waypoint.

## Key support code already in place

- **`Library/TrackerNav.py`** — closed-loop go-to-pose using overhead tracker. `wait_for_stable_pose` settles on yaw + position spread before returning. `post_step_delay_s = 0.5 s` buffer after each motor command (caller responsibility — there's a long doc comment in the function explaining why and why this isn't baked into the helper). `max_yaw_iterations = 8` (covers a full ~180° flip with calibration uncertainty).
- **`Library/AcquisitionPlanner.py`** — geometry helpers (`min_dist_point_to_walls`, `min_dist_segment_to_walls`), arena loader, sequential plan builder, save/load JSON, diagnostic plot. Uses convex hull of wall points as the feasibility region (the bounding box from `meta.json` is the camera FOV and overshoots the arena). Convex-hull approximation is fine for roughly-convex arenas; switch to alpha-shape if we ever annotate an L-shape.
- **Rotation calibration** — re-measured per-robot on 2026-05-06 with the new settled-poll path; values are in `Library/Settings.py`. `SCRIPT_CalibrateRotation.py` was updated to use `wait_for_stable_pose` and a `POST_STEP_DELAY_S = 1 s` buffer.

## Parameters settled on (planner)

- `CLEARANCE_MM = 250` — below this the emit pulse masks first echoes (sonar-physics constraint, not just safety).
- `MIN_STEP_MM = 300` — slightly bigger than clearance so consecutive waypoints aren't on top of each other.
- `TARGET_K = 150` waypoints.
- `N_YAWS = 8` per waypoint (45° spacing, random offset φ ∈ [0, 45°) per position) → ~1200 pings per session.
- Tour grown sequentially: each new waypoint must be reachable from the previous via a straight segment that maintains clearance. No separate TSP step needed — the order *is* the sampling order.

## Sampling rationale (don't forget tomorrow)

We almost picked "stratify on 3-slice min-distance" (the SonarModel's targets) but rejected it: two distinct `(x, y, yaw)` poses can produce the same 3-slice profile via different full-wall geometries, hence different envelopes. Deduplicating on labels would throw away exactly the input-space diversity the model needs. Random uniform over feasible `(x, y)` + multi-yaw per position naturally covers close-wall configurations because they're a non-trivial fraction of feasible area, and Protocol A misses them entirely by design — that's where Protocol B's leverage comes from.

## Test status

- `Library/TrackerNav.py` validated on the robot via `SCRIPT_SmokeTestTrackerNav.py`. Rotation calibration good after re-measurement; pings happen at each waypoint with valid sonar.
- `SCRIPT_BuildArenaGeometry.py` confirmed working against the new `AcquisitionArenas/` parent folder.
- `SCRIPT_BuildAcquisitionPlan.py` produces a clean plan against `Acquisition01/` (latest plan in `plans/`). The polygon-containment fix is in — first run had all positions outside the arena, second run with convex-hull constraint is correct.

## Open items (later)

- `client2`/`client3` in `Library/Settings.py` inherit `client1`'s rotation calibration via the dataclass `default_factory`. Per-robot overrides not yet wired.
- `AcquisitionSessions/` folder doesn't exist yet — runner will create on first execution.
