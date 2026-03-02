# Pipeline Issues

Found by full cross-file review of Steps 1–4 and all Library modules.

---

## Issue 1 — IID sign convention is inconsistent across the pipeline
**Severity: Moderate — can corrupt analysis and future code reuse**
**Status: FIXED**

Convention is now unified to `locate_echo` standard throughout: **positive IID = wall on right**.

| Where | Formula | Positive means |
|---|---|---|
| `AcousticProcessing.locate_echo` → `corrected_iid` | `20 * log10(right_sum / left_sum)` | wall on **right** |
| `EchoProcessor.iid_from_distance_window` → `iid_db` | `10 * log10(right_energy / left_energy)` | wall on **right** ✓ |

**Files fixed:** `Library/EchoProcessor.py:154`, `SCRIPT_TrainEchoProcessor.py:366,412,432,517,522,545,550,820,828,848,858,880,934`, `SCRIPT_TestEchoProcessor.py:117,140`

---

## Issue 2 — Emulator `left_min`/`right_min` labels are swapped
**Severity: Minor — non-breaking but deceptive**

In both `SCRIPT_TrainEmulator.build_profile_features` and `Emulator.build_profile_features`,
the azimuth bins go from −30° (right) to +30° (left). But:

```python
half = steps // 2
left_min = np.min(p[:, :half], axis=1)  # bins 0–9 = right-side azimuths (mislabeled "left")
right_min = np.min(p[:, half:], axis=1) # bins 10–20 = left-side azimuths (mislabeled "right")
asym = left_min - right_min             # physically = right_min − left_min
```

`asym > 0` actually means the **right** wall is closer. Because the mislabeling is
identical in both the training script and the runtime `Emulator` class, the model receives
consistent features and training/inference is unaffected. But interpreting `asym` as
"left minus right" would give the wrong wall side.
**Status: FIXED**

`left_min`/`right_min` now correctly reflect physical sides (bins `:half` = right-side
azimuths → `right_min`; bins `half:` = left-side azimuths → `left_min`).
`asym = left_min - right_min > 0` now means right wall closer, consistent with IID convention.
`left_i`/`right_i` in local slope renamed to `prev_i`/`next_i`.

**Files fixed:** `SCRIPT_TrainEmulator.py:157–165`, `Library/Emulator.py:282–290`,
`SCRIPT_TrainEchoProcessor.py:684–686`, `SCRIPT_TestEchoProcessor.py:86–88`

---

## Issue 3 — Stale design comment: rotate1 is NOT fixed to zero
**Severity: Minor — documentation only**
**Status: FIXED**

Docstring updated to reflect that the policy controls both rotate1 and rotate2 (±90° each)
and that the history includes both previous rotations.

**File:** `SCRIPT_TrainPolicy_GA_HistoryNN.py:6`

---

## Issue 4 — Fragile wall-coordinate unit detection in EnvironmentSimulator
**Severity: Moderate — silently wrong for non-standard arena sizes**
**Status: FIXED**

`ArenaLayout._load_walls` uses heuristic thresholds to decide the unit of the wall
point cloud loaded from DataProcessor:

```python
if max_coord > 10000:      # → multiply by 1000 (treat as meters)
elif max_coord < 100:      # → multiply by mm_per_px (treat as pixels)
else:                      # → assume mm already
```

`DataProcessor.wall_x/wall_y` are already in mm (produced by `mask2coordinates` which
explicitly computes mm coordinates). For the current ~2400×1800mm arenas, `max_coord ≈ 2400`
falls in the "assume mm" branch — correct, but by coincidence. For arenas with max
coordinates > 10,000mm or < 100mm the code would silently corrupt wall positions.
The conversion is unnecessary and should be removed.

**File:** `Library/EnvironmentSimulator.py:136–147`

---

## Issue 5 — Wasted sonar queries inside `simulate_robot_movement`
**Severity: Minor — efficiency**
**Status: FIXED**

`Evaluator.episode` calls `self.sim.get_sonar_measurement(x, y, yaw)` for policy
decisions, then calls `self.sim.simulate_robot_movement(...)`. Inside that call,
`simulate_robot_movement` computes three additional sonar measurements (after rotate1,
after rotate2, after drive), which are returned in the state dict but never read by the
evaluator. Each emulator forward pass is not free; this triples the emulator calls per
step unnecessarily.

**Files:** `Library/EnvironmentSimulator.py:445–520`,
`SCRIPT_TrainPolicy_GA_HistoryNN.py:329`

---

## Issue 6 — Wrong azimuth direction in `robot2world` example comment
**Severity: Trivial — documentation only**
**Status: FIXED**

`DataProcessor.py:276`:
```python
azimuths = [0, 90, 180, 270]  # forward, right, backward, left
```
Based on the rotation math and `arctan2(rel_y, rel_x)`, positive azimuths are
counter-clockwise = **left**. Should read `# forward, left, backward, right`.
This directly contradicts the description two lines above it that correctly states
"90° = Up/Left".

**File:** `Library/DataProcessor.py:276`

---

## What was checked and confirmed correct

- **Channel reordering**: Client correctly reorders hardware ADC channels `[emitter, left, right]`
  before storing; `DataProcessor.load_sonar` correctly extracts columns 1 and 2 as `(left, right)`;
  EchoProcessor correctly treats col 0 as left, col 1 as right.
- **Profile azimuth convention**: `DataProcessor.get_profile` and
  `EnvironmentSimulator.ArenaLayout.compute_profile` use identical formulas and bin ordering.
  Training and simulation profiles are consistent.
- **Emulator feature augmentation**: `build_profile_features` code in the training script and
  in `Emulator.py` are identical; training and inference features match.
- **Normalization/denormalization**: EchoProcessor and Emulator both save and restore norm stats
  faithfully; calibration is applied consistently at both train and inference time.
- **Cross-environment GA training**: Sessions are correctly partitioned; empirical starts are
  correctly seeded per generation and per environment.
- **Collision detection**: `_compute_safe_endpoint` correctly marches in increments and halts
  before wall contact; IID deadband correctly suppresses steering output.

---

## Priority order

| Priority | Issue |
|---|---|
| ~~Fix soon~~ Done | #1 — IID sign convention unified to `locate_echo` standard (positive = right wall) |
| ~~Fix soon~~ Done | #4 — Fragile unit-detection heuristic removed; wall coords are always in mm |
| ~~Clean up~~ Done | #2 — `left_min`/`right_min` corrected; `asym` sign now consistent with IID |
| ~~Clean up~~ Done | #3 — Docstring corrected: policy controls both rotate1 and rotate2 |
| ~~Optimise~~ Done | #5 — `compute_sonar=False` added; training scripts skip 3 redundant emulator calls per step |
| ~~Trivial~~ Done | #6 — Azimuth example comment corrected: 90° = left, 270° = right |
