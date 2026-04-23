> **Do not edit without explicit user consent.**

> **Implementation:** the canonical implementation of this rationale lives in three files:
> `SCRIPT_TrainEmulator.py` (emulator training), `SCRIPT_TrainPolicy.py` (policy training),
> and `SCRIPT_RunPolicy.py` (deployment). Consistency checks should cover all three.

# Project Overview

Modelling a bat that learns to use its sonar system. Data was collected with a robot equipped with a body-fixed sonar system in various arenas, using `SCRIPT_DataAcquisition.py`.

---

## Emulator

The emulator predicts, from a geometric profile of the local arena, the sonar readings (IID and distance) the robot would receive at that position and heading. It is a single 1D CNN with two regression heads — one for IID and one for distance — both trained on echo-present samples only. Echo presence is not predicted explicitly.

### Profile (input)

A profile is a 1D array of wall distances sampled at evenly-spaced azimuth angles centred on the robot's current heading. The same profile is the input to both CNN heads.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `opening_angle` | 220° | Broad enough to give the CNN lateral context, including walls that influence IID even outside the sonar's forward beam |
| `resolution` | 11°/bin | 220° / 20 bins — fine enough to resolve spatial variation, compact enough to keep input small |
| `profile_method` | `min_bin` | Also available: `ray_center` (see `DataCollection.load_profiles()`) |

Distances in the profile are in **mm**.

### IID and distance (shared CNN, two heads)

**Targets:**
- `corrected_iid` from the sonar package — a single float per step, in dB. IID > 0 means wall on the right; IID < 0 means wall on the left.
- `corrected_distance` from the sonar package — in mm.

The full 220° profile (20 bins) is used as input to both heads. A broad profile is necessary because IID encodes the lateral asymmetry of the environment, which requires context beyond the narrow forward beam. The distance head uses the same input; the CNN learns which parts of the profile are relevant for each output.

**Profile normalisation:** each profile is divided by `max_dist_mm` (a fixed global scale), then z-scored using training-set per-bin mean and std. This preserves absolute distance information across profiles.

> **Do NOT normalise each profile by its own mean (per-profile normalisation).** This is acoustically incorrect: IID magnitude depends on absolute distance, not just the shape of the profile. A profile at a sufficiently large distance produces IID ≈ 0 regardless of its shape, because the sonar echoes are too weak to carry lateral information. Per-profile normalisation would make all such distant profiles look identical to nearby ones, destroying the distance cue the CNN needs.

**Model architecture:**

```
Conv1D(in=1,  out=32, kernel=5, padding=2) → ReLU   ┐
Conv1D(in=32, out=64, kernel=5, padding=2) → ReLU   ┘ shared backbone
AdaptiveAvgPool1d(8) → flatten → FC(64×8=512 → 64) → ReLU

→ FC(64→32) → ReLU → FC(32→1)   [IID head: predicted IID in dB]
→ FC(64→32) → ReLU → FC(32→1)   [distance head: predicted distance in mm]
```

**Loss:** `IID_MSE + dist_loss_weight × distance_MSE`, computed on echo-present samples only. Both targets are z-scored independently (using echo-present training statistics) before computing MSE, so the two losses are in comparable units.

### Echo presence

Echo presence is **not** predicted by the emulator. Samples with `corrected_distance ≥ max_dist_mm` are excluded from training (their distance and IID values are sentinel values set by `AcousticProcessing`, not real measurements). At inference the networks simply extrapolate: a flat, far-away profile naturally produces a large predicted distance and near-zero predicted IID, which is the correct behaviour.

### Symmetrised inference

By physical definition, mirroring a profile horizontally (left↔right) must negate the IID and leave the distance unchanged. A CNN trained on finite data may not learn these (anti)symmetries exactly. To enforce them at inference time, each profile is passed through the network twice — once as-is and once flipped — and the outputs are combined:

```
iid_db   = (iid_normal  − iid_flipped)  / 2   # antisymmetric under flip
distance = (dist_normal + dist_flipped) / 2   # symmetric under flip
```

This guarantees the physical constraints are satisfied regardless of any residual asymmetry in the trained weights, at the cost of doubling inference time.

### Data and train/validation split

**Data source:** sessions B01–B05, loaded via `DataCollection`. Each data point pairs a profile (computed from the arena geometry at the robot's logged position and heading) with the `corrected_iid` and `corrected_distance` from the sonar package recorded at that step.

**Train/validation split** is quadrant-based to test spatial generalisation: the emulator is trained on data from most spatial regions and validated on held-out regions it has not seen, simulating positions the policy will visit that were not covered during data collection.

The split is specified as a dict mapping session name to a list of quadrant indices (0–3) to withhold for validation. All remaining data from all sessions goes to training. Example:

```python
validation_quadrants = {
    "sessionB01": [3],
    "sessionB03": [3],
}
```

This withholds ~12.5% of data per listed session while keeping all other sessions' data (including all quadrants of unlisted sessions) in training.

**Evaluation:** report Pearson r and RMSE for IID and distance separately on (1) training data (echo-present) and (2) held-out validation quadrants (echo-present).

---

## Policy: Architecture and Step Sequence

Using the emulator, a policy is trained to control the robotic bat. The policy is a neural network that produces two rotations per step — **rotation 1** and **rotation 2**.

This models a bat's ability to measure in a different direction (via head rotation) than the direction of flight.

### Step sequence

**Phase 1 — look:**

1. Present the policy with *n* previous values of:
   - sonar distance (emulator output for the *n* previous steps + 1 zero)
   - sonar IID (emulator output for the *n* previous steps + 1 zero)
   - rotation 1 (the *n* previous values + 1 zero)
   - rotation 2 (the *n* previous values — no zero appended, because rotation 2 from the previous step is already available)
2. Network produces **rotation 1**.
3. Robot rotates by rotation 1 degrees.
4. In this orientation, a sonar measurement is taken: the local profile is extracted and passed to the emulator to obtain IID and distance.

**Phase 2 — move:**

5. Present the policy with *n* previous values of:
   - sonar distance (previous *n* steps + new distance)
   - sonar IID (previous *n* steps + new IID)
   - rotation 1 (previous *n* values + new rotation 1)
   - rotation 2 (previous *n* values)
6. Network produces **rotation 2**.
7. Robot rotates by rotation 2 degrees.
8. The robot's new heading is `original heading + rotation1 + rotation2` (the net body rotation per step is the sum of both rotations).
9. Robot drives straight for a fixed distance. This distance should be calibrated to match the intercall distance observed in bats.

---

## Policy: IID Bilateral Symmetry

The policy network is trained exclusively in a canonical **"wall-on-right" frame**: IID values presented to the network are always non-negative. When the physical IID is negative (wall on left), the IID sign is flipped before feeding the network and the output rotation is negated, so the robot still turns the correct physical direction. History is always stored in this canonical frame (absolute IID, rotations reflected accordingly).

This symmetry wrapper is applied identically during training (in the simulator) and during deployment (on the real robot). Without it, the GA can find a degenerate solution that always turns in one direction regardless of sonar input.

---

## Policy: History Initialisation

The policy uses a history buffer of *n* steps. At episode start this buffer contains no real sonar data. Two naive approaches both cause problems:

- **Zero-fill:** the network can detect "all zeros = episode start" and behave differently at the start of an episode than mid-episode, which will not generalise well.
- **Random noise fill:** the network learns that early history is untrustworthy and may learn to discount it, wasting the history mechanism.

**Adopted solution — honest zero-fill:** both training and deployment initialise the history buffer to zeros. This is the truthful representation of episode start — the robot genuinely has no prior information — and is fully consistent between training and deployment. If the policy learns a distinct start-of-episode behaviour, that is legitimate: it is in a genuinely different situation at the start.

> **Open question:** is zero-fill actually the best strategy? The alternative — random pre-fill in training — prevents the network from exploiting the start-condition cue, at the cost of giving it dishonest history. Whether the policy benefits from, or is harmed by, knowing it is at the start of an episode is an empirical question worth revisiting.

---

## Policy: Training with a GA

The policy is trained with a genetic algorithm (GA), assessed on two criteria: (1) paths should be smooth, and (2) crashing should be rare.

### Fitness function

1. Compute the centroid (*x_c*, *y_c*) of all positions visited during an episode.
2. Divide 360° into angular bins of width *W* degrees (e.g. 10°).
3. For each bin, find all path points whose angle from the centroid falls in that bin and record the mean distance from the centroid. If no points fall in a bin, its distance is 0.
4. The **raw fitness** is the mean of these per-bin mean distances across all bins. This rewards paths that are consistently far from the centroid in every direction. Using the mean (not max) per bin prevents a degenerate strategy where the robot shoots briefly to the walls in each direction and returns to the centre — the robot must spend sustained time far from the centroid to score well.
5. If the episode ends in a collision, the raw fitness is multiplied by a **collision discount factor** (< 1).
6. A **survival** factor penalises early termination:

```
survival = steps_survived / max_steps
```

7. A **jitter penalty** is applied multiplicatively to enforce smooth paths:

```
jerk_t         = |(rotate1_t + rotate2_t) − (rotate1_{t−1} + rotate2_{t−1})|   # physical turns
jitter_factor  = 1 − w_smooth × mean(jerk_t) / max_possible_jerk
fitness        = coverage × survival × jitter_factor × collision_discount
```

where `max_possible_jerk = 2 × max_net_rotation_deg` (the maximum heading reversal per step, governed by the net-rotation cap described below). A smooth arc has low mean jerk; left-right oscillation has high mean jerk. Consistent wall-following turns are not penalised — only reversals are. `w_smooth` controls penalty strength (0 = disabled, 1 = full weight).

### Net rotation cap

The net body rotation per step is hard-capped: `|rotate1 + rotate2| ≤ max_net_rotation_deg`. After the policy produces `rotate2`, it is clipped so the cap is not exceeded. This prevents the robot from spinning excessively in a single step, regardless of what the policy outputs. `max_possible_jerk` in the jitter penalty is defined relative to this cap.

### Distance floor (sonar saturation)

The real sonar cannot return distances below **300 mm** — any wall closer than this produces a saturated reading of 300 mm. To keep training consistent with deployment, the emulator's predicted distance is clamped from below at `min_dist_mm = 300.0` mm before being passed to the policy or stored in history:

```python
dist_mm = max(cfg.min_dist_mm, min(emulator_distance_mm, cfg.max_dist_mm))
```

Without this floor, the emulator may predict sub-300 mm values when the simulated robot gets close to a wall (geometrically correct for the simulation, but unreachable on the real robot). The policy would then learn to react to distance values it will never observe during deployment. Clamping ensures that 300 mm is treated as a saturated "at or closer than minimum range" signal, consistent with its meaning on the real robot.

### IID noise injection

During training (fitness evaluation and trajectory plotting), Gaussian noise with std `iid_noise_db` is optionally added to the emulator's IID output before it is fed to the policy. This prevents the policy from overfitting to the emulator's exact IID values and encourages robustness to the measurement noise present on the real robot. Set `iid_noise_db = 0` to disable.

### Crash-start pool

To prevent the GA from ignoring collision-prone situations, a **crash-start pool** is maintained throughout training. When a trajectory episode ends in a collision, the robot's position and heading `crash_backtrack_steps` steps before the crash are added to the pool. In each subsequent generation a fixed number of guaranteed episodes are started from pool positions (in addition to randomly sampled starts), forcing every genome to face previously lethal situations.

Pool entries are **retired** after each generation: the current best-genome policy is re-run from each pool start, and any start it now navigates without crashing is removed. This prevents stale easy-to-solve starts from accumulating and keeps the pool focused on genuinely difficult positions. The pool is capped at `max_crash_starts_per_session` entries per session.

---

### Architecture choice

The policy is an **MLP** (not an RNN). An RNN was considered but rejected: although it has fewer parameters, each parameter has compounding effects across time steps, making the GA fitness landscape more rugged. The MLP with explicit history has a smoother, more GA-friendly landscape and maps cleanly onto the input structure described above.

The network has a **single output neuron** used for both rotation 1 and rotation 2. A two-output variant was considered but rejected: because the inputs to the two calls are systematically different (zeros vs. actual measurements in the current slot), the same function can produce meaningfully different values for r1 and r2 without needing separate output weights. Keeping one output neuron reduces genome size, which directly benefits GA search.

---

## Policy: Variation of History Length

To understand how much the policy benefits from memory, we train separate policies for several values of `history_len` (e.g. 1, 3, 5, 10). The goal is best performance at each history size, not a fair comparison between equally-sized networks, so the network is allowed to scale naturally with history.

Memory and head–body separation are treated as a coupled pair: the baseline has neither, and all history policies (`history_len > 0`) have both. This coupling is principled — a decoupled head is only useful if the robot can remember where it looked and what it found across steps. Without memory, a free head simply collapses to a GA-optimised fixed look angle, which adds no adaptive value.

### Network scaling with history

The input dimension is `4 * history_len + 3` (or `4 * history_len + 2` when `include_r1_in_input=False`), so it grows with history length. The hidden layer sizes are kept fixed across all runs. The expressivity bottleneck for short-history policies is the lack of temporal information, not network capacity, so scaling hidden sizes with history is not expected to help.

### Baseline: history_len=0, include_r1_in_input=False, force_aligned=True

The baseline agent has no memory and no head–body separation. With `force_aligned=True`, rotate1 is always 0: the head is fixed to the body and the sonar always measures straight ahead. The single network call (Phase 2) receives only `[dist_current, iid_current]` and produces the total body rotation for that step.

The baseline therefore learns only how to scale its body rotation as a function of the current sonar reading — nothing more. It is the correct lower bound for comparison with memory-augmented policies.

`r1_current` is excluded from the Phase 2 input (`include_r1_in_input=False`) for consistency with the rest of the framework, though it has no effect here since rotate1 is always 0.

### history_len = 1

`history_len = 1` is **not** a reactive baseline — it has one step of memory. Tracing through `build_input`:

- **Phase 1 (look):** input is `[prev_dist, 0, prev_iid, 0, prev_r1, 0, prev_r2]` — the network uses the previous step's measurement to decide where to look.
- **Phase 2 (move):** input is `[prev_dist, dist_current, prev_iid, iid_current, prev_r1, r1_current, prev_r2]` — both the previous and current measurements are available.

This is the minimal policy that makes meaningful use of the two-phase structure with memory.

### Hall of fame

The top-N genomes seen across all generations are retained in a **hall of fame** (`top_policies/rank001.json` … `rankNNN.json`). This provides a pool of high-quality policies from different points in training, which is useful for post-hoc analysis and deployment: the overall best policy may appear at any generation, not just the final one.

### Multi-run training

`SCRIPT_TrainPolicy.py` loops over `HISTORY_LENGTHS`. A value of 0 trains the baseline (`PolicyTraining/<condition>_h00`); any other value trains the standard config for that history length (`PolicyTraining/<condition>_h01`, `PolicyTraining/<condition>_h10`, etc.). All other settings (GA parameters, fitness function, architecture hidden sizes) are identical across runs.

---

## Policy: Burst Variant (exploratory)

Implemented in `SCRIPT_TrainPolicy_Burst.py`. The motivation is bat echolocation, where calls are grouped into bursts — a rapid volley of pulses fired in quick succession. The inter-pulse interval within a burst is too short for meaningful movement between calls, so the entire burst is best modelled as multiple measurements taken at the same position.

### Step sequence (burst)

Each step takes `N_LOOKS` sonar measurements before committing to a drive direction:

1. **Look planning:** Build input from history only (current slots zeroed) → MLP → `N_LOOKS` look angles + 1 drive output. The look angles are all planned simultaneously before any measurement is taken, modelling the pre-programming of a burst volley.
2. **Measurement:** Take `N_LOOKS` sonar measurements at the planned look directions (all from the same position).
3. **Drive planning:** Build input from history + all `N_LOOKS` measurements → same MLP → drive rotation r2.
4. **Execute:** Body rotates by r2, drives forward fixed distance.

### IID symmetry (two flips)

Because look angles are planned before measurement, the canonical frame for look planning must use the previous step's final IID sign (`flip_look`). The drive rotation uses the current step's final measured IID sign (`flip_drive`), matching the original policy's behaviour for r2. History stores canonical values throughout.

### Architecture

The MLP has `N_LOOKS + 1` outputs (one per look angle plus r2). Input dimension: `3 * N_LOOKS * (history_len + 1) + history_len`. This is larger than the single-policy input, making GA search harder for the same population size and generation count.

### Status

Exploratory. Early results suggest the burst policy converges more slowly than the single-look policy (larger genome) and reaches a slightly higher final collision rate. Whether the richer within-step information from multiple looks translates to better coverage or collision avoidance at longer training budgets is an open question.

---

## Deployment on the Real Robot

After training, the policy is applied on the real robot using a script similar to `SCRIPT_DataAcquisition.py`. The same sonar data collection and processing pipeline is used, yielding a `sonar_package` per step; `corrected_iid` and `corrected_distance` from that package are fed into the trained policy.

### Physical rotation sequence (each step)

1. Decide rotate1 using the policy (with IID symmetry wrapper applied to last step's IID).
2. Physically rotate the robot's body by rotate1 degrees.
3. Take a sonar measurement (`corrected_iid`, `corrected_distance`).
4. Decide rotate2 using the policy (with IID symmetry wrapper applied to current IID).
5. Physically rotate the robot's body by rotate2 degrees.
6. Drive forward a fixed distance calibrated to match bat intercall distances.
7. Append canonical (flipped) values to the history buffer.

> Because the sonar is body-fixed, rotate1 is a genuine physical rotation — it is not a virtual "look direction" as in the emulator-based simulator.

### IID symmetry wrapper (identical to training)

If `corrected_iid < 0` (wall on left): pass `abs(corrected_iid)` to the network and negate the output rotation to obtain the physical rotation. Store `abs(corrected_iid)` and the canonical (negated) rotation in the history buffer.

### History initialisation

Initialise the history buffer to zeros, consistent with how training episodes are initialised.

### Stopping condition

The robot runs for a fixed number of steps (`max_steps`, a configurable parameter). The script can also be interrupted manually.

### Data logging (per step)

- `sonar_package` (full, as returned by `client.read_and_process`)
- position from tracker: *x*, *y*, `yaw_deg`
- rotate1, rotate2, net rotation (rotate1 + rotate2)
- drive distance
- Additional fields can be added as needed.
