> **Do not edit without explicit user consent.**

> **Implementation:** the canonical implementation of this rationale lives in three files:
> `SCRIPT_TrainSonarModel.py` (sonar model training), `SCRIPT_TrainPolicy.py` (policy training),
> and `SCRIPT_RunPolicy.py` (deployment). Consistency checks should cover all three.

# Project Overview

Modelling a bat that learns to use its sonar system, with a specific focus on **vicarious learning**: a single generic sonar model, trained once on real sonar data, is used to mentally rehearse in any new arena so that a policy can be fit to that arena without the robot physically crashing in it.

The pipeline has three stages:

1. **Generic sonar model.** Train a single 3-slice distance model on pooled real sonar data (sessions B01–B05). The model maps a stereo sonar envelope to three (distance, σ) pairs covering the forward cone. It is trained once and reused across all downstream arenas.
2. **Arena specification.** For each target arena, extract its layout from an annotated overhead image into the same edge format used during sonar-model training, and define a target path through it. Together these specify what the robot should do in that arena.
3. **Arena-specific policy.** Using the trained sonar model as the sensor inside a geometric simulator of the target arena, train a policy by **behavioural cloning of a pure-pursuit teacher** that knows the target path. One policy per target arena.
4. **Deployment + cross-arena control.** Each policy is deployed on the real robot in its matched arena. Policies are then cross-swapped (policy_A in arena_B, and vice versa) as the primary control.

The central empirical claim is that the matched condition outperforms the swapped condition. A positive result simultaneously demonstrates (a) that policies are genuinely arena-specific and (b) that the simulator-driven vicarious adaptation is doing real work.

Real-data collection for the sonar-model sessions uses `SCRIPT_DataAcquisition.py`.

---

## Sonar Model

The sonar model predicts, from a stereo sonar envelope, the minimum wall distance (with predictive σ) in each of three angular slices of a forward cone. It is trained **once** on pooled data from `EmulatorTrainingData/` and held fixed across all downstream policy-training runs. Its generality is what makes vicarious learning possible.

### Geometry and slices

- **Forward cone:** ±`cone_half_deg` around boresight (currently 35°).
- **Slices:** the cone is split into three equal angular thirds — `right`, `center`, `left` — each `2/3 × cone_half_deg` wide. Names follow the project `+az = LEFT` convention: the `right` slice covers the most-negative-azimuth third (robot's physical right), the `left` slice covers the most-positive third.
- **Profile (ground truth at training/sim time):** a 1-D array of wall distances sampled at evenly-spaced azimuth bins covering `opening_angle` (currently 270°) over `profile_steps` bins (currently 90), centred on the robot's heading. The per-slice training target is the minimum wall distance whose bin centre falls within that slice. `profile_method` (`ray_center` or `min_bin`) controls how each profile bin is computed from the wall point cloud.

All of these parameters are recorded in `SonarModel/slices_feature_params.json`. The simulator and policy-training pipeline read them from that file, so retraining the sonar model with different geometry automatically propagates downstream.

### Architecture (`SonarSlicesUQ`)

A single 1-D CNN backbone embeds the left and right sonar envelopes independently into feature vectors `z_L` and `z_R`. Three heads then produce (mean, log σ²) per slice:

- **Center heads** operate on `(z_L + z_R) / 2`. By construction, swapping L and R leaves the center prediction unchanged.
- **Side heads** share weights and are applied with channels swapped:
  - `right_*` receives `concat(z_L, z_R)`  (zL-emphasized → trained on the negative-azimuth slice)
  - `left_*`  receives `concat(z_R, z_L)`  (zR-emphasized → trained on the positive-azimuth slice)
  Swapping the L and R sonar inputs exactly swaps the left and right predictions.

This bakes in the physical L/R symmetry of the robot. Each head is a small 2-layer MLP.

### Envelope normalisation

Each ping's L and R envelopes are min-max scaled per channel to a fixed range (`[0, 1]`) along the time axis before they reach the conv stack. Applied identically in `SCRIPT_TrainSonarModel.load_data` and inside `SonarModel.predict_from_envelope` at inference; the chosen mode is recorded in `slices_feature_params.json` under `envelope_norm` so train and deploy paths can't drift apart. Legacy feature_params files without that field default to a no-op for backward compatibility.

The motivation is sim-to-real robustness on the absolute amplitude of the receive envelope. The emit-pulse peak and the post-pulse signal level both depend on battery state and analog-receive-path drift; we observed empirically that the same robot in the same arena can produce envelope peaks at 25 k counts on one day and 28–30 k counts on another, with no firmware or arena change. With raw amplitudes feeding the conv stack, this between-session gain drift pushes inputs out of the model's training distribution and degrades predictions sharply. Per-ping normalisation removes absolute amplitude as a cue — the conv stack reads only envelope *shape* — and the model becomes invariant to receive-path gain.

The upper reference for the rescale is the max within the first `ref_window` samples of the envelope (currently 10), not the global max. This anchors the scale to the **emit-pulse region**, which is always present at the start of every ping. Anchoring on the global max would silently mis-scale pings where a strong close-wall echo exceeds the emit-pulse peak — those would be normalised differently from "normal" pings even though their underlying signal interpretation should be consistent. With emit-anchored scaling, post-emit echoes simply produce normalised values proportional to their amplitude relative to the emit drive (sometimes > 1 for very close walls), giving the conv stack a consistent feature scale. The lower reference is the global min (the noise floor).

The trade-off is loss of an absolute-distance hint that lives in raw amplitude (closer wall ↔ stronger echo, by spreading-loss). Empirically, echo *timing* is a much stronger distance cue than raw amplitude, so this loss is small; whether it's negligible is testable by comparing held-out NLL between a normalised SonarModel and a non-normalised one.

### Loss and training

For each slice the loss is **Gaussian negative log-likelihood**:
`0.5 × (log σ² + (μ − target)² / σ²)`,
with log-σ² clamped to `[LOG_VAR_MIN, LOG_VAR_MAX]`. The first `WARMUP_EPOCHS` use plain MSE on the means only, then training switches to NLL so the σ heads can fit residual scale without dragging the means around at initialisation.

Training data is split by **quadrant** per session: one quadrant per session is held out for validation (≈25% of each session), and the held-out set defines both the val-NLL early-stopping signal and the empirical σ_sim lookup described below.

### Two inference paths from one model

`Library/SonarModel.py` wraps the trained checkpoint and exposes two callable interfaces with the same six-key output (`distance_{L,C,R}_mm`, `sigma_{L,C,R}_mm`):

- **`predict_from_envelope(L, R)`** — used at deployment. The real sonar envelope is z-scored using the saved training mean/std and run through the network; σ is the network's heteroscedastic per-ping prediction.
- **`predict_from_profile(profile, rng)`** — used inside the simulator. The geometric per-slice minimum distance is taken from the profile and corrupted with `N(0, σ_sim_slice(d_true))`. `σ_sim` for each slice is a linear interpolation through per-bin empirical residual standard deviations measured on the validation split, stored in `feature_params.json`.

The σ in the two paths is **different by design**: at sim time we don't have a sonar input to condition on, so we use the marginal residual scale; at deploy time the network's per-ping σ is more informative and is used directly. They agree only insofar as the σ heads have correctly learned the marginal distance-conditional residual scale.

### Why σ, why three slices

- σ lets the policy reason about confidence (it can ignore high-σ readings or weight them less). In practice the current policy is configured with `use_sigma=False` and the σ outputs are not consumed — but the model is trained with them so the option is on the shelf.
- Three slices give the policy a coarse left/center/right asymmetry signal without exposing it to the full, high-dimensional profile. Cheapest possible spatial input that still preserves "is the obstacle on my left or my right?".

### Diagnostics

Per-slice scatter and σ calibration plots, an L/C/R collapse check, an "overall min" comparison against a single-output distance model, and per-slice σ_sim fits are written alongside the checkpoint. Val-set NLL is the standalone diagnostic used to flag a bad sonar model when downstream policy results are weak.

---

## Arena Specification

Each target arena is specified by:

1. An **annotated overhead image** processed into wall edges via `SCRIPT_BuildArenaGeometry.py`. The wall-cloud format is the same one used for the sonar-model training sessions.
2. A **target path** (`TargetArenas/<arena>/target_path.json`) — a closed polygonal loop of waypoints in arena (x, y) mm coordinates that the robot is expected to follow. Loaded and densified by `Library/TargetPath.py`.
3. A **release box and direction arrow** inside the path JSON, specifying the rectangle and yaw from which the real robot is released for an experimental run. Used both for training start-pool sampling and for matched real-world deployment conditions.

Target arenas live under `TargetArenas/`, one subfolder per arena, and are independent of the sonar-model training sessions. Multiple arenas can be specified to support cross-arena comparison.

---

## Simulator

`Library/EnvironmentSimulator.py` couples the loaded `SonarModel` with an `ArenaLayout` (wall point cloud + arena bounds). It exposes:

- `get_sonar_measurement(x, y, yaw)` — geometry → profile → noisy 6-key sonar dict.
- `simulate_robot_movement(...)` — applies a `(rotate1, rotate2, drive)` action with collision-aware drive. `rotate1` is included for compatibility but the current policy pipeline always sets it to 0 (single rotation per step). The drive is collision-checked against both the wall point cloud (with `robot_radius_mm` clearance) and the arena boundary; if the segment is blocked the robot stops at the safe endpoint and the step is flagged `collision.drive_blocked=True`.
- `reseed(seed)` — resets the σ_sim noise generator so a rollout is fully reproducible from a single integer seed.

All sonar geometry parameters (cone, profile, σ_sim) are pulled from the loaded `SonarModel`. The simulator has no parallel config of its own.

---

## Policy

For each target arena, a separate policy is trained inside the simulator instantiated with that arena's edges. Policies are not shared between arenas.

The policy is a **vanilla RNN** with hidden size 32. Per step it produces a single rotation (in degrees), clamped to `±max_rotate_deg`. After this rotation the robot drives a fixed `fixed_drive_mm` (currently 125 mm) — the inter-call distance is calibrated separately and is not adjusted by the policy.

### Step sequence (training and deployment)

1. Take a sonar measurement at the current pose.
2. Form the observation vector and feed it (with the previous hidden state) to the RNN.
3. Read the rotation output. Rotate the robot in place by that amount.
4. Drive forward by `fixed_drive_mm`.
5. Repeat.

The recurrent hidden state carries information across steps; there is no explicit history buffer, no two-phase look/move, and no head–body separation.

### Observation vector

Canonical layout, all components scaled to roughly [-1, 1]:

- `distance_left_mm  / max_dist_mm`
- `distance_center_mm / max_dist_mm`
- `distance_right_mm / max_dist_mm`
- (optional) `sigma_left_mm / max_sigma_mm`, `sigma_center_mm / max_sigma_mm`, `sigma_right_mm / max_sigma_mm` — included only when `cfg.use_sigma=True`.
- `prev_rot_deg / max_rotate_deg`

Distances are **clamped** to `[min_dist_mm, max_dist_mm]` before scaling (`min_dist_mm` reflects the minimum range the real sonar can return — anything closer saturates). σs are clamped to `[0, max_sigma_mm]`. These same clamps apply at training and deployment; if they drift apart the policy will see out-of-distribution inputs on the real robot.

`prev_rot` is the **commanded** rotation from the previous step (what the policy or teacher asked for), not the motor-noisy executed rotation. The robot knows what it asked, not what its motors did.

### Teacher: pure pursuit

The training teacher is **pure pursuit** along the target path:

1. Project the current `(x, y)` onto the closest segment of the densified path.
2. Advance `teacher_lookahead_mm` (currently 200 mm) along the path in arc-length order.
3. Take the rotation that points the robot's current yaw at that lookahead point, clamped to `±max_rotate_deg`.

Direction along the loop is fixed by arc-length order, so a robot starting "the wrong way" is commanded to U-turn at first. The teacher's target is purely a function of `(x, y)` — yaw-independent — which makes the teacher field a clean 2-D vector field over the arena.

### Training: behavioural cloning + DAGGER-lite

Behavioural cloning of the teacher, with masked MSE loss + BPTT through the full episode.

To prevent compounding errors from off-path drift, three noise sources are injected during rollout:

- **Teacher perturbation** (`teacher_perturb_prob=0.30`, σ=30°): with some probability, the rotation actually executed in the simulator is the teacher's rotation plus Gaussian noise. **The training label remains the clean teacher rotation** — only the commanded rotation is noisy. This generates recovery examples in the dataset (DAGGER-style).
- **Per-step motor execution noise** (3° σ on rotation, 5 mm σ on drive): zero-mean Gaussian added on top of the commanded rotation/drive before the simulator step, but `prev_rot` fed back to the policy stays at the commanded value. This is the high-frequency component of sim→real motor imperfection — averages out within ~10 steps.
- **Per-episode kinematic gain bias** (`motion_rot_gain_range_pct=0.15`, `motion_drive_gain_range_pct=0.05`): one `rot_gain ∼ U(1−0.15, 1+0.15)` and one `drive_gain ∼ U(1−0.05, 1+0.05)` are sampled at episode reset and applied multiplicatively to every commanded action for the whole rollout (`actual = commanded × gain + per-step noise`). This is the *systematic* sim→real component the per-step noise can't capture: the real robot exhibits a sustained ~10 % rotation gain mismatch and ~5 % drive gain mismatch (measured by `SCRIPT_CalibrateRotation.py` and per-step displacement on `SCRIPT_RunPolicy.py` traces). With only zero-mean per-step noise, those biases compound undetected; with a per-episode bias the policy is forced to use sonar feedback to discover and compensate within an episode. The samples are logged to `<output_dir>/motion_noise_log.tsv` for verification.

Episodes terminate at `max_steps` (currently 150) or when the simulator reports a wall collision.

### Starting positions

Training uses a pool of starts inside the path's `start_box` with yaw drawn from the box's `start_arrow` direction plus Gaussian noise (10° σ). The disjoint train/val split is over this pool, so val tests on unseen starts but within the same release box (matching the real-robot experimental setup).

This is a deliberately narrow start distribution — it matches the experimental release condition. A policy that fails far outside the box is not necessarily broken; it has simply not been trained for that state. If wider robustness is needed, the start pool should be widened or a curriculum added.

### Saved policy

Each best-on-val checkpoint is written to `PolicyTraining/<condition>/best_policy.json` with everything needed to rebuild the obs vector and run the RNN forward pass:

- `obs_layout` — names of the obs vector components, in order.
- `genome` — flat array, ordered `[W_xh, W_hh, b_h, W_hy, b_y]`, reshaped using `hidden_size` and `in_dim`.
- `hidden_size`, `in_dim`, `out_dim`, `use_sigma`, `max_rotate_deg`, `fixed_drive_mm`,
  `min_dist_mm`, `max_dist_mm`, `max_sigma_mm`.

Deployment must read the clamps and obs layout from this file rather than re-hard-coding them.

### Architecture choice

A vanilla RNN with explicit BPTT was chosen over an MLP-with-buffered-history because:

- The recurrent hidden state is a more compact representation of arbitrary-length history than a fixed-size buffer.
- Path following needs implicit localisation along the loop ("how far around am I?"), which the recurrent state can carry naturally.
- Training is supervised (BC), so the GA-friendly considerations that motivated MLPs in earlier prototypes do not apply.

---

## Deployment on the Real Robot

The trained policy is run on the real robot in its arena via `SCRIPT_RunPolicy.py`. The same data-collection and processing pipeline as `SCRIPT_DataAcquisition.py` produces the per-step `sonar_package`; the L and R envelopes are passed through `SonarModel.predict_from_envelope` to obtain 3-slice distances and σs, exactly as during training.

### Per-step sequence

1. Take a sonar measurement (`predict_from_envelope` on the real envelope).
2. Apply the same clamps as training (`min_dist_mm`, `max_dist_mm`, `max_sigma_mm` from `best_policy.json`).
3. Form the observation vector (same layout as training) and run one RNN step from the carried hidden state.
4. Physically rotate the robot's body by the policy's output (clamped to `±max_rotate_deg`).
5. Drive forward `fixed_drive_mm`.
6. Set `prev_rot = commanded rotation`. Continue.

There is no IID symmetry wrapper, no explicit history buffer, and no two-phase look/move — those concepts belonged to an earlier prototype and are not part of the current pipeline.

### History initialisation

The RNN hidden state starts at zero. Training does the same.

### Stopping condition

Fixed `max_steps`, configurable per run; manual interrupt is also supported.

### Data logging (per step)

- `sonar_package` (full, as returned by `client.read_and_process`)
- 6-key sonar prediction (3 distances + 3 σs) before and after clamping
- Position from external tracker: `x`, `y`, `yaw_deg`
- Commanded rotation, drive distance
- Hidden state (optional, useful for offline analysis)

---

## Acquisition-Protocol Comparison (Planned)

The current `SCRIPT_DataAcquisition.py` collects sonar training data while the robot runs a **sonar-IID-driven** obstacle-avoidance policy: distance, IID, and a small turn-probability decide each action. The pose distribution that lands in the training set is therefore whatever those dynamics produce — predominantly mid-corridor, walls ≈ 0.8–1.5 m away, with close-wall and corner configurations systematically avoided. Per-step diagnostics on early deploy runs (`PolicyRuns/.../step_metrics_analysis.png`) show that close-wall and corner poses are exactly where the trained SonarModel's residuals and σ are largest. The model is weakest on the configurations the acquisition policy never visits.

A planned alternative uses the **overhead tracker plus the annotated arena map** to actively steer the robot. The simulator's `compute_profile` already supplies geometric truth at any pose; a tracker-feedback nav controller can drive the robot through a deliberately chosen sequence of (x, y, yaw) poses, including ones close to walls, in corners, and through narrow gaps. At each pose the robot rotates in place and pings at multiple yaws, providing direct supervision on rotation invariance from a fixed location. Three protocol variants form a controlled comparison:

- **A — sonar-IID-driven** (existing `SCRIPT_DataAcquisition.py`). Avoidance algorithm picks actions from sonar.
- **B — vision-coverage-driven** (new acquisition script). Tracker + arena map picks actions to maximise coverage of pose / profile-feature space, deliberately targeting under-represented configurations.
- **C — random teleport** (control). Pose chosen uniformly at random from feasible space each step. Isolates "deliberate non-IID exploration" from the simpler "any non-IID protocol."

All three share arena, hardware, supervisory signal (`compute_profile` against the arena map), SonarModel architecture, training pipeline, and total number of pings. The only manipulated variable is the pose distribution. Pre-registered metrics: held-out NLL on a fixed test set spanning all configuration classes (open / corridor / corner / narrow-gap / very-close-wall), per-configuration-class error, σ calibration on held-out, and downstream policy performance at deploy time.

**Scientific framing.** This implements **active cross-modal calibration**: vision (the overhead tracker plus the arena map) supplies *both* the supervisory targets and the exploration policy that gates which acoustic configurations the agent encounters. The biological analogy is a developing or environmentally-perturbed bat using non-sonar cues to deliberately expose itself to under-sampled acoustic scenes, accelerating sonar-interpretation learning. The substantive empirical claim is the size of the gap **on configurations both A and B cover** (open arena, normal corridors); a gap on corner / close-wall configurations alone is partly tautological because A by construction doesn't visit them.

---

## Cross-Arena Control

The primary baseline against the matched (arena_X policy in arena_X) condition is a **cross-arena swap**: deploy policy_A in arena_B, and policy_B in arena_A. Run the same data-logging and metrics as the matched deployments, so the two conditions are directly comparable.

If matched outperforms swapped, the experiment simultaneously supports two claims:

1. Policies are arena-specific — there is real structure that a generic one-size-fits-all policy would miss.
2. The simulator-driven BC captures enough of that structure to produce arena-specific tuning without on-robot training.

If matched does **not** outperform swapped, the interpretation depends on the diagnostics: weak sonar model (val NLL is poor / σ poorly calibrated), insufficiently arena-specific teacher field, or BC instability are the main candidates to check.

With three or more target arenas, all off-diagonal swaps can be run to strengthen the design.

---

## Out of Scope (Parked)

- **Burst policy variant** (multiple within-burst measurements per step) — biologically interesting but shelved until the vicarious-learning story is established.
- **Mapping / pose-graph SLAM / spatial-information analyses** — scripts on `new_ideas` (`SCRIPT_PoseGraphSLAM*.py`, `SCRIPT_TakeEnvSnapshot.py`) target a later phase where the robot builds its own spatial representation rather than receiving an annotated arena image.
- **GA-trained policies, IID-emulator dual-head architectures, two-phase look/move, fixed-size history buffers, hall-of-fame** — earlier-prototype machinery that has been replaced by the BC + RNN + 3-slice-distance pipeline described here.
