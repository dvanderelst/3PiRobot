# Pipeline Simplification: Removing the EchoProcessor

## Decision
Remove the EchoProcessor CNN from the sonar processing pipeline and train the
Emulator directly on `corrected_iid` and `corrected_distance` from the sonar package.

## Background
The EchoProcessor was introduced as a "smart threshold" to suppress phantom echoes
(likely floor reflections caused by incorrect sonar tilt) and recover the true wall
echo distance and IID.  Accuracy checks (SCRIPT_CheckRobotAccuracy.py) comparing
sonar distance against closest visual distance from the arena profile revealed:

- Sessions B01, B02, B03: clean data, threshold-based distance tracks visual distance
  well.  The simplified pipeline works reliably.
- Sessions B04, B05: many phantom echoes (sonar distance << closest visual distance).
  Points well below the 1:1 line indicate spurious early returns, not wall echoes.

Root cause: a change in sonar tilt between session B03 and B04 caused the sonar beam
to clip the floor, producing near-field phantom echoes.  The EchoProcessor traded
this problem for a new one: because short-distance examples were rare in training data,
its CNN regressed toward the mean and systematically *overestimated* short distances —
dangerous for a wall-following policy.

## Real Fix
Correct the sonar tilt on the robot so that phantom echoes no longer occur.  Validate
with a new session that should resemble B01–B03.  Once hardware is correct, the
threshold-based `corrected_distance` / `corrected_iid` from AcousticProcessing is
reliable and the EchoProcessor is unnecessary.

## New Pipeline

### Emulator training
```
geometric distance profile  →  Emulator  →  (distance_mm, iid_db)
```
Targets: `sonar_package['corrected_distance']` and `sonar_package['corrected_iid']`
from clean sessions (correct sonar tilt).  No EchoProcessor in the loop.

### Policy training (simulation)
```
robot position + arena layout  →  geometric profile  →  Emulator  →  (distance_mm, iid_db)  →  Policy
```

### Policy deployment on robot
```
sonar ping  →  AcousticProcessing  →  sonar_package  →  corrected_distance / corrected_iid  →  Policy
```

The Emulator learns the same mapping that AcousticProcessing computes on the robot,
minimising the sim-to-real gap without any intermediate learned model.

## Implementation steps
1. Fix sonar tilt on robot.
2. Collect new validation session; run SCRIPT_CheckRobotAccuracy.py to confirm clean data.
3. Retrain Emulator on clean sessions using `corrected_distance` / `corrected_iid` as targets.
   Drop `echo_processor_dir` dependency from `Emulator.load()`.
4. Remove EchoProcessor from `SCRIPT_RunPolicy.py`; replace with direct sonar_package fields:
   - `iid_db  = sonar_package['corrected_iid']`
   - `dist_mm = sonar_package['corrected_distance'] * 1000.0`
5. Retrain policy using updated Emulator.

## Key files
- `Library/Emulator.py`          — remove EchoProcessor dependency; update training targets
- `Library/EchoProcessor.py`     — no longer used at runtime (keep for reference or delete)
- `Library/EnvironmentSimulator.py` — no change needed (uses Emulator via `get_sonar_measurement`)
- `SCRIPT_RunPolicy.py`          — replace EchoProcessor block with sonar_package fields
- `SCRIPT_CheckRobotAccuracy.py` — use to validate new sessions before retraining
