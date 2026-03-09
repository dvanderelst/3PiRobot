# Sign and Coordinate Conventions

This document records the conventions used throughout the pipeline.
Violating any of these will silently invert IID or rotation behaviour.

---

## 1. IID sign

**Positive IID = wall on the robot's physical RIGHT.**

Computed as `10 * log10(right_energy / left_energy)` by EchoProcessor.
All components must agree:

| Component | Source |
|---|---|
| `AcousticProcessing` — `corrected_iid` | raw hardware channels, right/left as wired |
| `EchoProcessor` — `iid_db` | `10*log10(right/left)` from sonar waveform |
| `DataProcessor` — profiles | positive azimuth = LEFT (CCW convention, see §3) |
| `Emulator` — `iid_db` | trained on EchoProcessor targets from DataProcessor profiles |
| `EnvironmentSimulator` — `iid_db` | emulator called with CCW-corrected profiles (see §4) |

---

## 2. Rotation sign

**Positive rotation = robot turns RIGHT (clockwise when viewed from above).**

Applies to:
- `client.step(angle=X)` — physical robot
- `rotate1_deg` / `rotate2_deg` in `EnvironmentSimulator.simulate_robot_movement`
- `rotate1` / `rotate2` returned by `PolicyController`

`rotate1` = sonar head turn (executed before the ping).
`rotate2` = body turn (executed after the ping, before driving).

---

## 3. Azimuth / profile convention

**Positive azimuth = CCW from forward = robot's physical LEFT.**

This is the standard mathematical convention (right-hand / CCW positive).

- Profile bins are ordered from `−opening_angle/2` (rightmost) to `+opening_angle/2` (leftmost).
- First half of the profile array = robot's RIGHT side.
- Second half of the profile array = robot's LEFT side.

Used consistently in `DataProcessor.load_profiles` and the emulator training data.

---

## 4. EnvironmentSimulator — y-axis correction

The arena image uses **image coordinates** where y increases **downward**.
In image coordinates, `arctan2(rel_y, rel_x)` gives a positive angle for walls
to the robot's physical RIGHT (opposite of the CCW convention above).

To align with the emulator's training convention, `rel_y_mm` is **negated** inside
`EnvironmentSimulator.get_relative_wall_coordinates` before the profile is computed.
This is a one-line fix; do not remove it.

---

## 5. Bilateral symmetry in HistoryNNPolicy

The policy is trained exclusively in a **canonical "wall-on-right" frame**:
- canonical IID = `abs(physical_iid)` — always non-negative
- canonical rotX = physical rotX reflected back to the positive-IID frame

`decide_rotate1` and `decide_rotate2` apply a sign-flip transparently:
- If `physical_iid < 0` (wall on left): IID is reflected, output rotation is negated.
- Always pass **raw physical IID** to these methods — do not pre-flip.

---

## 6. Bug history

**EnvironmentSimulator y-axis bug (fixed):**
Before the `rel_y_mm` negation was added, the simulator produced
left-right-flipped profiles. The emulator then predicted IID with the **wrong sign**
(negative for wall-on-right). The policy trained under this convention learned
inverted bilateral symmetry and turned **toward** walls on the physical robot.
Fix: negate `rel_y_mm` in `get_relative_wall_coordinates`.
Consequence: any policy trained before this fix must be retrained.
