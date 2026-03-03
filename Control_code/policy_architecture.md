# Policy Architecture: Two-Head Shared-Encoder MLP

See `SCRIPT_TrainPolicy.py` → `HistoryNNPolicy`.
Default hyperparameters: `h1=16, h2=8, history_len=5, feature_dim=6`.

---

## Biological motivation

The robot mimics a bat's two-stage sensing and steering loop:

1. **Head turn (rotate1)** — the bat *points* its sonar before emitting the pulse.
   The echo direction is chosen *before* the echo exists.
2. **Body turn + flight (rotate2)** — *after* hearing the echo the bat steers and flies.

These two decisions are causally ordered: rotate1 determines *where* the measurement
comes from; rotate2 is informed *by* that measurement.  A single network outputting
[rotate1, rotate2] simultaneously cannot respect this ordering — it either sees the
current echo for both (impossible: rotate1 hadn't happened yet) or for neither
(wastes information that is genuinely available when rotate2 is decided).

---

## Network diagram

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │  INPUTS                                                              │
 │                                                                      │
 │  History buffer  (history_len=5 steps × 6 features = 30 values)     │
 │                                                                      │
 │  Each row =  [ iid_norm | dist_norm | rot1_norm | rot2_norm |        │
 │                drive_norm | blocked ]                                │
 │              step t-5  (oldest, zero-padded if not yet available)    │
 │              step t-4                                                │
 │              step t-3                                                │
 │              step t-2                                                │
 │              step t-1  (most recent)                                 │
 │                                                                      │
 │  Flattened → hist_vec  shape (30,)                                   │
 └──────────────────────┬───────────────────────────────────────────────┘
                        │  (30,)
               ┌────────▼────────┐
               │  Shared Encoder │   W1 (16×30)  b1 (16,)
               │   tanh( W1·x    │   496 params
               │        + b1 )   │
               └────────┬────────┘
                        │  h1  (16,)
           ┌────────────┴─────────────────────────────┐
           │                                          │
           ▼                                          │
  ┌─────────────────┐                                 │  + current measurement
  │   HEAD 1        │                                 │  (taken after rotate1
  │   rotate1       │                                 │   is executed)
  │                 │                                 │
  │  W2a (8×16)     │             [ h1 (16,) | iid_norm {-1,0,+1} | dist_norm [0,2] ]
  │  b2a (8,)       │                                 │  (18,)
  │  tanh           │                                 ▼
  │  136 params     │                      ┌─────────────────┐
  │                 │                      │   HEAD 2        │
  │  W3a (1×8)      │                      │   rotate2       │
  │  b3a (1,)       │                      │                 │
  │  tanh           │                      │  W2b (8×18)     │
  │  9 params       │                      │  b2b (8,)       │
  │                 │                      │  tanh           │
  │  out ∈ [-1, 1]  │                      │  152 params     │
  └────────┬────────┘                      │                 │
           │                               │  W3b (1×8)      │
           │ × max_rotate1_deg (90°)       │  b3b (1,)       │
           │                               │  tanh           │
           ▼                               │  9 params       │
        rotate1 (deg)                      │                 │
     ∈ [-90°, +90°]                        │  out ∈ [-1, 1]  │
                                           └────────┬────────┘
                                                    │
                                                    │ × max_rotate2_deg (90°)
                                                    │
                                                    ▼
                                                rotate2 (deg)
                                             ∈ [-90°, +90°]
```

---

## Deadband

Both heads suppress their output when the relevant IID is near zero (open space, no
clear wall signal):

| Head   | Deadband input         | Threshold        | Output when in deadband |
|--------|------------------------|------------------|-------------------------|
| Head 1 | `last_iid_db`          | `< deadband_db`  | rotate1 = 0             |
| Head 2 | `current_iid_db`       | `< deadband_db`  | rotate2 = 0             |

`deadband_db = 0.25 dB` by default.  Head 1 uses the *previous* step's IID (the
most recent measurement available before looking); Head 2 uses the *current* step's
IID (just measured at the look direction).

---

## Step sequence (one navigation step)

```
 t=0  ────────────────────────────────────────────────────────────────►  t=1

  1. Build hist_vec from the last 5 history entries (zero-pad if < 5)

  2. h1 = tanh( W1 · hist_vec + b1 )          ← shared encoder

  3. rotate1 = Head1( h1 )                     ← look decision
       deadband check on last_iid (previous step's measurement)

  4. Execute rotate1: body turns to  yaw + rotate1

  5. Measure sonar at (x, y, yaw + rotate1)
       → iid_db   (IID in dB,  clipped to ±12 dB → iid_norm = iid/12)
       → dist_mm  (distance,   clipped to 2000 mm → dist_norm = dist/2000)

  6. rotate2 = Head2( [h1 | iid_norm | dist_norm] )   ← steer decision
       deadband check on current iid_db

  7. Execute rotate2 + drive 100 mm
       new body yaw = yaw + rotate1 + rotate2

  8. Append to history:
       [ iid_norm, dist_norm, rot1_norm, rot2_norm, drive_norm, blocked ]
       (rot1_norm = rotate1/90,  rot2_norm = rotate2/90,  pre-mirroring)
```

---

## History features (6 per step)

| Index | Feature    | Range      | Meaning                                        |
|-------|------------|------------|------------------------------------------------|
| 0     | iid_norm   | {−1, 0, +1}| Pure IID sign: +1 = right wall closer, -1 = left wall closer, 0 = deadband |
| 1     | dist_norm  | [0, 2]     | Distance at look direction ÷ 2000 mm           |
| 2     | rot1_norm  | [−1, +1]  | Head turn this step ÷ 90° (pre-mirroring)      |
| 3     | rot2_norm  | [−1, +1]  | Body turn this step ÷ 90° (pre-mirroring)      |
| 4     | drive_norm | [0, 1.5]  | Executed drive last step ÷ 100 mm              |
| 5     | blocked    | {0, 1}    | Whether last drive was physically blocked      |

IID sign convention: **positive = right ear louder = right wall closer**.
Correct wall-avoidance response: `sign(rotate2) = −sign(iid)`.

---

## Parameter count (default h1=16, h2=8, history_len=5)

| Layer              | Weight shape  | Bias shape | Params |
|--------------------|---------------|------------|--------|
| Shared encoder     | W1  (16 × 30) | b1  (16,)  | 496    |
| Head 1 hidden      | W2a  (8 × 16) | b2a  (8,)  | 136    |
| Head 1 output      | W3a   (1 × 8) | b3a  (1,)  | 9      |
| Head 2 hidden      | W2b  (8 × 18) | b2b  (8,)  | 152    |
| Head 2 output      | W3b   (1 × 8) | b3b  (1,)  | 9      |
| **Total**          |               |            | **802**|

Head 2 hidden layer is wider by 2 (18 vs 16) because `[h1 | iid_norm | dist_norm]`
concatenates the current measurement onto the shared encoding.

---

## Training

- **Algorithm**: Genetic Algorithm (no backprop). Genome = all 802 weights flattened.
- **Directional bias prevention**: random per-episode mirroring (50% probability by default).
  Mirroring negates IID inputs and rotation outputs, forcing the policy to learn symmetric
  wall-following behavior that works equally well for left and right walls.
  `fitness = mean(reward across all episodes, both normal and mirrored)`
- **Fitness**: sum of per-step sinuosity rewards
  `reward = 1 − w_turn_penalty × (sinuosity − 1)`,  episode terminates on wall hit.
- **Arena**: trained on multiple distinct physical arenas (sessions B02–B05),
  validated on a held-out arena (session B01).
