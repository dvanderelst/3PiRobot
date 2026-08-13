# Project Handoff

This file is the project-level Claude memory. End each session by updating the relevant section(s). Start each session by reading this file first.

The `~/.claude` auto-memory is machine-local and does not follow this project across machines; anything that needs to be portable lives here.

## Project at a glance

- **Paper:** *Vicarious sonar learning in a bat-inspired robot* (`Paper/main.tex`). Framework: cross-modal inverse training → cross-modal direct learning → cross-modal vicarious learning, instantiated on a bat-like robot. **Uncommitted paper work is now committed (`bb47507`), with agent notes `\dnote[10..16]` marking every passage the pole-range head supersedes:** four heads not three (Par 20, `fig:network`, the objective in Par 21); all inverse Results figures and `tab:inverse-results` predate the range head (D13, which also carries the CV evidence and a warning **not** to quote pole recall off a single split); `fig:inverse-results` needs a pole-range panel (D14); the trial design is resolved (D15); and the stop criterion is superseded (D16 — including that the paragraph's own claim that the tracker was "used only to score the run" was **false** until this change). Paper figures deliberately left uncommitted: `fig_robot.pdf` grew 4.5 → 17.5 MB, worth checking before it enters history permanently. **Experiment 1 loose ends, recorded in `\dnote[17]` (restored to `main.tex` 2026-08-07 — the note was written at `a33c350` and silently dropped by the language-edit passes without its items being fixed):** (1) the **1400-rollout simulation appears nowhere in the paper** (that paragraph is commented out), so nothing states how the two pole placements were chosen — it belongs in the Methods; (2) Methods says *trials*, Results says *runs* — pick one; (3) **the P2/S5 sonar and vision runs did not share a controller seed**, unlike the other nine pairs, so that one pair is not matched on the controller's random draws. Also still open from `\dnote[18]`: the biological framing for Experiment 2 (landmarks vs path integration, the Neuweiler & Mohres precedent in scratchpad section 6) is not yet written.
- **Robot:** Pololu 3pi+ 2040 with three MaxBotix MB1360 sonars (two ears + emitter). Control code under `Control_code/`. Firmware under `Robot_code/`.
- **Compute topology:** code lives in Dropbox so it syncs across the user's dev machines; the project's authoritative remotes are on GitHub. The **PyLorex tracking server runs on a separate dedicated computer**, not on the dev machine the user is currently typing on. When PyLorex code or calibration changes, the server-computer needs a `git pull` + a tracker restart before any data acquisition — otherwise the server keeps serving whatever it loaded at last startup. Control scripts in `Control_code/` connect to both the robot and the tracker server.
- **Key docs:**
  - `rationale.md` (root) — stable architecture doc for the robot pipeline (SonarModel, simulator, policy, deployment). Do not edit without explicit user consent.
  - `Paper/style.md` — writing-style reference for `main.tex`. UK English, no em-dashes, hand-numbered paragraphs. Read before editing the paper.
  - `Paper/dieters_scratch_pad.md` — the user's working notes for Discussion / Proposal. Deliberately untracked. Read and update when referenced.
  - `Robot_code/readme.md` — hardware spec (3pi+ 2040, MB1360 datasheet links).

## Working conventions

- **Discuss-then-apply rhythm.** Propose concretely, wait for "yes" / "go ahead", then apply + commit. The user revises hard between turns.
- **Domain authority.** The user is the bat-echolocation domain expert. Trust their biology calls. Push back on prose, not biology.
- **Dropbox co-edit hazard.** The project lives in Dropbox; the user may edit between turns. Commit promptly after each batch.
- **Build cwd for the paper.** `pdflatex main.tex` must run from `Paper/`, not the project root (a stale `resources/main.tex` exists elsewhere).
- **Scratchpad is untracked.** Update `Paper/dieters_scratch_pad.md` when the user references it; never commit it.
- **Mapping / SLAM is held back for a separate paper.** Do not propose adding it to the current paper.
- **Commit style.** Topical, concise. Prefer separate commits per concern when the working tree mixes user-initiated and agent-initiated changes. Bisect-friendly milestone commits (calibration recovery, first clean deploy) should announce "this worked" + what specifically unblocked them in the body.
- **Editor / matplotlib.** User is on PyCharm. Default inline backend is fine for non-interactive scripts; force `TkAgg` only for scripts that need real GUI events.

## Where to pick up

- **Paper:** on branch `direct-learning-poletask`. **Introduction fully reworked (2026-06-20/21).** Now a clean, contiguous **Par 1–8**: overlap → inverse model + asymmetry → cross-modal inverse training (defined in Par 3) → sonar (consolidated) → vision → synthesis → "this paper" (two tasks) → biological-plausibility close (Par 8). US spelling throughout; terminology standardized to *cross-modal inverse training*. See Paper state (2026-06-21) for detail. **Next:** the intro has **no Discussion hand-off** (the direct/vicarious pointer was dropped) — draft the (briefer) Discussion treatment of direct + vicarious learning, drawing on the retired material commented after `\end{document}` (vision-as-internal-model / planning argument, Mugan2020/Bennett2023, `\cnote{5}`); reconcile Par 7 "pole" vs Methods "wooden dowel of 25 mm"; confirm Task 2 wording ("path integration + landmark recognition") matches the actual experiment; trim Par 5 small-mammals/rodents acuity redundancy. **Inverse-model Methods (Par 20–22) + Results (Par 23–27, `tab:inverse-results`, `fig:inverse-results`) are drafted and committed (`b53644e`, `7354057`)** around the B architecture + spatial-holdout deployment model (in-sample vs held-out; no generalization claim). **Experiment 1 Methods and Results are now drafted and committed (2026-08-04, `3dee986`/`201b101`/`45c71f3`)** — see Paper state 2026-08-04. **Next on Experiment 1**, all recorded in `\dnote[17]`: the two pole placements are never explained now that the simulation paragraph is cut (the 1400-rollout sweep belongs in Methods, with how the placements were chosen); Methods says *trials* where Results says *runs*; and the P2/S5 sonar/vision pair did not share a controller seed, unlike the other nine. **Then:** the intro still has **no Discussion hand-off** (the direct/vicarious pointer was dropped) — draft the (briefer) Discussion treatment of direct + vicarious learning, drawing on the retired material commented after `\end{document}` (vision-as-internal-model / planning argument, Mugan2020/Bennett2023, `\cnote{5}`); reconcile Par 7 "pole" vs Methods "wooden dowel of 25 mm"; confirm Task 2 wording ("path integration + landmark recognition") matches the actual experiment; trim Par 5 small-mammals/rodents acuity redundancy. Also open: an unfinished `\dnote` at Par 2 on the owl's az/el mapping being a simple 2D→2D map against our object-based inverse — the sentence trails off at "take a look at this paper:". Experiment-1 paragraphs use descriptive `% comments` (not `% Par N` — renumber pending).
- **Code:** on branch `direct-learning-poletask`. **Experiment 1 is DONE — all 20 robot runs complete (5 starts x 2 pole placements x sonar/vision), written up in the paper in a separate session.** **Experiment 2 (path following) is now UNBLOCKED**: all three blockers recorded on the morning of 2026-08-07 were cleared the same day (see Code state 2026-08-07). The simulator runs on a fitted inverse *error model*, the policy observation carries the class and pole channels, `default_Path02` is trained, and the deploy chain loads and steps end to end.
  **Next, in order — REVISED 2026-08-08.** The horizon result (Performance notes 2026-08-08) changes the order below: the inverse now comes *before* the path redraw, because how far the robot can see determines how much clearance a usable path can afford. Do not redraw against a 1 m horizon.

  0. **DONE 2026-08-08.** Extended-horizon inverse landed: range head masked, verified, deployed (`67c68c0`), error model refitted. `SonarModel/inverse_deploy_*` is now the uncapped model; the 1 m predecessor is at `SonarModel_archive/2026-08-08_deploy1m/`.
  1. **DONE 2026-08-12. Acquisition06A collected — 640/640 pings, zero tracker misses.** The far-range hole is filled: beyond 1700 mm went from 11 pings to 272, and the training set now runs to 3196 mm instead of 1911. Full numbers in Performance notes 2026-08-12 (evening); arena and plan reasoning in the entry below it. **The `none`/true-silence question remains unsettled** — the arena tops out at ~3.2 m in-cone, so it supplies far-but-present and no silence.
  1b. **DONE 2026-08-12. Inverse retrained on Acq01A–06A** (deploy + 4-fold CV), predecessor archived at `SonarModel_archive/2026-08-12_deploy_preacq06/`. Phantom largely fixed, far range better but not solved, 1400–1700 mm now the worst band anywhere. Numbers in Performance notes 2026-08-12 (late).
     - ⚠️ **`SonarModel/inverse_error_model.json` is still the pre-Acq06 fit.** `SonarModel/` is internally inconsistent — **do not train any policy against the simulator until it is refitted** (`SCRIPT_FitInverseErrorModel.py`).
     - **`SCRIPT_CheckPoleSignal.py` has not been run and would understate this data** — it reduces each ping by *global argmax*, which on a pole-with-wall-behind ping locks onto the wall. Restrict it to near range or window the features around the labelled range; do not read a weak result as "no far-pole signal".
  1c. **DECIDED 2026-08-12 (night), by experiment. The inverse work is scoped and closed; the goal is to make Experiment 2 run, not to perfect the model.** Five directions were tested and four rejected — see Performance notes 2026-08-12 (night). **Closed: the three-way output** (unnecessary — p is already calibrated at T=1.00, ECE 0.010, so "class unresolved" is just p≈0.5); **widening the cone** (±50 a wash, ±70 worse, and the 1.4 m cliff is present at every width, so it is not a labelling artifact); **simplifying to distance+azimuth+p(pole)** (no range gain, loses the wall profile, degrades pole bearing); **investigating 1400–1700** (it was noise); **loosening the pole-range cap** (confident pole claims already sit at a median 692 mm, so widening buys distance only on the unreliable ones).
     - **The single adopted change: add the class-agnostic nearest-reflector range head.** Validated out-of-fold — tracks to 2579 mm at ~11% error with honest σ, and works equally for walls and poles in the band where classifying them is a coin flip. Keep the pole-range head masked at 1 m as the terminal-stop signal. This is what gives Experiment 2 a landmark-distance channel that means something beyond a metre.
     - **Design summary to build against:** inside ~1.4 m the full local feature works (class, wall profile, pole azimuth + range). Beyond it the only honest outputs are the agnostic range and a blurry wall profile (300 mm RMSE at 1500–2000), with p(pole) telling the controller which regime it is in.

  1d. **Then, in order, to finish Experiment 2.**
     1. **DONE 2026-08-13.** Class-agnostic range head in the trainer (`8a3936a`), emitted by the error model (`d6182dd`), and wired into the policy observation as `use_agn` (`38c9e63`). Obs width 14 → 16 with poles and σ; defaults False on load so all six existing artifacts still validate. Verified that the deployed inverse and `InverseErrorModel.observe()` populate every key `encode_obs` reads, which is the check the 2026-08-07 `p_wall` fault existed to teach. `SCRIPT_TrainPolicy` defaults `use_agn=True` for the Path04 retrain.
     2. **DONE 2026-08-12 (`d6182dd`).** Error model fixed and refitted: bins extended to 2500+, per-ping posteriors replace the oracle, `agn_dist_mm` emitted. See Code state. **Remaining from this item: add the `agn_dist` channel to `Library/Policy.py`'s observation layout** — a deliberate change, since it moves the obs width (14 → 16 with σ) and so invalidates `default_Path02`. Fine to do when the policy is retrained for the new path anyway. **Expect simulated policy performance to drop against the old run**; the simulator is now much closer to the real sensor, so the two are not comparable.
     3. **DONE 2026-08-13. Path04 v3 is the path** — 41 waypoints, 8.8 m, min clearance 455 mm (370 usable), turns max 46°, detects 66% / 81% of a 200 / 300 mm lateral drift. Numbers and the two rejected redraws in Performance notes 2026-08-13 (later). `SCRIPT_AnalysePathRun.py` now reports drift detectability and what-is-ahead, so this is checkable without re-deriving it. **Do not judge a path by clearance, by `cls != none`, by object-facing, or by `EXPT_gaze_path.py` alone** — each of those missed a real failure during this iteration.
     4. **DONE 2026-08-13 (`db22a14`).** `motion_rot_bias_deg` (default 3.0) drawn per episode as U(−x,+x) and added to every step, applied *after* the rotation clip since curl is accumulated while driving rather than commanded. Verified that before the change a commanded 0° produced exactly 0° of motor rotation every episode — the defect. Logged per episode in `motion_noise_log.tsv`.
     5. **Policy trained 2026-08-13** (`PolicyTraining/default_Path04/`, val_mse 114.3, 5-10% collision rate in sim; Performance notes 2026-08-13 evening). **Next: recalibrate (`SCRIPT_CalibrateRobot.py`) and deploy** — `SCRIPT_RunPolicy.py` with `POLICY_INPUT_SOURCE="live"`. The deploy chain reads `min_dist_mm`/`max_dist_mm`/`use_agn` from the policy artifact, so no deploy-side constant needs changing; the inverse must be the current `SonarModel/inverse_deploy_*`, which has the agnostic range head.
     - **Also worth doing, cheap:** a confidence threshold in `feature_from_inverse` instead of argmax. It restores the controller's abstain path, currently dead code, and 0.7–0.8 gives 92–95% accuracy on 57–69% of pings.
     - **Experiment 1 goes last**, on whatever the final inverse turns out to be — re-running it now risks a third run, or two differently-scoped inverses in the paper.
     - **Parked, with a new justification:** gaze. The 2026-08-08 measurement parked it because the horizon extension solved the detection problem it was proposed for, but off-cone competition costs 15 points of class accuracy at matched range, and rotating to increase angular separation from the flanker attacks that directly — now steerable by the calibrated confidence. Not needed to finish Exp 2.
     - **Open and unanswerable here:** whether the sonar could classify an *uncontested* object at 2.5 m. Only 27 such pings exist, and the room caps them at ~1.9 m; settling it needs a space ~5 m across.
  2. **Then re-run Experiment 1** on the new inverse (user's call 2026-08-08: cost is low and it should improve — first perception moves out from 798 mm and the 800–1000 mm blind ring goes away, and the paper avoids describing two differently-scoped inverses). **Re-running before Acquisition06 risks the phantom mechanism**, since Exp 1 wanders in an open arena and the current model asserts `pole` beyond ~1.5 m; the old model's pole precision there was 97.9%, which is the number to beat. Note the protocol constants: the runs used `APPROACH_STOP_MM=400` / `ALIGN_MIN_DETECTIONS=3`, while HEAD carries 500 and 6.
  3. **Then redraw the path.** `default_Path02_run02` completed **2.1 laps (110 steps)** with lap 2 tracking lap 1 to 116 mm, then **crashed at steps 116/117** near (-227, -1120). The cause is geometric, not control:

     | | |
     |---|---|
     | Path02 min clearance (centreline -> obstacle) | 155 mm |
     | robot radius | 85 mm |
     | **usable margin at the tightest point** | **70 mm** |
     | robot tracking error: mean / median / 90th / max | 119 / 94 / **266** / 341 mm |

     **49% of the path has less margin than the robot's 90th-percentile tracking error**, so collisions there are structural — no amount of calibration fixes it. (At the crash the robot had drifted 421 mm off-path into a region with 595 mm of clearance, so the *tightest* sections have not even been tested yet.)
     **This is the direct cost of why Path02 was chosen.** It reached more informative perception than Path01 precisely by hugging walls and blocks; Path01's min clearance is 249 mm (164 mm usable) against Path02's 155 mm (70 mm). Closer to obstacles = better sonar = less room for error. **Redraw with a minimum clearance of ~350 mm** (about Path02's current median), i.e. the 266 mm p90 tracking error plus the robot radius, then re-measure informativeness. Use `SCRIPT_DefinePath.py`; it already draws the 800 mm pole landmark ring.
     ⚠️ **2026-08-08: the trade-off stated above is largely an artifact of these two particular routes and no longer binds.** Binning all 960 path poses by clearance, informative fraction at a 1400 mm horizon is **78–100% at every clearance band and is not monotonic in clearance** — clearance is omnidirectional, perception is a ±35° cone, so a pose far from everything can still face a wall a metre away. On Path01 the horizon extension alone took recall-weighted perception from 48.3% to 78.6% of steps (blind steps per lap ~38 → ~16). **Route the new path for safety and perception together; do not reintroduce wall-hugging to buy perception.** Then measure it before training with both `SCRIPT_AnalysePathRun.py` (informativeness at the deployed horizon) and `EXPT_gaze_path.py` (local position uncertainty — if it lands near Path02's 251 mm the parked gaze idea stays parked; near Path01's 803 mm, reopen it before training). See Performance notes 2026-08-08 (evening).
     **Tooling: `SCRIPT_AnalysePathRun.py` (added 2026-08-07) reproduces every number above** — path clearance profile, informative-perception fraction, robot tracking error, the "% of path tighter than p90 error" verdict, the required minimum clearance, per-step yaw residual and drive scale, and crash locations with their local clearance. Run it on a candidate path **before** training against it: `python3 SCRIPT_AnalysePathRun.py <Arena> [<RunSession>]`. Arena-only mode skips the run analysis.
     ⚠️ **Informative-perception metric mismatch — do not mix the two.** `SCRIPT_AnalysePathRun.py` measures *true geometry, facing along the path, `cls != none`*, and gives **Path02 = 85.5%**. The **26% / 49%** figures quoted for Path01/Path02 in the 2026-08-07 Performance-notes entry came from a different measurement whose definition was not recorded — it reported 73.5% wall yet 49% informative, so "informative" there was a **subset of wall hits**, not simply "not none" (possibly range-limited, or measured through the error model rather than true geometry). Treat the old 26%/49% as unreproducible. The qualitative trade-off (clearance vs informativeness) holds under either definition. **Both paths were re-derived under the script's definition on 2026-08-08 and the numbers are in that Performance-notes entry** (at 1000 mm: Path01 56.6%, Path02 85.5%), swept across horizons — use those, and no longer quote a single informativeness figure without saying which horizon it assumes.
  4. **Then collect repeats and the ablation conditions**, on the robot rather than in sim (user's call, 2026-08-07): `use_poles=False` and the blind control. The plumbing exists. Whether the *simulated* policy degrades without the pole channel says something about the simulator, not the robot. Retraining is needed for the new path anyway, so fold the training-noise fix (item 6) in at the same time.
  5. **`RunPolicy` has no arena guard**, unlike `RunDirectPolicy`'s `check_arena_matches_pole()`. A moved block gives a confusing failure rather than a clear one. Worth adding if the furniture moves between sessions.
  6. **Add an additive rotation bias to the training motion model.** `SCRIPT_TrainPolicy.py` perturbs motion with `rot_motor = rot_exec * rot_gain + N(0, 3deg)` — every term is either multiplicative on the commanded angle or zero-mean. **There is no additive bias term anywhere.** The real robot's fault is exactly a fixed additive offset (about -1.1 deg/step even after calibration, uncorrelated with the commanded angle), so it is a perturbation the policy has never met in training, and the multiplicative gain cannot emulate it (at `rot_exec = 0` the gain does nothing but the robot still sheds heading). Suggested: draw a per-episode `rot_bias ~ U(-3, +3)` deg and add it to `rot_motor`. run02 succeeded *despite* this, not because of it — worth making it robust by design. The same gap is why the pre-flight preview says nothing about calibration: it inherits this noise model.
  7. **Tracker settling is the weak link in the measurements.** ~3% of run02 steps are glitches, and the paired +18.2/-23.1 deg residuals at steps 44/45 are **one bad yaw read**, not two bad steps (a wrong `yaw[45]` biases the step before and after equally and oppositely). The calibration run threw several "pose did not stabilise within 8.0s" warnings for the same reason. A bad pose also feeds the policy a wrong `prev_rot`.
- **Uncommitted and deliberately left**: one-line `Path02` arena-name switches in `SCRIPT_DefinePath.py` and `SCRIPT_TakeEnvSnapshot.py` (session state, same pattern as the 2026-07-28 note); `Paper/introduction_logical_analysis.md` deleted but unstaged; new paper figure resources under `Paper/images/image_resources/`.

---

## Paper state

*Last updated: 2026-08-12.*
*Current branch for ongoing work: `direct-learning-poletask`. `main` carries up through the direct-learning rename + Par 9 task commitment.*

### PLANNED — rewrite the inverse Results around what the model can and cannot recover (2026-08-12)

**Not drafted. Do this after the Experiment 2 work, so the numbers are final.** Decided 2026-08-12 with Dieter. Not extra work: `\dnote[13]` already flags that `tab:inverse-results` and `fig:inverse-results` predate the range head and `\dnote[14]` asks for a pole-range panel, so these have to be redone anyway. The question is only what replaces them, and an aggregate metrics table is now the weakest option available. All supporting numbers are in Performance notes 2026-08-12 (night) and the two entries above it.

**Claims to make, strongest first:**

1. **Different echo cues survive to different ranges, and the pattern is principled.** Distance is a monaural time-of-flight cue and holds to 2.5 m at ~11% error; azimuth needs a binaural comparison and dies at ~1.4 m; class needs fine temporal/spectral structure and dies at the same point. This is a statement about what echoes support, not only about our network.
2. **The hard part is attribution, not sensing.** Same architecture, same data: asked "how far is *the pole*" the head saturates at ~750 mm; asked "how far is *the nearest reflector*" it tracks to 2579 mm, and does so equally for walls and poles (269 vs 280 mm beyond 2 m) in the band where discriminating them is at chance. **The model can locate what it cannot name.**
3. **A vision-supervised sonar inverse can be well calibrated.** Fitted temperature 1.00, ECE 0.010, p ≥ 0.9 → 97.9% accurate. It knows when it does not know, which is what makes a noisy inverse usable by a controller — a better result than a higher accuracy figure would be.
4. **Discrimination degrades under off-cone competition**, 80.6% vs 65.1% at matched range: perception of a target depends on what else is in the beam, not only on the target.

**Hedge the 1.4 m number carefully.** It is measured in a 3.5 × 4.2 m arena where essentially every distant target is flanked — uncontested far targets provably cannot exist there beyond ~1.9 m (27 such pings in the whole dataset). **Lead with the mechanism and let the number follow:** "discrimination fails once a nearer reflector lies outside the analysed cone; in our arena that condition holds for effectively all targets beyond ~1.4 m." Honest, and it generalises.

**What it costs:** a Methods paragraph for the second (class-agnostic) range head; one figure, suggested as four panels — class accuracy vs range, azimuth error vs range against a chance baseline, both range heads vs range on one axis, and a reliability curve; and a softening of **Par 22**, which currently says the held-out split is only an overfitting guard with no generalisation claim. It is more than that now.

**Why it earns its place structurally:** it explains why Experiment 1 works (everything in it happens inside 1.4 m) and what constrains Experiment 2 (the landmark channel), rather than sitting as a standalone characterisation.

### Experiment 1 Methods shortened + Results drafted (2026-08-04; commits `3dee986`, `201b101`, `45c71f3`)

- **Methods** shortened by the user; the agent pass fixed typos and rewrote the **400 mm stop rationale**. The shortened draft attributed the threshold to emission/echo overlap saturating the sensors below 400 mm, which contradicts our own data: the overlap sits near 88 mm, training carries pole echoes down to 253 mm, and recall is ~55% in the 200–300 mm band, all of which require measurable echoes there. Restored reasons: **no training data below 253 mm** (planner clearance 250 mm, so any estimate below is extrapolation in exactly the regime the stop occupies) and **recall falling as the robot closes**. `Vanderelst2026` kept as "a comparable criterion" without asserting its number (the JEB study used 50 cm, the paper uses 40).
- **Results** `\subsection{Experiment 1}`: three live paragraphs (outcomes; the cost of sonar; per-step perception), plus `tab:direct-results` and `fig:direct-results`. Two further paragraphs (the range asymmetry; the simulated controller ceiling) were drafted and are **commented out in place**, not deleted. Numbers in Performance notes 2026-08-04.
- **`fig:direct-results`** (`Paper/images/scripts/fig_direct_results.py`): 2×2, columns sonar/vision, rows P1/P2, five trajectories per panel colored by start (ColorBrewer Dark2), plus a dot at every pose where a pole was perceived and an open circle at every pose where one was present within 1 m and missed, and the 400 mm stop ring. The misses fall at the outer edge of the detection shell (12 of 15 between 800 and 1000 mm).
- **`Paper/images/scripts/exp1_stats.py`** is the aggregation behind every number in the subsection. It recomputes ground truth from each run's `arena_features.npz` (the logged referee columns are nearest-over-all-directions, for collision scoring, not nearest-in-cone), and `check_vision_identity` is its correctness check: vision must reproduce the recomputed truth on all 410 vision steps, which it does.
- **A second figure was built and then dropped.** Range × bearing occupancy matrices, acquisition vs runs. Dropped once the sampling explanation failed (see Performance notes); script and outputs deleted, restorable from this entry's reasoning if ever wanted.
- **Only figure PDFs are tracked** (`fig_direct_results.pdf`); the SVG and the inspection PNG stay untracked, as for the other figures.

### Inverse-model Methods + Results, and Experiment 1 Methods draft (2026-06-24; commits `b53644e`, `7354057`; Experiment 1 uncommitted)

Built on the deployed spatial-holdout inverse (Code state + Performance notes 2026-06-23).

- **Methods "Training the inverse model"** finalized around the **B** architecture (`SonarSlicesUQ_Wall3`) and the single spatial-15%-holdout deployment model: Par 20 architecture (with `fig:network` + caption describing B's single symmetric 3-output wall head and `z_{LR}`/`z_{RL}` orderings), Par 21 training (masked three-part objective, GNLL, input/target scaling), Par 22 validation. **Par 22 states the framing explicitly:** no generalization claim, the held-out split is an overfitting guard, the experiments are the validation. `tab:inverse-params` updated (validation-holdout row; seed 42/0).
- **Results "Inverse model"** (Par 23–27 + `tab:inverse-results`) report **in-sample vs held-out** as two columns (class acc 85.5/83.9%, pole-az 10.1/12.1°, wall RMSE L/C/R 304/311/271 vs 330/397/248); the small gap stands in for the overfitting check. New **`fig:inverse-results`** (generator `Paper/images/scripts/fig_inverse_results.py`): confusion + pole-az + wall-depth true-vs-predicted on the held-out set, single-row layout, recomputed from the deployed model so figure and table agree. Held-out center wall RMSE (397) flagged as single-split sampling noise.
- **Experiment 1 Methods (obstacle avoidance + target approach)** drafted as a new `\subsection` before Results (**uncommitted**). Grounded in `SCRIPT_RunDirectPolicy.py`: the task (approach a pole, avoid walls, sonar alone), the modality-agnostic local feature (sonar via the inverse; vision read locally from arena geometry — not a map), the reactive controller's three branches (pole-approach / wall-avoid / empty-wander), and the overhead-tracker referee with its outcomes (reached-pole < 88 mm, collision < 20 mm, corner jam, 200-step cap). Paragraphs use descriptive `% comments` (not `% Par N`) pending a renumber.
- **Open (Experiment 1), as 4 `\cnote`s — the drafting surfaced these:** (1) **trial design** (the script runs one trajectory per invocation; decide start positions, repeats, and conditions — sonar alone vs sonar+vision±sim); (2) **summary measure** (nothing aggregated in-script — pick success rate / steps-to-pole / per-step class agreement / example trajectories); (3) report the **sim** condition or only sonar+vision; (4) controller **constants** in prose vs a parameter table. Resolving (1) unblocks the protocol/measures paragraphs and Experiment 1 Results.

### Introduction restructure + polish (2026-06-21, commits `036d893`, `456de36`, `06caeeb`, `0fc8a75`)

Building on the 2026-06-20 refocus (below), the Introduction was restructured (the user reworked it heavily between turns) and is now a clean, contiguous **Par 1–8**. Paragraph numbers here supersede those in the 2026-06-20 entry. Flow: 1 overlap across modalities → 2 inverse model + accessibility asymmetry → 3 cross-modal inverse training (mechanism + first-use `\emph{}` definition) → 4 sonar (consolidated) → 5 vision → 6 synthesis → 7 this paper (two tasks) → 8 biological plausibility.

- **Option B restructure (`456de36`).** The sonar argument is now ONE contiguous paragraph (Par 4): distance directly available → richer features need computation (overlapping echoes, cochlear integration) → not image-like / no 3D reconstruction (seeing-with-sound, our prior work, behavioural+neural evidence, **echo-to-depth ML failures**) → therefore the inverse is feature-based. Vision (Par 5) follows, then a single synthesis paragraph (Par 6). Previously the sonar argument was split across two paragraphs interleaved with vision.
- **Terminology standardized** to *cross-modal inverse training* throughout, defined once with `\emph{}` in Par 3 (was mixed with "cross-modal inverse model learning" in the intro).
- **US spelling (`06caeeb`).** `main.tex` converted UK→US (20 occurrences); `Paper/style.md` policy flipped UK→US. Bib titles keep their published spelling.
- **Par 5 (vision) built out.** Acuity verified against sources: laryngeal microbats ~0.6 cyc/deg (Cechetto2020; behaviourally cross-checked via Manske *Desmodus* 48′ stripe → ~0.6); *Rousettus* ~2.8, pteropodids ~3.8; human ~30 (DeSousa2022). Bell & Fenton's *Macrotus* figures are minimum-VISIBLE (detection) angles, NOT resolution — not convertible to cyc/deg, kept distinct in the prose. Depth: binocular overlap (*Macrotus* ~50°, ~2× others; Bell1986) + monocular cues (Kugler2019).
- **Bib (`036d893`).** Added + cleaned echo-to-depth refs (Brunetto2023, Zhang2025, Christensen2020a, Frank2020a, Parida2021) and Kugler2019; deduped Cechetto2023 (kept the complete copy, italicized species); stripped imported entries (abstracts, a ~200-line annote, file paths, keywords). DeSousa2022 added with the intro polish (`0fc8a75`).
- **Par 8 biological plausibility (`0fc8a75`).** New closing paragraph: realistic complex echoes + sonar poorer than real biosonar → feasible in principle. The old commented-out constraints draft was deleted; its Discussion pointer (direct/vicarious) and "not modeling a specific finding" disclaimer were **dropped** — so the intro currently has no Discussion hand-off.
- **Still open:** draft the Discussion (direct + vicarious learning) from the retired block after `\end{document}`; reconcile Par 7 "pole" vs Methods "wooden dowel of 25 mm"; confirm Task 2 wording vs the real experiment; trim Par 5 small-mammals/rodents redundancy.

### Introduction refocus (2026-06-20, commit `90f2a6a`)

The Introduction was restructured per the user's `Paper/new_outline.md` (telegram-style target structure, kept untracked). Goal: refocus the intro on the **empirical contribution** rather than the direct/vicarious-learning dichotomy. The intro now flows: feature-ease asymmetry (Par 1) → inverse model (Par 2) → cross-modal inverse training (Par 3) → bats + sonar's 3D limits (Par 4) → bat vision (Par 5) → **new gap paragraph (Par 6)** → noisy-but-usable inverse + two tasks (Par 9) → constraints + Discussion pointer (Par 10).

- **New gap paragraph (Par 6).** Positions prior sonar-classification work as stopping short of action. Citations verified against Zotero full texts (2026-06-20): Yovel2011 (perceptual classification review), Vanderelst2016 (place recognition), Achutha2021 (efficient-encoding perceptual benchmarks), Wang2022 (vision-supervised teacher/student gap detection — closest precedent, still no action), Eliakim2018 (Robat acts on echo *geometry*, learned classifier not in the control loop). Four new bib keys added: `Yovel2011`, `Eliakim2018`, `Wang2022`, `Achutha2021`.
- **"This paper" paragraph (Par 9) reframed.** Two tasks stated plainly: (1) obstacle avoidance + target approach; (2) preset path with path integration + landmark recognition. No "direct/vicarious learning" labels. Note "cardboard pole" → "pole" in this paragraph (Methods still says "wooden dowel of 25 mm diameter" — reconcile pending).
- **Retired material.** The vicarious-learning intro (old Par 6), the vicarious half of old Par 9, and the inverse-scope-asymmetry `\cnote` are now a commented block after `\end{document}`, tagged for salvage. The unique "vision-as-internal-model / planning" argument (Mugan2020, Bennett2023) and the `\cnote{5}` both rode along there; relocate to a **briefer Discussion pickup** of vicarious learning.
- **Numbering** kept minimal: the gap paragraph reused the freed Par 6 slot, so Par 9/10 and all Methods/Discussion numbers are untouched.
- **Still open (next pass):** Par 10 typos, echo2depth `\cnote` forward-ref, confirm Task 2 description matches the real experiment, draft the Discussion vicarious pickup.

The framework now names three operations cleanly: **cross-modal inverse training**, **cross-modal direct learning**, **cross-modal vicarious learning**. The canonical definitions live in the comment block at the top of `\section{Discussion}`. The rename (from "feature transfer" to "direct learning") highlights the no-internal-model contrast, mirroring the psychology literature's *direct* ↔ *vicarious*.

Par 9 commits the paper to a specific direct-learning task: a Holland/Finger-mimicking pole-arena demo. The inverse is **two-headed**: classify the nearest reflector (wall or cardboard pole); output a 3-direction depth profile when it is a wall, and the pole's azimuth when it is a pole. The action policy is a hand-coded reactive rule on this output (approach pole, avoid wall), explicitly framed in Par 9 as a stand-in for the trial-by-trial association a bat would acquire under reinforcement.

### Three design constraints locked in (2026-05-20)

1. **Vision stays local-only during direct learning.** The overhead camera supplies the same kind of local feature the sonar inverse produces (class label + class-conditional feature, extracted from the robot's instantaneous pose), not a global obstacle map. Otherwise the supervision signal is generated by a map-using process and the direct/vicarious distinction collapses.
2. **Selector logic is "class of nearest object."** User accepted the trade-off (robot has to wander a bit if a wall is closer than the pole). Pole-priority would be more aligned with the policy's needs but adds complexity.
3. **No short-term memory in the direct-learning policy.** Geometry suggests a single 3-point profile + class label is enough. If empirics force a window, both vision-teacher and sonar-student get the same window.

### Design refinement (2026-05-21)

Par 9 restructured around **two parallel blocks** rather than three. Each block (direct, vicarious) opens with its inverse model, then describes the task and what we test. The standalone "First, we show inverse training is feasible" block is folded into the two demonstrations.

Fourth design constraint added: **asymmetric inverse-model scope between the two operations.** Direct learning uses a specialised inverse trained on data from a single arena (does not need to generalise beyond it); vicarious learning uses a general inverse trained across multiple arena layouts (must support sonar reconstruction of unfamiliar environments). Same two-headed architecture, different training distributions. Implication for Methods and Results: each operation gets its own inverse-training subsection, and the Par 9 1a/1b structure (inverse, then task) is the template for Results.

Par 14 robot diameter fixed (90 → 96 mm) to match the actual Pololu 3pi+ 2040 spec, prompted by the new larger top-plate marker (96.1 × 96.1 mm card) making the old number obviously wrong.

### Methods drafting started (2026-05-20)

First Methods subsection drafted: `\subsection{Arena and Robot}` (Par 11–17), placed between Introduction and Discussion. Covers arena geometry (panel walls with foam facing, pole obstacles), robot platform (Pololu 3pi+ 2040 + HiLetgo ESP8266 ES), MaxBotix sonar payload geometry and acoustic parameters, overhead-camera tracking, and arena-layout digitisation by back-projection. Past tense throughout per corpus convention. Discussion paragraphs renumbered to Par 18–29.

Remaining Methods subsections (sonar data acquisition protocol, two-headed inverse training, direct-learning policy, vicarious-learning policy training) not yet drafted; will be filled in as the empirical work settles.

### Drafting note macros

`\cnote{N}{...}` (Dieter → Claude, blue) and `\dnote{N}{...}` (Claude → Dieter, red), defined in the preamble. Use inline at the point the note refers to; the numeric tag N is for cross-reference in conversation. Greppable via `grep -n '\\cnote\|\\dnote' Paper/main.tex`. Sublime Text highlighting is provided by `~/Dropbox/Scripts_and_Settings/SublimeText/DraftNoteHighlight.py` (symlinked into every machine's `Packages/User/`). Replaces the older `% >>> Cn:` line-comment convention.

### Branch reasoning

`direct-learning-poletask` exists because the Par 9 task commitment is empirically contingent. If the inverse turns out untrainable in the current configuration, Par 9 (and the Methods that follow) need revision. The naming rename is general improvement and stays on `main` either way. If the empirical work succeeds, merge back; if it fails, revert Par 9 on `main` and rethink.

### Open items

- **Par 10 typos.** `off`→`of`, `readility`→`readily`, `emssions`→`emissions`, `opossed`→`opposed`, `typocally`→`typically`, `freqency`→`frequency`.
- **C5 at Par 7** (now `\cnote{5}{...}`): "We should probably add some more arguments for the existence of internal models supported by vision." Scratchpad §2 and §3 material likely lands here.
- **Pole material/diameter inconsistency.** Methods Par 13 calls them *wooden dowel of 25 mm diameter*; Intro Par 9 still calls them *cardboard pole*. Reconcile once the physical setup is final. (Code side resolved 2026-05-20: `POLE_RADIUS_MM = 12.5` in `SCRIPT_BuildArenaGeometry.py`.)
- **Scratchpad §6 (Neuweiler & Möhres 1967) Discussion landing.** Strong precedent passage (§6.A — the "Raumbild" quote) and the wing-folding finding (§6.2) deserve prominent placement; suggested lead per §6.4.
- **Scratchpad §2 (rich-sense / specialised-sense architecture)** and **§4 (vision-as-calibrator bias)** — proposal-direction material, not yet drafted.
- **Remaining Methods subsections, Results, Abstract** — not drafted.

### Suggested first move (next paper session)

The inverse has trained successfully under the canonical close-range setup (see Code state). The reasonable next-paper directions, in increasing scope:

- **Quick paper-only edits.** Par 10 typo cleanup → `\cnote{5}` at Par 7 → §6 Neuweiler & Möhres landing → reconcile Par 9 "cardboard pole" against Methods Par 13 "wooden dowel of 25 mm diameter".
- **Draft the inverse-training Methods subsections.** Two needed now (specialised inverse for direct learning; general inverse for vicarious learning). Both use the same architecture and supervision protocol; the difference is the training distribution. Section structure should mirror the Par 9 1a/1b template.
- **Draft Results section 1 (direct-learning inverse).** Use the close-range Acq01A+Acq02A numbers as the current best estimate, but flag that more sessions are coming. Per-head performance, the cross-modal supervision protocol, and the choice of overhead-camera-derived ground-truth labels all live here.

---

## Code state

*Last updated: 2026-08-13.*
*Current branch for ongoing work: `direct-learning-poletask`.*

### Training-pipeline pre-flight review (2026-08-13)

Commits `223c1a1`, `e860e2c`, `83363e0`, `38c9e63`, `db22a14`. Run before the Path04 training and worth having done: six faults, four of them one bug wearing four hats. **The lesson to carry: `MAX_RANGE_MM = 1000` is a LABELLING constant and stopped meaning "how far the sensor sees" the moment `FAR_LABEL_MODE` became `"true_class"`. Every place that read it as a horizon was wrong.**

- **The simulated sensor went blind past 1 m.** `EnvironmentSimulator` took its horizon from the error model's `provenance.max_range_mm`, which the fitter wrote from `MAX_RANGE_MM`. `true_local_feature` then returned class `none` beyond it, so the sim emitted a clean **`p_none = 1.0`** at 1200/1800/2500 mm — while the deployed inverse has no abstain output at all and its `p_none` maxes at 6.5e-4 over 2775 pings. A policy would have learned to read a channel that is identically zero on the robot. **The fitter now records `sensor_horizon_mm` separately** (derived from the data span, 3356 mm); far-range failure is carried by the confusion and posterior tables, where it was measured.
- **`encode_obs` clamped every distance at `max_dist_mm = 1000`,** so the new agnostic range channel read exactly 1.000 at 1000/1500/2000/2500/3000 mm — saturated precisely where it is the only usable signal, discarding the whole point of the head at the last step. Raised to **2500**, the validated span of both that head and the wall slices. Cost: compressed near-range resolution.
- **The no-hit slice sentinel was also `max_range_mm`.** With the clamp raised, 1000 mm would have read as "a wall at a metre" rather than "nothing in range". Now tied to the horizon.
- **`get_clean_measurement` did not emit `agn_dist_mm`** — the deploy preview's path, so it would have pinned the channel at zero via `encode_obs`'s `.get` default. Exactly the fault that function exists to prevent for `p_wall`/`p_pole`/`p_none`.
- **Poles were not in the simulator's collision geometry** (`_segment_collides_with_walls` tested `arena.walls` only), so the simulated robot drove through the pole and never got a collision signal for it. Found by chasing Dieter's observation that the pole was missing from the teacher-rollout plot — a good argument for drawing things. Poles are now tested with their radius added to the clearance, rather than appended to the wall cloud, because that cloud is a *surface* sampling while a pole is a *centre*. All three rollout plots now share a `_draw_poles` helper.
- **`use_sigma` stays OFF, and not as an ablation — it would be an oracle.** `observe()` emits the fitted per-BIN σ, so σ is constant within a true-distance bin while the distance is noisy: at a true 400 mm the reported distances scatter 76–648 mm and σ reads exactly 155 every draw. σ therefore identifies which of the 7 wall-slice bins the TRUE distance falls in, exactly — worth *more* than the class-posterior oracle removed in `d6182dd`, because the distance channel is so noisy. **The fix, if the channel is wanted: have the fitter keep the model's actual per-ping predicted σs per bin and have `observe()` draw one, exactly as it now does for the class posterior.** Recorded in the config comment so it is not flipped back unknowingly.
- **Additive rotation bias added** (`motion_rot_bias_deg`, default 3.0, drawn per episode as U(−x,+x)). Every other perturbation was multiplicative on the commanded angle or zero-mean; the robot's dominant fault is neither. Verified before the change that a commanded 0° produced exactly 0° of motor rotation every episode. Applied **after** the rotation clip, since curl is accumulated while driving rather than commanded, and must not be clipped away when the commanded angle is already at the stop.

### Simulator error model fixed and refitted (2026-08-12)

Commit `d6182dd`. Two defects, both optimistic, both located exactly where Experiment 2 operates. The error model is what a simulation-trained policy learns its sensor from, so these were shaping the policy more than any inverse change.

- **The confusion bins stopped at `MAX_RANGE_MM`.** Harmless while `FAR_LABEL_MODE="none"` put everything beyond it in the abstain class; wrong under `"true_class"`, where real walls and poles reach 3196 mm. `_pick_probs` falls back to the *nearest* bin rather than failing, so **a wall at 2.5 m was classified at the 900–1000 mm rate — 89% correct at any range, against a measured 49–59%.** Bins now run to 2500+.
- **The class probabilities were an oracle.** `p_wall`/`p_pole` were the bin's mean confusion row: identical for every ping in a bin, and a deterministic function of the *true* class (wall rows 0.013–0.109, pole rows 0.727–0.923, disjoint). A policy reading those channels could recover ground truth even on pings where the sampled label was wrong — and, pulling the other way, it carried no per-ping information, so the policy could never learn that a low posterior marks an unreliable reading. **The fitter now stores the model's actual per-ping posteriors per (true class, range bin) and `observe()` draws one.** Measured: p_pole medians for true wall vs true pole are 0.00 vs 0.80 at 200–350 mm, 0.10 vs 0.69 at 900, **0.51 vs 0.55 at 1400, and 0.43 vs 0.43 at 2000** — separated where the sensor is informative, indistinguishable where it is not.
- **`agn_dist_mm` / `agn_dist_sigma_mm` are now emitted**, since the deployed inverse emits them on every ping. This closes the landmine flagged in the entry below.
- **Result after refit:** simulated class accuracy falls with range (97% at 300 mm, 64% at 1200, 50% at 1600) instead of sitting flat at 89%, and `agn_dist` tracks truth (399 / 698 / 1057 / 1441 / 2205 / 2958 mm for true 300–3000).
- **Expect the simulated policy to look *worse* than `default_Path02` did.** It will be training against a sensor much closer to the real one, so sim numbers are not comparable across this change. That is the point of the fix, not a regression.
- Fixed a latent reporting bug while here: a bin with `n == 1` has `sigma = None`, which the printers could not format. Such bins are now marked thin. `_pick_bin` already refused them, so no fitted model was ever affected.
- **Existing policies still load and run** — obs dims stay 4/7/9/14 and the extra key is ignored until the observation channel is added deliberately.

### Class-agnostic range head added to the inverse (2026-08-12)

Commit `8a3936a`. The single adopted outcome of the design experiments in Performance notes 2026-08-12 (night). Predecessors archived at `SonarModel_archive/2026-08-12_deploy_preacq06/` (pre-Acq06) and `2026-08-12_deploy_acq06_noagn/` (Acq06 data, no agnostic head — **the model every number in that Performance-notes entry was computed on**).

- **Two range heads now, answering different questions.** `pole_dist_*` stays masked to pole-class pings within 1 m: the terminal approach stop fires on it and needs an unbiased close-range estimate. `agn_dist_*` trains on every ping at every range and answers "how far is the nearest thing, whatever it is" — a time-of-flight question needing no classification.
- **Out-of-fold, it does not saturate**: predicted mean 450 / 758 / 1062 / 1338 / 1813 / 2157 / 2524 mm across the bands, RMSE 141–401 mm, 12.6–13.7% of range beyond 2 m, z_std 0.85–1.70. Equal for both classes at range (wall 340, pole 303 mm beyond 2000). Deploy split: RMSE **225 mm** against a 546 mm constant-mean baseline.
- **It costs about 2 points of class accuracy** — CV 79.5% ± 2.0% against 81.6% ± 2.7%, every fold same-or-lower, so a real multi-task cost rather than noise. `LOSS_W_AGN_DIST` is the knob. Accepted because class is at chance past 1.4 m regardless.
- **Backwards compatible.** The head is off by default and gated on `architecture.wall3_agn_dist` in `feature_params`; all three deploy-era archives verified still loadable with `agn_dist_divisor` None. `predict_from_envelope` emits `agn_dist_mm` / `agn_dist_sigma_mm` only when the head exists, so callers must check membership as they already do for `pole_dist_mm`.
- ⚠️ **Landmine for the next step: `InverseErrorModel.observe()` does NOT emit `agn_dist_mm`.** The moment `Library/Policy.py` adds the channel, the simulator and the robot will disagree — the same shape of fault as the `p_wall` bug of 2026-08-07, which was silent because `encode_obs` reads with a `.get(..., 0.0)` default. **Add the key to the error model in the same change that adds the observation channel.**

### Geometry-aware yaw selection in the acquisition planner (2026-08-12)

Commit `28b7692`. Motivated by Acquisition06: uniform yaws are geometry-blind, so a session collects whatever range distribution the arena happens to offer, which in an open room meant ~44% of pings under 1 m — a regime the existing 2135 pings already cover densely.

- **`build_plan` gains `yaw_mode`** (`"uniform"`, the unchanged default, or `"far_biased"`), plus `n_far_yaws` / `cone_half_deg` / `yaw_min_sep_deg`. `far_biased` picks the `n_far` headings with the greatest nearest-in-cone range and fills the rest from the *shortest* looks, both greedy under a minimum angular separation, and consumes no RNG — it is a deterministic function of position and arena.
- **`cone_ranges()` deliberately mirrors `AcquisitionSessionLoader.nearest_reflector_in_cone`** — same pole-surface convention, same centre-angle cone test. The planner's job is to predict the label the loader will later attach, so the two must be changed together; there is a comment saying so in both senses.
- **Positions and yaws now draw from separate RNG streams.** They shared one, so any change to yaw selection shifted the position stream and silently produced a different tour, making the two modes incomparable. Verified: at a fixed seed the two modes give byte-identical positions, route and rejection counts. **Cost: plans built before this commit no longer reproduce from their recorded seed.** The saved plan JSONs are the record, so that is a fair trade — but note `AcquisitionArenas/` is gitignored, so those JSONs live only in Dropbox.
- **`AcquisitionPlan` gained `yaw_mode` and `n_far_yaws`, both defaulted**, so `load_plan` still reads pre-existing plans (all six verified loadable) and `reorder_tour` carries them through. The diagnostics panel-1 docstring now records that its "modal heading = bug" reading inverts under `far_biased`.
- **`reorder_tour` truncation is the hazard to watch, not the sampler.** It refuses wall-crossing legs and truncates when the nearest-neighbour tour gets stuck. On the first far-biased build it dropped 15 of 131 waypoints, all from one contiguous region, costing two of the four poles a third of their close-range coverage. `MAX_ATTEMPTS_PER_STEP` 200 → 1000 fixed it (drop of 3). It prints `reorder: truncated at N/M` — read that line.

### PLANNED — the next inverse: a three-way output and a class-agnostic range head (2026-08-09)

**Not implemented.** Design decided 2026-08-09, from Dieter's point that "nothing detectable" and "something there but ambiguous at range x" are different states carrying different information, and the second is still useful to a controller. Depends on Acquisition06.

- **Target output structure.** Three states rather than today's forced wall/pole call:
  1. detected and classified (wall or pole, with its geometry),
  2. **detected, class unresolved, with a range** — a calibrated ~50/50 instead of today's confident `pole`,
  3. nothing detectable (only if such data turns out to exist; see Acquisition06).
- **The architectural gap: nothing currently estimates range for an unclassified object.** The pole-range head is masked to pole-class pings and the wall slices to wall-class pings, so an ambiguous return has no distance at all. **Add a class-agnostic nearest-reflector range head**, trained across the full range, and leave the pole-range head as it is — masked to 1 m, purely the terminal-stop signal. The rationale for expecting this to work is in the record: range is a monaural time-of-flight cue and was the best-performing regression head (106 ± 11 mm), while azimuth needs a binaural comparison. Range should stay recoverable well past where wall-vs-pole discrimination collapses.
- **What it buys.** For Experiment 1, the option to approach an ambiguous distant return to resolve it — active sensing, and biologically apt. For Experiment 2, a landmark distance that means something beyond a metre, which the saturating pole-range head cannot provide (Performance notes 2026-08-09).
- **Target model after Acquisition06, briefly.** Wall vs pole out to ~1.7 m (from ~1.4 today) with close range unchanged — this is the one real bet, since the 1400–1700 band currently rests on 42 wall pings, and beyond ~1.7 m expect little. Beyond that, state 2 above instead of a confident phantom. Wall slices unchanged close in (166 mm at 200–500) and better beyond 1.5 m once far walls are trained rather than extrapolated.
- **Caveat on the 50/50 signal.** p(wall) ≈ p(pole) is only informative if *calibrated*, and today it is not — beyond 1.7 m the model reports 91.5% confidence while being wrong. Calibration in that regime needs examples from that regime, which is exactly what Acquisition06 supplies.

### Uncapped inverse adopted and deployed (2026-08-08)

Commit `67c68c0`. Metrics and the phantom-gate replay in Performance notes 2026-08-08 (later).

- **`FAR_LABEL_MODE` now defaults to `"true_class"`** — the deployed inverse no longer relabels beyond-1 m pings to `none`. Setting it back to `"none"` restores the old behaviour exactly.
- **`POLE_DIST_TRAIN_MAX_MM = 1000`** caps what the *range head* trains on, separately from what the classifier sees. That head regresses toward the mean of its span, so widening the span moved its near-range bias +86 → +149 mm and dragged the terminal approach stop with it. Masking restores +66 mm, better than the model previously flown. **The head then saturates at the ceiling and its σ does not flag it** (z_std 2.56 beyond 1 m) — `trained_max_mm` is in `feature_params.json`, and a reported range near the ceiling means "at least 1 m", not a measurement. Azimuth is deliberately *not* masked: a noisier bearing costs efficiency, a biased range moves the stop, and masking azimuth would leave far poles detected with a meaningless bearing.
- **`SonarModel/inverse_error_model.json` was refitted** and must be refitted after *any* inverse retrain — the simulator draws its perception from it, so a stale file trains the policy against a sensor that no longer exists.
- **The `direct_pole_demo1` phantom gate no longer passes** (6 of 10 phantoms return as `pole`) and cannot, since the model has no abstain output. It is not a valid regression gate for this architecture; replacing it needs Acquisition06 first. Do not read its failure as "the new model is worse" — see the Performance-notes entry, where uncapping is shown to have *shrunk* the phantom zone from ~1.0 m to ~1.4 m.

### Inverse range horizon made configurable and measured (2026-08-08)

Commits `7ffa520`, `8f4bee2`. Numbers and the reasoning behind them in Performance notes 2026-08-08.

- **`SCRIPT_TrainInverseModel.py` gains `FAR_LABEL_MODE`** (`"none"`, the unchanged default, or `"true_class"`), which decides whether beyond-`MAX_RANGE_MM` pings are relabelled to the abstain class or keep their true wall/pole label. **`POLE_DIST_NORM_MM` is split out of `MAX_RANGE_MM`**: the two coincided only because the labelling cut also served as the pole-range normalisation scale, and moving the cut while they are fused silently reweights that head's loss against the others. Defaults are inert — the canonical path is byte-identical.
- **`run_fold` gains `oof_sink`**, an optional list that collects each fold's held-out indices and raw predictions. Calibration and range-stratified questions cannot be answered from the aggregate JSONs, which record only summary metrics.
- **Bug fixed (`7ffa520`): `main()` never passed the pole-distance arguments to `run_fold`**, so the range head trained on no loss in every CV fold while still emitting (untrained) numbers. `main_deploy()` was always correct, so no deployed model was affected — but any pole-range figure read off the CV path before this commit is meaningless.
- **`EXPT_range_horizon.py`** runs the labelling conditions under identical data, folds and seed and writes `TempOutput/RangeHorizon/`. Two design points worth keeping: it reports abstention *separately* from discrimination (raw accuracy conflates them and inverts the conclusion), and it re-analyses from the saved `oof_*.npz` rather than caching report JSON, so updated metric code never gets mixed with stale definitions. `main(only=[tag])` reruns one condition and re-scores the rest for free.
- **Nothing deployable was produced.** The experiment wrote CV folds under `SonarModel/expt_rh_*`; `SonarModel/inverse_*` is untouched and the 1 m deploy model is archived at `SonarModel_archive/2026-08-08_deploy1m/`.

### Experiment 2 unblocked: simulator, observation and deploy chain (2026-08-07)

Commits `450db06`, `f4769f0`, `668af6b`, `269faa8`, `3b6137a`, `1424917`, `834a340`, `ef78c1e`, `19d0f47`.

- **The simulator's sensor is an error model, not an echo simulator.** `SCRIPT_FitInverseErrorModel.py` fits `SonarModel/inverse_error_model.json` from the 2135 labelled echoes: wall-slice bias+sigma, a 3x3 class confusion conditioned on true class **and range**, pole azimuth/range residuals, and phantom-pole geometry. `Library/InverseErrorModel.observe()` emits exactly the keys the deployed inverse emits. It **generates phantoms rather than suppressing them** (15.7% of true-`none` become `pole`, at 611 +/- 149 mm) — a policy trained without them would never meet one. Documented simplification: draws are independent per step, while the real error is likely correlated with pose.
- **`Library/LocalFeature.py`** now holds `LocalFeature`, `feature_from_geometry`, `true_local_feature` and `slice_profile`, extracted from `SCRIPT_RunDirectPolicy.py`. Both experiments rest on the claim that the controller is indifferent to the modality that produced its input; that only holds if every producer emits the same object, so a second implementation free to drift was a real risk. Verified identical on 6000 random poses plus 5 byte-identical seeded rollouts. **The extraction silently deleted four unrelated functions** (`write_run_summary`, `_json_num`, `at_stop_distance`, `alignment_rotation`) because the slice ran to the wrong `def`; it **compiled cleanly** — Python resolves names at call time — and was caught only by the behavioural baseline. Slice to the *next* `def`, and never treat a clean import as evidence.
- **Observation widened.** `Library/Policy.py` gained `_OBS_CLASS` (p_wall/p_pole/p_none), `_OBS_POLE` (az, dist) and `_OBS_POLE_SIGMA`; `make_obs_layout(use_sigma, blind, use_poles)` yields 4/7 wall-only, 9/14 with poles, 1 blind. Pole geometry is **zeroed, not NaN or sentinel**, when no pole is reported. `use_poles` defaults **False** on load so legacy policies still validate (verified: 9d, 4d and 1d all load).
- **Deploy chain migrated** (`ef78c1e`). `SCRIPT_RunPolicy.py` was dead — it still called `SonarModel.load` and passed `sonar_model_dir=`. Now `InverseModel(fold="deploy")`, slice reduction via `LocalFeature.slice_profile`, and the simulator defaults its own error-model path. **The third fault was silent and is the one to remember: `predict_from_envelope` never emitted `p_wall`**, and `encode_obs` reads it with a `.get(..., 0.0)` default — so the robot would have flown with that channel pinned at zero while training saw it vary, with nothing to announce it. The pre-flight preview had the same shape of bug (it built only the three wall distances, leaving all three class posteriors at 0.0 — not a distribution, and a state absent from training); it now uses a new `EnvironmentSimulator.get_clean_measurement()` emitting the full key set with perfect classification and zero sigma.
- **Preview robustness** (`19d0f47`). The end-reason marker map carried `"profile_fail": "?"`, not a valid matplotlib marker — latent until a rollout first ended that way on the real run, taking down the plot *after* all 8 rollouts had been computed, with the robot connected and the operator waiting. Marker is now `"v"`; the plot is wrapped so a drawing fault reports the end counts instead of aborting a live session. (`profile_fail` = no wall ray returned a hit, i.e. the robot left the arena.)
- **`SCRIPT_DefinePath.py`** reads `arena_features.npz` directly (its `EnvironmentSimulator` dependency was itself blocked by the dead `SonarModel.load`) and draws poles with an 800 mm dashed landmark ring. The start pool now samples the drawn `start_box` + `start_arrow`; the 50 mm Gaussian is fallback only, and yaw noise is 20 deg — starting the robot inside a 50 mm box is impractical in the room.

### Model archive convention (2026-07-29)

Superseded inverse models live in `Control_code/SonarModel_archive/<date>_<name>/`, gitignored like `SonarModel/` itself (binary checkpoints; Dropbox is what preserves them, and `git clean -xdf` would remove them). Currently `2026-06-08_2class/` (the model whose phantom poles motivated the abstain class) and `2026-07-29_prerange/` (the deploy model the paper currently reports). **Each snapshot must include `inverse_feature_params.json`:** it is shared across folds, so training rewrites it with the current architecture flags, after which `InverseModel.load` builds *that* architecture for every checkpoint reading it. A `.pth` alone will not roll back. Both snapshots verified loadable from the archive.

### Experiment 1 setup settled + backlog committed (2026-07-28)

Commits `8cebc03`, `aae0f78`, `6ee8b5a`, `fcf04bf`, `e424646`, `dfa9ba8`.

- **Start poses.** `SCRIPT_DigitizeStartPoses.py` records the tape-marked starts by placing the robot and reading the tracker — same pose source as the runs, so marker bias is shared rather than a fresh error term between two measurement paths. Median over N reads, circular median for yaw, spread reported per mark. Output in `TempOutput/StartPositionDigitization/` is a **planning artifact, deliberately untracked**: the experimental record is step 0 of each run's `trajectory.tsv`.
- **Walls-only arena.** Annotate green wall polylines, no blue dabs → `arena_features.npz` with walls plus `pole_radius_mm`, so candidate poles can be injected in simulation. `BuildArenaGeometry` `ROOTS` gained the folder. 3471 wall points, closed boundary, cross-camera agreement **median 16 mm** (90th 25, max 29) — tighter than the 2026-05-26 series.
- **Design decisions, with the reasoning that cost time to find:** a trial is trivial only when the pole is visible at step 0, which needs *both* inside the ±35° cone *and* within the 1 m horizon — filtering on bearing alone discards good trials (a pole 1° off-axis at 2.9 m is invisible) and made some starts look unpairable. Pole separation stays a **hard floor** even though pole position is not a factor of interest, because each placement costs a snapshot/annotate/build/recalibrate cycle; demoting it to a tie-break produced placements 250 mm apart. Sweep seeds derive from coordinates, not candidate index, so scores are stable when the grid changes.
- **Still uncommitted (session state, intentionally):** arena/session names and run counters in `BuildAcquisitionPlan`, `RunPolicy`, `TakeEnvSnapshot`, `TrainPolicy`, `VisualDataAcquisition`, `CalibrateRobot`. Note `CalibrateRobot` `DRIVE_REPEATS` 16 → 12 is arguably a real change, not just state.

### Active thread

**Cross-modal direct-learning inverse for the pole/wall Par 9 task.** End-to-end pipeline now exists:

1. `SCRIPT_BuildArenaGeometry.py` extracts walls (green polylines) AND poles (blue dabs → blob centroid → back-projected through z = 610 mm) per camera, merged across cameras at 50 mm radius. Output: `arena_features.npz` with `x_mm`, `y_mm`, `kind` (0=wall, 1=pole), `source_camera`, `pole_radius_mm`. Pole radius = 12.5 mm (25 mm-diameter wooden dowels).
2. `Library/AcquisitionPlanner` reads walls + poles + radius, enforces pole clearance at `clearance_mm + pole_radius_mm` for waypoints and segments. Plans draw poles as purple circles.
3. `Library/AcquisitionSessionLoader` gained `load_session_inverse` / `load_data_inverse` returning per-ping `class_label`, `pole_azimuth_deg`, and (as of 2026-05-21) `near_dist_mm` via `nearest_reflector_in_cone`. Cone matches the model's ±35° slice cone; tie on distance goes to wall; pole distance measured to surface (centre − radius).
4. `Library/SonarModel.SonarSlicesUQ_TwoHeaded` shares the wall-only trunk and adds a class head (symmetric under L↔R) and pole-azimuth head (mean antisymmetric, log_var symmetric). Symmetry verified by construction. **Wall-head update (2026-06-22):** the deployed architecture is now `SonarSlicesUQ_Wall3(symmetric=True)` ("B") — one shared 3-output wall head run on both ear orderings (left/right swapped, center averaged), replacing the side-head + z_sym-center arrangement; class and pole-az heads unchanged. Chosen as performance-equivalent but simpler to argue (head-variant sweep, Performance notes 2026-06-22). `TwoHeaded` retained for loading older artifacts; `InverseModel.load` dispatches on the `model_class` recorded in `feature_params`.
5. `SCRIPT_TrainInverseModel.py` trains the combined model with CE(class) + masked GNLL(wall slices) + masked GNLL(pole az). Pole-az normalised by /CONE_HALF_DEG to keep antisymmetry exact. Writes to `SonarModel/` with prefix `inverse_`. **Canonical config (2026-05-29):** `ACQUISITION_SESSIONS = ["Acquisition01A", "Acquisition02A", "Acquisition03A"]`, `CV_QUADRANTS = [0, 1, 2, 3]`, `MAX_RANGE_MM = 1000.0`, `EPOCHS = 60`. Script refactored to loop the 4-fold quadrant CV internally — one invocation runs all folds, writes per-fold artifacts (`inverse_q{0..3}_*`) plus aggregated `inverse_cv_results.json`. `split_indices` raises if val-quadrant keys don't match loaded sessions (catches config drift). Legacy wall-only `SCRIPT_TrainSonarModel.py` retired 2026-05-29 (recoverable from git); `Library.SonarModel.SonarSlicesUQ` architecture stays for loading existing wall-only artifacts. **3-class update (2026-06-12):** `CLASS_NAMES = ["wall", "pole", "none"]`, `n_classes=3`; `load_and_filter` now *relabels* empty-cone + beyond-`MAX_RANGE_MM` pings to `none` (an abstain class) instead of dropping them — gives the classifier an explicit "nothing within 1 m" output and confines wall/pole regression to the in-range regime (wall/pole loss masks key off classes 0/1, so `none` feeds only the cross-entropy). `SonarModel.predict_from_envelope` now also returns `p_none`. Sessions are Acq01A–05A. See Performance notes 2026-06-12. **Deployment-training update (2026-06-23):** `main_deploy()` (now the script entry point) trains the single *deployed* model on a spatial 15% holdout per session (`HOLDOUT_FRAC=0.15`, `HOLDOUT_SEED=0`), early-stopping on the held-out region and reporting in-sample vs held-out (`run_deploy` / `spatial_holdout_mask` / `_subset_metrics`); writes `inverse_deploy_*` + the shared `inverse_feature_params.json`. **No generalization claim** — the held-out region is an overfitting guard only; the experiments are the validation ([[project_validation_purpose]]). The 4-fold quadrant CV (`main()`/`run_fold`) is kept for diagnostics/EXPT but is no longer the deployed/reported path. `load_data_inverse` gained `return_poses`. See Performance notes 2026-06-23.
6. `SCRIPT_CheckPoleSignal.py` is a standalone hand-feature LR diagnostic. Rerun after each new acquisition as a signal-floor sanity check before retraining.
7. `SCRIPT_RunDirectPolicy.py` is the Par 9 direct-learning demo: one reactive controller (`ReactiveController`) on a modality-agnostic `LocalFeature` ({pole→azimuth, wall→3 slice depths, empty→scan}), with `SENSE_SOURCE ∈ {sonar, vision, sim}`. Sonar feature comes from `InverseModel.predict_from_envelope` (3-class → `feature_from_inverse`, where class 2 `none` maps to `"empty"`); vision/sim from arena geometry (`feature_from_geometry`, with an optional `max_range_mm` horizon flag `GEOM_RANGE_HORIZON_MM`, default `None` = full sight — training always applies the horizon, deploy vision doesn't need to). Outputs per run: `trajectory.png` (colored by perceived class), `trajectory.tsv` (pose + feature + `p_pole`/`p_none` + referee + action), env/code snapshot, and sonar dills. Pre-emptive pole-stop avoids overshoot-bumping. `INVERSE_FOLD = "deploy"` now loads the spatial-holdout model (2026-06-23). Working-tree edits remain **uncommitted** and entangled with the user's live tuning constants: the `INVERSE_FOLD` repoint plus the earlier 3-class/horizon/`p_none`/dill/color deploy-side additions.

### Empirical results

See Performance notes (below) for the canonical record. Most recent: 2026-05-29 (3-session 4-fold CV, close-range). The 2026-05-21 CNN×distance stratification table is the empirical basis for `MAX_RANGE_MM = 1000.0` (CNN beats LR-baseline at every distance < 1700 mm; both collapse beyond). Pole-az sign trick: regression head doesn't localise reliably but its sign threshold-at-0 is ~68% accurate — keep the regression head, don't train explicit sign-BCE.

### Bisect targets

- **First clean policy deploy: `eef35c7`** (wall-only-policy milestone from the pre-pole era). What unblocked it: recalibrated drive curl (`drive_yaw_curl_deg_per_mm` −0.03693 → −0.01243) and distance scale (0.992 → 0.9972); quantified tracker noise (σ_yaw ≈ 0.6° per fresh frame, fresh-frame rate ≈ 0.8 Hz) and relaxed `wait_for_stable_pose` defaults (`yaw_tol_deg` 0.5 → 2.0, `timeout_s` 5 → 8). The 0.8 Hz cap is a DVR/RTSP bottleneck tracked in `PyLorex/TODO.md`.
- **Pole feature pipeline first end-to-end run: `a7328bc`** (two-headed inverse loader + architecture + trainer; produced the Acq01A-only baseline).
- **Close-range canonical milestone: `f9d4222`** (`Control: restrict inverse training to close-range pings (<= 1 m)`). What unblocked the new numbers: adding Acquisition02A roughly doubles the data, and the close-range filter cuts out the 1–1.7 m dead zone where signal collapses. This is the commit where the 80% / 87% wall recall / 64% pole recall picture stabilises across folds.

### Naming convention

- Wall-only policy training: `<CONDITION>_<TARGET_ARENA>[_blind]` → e.g. `default_Target02`, `default_Target02_blind`.
- Two-headed inverse artifacts: prefix `inverse_` under `SonarModel/`, parallel to the wall-only `slices_` prefix.

The `*.copy` and `code_*.zip` snapshots inside `PolicyRuns/.../files/` keep the old name on purpose — frozen records of the deploy at the time. **Do not rename or modify these.**

### Open items

- **Marker-height correction — distortion-aware ray-plane (2026-05-26).** `Lorex.py:get_aruco` around line 509 computes `floor_xy_mm` via ray-plane intersection at z = `Settings.marker_height_mm` directly. Uses K + dist (for the camera ray, distortion-aware across the full FOV) + R + PnP `t = -R.T @ C_pnp`, all four jointly from `solvePnP`. H_raw is no longer used in the marker pipeline; its linear extrapolation outside the dot-grid calibration footprint was the dominant remaining bias before this change. **Important non-substitution:** plumb-line `C_measured` (from `c_measured_{cam}.json`) is NOT swapped into the ray-plane — PnP's `R` and `t` are jointly fit, and substituting `C` while keeping PnP `R` shifts the principal-point projection by ~300 mm (verified during the 2026-05-26 session). `C_measured` is consumed only at calibration time (by `script_set_camera_center.py`, to derive `shark2tiger_delta_{x,y}` from the inter-camera physical distance). The previous H_raw + λ formulation is still in the code as a numerical fallback if the ray-plane setup degenerates (ray near-parallel to floor). Settings exports `marker_height_mm = 150.0`, `marker_height_tolerance_mm = 20.0`.
- **PyLorex IPPE corner-order bug (noted, NOT fixed).** The function-local `obj_square` in `LorexLib/Lorex.py:detect_aruco` is in CCW (BL,BR,TR,TL) order while `cv.aruco.detectMarkers` returns image corners in CW (TL,TR,BR,BL) order. This makes `IPPE_SQUARE` return a degenerate solution; `t_board_tag[2]` is nonsensical (~2900 mm instead of marker height ~150 mm). The yaw output is *implicitly calibrated* to the wrong R_cam_tag via `aruco_yaw_offset_deg`/`aruco_forward_axis`, so changing corner order without recalibrating yaw would break navigation. Marker-height bias fix above is independent of this (it uses ray-plane intersection, not PnP depth), so no urgency. Fix later by changing corner order + re-calibrating yaw offset together.
- **Camera-system calibration (2026-05-26, mostly resolved).** Geometry anchored to physical measurements via `c_measured_{tiger,shark}.json` (each camera's nadir + height) + `camera_system.json` (derived `shark2tiger_delta_{x,y}` from physical inter-camera distance 2180 mm). All three written by `script_set_camera_center.py`; re-run when cameras or plumb marks move. Cross-camera `|Δ|` is 15–40 mm in arena interior, up to ~125 mm at x extremes — calibration-grade for the pole/wall paper. Residual `dx`-linear-with-x gradient is most likely a small relative R/t mismatch between cameras; full bundle adjustment would fix it but isn't a blocker. See Performance notes 2026-05-26 for the comparison series.

- **Plank-based bundle calibration — implemented as Phase 5 checker, Phase 6 source deferred (2026-05-27).** Tooling: `PyLorex/script_{capture,diagnose,calibrate}_plank.py`. Full procedure in `PyLorex/Docs/calibration_process.md`. Phase 6 (adopt bundle outputs as canonical calibration) blocked on `Environment.py` warp refactor — without that, build-geom and runtime tracker would use different projection frames and `arena_features.npz` would overlap walls by 500–700 mm at the periphery. Plank-residual std ≈ 25 mm against the bundle calibration.

- **PyLorex calibration tooling.** Scripts under `PyLorex/`: `script_collect_intrinsics_live.py` (hands-free intrinsics with coverage map), `script_set_camera_center.py` (writes `c_measured_*.json` + `camera_system.json`), `script_check_camera_agreement.py` (manual-placement cross-camera floor-projection diagnostic, applies `shark2tiger_delta` internally), `script_run_homography.py` (per-dot H_raw fit diagnostics; `ransac_thresh_px` arg is actually in destination units / mm), `script_generate_aruco_markers.py` (per-ID ArUco SVGs + tiled sheets for laser cutting), plank tooling (see Plank item above). Per-script details live in the script docstrings.
- **Decision threshold / class balance tuning.** Classifier uses plain argmax. With the 3rd `none` class (2026-06-12) pole recall dropped to ~70% (from ~78%) via a pole↔none trade-off; a lowered pole threshold or CE class-weighting could recover some pole recall without much wall cost. Revisit after seeing the 3-class model deployed.
- **Blind ablation thread is paused.** Plumbing exists (`BLIND` flag in `SCRIPT_TrainPolicy.py`; `blind` flag in `Library/Policy.py` accepted everywhere; `meas_dict` allowed to be `None`). Resume after the pole/wall inverse has a stable Acquisition baseline.
- **PyLorex frame-rate bottleneck.** Server reports 7-8 Hz internally; client sees ~0.8 Hz of distinct reads. Tracked in `PyLorex/TODO.md`. Once fixed, tighten `wait_for_stable_pose` defaults back toward the pre-2026-05-10 values (yaw_tol 0.5°, timeout 5s).
- **Re-measure tracker noise.** `SCRIPT_MeasureTrackerNoise.py` was last run before the calibration overhaul; current `wait_for_stable_pose` tolerances (yaw_tol_deg = 2.0, timeout_s = 8) were sized off the 2026-05-11 numbers (σ_yaw ≈ 0.6°). Since then calibration was overhauled (2026-05-26), the 81 mm marker was fitted, and the projection switched H_raw → ray-plane. Visual data acquisition settles fine, so tolerances aren't too tight — but they may now be looser than the actual noise floor. Re-running takes ~30 s and would let us tighten and speed up per-waypoint settling.
- **Re-run robot calibration before next policy deploy.** `SCRIPT_CalibrateRobot.py` (drive_yaw_curl_deg_per_mm, drive_distance_scale) drifts with battery state, gear wear, and tire compression. The pre-pole-era clean deploy (eef35c7) was specifically unblocked by re-tuning these constants. Before the next `SCRIPT_RunPolicy.py` deployment on the real robot, re-calibrate so the drive model matches current robot state — flagged 2026-05-29 by dieter as a "don't-forget" gate.
- **Per-robot calibration.** `Settings.py` `ClientConfig` `default_factory` carries Robot01's calibration table; `client2` / `client3` inherit it. Only matters when actually deploying on those robots.
- **Diagnostics directory.** `Control_code/Diagnostics/` is currently untracked. If outputs there ever become worth versioning, add to `.gitignore` explicitly so the policy is intentional.
- **Uncommitted working-tree changes (long-standing user WIP, do not bundle into agent commits):** `Library/Settings.py` (rotation table + drive curl/scale, ongoing), `Library/DataProcessor.py`, `SCRIPT_CalibrateRobot.py`, `SCRIPT_RunPolicy.py`, `SCRIPT_TakeEnvSnapshot.py`, `SCRIPT_TrainPolicy.py`, `SCRIPT_BuildAcquisitionPlan.py`, `SCRIPT_BuildArenaGeometry.py`, `SCRIPT_ProbeTrackerAtWall.py`. Also `Paper/main.tex` carries the user's Par 10 Holland/Finger insertion uncommitted. **`SCRIPT_RunDirectPolicy.py` additionally carries agent-authored deploy-side work (3-class `none` handling, `GEOM_RANGE_HORIZON_MM`, `p_none` logging, trajectory color, sonar dills) that is validated but uncommitted** because it interleaves with the user's live tuning constants (`DRIVE_MM`, `BOUNCE_TRIGGER_MM`, `WALL_JAM_MM`, `SENSE_SOURCE`, `SESSION`) — commit it with those hunks reverted/restored, as in prior agent commits. Always inspect `git diff` before staging anything in this repo.

---

## Performance notes

Chronological record of model and robot-experiment performance, written when measured. Each entry should include the date, what was measured, the config (sessions, key flags, model identity), the commit at the time of measurement, and the metrics — enough to be interpretable months later without re-deriving anything. **Append new entries at the top so the most recent is read first.** Don't edit older entries; if a measurement is re-done later, write a new entry that references the prior one.

This exists because `SonarModel/`, `PolicyTraining/`, and `PolicyRuns/` are all gitignored, so per-run JSONs get overwritten and historical numbers are otherwise lost.

### 2026-08-13 (evening) — Recalibration before the Path04 deploy: yesterday's drive constants were outliers

`SCRIPT_CalibrateRobot.py`, both phases, `DRIVE_MM=150` × `DRIVE_REPEATS=15`, 5 reps per angle. Raw samples in `Library/RobotCalibration/Robot01_calibration.json`. **This resolves the uncertainty flagged in the 2026-08-12 entry.**

| date | curl (deg/mm) | scale | drive reps |
|---|---|---|---|
| 2026-06-08 | −0.01993 | 0.9873 | — |
| 2026-08-07 | −0.01244 | 0.9807 | 10 |
| 2026-08-12 | **−0.02609** | **1.0036** | 15 |
| **2026-08-13** | **−0.01444** | **0.9772** | 15 |

- **Both of the 2026-08-12 drive constants were bad measurements, not drift.** Curl is back to −0.01444, inside the −0.012 to −0.020 range of the other three sessions; −0.02609 stands alone. Scale is back below unity at 0.9772, near 0.9807 and 0.9873; 1.0036 remains the only reading above 1.0 ever recorded. **The 2026-08-12 entry's worry that "the two drive constants moved a lot and do not share one story" is retired** — they moved because the measurement was noisy.
- **The deployed curl is now close to what run02's behaviour implied.** run02 suggested an in-run curl of −0.0177 against −0.0124 measured at the time; −0.0144 sits between them. The 2026-08-07 "unresolved discrepancy" looks less like a protocol mismatch and more like between-session measurement noise, consistent with what the 2026-08-12 entry already suspected.
- **The rotation table is better than yesterday's.** Mean |error| **0.78°** against a mean per-angle sd of **1.04°** (2026-08-12: 1.45° against 1.00°). Yesterday's clearest real effect — ~3° under-rotation at ±40° — is gone: now +1.38 / −1.16.
- Four of ten entries still exceed one sd (−40, −30, −20, +40) but they do not form a pattern: two over-rotate, two under. With 5 reps per angle that is about what chance gives. Do not read individual cells.
- **−30 is the flakiest cell for the third session running**: error −1.86°, sd 1.22, range 3.27°. Flagged on 2026-08-07 (one wild rep) and 2026-08-12 (sd 1.72, range 4.97). Three sessions makes it a pattern rather than luck — something about that command is less repeatable than its neighbours. Not worth chasing now; worth remembering if a run misbehaves on left turns near 30°.

### 2026-08-13 (evening) — Path04 policy trained: comparable loss to Path02 on a harder sensor, 5-10% collision rate

`PolicyTraining/default_Path04/`, commit at training `83363e0`. Config: `use_poles=True`, **`use_agn=True`**, `use_sigma=False`, `max_dist_mm=2500`, `motion_rot_bias_deg=3.0`, obs width **10**, 2000 epochs.

- **`val_mse` 114.3 (RMSE 10.7° on the rotation command), against `default_Path02`'s 114.65** — essentially identical, and that is the result. This policy faces a materially harder sensor: no class oracle, no abstain flag, a horizon that no longer goes blind at 1 m, and an additive rotation bias the old one never met. Matching the old loss under those conditions is what the day's fixes bought. (`default_Path01_figure8` was 165.1 for scale.) Train and val track closely with no overfitting gap.
- **Batch evaluation, 60 rollouts from the release box, `max_steps=150` (2.6 laps if perfect):**

  | | |
  |---|---|
  | collisions | 3/60 (5%) |
  | laps completed | median **2.52** of 2.6 |
  | cross-track error | median **55 mm**, p90 146, p99 284, max 750 |
  | steps beyond the 370 mm usable margin | **0.3%** |

- **Collision rate vs the per-episode bias** (n=100 each): 6% at zero bias, 9% at the measured 1.08°, 15% at the training maximum 3.0°. **The bias is not the dominant cause** — the per-step 3° Gaussian and the gain noise produce 6% on their own. Note these are noisy: at n≈100 and p≈0.08 the standard error is ~2.7%, and the runs used different seeds, so 6% and 9% are not distinguishable. **Read the collision rate as roughly 5-10% per 2.5-lap run.**
- **Every collision is with B1 (−31,−612) or B2 (−394,−2137)**, the two interior blocks the path threads between — never the pole, never the boundary. Locations cluster at (−200…−500, −1900…−2250) and (0…180, −500…−820). If the rate wants reducing, nudging the path away from those two is a small edit rather than a redraw.
- **For scale on how much better this is than Path02:** that path crashed on its *first* real run at lap 3, with 49% of its length tighter than the robot's p90 tracking error. Here 0.3% of steps exceed the margin.
- **Caveat.** All of the above is simulation, and the simulator is now much closer to the real sensor than it was but still fitted from 2775 pings in five arenas. The number that matters is the robot.

### 2026-08-13 (later) — Path04 v3 adopted: safe, smooth, and perceptually adequate

Three redraws, and the iteration is the lesson. Numbers below are from `SCRIPT_AnalysePathRun.py`, which now reports drift detectability directly (commit for that alongside).

| | Path02 | v1 | v2 | **v3 (adopted)** |
|---|---|---|---|---|
| min clearance | 155 | 293 | 472 | **455** |
| usable margin | 70 | 208 | 387 | **370** |
| median clearance | 355 | 470 | 603 | 560 |
| median nearest-AHEAD | 614 | — | 1067 | 946 |
| steps with <1 m ahead | 86% | — | 42% | 56% |
| detect 200 / 300 mm lateral | 76 / 93% | 75 / 92% | 56 / 73% | **66 / 81%** |
| turns median / max | 17 / 61° | 19 / 59° | 16 / 71° | **14 / 46°** |

- **v3 is the first path that is simultaneously safe, smooth and usable.** Its 370 mm usable margin exceeds even the largest tracking error ever recorded (341 mm), against Path02's 70 mm; turns max 46° against the policy's 90° `max_rotate_deg`.
- **v2 is the instructive failure.** Told to widen four pinch points, the redraw pushed the *whole* path out — median clearance 470 → 603 mm — and lost 19 points of detectability while remaining perfectly safe and "informative". **The mechanism: it ended up looking at things 1–1.4 m away, where the model's error (236–401 mm) exceeds the change a 200 mm sideways shift produces.** The readings were there and useless. Clearance and "informative" both missed this, which is why the diagnostic was added.
- **What is AHEAD is the controllable quantity, not clearance.** Clearance is omnidirectional; the cone is forward. A path can sit 470 mm from a wall to its side and see nothing closer than a metre in front. Aim legs *at* things and turn away at ~550 mm.
- **But the sub-500 mm band is a genuine trade, not a drawing error.** To have something 400 mm ahead your clearance to it is 400 mm. v3 has 0% of steps under 500 mm ahead against Path02's 33%; that gap cannot be closed without giving back the safety margin, and should not be. It is most of the residual difference in detectability.
- **Why 66% is judged sufficient.** The metric is per-step and pessimistic: it asks whether a *single* reading moves detectably. Drift is persistent and the RNN accumulates, so a displacement visible on two steps in three is caught within a few steps. run02 held lap-to-lap at 116 mm on 76%; 66% is the same regime.
- **Caveats on the metric, since it has been wrong before.** It uses the model's measured per-band σ against *true* geometry, so it is an upper bound on what is detectable — real detection also needs the estimate to be unbiased, and there is a known +50 to +87 mm bias below 1.5 m. And it treats steps independently, which understates paths whose evidence accumulates.

### 2026-08-13 — Path04 arena and path: the right metric is drift detectability, and by it the drawn path is fine

Offline geometry only. **The main content of this entry is a correction: two metrics used earlier to judge paths are wrong, and the conclusions drawn from them were wrong.** Artifacts in `TempOutput/BlockLayout/` (gitignored).

- **The arena was rebuilt.** Path03 had only 2 interior blocks — the room was cleared for Acquisition06 and little went back — which is why it faced the boundary 79.7% of the time. Path04's arena has **4 blocks** at (−31,−612), (−394,−2137), (1180,593), (1464,−1766), plus the pole at (1415,−624). It holds a 10.6 m loop at 435 mm clearance while offering 55.2% object-facing, a combination neither predecessor had: Path02's arena loses its loop entirely above 350 mm clearance, which is why that path ran at 155 mm and crashed.
- **WRONG METRIC 1: object-facing fraction predicts nothing.** A generated smooth loop ("cand2") faced objects 65.7% of the time, better than Path02's 62.5%, and was five times worse at localising. Do not use it to choose a path.
- **WRONG METRIC 2, and this one is in the record: `EXPT_gaze_path.py`'s distinctiveness ignores motor history.** It asks whether two poses are confusable given their *sensory* windows. The RNN's state is driven by motor commands as well, and on a loop with varied turns those are nearly a unique signature. Including commanded rotation in the feature collapses the differences: aliasing goes 50.7 → 5.3% (Path01), 82.4 → 5.4% (Path03), 71.4 → 5.7% (cand2), and every path becomes essentially unambiguous. **The path ranking that metric produced is an artifact.** (Dieter's correction. Absolute numbers from `distinct_motor.py` are not comparable to the gaze script's — different features and tolerances — but the sensory→+motor direction is the finding.)
- **Why even the corrected version is not the question.** The motor sequence is what the policy *commanded*, not an observation of where the robot *is*. It pins position on the *intended* path; the whole problem is that execution drifts from intention at −1.08°/step. So motor history answers "where am I along the loop" and cannot answer "how far have I drifted off it".
- **RIGHT METRIC: can perception detect a displacement from the path?** Fraction of steps where the three slice readings shift by more than the model's measured per-band σ when the robot is displaced:

  | path | min clr | 100 mm lateral | 200 mm | 300 mm | 10° yaw |
  |---|---|---|---|---|---|
  | Path02 | 155 | 34% | **75%** | **93%** | 41% |
  | Path01 | 220 | 27% | 61% | 77% | 29% |
  | **Path04 as drawn** | 293 | 18% | **75%** | **92%** | 33% |
  | Path03 | 438 | 11% | 42% | 55% | 15% |
  | cand2 (generated) | 482 | 11% | 29% | 46% | 29% |

  **Path04 as drawn matches Path02** at the 200–300 mm displacements that matter against a 266 mm p90 tracking error. Both generated candidates are much worse.
- **Proximity does matter, but the threshold is lower than the earlier entries claim.** Detectability needs the readings to be sensitive to displacement, which needs nearby geometry: a wall at 500 mm shifts measurably when the robot moves 200 mm sideways, one at 1.5 m barely does. Path04 achieves it with **median clearance 470 mm** and min 293. Path03 at min 438 does not. So the requirement is a *median* around 450–500 mm, not a 435 mm *minimum*. **This also qualifies the 2026-08-08 claim that the clearance/perception trade-off "does not bind": it does not bind for informativeness (`cls != none`), but it does for drift detectability, which is what Experiment 2 actually needs.**
- **Recommended: keep Path04's shape; push four pinch points out to ~350–400 mm.** Steps 22–28 (293 mm, near (1222,−1375)→(1027,−2102)), 38–43 (309 mm), 17–19 (359 mm), 10–12 (385 mm) — six of 61 steps. Detectability comes from the path's overall proximity rather than its tightest points, so this should be safe, but re-run the check after: the metric has been wrong twice.
- **Two generated candidates were built and rejected, both instructive.** A star-shaped tour aiming at each block in turn scored best of anything on distinctiveness (101 mm) but had **median 126° / max 177° turns against the policy's 90° `max_rotate_deg`** — undrivable. Constraining turns to ≤55° produced a smooth loop that was safe (397 mm usable) and useless (29% detectability at 200 mm). **Facing an object from a distance requires pointing at it, and pointing requires turning**; smooth loops can only face things by passing near them, which is what Path02 did.

### 2026-08-12 (night) — What the inverse can and cannot do, settled by experiment: distance survives to 2.5 m, azimuth and class die at 1.4 m

Six experiments on the retrained model, all out-of-fold on Acq01A–06A, all reusing the same folds and seed. **This entry supersedes several claims in the entry below it** — see "corrections" at the end. Artifacts under `SonarModel/inverse_{agn,agnaz,c50,c70}_q*` (diagnostic only; `inverse_deploy_*` untouched).

- **The one result that matters most: a CLASS-AGNOSTIC range head works far past where everything else fails.** Same architecture, same folds; only the range head's target changed from "distance to the pole (pole pings, ≤1 m)" to "distance to the nearest in-cone reflector, any class, no cap".

  | true range | n | predicted mean | bias | RMSE | % of range | z_std |
  |---|---|---|---|---|---|---|
  | 0–500 | 682 | 473 | +93 | 198 | 51.9% | 0.98 |
  | 500–1000 | 1163 | 768 | +36 | **138** | 18.8% | 0.92 |
  | 1000–1400 | 509 | 1088 | −87 | 220 | 18.7% | 0.95 |
  | 1400–1700 | 149 | 1325 | −215 | 401 | 26.1% | 1.33 |
  | 1700–2000 | 96 | **1793** | −59 | 320 | 17.2% | 1.28 |
  | 2000–2500 | 110 | **2167** | −84 | **244** | **10.9%** | 0.84 |
  | 2500+ | 66 | **2579** | −158 | 320 | 11.7% | 0.76 |

  **No saturation, and σ is honest** (z_std 0.76–1.33 against the masked pole head's 2.1–2.9). Decisively, it works *equally well for both classes where telling them apart is a coin flip*: at 2000+, wall RMSE 269 mm and pole RMSE 280 mm, both 11.3% of range. Cost is near-range precision (0–500 RMSE 198 vs the masked pole head's 110–151, and near-range *pole* ranging degrades to 269 mm because walls outnumber poles in the shared target) — **so add this head and keep the pole-range head masked at 1 m as the terminal-stop signal**, exactly as the 2026-08-09 design proposed. No collateral damage: class accuracy 0.812–0.828 vs 0.792–0.847, wall slices 254/301 vs 260/289 mm.
- **The three cues fail at different ranges, and the pattern is physically coherent:**

  | cue | mechanism | usable to |
  |---|---|---|
  | **distance** | monaural time-of-flight — only needs an echo detected | **2500+ mm**, ~11% error |
  | azimuth | binaural comparison — needs SNR in *both* channels | ~1400 mm |
  | class | fine temporal/spectral structure | ~1400 mm |

  Anything requiring a comparison or fine structure dies together at 1.4 m; the cheap cue survives.
- **Widening the cone does NOT help** (`CONE_HALF_DEG` 35 → 50 → 70, one constant, everything else identical). Lift over the per-band majority baseline: at 0–500, +18.8 / +18.2 / +14.8; at 500–1000, +25.6 / +23.0 / +19.8; at 1000–1400, +13.8 / **+15.2** / **+0.4**. Overall 81.6% / 80.4% / 76.9%. ±50 is a wash; ±70 destroys the last informative band. **Critically, the collapse happens at ~1400 mm at every cone width, so the boundary is not a labelling artifact.** Outer slices at ±70 cost 397–453 mm RMSE against ±35's 271–331, because they cover 23–70° off-axis where the beam barely hears; the centre slice improves (210–278 vs 254–301) since it is wider and its min-depth more robust.
- **Simplifying to (nearest distance, nearest azimuth, p(pole)) was tested and rejected.** Class-agnostic azimuth *is* learnable close in — wall bearing RMSE 14.1° at 0–500 against a 26.7° zero-predictor baseline, left/right 88% — refuting the worry that an extended surface has no recoverable bearing. But it hits the same cliff: 26.0° at 1400–2000 against a 25.3° baseline, and 29.8° vs 25.5° at 2000+, i.e. **worse than always saying "straight ahead"**. It also costs the 3-slice wall profile (still good at 300 mm RMSE at 1500–2000, and the main spatial signal for Exp 2) and degrades pole bearing (left/right 63–74% vs the pole-only head's 81%, since walls outnumber poles 1765:1010). **No range gain, real losses.**
- **Off-cone competition is real but the decisive test is impossible in this room.** Controlling for range, at 1000–1400 mm: "clean" poses (nothing within ±70° more than 100 mm nearer than the target) score **80.6%** (n=139) against **65.1%** for contested (n=370) — a 15-point effect at matched range. *An earlier version of this analysis was confounded*: binning by off-cone advantage without controlling range also sorts by range (mean 1220 → 2180 mm across the bins). Above 1400 mm the clean set is **15, 11 and 1 pings** — 27 in total. That is not a sampling failure but the room's ceiling: the most open waypoint sits ~1.65 m from the nearest wall, and a side wall 1.75 m away still enters a ±70° window at ~1.86 m, so uncontested targets **cannot exist beyond ~1.9 m here**. **Whether the sonar could classify an uncontested object at 2.5 m is untested and untestable in this arena** — it needs a space ~5 m across or an absorbing boundary. Mechanism remains open: direct temporal overlap is ruled out (echoes 800 mm apart arrive ~4.7 ms apart, easily separable); live candidates are attribution, dynamic range under global normalisation, and reverberant masking from a strong near reflector.
- **The classifier is already a calibrated probability, and we were discarding it.** Fitted temperature **1.00**, overall ECE **0.010** — no scaling needed. Thresholding gives a clean reliability curve: p ≥ 0.7 → 92.0% accurate on 69.2% of pings; p ≥ 0.9 → **97.9%** on 43.9%. Low-confidence pings land monotonically at range (8.7% of the 0–500 band below p = 0.7, rising to 89.4% beyond 2500). **So the "class unresolved" state needs no third class and no label-space cut — it is p ≈ 0.5, and it already exists.** Caveat, which flattered an earlier reading: p's selectivity comes largely from tracking *range*; within the far bands it does not separate right from wrong.
- **`feature_from_inverse` throws that away.** It takes `int(pred["class_label"])`, the argmax, and returns a hard string; `p_pole`/`p_none` ride along as logging only. And since `none` has had zero training members since 2026-08-08, **`p_none` maxes at 6.5e-4 and argmax never picks class 2 in any of 2775 pings** — the `cl == 2` → `"empty"` branch is dead code, so the controller's scan-instead-of-chase behaviour cannot fire. Fix is a confidence threshold in that one function, not a model change.
- **Path01 re-scored at each horizon** (offline, error-model-free: true geometry per 150 mm step, weighted by the model's measured per-band accuracy). E[correct]: 51.9% at a 1000 mm horizon → **71.9% at 1400** → 74.9 / 76.9 / 77.2% at 1700 / 2000 / 2500. On correct-**and**-confident (p ≥ 0.8) it is flat from 1400 onward at 50.8%. **The useful horizon is 1400 mm; beyond it buys nothing.** Path01's weakness is that 30.1% of its steps sit in the 1000–1400 band where accuracy is 65.4% and confident-correct only 29.0%.
- **Corrections to the entry below.** (a) "1400–1700 mm is now the worst band anywhere" is **not supported** — bootstrapped, 1000–1400 beats it by +21.7 pp [CI +12.8, +30.6, P<0.001], but 1700–2000 by only +11.7 pp [CI −0.9, +24.4] and 2000–2500 by +8.7 pp [CI −3.7, +20.6]; all four far bands have overlapping CIs. There is a cliff at 1400 and then a flat plateau near chance, and the apparent dip was noise on n=149. Its ECE of 0.192 is estimated on the same 149 pings and is no more trustworthy. **Option (c) in Where-to-pick-up is therefore closed.** (b) Near-ties are a real but small effect — when a wall and pole sit within 200 mm of each other in the cone the label is acoustically arbitrary; prevalence 19.3% / 24.8% / 42.5% across 1000–1400 / 1400–1700 / 1700–2000, and excluding them lifts those bands by only 3–5 points. (c) "Far range is uninformative" should read **uninformative for class and azimuth, not for distance**.

### 2026-08-12 (late) — Inverse retrained on Acq01A–06A: the phantom is largely fixed, far range is better but not solved, and 1400–1700 mm is now the worst band anywhere

Commit at measurement `cb0545c`. `main_deploy()` then `main()` (4-fold quadrant CV), **config otherwise untouched** — only `Acquisition06A` was added to `ACQUISITION_SESSIONS`, so everything here is attributable to the data. 2775 pings, 930 beyond 1 m kept at true class, max 3196 mm. Predecessor archived at `SonarModel_archive/2026-08-12_deploy_preacq06/` (verified loadable, includes its error model as the "before" reference).

- **Deploy model:** held-out class acc **79.2%** (n=418), in-sample 84.7%, pole-az RMSE 13.81°, wall RMSE R/C/L 257/372/314 mm, best epoch 59. **Do not compare 79.2% against the predecessor's 86.6%** — the held-out set now runs to 3.2 m where discrimination is genuinely harder, so the aggregate had to fall. Same trap as the 2026-08-08 entry's warning about 86.6 vs 83.9.
- **CV:** class acc **81.6% ± 2.7%** (per-fold 84/79/85/79), wall prec/rec 87.4/83.0, pole prec/rec 72.8/79.0, pole-az 15.56° ± 0.75°, wall RMSE 293/312/325 mm.
- **The controlled comparison is out-of-fold on Acq06A only** — new = CV OOF, old = fully unseen (it never saw any Acq06 data). The old model is in-sample on Acq01A–05A, so any table including those pings flatters it.

  | band | n (wall/pole) | new acc / bal | old acc / bal | new wall rec | old wall rec |
  |---|---|---|---|---|---|
  | 0–500 | 65/33 | **85.7 / 78.8** | 81.6 / 72.7 | 100% | 100% |
  | 500–1000 | 64/71 | **78.5 / 79.6** | 65.2 / 66.8 | 100% | 98.4% |
  | 1000–1400 | 30/49 | 64.6 / **69.5** | 67.1 / 67.0 | **90.0%** | 66.7% |
  | 1400–1700 | 37/30 | 38.8 / **37.7** | 49.3 / 54.1 | 48.6% | 8.1% |
  | 1700–2000 | 51/34 | **58.8 / 60.8** | 40.0 / 50.0 | **51.0%** | **0.0%** |
  | 2000–2500 | 57/53 | **56.4 / 57.6** | 48.2 / 50.0 | **22.8%** | **0.0%** |
  | 2500+ | 25/41 | 48.5 / 39.0 | 62.1 / 50.0 | **0.0%** | **0.0%** |
  | overall | 640 | **64.2 / 64.2** | 59.7 / 60.1 | | |

- **The phantom is largely fixed.** The old model's wall recall is **exactly 0.0% in every band beyond 1700 mm** — it never says wall out there, and its 50.0% balanced accuracy in those bands is the arithmetic signature of always guessing one class. Its apparently respectable raw accuracy at range is entirely the class composition of the pings it was scored on. The new model recovers 51.0% / 22.8% wall recall at 1700–2500 and wins the fair test overall.
- **Two bands went backwards, and the pattern is not monotonic in range.**
  - **1400–1700 is the worst band anywhere: balanced accuracy 37.7%, below chance, with confidence 23 points ahead of accuracy.** Worse than the harder bands on either side. This is precisely the band where the two datasets mix ~50/50 (82 old-arena pings, 67 Acq06 ones); pooled, the model gets 55.0% there on old-session pings against 38.8% on Acq06 ones. Hypothesis worth testing: the same nominal range in a cluttered room and in an open one are different acoustic problems, and off-cone clutter is the confound.
  - **2500+ has collapsed back to always-pole** (wall recall 0.0%), on 25 wall examples in the whole dataset at that range. Not enough data, not a model defect.
- **Calibration improved where accuracy did** (500–1000 gap −4.4, 1700–2000 +0.9) and stayed bad where it did not (1400–1700 **+23.4**, 2500+ +11.7).
- **A methodological correction worth remembering.** The first pass compared the two models on the deploy run's single 96-ping spatial holdout and concluded the new model was much *worse* at range (15/51 on far pings). That was 51 pings in one contiguous patch, and the CV reversed the sign. **One spatial holdout patch is not an estimate of far-range behaviour** — use the 4-fold OOF for any range-stratified claim.
- **This locates the boundary the three-way output was designed around.** The 2026-08-09 note assumed a "detected, class unresolved, with a range" state was needed beyond *some* range. The data now measures it: balanced accuracy is 69.5% at 1000–1400 and at chance from 1400 up, while confidence stays high. Forcing a wall/pole call above ~1400 mm is what manufactures a confident wrong answer; that is where state 2 belongs.
- **`SonarModel/inverse_error_model.json` is STILL THE PRE-ACQ06 FIT.** The deploy model was replaced but the error model was not refitted, so `SonarModel/` is internally inconsistent — **nothing may be trained against the simulator until it is refit.**

### 2026-08-12 (evening) — Acquisition06A collected: the far-range hole in the training set is filled

Session `AcquisitionSessions/Acquisition06A`, arena `Acquisition06/env_0001_2026-08-12T12-44-36`, plan `plan_2026-08-12T13-28-12.json` (far-biased, 128 × 5). Commit at measurement `cfb4dc2`.

- **Ran clean: 640 of 640 pings kept, zero tracker misses**, so no pings were lost to `drop_pose_fallback`. Executed poses sat a median **18 mm** from their planned waypoints (p90 31, max 148) — the recalibration plus closed-loop nav held.
- **The plan predicted the yield almost exactly**: 314 far (1.5–3.5 m) pings achieved against 313 planned, 201 beyond 1911 mm against 197, identical p50 of 1441 mm, max 3196 vs 3175 mm. So the offline `cone_ranges` scoring is trustworthy for sizing future sessions — plan first, measure second.
- Session class balance: wall 329 / pole 311 (48.6% pole), no empty cones.
- **Pooled training set, before → after** (nearest-in-cone, ±35°):

  | band | Acq01A–05A | Acq01A–06A | wall | pole |
  |---|---|---|---|---|
  | 0–500 | 584 | 682 | 517 | 165 |
  | 500–1000 | 1028 | 1163 | 747 | 416 |
  | 1000–1400 | 430 | 509 | 283 | 226 |
  | 1400–1700 | 82 | **149** | **79** (was 42) | 70 |
  | 1700–2000 | 11 | **96** | 57 (was 6) | 39 |
  | 2000–2500 | **0** | **110** | 57 | 53 |
  | 2500+ | **0** | **66** | 25 | 41 |
  | total | 2135 | **2775** | 1765 | 1010 |

  **Beyond 1700 mm: 11 pings → 272.** The band the 2026-08-09 design note called "the one real bet" (1400–1700, resting on 42 wall pings) now carries 79, and every new band is roughly class-balanced rather than one-sided.
- **Consequences for the numbers quoted in earlier entries.** The 2026-08-08 statement that "cap1700 is fullrange, not fullrange with a guard" rested on the abstain class having 11 members beyond 1700; that is now 272, so every candidate-cut count in that entry is stale. Data now tops out at **3196 mm**, not 1911, so "uncapped means out to ~1.7 m on this dataset" no longer holds either.
- **Watch item for the retrain:** the top band is pole-heavy (2500+ is 25 wall / 41 pole), so the far-range prior will lean pole — the same direction as the phantom failure. The honest check is far-range **calibration** (is p(pole) right when it says 0.6?), not accuracy alone.

### 2026-08-12 — Robot recalibration before Acquisition06: small-angle over-rotation gone, large-angle under-rotation new, drive constants moved a lot

`SCRIPT_CalibrateRobot.py`, both phases, 5 reps per angle, `DRIVE_MM=150` × **`DRIVE_REPEATS=15`** (was 10 on 2026-08-07). Raw per-angle samples preserved in `Library/RobotCalibration/Robot01_calibration.json`. Constants live in `Library/Settings.py` (uncommitted user WIP, as always). Compare against the 2026-08-07 (evening) entry.

| cmd | new median | error | sd (5 reps) | 2026-08-07 | 2026-06-08 |
|---|---|---|---|---|---|
| −40 | −36.67 | **+3.33** | 1.19 | −40.01 | −39.80 |
| −30 | −31.67 | −1.67 | 1.72 | −29.91 | −29.79 |
| −20 | −20.93 | −0.93 | 1.29 | −18.74 | −20.79 |
| −10 | −11.94 | −1.94 | 0.81 | −11.19 | −10.92 |
| −5 | −5.47 | −0.47 | 0.81 | −6.69 | −5.13 |
| +5 | +4.75 | −0.25 | 0.41 | **+6.98** | +5.14 |
| +10 | +11.23 | +1.23 | 0.78 | +11.19 | +10.65 |
| +20 | +20.33 | +0.33 | 0.64 | +20.55 | +19.62 |
| +30 | +28.51 | −1.49 | 1.04 | +28.98 | +28.18 |
| +40 | +37.18 | **−2.82** | 1.34 | +38.74 | +38.80 |

- **Drive:** curl **−0.02609** deg/mm (was −0.01244, before that −0.01993), scale **1.0036** (was 0.9807, before that 0.9873).
- **The fresh-battery small-angle over-rotation has reverted.** 2026-08-07 measured ±5 → +6.98/−6.69 with fresh batteries and read it as a real effect; it is now +4.75/−5.47, i.e. back to the 2026-06-08 behaviour. That supports the battery-state explanation rather than motor wear.
- **New and in the opposite direction: ±40 now under-rotates by ~3°**, in both directions, where both previous tables were accurate there. At −40 the error exceeds the per-angle sd. Large rotations falling short while small ones are fine is what one would expect from less available torque.
- **Most of the rest is noise.** Mean |error| across the table is 1.45° against a mean per-angle sd of 1.00° on 5 reps (ranges up to 5°). The ±5, ±20 and −30 entries are within one sd of zero error — do not read them as real effects. **−30 is repeatedly the flakiest entry**: sd 1.72 and a 4.97° range here (samples −27.73, −30.80, −31.86, −31.67, −32.70), and the 2026-08-07 run also flagged −30 with a wild rep. The +5 cell has only 4 samples; one rep was lost, presumably a tracker miss.
- **The two drive constants moved a lot in five days and do not share one story.** Curl doubled, and the scale crossed 1.0 for the first time on record (history 0.992 → 0.9972 → 0.9873 → 0.9807 → 1.0036). 2026-08-07 attributed its low scale partly to voltage sag over the drive phase (chords 149.5 → 143.9 mm across 10 reps); today ran **15** reps, i.e. more continuous driving, which should sag further and push the scale *down*, yet it rose 2.3%. Both values sit inside the historical range (curl has spanned −0.0124 to −0.0369), so nothing is implausible — but this is not a settled measurement.
- **Bearing on the 2026-08-07 unresolved discrepancy:** run02 implied an in-run curl of −0.0177 against −0.0124 measured in consecutive straight drives. The measured value is now −0.0261, which *exceeds* the run02-implied figure, so the sign of the gap has flipped. That weakens the "calibration protocol does not match deployment" hypothesis as a systematic effect and favours plain between-session variability in curl.
- **Fit for Acquisition06, not yet fit for a deploy.** Acquisition labels come from `executed_pose` (tracker at ping time), so drive error costs waypoint accuracy and nothing else — these constants are good enough to run the session. **Re-measure before the Exp-1 re-run and before any Exp-2 policy deploy**, where the drive model does feed the result; if curl comes back near −0.0124 then −0.0261 was the outlier.

### 2026-08-12 — Acquisition06 arena and plan: the far-range gap is closed by geometry, and a 25 mm dowel does echo at 2.85 m

Offline geometry plus one bench measurement; no session run yet. Commit at measurement `28b7692`. **`AcquisitionArenas/` is gitignored, so the arena and plan artifacts these numbers describe exist only in Dropbox** — hence the detail here.

- **Arena** `AcquisitionArenas/Acquisition06/env_0001_2026-08-12T12-44-36`: bare boundary 3543 × 4166 mm, one closed structure, **no interior blocks**, 3325 wall points (median spacing 4.5 mm, max gap 18.6 mm, no pose blind at every heading). Four poles at (−792, −1880), (1082, −2349), (1071, 529), (−1218, −376), standoffs **477–607 mm** from the nearest wall, separations 1563–3045 mm.
- **The old arenas were geometrically incapable of the missing data.** Uniform survey — 150 mm position grid at ≥250 mm clearance, 10° headings, nearest-in-cone at ±35°:

  | arena | far band 1.5–3.5 m | p90 in-cone | max in-cone |
  |---|---|---|---|
  | Acquisition01 | 2.0% | 1179 mm | 1904 mm |
  | Acquisition02 | 1.4% | 1195 | 1815 |
  | Acquisition03 | 3.3% | 1286 | 2086 |
  | Acquisition04 | 2.2% | 1184 | 2008 |
  | Acquisition05 | 3.5% | 1298 | 1994 |
  | **Acquisition06** | **29.8%** | **2145** | **3160** |

  Nothing in Acq01–05 is beyond ~2.1 m in-cone from any legal pose at any heading — **the 1911 mm ceiling in the training set is the rooms, not the waypoint choices.** With the 4 poles removed Acq06 would reach 36.1% / 3277 mm, so the poles cost 6.3 points of far coverage.
- **Interior boxes were considered and rejected.** Simulating Path02's five 230–315 mm blocks into this arena collapses the far band **34.4% → 1.9%** and caps sight lines at 2.5 m — the same signature as Acq01–05, i.e. interior clutter is exactly what capped them. Three peripheral blocks would cost 34.4% → 20.0%. Poles are far cheaper per unit of far-range ambiguity, and a 25 mm dowel plus a 3.5 m boundary wall *bracket* a 250 mm block, so far competence on deployment-arena blocks becomes interpolation rather than extrapolation.
- **A 25 mm dowel returns a usable echo at 2.85 m.** Measured on the robot facing a pole with a wall >2.5 m behind: ~3600 a.u. above a ~6900 baseline, roughly **12 dB below** the wall echo at 3.45 m and ~9× the local ripple (eyeballed off the plot). This is what makes Acq06's far-pole labels sound rather than phantom, and it was the main risk against the design.
  - **The echo detector misses it and that does not matter.** `locate_echo` thresholds L/R only, flat ~11300 beyond 0.75 m; the pole peak at 10500 falls just under, so `selection_mode='max'` takes the wall and reports 3.34 m. **The inverse trains on raw `sonar_data` envelopes** (`AcquisitionSessionLoader.py:324`, `:458`) and never touches `locate_echo` or `corrected_distance`, so the training path is unaffected. Consequence: the acquisition-time `corrected_distance`/`corrected_iid` sanity display will disagree with the geometric label on far-pole pings — a false alarm, not a labelling error.
  - **`SCRIPT_CheckPoleSignal.py` WILL mislead on this session.** It reads raw envelopes but reduces each ping by *global argmax*, which on a pole-with-wall-behind ping is the wall — so all six hand features describe the wrong echo. Both it and the detector assume one echo per ping, which held only while poles were met close up with nothing behind them. The CNN is the only consumer that sees both echoes and their spacing.
  - Envelope input is the full 200 samples (~3.7 m), no truncation, and `sonar_norm` is a single global scalar (`SCRIPT_TrainInverseModel.py:707`) with no per-index or range-dependent gain — so the far echo enters at roughly 1σ. Learnable, but far-range sensitivity is to be *verified* in the trained model, not assumed.
- **Plan chosen: `plans/plan_2026-08-12T13-28-12.json`** — far-biased, `n_far=3`, seed 1786555688, `MAX_ATTEMPTS_PER_STEP=1000`. Scored against the uniform plan built the same day:

  | | uniform 13-03-09 | far-biased 13-20-40 | **far-biased 13-28-12 (chosen)** |
  |---|---|---|---|
  | positions / pings | 132 / 660 | 116 / 580 | **128 / 640** |
  | far 1.5–3.5 m | 203 (30.8%) | 281 (48.4%) | **313 (48.9%)** |
  | beyond 1911 mm | 116 | 182 | **197** |
  | 2500 mm+ | 22 | 69 | **74** |
  | under 1000 mm | 296 | 212 | 236 |
  | p50 in-cone | 1101 mm | 1454 | 1441 |
  | tour | 45.4 m | 34.6 m | **34.1 m** |
  | interior within 400 mm of a waypoint | 90% | 82% | 88% |
  | waypoints ≤1 m of each pole | 25/24/20/27 | 17/27/20/16 | **24/24/23/21** |

  The chosen plan beats the uniform one on every axis that matters while running **20 fewer pings and 11 m less driving**. Far-band class splits 153 wall / 160 pole — marginally pole-heavy and truthful, so the new data will *not* teach "far and open ⇒ wall" as a prior; remember that when reading the retrained model's far-range behaviour. Still, 153 far-wall pings is more than triple the 42 the 1400–1700 band currently rests on.
- **Yaw-mode effect isolated** at a matched seed (identical 123 positions, identical 41.9 m route, identical rejection counts): far band **30.4% → 47.2%**, beyond 1911 mm **115 → 199**. So the mode does the work; position count only scales the totals.
- **`reorder_tour` truncation cost real coverage on the first far-biased build** (13-20-40): 131 built → 116 kept, the 15 dropped all in x [−1366, −37], y [−1826, 81], which is where pole 4 sits. Poles 1 and 4 lost about a third of their close-range waypoints. Raising `MAX_ATTEMPTS_PER_STEP` 200 → 1000 reduced the drop to 3 and restored balance. At 4000 the drop is 1–2 with ~167 positions, but the session grows to ~835 pings / 47 m for diminishing returns.

### 2026-08-09 — Per-distance error of the deployed heads: wall slices are better close in than the aggregate suggests, and the pole-range head saturates at ~740 mm

Read off the deployed uncapped model (`67c68c0`) via the out-of-fold predictions in `TempOutput/RangeHorizon/oof_fullrange_maskedrange.npz`. No retraining; reproducible from that file.

- **Wall slices, by TRUE slice distance** (the deploy report's 292/392/321 mm is an aggregate over all wall pings and understates close range):

  | true distance | n | bias | RMSE | predicted σ |
  |---|---|---|---|---|
  | 200–500 mm | 899 | +58 | **166 mm** | 124 |
  | 500–750 | 897 | +51 | 196 | 145 |
  | 750–1000 | 758 | +87 | 232 | 172 |
  | 1000–1500 | 933 | +64 | 263 | 190 |
  | 1500–2000 | 397 | −43 | 333 | 356 |
  | 2000+ | 424 | **−524** | **778 mm** | 570 |

  Two things to carry forward: there is a consistent **+50 to +87 mm over-estimate below 1.5 m**, and unlike the pole-range head the wall head's **σ is roughly honest** (tracks RMSE until 2 m, understating only in the 2000+ band where the head saturates).
- **The pole-range head saturates at ≈740 mm.** At a true 1238 mm mean it reports 742 mm (bias −496). So an approach to a pole at 1.5 m sees the reported range *stick* near 740 mm for the first several 150 mm steps, then begin tracking properly as the true range falls inside the trained span. It lags, it does not jump.
- **Experiment 1 is unaffected by that.** `at_stop_distance()` is a plain threshold (`pole_dist_mm <= APPROACH_STOP_MM`, 500 mm) with no range-rate or consistency check, so a frozen reading simply means "keep approaching", which is correct. And the **asymptote (~740 mm) sits comfortably above the 500 mm stop, so a genuinely distant pole can never trigger a premature stop** — the failure worth worrying about does not arise.
- **Experiment 2 IS affected, and this is the real cost.** The policy's pole-distance observation channel cannot separate a pole at 1.1 m from one at 1.8 m — both read ≈740 mm. The landmark channel therefore carries bearing but essentially **no distance information beyond a metre**, which is precisely the signal the path-integration-plus-landmark story would lean on. Training stays self-consistent (the simulator's error model is fitted from this same model, so it reproduces the saturation and the policy learns the compressed mapping), but the information is genuinely absent. **This is the strongest argument for the planned class-agnostic range head** — see Code state 2026-08-09.

### 2026-08-08 (evening) — The clearance/perception trade-off is much weaker than believed; the gaze proposal is measured and parked

- Commit `a4dc548`, `EXPT_gaze_path.py`. All offline geometry, no robot. Two questions: what a per-pose *looking direction* decoupled from the driving direction would buy, and whether what the robot perceives tells it **where it is**.
- **Path01, the original problem, before vs after the horizon extension** (recall-weighted, i.e. the model actually classifying correctly, using each era's own measured per-band recall; the old model's figure already absorbs its blind ring since abstentions count as misses):

  | | share of steps | of 72 steps/lap | blind steps/lap |
  |---|---|---|---|
  | before (1 m cap, old model) | 48.3% | ~35 | **~38** |
  | after (uncapped, new model) | **78.6%** | ~57 | **~16** |

  Geometric equivalents 56.6% → 87.1%. **This is the problem Path02 was drawn to solve, solved without touching the controller.**
- **The clearance-vs-informativeness trade-off is largely an artifact of those two particular routes.** Pooling all 960 path poses and binning by clearance, informative fraction facing along the path:

  | clearance | n | @1000 mm | @1400 mm | @1700 mm |
  |---|---|---|---|---|
  | 100–200 mm | 51 | 82.4% | 100% | 100% |
  | 200–300 | 167 | 85.0% | 92.2% | 94.0% |
  | 300–400 | 191 | 66.5% | **78.5%** | 96.3% |
  | 400–600 | 318 | 68.9% | 95.0% | 99.1% |
  | 600+ | 233 | 68.2% | 100% | 100% |

  At 1400 mm it is **78–100% at every clearance band, and not monotonic in clearance at all** — because clearance is the distance to the nearest obstacle in *any* direction, while perception depends on the ±35° cone ahead. A pose far from everything can still be looking down a corridor at a wall a metre away. **So "closer to obstacles = better sonar = less room for error" (the 2026-08-07 framing, still quoted in Where to pick up) does not hold once the horizon reaches 1400 mm.** A ~350 mm-clearance path can be routed for safety and perception at once.
- **Perception alone cannot localise on either path** — 70–97% of poses are indistinguishable from a pose metres away (confusability = within one σ of the deployed model's own error; 1/3/5-step windows; gaze and sequence length barely move it). One pole per arena, and wall depth is only good to ~300 mm, so the profile is a weak place code.
- **That is the wrong bar, and the right one resolves the run02 paradox.** The RNN integrates, so it only needs *local* refinement. Residual position uncertainty given you already know where you are to within 1.5 m (median, 5-step window): **Path02 @1400 = 251 mm**, alongside the 116 mm lap-to-lap overlay actually measured. So run02 is explained: dead reckoning plus a weak quarter-metre perceptual correction, enough to stop the −1.08°/step bias accumulating, nowhere near enough to say where it is on the loop.
- **Gaze: measured, and parked.** It does **not** address the problem it was proposed for, because the horizon extension already did (detection is near-saturated either way). Its remaining value is localisation on *open* paths — Path01 803 → **355 mm** local uncertainty — while being neutral-to-harmful on wall-hugging ones (Path02 251 → 489 mm). Costs a second action output, an extra rotation per step (~4° rep-to-rep noise at small commanded angles), and a gaze command that depends on the position estimate it is meant to improve.
  - ⚠️ **These numbers flatter gaze**: it is aimed here from the robot's TRUE pose. A real robot aims from its believed pose, so the benefit degrades exactly where the estimate is poor. Facing along the path needs no self-knowledge at all.
  - **Decision rule for the redrawn path:** run `EXPT_gaze_path.py` on it. Local uncertainty near Path02's 251 mm → gaze is settled as unnecessary. Near Path01's 803 mm → reopen it *before* training, not after.
  - What none of this measures is the yardstick that actually matters — how easily the path is **learned** and how well disturbances are **corrected**. Those are closed-loop properties of policy plus dynamics plus noise. Answering them means training under each condition in sim, which tests learnability and stability honestly but says as much about the simulator as the robot on the perception side.

### 2026-08-08 (later) — Uncapped inverse adopted and deployed; the phantom-pole diagnosis of 2026-06-12 was wrong

- Commit `67c68c0`. Follows directly from the entry below, which is the evidence for the adoption. `FAR_LABEL_MODE="true_class"` is now the script default, plus `POLE_DIST_TRAIN_MAX_MM=1000` masking the pole-range head. Predecessor archived at `SonarModel_archive/2026-08-08_deploy1m/`.
- **The range-head mask works and beats the model we were flying.** Signed bias at 0–500 mm: **+66 mm** (masked) vs +86 (cap1m) vs +149 (unmasked fullrange); RMSE 128 vs 134 vs 233 mm. The nominal 500 mm terminal stop therefore fires at ≈434 mm true, against 414 (cap1m) and ~350 (unmasked). Below 1 m the classifier is untouched by the mask (91.8% vs 91.9% acc); pole recall given commitment 82.6 vs 84.7%, inside the 9–11 point per-fold noise. Close-range wall recall **95.7%, best of all conditions**. Close-range pole-az degrades mildly, 12.95 → 13.71° at 0–500.
- **The masked head saturates beyond its training span and its σ does not flag it** (1000–1700 mm: bias −496 mm, z_std **2.56**). Treat a reported pole range near 1000 mm as "at least 1 m", never as a measurement; `trained_max_mm` is recorded in `feature_params.json`.
- **Deployed model** (`main_deploy()`, spatial 15% holdout seed 0, best epoch 43, n_train 1813 / held-out 322): held-out class acc **86.6%**, wall prec/rec 91.4/86.2, pole prec/rec **80.3/87.3**, pole-az RMSE 15.01°, wall RMSE R/C/L 292/392/321 mm. In-sample 89.4%, pole-az 12.67°. **Do not compare 86.6% against the old 83.9%** — this is now effectively a **2-way** problem (`none` has zero members), the old was 3-way. The controlled comparison is the CV head-to-head in the entry below; that is the one to quote. Wall recall 86.2% is over *all* ranges and is dragged down by the far band (~64% at 1000–1700); at close range it is 95.7%.
- **Error model refitted** (`SCRIPT_FitInverseErrorModel.py`, required — the simulator draws its perception from this file and would otherwise model an inverse that no longer exists). **Phantom generation has moved**: from 15.7% of true-`none` pings to **10.9% of true-`wall`** pings, i.e. roughly 4% → 6% of all pings. Fitted pole-range bias +60 mm at 200–500 (matches the CV's +66) and −111 mm at 900–1000 (saturation beginning). Wall slices now show the same saturation beyond 2 m (bias −352 to −629 mm), which the simulator will now reproduce.
- **The phantom-pole regression gate no longer passes, and replaying it corrected a two-month-old misdiagnosis.** Of the 10 historical `direct_pole_demo1` phantoms, **6 (steps 0–5) are called `pole` again** at 0.56–0.91 (2-class era: 0.97–0.99; 3-class cap1m: `none`). Steps 6–9 now read `wall`, and step 9's old boundary miss is fixed.
  - **The 2026-06-12 entry's explanation — "binaural front/back ambiguity placing the rear echo at az≈0" — is not supported by the data.** Reconstructed true geometry at those steps: the pole sits **behind** the robot at az ±163…180°, moving 186 → 945 mm as it drives away, while the model's reported pole range stays **flat at 712–744 mm**. Nothing is at ~720 mm in *any* direction (nearest wall at any bearing 1066–1374 mm), and the pole-range head's training mean is **642 mm**. So the model is emitting its prior, not detecting the pole. Dieter's physical objection is what surfaced this: a rear echo would need a multi-bounce path, and there is no such echo.
  - **The real cause is an in-cone reflector beyond the model's competence.** At those steps the nearest in-cone wall is **2282 → 1539 mm**. Given an envelope it cannot match to any wall it knows, the model falls back to `pole` at a default range. Uncapping **reduced** this failure rather than causing it: the boundary moved from ~1.0 m to ~1.4 m (steps 6–7, walls at 1382/1199 mm, are now correctly `wall`; step 5 at 1539 mm is marginal at 0.56).
  - **Consistency filtering does not catch it.** Reported azimuths across the six steps are −0.8, −1.1, +6.5, +1.8, −2.5, −0.3 — all inside `ALIGN_TOL_DEG=15` for six consecutive steps, so `ALIGN_MIN_DETECTIONS=3` is satisfied. The stop-requires-n-detections rule defends against *independent* false positives; this phantom is **systematic**, persisting exactly as long as the robot faces open distance.
- **Consequence: the residual failure zone is "nearest in-cone reflector beyond ~1.5 m", and the training set has no such data** (max in-cone range 1911 mm, and no ping pairs a distant in-cone target with anything else). Exposure is 4.4% of Path01 poses, 0% of Path02, and higher for Exp-1-style wandering. **This is what Acquisition06 is for** — see Where to pick up.

### 2026-08-08 — The inverse's 1 m cap: what it costs, and why it cannot simply be moved outward

- Commits `7ffa520`, `8f4bee2`. Offline, no robot. `EXPT_range_horizon.py`: 4-fold quadrant CV per condition, identical data / folds / seed (Acq01A–05A, 2135 pings, 60 epochs), only the far-ping labelling and the cut differing. Artifacts in `Control_code/TempOutput/RangeHorizon/` (gitignored); the `oof_*.npz` carry every out-of-fold prediction, so anything here can be re-scored **without retraining** (`main(only=[...])` re-analyses from the npz). The deployed 1 m model was archived first to `SonarModel_archive/2026-08-08_deploy1m/`, including `inverse_feature_params.json`.
- **The cap is a labelling choice, not a sensor limit.** Confirmed **0 empty-cone pings** in these arenas, so `none` has always meant "beyond 1 m", never "nothing there". Data tops out at **1911 mm**; band populations 584 / 1028 / 512 / 11 for 0–500 / 500–1000 / 1000–1700 / beyond. "Uncapped" therefore means "out to ~1.7 m" on this dataset, not to arena scale.
- **Headline, on pings closer than 1000 mm (identical in every condition):**

  | condition | acc | bal | abstain | acc given committed | pole recall given committed |
  |---|---|---|---|---|---|
  | cap1m (canonical) | 84.9% | 79.8% | **9.3%** | **93.6%** | **88.2%** |
  | fullrange | 91.9% | 89.8% | 0.0% | 91.9% | 84.7% |
  | cap1700 | 92.1% | 89.9% | 0.0% | 92.1% | 84.5% |

- **Raw accuracy misleads; the decomposition is the finding.** Uncapping does **not** improve discrimination — conditional on committing to an answer, cap1m is slightly *better* (93.6 vs 91.9). The entire +7 pp is that cap1m **abstains on 150 of 1612 sub-1 m pings**, every one a real reflector, **112 of them poles**, median true range **874 mm** (90th 981, 10th 336). A hard cut in *label* space cannot produce a sharp boundary in *envelope* space, so the model hedges just inside its own cut; it bites poles hardest because a pole returns less energy than a wall at the same range and so resembles the far `none` training set. **This is the mechanism behind Exp 1's 25% pole recall in the 800–1000 mm band.**
- **Far echoes are classifiable and honestly calibrated to ~1.7 m.** fullrange balanced accuracy by band 91.2 / 88.9 / **76.0** / 50.0%; overconfidence gap +1.0 / −1.6 / **+4.4** / +47.6 pp; ECE .025 / .028 / **.044** / .476. **Pole detection does not degrade with range at all** (recall 84.1 / 84.9 / 87.6%); wall recall does (64% at 1000–1700), i.e. at range the model leans toward `pole`. This is not new capability — the 2026-05-21 stratification already implied ~68% balanced accuracy in that band; the label was suppressing it.
- **Beyond ~1.9 m it fails exactly as the 2-class model did, and the guard cannot be preserved by moving the cut.** All 11 pings past 1700 are called `pole`, including all 6 true walls, at p_pole 0.91–0.99. With the cut *at* 1700 the abstain class has 11 examples and is simply ignored (`none` recall **0%**, mean confidence 91.5%) — so **cap1700 is fullrange, not "fullrange with a guard"**. The guard at 1000 worked only because `none` was 523 pings (24.5%). Counts beyond candidate cuts: 1000→523, 1200→265, 1300→162, 1400→93, 1500→50, 1700→11. A cut near 1200 is the last one with a plausibly learnable abstain class; untested.
- **The pole-range head regresses toward the mean, and widening its span makes that worse.** Signed bias at 0–500 mm: cap1m **+86 mm**, fullrange **+149 mm**; fullrange at 1000–1700 is **−219 mm**. RMSE at 0–500: 134 mm (cap1m) vs 233 mm (fullrange). **Consequence:** the Exp-1 terminal stop fires on *predicted* range < 500 mm, so under cap1m it really fires near 414 mm true — consistent with the recorded 276–372 mm arrivals — and under fullrange near 350 mm, pushing toward the sub-253 mm regime with no training data. **Fix before deploying: mask the range head to poles within 1 m (range AND class) while leaving the classifier full-range.** Not yet done or verified.
- **Pole-azimuth σ is overconfident in every condition** (z_std 1.2–1.55 where 1.0 is honest), i.e. the head understates its own error by 20–50%. Pre-existing and unrelated to this change, but the policy consumes σ and `InverseErrorModel` is fitted from these outputs.
- **Path informativeness at a longer horizon** (`SCRIPT_AnalysePathRun.py`'s definition: true geometry, facing along the path, `cls != none`) — this is the payoff, and it changes the Experiment 2 plan:

  | horizon | Path01 informative | Path01 nothing-in-range | Path02 informative | Path02 nothing-in-range |
  |---|---|---|---|---|
  | 1000 mm | 56.6% | 43.4% | 85.5% | 14.5% |
  | 1200 mm | 74.3% | 25.7% | 95.8% | 4.2% |
  | 1400 mm | 87.1% | 12.9% | 97.8% | 2.2% |
  | 1700 mm | 95.6% | 4.4% | 100.0% | 0.0% |

  **Path01 at a 1400 mm horizon (87.1%) beats Path02 at the current 1 m horizon (85.5%)** — informative perception no longer requires hugging obstacles, which is exactly what crashed run02. The same table sizes the un-guarded exposure: poses with nothing within 1.7 m are **4.4% of Path01 and 0% of Path02**. **Caveat: this does not make Path01 usable** — its 249 mm clearance leaves 164 mm usable against the 266 mm p90 tracking error. It makes a *new* ~350 mm-clearance path affordable, because clearance no longer has to be traded against perception.

### 2026-08-07 (evening) — Experiment 2 runs on the robot: 2.1 laps of Path02, then a collision; robot recalibrated

- Commits at measurement: `19d0f47` (deploy chain) plus the Settings.py recalibration. Run: `PolicyRuns/default_Path02_run02`, policy `default_Path02`, `POLICY_INPUT_SOURCE="live"`.
- **Result: the policy follows the path, but Path02 has too little clearance to survive.** 110 steps = **2.1 laps**, threading between the six interior blocks, then **collisions logged at steps 116/117** (`crashes.tsv`) near (-227, -1120).
- **The collision is geometric, not a control failure.** Path02's minimum clearance is 155 mm against an 85 mm robot radius = **70 mm usable margin**, while measured tracking error is mean 119 / median 94 / **90th 266** / max 341 mm. **49% of the path has less margin than the 90th-percentile tracking error** — collisions are structural. At the crash the robot had drifted 421 mm off-path into a region with 595 mm clearance, so the tightest sections remain untested. **This is the price of the Path01 -> Path02 switch**: more informative perception was bought by hugging obstacles, and Path01's 249 mm min clearance (164 mm usable) is what was given up. (The 26%/49% figures in the earlier entry below are **not reproducible** — see the metric-mismatch warning in Where to pick up; use `SCRIPT_AnalysePathRun.py` for any new comparison.) Any replacement path needs a min clearance near **350 mm** (p90 error + robot radius) and should have its informativeness re-measured. Deviation of every post-lap-1 step from the full first-lap track: **mean 116 mm, max 291 mm** (lap-2 second half tightest at 88 mm mean). Step size is 150 mm, arena 3538 x 4188 mm.
- **This is the key inference:** the residual heading bias is **-1.08 +/- 0.20 deg/step** (n=106, 3 outliers excluded). If that accumulated freely, one 53-step lap would shed **~57 deg** of heading and lap 2 would not overlay lap 1. It does overlay, so **the policy corrects from perception rather than dead-reckoning** — the RNN integrates, but sensory input pulls it back each step.
- **Recalibration, both phases** (constants had not been touched since 2026-06-08; batteries were changed before run01):
  - *Phase 1, rotation, in-place, no space needed.* With fresh batteries the robot **over-rotates at small angles**: +-5 deg gave +6.98/-6.69 (was +-5.13), +-10 gave +-11.19 (was +10.65). Symmetric in both directions, so a real effect rather than noise. New table written; old one recoverable from `037c85e`. Note the tracker failed to settle several times and -30 deg had a wild rep (-5.85, std 9.75) — the medians absorbed it, but the small-angle entries have ~4 deg rep-to-rep spread and are the least certain.
  - *Phase 2, forward drive.* Run at **`DRIVE_MM=150` (matches the deploy step, so curl is measured at the operating point) x `DRIVE_REPEATS=10`**, needing only **~1.7 x 0.9 m** — note the path **curves** with compensation off, so it is not a straight corridor; the previous 16 x 200 mm setting needed 3.2 m. Result: **curl -0.01244 deg/mm** (was -0.01993), **scale 0.9807** (was 0.9873). Chords trended downward across the 10 reps (149.5 -> 143.9 mm), which looks like voltage sag over ~50 s of continuous driving.
- **An unresolved discrepancy, recorded because it will resurface.** Reconstructing run01 from the constants active then plus the behaviour both phases measured predicts **+1.25 deg/step**; run01 observed **-2.66**. Gap **-3.91 deg/step**, unexplained. Same robot and same batteries, so this is not a hardware change. run02 then came in at -1.08 rather than the predicted -0.75, implying an in-run curl of **-0.0177 deg/mm against the -0.0124 measured in consecutive straight drives**. **Leading hypothesis: the calibration protocol does not match deployment** — Phase 2 drives ten times *consecutively*, while every deployed step is *rotate, then drive*, and the drivetrain may carry state out of an in-place rotation. Against it: Experiment 1 also rotates-then-drives and its residual (+0.49) matched the calibrated curl fine. Not resolved; the magnitude no longer matters for task success, but it means calibration numbers should be treated as approximate for deployment.
- **Method note.** Yaw residual = commanded `rot_deg` vs tracker `diff(yaw_deg)`, restricted to full-drive steps, using a 3-MAD outlier filter. Adjacent equal-and-opposite residuals indicate a single bad pose read, not two bad steps — check for that before believing a large residual.

### 2026-08-07 — Path02 policy trained; robot calibration measured stale from 10 steps

- Commits at measurement: `834a340` (training), `19d0f47` (deploy chain). Policy in `PolicyTraining/default_Path02/`; the Path01 run is preserved as `PolicyTraining/default_Path01_figure8/`.
- **Path choice bound the result more than the sensor did.** Fraction of steps along each path where the sensor returns something usable:

  | path | wps | length | min clr | % wall | % none | % informative | pole in-cone wps | true pole detections/lap |
  |---|---|---|---|---|---|---|---|---|
  | Path01 (figure-8) | 38 | 10874 mm | 249 mm | 37.1% | 44.0% | 25.9% | 5 | 3.5 |
  | Path02 (serpentine) | 42 | 8433 mm | 155 mm | 73.5% | 16.3% | **49.0%** | 3 | 2.2 |

  The ambient ceiling for this arena, sampling all headings, is **82%** informative — so moving the path, not changing the sensor, was the available lever, and it nearly doubled useful perception.
- **Training** (`TARGET_ARENA="Path02"`, `use_poles=True`, `use_sigma=False`, `fixed_drive_mm=150`, `max_dist_mm=1000`, `teacher_lookahead_mm=240`, start yaw noise 20 deg, 2000 epochs): **best val 114.65 at epoch 1402**, against **165.1** for the Path01 figure-8. **6 of 6** rollouts follow the serpentine (Path01: 3 of 6). No collisions despite the 155 mm minimum clearance, and no ambiguous-crossing failure — a serpentine has no self-intersection, so position never maps to two correct headings. **Caveat:** much of the 165 -> 115 drop is that the regression problem got *easier* (the sensor informs twice as often), not that the policy got better; the trajectories are the stronger evidence. The last ~600 epochs diverged (final ~126-130, train ~104); `best_policy.json` is written only on val improvement, so the saved artifact is epoch 1402. Consider cutting `n_epochs`.
- **Yaw-residual step change, 2026-07-30 -> 2026-08-07.** Method: commanded `rot_deg` vs tracker `diff(yaw_deg)`, restricted to full-drive (>100 mm) steps, since `Client.step()` pre-compensates the drive curl (`angle += -distance_mm * drive_curl_dpm`) so the expected net yaw change equals the command.

  | | n | mean residual |
  |---|---|---|
  | Exp 1, 2026-07-30, pooled | 860 steps | +0.49 deg |
  | Exp 1, per-run means | 15 runs | **+0.52 +/- 0.49** (range -0.58 .. +1.26) |
  | Path02 run01, 2026-08-07 | 9 steps | **-2.66** (sd 1.26) |

  Path02 is **z = -6.5** against the Exp-1 run-to-run distribution; **0 of 15** Exp-1 runs are that extreme. Regressing Exp-1 residual on distance driven gives slope +0.00338 deg/mm -> implied actual curl -0.0166 vs calibrated -0.0199 (0.83x, i.e. fine); Path02 implies -0.0376 (1.89x). Zero-drive steps in Exp 1 show +0.09 deg, isolating the effect to the drive curl and clearing the rotation calibration table itself.
  **Artifacts ruled out:** log pairing (both alignments tested — the correct one has sd 1.26 vs 8.88), position within run (Exp-1 early vs late differ by 0.31 deg), rotation chunking (`rotate_in_substeps` cap is 34.5 deg; every Path02 command was inside it, so it would have issued one call just like `RunPolicy`), gain error (residual uncorrelated with commanded angle, r ~ -0.05 both datasets), run-to-run variance (above).
  **Interpretation:** a step change, not gradual drift — stable across two months and 20 runs, then a jump within eight days. That fits a discrete physical event rather than wear, so recalibration should be read as a measurement before it is accepted as a fix. **Caveat:** the Path02 side is 9 consecutive steps from a single run, so they are not independent samples; the comparison against 15 Exp-1 run-means is what carries the weight, not the within-run t.
  **Consequence:** Experiment 1 does not need rerunning (see Where to pick up). Exp 1 is closed-loop and absorbs a constant heading bias; the Exp 2 RNN integrates it into unbounded error, which is why calibration gates Exp 2 and not Exp 1.

### 2026-08-04 — Experiment 1 (obstacle avoidance + target approach): 20 robot runs, 20/20 success

- Runs performed **2026-07-30**; written up 2026-08-04 (paper commits `3dee986`, `201b101`, `45c71f3`). Deployed inverse, `INVERSE_FOLD="deploy"`. Every number below is reproducible via `Control_code/.venv/bin/python3 Paper/images/scripts/exp1_stats.py`.
- **Design as run: full factorial**, 5 tape-marked starts × 2 placements (P1, P2) × 2 modalities = **20 runs**. This supersedes the 9-trial / 18-run plan in the 2026-07-28 entry.
- **Protocol as actually run**, read from each run's frozen `code_*.zip` and `run_summary.json`: `APPROACH_STOP_MM=400`, `ALIGN_MAX_STEPS=10`, `ALIGN_MIN_DETECTIONS=3`, `ALIGN_TOL_DEG=15`. **HEAD of `SCRIPT_RunDirectPolicy.py` now carries 500 and 6**, so re-running from HEAD would not reproduce the protocol the paper describes.
- **Outcome: 20/20 `reached_aligned`.** No collisions, corner jams or timeouts in either modality. **Zero bearing corrections in any run** — the perceived azimuth was already inside the 15° tolerance when the stop fired. Smallest true wall clearance 212 mm (sonar) / 216 mm (vision) against a 20 mm collision criterion. True pole distance on arrival 276–372 mm (sonar), 322–390 mm (vision). Tracker-verified final heading error: median 5.1° sonar (max 11.2°), 2.2° vision (max 7.4°).
- **Cost of sonar.** Sonar median 74 steps (12–154) and 10.16 m (0.84–22.09); vision 34 steps (13–91) and 2.64 m (0.76–11.60); sonar slower in 8 of 10 pairs. Search vs approach split: sonar 53 then 10 steps, vision 4 then 25. True pole range at first perception: 798 mm sonar, 2038 mm vision. The gap is the sensory horizon, not feature quality; the approach phases are not an efficiency comparison, since vision's begins ~2 m out.
- **Per-step perception (sonar, n=781 steps, ground truth with the 1 m horizon applied).** 95.0% overall; recall 93.1% wall, 86.4% pole, 97.5% none; pole precision 97.9% (2 false poles in 97 calls). Feature empty on 522/781 steps, pole reported on 97. Vision: never empty in 410 steps, pole on 234. Pole recall by true range: 94.7% (<400), 100% (400–600), 96.4% (600–800), **25% (800–1000, n=16)**.
- **Why that beats the held-out benchmark (80.1% acc / 63.3% pole recall / 75.0% pole precision) — tested, and it is NOT the closed loop "sampling where the model is good".** That framing was drafted, then refuted, twice. Within matched range × bearing cells the model is not better during runs by any margin the data supports (<500 mm & |az|<15°: 88.7% acquisition vs 98.0% deployment; ≥500 & <15°: 83.6 vs 87.1; ≥500 & ≥15°: 62.7 vs 67.9), and deployment is **worse** in the 800–1000 mm band (25% vs 55.5%). Sub-bin position was ruled out (median range and |az| within each cell match to within 0.5° and 20 mm), as was pole isolation (deployment poles 675 and 669 mm from the nearest wall, acquisition median 716 mm). **The actual cause is scene complexity:** acquisition arenas held 4–5 poles each, so a second pole stood in the cone behind the nearest in **44% of acquisition pole echoes and 0% of deployment poses**; and acquisition includes 40 echoes where a wall was within 300 mm of the pole distance, where recall collapses to 35%, a geometry an approach never produces. The paper says this plainly. **Do not write it up as behavior improving the inverse.**
- **Data integrity, two items.** (1) `direct_P2_S4_sonar_repeat01` was **missing from `PolicyRuns/`** and was recovered from `PolicyRuns/backupP2.zip`; it is now restored. (2) The reason it went missing: `direct_P2_S5_sonar_repeat01/run_summary.json` records `session` and `start` of **S4** and carries S4's controller seed (`crc32("P2|4")`), because `START` was left at 4 while the robot stood on mark 5 (its step-0 pose is the S5 mark). Both runs are intact and distinct (108 vs 79 steps, no stale dills). Consequences: **9 of 10 sonar/vision pairs share a controller seed, P2/S5 does not**, and anything keying off `run_summary.json` rather than the folder name will mislabel that run. `exp1_stats.py` keys off folder names throughout and documents this.

### 2026-07-29 — Pole-range head added to the inverse (no classifier cost); terminal approach moved onto perception

- Commits `90e24df` (model + trainer), `b96750b` (deploy side). Pre-range model preserved at `Control_code/SonarModel.bak_prerange/` — **including `inverse_feature_params.json`, which is shared across folds; the `.pth` alone will not roll back.**
- **Why.** The direct-policy stop fired on tracker geometry in *both* conditions, so sonar runs were terminated by vision. A sonar run that never perceived the pole could blunder within the stop radius and be scored a success the inverse played no part in — an artifact that inflates sonar in the flattering direction and cannot occur in the vision condition. Fixing it needs a pole range from the inverse, which previously emitted azimuth only.
- **Why 500 mm, not 88 mm.** The old threshold was unreachable twice over: it sits inside the emission/echo overlap where there is no echo, and there is **no training data below 253 mm** — the acquisition planner held waypoints `CLEARANCE_MM=250` off every reflector, so the one regime the stop lives in was never sampled. 27.7% of pole pings fall below 500 mm. Matches the 50 cm criterion of the earlier JEB study with these sensors.
- **Pole-range performance: 106 ± 11 mm RMSE (4-fold CV), 99 mm on the deploy split, against a 213 mm constant-mean baseline.** Best-performing regression head in the model. Range is a monaural time-of-flight cue, so it is the easy head — unlike azimuth, which needs a binaural comparison.
- **No classifier cost.** 4-fold quadrant CV, both arms under the same code: class acc **83.9 ± 4.0** (head on) vs **83.7 ± 2.9** (off) = 0.05σ; pole recall **71.2 ± 11.0** vs **67.4 ± 9.2** = 0.38σ; pole precision 70.3 ± 5.6 vs 72.7 ± 7.9 = 0.36σ; pole-az **13.3 ± 1.1** vs 14.2 ± 1.2 = 0.77σ. All under 0.8σ, the two largest favouring the head. Head-off reproduces the 2026-06-22 B baseline (83.1%/64.2%), confirming today's trainer edits are inert when the head is off.
- **Methodological note worth keeping.** A single spatial-holdout split suggested a **−3.8 point** accuracy loss and −5.6 pole recall. That was noise: per-fold σ on pole recall is 9–11 points, and the split has only 90 held-out pole pings, where one ping is 1.1 points. A loss-weight sweep (0.1/0.3/0.5/1.0) chasing the apparent effect found pole recall swinging 80→60→76→63% with **no monotonic trend** — i.e. it was measuring seed noise. **Do not read pole recall off a single split.** CV over all 2135 pings settled it in ~40 min.
- **Deployment gate: `direct_pole_demo1` phantom replay passes 10/10** (p_none 0.68–0.92 where the 2-class model fired `pole` at 0.97–0.99); real in-range walls (steps 10–17) still `wall`. New observation not in earlier gate reports: **step 9, a wall at 884 mm, is called `none`** — a boundary miss near the 1000 mm training cut, benign behaviourally (the robot wanders rather than wall-avoids at 88 cm). Unknown whether earlier models got step 9 right; only the phantom count was recorded.
- **Sim baseline for the bearing regulation: 20 runs, zero corrections needed**, final bearings already within 11.2° on arrival. That is the noiseless reference against which the sonar correction count is read.

### 2026-07-28 — Sim sweep: controller ceiling + Experiment 1 pole placements

- Commit `8cebc03`. Offline, no robot. `run_sim` from `SCRIPT_RunDirectPolicy` imported verbatim, so the same `ReactiveController`, referee and tuned constants (`DRIVE_MM` 150, `BOUNCE_TRIGGER_MM` 750, `WALL_JAM_MM` 250).
- Config: walls-only arena (3471 pts), 5 recorded starts, `HORIZON_MM` 1000 (matching the 3-class inverse's abstain range), 10 seeds per candidate×start, `MAX_STEPS` 200.
- **Controller ceiling — the headline.** 500 mm grid, 28 candidates, **1400 rollouts: every interior position reached from every start; zero collisions, zero jams.** All failures were timeouts. With perfect perception the reactive rule never crashes and never sticks, so collisions or jams on the robot are attributable to perception or drive error, never to the rule.
- Reachability therefore does **not** discriminate (0.84–1.00 across candidates). Median steps does: 11–67, a six-fold spread.
- 250 mm grid re-run: 104 candidates. Chosen **P1 (−1203, −84)** serving S1–S5 and **P2 (−203, −2084)** serving S2–S5 — 2236 mm apart, 9 trials, 62–99 sim steps each, ~701 steps per condition, 18 robot runs.
- **Caveat.** Perfect perception and no drive error. This is a geometric feasibility filter: positions that fail here cannot be rescued by better perception, but passing says nothing about sonar. The experiments are the test.

### 2026-06-23 — Cross-version comparison: spatial-holdout deploy vs earlier 3-class CV models (no regression)

Asked whether the deployed B spatial-holdout model (entry below) does worse than earlier inverse versions. Read from the on-disk results JSONs (`inverse_base_cv_results.json`, `inverse_cv_results.json`, `inverse_deploy_results.json`). All numbers are on unseen data — earlier rows are 4-fold quadrant CV out-of-fold (each model trained on 75%, every ping scored once out-of-fold, averaged); the deploy row is the single 15% spatial held-out set (trained on 85%), with in-sample shown for reference.

| model | eval | class acc | wall rec | pole rec | pole-az RMSE | wall RMSE L/C/R (mm) |
|---|---|---|---|---|---|---|
| base 3-class | CV out-of-fold | 84.0% | 93.9% | 58.2% | 13.2° | 313/318/281 |
| B 3-class | CV out-of-fold | 83.1% | 93.0% | 64.2% | 13.7° | 321/323/291 |
| **B spatial (deployed)** | **held-out 15%** | **83.9%** | **91.8%** | **68.9%** | **12.1°** | **330/397/248** |
| B spatial (deployed) | in-sample | 85.5% | 95.2% | 70.8% | 10.1° | 304/311/271 |

- **Verdict: no regression — equivalent, slightly better on pole metrics.** Classification is flat (held-out 83.9% ≈ 84.0% / 83.1% CV). Pole-az is the best of the lot (12.1° vs 13.2° / 13.7°) and pole recall the highest (68.9% vs 58–64%) — the metrics the pole-approach task leans on. Walls comparable: held-out left/right as good or better (right 248 best), and the held-out **center 397 mm** is single-split sampling noise (in-sample center 311, in line with history), already flagged in the paper.
- **Caveats.** The comparison mixes eval protocols (4-fold CV vs a single 15% spatial split), so the deploy held-out is noisier — hence the center wobble; in-sample/held-out bracket the CV figures. Pole recall is the high-variance metric across runs (e.g. the 2026-06-12 base entry recorded 69.7% vs the EXPT base JSON's 58.2% here), so read pole recall as a band, not a point. 2-class era (90.4%) is not comparable (2-way vs 3-way). The phantom-pole gate (10/10, entry below) independently confirms no behavioral regression.
- References: 2026-06-23 deploy entry (below), 2026-06-22 B head-variant sweep, 2026-06-12 base 3-class.

### 2026-06-23 — Deployment inverse: single spatial 15% holdout (supersedes 4-fold CV for the deployed model)

- Commit: `f6c969d` (`Control: add spatial-holdout deployment training for the inverse`) on `direct-learning-poletask`. `SCRIPT_TrainInverseModel.py` gains a deployment path (`main_deploy()`, now the script entry point; CV `main()`/`run_fold` untouched, EXPT still works); `AcquisitionSessionLoader.load_data_inverse` gains `return_poses`. `SCRIPT_RunDirectPolicy.py` `INVERSE_FOLD` "q0" → "deploy" (edited in the working tree, left **uncommitted** with the user's live tuning WIP).
- **Framing change (decided 2026-06-23, see [[project_validation_purpose]]).** We make NO generalization claim for the inverse; its real test is the behavioral experiments. Held-out data is now only an overfitting guard. So the deployed model is a single train/val split, not CV: per session, hold out a contiguous spatial region (the 15% of pings nearest a seeded random anchor; **seed=0**, chosen for balanced per-session pole coverage — ≥14 poles/session — before training), train on the other 85% with early stopping on the held-out region, and report in-sample vs held-out (the small gap is the overfitting check). The 2026-06-22 4-fold CV B model is superseded for *deployment*; that entry's CV numbers are no longer what the paper reports.
- Config: Acq01A–05A, `MAX_RANGE_MM=1000`, `CONE_HALF_DEG=35`, `SonarSlicesUQ_Wall3(symmetric=True)`, 15% spatial holdout (seed 0), 60-epoch budget, best epoch **45**. n_train 1813 / n_held-out 322 (held-out poles n=90).
- **Class accuracy: in-sample 85.5% • held-out 83.9%** (small gap → not overfit).
- **Wall: in-sample prec/rec 92.6/95.2 • held-out 90.6/91.8.**
- **Pole: in-sample prec/rec 73.9/70.8 • held-out 81.6/68.9.**
- **None: in-sample prec/rec 78.6/76.5 • held-out 75.3/85.9.**
- **Pole-az RMSE: in-sample 10.1° (n=387) • held-out 12.1° (n=90)**; MAE 7.3°/9.2°; σ_med ≈10°.
- **Wall RMSE L/C/R (mm): in-sample 304/311/271 • held-out 330/397/248.** Held-out center (397) rests on few held-out wall pings and is noisier than the even in-sample profile (single-split sampling, flagged in the paper).
- **Deployment validation:** `inverse_deploy_*` loads via `InverseModel.load(fold="deploy")`; the `direct_pole_demo1` phantom-pole gate passes **10/10** (all old confident phantom poles → `none`).
- Paper: Methods Par 22 + Results Par 23–27 + `tab:inverse-results` rewritten around this (in-sample vs held-out two columns); paper edits uncommitted (user reviewing).

### 2026-06-22 — Wall-head architecture sweep + adoption of the single symmetric head (B) as canonical inverse

- Commit: `ed66804` (`Control: adopt single symmetric wall head (B) as canonical inverse`) on `direct-learning-poletask`. New `EXPT_head_variants.py`; `Library/SonarModel.py` gains `SonarSlicesUQ_Wall3` + a `model_class`-dispatching `InverseModel.load`; `SCRIPT_TrainInverseModel.py` selects architecture via `MODEL_CLASS`/`MODEL_KWARGS`/`MODEL_CLASS_NAME` (recorded in `feature_params`).
- **Motivation (expository, not numerical).** The deployed base inverse (`SonarSlicesUQ_TwoHeaded`) used *two* wall mechanisms — center regressed from the averaged embedding `z_sym=(zL+zR)/2`, flanks from the full binaural concat with weight-tying — which is awkward to justify in Methods. The sweep tested whether a single uniform head does as well.
- **Sweep (`EXPT_head_variants.py`, identical data/folds/seed, only the wall head swapped; 5 sessions Acq01A–05A, `MAX_RANGE_MM=1000`, 3-class, 4-fold CV @ 60 ep).** Interrupted by a power outage mid-variant-A (base + B complete, A had only folds q1/q2); resumed A to finish.

  | variant | wall head | class acc | pole-az RMSE | wall L/C/R RMSE (mm) |
  |---|---|---|---|---|
  | base | side heads + z_sym center (old deployed) | 84.0% | 13.22° | 313/318/281 |
  | **B** | one 3-out head, symmetry enforced over LR/RL | 83.9% | 14.28° | 323/319/292 |
  | A | one 3-out head, LR only, asymmetric | 83.8% | 13.94° | 349/333/299 |

- **Verdict: adopt B.** All three are within per-fold noise on class accuracy and wall slices (σ 17–47 mm). The only gap that looks real is pole-az (base 13.22° vs B 14.28°, ~1.3σ), but the pole-az and class heads are *byte-identical* across variants — only the wall head differs — so that gap is shared-trunk training coupling, not an architecture effect. B wins on the argument that matters: one uniform symmetric head, same symmetry logic as the class/pole heads, far simpler to state. A (deliberately asymmetric) was worst on wall RMSE, confirming the L↔R mirror symmetry is a correct inductive bias, not a limitation.
- **Canonical `inverse_` retrained as B (deployed model, fold q0 is the deploy default).** Class acc **83.1% ± 2.4%** (per-fold 86/79/84/84); wall precision 92.4% ± 2.2% / recall 93.0% ± 3.1%; pole precision 71.9% ± 6.2% / recall 64.2% ± 4.7%; none precision 73.3% ± 2.3% / recall 77.5% ± 10.0%; **pole-az RMSE 13.70° ± 1.01°** (per-fold 12.3/14.8/13.1/14.6); wall RMSE right 291 ± 23, center 323 ± 47, left 321 ± 16 mm. (Small deltas vs the sweep's B row are run-to-run CPU nondeterminism, same family.)
- **Phantom-pole recheck (the deploy gate).** Replayed the saved `direct_pole_demo1` pings (`PolicyRuns/older/direct_pole_demo1/`, fed `sonar_data[:,1]`/`[:,2]` exactly as `SCRIPT_RunDirectPolicy.py`) through the new B fold-q0: **10/10** old phantom poles (base-2-class fired `pole` at p≈0.97–0.99) now classify `none` at p_none 0.67–0.94; real in-range walls (steps 10–17) still classify `wall`. B preserves the abstain fix — and beats the 2026-06-12 base-3-class result (8/9).
- Reference: 2026-06-12 (base 3-class, 84.4% acc / 69.7% pole recall / 13.17° pole-az). Old 2-class still at `Control_code/SonarModel.bak_2class/`. base architecture still loadable via the `model_class` dispatch.

### 2026-06-12 — Two-headed inverse retrained with a 3rd "none" class, Acq01A–05A, close-range, 60-epoch CV

- Commits: `Library/SonarModel.py` (`p_none` readout) and `SCRIPT_TrainInverseModel.py` (3-class) on `direct-learning-poletask`, this session. `SCRIPT_RunDirectPolicy.py` deploy-side (class 2 → `"empty"`, `GEOM_RANGE_HORIZON_MM` flag, `p_none` logging) validated but uncommitted.
- **Motivation.** In sonar deployment the 2-class inverse emitted confident phantom poles in open space (`direct_pole_demo1`: drove off after a pole that was 186 mm *behind* the robot). Diagnosed via the new per-step `trajectory.tsv` `p_pole` logging + saved ping dills: the model had no abstain option — training *dropped* empty-cone and beyond-1 m pings, so an out-of-distribution far/rear-pole ping collapsed to `pole` at p≈0.99 (compounded by binaural front/back ambiguity placing the rear echo at az≈0). Fix: relabel those dropped pings to a new `none` class instead of discarding.
  - ⚠️ **The causal half of this paragraph was disproved on 2026-08-08 — see that entry.** The rear-pole / front-back-ambiguity story does not survive checking the geometry: the model's reported range stays flat at its 642 mm prior while the pole moves 186 → 945 mm behind, and nothing sits at that range in any direction. The actual trigger is an **in-cone reflector beyond the model's competence** (a wall at 1.5–2.3 m). The *observation* (confident phantom poles) and the *fix that worked* (an abstain class) both stand; only the explanation was wrong. Entry left otherwise unedited per the append-only convention.
- **Data (relabel pass).** 2135 pings → wall 1135 (53%) / pole 477 (22%) / none 523 (25%). All 523 `none` are **beyond-1 m** pings (mostly 1.0–1.5 m); there are **zero** true empty-cone pings in these arenas, so `none` operationally means "nearest reflector > 1 m." Balanced and trainable.
- Config: Acq01A–05A, `MAX_RANGE_MM = 1000.0`, `CONE_HALF_DEG = 35°`, `SonarSlicesUQ_TwoHeaded` with `n_classes=3`, 4-fold quadrant CV at 60 epochs, CPU.
- **Class accuracy: 84.4% ± 2.2%** (per-fold 85 / 81 / 84 / 87). Not comparable to the 2-class 90.4% — harder 3-way split.
- **Wall: precision 93.7% ± 1.8% • recall 92.8% ± 1.3%** — unchanged vs 2-class; wall head unaffected.
- **Pole: precision 72.5% ± 6.1% • recall 69.7% ± 5.0%** — down from 2-class 82.8 / 78.1; expected pole↔none trade-off.
- **None: precision 75.8% ± 6.2% • recall 78.8% ± 10.0%** (new class).
- **Pole-az RMSE: 13.17° ± 0.65°** — slightly better than 2-class 15.18°, σ tightened.
- Wall slice RMSE: right 278 ± 15, center 322 ± 48, left 307 ± 20 mm — essentially unchanged.
- **Deployment validation (the decisive test).** Re-ran the exact `direct_pole_demo1` phantom pings (saved dills) through the new fold-`q0` model: **8 of 9 prior phantom poles now classify `none` at p_none 0.77–0.95** (were `pole` at 0.97–0.99); the 9th is a borderline pole (p=0.56); the real in-range walls (steps 10–17) still classify `wall`. So the retrain fixes the actual failure — the controller will scan instead of chasing rear-pole ghosts.
- Old 2-class model backed up to `Control_code/SonarModel.bak_2class/` (`SonarModel/` is gitignored — only copy).
- Reference: 2026-06-04 2-class entry (90.4% acc / 78.1% pole recall / 15.18° pole-az).

### 2026-06-04 — Two-headed inverse, Acq01A + Acq02A + Acq03A + Acq04A, close-range, 60-epoch CV

- Commit: working-tree state on `direct-learning-poletask` (only `ACQUISITION_SESSIONS` list change vs the 2026-05-29 entry — Acq04A added; config otherwise unchanged).
- Sessions: Acq01A (505 pings), Acq02A (315), Acq03A (~300), Acq04A (~300+). 4-fold CV n_train per fold ~928–989, n_val ~289–350 (was ~700 / 150–170 at 3 sessions).
- Config: `MAX_RANGE_MM = 1000.0`, `CONE_HALF_DEG = 35°`, `SonarSlicesUQ_TwoHeaded`, 4-fold quadrant CV at 60 epochs.
- **Class accuracy: 90.4% ± 1.6%** (per-fold 91.5 / 88.9 / 92.5 / 88.9). Crossed 90% mean; σ down from 2.7 → 1.6 vs 3-session.
- **Wall recall: 94.7% ± 0.7%** • Wall precision: 92.5% ± 2.2%.
- **Pole recall: 78.1% ± 3.4%** • Pole precision: 82.8% ± 6.2%. Operating-point shifted vs 3-session: recall −3.2 pp, precision +4.9 pp (model more discriminating; threshold can be lowered at deploy to recover prior recall).
- **Pole-az RMSE: 15.18° ± 1.20°** (per-fold 15.0 / 17.1 / 13.7 / 14.9). Mean essentially flat vs 3-session (15.27°); σ halved from 2.32 → 1.20. Likely near a real signal floor against the ~20° empirical pole-az std.
- Wall slice RMSE: right 277 ± 55, center 345 ± 48, left 320 ± 50 mm. Right and left improved ~18–19 mm vs 3-session; center essentially flat.
- best_epoch per fold: 38 / 23 / 39 / 50 — comfortably inside the 60-epoch budget.
- **σ-tightening is the headline finding.** Per-fold σ halved on most aggregate metrics: class acc 2.7 → 1.6, wall recall 1.4 → 0.7, pole recall 7.5 → 3.4, pole precision 8.6 → 6.2, pole-az RMSE 2.32 → 1.20. Strong evidence that spatial-coverage variance was the dominant residual noise — arena diversity (more sessions in different layouts) matters more than ping density per arena.
- **q=1 unstuck.** Previously the persistent weak fold (class acc 84.9% in 2-session, 84.6% in 3-session; pole RMSE 22.2° → 19.2°). 4-session lifts it to 88.9% class, 17.1° pole RMSE. Acq04A added the spatial coverage that q=1 had been short of.
- **Means plateauing.** Class acc near its ceiling, pole-az RMSE plateaued. Marginal value of a 5th session is small for the regression head; remaining gain would be further σ-tightening. Acq05 still planned opportunistically.
- Reference points: 2026-05-29 3-session entry (88.9% / 81.3% / 15.27°), 2026-05-29 2-session entry (89.4% / 81.7% / 16.40°), 2026-05-21 baseline (82.7% / 69.8% / 17.19°).

### 2026-05-29 — Two-headed inverse, Acq01A + Acq02A + Acq03A, close-range, 60-epoch CV

- Commit: `664b335` + working-tree refactor of `SCRIPT_TrainInverseModel.py` (CV loop wired into `main()`, `CV_QUADRANTS = [0,1,2,3]`, `EPOCHS = 60`, per-fold artifacts prefixed `inverse_q{0..3}_*`, `split_indices` guards against `val_quadrants`-vs-loaded-sessions drift).
- Sessions: Acq01A (505 pings, new marker), Acq02A (315 pings), Acq03A (~300 pings). Old pre-marker 01A/02A archived in `AcquisitionSessions_OLD.zip`.
- Config: `MAX_RANGE_MM = 1000.0`, `CONE_HALF_DEG = 35°`, `SonarSlicesUQ_TwoHeaded`, 4-fold quadrant CV. ~700 training pings per fold.
- **Class accuracy: 88.9% ± 2.7%** (per-fold 91.7 / 84.6 / 90.7 / 88.7)
- **Wall recall: 92.3% ± 1.4%** • Wall precision: 92.4% ± 3.9%
- **Pole recall: 81.3% ± 7.5%** • Pole precision: 77.9% ± 8.6%
- **Pole-az RMSE: 15.27° ± 2.32°** (per-fold 14.3 / 19.2 / 13.0 / 14.6)
- Wall slice RMSE: right 296 ± 49, center 342 ± 47, left 338 ± 36 mm
- best_epoch per fold: 48 / 22 / 42 / 52 — comfortably inside the 60-epoch budget.
- **q=1 is the weak fold in this dataset** (84.6% class, 70.8% pole rec, 19.2° pole-az RMSE). Best at epoch 22 — identical in 2-session 150-ep, 2-session 60-ep, and 3-session 60-ep runs → spatial/coverage issue in that region of these arenas, not a training-schedule issue. UQ head correctly signals high σ_med 15.8° on q=1.
- **vs 2-session run earlier today** (Acq01A + 02A, same config, 60 ep): pole-az RMSE 16.40° → 15.27° (mean down, σ 3.5 → 2.3); σ tightened on class acc (3.6→2.7), wall recall (2.1→1.4), pole recall (9.0→7.5); class acc mean flat — plateauing near ceiling. Wall slice RMSEs improved 7–29 mm.
- **vs 2026-05-21 baseline** (Acq01A + 02B, two sessions, 435 close-range pings): class acc 82.7 → 88.9, pole recall 69.8 → 81.3, pole precision 70.6 → 77.9, pole-az RMSE 17.19° → 15.27°. Post-calibration inverse fits as well as or better than the pre-calibration version, on different arenas and with more data.
- **Training schedule experiment.** Earlier today the same 2-session CV ran at `EPOCHS = 150` (per-fold best_epoch {21, 37, 71, 96}) and again at `EPOCHS = 60` ({21, 31, 37, 52}). Aggregated CV means were statistically indistinguishable; q=1 and q=2 produced byte-identical val_total because their best models were already found before epoch 60. The long tail of training was wasted compute; `EPOCHS = 60` is now the canonical setting. Post-warmup pole loss does diverge in some folds (climbs to +11 by epoch 150) but best-save catches the convergent model before it matters; LR decay / loss reweighting deferred as cosmetic.
- **LR signal-floor diagnostic** (`SCRIPT_CheckPoleSignal.py`, same data, full range): 72.2% CV acc vs 70.5% wall prior. Distance-stratified acc 80.5 / 69.6 / 57.7 / 68.8% in TOA bins 0–30 / 30–60 / 60–100 / 100–190 samples. CNN-LR gap at close range ~15 pp — CNN doing real envelope work the hand features can't.

### 2026-05-27 — Wall-probe baseline after plank-bundle attempt + revert

- Branch: PyLorex `main` (fast-forwarded from `arena-frame-calibration`, commit `9addb64`). Calibration files restored from a backup zip of the pre-bundle pipeline (`pose_*.{json,npz}`, `camera_system.json`, `c_measured_*.json`).
- Diagnostic: new `PyLorex/script_capture_wall_probe.py`, robot driven flush against several walls; bias = `(tracker-distance-to-nearest-wall) − 48 mm` (robot radius).
- 6 snapshots, per-camera biases (mm):

  | snap | tiger | shark |
  |---|---|---|
  | 1 | −31  | +70 |
  | 2 |  −9  |  —  |
  | 3 | +60  |  —  |
  | 4 | +59  |  —  |
  | 5 | +72  |  —  |
  | 6 |  −1  | +59 |

- Pattern: shark mildly systematic positive (~+65 mm, n=2); tiger spans −31 to +72 mm (no obvious systematic). Overall mean ≈ +25 mm.
- **Interpretation.** Same baseline as before today's experiments — the revert restored the working pipeline. During the day a bundle-adjusted calibration was tried (`script_calibrate_plank.py`) and produced 100–200 mm wall biases via a frame mismatch between `H_raw`-using build-geom and ray-plane-using tracker; reverted to the pre-bundle state.
- **Causal chain** (because the diagnostics on 2026-05-26 vs 2026-05-27 measure different things and it's easy to confuse them). The ≤75 mm wall-probe baseline itself was established by the 2026-05-26 work — specifically the ray-plane projection switch in `Lorex.get_aruco` + plumb-line `camera_system.json` correcting `shark2tiger_delta_y` from −1400 to −1920. The 2026-05-26 perf note below records the cross-camera-disagreement diagnostic that confirmed the fix at the time, using `script_check_camera_agreement.py`; `script_capture_wall_probe.py` didn't exist until today, so wall-flush *absolute* biases weren't measured at the time. Today's plank tooling did not change the live calibration — its only quantitative contribution was independent confirmation (`script_diagnose_plank.py`: plank residual std ≈ 25 mm) that the 2026-05-26 calibration is healthy. The plank-bundle adoption attempt was a separate experiment, rolled back. See `PyLorex/Docs/calibration_process.md` "Design notes / why this pipeline" for the longer architectural version.
- The plank quantitative diagnostic (`script_diagnose_plank.py`) quoted in `calibration_process.md` Phase 5 (mean ≈ +1.5 mm, std ≈ 25 mm, max ≈ 90 mm at tiger far corner) was measured against the bundle-fit calibration, not against the current reverted one. Re-measure against the current calibration when convenient to update those numbers as the authoritative healthy-baseline for the live pipeline.

### 2026-05-26 — Cross-camera tracking agreement after calibration overhaul

- PyLorex changes (working tree, branch unchanged): `Lorex.py:get_aruco` switched to ray-plane intersection at `z = marker_height_mm` (K + dist + R + PnP `t`, all from solvePnP); new `script_set_camera_center.py` writes `c_measured_{cam}.json` + `camera_system.json`; `Settings.py` reads `shark2tiger_delta_{x,y}` from `camera_system.json` with hardcoded fallback; `script_check_camera_agreement.py` now applies `shark2tiger_delta` internally + prints PnP/measured C comparison in the static header.
- Calibration anchor: physical inter-camera distance 2180 mm (tape, plumb-line nadirs), `Cz` 2950 mm both cameras (tape). Derived `shark2tiger_delta_{x,y} = (0, −1920)` mm; previous hardcoded `(0, −1400)` was wrong by 520 mm.
- Diagnostic: `script_check_camera_agreement.py`, 15 manual placements in the FOV overlap (`x ∈ [−1000, +1500]`, `y ∈ [−1300, −870]`).
- **`dy`: mean +7 mm, range [−32, +40] (≈72 mm peak-to-peak), no structure visible.**
- **`dx`: range [−50, +125] (≈175 mm peak-to-peak), clean positive linear slope ~+0.07 mm/mm with `x_mean`.**
- **`|Δ|`: 15–40 mm in arena interior, up to ~125 mm at `x ≈ +1500`.**
- Reference for comparison (same script, same arena):
  - Before any 2026-05-26 fixes (handoff state on 2026-05-22 evening, with stale `delta_y = −1400`): `dy` U-shaped, mean ~−360 mm + ~150 mm peak-to-peak.
  - After `delta_y = −1920` but before H_raw → ray-plane switch: `dy` shifted to mean +165 mm with the same U-shape + linear `dx(x)` slope. `|Δ|` 100–260 mm.
  - After ray-plane switch (this entry): `dy` U-shape gone; `dx(x)` slope still present but flipped sign vs the H_raw era. `|Δ|` 15–125 mm.
- H_raw intrinsic-fit diagnostics from `script_run_homography.py` on the same calibration run: 50/50 inliers, RMS 0.23 mm (tiger) / 0.42 mm (shark), max residual <1.3 mm. PnP reprojection RMSE <1.05 px. Fits are tight *at* the dot grid; the previous bias was purely linear extrapolation outside the dot footprint.

### 2026-05-21 — Two-headed inverse, Acq01A + Acq02B, close-range

- Commit: `088c16d` (canonical at time of measurement)
- Config: `MAX_RANGE_MM = 1000.0`, `CONE_HALF_DEG = 35°`, `SonarSlicesUQ_TwoHeaded`, 4-fold quadrant CV (`SCRIPT_TrainInverseModel.py`)
- Data: 435 close-range pings retained (308 wall, 127 pole) from 460 raw
- **Class accuracy: 82.7% ± 5.9%** (per-fold 75 / 84 / 89 / 83)
- **Wall recall: 87.9% ± 4.4%** • **Pole recall: 69.8% ± 16.6%** • **Pole precision: 70.6% ± 0.8%**
- **Pole-az RMSE: 17.19° ± 1.98°** (still data-starved; empirical pole-az std ≈ 20°)
- Wall slice RMSE (q=3 fold only, val n=81): right 328, center 357, left 334 mm — no aggregated multi-fold number recorded
- q=0 is the persistent weakest fold but no longer an outlier

### 2026-05-21 — Two-headed inverse, Acq01A + Acq02A, close-range (superseded)

- Commit: `f9d4222`
- Same config as the entry above; 379 close-range pings (265 wall, 114 pole)
- Class accuracy: 79.8% ± 7.8% (per-fold 69 / 83 / 88 / 79)
- Wall recall: 86.9% ± 3.9% • Pole recall: 63.6% ± 21.9% • Pole precision: 65.6% ± 9.0%
- Pole-az RMSE: 16.62° ± 1.66°
- Sign-of-azimuth experiment (regression-trained, threshold at 0): 67.5% ± 8.5%. Sign-BCE-trained instead: 61.6% ± 4.5% (worse — denser supervision wins).
- Superseded by the entry above when 02A was replaced by the 02B redo.

### 2026-05-21 — CNN out-of-fold distance stratification

- Trained on all data (full range, Acq01A + Acq02A), OOF predictions stratified by TOA-derived one-way distance:

| range (mm)   | n_wall | n_pole | CNN acc | wall_rec | pole_rec | sign_acc | LR-baseline acc |
|--------------|--------|--------|---------|----------|----------|----------|-----------------|
| 0 – 500      | 53     | 26     | 0.87    | 0.92     | 0.77     | 0.69     | 0.79            |
| 500 – 1000   | 155    | 61     | 0.81    | 0.89     | 0.61     | 0.69     | 0.66            |
| 1000 – 1700  | 88     | 45     | 0.65    | 0.60     | 0.76     | 0.67     | 0.50            |
| 1700 – 3250  | 10     | 17     | 0.52    | 0.40     | 0.59     | 0.47     | 0.70 (n=10)     |

This is the empirical basis for `MAX_RANGE_MM = 1000.0` in the canonical training config.

### 2026-05-20 — Two-headed inverse, Acquisition01A only (initial baseline)

- Commit: `a7328bc` (two-headed pipeline first end-to-end run)
- Config: no range filter, `CONE_HALF_DEG = 35°`, 4-fold quadrant CV; 237 valid pings, 35% pole
- Class accuracy: 73.1% ± 11.4% (per-fold 55, 85, 85, 73)
- Pole-az RMSE: 19.9° ± 1.7° (uninformative; matches data std)
- Wall recall: 82% ± 4% • Pole recall: 58% ± 19%
- Hand-feature LR signal-floor diagnostic: 66% (vs 65% wall prior — barely above)

### 2026-05-11 — Wall-only policy first clean deploy (pre-pole)

- Commit: `eef35c7` (wall-only-policy milestone, last in the pre-pole era)
- No on-disk wall-only RMSE preserved (`SonarModel/slices_results.json` was overwritten when the two-headed pipeline landed). `SCRIPT_TrainSonarModel.py` references a hardcoded baseline ("distance-only model: 142 mm") but the wall-only slices RMSE itself is not recorded anywhere committed.
- Drive recalibration that unblocked the deploy: `drive_yaw_curl_deg_per_mm` −0.03693 → −0.01243, `drive_distance_scale` 0.992 → 0.9972.
- Tracker noise: σ_yaw ≈ 0.6° per fresh frame; fresh-frame rate ≈ 0.8 Hz (DVR/RTSP bottleneck, see `PyLorex/TODO.md`).
