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

- **Paper:** on branch `direct-learning-poletask`. **2026-08-24 closed both open experimental items** (Experiment 1's sonar re-run, and the sensory-deprivation control in both arenas), and **2026-08-25 closed the last one** (the partial sensory clamp, Code item 0c). **No robot work is outstanding.** From here the paper is the critical path. **Experiment 1 Results is DONE (2026-08-25, `cb5fac3` + `1b303fa`)**: rewritten on the re-run, revised by Dieter, no draft notes left in it. Read Paper state 2026-08-25 before touching any of its numbers — three of them are traps (pole recall must be quoted inside the 1200 mm veto, the comparison against the model's own evaluation has to be range-matched, and first-perception ranges are over arrivals only), and the explanation the section carried since 2026-08-04 was refuted by measuring it.  **Methods 2.2.1 (acquiring and labeling) is rewritten for the six-arena dataset, and figs 2 and 3 are regenerated (2026-08-21)** -- see Paper state 2026-08-21 for the numbers and for two traps worth reading before touching a figure (the `\topfraction` float cascade; arena 5 has no interior walls either). **§2.2.2, `tab:inverse-params`, and the whole inverse-model Results are DONE (2026-08-21/23)** -- see Paper state 2026-08-22/23 for what changed and for four figure defects worth reading before building another figure. The section has no draft notes left in it. **Next: the Discussion. Every Results subsection is now done.** (1) *Experiment 1 Results is DONE* (2026-08-25). (2) *Experiment 2 Results is DONE* (2026-08-23) **except the control subsection, which now has two conditions to report rather than one — see Paper state 2026-08-25 (later), which also lists what `fig_sensory_clamp.py` needs**: both arenas, the closing synthesis, three figures, and the curl paragraph -- see Paper state 2026-08-23, which also records a wrong argument about path integration that sat in the paper for several commits and should not be re-derived. The one piece left in that section needing no data is `\dnote[18]`'s surviving item, the biological framing (landmarks against path integration, Neuweiler and Möhres 1967, scratchpad section 6). (3) Still open elsewhere: `\dnote[22]` (which kind of landmark, in the Discussion) and `\cnote[5]`. **`\dnote[17]` is closed** — its seed item was fixed by the re-run, *trials*/*runs* was unified, and Dieter dropped the pole-placement item as not worth the space, deleting the note. **Introduction fully reworked (2026-06-20/21).** Now a clean, contiguous **Par 1–8**: overlap → inverse model + asymmetry → cross-modal inverse training (defined in Par 3) → sonar (consolidated) → vision → synthesis → "this paper" (two tasks) → biological-plausibility close (Par 8). US spelling throughout; terminology standardized to *cross-modal inverse training*. See Paper state (2026-06-21) for detail. **Next:** the intro has **no Discussion hand-off** (the direct/vicarious pointer was dropped) — draft the (briefer) Discussion treatment of direct + vicarious learning, drawing on the retired material commented after `\end{document}` (vision-as-internal-model / planning argument, Mugan2020/Bennett2023, `\cnote{5}`); reconcile Par 7 "pole" vs Methods "wooden dowel of 25 mm"; confirm Task 2 wording ("path integration + landmark recognition") matches the actual experiment **— RESOLVED 2026-08-14: it does, but specify *which kind* of landmark. The robot navigates on boundary geometry, not object identity; both are landmarks in the spatial-cognition sense. See Performance notes 2026-08-14 (landmark removal)**; trim Par 5 small-mammals/rodents acuity redundancy. **Inverse-model Methods (Par 20–22) + Results (Par 23–27, `tab:inverse-results`, `fig:inverse-results`) are drafted and committed (`b53644e`, `7354057`)** around the B architecture + spatial-holdout deployment model (in-sample vs held-out; no generalization claim). **Experiment 1 Methods and Results are now drafted and committed (2026-08-04, `3dee986`/`201b101`/`45c71f3`)** — see Paper state 2026-08-04. **Next on Experiment 1**, all recorded in `\dnote[17]`: the two pole placements are never explained now that the simulation paragraph is cut (the 1400-rollout sweep belongs in Methods, with how the placements were chosen); Methods says *trials* where Results says *runs*; and the P2/S5 sonar/vision pair did not share a controller seed — **RESOLVED 2026-08-21**, the re-run used `START=5` so all ten pairs now match. **Three Exp-1 numbers also changed in the re-run** (cost of sonar, "zero bearing corrections", arrival range): Performance notes 2026-08-21 (evening). Also open: an unfinished `\dnote` at Par 2 on the owl's az/el mapping being a simple 2D→2D map against our object-based inverse — the sentence trails off at "take a look at this paper:". Experiment-1 paragraphs use descriptive `% comments` (not `% Par N` — renumber pending).
- **Second arena — Path07. DEPLOYED AND WORKING, and the manipulation series is complete.** Artifact `PolicyTraining/default_Path07/best_policy_survival.json`, deploy config `70edfa1`. Six 500-step runs, none with a collision: baseline **42 mm** median cross-track and a replicate at **42 mm** (window medians agreeing to ≤15 mm, so the arena has a floor under it, like Path04's ±1 mm); pole displacement twice, in different directions and magnitudes, both a near-null (~30 mm of route shift); P0+P1 removal (+66 east / +102 north over the north lobe, six laps, never negative, baseline elsewhere); and a wall **added** ~450 mm inside the southern boundary and across the trained path (the boundary itself never moved — "displaced" here and in the 2026-08-20 notes is wrong, see Paper state 2026-08-23), which the route stood off by 416 mm at its southernmost point — near-complete capture, against the Path04 blocks' one third. **For thin dowels presence matters and position barely does; for walls position is nearly everything.** Numbers, the measured arena inventory per run, and the live-vs-clean slice analysis that corroborates the channel ablation on the robot are in Performance notes 2026-08-19 → 2026-08-21.
  **Three rules that came out of this arena and bind everywhere downstream:** (1) **never judge a policy by `val_mse`**; (2) **check `Settings.py` holds non-identity calibration before every deploy** (`SCRIPT_RunPolicy` now refuses otherwise); (3) **the inverse is blind below ~400 mm**, so the robot has no sonar collision avoidance at contact range. Detail in Performance notes 2026-08-18 (late).
  **Still open here:** the simulator is systematically pessimistic — its cross-track runs 1.4–1.7× the robot's on *both* paths, because `motion_rot_bias_deg = ±5` was set from badly-calibrated sessions, so every sim survival figure quoted for Path06/Path07 is probably too low. Re-derive it from curl measured in verified-calibrated sessions only. The Path04 landmark series is **not** invalidated by any of this — but do not retrain a Path04 policy without redoing the whole series.
- **Path06 (figure-of-eight) — SUPERSEDED by Path07, arena retired to `TargetArenas/old_stuff/`.** Trained and deployed once; it survived only 27–47% in simulation against Path04's 82.5%, and its single robot run did two clean laps then lost the path at −4.69 °/step of curl, which confounds it. Kept only because two mechanisms proposed for its failure were tested and refuted — **do not act on the "swap dowels for blocks" advice.** Full reasoning in Performance notes 2026-08-18.
- **Code:** on branch `direct-learning-poletask`. **All robot work for both experiments is complete as of 2026-08-25**, item 0c included. Nothing is open on the robot; the paper is the only critical path. Experiment 1: 20 runs 2026-07-30, re-run on the current inverse 2026-08-21 (sonar arm only, 10/10). Experiment 2: two arenas, a replicate on each, pole removal, pole displacement ×2, wall displacement. The 2026-08-08 plan that carried this section for two weeks is fully discharged — extended-horizon inverse (`67c68c0`), Acquisition06A, the retrain on Acq01A–06A, the class-agnostic range head, Path04 v3, the per-episode rotation bias (`db22a14`), the trained policies, the robot ablations, and the path redraw. The reasoning and every number live in Code state and Performance notes 2026-08-08 → 2026-08-21; do not re-derive them here.

  **What is actually open, in rough priority.**

  0. ✅ **Experiment 1's sonar arm re-run — DONE (2026-08-24).** Ten runs on `17af398`; the vision arm was not re-flown. The design is matched and verified, not assumed: `drive_mm` on `approach` rows is 75.0 in all twenty runs, the five protocol constants are identical across the arm, and **all ten controller seeds match their vision partners** — which also closes `\dnote[17]`'s third item. **The headline moved: sonar is 9/10, not 10/10** (P2/S3 exhausted `MAX_STEPS=200` mid-approach, 437 mm out against a 400 mm stop). Dieter's call: **keep the failed run**, no re-fly, no exclusion. Numbers, the failure analysis and the three remaining wording choices are in Performance notes 2026-08-24 (later). Runs are filed in `PolicyRuns/Poles/` alongside their vision partners (done 2026-08-25).
  0b. ✅ **Sensory-deprivation control — DONE, both arenas (2026-08-24).** Six runs with `CLAMP_SENSING = "shuffle"`, six collisions, none past 0.52 of a lap, against intact runs of 8.53 and 6.36 laps with no contact. Numbers, seeds, pool provenance and the two ways a "replicate" of this condition turns out not to be one are in Performance notes 2026-08-24 — **read that before re-running anything here.** The `"const"` variant was flown twice and then retired by Dieter in favour of shuffle (parked in `old_stuff`); the simulated `dead_reckoning` counterpart is deliberately **not** being reported, so the claim rests on the six robot runs alone. Still true and still the point: this shows the controller is not input-invariant, i.e. not running open-loop. It does **not** measure how far dead reckoning carries the robot — qualitative only, never a percentage. It retires the commented-out Experiment 2 Methods paragraph ("the input has no null value, so sensory data cannot be withheld, only replaced") by making the point empirically.
  0c. ✅ **Partial sensory clamp — DONE, both arenas (2026-08-25).** Six runs, six collisions, none past 0.56 of a lap; the two conditions are indistinguishable and the direction flips between arenas. Numbers, the seed trap that cost a Path04 re-fly, and the battery/calibration change the result survived are in Performance notes 2026-08-25 (later) — **read that before re-running anything here.** Dieter's call: **keep both conditions in the paper**, keep_agn as the headline and the full shuffle as the coherent-input reference; neither supersedes the other, because each answers the objection the other leaves open. The original plan text follows, still accurate as a recipe. Runs are filed in `PolicyRuns/Paths/`.

     **Was:** implemented, simulated, robot runs pending (planned 2026-08-25 morning). Dieter's sharpening of 0b: shuffle everything *except* the class-agnostic range head, because a reader need not accept that head as part of the inverse model. Code at `fc55b1a` (`CLAMP_EXEMPT_KEYS`, default `()` so 0b's condition is unchanged); the recipe and the matched seeds are in the comment block above that setting. Simulation on both arenas says it collapses exactly like the full shuffle (0.36 / 0.73 laps, 24/24 collisions) while the counterpart `shuffle_agn_only` laps normally on Path04 — see Performance notes 2026-08-25 (PREDICTION), and **write a new entry with the robot numbers rather than editing that one.** Plan: 3 runs per arena, same pools as 0b, seed forced to each run's full-shuffle partner. **Read Performance notes 2026-08-24 first** — the pool/seed traps and the fact that a pole contact leaves no `crashes.tsv` row (count contacts from the trajectory) apply unchanged here.

     Everything the settings block needs, in one place, so it need not be reassembled from three:

     | | Path04 | Path07 |
     |---|---|---|
     | `POLICY` / `ARENA` | `default_Path04` / `Path04` | `default_Path07` / `Path07` |
     | `POLICY_FILE` | `best_policy.json` | `best_policy_survival.json` |
     | `CLAMP_SHUFFLE_POOL_RUN` | `Paths/default_Path04_run01` | `Paths/default_Path07_run01` |
     | `CLAMP_SHUFFLE_SEED`, run01/02/03 | 4115754309 / 3436990848 / 3685172675 | 3828090684 / 3718331385 / 3403296698 |

     Plus, for every run: `CLAMP_SENSING = "shuffle"`, `CLAMP_EXEMPT_KEYS = ("agn_dist_mm", "agn_dist_sigma_mm")`, `VARIATION = "shuffle_keep_agn"`, and a fresh calibration before each arena. **The banner must print `EXEMPT (still live): agn_dist_mm, agn_dist_sigma_mm`** — if it does not, `CLAMP_EXEMPT_KEYS` did not take and the run is 0b's condition, not this one. Record the results in the same columns as the 2026-08-24 table (steps, laps, median cross-track, what was hit), and file the runs under `PolicyRuns/Paths/`.

     To redo the simulation: `cd Control_code && python SCRIPT_Ablations.py PolicyTraining/default_Path07 best_policy_survival.json` (Path04 takes `best_policy.json`); ~1 min per arena, writes `ablations*.png/.svg` into the policy folder.
  1. **The writing — now the only critical path; no robot item is open.** All three Results subsections and both Methods are done as of 2026-08-25. Experiment 2's control section needs one revision: it now reports **two** conditions, not one (see Paper state 2026-08-25 (later)). What is left is the Discussion, which needs a ground-up rewrite, plus `\dnote[18]`'s biological framing for Experiment 2. **Do not reuse any Experiment 1 number or figure quoted before 2026-08-25**; the figure was rebuilt with the re-run and the veto-based overlays.
  2. ✅ **`exp1_stats.py` reads the current layout — DONE (2026-08-25, `cb5fac3`).** `run_dir()` resolves `PolicyRuns/Poles/` and falls back to the root so a run still being flown works. It also gained `check_design_matched()`, which asserts the 75 mm approach step and the shared controller seed across all twenty runs, i.e. exactly the two faults that have happened. Still open, deliberately: `DATA_FOLDER` in `SCRIPT_RunDirectPolicy.py` writes a fresh run to the root and the move into `Poles/` is manual (that file carries Dieter's session constants).
  3. **`RunPolicy` has no arena guard**, unlike `RunDirectPolicy`'s `check_arena_matches_pole()`. A moved block gives a confusing failure rather than a clear one. Worth adding if the furniture moves between sessions.
  4. **Tracker settling is the weak link in the measurements.** ~3% of steps are glitches, and paired equal-and-opposite yaw residuals are **one bad read**, not two bad steps (a wrong `yaw[n]` biases the step before and after equally and oppositely). `SCRIPT_AnalysePathRun.py` now flags these pairs explicitly. A bad pose also feeds the policy a wrong `prev_rot`.
  5. **Optional extras on the landmark thread**, none needed for the paper: a B1-only displacement repeat on Path04 (the two boxes moved together, so nothing separates them; prediction is that the shift halves); landing `SCRIPT_CheckArenaDelta.py` (the method that works is in Performance notes 2026-08-21, not the 2026-08-17 one); a flat-panel run where the pole was, to change acoustic class while holding range; and a smaller wall displacement (~200 mm, path still flyable) to get a graded capture fraction for walls.
  6. **Parked: gaze.** Off-cone competition costs 15 points of class accuracy at matched range, and rotating to increase angular separation attacks that directly — now steerable by the calibrated confidence. Not needed for either experiment.
  7. **Open and unanswerable in this room:** whether the sonar can classify an *uncontested* object at 2.5 m. Only 27 such pings exist and the arena caps them at ~1.9 m; settling it needs a space ~5 m across.

  **Two measurement cautions that still bind.** (a) **Do not mix informative-perception metrics.** `SCRIPT_AnalysePathRun.py` measures *true geometry, facing along the path, `cls != none`* and is the definition to use; the 26%/49% figures quoted for Path01/Path02 in the 2026-08-07 notes came from an unrecorded definition and are unreproducible. Never quote a single informativeness figure without saying which horizon it assumes. (b) **`SCRIPT_CheckPoleSignal.py` would understate the current data** — it reduces each ping by *global argmax*, which on a pole-with-wall-behind ping locks onto the wall. Restrict it to near range or window the features around the labelled range; do not read a weak result as "no far-pole signal".

- **Uncommitted and deliberately left**: session-state constants only — arena/session names in `SCRIPT_DefinePath.py`, `SCRIPT_TakeEnvSnapshot.py`, `SCRIPT_TrainPolicy.py` (Path07), `SCRIPT_RunPolicy.py` (repeat/variation), `SCRIPT_BuildAcquisitionPlan.py` and `SCRIPT_VisualDataAcquisition.py` (Acquisition06), `SCRIPT_CalibrateRobot.py`'s phase toggle, and a commented-out ping loop in `SCRIPT_TestRobot.py`.

---

## Paper state

*Last updated: 2026-08-25.*
*Current branch for ongoing work: `direct-learning-poletask`. `main` carries up through the direct-learning rename + Par 9 task commitment.*

### Experiment 2's control becomes two conditions, not one (2026-08-25, later)

The partial sensory clamp flew on both arenas (Performance notes 2026-08-25 (later)), and Dieter's call is to **report both it and the full shuffle**. That changes the control subsection and `fig:sensory_clamp`, neither of which has been rewritten yet.

**Why both.** Each condition is vulnerable to the objection the other is not, so neither supersedes the other:
- **Full shuffle** feeds a *coherent* observation — a real measurement from the wrong place. Its weakness: a reader can answer that the class-agnostic range head was never part of the inverse model anyway (monaural time-of-flight, no class, no bearing), so the condition only shows the robot needs *a* sonar signal.
- **`shuffle_keep_agn`** closes exactly that, leaving the agnostic head live and destroying everything distinctively the inverse model's. Its weakness: the input is *spliced* (a live range beside wall slices from another pose), off the training manifold in a way the full shuffle is not.

The control for the splicing objection is `shuffle_agn_only` — equally spliced, laps normally — which exists **only in simulation** and is weak on Path07 (14/24 collisions there). If the paper leans on the splicing argument, that limit has to be stated; do not quote the Path04 simulation alone.

**How to report it.** `shuffle_keep_agn` is the headline condition; the full shuffle is the coherent-input reference, in a sentence with its numbers. **Do not round the two into one number** — the means differ and the direction *flips between arenas* (Path04 0.27 keep_agn vs 0.35 full; Path07 0.53 vs 0.47), which is what no real difference looks like. Say indistinguishable, make no directional claim.

**`fig_sensory_clamp.py` needs work.** It currently hard-codes three runs per panel (`default_Path0{4,7}_run0{1,2,3}_shuffle`, lines 48–50) and `C_CLAMP` has exactly three colours; `run_path()` searches `PolicyRuns/Paths` then the root, so the new runs resolve without change. Six saturated trajectories per panel will be too busy — plot `keep_agn` saturated and demote the full shuffle toward the grey treatment the intact run already gets, or split into a 2×2. The `_numbers.json` sidecar it writes will need the new runs too.

**Wording that must not drift.** The paper says *shuffled input*, never *clamp* (`fac0df9`), and the `"const"` variant is retired and unreported. That convention now has to cover two shuffled conditions — name them explicitly rather than saying "the shuffled runs" and leaving the reader to guess which.

### Experiment 1 Results rewritten on the re-run, and one explanation refuted by measuring it (2026-08-25; commits `cb5fac3`, `1b303fa`)

**The section is complete.** Every number moved, `tab:direct-results` and `fig:direct-results` were rebuilt, and Dieter's revision pass then restructured it: the failed run folds into the opening paragraph, the approach-behaviour paragraph sits ahead of the perception one, the dead commented-out blocks are gone (including `\dnote[17]`, dropped by Dieter's call that the pole placements need no discussion), and a closing summary hands off to Experiment 2. **No draft notes are left in Experiment 1.**

🔴 **The old explanation for why in-run perception beats the model's own evaluation is WRONG, and it had been in the paper since 2026-08-04. Do not put it back.** It said the training echoes were harder because in $44\%$ of them a second pole stood in the cone behind the nearest one, a situation the single-pole Experiment 1 arena never produced. Measured out of fold on the six-session set, pole echoes with a second pole in the cone are classified **better**, not worse: $77.0\%$ against $66.6\%$, and the same sign in every band out to $1200$\,mm. (The current figure for the statistic itself is $39.6\%$ of pole-nearest echoes, not $44\%$; the $44$ predates Acquisition06A.) This is the third mechanism in this project asserted from plausibility and refuted on contact with the data.

**What replaces it is measured, and it accounts for most of the gap.** Two conditions the runs mostly satisfy and the acquisition grid did not: a single pole in the cone, and the pole near the centre of the field, because the controller turns toward whatever it perceives. Numbers in Performance notes 2026-08-25. The text now says the centring restriction does most of the work, which is true and is the opposite of the emphasis the refuted version had.

**Three comparison traps this section walks past, worth knowing before quoting any of its numbers.**
- **Pole recall must be quoted inside the $1200$\,mm veto.** Over all ranges it is $56.7\%$, which sounds bad and means nothing: beyond the veto a pole call is refused by design, so every pole out there is scored as missed by construction. Inside it, $88.1\%$.
- **The comparison against the model's own evaluation has to be range-matched.** The runs and the acquisition set have very different range distributions, so an unrestricted comparison is not a comparison. Like-for-like within $1200$\,mm: $94.3\%$ in the runs against the quadrant models' $87.8\%$.
- **"Better than during training" is the wrong phrase** and was caught in review. The reference figures come from the quadrant models, none of which trained on the echoes it scored. In-sample is a different number ($83.4\%$). The text says "on the acquisition data".

**The cost of sonar now rests on the paired per-trial path ratio** (median $\times 1.75$, range $\times 1.04$ to $\times 5.03$), with the medians still in the table. The ratio of medians moves by $0.55$ when the failed run is dropped where the paired ratio moves by $0.05$, and the ten trials differ in difficulty by an order of magnitude, so a ratio of medians across unpaired trials is the wrong summary for a paired design.

**Two numbers that read as contradictions and are not.** (a) The final-pose bearing under sonar is now "at most $16.4^{\circ}$", outside the $15^{\circ}$ alignment tolerance, because the tolerance is on *perceived* azimuth and that head has ~$14^{\circ}$ RMSE. (b) The first-detection range is quoted as $1018$ to $1140$\,mm, but the failed run first perceived the pole correctly at $578$\,mm. Those rows are over arrivals, which the table caption states; the failed run is the only one outside the band and it falls well inside it.

### Considered and deliberately NOT done: the drop-in / route-recovery test (2026-08-23, Dieter's call)

Release the robot somewhere on the trained path -- or anywhere in the arena -- and see whether it can pick the route up from there on sensory data alone. It looks like the natural next experiment after Experiment 2's manipulations, and it is not one: it asks whether the inverse model's output supports **self-localization**, which is the subject of the separate mapping/SLAM paper. Running it here would import that paper's question into this one.

**The sharper version of the reason, worth having when the Discussion is written.** It is not only that the drop-in test is *about* self-localization; it is that the current design deliberately avoids that question. Every Experiment 2 run starts from the same release region with the recurrent state at zero, so the controller never has to work out where on the loop it is -- only to run forward from a known start, corrected by what it senses. Drop it in at an arbitrary pose and the only way to pick up the route is to recognize the place. The two papers ask different questions and this one is built so it need not answer that one.

**Two things to say in the Discussion when it gets written (Dieter agreed, 2026-08-23, but the Discussion needs a ground-up rewrite first, so nothing was added to `main.tex`).** (1) State the limitation plainly rather than only deflecting: we do not test whether the robot can determine where it is, every run began from the same release region, and "corrects a sequence run from a known start" is a weaker claim than "follows the route". It is a chosen limitation, not one we were caught by, and saying so costs nothing. (2) Point at the separate paper for whether the inverse model's output supports self-localization. Expect the question to arise on its own once the sensory-clamp result is in -- a route that collapses without sensing invites "so could it find the route from sensing alone?".

**Dieter raised this himself and ruled it out in the same breath, and asked to be reminded of the reasoning if it resurfaces**, because he expects to forget why he dropped it. If it comes up, say it was considered and deferred, and give the reason rather than re-arguing it. `CLAUDE.md` already carries the general rule that mapping and SLAM stay out of this paper; this is the specific experiment that rule catches.

### Experiment 2 Results written, and a wrong argument about path integration caught (2026-08-23; commits `5653dff` -> `c85121b`)

**The section is complete**: shared opening, `\subsubsection{Arena 1}`, `\subsubsection{Arena 2}`, and a closing `\subsubsection{Both arenas together}`. Figures `fig_arena1_runs`, `fig_arena2_runs` (four maps + a sliding-window profile each) and `fig_control_loop`. Only `\dnote[18]`'s biological framing is left in it.

🔴 **The paper argued the path-integration point wrongly for several commits, and the wrong version is superficially convincing. Do not re-derive it.** The bad argument: the policy's only non-sonar input is `prev_rot`, which is its own output (`prev_rot = rotate_cmd`, `SCRIPT_RunPolicy.py:637,652`; the trainer matches at `:1054`), the forward pass is deterministic and `initial_hidden()` is zeros, therefore `h_t` is a function of the echo history and the route is "sonar-guided throughout". **The last step does not follow.** A function may ignore its arguments: the recurrence can run almost entirely on `W_h h_t`, consulting the input barely or not at all. And the robot does not need a motion *sensor* to dead-reckon, because `fixed_drive_mm = 150` never varies -- a step count IS an odometer, and a 32-unit RNN has room for a phase variable. Path integration is not merely available to this controller, it is available in the cheapest possible form. Corrected in `195c4c6`: the Methods now pose sensing-versus-internal-sequence as an *empirical* question and the manipulations answer it.

**What survives of the argument, and is still in the paper:** the efference copy carries no information the echo history did not already determine, since the command was computed from it. That is a claim about redundancy, not about dependence, and it is the one `fig_control_loop` draws.

**`fig_control_loop` went through two failed designs before the third worked.** (1) A dead-ended arrow for the missing odometry: a blocked arrow needs something visible to be blocked *at*, and drawing a thing that does not exist as a line that does is a conceit that only reads if you already know the point. (2) The efference copy as a self-loop out of the network and back: that is the standard picture of *hidden-state recurrence*, which this network also has and which is NOT drawn, so the one arc on the page reads as the thing it is not. The version that works, Dieter's layout: features are a label on an arrow rather than a box; the motor command arrives as a second input *parallel* to the features; the same label goes on the arrow to the motors, because it is the same number; and execution error leads to a `New pose` box whose only route back is the next emission. **Absence is shown by there being no arrow.**

**Curl is now in the paper, and it needed the calibration to be introduced first.** `Library/Client.py:191` pre-compensates every drive by `-distance_mm * drive_yaw_curl_deg_per_mm`, *after* the policy has issued its rotation, so anything measured from `rot_deg` against tracker yaw is a **residual**, not the physical fault. Numbers in Performance notes 2026-08-23 (curl). Methods §2.1 now states the calibration; the aggregate numbers close "Both arenas together".

**Two number traps found while writing.** (a) The Arena 2 displacement medians read "43 and 46 mm"; every path sampling gives **42** and 46, and the 43 was carried across from the baselines (42.7, 42.8). Fixed in `c85121b`. (b) `SCRIPT_AnalysePathRun.py` samples the path with `densify(n=12)` -- twelve points per *waypoint segment* regardless of its length -- so its deviation figures run slightly high and vary with how the path was drawn. It agreed with uniform 25 mm sampling to within a millimetre here, but **spot-check any number quoted from that script's console output** rather than from the figure scripts.

⚠️ **The Arena 2 wall was ADDED, not displaced.** The arena's own boundary stayed where it was; panels were placed about 450 mm inside it. Earlier entries (and the Where-to-pick-up bullet, now corrected) say "displaced", which is what the run photographs disprove. The geometry in `arena_features.npz` is the *trained* layout in every run and was never re-digitised, so every manipulation drawn in figs 8 and 9 comes from the recorded measurements and each run's own photograph.

### Inverse-model Results rewritten and figured (2026-08-22/23)

**The section is complete.** No `\dnote`/`\cnote` left in it; 20, 23 and 24 are all discharged. Dieter wrote the prose, the agent supplied and checked the numbers. Every figure number comes from a `*_numbers.json` written by the same script that draws the figure, so prose and figure cannot drift.

**Terminology, now used throughout:** the **quadrant models** (four, one arena quadrant held out each) and the **deployed model** (a fifth, trained on 85% of a single contiguous patch). Methods 2.2.2 was rewritten around that framing by Dieter and reads far better than the two-splits version it replaced. The word *out-of-fold* is gone from the paper.

**Fig. 4 (`fig_inverse_limits`), one panel per output head:** A class, B reliability, C pole azimuth, D pole range, E nearest-reflector range, F wall depth. Scheme: solid+filled = quadrant models, dashed+open = deployed, grey dotted = baselines (not models); colour = subject (blue both classes, red pole, brown wall). C and D share axes so the masked head's flatness reads against the agnostic head's diagonal.

**Fig. 5 (`fig_perception_examples`) and Fig. 6 (`fig_pole_examples`)** show individual perceptions, which the section otherwise lacked entirely. Both pick examples by rule and say so in the caption. Fig. 6 selects among *echoes the model called a pole*, since those are the only ones whose pole channels the robot uses; D--F then show the model confidently reporting a pole at ~800 mm when it is at 1.3--2.4 m, which is the failure the 1200 mm veto exists to catch.

**Supplementary:** `tab:inverse-results` moved there (it is the overfitting check and carries no argument), joined by `fig:pole-range-cap` from `EXPT_pole_range_cap.py`. The main-text Results now carry no table at all, deliberately -- every number in the section is range-dependent and a table flattens exactly the structure the section is about.

**Four defects found in the agent's own figures, all by Dieter looking at the drawing.** Worth reading before building another one:
- 🔴 **Panel B compared the deployed model against a baseline computed over a different set of echoes** (its own 17--23 per band, against the baseline over all of them), so it appeared to beat a rule it had never been measured against. Like for like beyond 2000 mm: baseline 4.88 deg, quadrant 5.68, deployed 6.83 -- it loses. **Rule: when a series is drawn from a subset, every reference line in the panel must come from that same subset.** The deployed series is now dropped from B.
- 🔴 **A left/right flip in Fig. 5.** `SLICE_NAMES` is `("right","center","left")`, index 0 = most negative azimuth = physical right; the drawing had them the other way. Caught by "the true arcs should touch the walls, and they don't". Confined to that figure -- the trainer, error model, deployed feature and panel F all pair `slice_t[:, i]` with `SLICE_NAMES[i]` correctly, checked.
- **Error bars in D traced the diagonal**, which is arithmetic once a head saturates (bias -1454 vs RMSE 1463 at 2000--2500, so mean+RMSE = true). Bars there are now the SD of the predictions, which says the right thing: the head does not become uncertain, it becomes *constant* (SD 53--61 mm far, against 90--136 near).
- **Dashed arcs rendered solid at some radii** because a `Wedge` of width 1 has two nearly-coincident arcs whose dash phases interleave. Arcs are polylines now.

**Two more agent claims that did not survive testing**, both corrected in the text: that the deployed model's holdout echoes are *easier* (difference 1.75 deg, CI [-0.68, +3.45], and band-matched it is not even consistent in sign), and the off-cone competition effect on class (see Performance notes 2026-08-23).

### Both experiments' Methods audited against the code and the runs (2026-08-21, late; commits `17af398`, `ddca375`, `1733fc3`, `93b73e9`, `f2fd64a`, `6fd4fce`)

Everything below was checked against a config, an artifact or a log rather than against the previous draft. The pattern worth carrying: **most of what was wrong here was code that had moved and text that had not.**

**Experiment 1.**
- 🔴 **The 75 mm pole-approach step was in the paper but not in the code**, and the 2026-08-21 re-run therefore drove 150 mm on every approach step while the reused vision runs drove 75. See Code state 2026-08-21 (late). The sonar arm must be re-run.
- The claim that sonar "did not perceive objects beyond its 1 m training range" is gone. The deployed inverse is uncapped; what limits the robot is class degradation past ~1.4 m plus the 1200 mm pole-range veto, which was undocumented and is now in the prose and in `tab:controller-params`.
- "When no reflector was within range, the feature was empty" removed: an empty sonar feature needs the untrained *none* class to win the argmax, and vision runs with `GEOM_RANGE_HORIZON_MM = None`, so in a closed arena neither modality can produce one.
- **The recall justification for the 400 mm stop was retired, and the emission-overlap justification put in its place.** Detail and numbers in Performance notes 2026-08-21 (night). Dieter's overlap argument was right and only its range was wrong (500 mm, measured at 250--340).
- Closest pole in training is **248.8 mm**, not 253.
- *trials*/*runs* unified on runs; `\dnote[17]` item 3 closed (the P2/S5 seed).

**Experiment 2.** Checked against `PolicyTraining/default_Path04/config.json` and `default_Path07/config.json`, which are identical, and against the eleven runs under `PolicyRuns/Paths/`.
- **Teacher lookahead is 240 mm, not 200.** `\dnote[18]` had warned the lookahead is implicitly a ratio to the step and would need scaling when the step moved 125 -> 150; that happened in the code and never reached the text.
- **The per-episode additive rotation bias was missing entirely** (`motion_rot_bias_deg = 5.0`, drawn U(-5,+5) per episode, added at every step *after* the rotation clip). It models drive curl, it is the robot's dominant fault, and it is the perturbation that actually forces the controller to correct drift from sensing -- so the paragraph was making that argument while describing only the multiplicative gains.
- 🔴 **The blind-controller condition was never run.** All eleven runs are of the sonar policy. Dieter's reasoning for dropping it rather than running it: sensing and ego-motion are combined non-separably in the controller, so degrading the sensory input degrades path integration too, and neither can be switched off alone. The paper now says so and asks the question by manipulating the arena instead. The blind comparison survives where it belongs, as `SCRIPT_Ablations.py`'s `dead_reckoning` in simulation on the same trained policy.
- The conditions paragraph now matches what was run: baseline, replicate (as the noise floor a manipulation must beat), removal and displacement of poles and walls, and the second arena with its own path and controller.
- `\dnote[18]` cut from 3363 characters to 929. Its three "assumed rather than established" items are all done; **the one live item is the biological framing.** Two of its own details had gone wrong: it listed the σ channels as part of the observation (σ is off by design, see below) and its echo count and pole-recall bands predate the six-session retrain.

**σ is deliberately not an input.** `use_sigma = False` in both deployed policies. The old "σ inputs are load-bearing" finding belongs to the retired 7-input wall-only policy (`rnn_sup_loop2_h32`) and is not evidence about this pipeline; in the current simulator σ is constant within a distance bin and would identify the true bin. The machine-local memory that said otherwise has been corrected.

### Methods 2.2.1 rewritten for the six-arena dataset; figs 2 and 3 regenerated (2026-08-21; commits `531cc0f`, `21e0c20`, `b5ba2d0`)

**`fig:arena_layouts` (fig 2).** Acquisition06 added as panel F. Layout is now a 2x3 grid (`NCOLS`) because six panels in one row leave arena 6's poles unresolvable; the plan overlay moved from arena 5 to arena 1 (`PLAN_ARENA`). Both are one-line switches in `fig_arena_layouts.py`.

- **Do not write that arena 6 is the one without interior walls.** Arena 5 has none either; it uses a deep boundary notch. Checked by eye at size after nearly writing the contrast into the caption.
- ⚠️ **The float cascade, because it will recur.** At `width=\textwidth` this figure plus its caption misses `\topfraction` (default 0.7 of textheight) by about two points. It is then deferred, and since LaTeX emits floats in order, **every later figure queues behind it and the whole set lands at the end of the document.** The fix needs BOTH `width=0.85\textwidth` and `\renewcommand{\topfraction}{0.85}` (now in the preamble with a comment): `topfraction` alone does nothing at full width, and 0.85 width alone stopped being enough once the caption grew. **Symptom to recognise: figures suddenly at the end of the PDF right after a figure got bigger or a caption got longer.** Caption length counts toward the same budget as the image.

**`fig:network` (fig 3).** Fifth head added (`agn_dist_head=True`), drawn as "Nearest-reflector range / any class, any range". Wiring was read off `SonarModel.forward` rather than assumed: the agnostic head is mirror-symmetric, combined as `(z_LR + z_RL)/2` like the pole-range head, not antisymmetric like pole azimuth. `fig_network.py`'s `CFG` is hard-coded; it agrees with the deployed `SonarModel/inverse_feature_params.json` today (every layer dim, plus `wall3_symmetric`/`wall3_pole_dist`/`wall3_agn_dist` all true) but nothing enforces that. Worth reading from `feature_params` with a `CFG` fallback if it drifts again.

**§2.2.1 rewritten** (uncommitted at time of writing). Six sessions, not five. **2775 echoes, not "about 2100"** (505/315/405/440/470/640 per session; 930 beyond 1 m; most distant 3196 mm; poles 4/4/4/4/5/4). The Barchi flight-time comparison survives the new count: "under 4 minutes" becomes "under 5".

- **The yaw protocol differs for session 6, and this was measured, not read off the config.** Arenas 1--5 are exactly 72 deg apart, uniform, at every position. Arena 6 is `far_biased`: three of five yaws down the longest sight lines, **minimum separation median 40 deg (range 40--60)**. The paper's independence argument (72 deg is well beyond the +-22 deg main lobe) is therefore **scoped to sessions 1--5**; at 40 deg the lobes overlap by about 4 deg. This is 640 of 2775 echoes, 23% of the training set, so it was not optional detail.
- 🔴 **THE `none` CLASS HAS ZERO TRAINING EXAMPLES.** `load_and_filter` prints `empty_cone=0` for all six sessions and `relabelled 0 pings to 'none'`. This is structural, not incidental: recording from inside a closed arena means some wall always falls within the +-35 deg cone, so **the only thing that ever produced a *none* label was the retired 1 m cut.** Recorded as `\dnote[19]`. Consequences NOT yet chased: the deployed model still has a three-way output whose third class is untrained and effectively unused (max `p_none` 6.5e-4 over all 2775 pings), while `fig:network`, `tab:inverse-params` and the Results all still describe wall/pole/none, and **Experiment 1 Results quote 97.5% recall for the abstain class, measured on the old 1 m-capped model.** Whether the paper keeps describing three classes is Dieter's call.

**Dieter's edits in the same pass.** He first trimmed the arena-6 rationale out of the caption as uneven detail (we do not explain the other arenas' boxes either), then reinstated it in the body once it emerged that session 6 also differed in *protocol*, not just layout: a deviation from a stated method needs its motive. He also added a `\cnote` questioning whether the Barchi 6 x 5 min flight-time argument is the right one. **An unclosed `\cnote` silently swallowed the whole labelling paragraph** (two braces opened, one closed; no LaTeX error, the only symptom was the page count moving and the text rendering red). Fixed. Worth remembering: unbalanced `\cnote`/`\dnote` braces fail silently and consume text until some later stray `}`.

### PLANNED — how to interpret the Experiment 2 manipulations (2026-08-21)

Dieter's observation: what was left of `\dnote[18]` is not a Methods gap at all. Methods says what we did; landmarks-against-path-integration is how to *read* what happened. The note stays where it is but is now marked as an interpretation item, and `\dnote[22]` marks the matching spot in the Discussion.

**The interpretive problem, and what already solves it.**

1. **The non-separability argument cuts both ways.** Methods now justifies manipulating the arena on the grounds that sensing and ego-motion cannot be pulled apart in the controller. True, but it means a *null* result under a manipulation cannot by itself be read as "the robot ignored that object" -- the contribution could be present and swamped. The two ~30 mm pole displacements are exactly this case and must not be reported as "poles are not used".
2. **The dissociation is what rescues it.** Pole *removal* moved the route (+66 east / +102 north over the north lobe, six laps, never negative, baseline elsewhere) while pole *displacement* did essentially nothing, twice, in different directions and magnitudes. **Presence matters, position barely does** -- and neither run alone supports that. **Report them as a pair; the comparison is the result.**
3. **The walls go the other way, and that is the headline.** A wall displaced 420--500 mm across the trained path pulled the route 416 mm, near-complete capture, against the Path04 blocks' one third. **For thin dowels presence matters and position barely does; for walls position is nearly everything.**

**What this licenses us to claim.** The controller navigates on **boundary geometry plus object presence, not object identity.** That is a narrower claim than "landmark recognition" and it is the honest one. It also picks the precedent: Neuweiler and Möhres (1967) is about a spatial *Raumbild* rather than a set of recognised objects, which is the same distinction (scratchpad section 6).

**Where each piece gets written.** The dissociation and the wall/pole contrast go in Experiment 2 Results, with the replicate (42 mm against 42 mm on Path07) as the noise floor that makes the null a null. The biological framing and the landmark-kind distinction go in the Discussion.

**The Introduction item is moot.** The old open item "confirm Task 2 wording (path integration + landmark recognition) matches the experiment" no longer applies: the Introduction's two-task paragraph is gone, and *landmark* now appears in the paper only in `\dnote[18]`, in the Schumacher2017 electrosensory example (Par 2), and in the Discussion's "familiar landmark types", which is what `\dnote[22]` flags.

### PLANNED — rewrite the inverse Results around what the model can and cannot recover (2026-08-12)

**Not drafted. Do this after the Experiment 2 work, so the numbers are final.** Decided 2026-08-12 with Dieter. Not extra work: `\dnote[13]` already flags that `tab:inverse-results` and `fig:inverse-results` predate the range head and `\dnote[14]` asks for a pole-range panel, so these have to be redone anyway. The question is only what replaces them, and an aggregate metrics table is now the weakest option available. All supporting numbers are in Performance notes 2026-08-12 (night) and the two entries above it.

**Required by Dieter (2026-08-21), recorded in the paper as `\dnote[20]` at `\subsection{Inverse model}`:** the rewrite must open with a *simple summary of the model's limits* -- what it can and cannot recover -- rather than an aggregate metrics table. That summary is also where the distance point lands, folded in from a `\cnote` that used to sit in Methods 2.2.2: **sonar does not give range for free.** A time of flight yields a distance only when one isolated reflector produced it; with several reflectors in the beam the hard part is attributing an echo to an object. That is exactly what the two range heads separate, and it is the same point as claim 2 below, stated from the reader's side.

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

*Last updated: 2026-08-25.*
*Current branch for ongoing work: `direct-learning-poletask`.*

### The partial sensory clamp, and its simulated counterpart (2026-08-25; commit `fc55b1a`)

Dieter's condition: the full-shuffle control invites the reply that the class-agnostic range head is not part of the inverse model anyway, so run the clamp with that head exempt. Simulated prediction in Performance notes 2026-08-25 (the PREDICTION entry) — it says the route collapses just the same.

**`SCRIPT_RunPolicy.py` gains `CLAMP_EXEMPT_KEYS`**, keys whose live value survives the clamp. **Default `()`, i.e. the condition flown 2026-08-24 is unchanged** — a stale exemption silently converting a plain control run into a partial one is exactly the class of fault this file has produced three times. The comment block above it carries the three lines needed to fly the new condition, including the six recorded shuffle seeds: force the seed to its full-shuffle partner's and run *i* draws the identical measurement stream, which makes the pair a matched comparison rather than two independent samples.

Two details that are easy to get wrong:
- **The draw pool stores the intact measurement, never the spliced one.** Otherwise an exempt live value re-enters the pool and gets redrawn into a *clamped* channel a few steps later.
- **`save_data` now records `policy_input`.** On an intact run it duplicates `sonar_prediction`; on a clamped one it is the only record of what the policy was actually handed, which otherwise survives only as (pool, seed).

The clamp banner names the exemption, so a partial clamp cannot be mistaken for a full one in the console log.

**`SCRIPT_Ablations.py` gains the shuffle family** — `shuffle_all`, `shuffle_keep_agn`, `shuffle_agn_only` — which replace their channels with a joint draw from the teacher-walk observation pool rather than with a median. That is what the robot condition does; the median family stays as it was. The draw uses its own generator so the motion-noise stream the median conditions were measured on is untouched. Also: `RUN_DIR` and `POLICY_FILE` from the command line (Path07 flew the survival artifact, and ablating a policy the robot never flew answers a question nobody asked), an `ONLY_CONDITIONS` filter, and a trajectory grid sized to the condition count instead of a fixed 2×4.

### Experiment 1's runs filed, and `exp1_stats.py` brought onto the current inverse (2026-08-25; commit `cb5fac3`)

**The ten sonar re-runs are now in `PolicyRuns/Poles/`** alongside their vision partners, so the twenty runs of Experiment 1 live in one place. This discharges the filing half of the 2026-08-24 plan item.

**`exp1_stats.py` reads `Poles/` and falls back to the root**, so a run still being flown resolves without editing the file. Three conventions in it changed with the uncapped inverse, and each would otherwise have produced a plausible wrong number:
1. **The $1$\,m abstain horizon is gone from the scoring.** The deployed model never emitted the abstain class in any of the twenty runs, so applying a horizon to the ground truth now only manufactures disagreement. Class agreement is scored against the raw truth.
2. **Pole recall is reported inside the `POLE_RANGE_VETO_MM` veto** as well as over all ranges, because beyond it a pole call is refused by design.
3. **`check_design_matched()` is new and asserts the two faults that have actually happened**: the $75$\,mm approach step on every `approach` row of all twenty runs, and a shared controller seed in all ten pairs. Both pass. A future re-run that quietly reverts either now fails loudly instead of producing a comparison that is not matched.

`fig_direct_results.py` follows the same change: missed poles are those inside the veto rather than inside $1$\,m, and a run that exhausts its step budget ends in a cross rather than a star so the one failure is not read as a tenth arrival.

**Still not done from that plan item:** `DATA_FOLDER` in `SCRIPT_RunDirectPolicy.py` still writes a fresh run to `PolicyRuns/` root, and the move into `Poles/` is manual. Left alone deliberately, since that file carries Dieter's session constants.

### The sensory-deprivation control landed, and three faults in the clamp path (2026-08-24; commits `f11037c`, `11b62f6`, `62b210c`)

`CLAMP_SENSING` existed (`e8e481d`) but had never been flown, and three things were wrong with the path around it. All three were found before or between runs, and each would have quietly weakened the result rather than failing loudly.

1. **The pre-flight preview ignored the clamp** (`f11037c`). `_simulate_rollout` fed `sim.get_clean_measurement` straight into `encode_obs`, so the preview drew a clean intact lap — precisely the wrong expectation for a control run, and useless for guessing where the robot would hit something. Now clamps what the policy sees while judging collision on the true geometry. It earned its keep on first use: 0/9 rollouts survived and it called the const run's circle-then-north-wall correctly.
2. **`"shuffle"` warmed its pool up from its own steps** (`11b62f6`), so step 0 drew the robot's own current measurement and the first ~10 steps were substantially veridical — against a condition that collapses by step 16. `CLAMP_SHUFFLE_POOL_RUN` now points the pool at a previous *intact* run in the same arena: realistic marginals, decorrelated from position at step 0.
3. **The draw was deterministic across runs** (`62b210c`). With ~500 prefilled measurements every draw lands in the prefill, so the input sequence is fixed by (pool, seed) — and with the seed a module constant, repeats differed only through how many steps the preview happened to simulate first. `reset_clamp_pool()` reseeds from the SESSION name and prints the value.

**The general shape, third time in this project:** a control condition looks like it works because nothing errors. The clamp banner, the preview, and the pool all behaved plausibly while measuring something other than what was intended. Results and numbers in Performance notes 2026-08-24.

### The 75 mm pole-approach step restored -- the SECOND thing lost the same way (2026-08-21, late; commit `17af398`)

**`fcaf26d` restored the terminal protocol but missed this one.** `POLE_APPROACH_DRIVE_MM = 75.0` -- the halved forward step taken while a pole is perceived -- had the identical history: in the working tree during the 2026-07-30 runs, never committed, gone when the tree returned to the committed state. The pole branch of `decide()` was returning the full `DRIVE_MM`.

**Proven from the logs, not inferred.** The `drive_mm` column of `trajectory.tsv` on `tag == "approach"` rows reads 75.0 in every 2026-07-30 sonar run and in every reused vision run, and **150.0 in every run of the 2026-08-21 sonar re-run**. So the two arms of Experiment 1 are not matched on the approach step, in a paired comparison whose headline is the cost of sonar in steps and path length. Effect is modest in absolute terms -- sonar takes only 5--8 approach steps per run, so roughly 0.4 m of the 5.54 m median path -- but the design claim "identical controller, different modality" does not hold for those runs.

Restored with its original comment verbatim from the frozen `code_*.zip`, wired through `ReactiveParams.pole_drive_mm`, and now written into `run_summary.json` next to `DRIVE_MM`. Verified by exercising the controller directly: perceiving a pole returns 75 mm, cruising returns 150 mm. The original comment's last line is the point -- *"Set in the shared controller, so sonar, vision and sim all slow down identically -- the comparison stays fair"* -- and that is exactly what broke.

⚠️ **THE SONAR ARM OF EXPERIMENT 1 MUST BE RE-RUN on `17af398` or later** (planned for Monday 2026-08-24). Ten runs, sonar only; the vision runs stay as they are. Every Experiment 1 Results number moves again, including the step and path-length medians, since approach steps will be half as long. **Do not spend effort on those numbers before the re-run.**

**The lesson, now twice over.** A constant that is reported in the paper and lives only in the working tree will be lost. Both recoveries came from `CodeLogger`'s frozen zips, which is the whole reason it exists. When restoring one, check for others: `fcaf26d` fixed what `\dnote[18]` had spotted and stopped there.

### Experiment 1's terminal protocol restored, and the pole-range veto landed (2026-08-21; commits `e81afeb`, `fcaf26d`)

Two changes to `SCRIPT_RunDirectPolicy.py`, both prerequisites for re-running the sonar arm on the current inverse.

**1. Range veto on the pole call (`POLE_RANGE_VETO_MM = 1200.0`).** Refuse a pole call when the class-agnostic range head reads beyond the cap. Measured, not guessed — see Performance notes 2026-08-21 (later). Loading a fold without the agnostic head now raises rather than silently disabling the veto. `None` disables it deliberately. **The veto touches only the direct policy**: `SCRIPT_RunPolicy` hands `encode_obs` the raw prediction dict, so Experiment 2 is unaffected.

**2. The published terminal protocol was not in the code, and had never been.** `APPROACH_STOP_MM = 400`, `ALIGN_MAX_STEPS = 10` and `ALIGN_MIN_DETECTIONS = 3` with its `n_consec` confirmation loop and the "unconfirmed → treat as spurious, carry on" path were written into the working tree on 2026-07-30 *while Experiment 1 was being run*. All 20 runs used them. **None of it was ever committed** — `ALIGN_MIN_DETECTIONS` appears in no revision of that file, and `b96750b` (2026-07-29, the last commit before the runs) already carried 500 / 6 / no confirmation. The tree later returned to the committed state and the rule was lost. It survived only inside each run's frozen `code_*.zip`, which is the whole reason `CodeLogger` exists.

Recovered from `PolicyRuns/old_stuff/direct_P1_S1_sonar_repeat01/code_*.zip` and ported on top of HEAD's library refactor (the as-run file also predates `feature_from_geometry` moving into `Library.LocalFeature`; that part was correctly left alone). The original comment is preserved verbatim — it records that the first real trial went straight from `empty` to reached in one step, having perceived the pole twice in 136 steps. `\dnote[18]` had spotted half of this ("the paper reports a 400 mm stop and ten corrections while the current script says 500 and six"); what it missed is that the confirmation *logic*, not just the constants, was gone.

**This matters more now, not less.** The veto puts pole precision at 88.2% — better than the ~70% that provoked the rule, but a single-ping arrival criterion at 88% still credits false detections as successes.

Also in `fcaf26d`:
- The `stop` row and the per-confirmation rows are logged again, and the sonar ping at the stop step is saved again, so the terminal phase is recoverable from `trajectory.tsv` and the dills rather than only counted in the summary.
- `run_summary.json` now records `align_min_detections` and `pole_range_veto_mm`. Both have drifted out of the script before; a run that does not say which it used is not reproducible.
- **`main()` raised a `KeyError` on every run** — `trial['sim_steps']`, in the startup print. The trial list was regenerated from the as-built arena annotations after the runs and that pass drops the simulated step estimate. Now read with `.get`.

**Before the re-run, two settings the script does not carry correctly:** `ARENA` at HEAD reads `"DirectP1"`, which does not exist — the arenas are **`DirectPole01` (P1)** and **`DirectPole02` (P2)**, confirmed from each run's `run_summary.json`. `check_arena_matches_pole()` would catch a mismatch, but it is a stop, not a fix. Verified in sim: a run logs `stop`, `confirm1`, `confirm2`, then `reached_aligned`, with the summary reporting 400 mm and ten corrections.

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

### 2026-08-26 — **Path04 baseline re-check: still 8.53 laps at 41 mm.** And the post-battery `+0.024` curl was a session transient, not the new normal

Amends the entry below (2026-08-25, later) on one point only; its numbers and conclusions stand. `default_Path04_run03_check` — intact run, `CLAMP_SENSING = "off"`, `CLAMP_EXEMPT_KEYS = ()`, `best_policy.json`, `MAX_STEPS = 500`, flown to confirm the policy still works after the battery change. **500 steps, 8.53 laps, no collision, median cross-track 41 mm** against the 37 mm of `run01`/`run02` — inside this arena's known noise floor. Per-100-step medians 45 / 44 / 47 / 33 / 40 mm, i.e. **flat across 8.5 laps**, so nothing is curling out within a run. `rot_deg` mean −5.3°, sd 14.5°, range ±44°: active steering, not a stuck bias.

**The correction.** That run flew on `drive_yaw_curl_deg_per_mm = -0.01151`, `drive_distance_scale = 1.0015` — Dieter recalibrated again between the keep_agn runs and this one. The curl has flipped **back** to roughly where every pre-battery session sat:

| session | curl | deg/step at 150 mm |
|---|---|---|
| full shuffle (2026-08-24) | −0.01444 | −2.17 |
| keep_agn precalib (2026-08-25) | +0.00591 | +0.89 |
| post-battery, two measurements | +0.02400 / +0.02226 | +3.60 / +3.34 |
| this run (2026-08-26) | **−0.01151** | **−1.73** |

So the `+0.024` figure was a property of *that calibration session*, not of the new batteries, even though it replicated twice within the session to 7%. **Do not read `+0.024` as the robot's post-battery baseline.** Two agreeing measurements inside one session are not evidence of stability across sessions — the calibration constants move by more than their within-session spread, which is exactly why the standing rule is to recalibrate before every deploy rather than to trust a recent value.

What this does **not** change: the three Path04 keep_agn runs of 2026-08-25 genuinely did fly on `+0.02226`, with the sign opposite to their full-shuffle partners' `-0.01444`, and they still landed inside the family. The claim that the collapse survives a large motor-calibration change stands as written.

### 2026-08-25 (later) — **The partial sensory clamp on the robot, both arenas: 6 runs, 6 collisions, none past 0.56 of a lap.** The prediction holds, and keeping the agnostic range head live rescues nothing

Discharges plan item 0c. **This is the measurement; the entry below it (PREDICTION) is the pre-registration and stands unedited.** Code at `fc55b1a`: `CLAMP_SENSING = "shuffle"` with `CLAMP_EXEMPT_KEYS = ("agn_dist_mm", "agn_dist_sigma_mm")`, so the class-agnostic range head keeps its live value and everything distinctively the inverse model's — three wall slices, three class posteriors, pole azimuth, pole range — is replaced by a joint draw from a 500-measurement pool of a previous intact run on the same path. `VARIATION = "shuffle_keep_agn"`; runs filed under `PolicyRuns/Paths/`. Each run's seed was forced to its full-shuffle partner's, so run *i* saw the same drawn measurement stream as run *i* of 2026-08-24, differing in exactly one channel.

**The result** (cross-track = distance to the closed target polyline, the same definition as the 2026-08-24 entry).

| | steps | laps | ct median | ended |
|---|---|---|---|---|
| Path04 keep_agn run01/02/03 | 11 / 27 / 10 | 0.19 / 0.46 / 0.17 | 210 / 73 / 295 | pole, wall, wall |
| Path04 shuffle run01/02/03 (2026-08-24) | 12 / 26 / 23 | 0.20 / 0.44 / 0.39 | 167 / 110 / 298 | pole, wall, wall |
| Path04 intact run01/02 | 500 / 500 | 8.53 / 8.53 | **37 / 37** | no collision |
| Path07 keep_agn run01/02/03 | 40 / 44 / 40 | 0.51 / 0.56 / 0.51 | 162 / 203 / 205 | wall, wall, wall |
| Path07 shuffle run01/02/03 (2026-08-24) | 35 / 37 / 40 | 0.45 / 0.47 / 0.51 | 100 / 99 / 149 | wall, wall, pole |
| Path07 intact run01/02 | 500 / 500 | 6.36 / 6.36 | **42 / 42** | no collision |

Mean laps: Path04 **0.27** keep_agn against 0.35 full shuffle; Path07 **0.53** against 0.47. **The two conditions are indistinguishable and the direction flips between arenas**, which is what no real difference looks like. Report them as one family with no directional claim — an earlier read on Path07 alone suggested keep_agn ran slightly longer, and the re-flown Path04 removed it. Simulation predicted 0.36 / 0.73; the robot gave 0.27 / 0.53, the same Path07 optimism the simulator showed for the full shuffle, in the safe direction.

**What this licenses.** The full-shuffle control (2026-08-24) invites the reply that the class-agnostic range head was never part of the inverse model — it is monaural time-of-flight to the nearest reflector, no class, no bearing, the one output a plain ranger also gives — so that control only shows the robot needs *a* sonar signal. This condition answers it: with that head live and everything else destroyed, the route collapses just the same. The controller is flying on the inverse model's distinctive outputs, not on a generic range channel. Everything the 2026-08-24 entry says about what the control does *not* license carries over unchanged: it is not a measure of how far dead reckoning carries the robot, and no percentage of path-following should be attributed to sonar.

**Both conditions are worth reporting, and neither supersedes the other.** Each is vulnerable to the objection the other is not. The full shuffle feeds a *coherent* observation (a real measurement from the wrong place) but leaves the agnostic-head objection open. This condition closes that but feeds a *spliced* observation — a live range beside wall slices drawn from another pose — which is off the training manifold in a way the full shuffle is not. The control for that objection is `shuffle_agn_only` (equally spliced, laps normally), which exists **only in simulation** and is weak on Path07, where shuffling the agnostic head alone costs 14/24 collisions. Dieter's call 2026-08-25: keep both conditions, report keep_agn as the headline and the full shuffle as the coherent-input reference.

**Three things worth knowing before re-running any of this.**

1. **Forcing all three runs to one seed makes them one sample, not three.** The first Path04 attempt bumped `REPEAT` without bumping `CLAMP_SHUFFLE_SEED`, so all three flew `4115754309`. Verified from the recorded `policy_input`: **18/18 steps with byte-identical shuffled channels** across the triplet, exempt channels differing at every step. This is the 2026-08-24 trap #2 in a new form — the draw is deterministic in (pool, seed), so a shared seed is a shared input sequence. The re-flown runs check out at 0/11, 0/10, 0/10 and 0/40, 1/40, 0/40. Those first three are parked at `PolicyRuns/old_stuff/default_Path04_run0{2,3}_shuffle_keep_agn_dupseed`.
2. **The matched seed does not fix the failure point.** Run01 and run02 tracked their partners closely (11 vs 12 steps, 27 vs 26) but run03 did not (10 vs 23), and Path07's pairs are looser still (40/35, 44/37, 40/40). Two of six is not a rule; do not build an argument on "the same draw produces the same collapse."
3. **The collapse survives a large motor-calibration change.** New batteries were fitted before the Path04 re-fly, flipping the measured curl from `-0.01444` to `+0.02226` deg/mm (−2.17 → +3.34 deg/step at 150 mm) — sign reversed, magnitude ~50% larger, replicated across two independent calibration runs (`+0.02400` then `+0.02226`, 7% apart) so it is real and not an outlier fit. `drive_distance_scale` 0.9627 → 0.9841. The Path04 keep_agn runs still land inside the family, so the result is not an artifact of any one session's drive constants. A third calibration point sits at `old_stuff/default_Path04_run01_shuffle_keep_agn_precalib` (0.31 laps at curl `+0.00591`), correct in every respect and retired only by the recalibration.

Also: **`crashes.tsv` was absent for five of the six runs** — the crash logger fires on a blocked drive and neither a thin dowel nor a glancing wall contact blocks the wheels. Contacts were scored from the trajectory endpoint against the measured arena geometry (nearest wall point vs nearest pole point). Path04 run01 ended 230 mm from the pole against 440 mm from the nearest wall; the other five were walls. Two Path07 runs (run01 and run03) died 85 mm apart on the same wall from independent draws, corroborating the 2026-08-24 observation that Path07's failure point is a property of the arena rather than of the draw.

### 2026-08-25 — **PREDICTION, not a measurement: the partial sensory clamp in simulation.** Keeping the class-agnostic range head live rescues nothing

**Read this as a pre-registration.** No robot ran. These are 24 simulated rollouts per condition under the full training motion model, produced by `SCRIPT_Ablations.py` at `fc55b1a` to decide whether a new robot condition is worth table time. The robot runs are planned for 2026-08-25 morning; when they land, write a new entry rather than editing this one.

**Why the condition exists.** The flown control (entry of 2026-08-24) replaces *every* feature the inverse reports. A reader can answer that the class-agnostic range head was never part of the inverse model to begin with — it is monaural time-of-flight to the nearest reflector, no class, no bearing, the one output a plain ranger also gives — so the control only shows the robot needs *a* sonar signal. Dieter's condition answers it directly: leave that head live, replace everything distinctively the inverse model's (three wall slices, three class posteriors, pole azimuth, pole range). Both deployed policies are the 10-channel `use_sigma=False` layout, so there are no σ channels involved.

**Conditions.** `shuffle_all` = the flown control. `shuffle_keep_agn` = the new one. `shuffle_agn_only` = the counterpart, which exists to answer the one objection to `shuffle_keep_agn`: a spliced measurement (live range beside wall slices drawn from another pose) is internally inconsistent and off the training manifold, so a collapse could be blamed on incoherence rather than on the missing channels. `shuffle_agn_only` is *equally* spliced. Every shuffle condition takes a **joint** draw — one pooled observation supplies all clamped channels at a step, as on the robot; per-channel draws would destroy the joint structure as well as the tie to position, and are a weaker condition.

| condition | Path04 laps / coll | Path07 laps / coll |
|---|---|---|
| full (intact) | 0.99 / 3-24 | 0.91 / 4-24 |
| `shuffle_all` | 0.40 / 24-24 | 0.75 / 24-24 |
| `shuffle_keep_agn` | **0.36 / 24-24** | **0.73 / 24-24** |
| `shuffle_agn_only` | 0.97 / 4-24 | 0.86 / 14-24 |

Path04 is `best_policy.json`, Path07 `best_policy_survival.json` — the artifacts actually flown in each arena. Median-clamp conditions in the same run are unchanged from earlier: `full` 0.99/3-24 and `dead_reckoning` 0.33/24-24 on Path04, matching the 2026-08-13 figures.

**What it predicts.** Keeping the agnostic head live is indistinguishable from destroying it too (0.36 vs 0.40; 0.73 vs 0.75), 24/24 collisions on both paths. Expect all six robot runs to collide and none to pass ~half a lap.

**Two reasons to trust the prediction, and one caveat.**
1. **The simulator reproduces the flown control on Path04.** `shuffle_all` gives 0.40 mean laps against the robot's 0.20 / 0.44 / 0.39, all colliding. On Path07 it is optimistic (0.75 against 0.45 / 0.47 / 0.51) but calls the same 24/24 collapse — consistent with the known pessimism/optimism wobble in `motion_rot_bias_deg`, and in the safe direction here (the sim over-estimates survival, and still says the condition collapses).
2. **The incoherence objection is answered on Path04**: `shuffle_agn_only` is equally off-manifold and laps normally (0.97 laps, 4/24 collisions against 3/24 intact), so splicing per se is not what kills the route.
3. **Caveat — that argument is weaker on Path07**, where shuffling the agnostic head alone costs 14/24 collisions, matching `no_agn_range`'s 14/24 in the median family. On that path the head genuinely carries work. Report it as-is; do not lean on Path04 alone.

### 2026-08-25 — **Experiment 1's perception, measured properly against the model's own evaluation.** The gap is real, it survives range-matching, and the explanation the paper carried for it is refuted

All from the twenty runs in `PolicyRuns/Poles/` through `Paper/images/scripts/exp1_stats.py` at `cb5fac3`, and from the four quadrant folds (`inverse_q0..q3`) out of fold over the 2775 six-session echoes. No new robot runs; this is analysis of the 2026-08-24 re-run recorded in the entry below, which stands.

**In-run class agreement against the raw truth** (nearest reflector in the $\pm 35^{\circ}$ cone, no horizon applied, 672 sonar steps):

| | all ranges | within 1200 mm |
|---|---|---|
| agreement | 85.3% (n=672) | **94.3%** (n=335) |
| wall recall | 98.3% | 98.5% |
| pole recall | 56.7% | **88.1%** |
| pole precision | 93.7% | 97.5% |
| majority baseline | 68.8% | 59.7% |

⚠️ **Never quote the 56.7%.** Beyond the 1200 mm veto a pole call is refused by design, so every pole out there is scored as missed by construction. Worse, beyond the veto in-run "accuracy" equals the band's base rate exactly (1400--1700: 80.2% against a majority of 80.2%; 1700--2000: 79.1% / 79.1%), because the veto forces every call to wall and the truth out there is mostly wall. **Past 1200 mm the in-run accuracy figure measures the arena, not the model.**

**Like-for-like against the quadrant models, restricted to the same 1200 mm** (2143 training echoes): agreement 87.8%, wall recall 94.7%, pole recall 73.9%, pole precision 87.6%, majority 66.5%. So the in-run advantage is real and survives range-matching. In-run accuracy also exceeds out-of-fold in *every* band (200--500: 97.8 vs 93.2; 500--750: 96.3 vs 91.3; 750--1000: 96.9 vs 82.7; 1000--1400: 79.5 vs 71.1).

🔴 **The paper's explanation for that advantage was wrong and is now deleted.** It held that the training echoes were harder because a second pole often stood in the cone behind the nearest one. Out of fold, pole echoes **with** a second pole in the cone are classified **better**: 77.0% (n=400) against 66.6% (n=610), and the same sign in every band to 1200 mm (200--500: 84.4 vs 71.3; 500--800: 90.0 vs 73.7; 800--1200: 74.4 vs 62.8; it only reverses past 1700, on 41 echoes). The statistic itself is also stale: on the six-session set it is 39.6% of pole-nearest echoes and 20.5% of all echoes, not the 44% in the paper, which predates Acquisition06A.

**What does explain it, measured.** The runs are not a random draw from the training distribution, in two ways, and applying both to the training echoes closes most of the gap:

| training echoes within 1200 mm, out of fold | n | accuracy | pole recall | wall recall |
|---|---|---|---|---|
| all | 2143 | 87.8% | 73.9% | 94.7% |
| ≤1 pole in the cone | 1729 | 88.8% | 68.6% | 95.6% |
| pole within 15° of the heading | 1741 | 91.7% | 78.1% | 94.7% |
| **both** | 1496 | **93.0%** | 76.8% | 95.6% |
| *the runs, for comparison* | *335* | *94.3%* | *88.1%* | *98.5%* |

**The centring restriction does the work; the single-pole one is small and mixed** (+1.0 point of overall accuracy through wall echoes, and it *lowers* pole recall, consistent with the refutation above). The sampling difference itself is large: over in-run steps with a pole nearest within 1200 mm the pole lies a median **4.8°** off the heading and within 15° on **80%** of them, against **17.0°** and **44%** in the training echoes. The mechanism is closed-loop: the controller turns toward what it perceives, so a pole seen once is centred on the next step. ⚠️ Do not push this further into a causal claim per band. Out of fold, centring helps below 800 mm (86.3 vs 68.5, 91.1 vs 71.4) and **reverses** at 800--1200 (62.3 vs 71.2).

**The veto is a hard ceiling on detection, and that is what makes first-perception range so repeatable.** Over all 119 correct pole calls of the sonar arm the largest true range is **1140 mm**; not one anywhere beyond it. First correct detection per arrival: 1018, 1025, 1059, 1060, 1068, 1071, 1105, 1131, 1139 mm, a 121 mm spread over five start poses and two placements. Vision, ungated, spans 765 to 2688 mm. **Quote first-perception numbers over arrivals only** and say so: the failed run first perceived the pole correctly at 578 mm and is the sole value outside the band.

**All eight spurious pole calls of the experiment came from wall echoes at 1.1--1.4 m** (in-cone truth 1110, 1143, 1182, 1263, 1290, 1302, 1354, 1383 mm), all in P2 runs. The veto cannot catch these: it tests the range to the nearest reflector whatever its class, and that range was inside 1200 mm. This is the class head failing at the range where Fig. 4A says it fails, not the range head under-reading.

**The failed run (P2/S3) step by step**, since "a perception miss that ran out of budget" undersells it. The pole was the nearest reflector within 1200 mm on eight of 200 steps: 1138, 1160, 1034, 850, 717 mm were all called **wall**, and only the last three (578, 507, 437) were called pole. It also made three false pole calls earlier (steps 59, 60, 149) that put it into approach mode toward nothing. So it had five earlier chances at ranges the other nine runs converted routinely, and latched on only at 578 mm with three steps left.

### 2026-08-24 (later) — **Experiment 1's sonar arm re-run on the restored approach step: 9/10, not 10/10.** The design is finally matched, and the cost of sonar depends on how the one failure is counted

Discharges the 🔴 item carried since 2026-08-21. Ten runs, sonar only; the vision arm was not re-flown. Script at `17af398` (the commit that restored the step), session constants `SENSE_SOURCE="sonar"`, `SUFFIX="_repeat01"`, `MAX_STEPS=200`, arenas `DirectPole01` (P1) and `DirectPole02` (P2), `START` 1→5 in each. Calibration for all ten: curl −0.0401 °/mm, drive scale 0.9838. Supersedes the 2026-08-21 (evening) entry, whose ten sonar runs are void.

**The design is matched now, and this was checked rather than assumed.** `drive_mm` on `approach` rows reads **75.0 in all ten sonar runs and all ten vision partners**. `approach_stop_mm` 400, `align_max_steps` 10, `align_min_detections` 3, `pole_range_veto_mm` 1200, `inverse_fold` `deploy` — identical across all ten. **All ten controller seeds match their vision partners**, which closes `\dnote[17]`'s third item (the P2/S5 pair was the one that did not share a seed). `n_align_corrections` is **0 in every run of both arms**, consistent with the 2026-08-21 finding.

| trial | sonar steps | sonar path mm | vision steps | vision path mm | path ratio |
|---|---|---|---|---|---|
| P1/S1 | 65 | 8614 | 29 | 1958 | ×4.40 |
| P1/S2 | 37 | 4408 | 34 | 2371 | ×1.86 |
| P1/S3 | 38 | 4678 | 26 | 1997 | ×2.34 |
| P1/S4 | 74 | 9736 | 52 | 5726 | ×1.70 |
| P1/S5 | 12 | 831 | 13 | 762 | ×1.09 |
| P2/S1 | 34 | 3949 | 30 | 2193 | ×1.80 |
| P2/S2 | 83 | 10958 | 50 | 6752 | ×1.62 |
| **P2/S3** | **200** | **29316** | 52 | 5831 | ×5.03 |
| P2/S4 | 92 | 12077 | 91 | 11600 | ×1.04 |
| P2/S5 | 37 | 4327 | 33 | 2905 | ×1.49 |

**Outcome: sonar 9/10 reached-and-aligned, vision 10/10.** Dieter's call is to **keep the failed run** — no re-fly, no exclusion.

| | all 10 pairs | 9 pairs (excl. P2/S3) |
|---|---|---|
| median steps, sonar / vision | 52 / 34 (×1.54) | 38 / 33 (×1.15) |
| median path, sonar / vision | 6646 / 2638 mm (×2.52) | 4678 / 2371 mm (×1.97) |
| **paired per-trial path ratio** | **median ×1.75** (range ×1.04–×5.03) | **median ×1.70** (×1.04–×4.40) |

**Prefer the paired per-trial ratio.** It moves by 0.05 when the failed run is dropped, where the ratio of medians moves by 0.55. The trials differ hugely in difficulty (vision alone spans 762–11600 mm), so a ratio of medians across unpaired trials is the wrong summary for a paired design.

**What went wrong in P2/S3 — a perception miss that ran out of budget, not a lost robot.** The pole was within 1200 mm on **85 of 200 steps** and the inverse called `pole` on **3** of them (6 `pole` calls in the whole run). But the run did not wander: the final three rows are `approach` steps with the true pole range closing 577 → 507 → **437 mm** when the step budget expired, against `APPROACH_STOP_MM = 400`. **It stopped 37 mm short of the success criterion.** `MAX_STEPS = 200` was the budget in all three generations (verified from the frozen code zips), so this is a genuine draw and not a changed rule. Re-flying this trial alone would be re-rolling one bad outcome; raising the budget would require re-flying all ten.

**Against the superseded generations** (both parked, see below). 2026-08-21 (150 mm approach, void): 10/10, 39 steps, 5540 mm. 2026-07-30 (original inverse): 10/10, 74 steps, 10156 mm. The successes-only path median **fell** to 4678 mm, which is what halving the approach step predicts — the same number of approach steps at half the distance.

**Where the data is.** New runs written to `PolicyRuns/direct_*_sonar_repeat01` (root) for filing into `PolicyRuns/Poles/`. The void 150 mm generation is `PolicyRuns/old_stuff/direct_*_sonar_repeat01_approach150`; the 2026-07-30 originals keep the plain name in `old_stuff`. Distinguish them by the `drive_mm` column on `approach` rows — 150.0 is the void generation.

**Three choices left for the paper**, none of them data questions: report medians over all ten or over the nine successes; how to word the failure (recommended: plainly, as a perception miss that exhausted the step budget mid-approach at 437 mm); and whether the cost-of-sonar claim rests on the paired ratio (recommended) or the ratio of medians.

### 2026-08-24 — **The sensory-deprivation control, both arenas: 6 runs, 6 collisions, and never more than half a lap.** Plus two ways a "replicate" of this condition can be no replicate at all

Discharges the 2026-08-24 plan item ("run the sensory-clamp control condition"). Commits `f11037c`, `11b62f6`, `62b210c`. `SCRIPT_RunPolicy.py` with `CLAMP_SENSING = "shuffle"`: every feature the inverse reports is replaced by a measurement drawn at random from a pool of 500 real predictions taken on the same path by the same policy — realistic marginals, no tie to where the robot actually is. Fresh calibration before each arena (Path04: curl −0.01444 °/mm, scale 0.9627; Path07: −0.0401, 0.9838).

**The result, measured the same way throughout** (cross-track = distance to the closed target polyline; this definition reproduces the handoff's recorded 42 mm Path07 baseline exactly).

| | steps | laps | ct median | ended |
|---|---|---|---|---|
| Path04 shuffle run01/02/03 | 12 / 26 / 23 | 0.20 / 0.44 / 0.39 | 166 / 109 / 298 | pole, wall, wall |
| Path04 intact run01/02 | 500 / 500 | 8.53 / 8.53 | **36 / 37** | no collision |
| Path07 shuffle run01/02/03 | 35 / 37 / 40 | 0.45 / 0.47 / 0.51 | 100 / 99 / 149 | wall, wall, pole |
| Path07 intact run01/02 | 500 / 500 | 6.36 / 6.36 | **42 / 42** | no collision |

Distance held collapses **22×** on Path04 (8.53 → 0.39 median laps) and **13.5×** on Path07 (6.36 → 0.47); cross-track inflates 4.6× and 2.4×. No run of the six reached 0.52 of a lap. Path07 replicates far more tightly (three runs spanning 0.06 laps) than Path04 (0.20–0.44), so its failure point is a property of the arena, not of the draw. **Compare arenas by distance, not lap fraction** — Path07's lap is a third longer (11792 mm, 78.6 steps/lap, against 8794 mm and 58.6), so its 37 steps is 5550 mm against Path04's 3450 mm.

**What this licenses, and what it does not.** A controller that follows the path open-loop is by definition invariant to its input; the route collapsed, so it is not. That is the whole claim. It is **not** a measure of how far dead reckoning carries the robot: the obs encoding has no null value, so this feeds a false reading rather than an absent one, and the controller may be steering wrongly rather than merely uninformed. Qualitative only — do not quote a percentage of path-following attributable to sonar.

**Two methodological traps, both found the expensive way.**

1. **`CLAMP_SENSING = "const"` is a closed autonomous system, so const runs are not replicates.** With a constant observation, a zero-initialised hidden state, and `prev_rot` fed back as the *commanded* value, the policy takes no input from the world at all and emits one fixed motor program. Two Path04 const runs produced **bit-identical commanded rotations at all 16 steps** (max abs diff 0.0), differing only through start pose and motor noise: both 16 steps / 0.27 laps, circling at −28.7 ± 5.9 °/step for ~11 steps (a fitted circle of R = 312 and 328 mm, max residual < 40 mm) before the rotation collapsed to ≈ +2 °/step and the robot ran into the north wall. Dieter retired the const condition in favour of shuffle; runs parked in `PolicyRuns/old_stuff/default_Path04_run0{1,2}_clamped_const`.
2. **A prefilled shuffle pool makes the draw deterministic too, unless the seed is per-run.** With ~500 prefilled measurements, a replay of 30 steps showed **0 of 30** draws landing on a live step — every one hits the prefill. So the input sequence is fixed by (pool, seed), and with the seed a module constant, repeats differed only through how many steps the pre-flight preview happened to simulate beforehand: real variation, but accidental, unrecorded, and gone entirely at `PREVIEW_N=0`. `reset_clamp_pool()` now reseeds from the SESSION name and prints it. Seeds actually flown — Path04 `4115754309` / `3436990848` / `3685172675`, Path07 `3828090684` / `3718331385` / `3403296698`; pools `Paths/default_Path04_run01` and `Paths/default_Path07_run01`, 500 measurements each, all 15 keys present.

Also worth knowing: **the pre-flight preview did not honour the clamp** (`f11037c`) — it fed `sim.get_clean_measurement` straight to `encode_obs` and drew a clean intact lap, the opposite of what the run would do. Fixed before the first run, and it immediately predicted the const circle-then-north-wall correctly, 0/9 rollouts surviving. And **a pole contact leaves no `crashes.tsv` row** — the crash logger fires on a blocked drive, and a thin dowel does not block the wheels. Two of the six runs ended against poles; count contacts from the trajectory, not from that file.

One further Path04 shuffle run (27 steps, 0.46 laps, 155 mm) was flown before the per-run seed landed and is parked at `old_stuff/default_Path04_run01_clamped_shuffle_unseeded`. It sits inside the family and is corroboration, not a discrepancy; its exact draw sequence is not replayable.

### 2026-08-23 — **Residual motor faults on all eleven Experiment 2 runs.** The robot flew a lap with 237° of heading it could not know about

Regenerated by `Paper/images/scripts/curl_stats.py` (commit `321afec`), which also writes `curl_stats.json`. Supersedes the four-session figures in the 2026-08-20 entry, which it reproduces: that entry's −0.54 and +3.01 °/step are Path07 `run01` and `run01_moved_poles`, and this fit gives −0.51 and +3.02 from the eleven runs.

**Method.** Each logged row carries the pose the robot sensed from and the rotation the policy commanded there, so the heading change to the next row answers that command: `dyaw = gain · rot_cmd + curl`. Trimmed least squares, because ~3% of tracker reads are glitches. The run's own calibration constant is read back out of its archived code zip, so correction, residual and implied physical curl stay together — **they are not interpretable apart**, since `Client.step()` applies the correction below the policy.

| | across the 11 runs |
|---|---|
| physical curl | −1.9 to −6.1 °/step |
| correction applied | +2.2 to +9.2 °/step |
| residual after correction | −1.2 to +3.0 °/step |
| rotation gain | 0.897 to 1.000 (SE per run 0.006-0.009) |
| largest residual, per lap | **237°** (Path07 `run01_moved_poles`, 79-step lap) |

**The confound resolves in the paper's favour, and better than the 2026-08-20 entry claimed.** All three Arena 2 manipulations on `run01` sat at +2.2 to +3.0 against baselines at −0.5 and −1.0, so curl and manipulation are confounded there. But the run carrying the **largest residual of the entire experiment**, +3.02, is `run01_moved_poles` — the pole displacement, which tracked at 42 mm against 43 mm baselines. A lap accumulating 237° of unaccounted heading produced no route change at all, so curl does not manufacture the removal and wall effects.

**The rotation error is one-sided: the robot never over-turned.** Every gain is at or below 1, the largest being 0.9998 ± 0.0067 — indistinguishable from unity but not above it, with the next at 0.993. Consistent with wheel slip during a turn, which the calibration table is fitted to add back and which grows again between calibration and run. This matters for the paper's argument: a signed fault accumulates over a lap where a symmetric one would partly cancel, and the Results now say so.

⚠️ **A tempting wrong explanation for the one-sidedness, already checked and ruled out.** `Library/Client.py` does `angle = int(angle + correction)`, truncating toward zero, which shaves up to a degree off every command in the shrinking direction — inherently one-sided, and at the observed mean commanded rotation of ~12° it predicts a uniform ~4% shortfall. But several runs come in at 0-2%, so it cannot be the driver: the calibration table is fitted on measured desired-versus-obtained pairs and already absorbs it. Do not re-derive this.

**Calibration goes stale between sessions.** The correction is fitted per session and over-corrects by up to 3 °/step by run time; the physical fault itself drifts roughly 2× across sessions. This is why the residual changes sign run to run.

### 2026-08-23 — **The inverse's limits, measured properly.** The 1.4 m boundary was a bin edge; the pole-range cap is still right, for a reason that has changed

All out-of-fold over the four quadrant folds (`inverse_q0..q3`), every one of the 2775 echoes scored once by the model that did not train on it. Numbers regenerate into `Paper/images/scripts/fig_inverse_limits_numbers.json`.

⚠️ **The fold prefix matters and the naming is a trap.** `inverse_q*` are the folds carrying BOTH range heads, i.e. the deployed architecture. **`inverse_agn_q*` is NOT the agnostic-head variant** despite the name -- it is the 2026-08-12 diagnostic in which the existing pole-range head was *retargeted*, so it has no separate agnostic head at all. Checked from the checkpoints' `state_dict` keys. **The range table in Performance notes 2026-08-12 (night) is from that variant, not from what is deployed.**

**Where each cue stops beating its baseline, measured without bins** (sliding 300-echo window, 300x bootstrap):

| cue | crossing | 95% CI |
|---|---|---|
| class vs local base rate | **1537 mm** | 1439--1884 |
| pole azimuth vs "straight ahead" | **2168 mm** | 1770--2301 |

**So the 1.4 m boundary quoted across this project since 2026-08-12 was an artefact.** That analysis binned at 1000/1400/1700 and concluded the cliff was at 1400; at 1400 the model is in fact still **7 points ahead** of the base rate. The two cues also do not fail together: class dies first, azimuth holds a few hundred millimetres longer (the CIs do overlap between 1770 and 1884, so do not quote a precise gap). Beyond 1700 mm class is *below* the base rate -- worse than ignoring the echo.

**Where 1400 reached the running system: nowhere.** The deployed veto is 1200 mm, set from measured precision/recall. The other occurrences are bin edges and analysis constants. The only one with influence is `SCRIPT_AnalysePathRun.py`'s `horizon=1400` default, used to judge whether a drawn path is perceptually adequate -- conservative against the measured 1537, so Path04 v3 and Path07 need no revisiting.

**The pole-range cap re-tested on the six-session data** (`EXPT_pole_range_cap.py`, three spans x four folds). Dieter's question was whether the 1 m cap is a remnant from before Acquisition06, since the original comparison ran on five sessions with 222 far poles and none past 1772 mm. It is not a remnant, but the evidence for it is now current rather than inherited:

| true range | cap1000 bias/RMSE | cap1700 | uncapped |
|---|---|---|---|
| 200--500 | **+36 / 103** | +123 / 230 | +167 / 279 |
| 1000--1400 | −403 / 455 | −134 / 230 | −81 / 240 |
| 2000--3300 | −1687 / 1717 | −1055 / 1102 | **−6 / 251** |

Collateral is a wash or slightly better uncapped (class .803 -> .812, azimuth 15.9 -> 15.4 deg). So uncapped genuinely works at landmark range, and pays for it with close-range bias in the band where the 400 mm stop is decided. `cap1700` is the worst of the three. **Keep the cap at 1 m** -- and note the decisive argument is not the trade-off but that *we already have far-range pole ranging by another route*: the agnostic head's target IS the pole's range whenever a pole is nearest (303 mm RMSE beyond 2 m), and `use_agn=True` in both deployed policies. What limits poles as landmarks is naming, not measuring.

**Out-of-cone competition, tested three ways.** Median agnostic-range |error| for echoes with vs without a nearer reflector outside the analysed cone:

| window | 500--1000 mm | 1000--1400 mm |
|---|---|---|
| ±50° | 57 -> 95 | 103 -> 254 |
| ±70° | 58 -> 73 | 83 -> 193 |
| ±180° | 60 -> 64 | 81 -> 153 |

Same sign under every definition, growing with range. **But the same test on CLASS does not survive**: at 1000--1400 the sign flips with the window (clean 77.0% vs contested 68.9% at ±70, but 57.9% vs 72.2% at ±180, where "clean" is 38 unusual poses). **So the recorded 80.6/65.1 off-cone class figures must not go in the paper.** Beyond 1400 mm nothing can be tested either way: 240 of 245 echoes at 1400--2000 are contested, and 176 of 176 beyond 2000.

**Other numbers now on record.** Pole-range head saturation: mean prediction 768 mm for all 429 poles beyond 1 m, and its σ does NOT flag the failure -- error grows 25-fold while σ grows by half, so at 2--3.3 m it reports ±145 mm while being wrong by 1.66 m. **A predicted variance is not an out-of-distribution detector**, which is the mechanism behind building the veto on the agnostic head instead. Wall slices by range: 273/268/266 mm at 200--500, peaking near 400 at 1--1.4 m, then *improving*, with the centre slice best of the three beyond 1700 (203--247). Calibration ECE 0.020; p>=0.9 -> 97.8% on 42.9% of echoes.

### 2026-08-21 (night) — **Three measurements taken while auditing the Experiment 1 Methods.** Pole recall no longer falls at close range, and the emission overlap ends near 300 mm

All three from the six-session training set through the deployed inverse (`InverseModel.load('SonarModel', fold='deploy')` over `load_and_filter()`, 2775 pings). Overall class accuracy on that set is 82.7%, which is the sanity check that the prediction path was wired right. ⚠️ Note the ear indexing: `sonar` is `(N, 200, 2)`, samples × time × ear, so the ears are `sonar[:,:,0]` and `sonar[:,:,1]`. Slicing the middle axis gives a time slice and yields ~0% recall, which looks like a broken model rather than a broken index.

**1. Pole recall by true pole range — flat, not falling.**

| band (mm) | n | recall |
|---|---|---|
| 200–300 | 30 | 83.3% |
| 300–400 | 69 | 85.5% |
| 400–500 | 66 | 83.3% |
| 500–700 | 171 | 87.1% |
| 700–1000 | 245 | 83.3% |
| 1000–1400 | 226 | 73.9% |

Absolute levels are optimistic (most of these pings were in training), but the *shape* is the point. **The paper's second justification for the 400 mm stop — recall falling from ~82% at 400–500 to ~55% at 200–300 — was a property of the retired 1 m-capped model and is now false.** Removed from the Methods.

**2. Closest pole in the training set: 248.8 mm** (planner clearance 250 mm). The paper said 253. The extrapolation argument for the stop is unaffected and is now exact.

**3. The emission ringdown, measured on the envelopes.** Median over pings and ears; 10 kHz sampling, so 17.15 mm of round-trip range per sample. Sensors saturated over samples 0–5 (**86 mm**); decayed to within 10% of the echo-field background by sample 14 (**1.40 ms, 240 mm**) and 5% by sample 16 (**1.60 ms, 274 mm**). So the emission runs about 1.5–2 ms and overlap ends near **250–340 mm**.

**This settles the 2026-08-04 disagreement.** That pass cut the emission/echo-overlap justification because "the overlap sits near 88 mm". That figure is the *saturation plateau*, not the end of the emission — both measurements are right and they measure different things. Dieter's overlap argument was correct; only its range (500 mm) was wrong. The full emission is the relevant quantity, and it makes the stop a positive argument: **400 mm puts the final range estimate just outside the overlap.** That is now the paper's second reason.

**Caveat kept out of the paper but worth holding:** training echoes reach 248.8 mm, *inside* the overlap, and the model still classifies them at 83% recall. Overlap degrades the echo, it does not destroy it. If a reviewer asks how there is training data at 249 mm given the overlap, that is the answer.

**Per-class counts, for the record:** wall 1765, pole 1010, none 0 (in-sample 1522/835, held-out 243/175).

### 2026-08-21 (evening) — **Experiment 1 re-run on the current inverse: 10/10 again, and the cost of sonar roughly halves.** Three published numbers change

Ten sonar runs, `PolicyRuns/Poles/direct_{P1,P2}_S{1..5}_sonar_repeat01`, commit at run `fcaf26d` (+ `c881d9f` after). `INVERSE_FOLD="deploy"` (current uncapped model with the agnostic range head), `POLE_RANGE_VETO_MM=1200`, and the restored protocol: `APPROACH_STOP_MM=400`, `ALIGN_MAX_STEPS=10`, `ALIGN_MIN_DETECTIONS=3`. Arenas `DirectPole01`/`DirectPole02`. **The vision arm was NOT re-run** — it never touches the inverse, and `CONTROLLER_SEED = crc32("{POLE}|{START}")` excludes `SENSE_SOURCE`, so the pairing holds. The vision runs were copied to `PolicyRuns/Poles/` and verified byte-identical to the originals in `old_stuff` (410 steps, as before).

**Outcome: 10/10 `reached_aligned`. No collision, no corner jam, no timeout.** The headline result survives the model change.

| | published (1 m-capped) | **re-run (veto 1200)** |
|---|---|---|
| outcomes | 10/10 | **10/10** |
| steps, median (range) | 74 (12–154) | **39 (11–78)** |
| path, median | 10.16 m | **5.54 m** |
| pole calls / precision | 97 / 97.9% | 88 / **94.3%** |
| first detection, median | 798 mm | **1089 mm** |
| recall 800–1000 mm | 25% (n=16) | **67% (n=15)** |
| recall 1000–1400 mm | 0% (n=32) | **50% (n=26)** |
| recall beyond 1400 mm | 0% | 0% |
| true pole distance on arrival | 276–372 mm | 188–408 mm |
| min wall clearance | 212 mm | 220 mm |

**Precision came in at 94.3%, above the 88.2% the offline replay predicted** (Performance notes 2026-08-21 later). The replay scored the model at the *flown* poses; in closed loop the robot spends proportionally more of its time near the pole, where class is reliable, so the open-loop figure is a lower bound. Worth remembering the next time a replay is used to gate a run.

**The confirmation rule never rejected a detection** — zero unconfirmed stops in ten runs. It cost nothing here and was still the right guard to restore: it is what stands between an 88–94% precision model and a criterion that ends a trial on one ping.

#### Three numbers in the paper change

1. **The cost of sonar roughly halves.** Median 74 → 39 steps and 10.16 → 5.54 m, against vision's unchanged 34 steps / 2.64 m. **Pairwise, sonar is slower than its vision partner in 6 of 10 pairs, not 8 of 10.** The claim survives but weakens — and the paper's own explanation ("the gap is the sensory horizon, not feature quality") is now directly demonstrated, since moving the horizon out is exactly what closed most of it.
2. **"Zero bearing corrections in any run" no longer holds.** Three of ten needed one correction (P1/S4, P2/S3, P2/S4). Arguably a better result: the azimuth head is exercised rather than bypassed.
3. **Arrival distances spread wider: 188–408 mm true, against 276–372.** P1/S4 arrived at a true 188 mm on a perceived 331 — inside the 253 mm band where the inverse has no training data, which is the regime the Methods argue the 400 mm stop exists to avoid. Needs a sentence; it is a range-head error at close range, not a control failure.

#### Per-trial variance is large in both directions — quote aggregates only

P2/S4 went 108 → 11 steps; P1/S5 went 12 → 73. A different class call at step *n* puts the robot somewhere else entirely from step *n+1*, so trials are not paired across model versions even though the seeds match. Steps per pair, vision / old sonar / new sonar:

| pair | vision | sonar old | sonar new |
|---|---|---|---|
| P1/S1 | 29 | 85 | 32 |
| P1/S2 | 34 | 36 | 30 |
| P1/S3 | 26 | 63 | 28 |
| P1/S4 | 52 | 154 | 36 |
| P1/S5 | 13 | 12 | **73** |
| P2/S1 | 30 | 136 | 42 |
| P2/S2 | 50 | 40 | 46 |
| P2/S3 | 52 | 68 | 75 |
| P2/S4 | 91 | 108 | **11** |
| P2/S5 | 33 | 79 | 78 |

#### Data notes

- **P2/S5 was run with `START=5`**, so it finally draws `crc32("P2|5")` and matches its vision partner. All ten pairs now share a seed; the 2026-08-04 caveat is closed.
- **`final_true_pole_dist_mm` / `final_true_wall_clear_mm` are `null` in all ten summaries.** `run_robot` called `write_run_summary` without the referee values — the same drift that lost the terminal protocol, fixed in `c881d9f` but after these runs. **No data is lost**: `trajectory.tsv` carries `pole_near_mm` and `min_wall_mm` per row, and every number in this entry comes from there.
- ⚠️ **`Paper/images/scripts/exp1_stats.py` cannot read these runs.** Its `run_dir()` resolves `PolicyRuns/direct_…`, and both arms now live under `PolicyRuns/Poles/`. Nothing here was produced by it; the aggregation needs repointing before the paper numbers are regenerated.

### 2026-08-21 (later) — **Offline replay: the current inverse would collapse Exp 1's pole precision from 97.4% to 61.4%.** Do not re-run on the bare argmax — gate the pole call first

`EXPT_replay_exp1_inverse.py` (added this session) replays the **current** deploy inverse over the stored envelopes of the ten 2026-07-30 sonar runs and scores it per step against the model those runs actually flew. Ground truth is recomputed from each run's own `arena_features.npz` via `nearest_reflector_in_cone`, the same way `exp1_stats.py` does it. **761 of the 781 logged steps** carry an envelope — every run stores exactly two fewer dills than `trajectory.tsv` has rows, systematically, in all ten.

**This is per-step perception only.** The poses come from the flown runs; a different class call means a different action and a different pose from that step onward, so nothing here predicts trajectories, step counts or outcomes.

#### The result

| | pole calls | precision (raw truth) | false calls past 1.4 m |
|---|---|---|---|
| flown (1 m-capped model) | 77 | **97.4%** | 0 |
| current inverse, bare argmax | 171 | **61.4%** | **55** |

Detection improves exactly as predicted — recall at 800–1000 mm goes 25% → 87.5%, at 1000–1400 mm 0% → 40.6%, and first perception moves from ~800 mm out to 1.0–2.7 m — **but 66 of 171 pole calls are false, 55 of them beyond 1.4 m or on an empty cone.** Exp 1's controller turns toward perceived poles, so this is the number that matters, and it is a real risk to the 20/20.

**The terminal stop is safe either way**: zero steps in 761 would fire the 400 mm stop without a pole truly within 600 mm. The pole-range head is masked at 1 m and saturates near 740 mm, so a distant phantom cannot reach it. The exposure is wasted steps and wall contact, not a spurious `reached_aligned`.

#### The fix, measured rather than assumed

A confidence gate on the pole call (`p_pole >= tau`, rejected calls falling back to the runner-up over {wall, none} so the robot keeps wall-following rather than dropping into the scan path), optionally with the **class-agnostic range head vetoing** class calls beyond the range where class is trustworthy — the range head being the one output validated past 1.4 m:

⚠️ **The veto sits on top of the argmax, it does not replace it.** `tau = 0.5` is exactly the argmax here (every argmax-pole call in these runs scores p_pole ≥ 0.5), so the rows below isolate the veto. A "veto with tau = 0" would call pole on everything nearby — 263 calls at 31.9% precision.

| rule | calls | precision | 800–1000 recall | 1000–1400 recall | false past 1.4 m | median 1st detection |
|---|---|---|---|---|---|---|
| flown model | 77 | 97.4% | 25.0% | 0% | 0 | 798 mm |
| argmax, no veto | 171 | 61.4% | 87.5% | 40.6% | **55** | 1088 |
| argmax + veto 900 | 71 | 98.6% | 6.2% | 0% | 0 | 774 |
| argmax + veto 1000 | 85 | 95.3% | 43.8% | 6.2% | 0 | 851 |
| **argmax + veto 1200** | 102 | **88.2%** | **81.2%** | 15.6% | **2** | **995** |
| argmax + veto 1400 | 119 | 81.5% | 87.5% | 31.2% | 12 | 1053 |
| tau 0.70, no veto | 75 | 98.7% | 31.2% | 3.1% | 0 | — |

**1200 mm is the knee, and it is the proposal.** It recovers essentially the whole 800–1000 mm gain (25% → 81.2%, against the argmax's 87.5%), moves median first detection from 798 to 995 mm, and costs 2 false calls past 1.4 m where the ungated model costs 55. Going out to 1400 buys 6 more points of recall for **six times** the far phantoms.

**And the false calls are safe ones.** At veto 1200 all 12 false calls sit at ≥800 mm true range (1 at 800–1000, 9 at 1000–1400, 2 beyond) — **none below 800 mm**. So a false pole always leaves the robot ≥5 steps of runway, and class is ~100% reliable under 800 mm, so the call corrects itself well before contact. That is the self-correcting approach behaviour, bounded.

**`tau` alone is a trap**: 0.70 restores precision by discarding exactly the long-range detections that motivate the re-run — its first-detection ranges land where the flown model already was. The veto is the better instrument because it targets the actual defect: the model is fine up close and unreliable far away, and the agnostic range head is the one output that still knows which regime it is in.

**Cost to the paper if 1200 is adopted:** pole precision as published drops 97.9% → 88.2%, and first perception moves 798 → 995 mm, so the "cost of sonar" paragraph and `tab:direct-results` need new numbers. Both are Results updates, not structural.

**The gate is confined to Experiment 1.** It lives in `feature_from_inverse`, which only `SCRIPT_RunDirectPolicy` uses; `SCRIPT_RunPolicy` hands `encode_obs` the raw prediction dict. Nothing here touches the Path04/Path07 artifacts.

#### Next

Land the gate in `feature_from_inverse` with the chosen (tau, cap) as named constants, re-run the **ten sonar runs only** holding `APPROACH_STOP_MM=400` / `ALIGN_MIN_DETECTIONS=3` (HEAD carries 500 / 6), and leave the vision arm alone — `CONTROLLER_SEED = crc32("{POLE}|{START}")` excludes `SENSE_SOURCE`, so pairing survives. Running P2/S5 with `START=5` repairs the one pair whose seeds never matched.

### 2026-08-21 — **Path07 replicate: 42 mm against 42 mm.** The noise floor is under the whole manipulation series, and the corrected moved-poles run reproduces the near-null

Two more 500-step runs on `PolicyTraining/default_Path07/best_policy_survival.json`, `POLICY_INPUT_SOURCE="live"`, no collision in either. `PolicyRuns/default_Path07_run02` (baseline replicate) and `default_Path07_run02_moved_poles` (the 08-20 moved condition re-run with the P2 oversight corrected). This closes the replicate item the 2026-08-20 and 2026-08-19 entries both flagged as blocking.

#### The replicate

| | baseline 08-19 (`run01`) | **baseline replicate 08-21 (`run02`)** |
|---|---|---|
| cross-track median | 42 | **42** |
| p90 / max | 114 / 248 | 96 / 183 |
| curl | −0.54 °/step | −1.13 |
| drive scale | 1.014 | 1.018 |

**Zero difference on the median**, and window medians agree to ≤15 mm (north lobe dx +10/+10, dy +6/+21; east+south dx −32/−22, dy +9/+5; rest of lap dx +2/−1, dy +1/+8). Path07 now carries the same licence Path04's ±1 mm replicate carried: **effect sizes down to ~30 mm on a window median are measurable on this arena**, and every effect in the 2026-08-20 entry clears that by 4–20×.

#### The corrected moved-poles run — bigger displacements, same near-null

Full tan-blob inventory of the fresh `arena.png` (all blobs >20 cm² inside the arena, >200 mm from the stored walls): **P2 is back at its trained position** (+1426,−196 against the baseline's +1423,−195 in image-centroid terms), P3 and P4 untouched, and the two moved poles went further and in different directions than on 08-20:

| | 08-20 (`run01_moved_poles`) | **08-21 (`run02_moved_poles`)** |
|---|---|---|
| P0 | ~215 mm SSW | **~800 mm ESE** |
| P1 | ~545 mm N | **~400 mm SW** |
| P2 | ~225 mm SW (oversight, carried into the removal run) | **at trained position** |

Result: median **45 mm**, north-lobe dy **+43** with per-lap values +94, +43, +47, +39, +32, +13, +30 — seven laps, all positive, ~+25–30 mm above the two baselines. **08-20's moved run gave +34 mm on the same measure**, from displacements roughly half the size and pointing elsewhere, at +3.01 °/step curl against this run's +1.39. The near-null replicates.

#### The ladder, with a floor under it

| manipulation | magnitude | route response | vs noise floor |
|---|---|---|---|
| pole displacement (two independent configs) | 0.4–0.8 m | **~30 mm** | ~2× |
| pole removal (P0+P1) | — | **~110 mm** (dx +56, dy +96 over baseline, six laps) | ~7× |
| wall displacement | 420–500 mm inward | **416 mm**, near-complete capture | ~28× |

**For thin dowels, presence matters and position barely does. For walls, position is nearly everything.** Note this reverses the Path04 block-displacement reading (~1/3 capture) in the pole case and exceeds it in the wall case — the operative variable is the reflector, not the manipulation type. Do not quote "displacement beats removal" or the reverse without saying what was displaced.

#### Why, from the recorded sonar — the robot-side corroboration of the channel ablation

`d_live − d_clean` per slice, where `clean` is the true geometry of the **trained** arena evaluated at the robot's actual pose. In a baseline this is inverse error; under a manipulation it is inverse error *plus* the manipulation, and it is independent of where the robot ended up.

| window | base01 | base02 | moved02 | removed | walls |
|---|---|---|---|---|---|
| north lobe .00–.25 | −88/−58/−92 | −95/−26/−67 | −131/−22/−63 | −94/−49/−52 | −269/−46/−49 |
| east+south .25–.45 | +4/−66/−3 | +9/−54/+3 | +35/−75/−23 | +30/−62/−7 | **−419/−441/−434** |

**The wall moved the sonar picture by exactly what it moved physically** — ~430 mm shorter on all three slices across the affected window. **The pole manipulations produce no comparable window-level signature** (≤20 mm from baseline). A 25 mm dowel at ~500 mm with a wall behind it barely perturbs the three-slice profile — which is the channel `SCRIPT_Ablations.py` identified as carrying the control, until now only in simulation.

⚠️ **The removal effect is real but its mechanism is NOT visible here.** Six laps, never negative, localised to the window where the removed poles are in-beam — but the window-median slice deltas are at baseline, so the effect must live in a minority of steps. **Do not write it up as "the pole dominated the slice reading."** Per-5%-bin deltas swing ±200–400 mm in both directions and are too noisy to attribute; settling the mechanism would need per-step attribution against the in-beam windows, which has not been done.

#### Status

**Experiment 2's robot work is complete**: two arenas, a replicate on each, pole removal, pole displacement ×2, wall displacement. What remains for Exp 2 is the write-up, which does not exist yet — no subsection, no figures. The thread now moves to Experiment 1.

### 2026-08-20 — **Three Path07 manipulations: removal shifts the route locally, an added wall on the path makes the robot re-route around it, displacement does almost nothing.** And the route survives a 3.5 °/step swing in motor bias

Four runs on one artifact, `PolicyTraining/default_Path07/best_policy_survival.json`, `POLICY_INPUT_SOURCE="live"`, commit at run `ec8ff25`. **500 steps each, no collision in any of them.** The 08-19 baseline is the entry below; the three manipulations are `PolicyRuns/default_Path07_run01_{moved_poles,removed_poles,added_walls}`.

#### What was actually in the arena, measured rather than assumed

`SCRIPT_RunPolicy.py` still copies the stored arena over the fresh capture, so `arena_features.npz` is byte-identical in all four runs and is **not** a record of what was present. The fresh `arena.png` is, and the dowels are recoverable from it directly (method note below).

| | P0 (+110,+736) | P1 (−1020,+404) | P2 (+1236,−128) | P3 (+496,−825) | P4 (−473,−1640) |
|---|---|---|---|---|---|
| moved_poles | ~215 mm SSW | ~545 mm N | ~225 mm SW | untouched | untouched |
| removed_poles | **removed** | **removed** | ⚠️ *still at its displaced position* | untouched | untouched |
| added_walls | back in place (±15 mm) | back in place | back in place | untouched | untouched |

⚠️ **P2 was left displaced through the removed-poles run — an oversight, confirmed by Dieter, not a design choice.** It cuts the right way for the contrast (`moved` and `removed` share the identical P2 placement, so their difference isolates P0+P1) but that run is "P0+P1 absent, P2 displaced", and should never be quoted as a clean removal.

`added_walls` places one wall along **y ≈ 0.051·x − 2626, from x ≈ −850 to +1700** (weighted line fit to the changed region, residual rms 35 mm). The five poles are all back at their trained positions, so the wall is the only change. **Read it as a wall DISPLACEMENT, not an addition** (Dieter's intent, and the geometry agrees): measured against the trained south boundary column by column, it brings the south wall inward by **+450 (x −300…0), +416/+440 (0…+600), +464/+489 (+600…+1200), +504 mm (+1200…+1500)**, tapering at both ends — i.e. ~420–500 mm over exactly the stretch the path's south arc runs along. The original wall is still physically behind it, so acoustically this is occlusion-displacement: from anywhere on the path the new wall is what returns the echo.

#### Tracking

| | baseline 08-19 | moved poles | removed P0+P1 | added S wall |
|---|---|---|---|---|
| cross-track median | 42 | **42** | 57 | 83 |
| p90 / max | 114 / 248 | 120 / 191 | 153 / 288 | 224 / 496 |
| curl (°/step) | −0.54 | +3.01 | +2.62 | +2.52 |
| drive scale | 1.014 | 1.004 | 1.008 | 1.007 |

#### Where the deviation sits — every effect is local and replicates lap on lap

World-frame median offset of the trajectory from the target path (dx, dy in mm), on one metric (5 mm densified centreline, nearest point) so the four compare to each other:

| window | baseline | moved | removed | walls |
|---|---|---|---|---|
| north lobe (phase 0.00–0.25) | +10, +6 | +12, +34 | **+66, +102** | +21, +39 |
| south arc (phase 0.30–0.45) | −17, +18 | +0, +20 | −6, +0 | **−219, +75** |
| rest of lap (0.45–1.00) | +2, +1 | +1, −1 | −9, +12 | −9, +21 |

**Removal.** Per-lap dx over the north lobe: **+51, +82, +69, +47, +67, +65**; dy **+79, +117, +125, +94, +103, +107** — six laps, never negative, against a baseline that scatters (dx +9, +29, +20, +24, −85, −3, +16). Cross-track in that window 134 mm vs 48 baseline, and **at baseline everywhere else in the lap** (19 vs 37 in the south arc, 54 vs 39 in the rest). The window is exactly at and just downstream of where the removed poles are in the ±35° cone (P1 in-beam at phase 0.90–1.00, P0 at 0.95–0.10) — the same "cost surfaces downstream of the missing object" signature as the 2026-08-14 Path04 removals.

**Displacement is a near-null, and that is the interesting part.** Moving P0/P1/P2 by 215–545 mm shifted the route by ~+28 mm north in the north lobe (per-lap dy +44, +5, +72, +31, +14, +49 — consistently positive but tiny) and left median tracking at 42 mm, identical to the baseline. **Removing the same two poles moved the route four times as far.** This is the reverse of the Path04 block displacement (a third of the cue displacement captured, all nine laps) and the obvious difference is what was manipulated: 25 mm dowels 450–530 mm off the path here, against large blocks there. Do not generalise "displacement > removal" from Path04 without saying which reflector.

**The displaced wall sits *on* the trained path, and the route follows it inward — near-complete cue capture.** The south boundary moved north by ~420–500 mm and the route's southernmost point moved north by **416 mm**: capture of roughly **0.85–1.0**, against the ~1/3 the Path04 blocks produced. Extended boundary reflectors dominate the three-slice wall profile, which is the channel the simulated ablations say carries the control. ⚠️ **The capture fraction may be saturated rather than genuinely near-unity**: partial capture would have driven the robot into the wall, so this manipulation cannot distinguish 0.85 from 1.0. A smaller inward move (~200 mm, leaving the path flyable) would give a graded number comparable to the blocks'. What it does settle is that the avoidance is **cue-driven, not constraint-driven** — there is no reactive obstacle avoidance anywhere in the controller and the inverse is blind below ~400 mm, so nothing but the changed percept kept the robot off the wall. The path centreline passes **43 mm** from the fitted wall line at (+921, −2536) — i.e. −42 mm of usable margin after the 85 mm robot radius, so the trained route is physically blocked. The robot never came closer than **510 mm**, spent **zero** steps within 300 mm of it, and its southernmost point was **−2091 mm against −2507 in the baseline (416 mm further north)**. Per-lap dx over the south arc: −196, −219, −218, −315, −163, −221, −184 — seven laps, all of them. There is no explicit obstacle avoidance anywhere in the controller; the re-route is what the changed sonar does to the state estimate.

#### The route survives a 3.5 °/step swing in motor bias — the strongest evidence that sensing corrects the integrator

The policy's only non-sonar input is `prev_rot`, the **commanded** rotation (`SCRIPT_RunPolicy.py` step 6, and the trainer matches). It never sees executed rotation — no wheel odometry, no IMU — so its dead-reckoning is blind by construction to curl and gain error. Across these four sessions curl ran from **−0.54 to +3.01 °/step**, roughly 237° of unaccounted heading per 79-step lap at the high end, and the route held at an identical 42 mm median. A controller running open-loop on integrated motor commands cannot do that.

⚠️ **Read the other way, this is also the day's main confound**: all three manipulations ran at +2.5…+3.0 curl against a baseline at −0.54. The moved-poles run rescues the comparison — same +3.0 curl, baseline tracking (42 mm overall, 49 vs 48 in the north lobe, 47 vs 39 in the rest of the lap) — so curl does not produce the removal or wall effects. That is an argument, not a noise floor. **The Path07 replicate is still missing and is still the next run.**

#### What the data does *not* show: path integration carrying the route

Worth recording so nobody re-derives it. At a 1400 mm horizon the robot is sensorily deprived almost never: **84–87% of steps have a reflector in the cone** (true geometry, ±35° slices; live sonar agrees at 82–90%), blind stretches run a **median of 3 steps and a maximum of 6–8** (≤1.2 m of travel), and cross-track does not grow across them (baseline median −11 mm end-to-end, −20 mm in the four steps after). There is no stretch long enough for the integrator to do visible work, so "it kept going with a pole gone" does not separate path integration from "the walls were still there". `SCRIPT_Ablations.py` carries the matching negative in its own docstring: absent a sustained disturbance a blind policy dead-reckons the loop in simulation.

**Decision (Dieter, 2026-08-20): no blind-policy run.** The paper's claims sit on the sensory side, and none of them assert that path integration alone can navigate. The supported statement is the one above — the policy carries a motor-efference integrator that sonar demonstrably corrects — which needs no PI-alone condition. `BLIND = True` in `SCRIPT_TrainPolicy.py` remains available (one line, automatic `_blind` artifact suffix) if that ever changes.

#### Method note — recovering the arena from `arena.png` works, plain differencing does not

The 2026-08-17 recipe (greyscale difference against a reference run, threshold, connected components) **fails across days**: the 08-19 → 08-20 lighting change dominates the difference image, and the largest blobs are all wall and floor illumination outside the arena. What does work, and is what produced the table above:

- **Dowels by colour**, per run, no reference image needed: `R−B > 35` and `R > 95` on `arena.png`, masked to more than 250 mm from the stored wall points, largest blob within 400 mm of the trained pole position. Present/absent is unambiguous; centroid differences between runs are good to ~15 mm because the parallax smear is identical across runs (the dowel appears as a streak, not a dot — compare centroids run-to-run, never centroid to true position).
- **Wall changes by differencing**, but restricted to a band around the region of interest and with a size threshold, then a weighted line fit through the blob centroids.

Both are ~40 lines. **Still worth landing as `SCRIPT_CheckArenaDelta.py`** and running right after placing an obstacle, before committing to a 500-step run — it would have caught the P2 oversight in seconds.

### 2026-08-19 — **Path07 deployed: 500 clean steps first attempt.** And the simulator is systematically pessimistic

`PolicyRuns/default_Path07_run01`, 500 steps, `POLICY_INPUT_SOURCE="live"`, artifact `PolicyTraining/default_Path07/best_policy_survival.json` (epoch 1600). Commit at run `70edfa1`. **No collision, 6.44 laps, no crash log.**

| | **Path07 run01** | Path04 baseline | Path04 replicate |
|---|---|---|---|
| cross-track median | **42 mm** | 36 | 37 |
| p90 | **114** | 89 | 94 |
| max | **248** | 188 | 185 |
| curl | −0.51 °/step | +0.51 | +1.78 |
| drive scale | 1.0159 | 1.0075 | 1.0017 |

**Slightly worse than Path04 but only slightly** (+5 mm median, +22 p90), and calibration held — compare the −4.69 °/step that wrecked `default_Path06_run01`. **No degradation across laps**: per-lap medians 61, 39, 35, 50, 74, 31, 21, worst in lap 5 and best in the last. Nothing accumulating.

#### ⚠️ The simulator understates real tracking on BOTH paths — the disturbance model is too harsh

Sim predicted 67.5% survival over 300 steps (~52% over 500); the robot did 500 clean first try. n=1, so that alone proves nothing. But cross-track shows a systematic offset:

| | sim median | robot median | ratio |
|---|---|---|---|
| Path04 | 62 mm | 36.5 mm | **1.70×** |
| Path07 | 60 mm | 42 mm | **1.43×** |

**Likely mechanism, and it is checkable:** training and evaluation draw `motion_rot_bias_deg ~ U(−5, +5)` per episode, so a typical simulated episode carries ~2.5 °/step of curl. A *well-calibrated* robot runs at 0.1–0.5 (−0.12 on Path06 run02, −0.51 here). The ±5 was set on 2026-08-13 from sessions reading −3.6 and −4.09 — which we now know were **badly calibrated** sessions, exactly the fault `7596853` guards against. **If ±5 is wrong, every sim survival figure quoted for Path06 and Path07 is too low, and the "Path07 fails 2.7× more per lap" conclusion is largely an artifact.** Worth re-deriving `motion_rot_bias_deg` from curl measured in verified-calibrated sessions only, before concluding anything more about route geometry.

#### Selection criterion: survival vs tracking — they measure different things

Prompted by Dieter's question. Note first that **`default_Path04` was val-selected, not tracking-selected** — every policy before 2026-08-18 was. On three artifacts evaluated on identical fresh seeds:

| artifact | survival | median xt | p90 | max |
|---|---|---|---|---|
| default_Path07 survival-sel | **67.5%** | 58–60 | 175–179 | **592–637** |
| seedB_Path07 survival-sel | 60.8% | 59–60 | 174–189 | 835–866 |
| default_Path07 val-sel | 53.4% | 60–65 | 178–206 | **869–1014** |

**Median tracking cannot separate them — 58–65 mm across all three.** What differs is the *tail*: worst excursion 592 → 1014 mm, and survival tracks it. Selecting on median tracking would have been near a coin flip between the best and worst artifact. The reason is that all of these follow the path competently most of the time; what differs is how often they reach an unrecoverable state, a rare-event property invisible in a median over ~18,000 steps.

**But tracking is the criterion the landmark work needs**: the displacement result rested on a ±1 mm replicate *of the median*, and resolving a +23 mm pole effect needs median stability, not survival. **The two are not substitutes — select on survival to get a policy that finishes runs, then confirm median stability with a replicate before measuring anything.** For Path07 the survival-selected artifact happens to win on both. Suggested trainer improvement: log p90 and max alongside the median at each survival evaluation and use p90 as a tie-break.

#### Next

**A replicate run is the priority** — Path04's ±1 mm replicate is what licensed every effect size in the landmark series, and Path07 has no noise floor yet. It also starts separating "the sim is pessimistic" from "we got lucky once".

### 2026-08-18 (night) — Second training seed on Path07: **no better**. The ~65% level looks like a property of the path, not of training luck

Ran the identical recipe with `seed = 7` instead of 42 (`CONDITION = "seedB"`, output `PolicyTraining/seedB_Path07/`), everything else byte-identical. Re-measured both winners on seeds 101/202, which neither was selected against:

| artifact | selection score | **fresh seeds** | max off-path |
|---|---|---|---|
| `default_Path07` survival-sel (ep 1600) | 76.7% | **67.5%** (66.7 / 68.3) | 592–637 mm |
| `seedB_Path07` survival-sel (ep 1000) | 63.3% | **60.8%** (60.0 / 61.7) | 835–866 mm |
| `default_Path07` val-sel | — | 53.4% | 869–1014 mm |

**6.7 points apart, ~1.1 SE — the two seeds are not meaningfully different.** `default_Path07` wins on the point estimate and on worst-case excursion, so it is the one to fly.

**The "two short runs beat one long run" idea is NOT supported.** Two independent seeds landing within noise of each other points the other way: **~65% looks like a ceiling set by Path07's perception, not by training luck** — consistent with the drift-detectability analysis (47.5% at 200 mm against Path04's 66%). If that holds, neither more seeds nor more epochs will move it much, and the way to do better is a route with more informative geometry.

⚠️ **Two process lessons, both mine.**

1. **Do not compare runs at a matched epoch.** At epoch 1000 seedB read 63.3% against run A's 31.7% and I called it a real 3.8-SE difference. It was a transient: seedB then went 55 → 55 → 43 → 25 while run A carried on to its 76.7 peak at epoch 1600. **The only fair comparison is each run's best, re-measured on fresh seeds.** The trace wanders far too much for any single epoch to mean anything.
2. **seedB died silently at epoch 1416 and went unnoticed for ~12 hours.** No traceback; epoch times ballooned 10s → 46s → 179s beforehand, the signature of an OOM kill or a suspend. The monitor was watching for `Traceback`/`MemoryError`/`Killed` and a silent kill produces none of them. **A watchdog on a long run must check the process is ALIVE, not just that it has not printed an error.**

### 2026-08-18 (evening) — Path07 policy trained. Survival selection is worth +14 points and **replicates**; Path07 fails 2.7× more often per lap than Path04

`PolicyTraining/default_Path07`, 2000 epochs, standard recipe (1000 episodes, hidden 32, rot_bias ±5, `use_agn=True`), first run with survival-based selection (`8e06001`). Path07 lap ≈ 79 steps.

**Head-to-head, re-measured on seeds the checkpoints were NOT selected against (n=60 × 2, 300-step cap):**

| artifact | seed 101 | seed 202 | mean | max off-path |
|---|---|---|---|---|
| **`best_policy_survival.json`** | 66.7% | 68.3% | **67.5%** | 637 / 592 mm |
| `best_policy.json` (val-selected) | 60.0% | 46.7% | **53.4%** | 1014 / 869 mm |
| Path04 `best_policy.json`, same test | 85.0% | 83.3% | **84.2%** | — |

**Survival selection: +14 points, on top of +15 on Path06. Two paths, so treat it as established.** It also yields a *steadier* policy — consistent across seeds (66.7/68.3 vs 60.0/46.7) and much smaller worst excursions.

⚠️ **The selection score overstates.** The winner scored **76.7%** at selection and **67.5%** on fresh seeds — a ~9-point winner's curse, about what taking the max of 19 draws at SE 6.4 predicts. `9079a5a` makes the trainer re-measure its winner on seeds 101/202 automatically. **Quote the fresh number, never the selection score.**

⚠️ **Do not compare raw survival across paths — normalise per lap.** A 300-step cap is 5.1 laps on Path04 but only 3.8 on Path07, so it is an *easier* test for Path07 and the raw gap understates the difference:

| | survival | laps | failure rate | half-life |
|---|---|---|---|---|
| Path04 | 84.2% | 5.1 | 0.038 /lap | 18.4 laps |
| **Path07 (survival-sel)** | 67.5% | 3.8 | **0.103 /lap** | **6.7 laps** |
| Path07 (val-sel) | 53.4% | 3.8 | 0.165 /lap | 4.2 laps |

**Path07 fails 2.7× more often per lap than Path04.** For a 500-step robot run (6.3 laps on Path07) that predicts roughly a 52% chance of finishing clean, against 82% for Path04 over its equivalent. Good enough to learn from on the robot; **not yet good enough for a manipulation series**, which needs 500 clean steps in every condition — that is what made the Path04 displacement result measurable.

**Does more training help? Unresolved, and the trace says why.** Fitting the 19-point survival trace: whole run **+27.1 pts/1000 epochs (t=6.05)**, epoch≥600 **+18.3 (t=2.52)**, epoch≥1000 +19.3 (t=1.36), epoch≥1300 +6.0 (t=0.20). No overfitting — train/val gap +9.6 and flat, val still falling at −8/1000. So the tail is **underpowered, not converged**. But the dominant feature is variance, not trend: consecutive evaluations 100 epochs apart run 40 → 40 → **76.7** → 53.3 → 61.7 → 36.7. **The optimiser wanders through good and bad regions and survival selection mostly harvests that, rather than the policy steadily improving.** Cheaper than more epochs: evaluate more often (more draws from the same training), or run a second seed and keep the better artifact.

**Contrast with Path06, which plateaued at ~45% by epoch 250 and never moved.** Path07 climbed the whole way. So "has the curve flattened" is path-specific and cannot be assumed.

### 2026-08-18 (late) — Calibration silently reverted; the crash mechanism found; checkpoint selection moved onto survival; Path07 drawn

Follows the entry below, which should be read first. Commits `7596853` (deploy guards), `8e06001` (survival selection).

#### ⚠️ `default_Path06_run01` flew with NO calibration applied. Check `Settings.py` before every deploy

`SCRIPT_CalibrateRobot.py` **resets `Library/Settings.py` to identity before measuring** — it has to observe raw firmware behaviour rather than residuals on top of an existing correction — and writes the measured values back only if you answer `y` to a final prompt. It also **skips that write silently** when the rotation table trips a CRITICAL flag. Either path leaves a good calibration JSON on disk beside an identity `Settings.py`, and `Client.step` reads `Settings.py`, not the JSON.

That is what happened. The JSON written at 12:04 held `drive_yaw_curl_deg_per_mm = -0.02927`; `Settings.py` held `0.0`. **−0.02927 × 150 mm = −4.39 °/step predicted, −4.69 measured.** The whole disturbance, accounted for. The 12:04 rotation table was also wild (commanded −30 → −28.8, −38.3, −46.6, −28.7, −40.7 across five repeats), which is very likely what tripped the CRITICAL flag.

**Fixed in `7596853`: `SCRIPT_RunPolicy.py` now refuses to start on identity constants** (`CHECK_CALIBRATION`). There is no case where flying uncalibrated is wanted.

After a clean recalibration (`-0.03013 °/mm`, scale `0.9897`), **`default_Path06_run02` tracked at a 50 mm median / 107 p90 over 58 steps — better than Path04's own baseline (63 / 108)** with curl at −0.12 °/step. The calibration was the whole story for run01. ⚠️ Note the robot's curl is now **2.4× the −0.01244 in force for all the Path04 runs** — something changed mechanically; it is compensated, but do not assume the rig is the one that produced the landmark results.

#### The run02 crash: the inverse is blind below 400 mm

run02 held the path for 58 steps, then went 91 → 200 → 313 → 354 → 399 → 463 mm over seven steps and grazed the block at (716, −1879) with 88 mm clearance. Live minus true, all slices pooled, from that run:

| true range | n | median error | \|err\| > 300 mm |
|---|---|---|---|
| 0–400 mm | 3 | **+573 mm** | **100%** |
| 400–700 | 4 | +137 | 25% |
| 700–1200 | 62 | **−10** | 3% |
| 1200+ | 128 | −119 | 27% |

The left slice over the final seven steps: true `840 → 687 → 595 → 458 → 351 → 218 → 152`; model `1038 → 928 → 945 → 491 → 924 → 750 → **1384**`. At contact it reported 1384 mm for something 152 mm away. Consistent with the training set having nothing below 253 mm. **The policy has no sonar-based collision avoidance inside ~400 mm — this is a hard constraint on how much clearance any route needs, not a tuning parameter.**

A second guard (`CROSS_TRACK_WARN_MM`, default *warn*) fires on 300 mm sustained 2 steps: replayed, it triggers at run02 step 63 (two before contact) and never on either clean Path04 baseline (max 185/188 mm). ⚠️ **Deliberately not an abort** — it also fires at step 35 of the B1+pole removal and step 32 of the B1+B2 displacement, runs whose full 500 steps were wanted. **It would not have prevented run02**: the robot hit something 463 mm off-path, where no launch-time rule reaches. It buys warning, not avoidance.

#### Checkpoint selection moved onto closed-loop survival (`8e06001`)

The single largest lever found all day, and it is free. See the commit and the entry below for the evidence. **Deploy `best_policy_survival.json`, not `best_policy.json`.**

**This does NOT invalidate the Path04 landmark series.** All five runs used one fixed artifact, so the manipulations are within-policy comparisons and are unaffected by how it was chosen; the ±1 mm replicate establishes the measurement floor independently. What changes is only the reading of its 82.5% survival — one arbitrary draw, not the recipe's ceiling. **Corollary: do not retrain a Path04 policy without redoing the whole series**, since runs from a different artifact cannot be pooled with the existing five.

#### Where perception fails, correctly this time

Path06's failure localised to a **dead zone at arc 42–60%** — 18% of the lap at 10% drift detectability against a 49% path mean — and run02 left the path at arc 48–58%, its floor. In simulation 29.3% of divergence onsets landed in that 17.8% of path (1.65×, ~1.9 se), with local detectability 41.5% at onset against a 51.2% median.

**It is not the absence of landmarks — objects are in cone on 75–84% of that zone. It is their distance: ~1500 mm against a 959 mm path median**, past the noise cliff where the model's 236–401 mm error exceeds the ≤200 mm a lateral drift can produce. Nor is it the crossing as a feature: the *same physical point* scores 9.8% on the arc-54% approach and 37.1% on the arc-98% approach, 60° apart. It is the approach *leg*, which runs through the arena's most open ground — and that is structural to a centred figure-of-eight.

**A general design rule fell out, and it is stronger than the 2026-08-08 note (which said the clearance/perception trade was weak). At the tight end it is *inverted*:**

| clearance band | share of lap | local detectability |
|---|---|---|
| tight (<460 mm) | 7.4% | **2.4%** |
| mid (460–600) | 70.5% | 57.5% |
| open (>600 mm) | 22.1% | **62.7%** |

Squeezing past something puts it **abeam**, outside the ±35° forward cone: it consumes the whole safety margin and contributes nothing. **Tight is not informative. Route for what is AHEAD at under a metre, and stop paying clearance for perception you do not get.**

#### Path07, and a caution about over-constraining

Furniture moved (block to (−329, −361), poles to (496, −825) and (−473, −1640)) and the path redrawn. Current: min clearance **436 mm**, drift 47.5 / 70.6 / 24.8%, ahead median 974 mm. An earlier 43-waypoint version scored 57.2 / 72.7 / 34.9% at 390 mm — **that edit bought +46 mm of clearance for ~10 points of detectability, the same bad trade as Path06 v1→v2.** Worth reverting toward, fixing only the one free tight spot at (−393, −980) where pushing left gains the full 100 mm per 100 mm and local detectability is only 12%.

⚠️ **Agent note, recorded so it is not repeated.** I spent a long stretch searching for an "optimal" path, using a `CLEAR_MIN` of 460–520 mm, and reported the arena as admitting no feasible figure-of-eight. **`SCRIPT_DefinePath.py` sets `MIN_CLEARANCE_MM = 400.0` and calls it the floor.** Path07 at 436 mm already clears the project's own standard; the infeasibility was self-inflicted. Dieter's correction was right, and the general lesson is that geometry analysis has had poor predictive value here — Path06 looked acceptable on paper and survived 27%. **Only closed-loop survival and robot runs have told us anything. Prefer training and testing over another round of route optimisation.**

### 2026-08-18 — Path06 (figure-of-eight) trained and deployed. **`val_mse` does not predict closed-loop behaviour, and never did** — plus a mechanism I asserted twice and got wrong twice

Second arena, `TargetArenas/Path06`: a figure-of-eight with a genuine self-crossing, 5 poles + 1 block, 38 waypoints, 10.6 m, ~71 steps/lap. Policy `PolicyTraining/default_Path06`, same recipe as Path04 (2000 epochs, 1000 episodes, hidden 32, rot_bias ±5, `use_agn=True`). Teacher fix at commit `2169831`.

#### ⚠️ THE FINDING THAT QUALIFIES EVERY OTHER NUMBER IN THIS FILE

**`val_mse` is not a usable selection criterion or quality measure.** Closed-loop evaluation in the simulator (300 steps, n=30–40, survival = reached the cap without a blocked drive):

| | Path04 | Path06 |
|---|---|---|
| val_mse | 118.9 | 131.0 |
| **survived 300 steps** | **82.5%** | **27.5%** |
| cross-track median / p90 / max | 62 / 178 / 719 mm | 76 / 286 / 1028 mm |
| laps completed (median) | 5.13 | 2.29 |

Near-identical loss, threefold difference in survival. Worse, *within* Path06 across 11 checkpoints spanning 2500 extra epochs, val ranged 125–136 while survival ranged 20–53% **with no relationship between them** — the lowest-val checkpoint (+500, val 130.4) was the worst survivor at 20%, and the best survivors had among the highest losses. The cause is standard behavioural cloning: per-step imitation MSE is dominated by the large curvature signal, the small corrections that hold the path barely register, and errors compound over 300 steps.

**Consequence: `best_policy.json` is selected on val, so every policy this project has shipped was selected close to arbitrarily — `default_Path04` included, whose 82.5% may not be the best that run produced.** Selection needs a closed-loop survival eval. At n=30 the standard error is ~9 points, which is why the table below is so noisy; use n≥100.

#### The Path06 failure is perception, not the route, the crossing, or the motors

2×2, same 300-step survival metric, n=30:

| | Path04 | Path06 |
|---|---|---|
| as trained (motion + sensor noise) | 87% | **27%** |
| motion noise OFF, sensor noisy | 100% | **50%** |
| motion noise ON, sensor clean | 97% | 27% |
| both OFF (pure geometry) | 100% | **100%** |

With no noise the route is driven perfectly, so the figure-eight is learnable and the policy learned it. With the trained sensor and *zero* motor disturbance it still fails half the time; that 50-point gap is sensing. Caveat: `get_clean_measurement` is out of distribution for a policy trained on error-model output (Path06's median cross-track is actually *worse* clean, 85 vs 51 mm), so read the two noisy-sensor rows as the real comparison and the clean rows only as proof the route is drivable.

**Where it dies: median cross-track at the moment of collision is 463 mm.** It gets lost, it does not clip things — so the 375 mm clearance squeeze at 17–21% of the lap is *not* implicated. Failures spread across the whole lap (9 in the first decile, 12 across 20–40%), the signature of accumulating error.

**The self-crossing is solved.** Of 29 failures, **zero** within 600 mm of the crossing. See `2169831`: the pure-pursuit teacher projected statelessly and picked branches on numerical noise (19/55 wrong before, 0/145 after).

#### More training: worth ~15 points, then flat

Continued from the epoch-1974 checkpoint for 2500 more epochs, measuring survival every 250:

```
 epoch    val  survive        epoch    val  survive
   +0   133.9    30.0%        +1250  125.4    46.7%
 +250   135.5    46.7%        +1500  130.0    43.3%
 +500   130.4    20.0%        +1750  125.3    46.7%
 +750   134.8    46.7%        +2000  128.6    36.7%
+1000   128.3    43.3%        +2250  127.1    53.3%
                              +2500  127.1    36.7%
```

All of the gain arrives by +250; the next 2250 epochs move val from 133.9 to 127.1 and survival not at all. The training curve had *not* flattened (Dieter's observation, and correct) — but Path04's curve has the same shape and reaches 82.5%, so "still descending" does not distinguish them. Ten checkpoints kept as `continued_ep*.json`.

#### ⚠️ Mechanism: it is RANGE, not reflector type. Two wrong claims retracted

**RETRACTED (1): "±5° rot_bias is too much for this path."** Sweeping rot_bias 0/1/3/5° gave flat survival on both paths (Path06: 27.5 / 35.0 / 45.0 / 27.5%; Path04: 80.0 / 77.5 / 75.0 / 82.5%). Motor disturbance is not the cause.

**RETRACTED (2): "a 25 mm dowel is a point reflector and therefore a poor drift detector; swap dowels for blocks."** Controlling for range this is false, and in one band backwards. Detection of a 200 mm lateral drift:

```
Path06         %steps   detect  |  wall   pole        Path04    detect | wall  pole
   0- 700 mm    14.9%    84.2%  |  100%    71%          12.9%    85.2% |  87%    0%
 700-1000 mm    38.4%    73.5%  |   74%    72%          39.5%    80.0% |  79%   86%
1000-1400 mm    33.7%    24.4%  |   14%    34%          39.0%    47.2% |  48%    0%
1400+    mm     13.1%     4.5%  |    0%     5%           8.6%    22.2% |  22%    -
```

Within a band poles and walls are comparable, and at 1000–1400 mm the **poles are more than twice as good as the walls** (34% vs 14%). Worked example: facing a pole at 1073 mm a 200 mm shift changed the slices by 295/327 mm (detected); facing a wall at 1180 mm the same shift changed them by 73–80 mm (invisible, against 236 mm noise). Where a distant pole *is* detected it is by the discrete mechanism — its range changes only ~16 mm at 1.2 m, but its bearing swings ~9.5° against 23.3°-wide slices, so it jumps slice and the vacated slice snaps to the 1400 mm horizon. Discreteness helps.

**What actually drives detection is range against the model's noise cliff.** The deployed inverse's 1σ range error goes 141 mm (<1 m) → 236 mm (1–1.4 m) → 401 mm (1.4–1.7 m), while the geometric signal from a 200 mm move can never exceed 200 mm. Beyond ~1 m the noise simply overtakes the signal.

**Decomposing the 62.9% vs 49.5% gap: the near bands contribute almost identically (0.426 vs 0.407). The entire deficit is the 1000–1400 mm band** — Path04 draws 0.184 of its total from it, Path06 only 0.082. And comparing like with like, Path04's *walls* at 1000–1400 mm detect 48% against Path06's 14%: same range, same reflector type, threefold difference. **That last step is unexplained.** Probably surface orientation relative to the drift direction (a wall you move parallel to does not change range at all), but it has not been tested and should not be asserted.

**The one safe design rule: below 1000 mm everything detects at 74–85% regardless of type.** Path06 has 53% of steps with something under a metre ahead. Raising that fraction is the lever that does not rest on an unproven mechanism.

#### Robot deploy — 2 good laps, then lost, but the run is confounded

`PolicyRuns/default_Path06_run01`, 210 steps, no logged crash. Cross-track median 68, p90 218, max 549 mm; 2.94 laps travelled. By lap:

| | median | p90 | max |
|---|---|---|---|
| lap 1 (steps 0–70) | 60 | 160 | 242 mm |
| lap 2 (steps 71–141) | 55 | 137 | 281 mm |
| **lap 3 (steps 142–209)** | **101** | **397** | **549 mm** |

Two clean laps at Path04-class tracking, then it comes apart — which matches the simulator's 27–45% survival at ~2.3 laps.

⚠️ **But do not read this as a clean test of the policy: session curl was −4.69 °/step, against +0.51 / +1.78 / +0.68 in the three Path04 runs.** Seven to ten times larger, sustained from step 0 (−4.69 / −4.50 / −5.28 by lap), with **43% of steps beyond |5|**. The policy was trained on `rot_bias ~ U(−5,+5)` drawn *per episode*, so a typical training episode sees ~2.5 and a sustained −4.7 sits in the far tail — note this also means the sim sweep above (which varied the *range* of the uniform) does not license "curl doesn't matter" for this run. **Recalibrate and re-run before drawing any conclusion about Path06 on the robot.** Doing two good laps under a disturbance Path04 never faced is arguably the more interesting reading.

### 2026-08-17 — Landmark **displacement**: the route follows the moved landmarks, on every lap. The strongest landmark result so far

Two runs, same policy `PolicyTraining/default_Path04` (rot_bias ±5), `POLICY_INPUT_SOURCE="live"`, commit `f08a4de`. The stored arena was not re-digitized, so `sim_prediction_clean` and all cross-track numbers stay on the common Path04 reference.

**Why displacement and not more removals.** Removal shows a cue is load-bearing; it cannot show the cue is *positional*, because "the scene changed and the state was perturbed" explains it equally well. Displacement discriminates: only a positional cue predicts a route shift whose **direction** follows the cue. This is the second rung of the ladder set out in the 2026-08-14 entry (removal → displacement → cue conflict → novel viewpoint).

#### Run 2 of 2 — the result. `PolicyRuns/default_Path04_run1_B1_moved`, 500 steps, no crash

**B1 and B2 both displaced ~310–330 mm east.** Measured, not assumed — see the method note below. B1 (+327, −44), B2 (+308, +17). B3, B4, B5 and the pole verified unmoved (local frame difference under the detection floor everywhere).

**The route displaced with them, in the same direction, on all nine laps.** World-frame median offset of the trajectory from the target path, over the affected stretch (54–75% of the lap):

| | dx | dy |
|---|---|---|
| run01 baseline | +9 | +7 |
| run02 baseline | +13 | +14 |
| **B1+B2 moved east** | **+107** | **+82** |

Per-lap dx: **+156, +84, +141, +107, +52, +101, +148, +126, +92** — nine laps, never negative. Outside that stretch the profile sits within ±45 mm of baseline, so the effect is localised, not a global drift.

**Read it as partial cue capture.** The route follows the moved landmarks by roughly **a third** of their displacement. Two blocks moved; three blocks and all four walls stayed. The stationary geometry anchors most of the estimate and the moved blocks pull it partway — which is exactly the graded outcome the animal-navigation displacement assay produces, and a stronger claim than any removal run can support.

**The robot is not degraded, it is driving a displaced route accurately.** |cross-track| median 57 mm vs 37 for both baselines, p90 188 vs ~90, max 429, 500 steps, no collision. Control is intact; what moved is where it thinks the route is. ⚠️ These three medians were recomputed on one metric (5 mm densified centreline, nearest-point) so they compare to each other; they are **not** on the same footing as the 63/64 mm quoted in the 2026-08-14 entries.

**The curl confound does not apply here.** Session curl was **−1.31 °/step** against +0.51 / +1.78 in the two baselines, and today's sessions run negative where August 14th's ran positive. It cannot produce this effect: a constant rotation bias acts uniformly around the loop, whereas this shift is confined to one quarter of it, returns to baseline everywhere else, and points in the direction the landmarks were moved.

**Limitation — no attribution between B1 and B2.** They moved together. A B1-only repeat is the obvious follow-up and has a quantitative prediction: the shift should roughly halve.

#### Run 1 of 2 — aborted at 36 steps, and why. `PolicyRuns/default_Path04_run02_B1_moved`

Designed as B1 → (−25, −864), 250 mm **due south, x unchanged**, chosen because it preserves all seven in-cone phases (33–39), gives a uniform −247 mm range shift with ≤2° bearing change, and keeps 377 mm usable clearance. **As actually placed, B1 went to (−200, −920)** — 180 mm west and 305 mm south, a 354 mm move to the SSW. Two consequences, both avoidable:

1. **Clearance fell to 466 mm centreline / 276 mm usable**, below the path's own 370 mm minimum, making it the tightest point in the arena.
2. **The in-cone window was no longer preserved** — at steps 31–35 the moved box sat at bearing +37° to +50°, outside the ±35° cone, exactly where the trained box would have been inside it. So the run was part displacement, part removal, which is the confound the due-south placement existed to avoid.

The run is still informative. Cross-track was **3 mm at step 26**, dead on the path; **step 27 is the first step the box is inside the beam**, and error then grows monotonically: 58, 92, 121, 163, 206, 246, 319, 388, 414, 427, 458 mm. Both baselines stay under 120 mm through the same stretch. The failure is a specific route error — x pinned at 334–342 mm for seven steps while the trained route bends west — followed by a late left turn that carried the robot into the box, ending 165 mm off its east face. The sensory side is unambiguous: at steps 33–35 the stored geometry puts B1 at 528–814 mm in the centre and left slices and the live sonar read 1718–2294 mm. **User's reading, and it fits: the robot holds its heading where the expected return is absent, turns late, and the late turn is what hits the box.** Same failure *shape* as the B1 removal run (missed turn onset, wide recovery) one leg earlier — what the box supplies is turn timing.

Caveat retained: session curl was −0.84 °/step against +0.44/+0.74 in the baselines over the same 37 steps. It was constant from step 0 while the robot tracked to 3 mm at step 26, so it does not explain the onset, and run01 spent 13% of its 37-step windows at ≤ −0.84 without ever leaving the path — but with n=36 and a confounded placement this run should not be quoted as a result on its own. Run 2 is the result.

#### Method note — measure the placement, do not eyeball it

**Differencing the run's `env_*/arena.png` against a reference run's locates every moved obstacle to ~15 mm, at zero cost.** Greyscale difference, Gaussian blur, threshold at ±18, connected components: the "brighter" blob is where an obstacle left, the "darker" blob is where it arrived. Validation: on run 1 the vacated blob landed at (−17, −627) against B1's true trained centre of (−21, −612).

This is the practical answer to the env-snapshot gap flagged on 2026-08-14. `SCRIPT_RunPolicy.py` still copies the stored arena over the fresh capture, so `arena_features.npz` is never a record of what was present — but the **fresh `arena.png` and per-camera warps are kept**, and they are enough. Full re-digitisation needs manual annotation; this does not. **Worth landing as `SCRIPT_CheckArenaDelta.py` and running right after placing an obstacle, before committing to a 500-step run** — it would have caught run 1's misplacement in seconds.

### 2026-08-14 (late) — The four-run landmark design, complete: a 1 mm replicate, and both landmarks are load-bearing

**Supersedes the 2026-08-14 entry below on three points** (the pole is *not* inert; the excess is *not* a session offset; it is *not* a distant-cue effect). The entry below is left as written — its data are correct, its conclusions about the pole were drawn before the replicate existed.

Four runs, 500 steps each, same policy `PolicyTraining/default_Path04` (rot_bias ±5), `POLICY_INPUT_SOURCE="live"`, commit `f290630`. Fresh batteries and a fresh `SCRIPT_CalibrateRobot.py` before each. The stored arena was never re-digitized, so `sim_prediction_clean` stays on a common reference throughout.

| run | folder | arena | median | p90 | max |
|---|---|---|---|---|---|
| baseline | `default_Path04_run01` | all in | 63 | 108 | 201 |
| **replicate** | `default_Path04_run02` | all in | **64** | **109** | **186** |
| pole removed | `default_Path04_run01_Pole_removed` | pole out | 86 | 180 | 620 |
| both removed | `default_Path04_run01_B1_Pole_removed` | pole + B1 out | 128 | 374 | 639 |

**THE METHODOLOGICAL RESULT, and it is worth as much as the experimental one: a 500-step run reproduces to +1 mm on median and p90 across a battery change and a recalibration.** Effects down to ~10 mm are measurable on this rig. Nothing else in this project has established that, and without it none of the rows above can be interpreted. **Re-measure this if the rig, path or policy changes — every effect size here is quoted against it.**

Sessions were also motor-matched, measured from the tracker rather than assumed (residual = actual yaw change − commanded rotation): median curl +0.51 / +0.71 / +0.68 °/step and drive scale 1.0075 / 1.0063 / 1.0071 across baseline / both-out / pole-out. Against a policy trained on ±5 °/step, those differences are nothing.

**Both landmarks are load-bearing, and B1 is worth about twice the pole** (+65 vs +23 mm). That ratio matches their acoustic salience: B1's removal moved the centre-slice reading by +563 mm, the pole's by under 214.

**Neither acts locally.** For each landmark the manipulated window itself runs at baseline and the cost appears elsewhere. B1's window (33–39) went −3/+15/+29/+33/+50/+74 while the corner three steps later blew out to **+154/+344/+483**; the pole's window (8–12) came in at **+5 mm**. Sorting the pole-out excess by whether the pole was even in the beam:

| | replicate (noise floor) | pole out |
|---|---|---|
| 11 phases with the pole **in** the ±35° beam | −15 mm | **+11 mm** |
| 45 phases with it **out** of the beam | +5 mm | **+31 mm** |
| …in-beam, near (<1.4 m) | +6 | +6 |
| …in-beam, far (≥1.4 m) | −20 | +18 |

**The excess is three times larger where the pole cannot be seen.** So it is not a moment-to-moment sensory effect, and the "distant beacon" hypothesis raised earlier the same day is refuted — the far in-beam phases are not the hot spot. What fits both landmarks is a **state** effect: the recurrent state is perturbed where the landmark is sensed, and the trajectory consequence surfaces downstream. Caveat on the +31: it is a median over 45 phases and per-phase replicate scatter is ±30 mm, so no individual phase in that set is interpretable; the systematic positive shift across all 45 is.

**Separate finding — the class head does not report pole presence.** p(pole) in the pole window is **0.445 with the pole physically there and 0.478 with it physically removed** (n=500 each); whole-lap 0.192 vs 0.195 vs 0.166 vs 0.170 across all four runs. Per-phase medians are near-identical run to run. The highest p(pole) anywhere on the lap is at phases 13–14 (0.82–0.92) — **B4's window, a 210 mm block, not a dowel.** The likeliest reading is that the head learned *compact/edge-like reflector vs extended surface*, which is acoustically real (a box corner and a 25 mm dowel both give a compact specular return) but is not the distinction the label encodes — blocks are `kind=0` in training. **This belongs in the paper's inverse Results as a negative result, not buried.** It also means the pole-removal run cannot test whether the policy uses the class channel: you cannot test indifference to a signal that never moved. The class-inertness claim still rests on the simulation ablation alone. **The manipulation that would test it on the robot: put a flat panel where the pole was, at the same range** — range held, acoustic class genuinely changed.

⚠️ **`SCRIPT_RunPolicy.py`'s env snapshot is useless as a record of the physical arena.** All four runs' `env_*/arena_features.npz` are byte-identical to `TargetArenas/Path04/env_0001_2026-08-13T11-43-30/arena_features.npz` (md5 `0575c177…`) — it copies the stored arena rather than re-digitizing. **No run in this series has an independent record of what was actually in the arena**, and during the pole-out run this cost real time chasing whether the pole had actually been removed. Fix before the next manipulation run; it is the only way to catch a setup that does not match intent.

Note on small n: at 93 steps the pole window read p(pole) 0.630 and looked like a phantom-pole regression. At 500 it is 0.478. **Nothing in this series should be read below ~300 steps** — the pole window carries only ~8 samples per 1.7 laps.

### 2026-08-14 — Landmark removal on the robot: the path degrades globally, and the error appears *downstream* of the removed object

> **⚠️ SUPERSEDED in part by the 2026-08-14 (late) entry above.** The conclusion drawn here and in the section below that the pole is inert was made before a baseline replicate existed. The replicate lands at +1 mm, so the pole's +23 mm is real. Data below stand; the pole conclusions do not.

`PolicyRuns/default_Path04_run01_B1_Pole_removed` (500 steps) against `PolicyRuns/default_Path04_run01` (500 steps, same policy `default_Path04` at rot_bias ±5, `POLICY_INPUT_SOURCE="live"`, fresh batteries, recalibrated before each). Commit at run `f290630`. Two objects removed together: the **pole** at (1415, −624) and **B1**, the central block at (−25, −614). The stored arena was deliberately *not* re-digitized, so `sim_prediction_clean` stays on a common reference across both runs.

| | baseline | removed |
|---|---|---|
| cross-track median | 63 mm | **128 mm** |
| p90 | 108 mm | **374 mm** |
| max | 201 mm | **639 mm** |
| min obstacle clearance | 364 mm | **85 mm** |
| steps under 250 mm clearance | 0 | **15** |

**The effect is real and large — and it is not where the objects were.** Per-phase medians (56-step lap; delta = removed − baseline):

```
B1 window   33  34  35  36  37  38 | 39   40   41   42
delta (mm)  -3 +15 +29 +33 +50 +74 |+154 +344 +483 +139

pole window  8   9  10  11  12     | 14   15   16   17
delta (mm)  +5 +57 +28 +33 +54     |+132 +198 +237 +285
```

Through each manipulated window the robot tracks near baseline; it falls apart on the following steps. **The prediction going in was a local excursion at B1 and nothing at the pole. Locality was wrong.**

**The corner at phase 39–41 is the mechanism, and it is concrete.** The path tangent swings 122° → 163° → −171° (top-left turn). Baseline rotates 15.5 / 23.5 / 16.8°; removed rotates **4.7** / 21.2 / 14.3°. It under-turns at the entry, carries straight on, then swings wide to 546 mm. **B1 was not an obstacle to avoid — it was the cue that the corner had arrived.** Seen from outside this reads as the robot hunting for a landmark before giving up; in the data it is a missed turn onset followed by a wide recovery.

**Not accumulating.** Lap medians — baseline 79/61/65/51/60/64/55/77/60, removed 130/135/113/129/117/203/**68**/171/109. Stationary but highly variable; lap 7 came back to near-baseline. The policy has not lost the path, it has become unreliable on it. (An earlier read at 300 steps flagged a possible progressive decline; the full run refutes that.)

**Sensory side — the pole removal was near-silent, as the geometry predicted.** Centre-slice live reading in the pole window 1273 → 1152 mm (the wall sits right behind the dowel, so removing it barely changes the profile). B1's removal *was* seen: 1741 → 2304 mm, +563. The class channel did register the pole's absence moderately — p(pole) in the pole window 0.445 → 0.306, above-0.5 on 46% → 32% of steps. **Note the correction: at 300 steps this looked like no change at all (0.403 / 41%); the full run shows a real but modest shift.**

**Limitation — this run cannot attribute.** Both objects came out together, and since the effects propagate around the loop, the +24 mm in the pole window cannot be assigned to the pole rather than to B1's disturbance arriving from upstream.

**NEXT RUN, decided 2026-08-14: pole out, B1 back in** (not the reverse). Power argument: lap-to-lap SD of a phase median is **24 mm** in baseline but **67 mm** in the removed condition, so testing pole-only against the tight baseline detects a ~20 mm shift where testing B1-only against the noisy both-removed run needs ~55 mm — about 2.7× more sensitive. It is also the right logic: the simulation ablation claims the class channel is inert, and inertness is demonstrated by a predicted null against a low-variance reference.
- Put B1 back at **(−25, −614)** within ~50 mm — it is the phase-39 turn cue, so a displaced B1 confounds the comparison. Leave the pole out. Recalibrate first. Run 500 steps for matched n. Do not re-digitize the arena.
- **Pre-registered prediction:** no phase departs from baseline by more than ~20 mm; in particular phases 8–12 and 13–19 sit at baseline.
- **Falsifier:** phases 14–19 went +132 to +285 mm with both objects out. If they light up again with only the pole gone, the pole was contributing and the B1-centred account is wrong.

#### Two hypotheses tested against this run the same day — one killed, one killed, and what survives

**KILLED: "dead-reckoning punctuated by re-registration at landmarks."** The natural story for downstream error is that the policy accumulates drift and re-registers when a landmark comes into view. It predicts error should fall when a *surviving* landmark enters the cone. It does not: median delta is **+53 mm with a surviving landmark in cone (27 phases) vs +55 mm with nothing in cone (17)**, and the two biggest failures happen *while* B4 (phases 14–17) and B5 (40–42) are in plain view. The sanity check settles it — in **baseline**, tracking is **62 mm with a block or pole in the cone and 61 mm with none**. Landmark visibility does not predict tracking quality. The reason: the split counted only blocks and the pole, but **the walls are always in the cone** — the three-slice profile never reads empty (median ~1300 mm everywhere on the lap). There are no landmark acquisition events to re-register on.

**KILLED: "it is just a reactive corridor-follower."** (User's objection, and the data backs it.) The reflex does exist — with the nearest slice under 600 mm the robot turns a median 42° (baseline) / 32° (removed), against 6–8° when over 1200 mm; at the 85 mm moment it commanded −50.9°. **But in baseline it enters that regime on 7 of 500 steps**, and its tightest moment all run was 364 mm at a mild −12.8°. Proximity avoidance is a rescue reflex at the margins; it does not generate the route. Decisive point: **a corridor-follower cannot get lost, and this one does** — it drove to 85 mm from the west wall and 164 mm past B4, repeatedly at the same phases each lap. You only end up somewhere you should not be if you hold an internal estimate of where you are and it is wrong.

**WHAT SURVIVES.** The RNN carries a spatial estimate updated by its own motor commands and corrected by boundary geometry. Removing B1 removed a correction, the estimate mis-registered, and the robot then confidently drove routes belonging elsewhere. This accounts for all three signatures: failures downstream of the manipulation, no rescue when other objects come into view, and near-wall approaches that are not wall-following failures.

**Terminology — resolved 2026-08-14, user's call and it is the right one.** An earlier draft of this entry contrasted "landmark recognition" with "depth-profile following". That was the narrow robotics sense of *landmark* = identifiable discrete object. In the spatial-cognition literature boundary and enclosure geometry **are** landmark cues (geometric module, reorientation work, boundary vector cells), so navigating by the shape of the enclosure *is* landmark-based navigation. **The paper's Task 2 wording "path integration + landmark recognition" therefore stands** — what it needs is a specification of *which kind*: boundary geometry, not identified objects. The real dissociation this project can demonstrate is **geometry vs object identity**, and the pole-only run is the test for it.

**The framing worth taking to the Discussion.** The behavioural criterion for landmark use (perturb a cue, observe a specific route change) is deliberately mechanism-agnostic — that is what makes it usable on animals. This robot passes it, and we can state exactly what is inside: three depth numbers per ping, an inert class posterior, a recurrent state, no object concept, none of it hand-designed. **So the behavioural signature of landmark use is satisfiable by a system with no landmark representation.** Note the further point that keeps this from being merely definitional: the outside view is a *family* of tests of increasing strength — removal (non-discriminating, both accounts pass) → displacement → cue conflict (two identical boxes, or furniture rotated against the walls) → novel-viewpoint transfer. There is real empirical content further down that list; the assay the field runs most often simply does not reach it. User's framing: we only ever get the outside view in a real bat, and hippocampal recordings do not close the gap because they show the map, not the echo-to-space transform — which is the layer this robot instantiates.

Note for the arena: the digitized Path04 arena has **five** free-standing blocks, not four — B1 (−25, −614), B2 (−397, −2138), B3 (1183, 593), B4 (1481, −1765), and B5 (−1395, −1541). Nearest-in-cone windows over the lap are B3 0–7 and 53–55, pole 8–12, B4 13–19, B2 21 and 29–32, B1 33–39, B5 40–43; phases 22–28 and 44–52 have nothing in the ±35° cone.

### 2026-08-14 — Retrain at rot_bias ±5: the failure mode from run01 is gone

`PolicyTraining/default_Path04` (new) vs `default_Path04_bias3` (the policy that ran 8.5 laps, renamed and kept as the control). Only `motion_rot_bias_deg` differs, 3.0 → 5.0 (`82532f4`). `val_mse` 118.90 vs 114.31 — 4% higher, which is the expected price of a wider disturbance distribution and **not** the number to judge this on.

Both policies, 40 rollouts per cell at FIXED rotation bias:

| bias °/step | old collisions | **new** | old track med/p90 | **new** |
|---|---|---|---|---|
| 0.0 | 10% | 8% | 53 / 144 | 63 / 180 |
| 1.0 | 2% | 10% | 55 / 150 | 60 / 170 |
| 2.0 | 2% | 5% | 61 / 169 | 63 / 180 |
| **3.0** | **18%** | **8%** | 68 / 178 | 65 / 176 |
| **4.0** | **25%** | **2%** | 85 / 212 | 60 / 157 |
| **5.0** | **40%** | **15%** | 105 / 261 | 81 / 214 |

- **The old policy collapses at its own training boundary** — 2% collisions at bias 2, then 18 / 25 / 40% past ±3, with tracking degrading 53 → 105 mm median. That is the robot behaviour of run01 reproduced in simulation, which is itself a check on the simulator.
- **The new one is flat across the range**, 60–81 mm median from bias 0 to 5, with no collision trend until the extreme edge.
- **The cost at low bias is small and probably noise.** Marginally looser p90 at bias 0–2 (170–180 vs 144–169), the expected price of spreading the same capacity wider. At n=40 the standard error on a 10% rate is ~5%, so the 0–2 rows are not distinguishable, and the 2% at bias 4 against 15% at bias 5 is noisier than it looks. What survives the noise is the 3–5 pattern.
- **Prediction to test on the robot:** run01 broke at step ~400 when battery sag pushed the residual past −3.6. This policy should hold through it. **The informative part of a deploy is precisely the part that failed last time**, so run long enough to reach it rather than stopping while clean.

### 2026-08-13 (night) — Path04 run01 on the robot: 8.5 laps, no collision, and the curl grows within a run

`PolicyRuns/default_Path04_run01`, policy `default_Path04`, `POLICY_INPUT_SOURCE="live"`, 500 steps. Commit at run `bb8ec51`.

- **500 steps = 8.5 laps, no collision.** Path02's run02 crashed on lap 3. That settles the path question.
- **Tracking error mean 46 / median 33 / p90 83 / max 451 mm**, against run02's 119 / 94 / 266 / 341. The analyser's verdict: 0% of the path is tighter than this run's p90, and a safe path would have needed only **168 mm** of clearance. We built 455.
- **THE FINDING: the yaw residual grows monotonically through the run**, while drive distance stays flat at ~152 mm — so it is rotation, not the drivetrain. Almost certainly battery sag over ~20 minutes of continuous driving.

  | steps | yaw residual | cross-track mean / max |
  |---|---|---|
  | 0–100 | **−0.36**°/step | 37 / 131 mm |
  | 100–200 | −1.32 | 31 / 85 |
  | 200–300 | −2.37 | 32 / 126 |
  | 300–400 | −3.24 | 38 / 122 |
  | 400–500 | **−3.60** | **87 / 451** |

- **The policy absorbs the disturbance up to almost exactly its training range, then stops.** Training drew `rot_bias ~ U(−3, +3)`. Tracking is flat at 31–38 mm mean right through the fourth fifth at −3.24°/step, then degrades sharply once the residual passes −3.6, outside anything it met. **That is the additive-bias mechanism (`db22a14`) validating itself: it works exactly as far as it was trained and no further.** Suggests widening `motion_rot_bias_deg` 3.0 → ~5.0 at the next retrain.
- **This explains the 2026-08-07 "unresolved discrepancy".** That entry recorded run01 predicting +1.25°/step against −2.66 observed, gap unexplained, and floated a calibration-protocol mismatch. If curl grows within a run, calibration — measured from a standing start on a rested battery — captures the *beginning* state while the run average is dominated by the degraded later state. Simpler than the protocol hypothesis, and it gets the sign right. **A per-run average yaw residual conflates two regimes and should not be quoted as one number.**
- **Sim-to-real: the error model is honest.** Real inverse vs geometry, bias +6 mm / RMSE **282 mm**; simulator vs geometry, −18 mm / **312 mm**. Ratio 0.90 — comparable, marginally pessimistic, which is the right direction. Per-band real errors (231 / 168 / 208 / 458 mm at 0–750 / 750–1250 / 1250–1750 / 1750+) track the model's held-out figures on an arena configuration it never saw. The inverse generalises to the new block layout.
- **Consequence for future path design: the clearance/perception trade-off is much weaker than the 2026-08-13 entries assume.** All of that analysis anchored on run02's 266 mm p90 tracking error, which forced ~435 mm clearance, which pushed readings into the imprecise 1–1.4 m band and cost drift detectability. This policy tracks at **83 mm p90**. At that error a ~250 mm-clearance path is safe — Path02 territory, where 86% of steps see something inside a metre against Path04's 56%. **A future path could hug much closer and perceive far better at no real safety cost.**

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
