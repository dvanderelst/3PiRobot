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
  **Next, in order:**
  1. **Experiment 2 runs on the robot, but Path02 is not usable as drawn — REDRAW THE PATH FIRST.** `default_Path02_run02` completed **2.1 laps (110 steps)** with lap 2 tracking lap 1 to 116 mm, then **crashed at steps 116/117** near (-227, -1120). The cause is geometric, not control:

     | | |
     |---|---|
     | Path02 min clearance (centreline -> obstacle) | 155 mm |
     | robot radius | 85 mm |
     | **usable margin at the tightest point** | **70 mm** |
     | robot tracking error: mean / median / 90th / max | 119 / 94 / **266** / 341 mm |

     **49% of the path has less margin than the robot's 90th-percentile tracking error**, so collisions there are structural — no amount of calibration fixes it. (At the crash the robot had drifted 421 mm off-path into a region with 595 mm of clearance, so the *tightest* sections have not even been tested yet.)
     **This is the direct cost of why Path02 was chosen.** It reached more informative perception than Path01 precisely by hugging walls and blocks; Path01's min clearance is 249 mm (164 mm usable) against Path02's 155 mm (70 mm). Closer to obstacles = better sonar = less room for error. **Redraw with a minimum clearance of ~350 mm** (about Path02's current median), i.e. the 266 mm p90 tracking error plus the robot radius, then re-measure informativeness — the compromise number is the honest one to report. Use `SCRIPT_DefinePath.py`; it already draws the 800 mm pole landmark ring.
     **Tooling: `SCRIPT_AnalysePathRun.py` (added 2026-08-07) reproduces every number above** — path clearance profile, informative-perception fraction, robot tracking error, the "% of path tighter than p90 error" verdict, the required minimum clearance, per-step yaw residual and drive scale, and crash locations with their local clearance. Run it on a candidate path **before** training against it: `python3 SCRIPT_AnalysePathRun.py <Arena> [<RunSession>]`. Arena-only mode skips the run analysis.
     ⚠️ **Informative-perception metric mismatch — do not mix the two.** `SCRIPT_AnalysePathRun.py` measures *true geometry, facing along the path, `cls != none`*, and gives **Path02 = 85.5%**. The **26% / 49%** figures quoted for Path01/Path02 in the 2026-08-07 Performance-notes entry came from a different measurement whose definition was not recorded — it reported 73.5% wall yet 49% informative, so "informative" there was a **subset of wall hits**, not simply "not none" (possibly range-limited, or measured through the error model rather than true geometry). **Re-derive both paths with the script's definition before comparing anything**, and treat the old 26%/49% as unreproducible. The qualitative trade-off (clearance vs informativeness) holds under either definition.
  2. **Then collect repeats and the ablation conditions**, on the robot rather than in sim (user's call, 2026-08-07): `use_poles=False` and the blind control. The plumbing exists. Whether the *simulated* policy degrades without the pole channel says something about the simulator, not the robot. Retraining is needed for the new path anyway, so fold the training-noise fix (item 4) in at the same time.
  3. **`RunPolicy` has no arena guard**, unlike `RunDirectPolicy`'s `check_arena_matches_pole()`. A moved block gives a confusing failure rather than a clear one. Worth adding if the furniture moves between sessions.
  4. **Add an additive rotation bias to the training motion model.** `SCRIPT_TrainPolicy.py` perturbs motion with `rot_motor = rot_exec * rot_gain + N(0, 3deg)` — every term is either multiplicative on the commanded angle or zero-mean. **There is no additive bias term anywhere.** The real robot's fault is exactly a fixed additive offset (about -1.1 deg/step even after calibration, uncorrelated with the commanded angle), so it is a perturbation the policy has never met in training, and the multiplicative gain cannot emulate it (at `rot_exec = 0` the gain does nothing but the robot still sheds heading). Suggested: draw a per-episode `rot_bias ~ U(-3, +3)` deg and add it to `rot_motor`. run02 succeeded *despite* this, not because of it — worth making it robust by design. The same gap is why the pre-flight preview says nothing about calibration: it inherits this noise model.
  5. **Tracker settling is the weak link in the measurements.** ~3% of run02 steps are glitches, and the paired +18.2/-23.1 deg residuals at steps 44/45 are **one bad yaw read**, not two bad steps (a wrong `yaw[45]` biases the step before and after equally and oppositely). The calibration run threw several "pose did not stabilise within 8.0s" warnings for the same reason. A bad pose also feeds the policy a wrong `prev_rot`.
- **Uncommitted and deliberately left**: one-line `Path02` arena-name switches in `SCRIPT_DefinePath.py` and `SCRIPT_TakeEnvSnapshot.py` (session state, same pattern as the 2026-07-28 note); `Paper/introduction_logical_analysis.md` deleted but unstaged; new paper figure resources under `Paper/images/image_resources/`.

---

## Paper state

*Last updated: 2026-08-04.*
*Current branch for ongoing work: `direct-learning-poletask`. `main` carries up through the direct-learning rename + Par 9 task commitment.*

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

*Last updated: 2026-08-07.*
*Current branch for ongoing work: `direct-learning-poletask`.*

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
