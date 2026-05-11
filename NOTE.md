# Project state — 2026-05-11

Picking up: train + deploy the **blind ablation** (`BLIND = True` at the top
of `SCRIPT_TrainPolicy.py`) and compare against the sighted baseline. The
sighted side has just hit its first clean on-robot deploy; the next move is
to demonstrate that this success depends on sonar, not dead-reckoning.

## Milestone — first clean policy deploy (2026-05-10)

`PolicyRuns/default_Target02_run06/trajectory.png` — full 354-step figure-8
on Target02, both loops tightly on path, no second-lap divergence. First
deploy that didn't unravel by lap 2 since training started for this arena.
The companion ablation sweep (`PolicyTraining/default_Target02/ablations.png`)
also ran end-to-end and shows the expected ranking (full > center/sides >
blind/no_prev_rot).

**Bisect target if anything regresses: `eef35c7`.** That commit message
spells out exactly what unblocked the run, so a future bisect lands here
with the context already attached. Don't squash or rewrite it.

What unblocked the run, in case the commit ever drifts away:
- Recalibrated drive curl (`drive_yaw_curl_deg_per_mm` -0.03693 → -0.01243)
  and distance scale (0.992 → 0.9972). The Friday-eve curl was inflated
  by uncorrected residuals — the lap-2 drift was a calibration loop
  error, not a policy issue.
- Quantified tracker noise (`SCRIPT_MeasureTrackerNoise.py`, since
  removed — output preserved in `Control_code/Diagnostics/tracker_noise_*`):
  σ_yaw ≈ 0.6° per fresh frame, fresh-frame rate ≈ 0.8 Hz. Relaxed
  `wait_for_stable_pose` defaults `yaw_tol_deg` 0.5 → 2.0 (~3σ) and
  `timeout_s` 5 → 8. The 0.8 Hz cap is a DVR/RTSP bottleneck on
  PyLorex's side (still tracked in `PyLorex/TODO.md`).

## Naming change — current folder layout

The historical `default_Target02_h32_nosigma` naming was simplified to just
`default_Target02` on 2026-05-11. Hidden size and the `_nosigma` flag
weren't varying across runs, so the suffixes added noise. New convention:

  `<CONDITION>_<TARGET_ARENA>[_blind]`  →  e.g. `default_Target02`,
  `default_Target02_blind`

Renamed in place (no history loss):
- `PolicyTraining/default_Target02_h32_nosigma/` → `default_Target02/`
- `PolicyRuns/default_Target02_h32_nosigma_run0{6,7}/` → `default_Target02_run0{6,7}/`
- Updated downstream pointers: `SCRIPT_RunPolicy.POLICY`,
  `SCRIPT_Ablations.RUN_DIR`, and the embedded `output_dir` field in
  `PolicyTraining/default_Target02/config.json`.
- The `*.copy` and `code_*.zip` snapshots inside `PolicyRuns/.../files/`
  keep the old name on purpose — those are frozen records of the deploy
  at the time. **Do not rename or modify these.**

## Blind ablation — what's wired and what to do

`SCRIPT_TrainPolicy.py` now has a top-level `BLIND` constant (right under
`CONDITION`). Default `False`. Flip to `True` to train a control policy
that sees only `prev_rot` (in_dim=1) — all sonar channels stripped.

Plumbing in `Library/Policy.py`: `make_obs_layout`, `encode_obs`,
`make_policy_dict`, and `Policy.__init__` all accept a `blind` flag. A
saved blind policy loads and runs end-to-end through the same
`Policy.load` deploy path as the sighted variant; `meas_dict` is allowed
to be `None` so the sonar measurement call can be (and is) skipped in
rollouts.

**Hypothesis being tested:** with motor-noise injection
(`motion_rot_gain_range_pct=0.15`, `motion_drive_gain_range_pct=0.05`,
plus per-step Gaussians) a blind policy cannot use sonar feedback to
correct sustained execution bias. Expected outcome: poor downstream
performance vs. the sighted baseline. Intended as **publication evidence**
that the sighted policy's success is sonar-driven rather than pure
dead-reckoning. Save the comparison plots; they're a load-bearing figure.

**Concrete next steps:**
1. `BLIND = True` at the top of `SCRIPT_TrainPolicy.py`, run training. New
   artifact lands at `PolicyTraining/default_Target02_blind/`.
2. Decide how to present the comparison. Two options worth a quick look:
   - Deploy the blind policy on the robot (same way as run06) and capture
     trajectory plots side-by-side with the sighted run. Most direct
     evidence but requires robot time.
   - Extend `SCRIPT_Ablations.py` to load both policies and produce a
     sighted-vs-blind comparison panel. Sim only, but instant.
   The user will steer; default to asking before doing the on-robot run.
3. If you do deploy: update `SCRIPT_RunPolicy.POLICY` to
   `default_Target02_blind`, bump `REPEAT`. Don't forget to flip it back
   to `default_Target02` for sighted comparisons.

## Other changes this session (committed)

- `862c77d` — removed three obsolete scripts:
  `SCRIPT_DiagnoseDriveCurl.py` (subsumed by Phase 2 of
  `SCRIPT_CalibrateRobot.py`), `SCRIPT_TimeTrackerRequest.py` (one-off
  latency probe; result captured in TrackerNav comments + PyLorex TODO),
  `SCRIPT_SmokeTestTrackerNav.py` (covered by `SCRIPT_RunPolicy.py` as
  the de facto end-to-end test).
- `SCRIPT_DefinePath.py` now forces `matplotlib.use("TkAgg")` before
  pyplot import. The picker depends on real GUI events (mouse/key/motion)
  which PyCharm's inline backend never delivers. Other scripts keep the
  default backend — the user is OK with PyCharm's inline image handler
  for non-interactive plots.

## User / workflow notes

- **Editor:** PyCharm. Default matplotlib backend (inline / SciView pane)
  is fine for most scripts; only force a windowed backend for scripts
  that need real GUI events.
- **Commit style for milestones:** when a change *works* and represents
  a recovery point (calibration that fixed a bug, first clean deploy,
  etc.), the commit message body should explicitly say "this worked"
  and list what specifically unblocked it. The point is bisect-friendly
  context — a future regression should land on the commit that flagged
  itself as known-good.
- **Bundling commits:** when the working tree mixes user-initiated and
  agent-initiated changes, ask before bundling. The user generally
  prefers separate commits per concern over one mixed commit.

## Open items

- **PyLorex frame-rate bottleneck.** Server reports 7-8 Hz internally but
  client sees ~0.8 Hz of distinct reads. Tracked in `PyLorex/TODO.md`.
  Once fixed, tighten `wait_for_stable_pose` defaults back toward the
  pre-2026-05-10 values (yaw_tol 0.5°, timeout 5s).
- **Per-robot calibration.** `Settings.py` `ClientConfig` `default_factory`
  carries Robot01's calibration table; `client2`/`client3` inherit it.
  Only matters when actually deploying on those robots.
- **Diagnostics directory.** `Control_code/Diagnostics/` is currently
  untracked (matches the gitignore pattern of `PolicyRuns/`,
  `PolicyTraining/`). If a future session decides outputs there are
  worth versioning, add `Control_code/Diagnostics/` to `.gitignore`
  explicitly to make the policy intentional rather than accidental.

## Commits since the last NOTE

1. `eef35c7` — Calibration + tracker tolerances tuned, first clean run
2. `862c77d` — Remove obsolete diagnostic / smoke-test scripts
3. `be30d86` — TrainPolicy: simplify output naming + add blind ablation

(Plus the pending working-tree change: `SCRIPT_DefinePath.py` TkAgg
force, and this NOTE.md rewrite — neither committed yet.)
