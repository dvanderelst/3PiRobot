# Project conventions for Claude

## Cross-session memory

Read `handoff.md` at the repo root before doing anything else. It's the
portable cross-machine project memory, written by the user and prior
Claude sessions. Treat its "Working conventions" as authoritative for
project-state context.

At the end of each session, proactively offer to update `handoff.md`:
  - Update the Code state or Paper state section that matches the work
    just done.
  - Update "Where to pick up" so the next session knows what's next.
  - Keep changes scoped to what actually shifted — don't restate stable
    parts.
  - Always confirm with the user before writing.

## Handoff structure

Keep `handoff.md` organised into three top-level state sections:
  - **Code state** — work in `Control_code/` and `Robot_code/`.
  - **Paper state** — work in `Paper/`.
  - **Performance notes** — chronological log of model/robot
    performance numbers, append-only at the top so the most recent is
    read first. Each entry: date, what was measured, config (sessions,
    flags, model), commit at time of measurement, metrics. Don't edit
    older entries; if a measurement is redone, write a new entry that
    references the prior one. This section is the canonical record —
    `SonarModel/`, `PolicyTraining/`, and `PolicyRuns/` are gitignored,
    so per-run JSONs get overwritten and would otherwise be lost.

Plus a "Where to pick up" pointer near the top that briefly says what's
next in each. When updating, edit only the section that matches what the
session actually shifted; the other sections stay untouched. Whenever a
model is trained or a robot experiment yields numbers worth keeping,
add a Performance-notes entry — that's the durable record.

## Workflow rules

- **Discuss-then-apply.** Propose concretely, wait for "yes" / "go
  ahead", then apply + commit. The user revises hard between turns.

- **Don't bundle user's pre-existing uncommitted changes into agent
  commits.** The user often has WIP edits to robot scripts (`Settings.py`,
  `SCRIPT_CalibrateRobot.py`, `SCRIPT_RunPolicy.py`, `SCRIPT_TrainPolicy.py`,
  `SCRIPT_TakeEnvSnapshot.py`, etc.) that predate your session. Stage only
  the files you actually modified. If your edits and the user's overlap in
  one file, temporarily revert the user's hunks, commit your own, then
  restore. Always inspect `git diff` before staging.

- **Commit style.** Topical, concise. Prefer separate commits per concern.
  Milestone commits ("this worked, here's what unblocked it") should be
  bisect-friendly — never rewrite or squash them.

- **Dropbox co-edit hazard.** The project lives in Dropbox; the user may
  edit between turns. Commit promptly after each batch so changes are
  durable across machines.

- **Domain authority.** The user is the bat-echolocation domain expert.
  Trust their biology calls. Push back on prose, not biology.

## Project gotchas

- **Paper build cwd.** `pdflatex main.tex` must run from `Paper/`, not
  the project root (a stale `resources/main.tex` exists elsewhere).

- **Scratchpad is untracked.** `Paper/dieters_scratch_pad.md` is in
  `.gitignore`. Update it when referenced; never commit it.

- **Mapping / SLAM is held back for a separate paper.** Don't propose
  adding it to the current paper.

- **Data folders are gitignored.** `AcquisitionArenas/`,
  `AcquisitionSessions/`, `PolicyTraining/`, `PolicyRuns/`, `TargetArenas/`,
  `SonarModel/`, etc. are output dirs — don't try to commit anything under
  them.
