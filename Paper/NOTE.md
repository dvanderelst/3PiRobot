# Paper state — 2026-05-15 (end of session)

The introduction has been rewritten from scratch under a substantially
reframed theoretical setup. Pars 1-7 are drafted; the paper builds clean
at 4 pages. The Methods, Results, and Discussion sections are not yet
drafted. The only uncommitted item is `dieters_scratch_pad.tex`,
intentionally untracked.

## File state

- `main.tex` — 7-paragraph introduction. No formal section. No figure.
  Builds clean.
- `references.bib` — 52 entries (was 40). 12 added this session.
- `STYLE.md` — unchanged.
- `dieters_scratch_pad.tex` — the user's scratch file. Untracked. Contains
  the crux statement plus parked notes for the Discussion and proposal
  (rich-sense / specialised-sense architecture, MacIver visual-range
  argument with Bennett verification, vision-as-calibrator bias as a
  proposal direction).
- `images/` — empty. Old `theory.drawio` and the in-progress
  `theory02.drawio` are archived in `resources/`.
- `resources/` — gitignored. Contains the v1 archive (`v1_main.tex`,
  `v1_issues.md`, `v1_NOTE.md`, `v1_outline.md`), the v1 proposal
  (`main.tex` + `references.bib`), Bennett's book (epub), the original
  Vanderelst-Peremans 2015 PDF, and both theory figures.
- `.gitignore` — now ignores both `resources/` and `oldstuff/`.

## The major reframe this session

The framework was inverted from the v1 three-regime setup
(inverse-training + forward-training + vicarious-learning, with
green/yellow/blue colour scheme) to a one-regime-plus-consequence
setup:

- **The framework operation is cross-modal inverse training**: when
  modality A has a simple (trivial / non-learned) inverse for a feature
  and modality B carries the same information but cannot read it off,
  A can supply training labels for B's inverse.
- **Vicarious learning is a downstream consequence**: once B's inverse
  produces the same features as A's, behaviour learned through A can
  be deployed through B alone. The forward model is *not* a framework
  primitive in this version; it is a methods-level implementation
  detail (the σ_sim noise injector in `predict_from_profile`).

The Information-theoretic statement section (v1 Pars 17-22) is removed
entirely. The DPI was load-bearing for the three-regime distinction;
under the reframe, it isn't doing work the prose can't do informally.

The A/B convention was **flipped**: A = simple inverse / supervisor;
B = inverse must be learned / deployed at test time. This is the
opposite of v1 (which had A = trained, B = supervisor). The scratchpad
crux uses the new convention.

The register was deliberately lowered. No `F_A ∩ F_B` notation in the
prose, no `\moddata → \estfeats` formal arrows. The inverse-model
concept is introduced in one sentence; the rest of the intro reads as
narrative anchored in animal examples.

## Introduction structure (Pars 1-7)

1. Modalities differ in what they encode (vision colour/2D layout
   faithful; bat echo distance faithful, layout lossy, colour absent).
2. Overlap with biological evidence (dolphins, rodents) + asymmetry of
   accessibility (vision-azimuth trivial vs sound-azimuth needs
   computation).
3. *Inverse model* defined; trivial vs learned with the bat-distance
   pathway (cortical distance representation) as the trivial-via-
   neural-pathway example and the binaural-azimuth pathway as the
   learned example.
4. Cross-modal supervision: barn owl developmentally, ferret SC,
   ventriloquism aftereffect in adult primates. Closing sentence flags
   the vision-as-calibrator bias and notes the framework's generality
   regardless.
5. Bat puzzle: cluttered niche + poor sonar reconstruction with
   behavioural and neural evidence + remarkable spatial competence
   nonetheless.
6. Bats also see; vision could teach sonar AND drive offline rehearsal;
   defines *cross-modal vicarious learning* formally; bat-at-night
   image.
7. Contribution: the robot. Vision-supplied 3-distance feature
   supervises sonar's inverse; scene model enables imagined rehearsal;
   sonar runs alone at deployment.

## Bibliography additions this session

- **Cross-modal calibration (mammals):** `KingCarlile1995`,
  `Recanzone1998`, `MuganMacIver2020`. The first two: filled with
  reasonable bibliographic info but **user should verify** against
  authoritative sources before final submission.
- **Bat spatial behaviour and sonar limitations:** `Holderied2006`,
  `Schnitzler2003`, `Yovel2009`, `Wiegrebe1996`, `Geberl2019`,
  `Warnecke2018`, `Barchi2013`, `VonHelversen2005`, `Stones1969`. All
  pulled verbatim from the v1 proposal bib (`resources/references.bib`)
  so author/year/journal info is accurate.

Many v1 citations are now uncited (the anatomy sequence in Par 1, the
information-theory citations, Marr / Larkin-Simon). They remain in the
bib for potential Methods or Discussion use.

## Things to know that aren't in the files

- **The Dropbox co-editing hazard still applies.** Paper/ is in
  Dropbox; the user co-edits `main.tex` between turns. Protocol:
  commit promptly after each batch; tell the user to reload before
  they edit.
- **The build cwd matters.** `pdflatex main.tex` must be run from
  `Paper/`, not from the project root or from `resources/`. The
  resources/ directory contains its own `main.tex` (the v1 proposal),
  which will be processed instead if cwd is wrong. Always
  `cd /home/dieter/Dropbox/PythonRepos/3PiRobot/Paper && pdflatex ...`.
- **The discuss-then-apply rhythm.** The user revises hard between
  proposals. Propose concretely, wait for "yes" / "go ahead", then
  apply + build + commit. They care about precise wording and will
  push back on loose claims.
- **The user is the bat-echolocation domain expert.** Their domain
  calls (especially around sonar limitations and spatial behaviour)
  are authoritative. Trust them.
- **The user drafts in `dieters_scratch_pad.tex`.** This is where new
  prose attempts and parked observations live. It's deliberately
  untracked. Read it (and update it) when the user references it.
- **Forward models are intentionally absent from the intro.** They
  reappear in Methods only, as the σ_sim noise injector in
  `predict_from_profile`. Do not reintroduce them in the intro
  without an explicit framework discussion first.
- **`theory02.drawio` is preserved in `resources/`** for proposal
  reuse, not for this paper. The user explicitly decided the paper
  needs no theory figure; the formalism returns in the proposal.

## Suggested first move

Three viable next moves, roughly in order of expected payoff:

1. **Methods section.** The pipeline is well-documented in
   `resources/v1_outline.md` and in the `Control_code/Docs/rationale.md`
   file. The Methods section is the natural next thing to draft,
   especially since it can introduce the forward-model machinery
   (`predict_from_profile`, σ_sim) that the intro deliberately omits.
2. **Refinement sweep of Pars 1-7.** The intro reads cleanly but has
   some typos in user-edited passages (`easility`, `retinoptopical`)
   and some places where the prose could tighten further. A
   paragraph-by-paragraph review with the user could close these out
   before Methods.
3. **Contribution paragraph (Par 7) sharpening.** Par 7 currently
   carries methodological detail (transducer count, behavioural
   cloning, pure-pursuit teacher) that could either stay (concrete
   contribution claim) or move to Methods (cleaner intro). Worth
   asking the user.

If unsure, ask the user which of these three they want next; do not
just start drafting Methods.

## Commit history this session

```
bccc44f Paper: gitignore oldstuff/ alongside resources/
3f436ef Paper: draft full introduction with bat-application framing
0f74e6c Paper: scaffold new main.tex around the reframed introduction
ace32f3 Paper: archive v1 (main.tex, issues.md, NOTE.md, outline.md)
```
