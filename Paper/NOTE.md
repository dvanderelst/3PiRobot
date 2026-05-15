# Paper state — 2026-05-14 (end of session)

The Introduction and Information-theoretic statement section are drafted and
have been through a thorough paragraph-by-paragraph rework this session.
Everything is committed (15 commits); the only uncommitted items are the
user's unrelated `Control_code/*` changes and `Paper/dieters_scratch_pad.tex`
(the user's scratch file, deliberately untracked).

- **Introduction:** Pars 1-16. **Information-theoretic statement:** Pars 17-22.
- Builds clean: 8 pages, no undefined references or citations, no overfull boxes.
- `references.bib`: 40 entries.

## What changed this session

**`Paper/` is now under git.** It was entirely untracked at the start; it is
now an initial import (`3366a21` supporting material, `97bb449` main.tex) with
a `.gitignore`, followed by 13 more commits. Commit promptly from here on.

**A file-loss incident, recovered.** Early on, `main.tex` was clobbered (a
stale editor buffer plus Dropbox sync overwrote uncommitted edits) — the
forward-inverse subsection, renumbering, and E3 edits were lost before the
first commit captured them. Recovered from the conversation transcript
(`4c87876`). This is why there is now a memory note on the Dropbox co-editing
hazard, and why the user moved to drafting in `dieters_scratch_pad.tex`.

**D4 / E3 / D5 resolved and implemented** — see the resolution notes in
`issues.md` Part I.

**Intro Pars 1-11 reworked, paragraph by paragraph** with the user:
- **Par 2** — "all sensory systems are limited" recast as a *consequence* of
  the channel/filter argument; the DPI-pointer sentence and the
  active-perception clause parked as `%` comments.
- **Par 3 (new)** — information vs representation; informational vs
  computational equivalence, with Marr's numerals example (Marr 1982, Larkin
  & Simon 1987).
- **Par 4 (the crux)** — "feature" formally defined (*a property of the world
  that some behaviour needs to know*); perception framed as the construction
  of a representation; closes with the bat echo example (Wiegrebe 2008, Suga
  1990, Kothari 2018).
- **Par 5** — the chain supports representations at different depths.
- **Par 6** — crystallised around *graded fidelity*: a chain carries a feature
  faithfully, lossily, or not at all (the narrative form of the formal
  Preserved/Lossy/Destroyed trichotomy).
- **Par 7** — the amodal-representation claim scoped per-feature (resolving an
  apparent contradiction with modality-specific features).
- **Pars 9-11 (forward-inverse subsection)** — Par 9 defines the pair, Par 10
  situates it across traditions ("what changes is only the state"), Par 11 is
  the division-of-labour argument, retargeted from blue-specific to a general
  hand-off into the three regimes.

**"readout" cut from the intro region (Pars 1-11).** It is process/product
ambiguous; the framework already has "the chain" (the operation) and "a
representation" (the product). **Downstream "readout" uses remain** — the
green/yellow/blue paragraphs, figure caption, contribution paragraph, and
formal section — see issues.md **A6**. This is the most pressing loose end:
the paper is term-inconsistent until A6 is done.

**Verb discipline established** (apply to any new prose): a chain *carries*
information about features (quantity axis); a representation *makes* features
*explicit* / leaves them *implicit* (format axis). Avoid "encode" (blurs the
present-vs-explicit distinction the paper is built on), "highlights",
"captures".

**Conant & Ashby cut.** It was included as the "deeper root" of the
forward-inverse pair, then dropped from Par 9 on review: its formal content
supports only "a regulator must contain a model", not the forward-inverse
*pairing*, and it is a control-theory result in a perception framing.
`ConantAshby1970` stays in `references.bib` uncited — a candidate for the
policy section or the Discussion, not a forgotten loose end.

**3 new bib entries** beyond the D4/E3/D5 batch: `Marr1982`, `LarkinSimon1987`,
`Wiegrebe2008`. `references.bib` is now 40 entries.

## Things to know that aren't in the files

- **Dropbox co-editing hazard** (there is a memory note on this). `Paper/` is
  in Dropbox and the user co-edits `main.tex`. Protocol: turn-taking; commit
  promptly after each batch; tell the user to reload before they edit. The
  user now drafts in `dieters_scratch_pad.tex` to avoid clobbering.
- **The user is the bat-echolocation domain expert** (their own published
  area). Trust their domain calls — e.g. the Wiegrebe-as-cochlear-model
  framing in the Par 4 bat example.
- **The discuss-then-apply rhythm.** The user revises hard and catches loose
  reasoning. Propose concretely, wait for "yes" / "go ahead", then apply +
  build + commit. They care a lot about verb/term discipline.
- **References not in Zotero.** 9 of the new references were hand-typed (not in
  the `vicarious_learning_paper` collection): ConantAshby1970, Sutton1990,
  HaSchmidhuber2018, RaoBallard1999, Friston2010, WolpertKawato1998,
  LarkinSimon1987, Marr1982, Wiegrebe2008. The user may want to add these to
  the Zotero collection so the curated bib stays complete.
- Earlier-session gotchas still apply: the dolphin "rotation between
  modalities" cross-modal story is a false memory; check the Zotero
  `vicarious_learning_paper` collection (`EIA59P9J`) before any library-wide
  search.

## File state

- `main.tex` — Introduction Pars 1-16, Information-theoretic statement Pars
  17-22. Builds clean, 8 pages.
- `references.bib` — 40 entries.
- `issues.md` — the revision queue; Par references synced to current
  numbering; Part I (D4/E3/D5) resolved; Part II is the active queue.
- `outline.md`, `STYLE.md` — unchanged this session. Read `STYLE.md` before
  writing prose: UK English, no em-dashes, term-first-symbol-after, `% Par N`
  numbering.
- `dieters_scratch_pad.tex` — the user's scratch file. Untracked, not paper
  source; leave it alone.
- `.gitignore` — excludes LaTeX build artifacts, `main.pdf`, JabRef backups,
  drawio `*.bkp`, and `resources/`.

## Suggested first move

`issues.md` Part II is the queue; the priority ordering is at the bottom.
Most pressing:

- **A6 — the downstream "readout" cleanup** (with **A5**, terminology
  alignment). The intro cut "readout" but the green/yellow/blue paragraphs,
  figure caption, contribution paragraph, and formal section still use it; the
  paper is term-inconsistent until this pass is done. A6 flags which spots are
  mechanical swaps and which need judgement (the green pathway and the figure
  caption should be drafted, not blind-swapped).
- Then **B1-B3** (precursor citations for the three pathways) and **A2**
  (logic-chain transitions — mostly addressed; the Par 1 → 2 link remains).
- **A4** is a real outstanding catch: the DPI-pointer sentence is parked, so
  `\ref{sec:info}` appears nowhere active and the formal Section dangles
  unreferenced. Decide whether to restore a pointer.
