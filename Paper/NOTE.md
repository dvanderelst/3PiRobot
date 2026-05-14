# Paper state — 2026-05-14

Picking up: the three **strategic framing decisions** (D4 / E3 / D5 in
`issues.md` Part I) are now **resolved and implemented**. Next is the
mechanical revision queue in Part II, starting with **B1-B3** (precursor
citations for the three pathways), then **A2** (logic-chain transitions).

## What changed this session

**D4 / E3 / D5 resolved.** The user's call: lean heavy on theoretical
framing now, since this text is also a proposal seed, and trim for the
paper later.

- **D4 -> dedicated subsection.** New `\subsection{The forward--inverse
  model pair}` between "Modalities preserve different features" and
  "Cross-modal learning lives in the overlap". Two paragraphs (new Pars
  7-8). Par 7 defines the forward-inverse pair and names the traditions
  (motor control, model-based RL, predictive coding, active inference,
  cybernetics) with grouped citations. Par 8 carries the D5 argument.
  Heavy on content, structure-preserving: the bat hook still opens Par 1.
- **E3 -> medium, heavy deferred.** Par 12 cites Bennett 2023 as an
  accessible synthesis on off-line simulation; Par 13 gains the
  world-models sentence (learned world models as a central missing
  component of contemporary AI). Heavy Discussion engagement parked for
  Discussion drafting.
- **D5 -> deployed full.** The counterfactual-vs-grounding argument is
  Par 8; the Conant-Ashby good-regulator theorem and Friston 2010 are
  cited in Par 7. Both anchors land, so the separate Par-8 parenthetical
  from the issues.md draft was dropped as redundant prose rather than
  omitted.

**8 new bib entries** in `references.bib`: JordanRumelhart1992,
WolpertKawato1998, Sutton1990, HaSchmidhuber2018, RaoBallard1999,
Friston2010, ConantAshby1970, Bennett2023. `references.bib` now has 37
entries. Build clean (7 pages, 322 KB PDF); no undefined references or
citations, no overfull boxes.

**Paragraph renumbering.** The new subsection added two intro paragraphs,
so every Par numbered 7+ shifted up by 2. The introduction now runs Pars
1-13; the Information-theoretic statement section is Pars 14-19. All Par
references in `issues.md` were updated to the new numbering.

## Things to know that aren't in the files

- **6 of the 8 new references are not in the Zotero library.** Only
  Jordan & Rumelhart 1992 (`JYG42T4T`) and Bennett 2023 (`2RPS3UXL`)
  were in the `vicarious_learning_paper` collection. Conant & Ashby 1970,
  Sutton 1990, Ha & Schmidhuber 2018, Rao & Ballard 1999, Friston 2010,
  and Wolpert & Kawato 1998 were hand-typed from canonical bibliographic
  data. The user may want to add them to the Zotero collection so the
  curated bib stays complete.
- **D5's drafted parenthetical was intentionally not used verbatim.** The
  issues.md D5 draft put Conant-Ashby + Friston in a Par-8 parenthetical;
  with the dedicated subsection those citations have a more natural home
  in Par 7's traditions sweep, so the parenthetical would have been
  redundant prose. The substance ("deploy full") is fully present.
- **Framework is still presented as established background, not the
  paper's contribution.** The new subsection was written with this
  guiding principle in mind, but the C1 contribution-framing audit should
  re-check Pars 7-8 specifically.
- Earlier session notes still apply: the dolphin "rotation between
  modalities" story is a false memory; bibkey gotchas (Winters & Reid is
  `Winters2010`; Boyer is a first name); check the Zotero
  `vicarious_learning_paper` collection (`EIA59P9J`) before any
  library-wide search.

## File state

Still nothing in `Paper/` is committed (`git status` reports `?? Paper/`).
A logically grouped commit is warranted at some point; the user prefers
separate commits per concern. Candidate split: (1) the 8 new bib entries,
(2) the D4 subsection + paragraph renumber in `main.tex`, (3) the E3
edits, (4) the `issues.md` / `NOTE.md` housekeeping.

- `main.tex` — Pars 1-13 (intro) + Pars 14-19 (info-theory), builds clean.
- `references.bib` — 37 entries.
- `issues.md` — Part I D4/E3/D5 marked resolved; Part II Par numbers
  updated; A1 and A2 partly overtaken by the new subsection (see the
  notes in those sections).
- `outline.md` — strategic plan, untouched this session.
- `STYLE.md` — UK English, no em-dashes, term-first-symbol-after,
  `% Par N` numbering. Read before writing prose.
- `resources/` — Vanderelst & Peremans 2015 precursor PDF; Bennett 2023
  epub (extracted text in `/tmp/bennett.txt` if needed). Git-ignored:
  large binaries / copyrighted, not paper source.
- `.gitignore` — excludes LaTeX build artifacts, `main.pdf`, JabRef
  backups, drawio `*.bkp` files, and `resources/`.

## Suggested first move

Open `issues.md` Part II. B1-B3 are the next queue items:

- **B1** (green pathway, Par 10) is fully open — Gupta, Hoffman & Malik
  2016 cross-modal distillation is the direct ML precedent.
- **B2 / B3** (yellow Par 11, blue Par 12) are now reduced to optional
  reinforcing cites inside those pathway paragraphs; the broad lineage
  (Rao & Ballard, Sutton, Ha & Schmidhuber) is already cited in the new
  Par 7.

Then A2 (the remaining Par 8 -> Par 9 transition), A5 (terminology:
standardise on "vicarious rehearsal"), and the small A1 readout-definition
tidy-up.
