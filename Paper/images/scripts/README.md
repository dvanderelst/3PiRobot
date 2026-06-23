# Paper figure scripts

Plotting code for the figures in `main.tex`. Each script reads its data
directly from `Control_code/` (the experimental code lives there) and writes
the finished figure up into `Paper/images/`, where `\graphicspath{{images/}}`
picks it up — so `\includegraphics{fig_inverse_cv}` just works.

## Conventions

- **One script per figure**, named after the figure's LaTeX `\label`:
  `\label{fig:inverse_cv}` → `fig_inverse_cv.py` → `images/fig_inverse_cv.pdf`.
- Start each script with `import style; style.setup()` and finish with
  `style.save(fig, "fig_inverse_cv")`. `style.py` holds the shared fonts,
  sizes, column widths, and the wall/pole/none colour scheme.
- Get all paths from `paths.py` (e.g. `paths.SONAR_MODEL`). It resolves the
  `Control_code` location from the script's own path, so scripts run from any
  working directory. If the data moves, fix it once in `paths.py`.

## Running

Use the Control_code virtualenv (already has numpy/matplotlib):

```
Control_code/.venv/bin/python3 Paper/images/scripts/fig_inverse_cv.py   # one figure
Control_code/.venv/bin/python3 Paper/images/scripts/make_all.py         # all figures
```

## Note on data freshness

`Control_code` outputs (`SonarModel/`, `PolicyRuns/`, ...) are gitignored and
overwritten between runs, so a figure reflects whatever was last produced
there. Regenerate the affected figure after re-running an experiment.

## Files

- `paths.py` — filesystem locations (Control_code data dirs, image output).
- `style.py` — shared matplotlib style + `save()` helper.
- `make_all.py` — run every `fig_*.py`.
- `fig_*.py` — one per figure (added as figures are written).
