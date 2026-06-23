"""Shared matplotlib style + save helper for paper figures.

Use at the top of every fig_*.py:

    import style
    style.setup()
    ...
    style.save(fig, "fig_inverse_cv")   # -> Paper/images/fig_inverse_cv.pdf

Keeps fonts, sizes, and the wall/pole/none colour scheme consistent across
figures, and writes output where \\graphicspath can find it.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive; these scripts only save files
import matplotlib.pyplot as plt  # noqa: E402

from paths import IMAGES  # noqa: E402

# Class colours, kept consistent with the deployed-trajectory plots.
COLORS = {"wall": "#1f77b4", "pole": "#d62728", "none": "#7f7f7f"}

# Beige axes background used across all figures.
BG = "#f4efe1"

# Column widths (inches) for a typical two-column journal page.
WIDTH_1COL = 3.4
WIDTH_2COL = 7.0


def setup():
    """Apply the shared rcParams. Call once at the start of a figure script."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "axes.facecolor": BG,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def save(fig, stem, dest=None):
    """Save <stem> as both PDF and SVG. Standalone figures go to Paper/images/
    (default); pass dest=paths.IMAGE_RESOURCES for building-block assets used in
    a composite SVG figure. Transparent figure margin, beige axes kept."""
    dest = dest or IMAGES
    outs = []
    for ext in ("pdf", "svg"):
        out = dest / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="none")
        outs.append(out)
    print(f"[style] wrote {dest.name}/{stem}.pdf + .svg")
    return outs
