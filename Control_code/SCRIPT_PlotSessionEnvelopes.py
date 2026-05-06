#!/usr/bin/env python3
"""
SCRIPT_PlotSessionEnvelopes.py

For each policy run (or training session) in `PolicyRuns/` (or
`SonarSessions/`), load all the saved sonar envelopes, compute the
sample-wise mean and standard deviation across pings, and plot one
mean ± SD trace per session on a single figure (one row for the L
channel, one for R). Useful for spotting between-session amplitude
shifts that aren't visible in any single-session view.

Hypothesis the plot is meant to test: hardware/environment state
shifts the captured envelope amplitude in clusters (e.g. some sessions
sit at ~25 k peak, others at ~28 k+), and the cluster identity is
session-level (consistent across all pings within a session).

Usage:
  .venv/bin/python SCRIPT_PlotSessionEnvelopes.py [SESSION_GLOB]

If no glob is given, defaults to today's run pattern.
"""
import glob
import os
import sys

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import Library.Settings as _settings
_settings.data_folder = "PolicyRuns"
from Library.DataProcessor import DataProcessor


DEFAULT_GLOB = "rnn_sup_loop2_h32_nosigma_run*"


def _load_session_envelopes(session_dir: str):
    """Return (L, R) arrays of shape (N, T), or (None, None) if loadable
    but empty / wrong shape, or skip with a print on actual failure."""
    proc = DataProcessor(session_dir, cache_dir=None, force_recompute=True)
    try:
        proc.load_sonar(flatten=False)
    except Exception as exc:
        print(f"  ⚠️  load_sonar failed for {session_dir}: {exc}")
        return None, None
    sd = np.asarray(proc.sonar_data, dtype=np.float32)
    if sd.size == 0 or sd.ndim != 3 or sd.shape[2] != 2:
        print(f"  ⚠️  unexpected sonar_data shape {sd.shape} in {session_dir}")
        return None, None
    return sd[:, :, 0], sd[:, :, 1]


def _per_session_stats(L, R):
    """Per-sample mean and std across pings (axis 0)."""
    return (L.mean(axis=0), L.std(axis=0),
            R.mean(axis=0), R.std(axis=0),
            int(L.shape[0]))


def main(glob_pattern: str = DEFAULT_GLOB):
    base = "PolicyRuns" if "rnn_sup" in glob_pattern else "SonarSessions"
    pattern = os.path.join(base, glob_pattern)
    sessions = sorted(glob.glob(pattern))
    if not sessions:
        print(f"No sessions found at {pattern}")
        return
    print(f"Found {len(sessions)} sessions matching {pattern}")

    cmap = plt.get_cmap("tab20" if len(sessions) > 10 else "tab10")

    fig, (axL, axR) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    rows = []
    for i, sess_path in enumerate(sessions):
        name = os.path.basename(sess_path)
        L, R = _load_session_envelopes(name if base == "SonarSessions" else os.path.relpath(sess_path, base))
        if L is None:
            continue
        mL, sL, mR, sR, n = _per_session_stats(L, R)
        x = np.arange(L.shape[1])
        c = cmap(i % cmap.N)
        rows.append((name, n, float(mL.max()), float(mR.max())))
        axL.fill_between(x, mL - sL, mL + sL, color=c, alpha=0.10)
        axL.plot(x, mL, color=c, lw=1.0, alpha=0.95, label=f"{name}  (n={n})")
        axR.fill_between(x, mR - sR, mR + sR, color=c, alpha=0.10)
        axR.plot(x, mR, color=c, lw=1.0, alpha=0.95, label=f"{name}  (n={n})")

    for ax, ch in ((axL, "L"), (axR, "R")):
        ax.set_ylabel(f"envelope ({ch} channel)")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("sample index")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle(f"Per-session mean ± SD envelope — {pattern}\n"
                 "Bands = ±1 SD across pings within the session.")
    fig.tight_layout()

    out_path = os.path.join("PolicyRuns" if "rnn_sup" in glob_pattern else "SonarSessions",
                            "_session_envelope_overlay.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote: {out_path}")

    print(f"\n{'session':<46} {'n':>4} {'L max':>7} {'R max':>7}")
    for name, n, lmax, rmax in rows:
        print(f"  {name:<44} {n:>4d} {lmax:>7.0f} {rmax:>7.0f}")


if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GLOB
    main(g)
