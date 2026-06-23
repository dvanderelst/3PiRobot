"""Figure element: a single example echo measurement, for illustration.

Plots the demodulated echo envelopes of the two ears from one recorded ping
versus sample index. Saved as SVG (vector) so it can be dropped into the
robot/setup composite figure, plus a PNG for quick inspection.

`load_ping` and `plot_echo` are reused by gen_echo_candidates.py.

-> Paper/images/fig_echo_example.svg
"""

import sys

import numpy as np
import dill

import style
from paths import ACQ_SESSIONS, CONTROL, IMAGE_RESOURCES

# dills pickle Library.* objects (e.g. ClientConfig); make them importable
sys.path.insert(0, str(CONTROL))

style.setup()
import matplotlib.pyplot as plt  # noqa: E402

SESSION = "Acquisition01A"
PING = "data00211.dill"          # clean echo at ~0.67 m; swap for another ping

# sonar_data columns (Library/Settings.py): 0 = emitter, 1 = right ear, 2 = left ear
LEFT_COL, RIGHT_COL = 2, 1


def load_ping(path):
    """Return the two ear envelopes (shared-max normalized) plus echo metadata."""
    sp = dill.load(open(path, "rb"))["data"]["sonar_package"]
    sd = np.asarray(sp["sonar_data"], dtype=float)
    left, right = sd[:, LEFT_COL], sd[:, RIGHT_COL]
    norm = max(left.max(), right.max()) or 1.0  # shared scale keeps the L/R (ILD) difference
    return dict(left=left / norm, right=right / norm,
                raw_distance=sp.get("raw_distance"), side=sp.get("side_code"))


def plot_echo(ax, p):
    x = np.arange(len(p["left"]))
    ax.plot(x, p["left"], color=style.COLORS["wall"], lw=1.4, label="Left ear")
    ax.plot(x, p["right"], color=style.COLORS["pole"], lw=1.4, label="Right ear")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Echo amplitude (norm.)")
    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(0, 1.05)


def main():
    p = load_ping(ACQ_SESSIONS / SESSION / PING)
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    plot_echo(ax, p)
    ax.legend(frameon=False, loc="upper right")

    style.save(fig, "fig_echo_example", dest=IMAGE_RESOURCES)     # asset: PDF + SVG
    fig.savefig(IMAGE_RESOURCES / "fig_echo_example.png", dpi=150)  # inspection


if __name__ == "__main__":
    main()
