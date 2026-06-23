"""Render a handful of example echoes so we can pick an interesting one.

Scans a session, selects pings with a located echo spread across range, and
writes one PNG per candidate into Paper/images/echo_candidates/. Once you pick
one, set PING in fig_echo_example.py to that ping.

    Control_code/.venv/bin/python3 Paper/images/scripts/gen_echo_candidates.py
"""

import glob
import sys

import numpy as np
import dill

import style
from paths import ACQ_SESSIONS, IMAGE_RESOURCES, CONTROL

sys.path.insert(0, str(CONTROL))
style.setup()
import matplotlib.pyplot as plt  # noqa: E402

from fig_echo_example import load_ping, plot_echo  # noqa: E402

SESSION = "Acquisition01A"
N = 12
OUT = IMAGE_RESOURCES / "echo_candidates"


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for f in sorted(glob.glob(str(ACQ_SESSIONS / SESSION / "data*.dill")))[::4]:
        try:
            sp = dill.load(open(f, "rb"))["data"]["sonar_package"]
        except Exception:
            continue
        if sp.get("echo_located") and sp.get("raw_distance"):
            rows.append((float(sp["raw_distance"]), f, sp.get("side_code")))
    if not rows:
        print("no candidates found")
        return
    rows.sort()
    idx = sorted(dict.fromkeys(np.linspace(0, len(rows) - 1, N).round().astype(int)))
    picks = [rows[i] for i in idx]

    for old in OUT.glob("echo_*.png"):
        old.unlink()
    for rng, f, side in picks:
        name = f.split("/")[-1].replace(".dill", "")
        p = load_ping(f)
        fig, ax = plt.subplots(figsize=(3.4, 1.8))
        plot_echo(ax, p)
        ax.legend(frameon=False, loc="upper right", fontsize=7)
        ax.set_title(f"{name}   r={rng:.2f} m   side={side}", fontsize=8)
        fig.savefig(OUT / f"echo_{name}_r{rng:.2f}_{side}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  echo_{name}_r{rng:.2f}_{side}.png")
    print(f"{len(picks)} candidates in {OUT}")


if __name__ == "__main__":
    main()
