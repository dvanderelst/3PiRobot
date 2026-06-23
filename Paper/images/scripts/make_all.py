"""Regenerate every paper figure by running each fig_*.py in this folder.

Run with the Control_code venv so numpy/matplotlib are available:

    Control_code/.venv/bin/python3 Paper/images/scripts/make_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def main():
    figs = sorted(SCRIPTS.glob("fig_*.py"))
    if not figs:
        print("no fig_*.py scripts yet")
        return
    for f in figs:
        print(f"=== {f.name} ===")
        subprocess.run([sys.executable, str(f)], check=True)
    print(f"done: {len(figs)} figure(s)")


if __name__ == "__main__":
    main()
