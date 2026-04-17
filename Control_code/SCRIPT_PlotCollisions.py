#!/usr/bin/env python3
"""Plot collision rates for all completed/running training runs."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

POLICY_DIR = "PolicyTraining"

# Colours and styles per run
RUNS = [
    ("sonar_h00",      "Sonar h00 (baseline)",   "#888888", "-",  "o"),
    ("sonar_h01",      "Sonar h01",               "#1f77b4", "-",  "o"),
    ("sonar_h05",      "Sonar h05",               "#2ca02c", "-",  "o"),
    ("sonar_h10",      "Sonar h10",               "#d62728", "-",  "o"),
    ("test_burst_h01", "Burst h01 (fixed)",        "#ff7f0e", "--", "s"),
    ("test_burst_h03", "Burst h03 (fixed)",        "#9467bd", "--", "s"),
]

fig, ax = plt.subplots(figsize=(11, 5))

for name, label, colour, ls, marker in RUNS:
    path = os.path.join(POLICY_DIR, name, "training_history.json")
    if not os.path.exists(path):
        print(f"  skipping {name} (not found)")
        continue
    with open(path) as f:
        h = json.load(f)
    coll = [v if v is not None else float("nan") for v in h["collision_rate"]]
    gens = list(range(len(coll)))
    suffix = "" if len(coll) == 50 else f" ({len(coll)} gens)"
    ax.plot(gens, coll, color=colour, linestyle=ls, marker=marker,
            markersize=4, linewidth=1.5, label=label + suffix)

ax.set_xlabel("Generation")
ax.set_ylabel("Collision rate (best genome)")
ax.set_ylim(0, None)
ax.set_xlim(0, 49)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_title("Collision rate across training runs")

plt.tight_layout()
out = os.path.join(POLICY_DIR, "collision_comparison.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")
