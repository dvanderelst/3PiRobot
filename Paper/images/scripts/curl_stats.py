"""Residual motor faults on the Experiment 2 runs, measured from the tracker.

The Methods argue that the controller cannot hold the path by counting its own
movements, because the motor faults do not average out over a lap and nothing
in its input reports them. This script measures how large those faults actually
were on the eleven deployed runs, so the claim rests on the runs the paper
reports rather than on the simulation the controller was trained in.

Each logged row holds the pose the robot sensed from and the rotation the
policy commanded there, so the heading change between consecutive rows answers
that command:

    dyaw = gain * rot_cmd + curl

The two faults are the two coefficients, and they are the same two the training
perturbation models: a multiplicative rotation gain and an additive per-step
curl. Fitted with a trimmed least squares, because about 3% of tracker reads
are glitches and a single bad yaw biases the step before and after it equally
and oppositely.

`curl` is what matters for the argument: it is unrelated to the commanded
rotation, so no amount of counting commands recovers it.

    Control_code/.venv/bin/python3 Paper/images/scripts/curl_stats.py
"""

import json
import os
import sys

from paths import CONTROL, POLICY_RUNS, SCRIPTS

sys.path.insert(0, str(CONTROL))
os.chdir(str(CONTROL))

import glob  # noqa: E402
import numpy as np  # noqa: E402

NAME = "curl_stats"
LOOP_MM = {"Path04": 8800.0, "Path07": 11800.0}
DRIVE_MM = 150.0
N_TRIM = 6


def fit(rot, dyaw):
    """Trimmed fit of dyaw = gain * rot + curl."""
    x, y = rot, dyaw
    g = c = float("nan")
    for _ in range(N_TRIM):
        A = np.c_[x, np.ones_like(x)]
        g, c = np.linalg.lstsq(A, y, rcond=None)[0]
        r = y - (g * x + c)
        keep = np.abs(r) < 3.0 * np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
        if keep.sum() < 50:
            break
        x, y = x[keep], y[keep]
    return float(g), float(c), int(len(x))


def main():
    rows = []
    for d in sorted(glob.glob(f"{POLICY_RUNS}/Paths/default_Path0*")):
        a = np.genfromtxt(os.path.join(d, "step_metrics.tsv"), delimiter="\t",
                          names=True)
        yaw, rot = a["yaw_deg"], a["rot_deg"]
        dyaw = (np.diff(yaw) + 180.0) % 360.0 - 180.0
        m = np.isfinite(dyaw) & np.isfinite(rot[:-1])
        gain, curl, n = fit(rot[:-1][m], dyaw[m])
        arena = "Path04" if "Path04" in d else "Path07"
        steps_per_lap = LOOP_MM[arena] / DRIVE_MM
        rows.append(dict(run=os.path.basename(d), arena=arena, n_kept=n,
                         rot_gain=gain, curl_deg_per_step=curl,
                         curl_deg_per_lap=curl * steps_per_lap,
                         steps_per_lap=steps_per_lap))

    print(f"{'run':<40} {'gain':>6} {'curl °/step':>12} {'°/lap':>8}")
    for r in rows:
        print(f"{r['run']:<40} {r['rot_gain']:>6.3f} "
              f"{r['curl_deg_per_step']:>+12.2f} {r['curl_deg_per_lap']:>+8.0f}")
    for arena in ("Path04", "Path07"):
        v = [r["curl_deg_per_step"] for r in rows if r["arena"] == arena]
        print(f"{arena}: curl {min(v):+.2f} to {max(v):+.2f} °/step "
              f"over {len(v)} runs")
    with open(SCRIPTS / f"{NAME}.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
