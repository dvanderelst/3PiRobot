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

What this measures is the curl that SURVIVED correction. `Library/Client.py`
pre-compensates every drive by `-distance_mm * drive_yaw_curl_deg_per_mm`,
using a constant calibrated per session, and it does so after the policy has
issued its rotation. The logged `rot_deg` is the policy's command, so the fitted
`curl` is the part the calibration failed to remove. Each run archives the code
it ran under, so the constant is read back from that zip and the physical curl
reported alongside the residual: the three numbers are only interpretable
together.

The residual is what matters for the argument: it is unrelated to the commanded
rotation, so no amount of counting commands recovers it, and the policy is told
neither the correction nor the leftover.

    Control_code/.venv/bin/python3 Paper/images/scripts/curl_stats.py
"""

import json
import os
import sys

from paths import CONTROL, POLICY_RUNS, SCRIPTS

sys.path.insert(0, str(CONTROL))
os.chdir(str(CONTROL))

import glob  # noqa: E402
import re  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402

NAME = "curl_stats"
LOOP_MM = {"Path04": 8800.0, "Path07": 11800.0}
DRIVE_MM = 150.0
N_TRIM = 6
CAL_RE = re.compile(rb"drive_yaw_curl_deg_per_mm:\s*float\s*=\s*(-?[0-9.eE+]+)")


def calibrated_curl(run_dir):
    """The curl constant the run actually executed under, from its code zip."""
    zips = glob.glob(os.path.join(run_dir, "*.zip"))
    if not zips:
        return float("nan")
    with zipfile.ZipFile(zips[0]) as z:
        for name in z.namelist():
            if name.endswith("Settings.py"):
                m = CAL_RE.search(z.read(name))
                if m:
                    return float(m.group(1))
    return float("nan")


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
        # What Client.step added to every command, and therefore the physical
        # curl implied by what was left over.
        comp = -calibrated_curl(d) * DRIVE_MM
        rows.append(dict(run=os.path.basename(d), arena=arena, n_kept=n,
                         rot_gain=gain, curl_deg_per_step=curl,
                         curl_deg_per_lap=curl * steps_per_lap,
                         correction_deg_per_step=comp,
                         physical_curl_deg_per_step=curl - gain * comp,
                         steps_per_lap=steps_per_lap))

    print(f"{'run':<40} {'gain':>6} {'corr':>7} {'resid':>7} {'raw':>7} "
          f"{'resid °/lap':>12}")
    for r in rows:
        print(f"{r['run']:<40} {r['rot_gain']:>6.3f} "
              f"{r['correction_deg_per_step']:>+7.2f} "
              f"{r['curl_deg_per_step']:>+7.2f} "
              f"{r['physical_curl_deg_per_step']:>+7.2f} "
              f"{r['curl_deg_per_lap']:>+12.0f}")
    res = [r["curl_deg_per_step"] for r in rows]
    raw = [r["physical_curl_deg_per_step"] for r in rows]
    lap = [abs(r["curl_deg_per_lap"]) for r in rows]
    print(f"\nall {len(rows)} runs: physical curl {min(raw):+.2f} to "
          f"{max(raw):+.2f} °/step; residual after correction {min(res):+.2f} "
          f"to {max(res):+.2f} °/step; largest residual {max(lap):.0f} °/lap")
    with open(SCRIPTS / f"{NAME}.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
