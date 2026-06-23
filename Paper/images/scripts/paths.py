"""Filesystem paths for the paper-figure scripts.

Everything is resolved from this file's own location, so the scripts work
regardless of the current working directory. Repo layout:

    3PiRobot/
      Control_code/            <- experimental code + (gitignored) outputs
      Paper/images/scripts/    <- this folder

Figure scripts read their data straight from Control_code/ and write the
finished figure up into Paper/images/ (where \\graphicspath looks).
"""

from pathlib import Path

_HERE = Path(__file__).resolve()
SCRIPTS = _HERE.parent          # Paper/images/scripts
IMAGES = _HERE.parents[1]       # Paper/images   <- standalone figures go here
IMAGE_RESOURCES = IMAGES / "image_resources"  # building-block assets for composite SVG figures
PAPER = _HERE.parents[2]        # Paper
ROOT = _HERE.parents[3]         # 3PiRobot
CONTROL = ROOT / "Control_code"

# Common data locations under Control_code. These outputs are gitignored and
# get overwritten between runs, so a figure reflects whatever was last produced
# there -- regenerate the figure after re-running the relevant experiment.
SONAR_MODEL = CONTROL / "SonarModel"          # inverse_cv_results.json, per-fold artifacts
POLICY_RUNS = CONTROL / "PolicyRuns"          # deployed trajectories
POLICY_TRAINING = CONTROL / "PolicyTraining"
ACQ_SESSIONS = CONTROL / "AcquisitionSessions"
ACQ_ARENAS = CONTROL / "AcquisitionArenas"

if not CONTROL.is_dir():
    print(f"[paths] warning: Control_code not found at {CONTROL}")
