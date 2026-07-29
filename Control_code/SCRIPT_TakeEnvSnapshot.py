"""
SCRIPT_TakeEnvSnapshot.py

Capture an overhead snapshot of the current arena into DATA_FOLDER/<session>/,
together with a copy of the Settings.py that was in effect when it was taken.

The snapshot is the input to the annotate -> SCRIPT_BuildArenaGeometry.py step
that produces arena_features.npz.

Where the settings copy goes
----------------------------
Beside the snapshot, in <session>/files/Settings.copy.

This used to be written by a DataStorage.DataWriter, which resolves its root
from the global Settings.data_folder. Nothing set that here, so the copies
landed in the default 'SonarSessions/' -- a folder that held nothing else, whose
name predates the move of sonar training data to AcquisitionSessions/, and which
was nowhere near the snapshot it described.

Writing the copy directly also removes a live hazard: DataWriter(autoclear=True)
rmtree's its target folder before writing. Pointing its root at DATA_FOLDER --
the obvious way to co-locate the two -- would therefore have deleted
TempOutput/<session>/ on every run, taking any previously built arena with it.
For session = 'StartPositionDigitization' that is the recorded start poses, the
trial list and the walls-only build.
"""

import shutil
from pathlib import Path

from Library import LorexTracker
from LorexLib.Environment import capture_environment_layout

DATA_FOLDER = 'TempOutput'

session = 'DirectTarget02'
tracker = LorexTracker.LorexTracker()
snapshot = capture_environment_layout(save_root=f'{DATA_FOLDER}/{session}')

# Provenance: which Settings.py produced this snapshot. `.copy` rather than
# `.py` so it is never importable, matching the convention in PolicyRuns/.
files_dir = Path(DATA_FOLDER) / session / 'files'
files_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2('Library/Settings.py', files_dir / 'Settings.copy')
print(f"Snapshot -> {DATA_FOLDER}/{session}   (+ files/Settings.copy)")
