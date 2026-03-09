"""
PolicyController — stateful wrapper for HistoryNNPolicy deployment on the real robot.

Mirrors Evaluator.episode() state management from SCRIPT_TrainPolicy.py, adapting it
for step-by-step real-time use instead of a fully simulated episode.

Ping ordering (DataAcquisition-style, one ping per step):
    ping → compute rotate1 (from last_iid) + rotate2 (from current ping) → rotate1 → rotate2 → drive
"""

import collections
import json
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(f):
        return default
    return f


# ---------------------------------------------------------------------------
# HistoryNNPolicy (copy from SCRIPT_TrainPolicy.py — kept self-contained so
# this module can be imported without pulling in the full training script)
# ---------------------------------------------------------------------------

class HistoryNNPolicy:
    """
    Two-head MLP with shared history encoder.

    BILATERAL SYMMETRY — CANONICAL FRAME DESIGN
    ============================================
    The network is trained exclusively in a canonical "wall-on-right" frame:
    IID values presented to the network are ALWAYS non-negative (wall on right
    or in deadband).  When the physical IID is negative (wall on left), both
    decide_rotate1 and decide_rotate2 flip the IID sign before running the
    network and negate the output rotation.  This gives correct physical
    behaviour on both wall sides without training on negative IID examples.

    IMPORTANT FOR DEPLOYMENT AND FUTURE READERS
    ============================================
    Do NOT pre-flip the IID before calling these methods.  Always pass the raw
    physical IID (can be negative).  The wrapper inside each method handles the
    flip transparently.  Removing or bypassing this wrapper will break bilateral
    symmetry.

    History features stored per step (always in canonical frame):
        [canonical_iid_norm, dist_norm, rot1_canonical_norm, rot2_canonical_norm,
         drive_norm, blocked]
    where canonical_iid = abs(physical_iid) and canonical_rotX = physical_rotX
    reflected back to the positive-IID frame.
    """

    def __init__(
        self,
        max_rotate1_deg: float,
        max_rotate2_deg: float,
        deadband_db: float,
        history_len: int,
        hidden_sizes: Tuple[int, int],
    ):
        self.max_rotate1_deg = float(max_rotate1_deg)
        self.max_rotate2_deg = float(max_rotate2_deg)
        self.deadband_db = float(deadband_db)
        self.history_len = int(history_len)
        self.hidden_sizes = tuple(int(v) for v in hidden_sizes)
        self.feature_dim = 6
        self.in_dim = self.history_len * self.feature_dim
        h1, h2 = self.hidden_sizes
        self.shapes = [
            (h1, self.in_dim), (h1,),   # shared encoder:       W1, b1
            (h2, h1),          (h2,),   # rot1 head hidden:     W2a, b2a
            (1,  h2),          (1,),    # rot1 head output:     W3a, b3a
            (h2, h1 + 2),      (h2,),   # rot2 head hidden:     W2b, b2b  (+2 = iid_n, dist_n)
            (1,  h2),          (1,),    # rot2 head output:     W3b, b3b
        ]
        self.params: List[np.ndarray] = [np.zeros(s, dtype=np.float32) for s in self.shapes]

    def genome_size(self) -> int:
        return int(sum(int(np.prod(s)) for s in self.shapes))

    def set_genome(self, genome: np.ndarray) -> None:
        g = np.asarray(genome, dtype=np.float32).reshape(-1)
        if g.size != self.genome_size():
            raise ValueError(f"HistoryNNPolicy genome must have length {self.genome_size()}")
        out: List[np.ndarray] = []
        off = 0
        for s in self.shapes:
            n = int(np.prod(s))
            out.append(g[off:off + n].reshape(s))
            off += n
        self.params = out

    def _shared_h1(self, hist_vec: np.ndarray) -> np.ndarray:
        x = np.asarray(hist_vec, dtype=np.float32).reshape(self.in_dim, 1)
        w1, b1 = self.params[0], self.params[1]
        return np.tanh(w1 @ x + b1.reshape(-1, 1))

    def decide_rotate1(self, hist_vec: np.ndarray, last_iid_db: float) -> float:
        """Head 1: decide where to look (before measuring).

        SYMMETRY WRAPPER: last_iid_db is the raw physical IID from the previous
        step.  If it was negative (wall on left last step), we assume the wall is
        still on the left, run the network in the canonical positive-IID frame,
        and negate the output so the head turns toward the correct physical side.
        Pass the raw physical IID — do NOT pre-flip.
        """
        phys_last = safe_float(last_iid_db, 0.0)
        flip = phys_last < 0.0
        canonical_last = abs(phys_last)
        if canonical_last < self.deadband_db:
            return 0.0
        h1 = self._shared_h1(hist_vec)
        w2a, b2a, w3a, b3a = self.params[2], self.params[3], self.params[4], self.params[5]
        h2 = np.tanh(w2a @ h1 + b2a.reshape(-1, 1))
        y = np.tanh(w3a @ h2 + b3a.reshape(-1, 1))
        rotate1_canonical = float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate1_deg
        return -rotate1_canonical if flip else rotate1_canonical

    def decide_rotate2(self, hist_vec: np.ndarray, current_iid_db: float, current_dist_mm: float) -> float:
        """Head 2: decide body turn after looking (current measurement injected).

        SYMMETRY WRAPPER: current_iid_db is the raw physical IID just measured.
        If negative (wall on left), we reflect to canonical positive-IID frame,
        run the network, and negate the output so the body turns the correct
        physical direction.  Pass the raw physical IID — do NOT pre-flip.
        """
        phys = safe_float(current_iid_db, 0.0)
        flip = phys < 0.0
        canonical_iid = abs(phys)
        if canonical_iid < self.deadband_db:
            return 0.0
        h1 = self._shared_h1(hist_vec)
        iid_n  = float(np.clip(canonical_iid / 12.0, 0.0, 2.0))
        dist_n = float(np.clip(safe_float(current_dist_mm, 1800.0) / 2000.0, 0.0, 2.0))
        h1_aug = np.concatenate([h1, np.array([[iid_n], [dist_n]], dtype=np.float32)], axis=0)
        w2b, b2b, w3b, b3b = self.params[6], self.params[7], self.params[8], self.params[9]
        h2 = np.tanh(w2b @ h1_aug + b2b.reshape(-1, 1))
        y = np.tanh(w3b @ h2 + b3b.reshape(-1, 1))
        rotate2_canonical = float(np.clip(y[0, 0], -1.0, 1.0)) * self.max_rotate2_deg
        return -rotate2_canonical if flip else rotate2_canonical


# ---------------------------------------------------------------------------
# load_policy helper
# ---------------------------------------------------------------------------

def load_policy(json_path: str) -> HistoryNNPolicy:
    """Load a HistoryNNPolicy from a best_policy.json file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    policy = HistoryNNPolicy(
        max_rotate1_deg=data["max_rotate1_deg"],
        max_rotate2_deg=data["max_rotate2_deg"],
        deadband_db=data["iid_deadband_db"],
        history_len=data["history_len"],
        hidden_sizes=data["hidden_sizes"],
    )
    genome = np.array(data["genome"], dtype=np.float32)
    expected = data.get("genome_size", policy.genome_size())
    if genome.size != expected:
        raise ValueError(
            f"Genome size mismatch: JSON has {genome.size} values, policy expects {expected}"
        )
    policy.set_genome(genome)
    return policy


# ---------------------------------------------------------------------------
# PolicyController
# ---------------------------------------------------------------------------

class PolicyController:
    """
    Stateful controller wrapping HistoryNNPolicy for real-time robot deployment.

    Mirrors the per-step state management from Evaluator.episode() in
    SCRIPT_TrainPolicy.py.  History is maintained in the canonical positive-IID
    frame exactly as during training, so the network sees the same feature
    distribution at inference time.

    Typical usage (DataAcquisition-style, one ping per step):

        ctrl = PolicyController(policy)
        ctrl.reset()
        for step in range(MAX_STEPS):
            sonar = client.read_and_process(do_ping=True)
            iid_db  = safe_float(sonar['corrected_iid'],      0.0)
            dist_mm = safe_float(sonar['corrected_distance'],  1.8) * 1000.0
            rotate1, rotate2, drive_mm = ctrl.step(iid_db, dist_mm)
            client.step(angle=rotate1);   time.sleep(0.15)
            client.step(angle=rotate2);   time.sleep(0.15)
            client.step(distance=drive_mm / 1000.0)
    """

    def __init__(self, policy: HistoryNNPolicy, fixed_drive_mm: float = 100.0):
        self.policy = policy
        self.fixed_drive_mm = float(fixed_drive_mm)
        # Internal state — initialised by reset()
        self.hist: collections.deque = collections.deque(maxlen=policy.history_len)
        self.last_physical_iid: float = 0.0
        self.prev_drive_norm: float = 0.0
        self.prev_blocked: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all history and reset state to episode start."""
        self.hist = collections.deque(maxlen=self.policy.history_len)
        self.last_physical_iid = 0.0
        self.prev_drive_norm   = 0.0
        self.prev_blocked      = 0.0

    def _build_hist_vec(self) -> np.ndarray:
        """Build the flat history vector with zero-padding for early steps."""
        if self.policy.history_len == 0:
            return np.zeros(0, dtype=np.float32)
        pad_n = self.policy.history_len - len(self.hist)
        if pad_n > 0:
            return np.concatenate(
                [np.zeros((pad_n * 6,), dtype=np.float32)] + list(self.hist), axis=0
            ).astype(np.float32)
        return np.concatenate(list(self.hist), axis=0).astype(np.float32)

    def compute_rotate1(self) -> float:
        """Compute head-1 rotation using last step's IID (no new ping needed).

        Call this before taking the current step's sonar ping.
        """
        hist_vec = self._build_hist_vec()
        return self.policy.decide_rotate1(hist_vec, self.last_physical_iid)

    def compute_rotate2(self, iid_db: float, distance_mm: float) -> float:
        """Compute head-2 rotation using the current sonar measurement.

        Call this after taking the current step's sonar ping.
        """
        hist_vec = self._build_hist_vec()
        return self.policy.decide_rotate2(hist_vec, iid_db, distance_mm)

    def update(
        self,
        rotate1: float,
        rotate2: float,
        iid_db: float,
        distance_mm: float,
        executed_drive_mm: float = None,
        blocked: bool = False,
    ) -> None:
        """Update canonical history after executing a step.

        Parameters
        ----------
        rotate1, rotate2 : float
            Actions that were commanded this step (degrees).
        iid_db : float
            Raw physical IID measured this step (dB).
        distance_mm : float
            Distance measured this step (mm).
        executed_drive_mm : float, optional
            Actual drive executed (mm).  Defaults to fixed_drive_mm.
        blocked : bool
            Whether the drive was blocked by a collision.
        """
        if executed_drive_mm is None:
            executed_drive_mm = self.fixed_drive_mm

        physical_iid  = safe_float(iid_db, 0.0)
        dist_mm_safe  = safe_float(distance_mm, 1800.0)
        flip          = physical_iid < 0.0
        canonical_iid = abs(physical_iid)
        canonical_rot1 = -rotate1 if flip else rotate1
        canonical_rot2 = -rotate2 if flip else rotate2

        canonical_iid_norm  = float(np.clip(canonical_iid  / 12.0,                       0.0, 2.0))
        dist_norm           = float(np.clip(dist_mm_safe   / 2000.0,                      0.0, 2.0))
        canonical_rot1_norm = float(np.clip(canonical_rot1 / max(self.policy.max_rotate1_deg, 1e-6), -1.0, 1.0))
        canonical_rot2_norm = float(np.clip(canonical_rot2 / max(self.policy.max_rotate2_deg, 1e-6), -1.0, 1.0))
        drive_norm          = float(np.clip(executed_drive_mm / max(self.fixed_drive_mm, 1e-6), 0.0, 1.5))

        self.hist.append(np.array(
            [canonical_iid_norm, dist_norm, canonical_rot1_norm, canonical_rot2_norm,
             drive_norm, 1.0 if blocked else 0.0],
            dtype=np.float32,
        ))
        self.last_physical_iid = physical_iid
        self.prev_drive_norm   = drive_norm
        self.prev_blocked      = 1.0 if blocked else 0.0

    def step(
        self,
        iid_db: float,
        distance_mm: float,
        executed_drive_mm: float = None,
        blocked: bool = False,
    ) -> Tuple[float, float, float]:
        """Convenience method: compute both actions and update history.

        Call this once per control cycle after the sonar ping.

        Returns
        -------
        (rotate1, rotate2, drive_mm) : Tuple[float, float, float]
            rotate1  — head-turn angle (degrees, + = right)
            rotate2  — body-turn angle (degrees, + = right)
            drive_mm — forward distance to drive (mm, always fixed_drive_mm)
        """
        rotate1 = self.compute_rotate1()
        rotate2 = self.compute_rotate2(iid_db, distance_mm)
        self.update(rotate1, rotate2, iid_db, distance_mm, executed_drive_mm, blocked)
        return rotate1, rotate2, self.fixed_drive_mm
