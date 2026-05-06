"""
Library/Policy.py

Single source of truth for the trained-policy artifact.

Loads the JSON written by SCRIPT_TrainPolicy.save_policy() and exposes the
same obs encoding + forward pass to both the training script (during student
rollouts and final save) and the deployment script (SCRIPT_RunPolicy.py).
This prevents the obs encoding (clamps, scales, channel order) from drifting
between train and deploy.

Top-level API:

    Policy.load(path) → Policy
    policy.encode_obs(meas_dict, prev_rot)  → np.ndarray, shape (in_dim,)
    policy.step(obs, hidden)                → (rotation_deg, new_hidden)
    policy.initial_hidden()                 → np.ndarray, shape (hidden_size,)

    encode_obs(...)        — stateless encoder, used at training time before a
                             policy artifact exists.
    make_obs_layout(...)   — canonical obs-vector channel names.
    make_policy_dict(...)  — the deploy-relevant subset of the saved JSON,
                             used by SCRIPT_TrainPolicy.save_policy() and by
                             rollout_student to build an in-memory Policy from
                             a live torch RNN.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple, Union

import numpy as np


# ── Canonical obs layout ─────────────────────────────────────────────────────

_OBS_BASE = [
    "distance_right_mm_norm",
    "distance_center_mm_norm",
    "distance_left_mm_norm",
]
_OBS_SIGMA = [
    "sigma_right_mm_norm",
    "sigma_center_mm_norm",
    "sigma_left_mm_norm",
]
_OBS_TAIL = ["prev_rot_deg_norm"]


def make_obs_layout(use_sigma: bool) -> List[str]:
    """Canonical obs-vector channel names. Train and deploy must agree."""
    return list(_OBS_BASE) + (list(_OBS_SIGMA) if use_sigma else []) + list(_OBS_TAIL)


# ── Stateless obs encoder ────────────────────────────────────────────────────

def encode_obs(
    meas_dict: Dict[str, Union[float, np.floating]],
    prev_rot: float,
    *,
    min_dist_mm: float,
    max_dist_mm: float,
    max_sigma_mm: float,
    max_rotate_deg: float,
    use_sigma: bool,
) -> np.ndarray:
    """Pack a 6-key sonar measurement + previous rotation into the policy obs.

    Distances are clamped to [min_dist_mm, max_dist_mm] and divided by
    max_dist_mm. σs are clamped to [0, max_sigma_mm] and divided by
    max_sigma_mm. prev_rot is divided by max_rotate_deg.

    Returns float32 array, shape (4,) when use_sigma=False, (7,) otherwise.
    """
    def clamp_d(v: Union[float, np.floating]) -> float:
        return max(min_dist_mm, min(float(v), max_dist_mm))

    def clamp_s(v: Union[float, np.floating]) -> float:
        return max(0.0, min(float(v), max_sigma_mm))

    obs: List[float] = [
        clamp_d(meas_dict["distance_right_mm"])  / max_dist_mm,
        clamp_d(meas_dict["distance_center_mm"]) / max_dist_mm,
        clamp_d(meas_dict["distance_left_mm"])   / max_dist_mm,
    ]
    if use_sigma:
        obs += [
            clamp_s(meas_dict["sigma_right_mm"])  / max_sigma_mm,
            clamp_s(meas_dict["sigma_center_mm"]) / max_sigma_mm,
            clamp_s(meas_dict["sigma_left_mm"])   / max_sigma_mm,
        ]
    obs.append(float(prev_rot) / max_rotate_deg)
    return np.asarray(obs, dtype=np.float32)


# ── Saved-artifact dict builder (deploy-relevant subset) ─────────────────────

def make_policy_dict(
    *,
    genome: np.ndarray,
    hidden_size: int,
    in_dim: int,
    out_dim: int,
    use_sigma: bool,
    max_rotate_deg: float,
    fixed_drive_mm: float,
    min_dist_mm: float,
    max_dist_mm: float,
    max_sigma_mm: float,
) -> dict:
    """Deploy-relevant subset of the saved policy JSON. SCRIPT_TrainPolicy
    .save_policy() adds training metadata (val_loss, epoch, ...) on top."""
    return {
        "policy_kind":    "vanilla_rnn",
        "hidden_size":    int(hidden_size),
        "in_dim":         int(in_dim),
        "out_dim":        int(out_dim),
        "use_sigma":      bool(use_sigma),
        "obs_layout":     make_obs_layout(bool(use_sigma)),
        "max_rotate_deg": float(max_rotate_deg),
        "fixed_drive_mm": float(fixed_drive_mm),
        "min_dist_mm":    float(min_dist_mm),
        "max_dist_mm":    float(max_dist_mm),
        "max_sigma_mm":   float(max_sigma_mm),
        "genome":         np.asarray(genome, dtype=np.float32).tolist(),
    }


# ── Loadable wrapper ─────────────────────────────────────────────────────────

class Policy:
    """Numpy wrapper for a trained vanilla-RNN policy artifact.

    The same forward pass that ran inside torch at training time, ported to
    numpy for inference. No torch import needed at deploy.
    """

    def __init__(self, params: dict):
        kind = params.get("policy_kind", "vanilla_rnn")
        if kind != "vanilla_rnn":
            raise ValueError(f"unsupported policy_kind: {kind!r}")

        self.hidden_size    = int(params["hidden_size"])
        self.in_dim         = int(params["in_dim"])
        self.out_dim        = int(params["out_dim"])
        self.use_sigma      = bool(params["use_sigma"])
        self.max_rotate_deg = float(params["max_rotate_deg"])
        self.fixed_drive_mm = float(params["fixed_drive_mm"])
        self.min_dist_mm    = float(params["min_dist_mm"])
        self.max_dist_mm    = float(params["max_dist_mm"])
        self.max_sigma_mm   = float(params["max_sigma_mm"])
        self.obs_layout     = list(params["obs_layout"])

        expected_in = 7 if self.use_sigma else 4
        if self.in_dim != expected_in:
            raise ValueError(
                f"in_dim={self.in_dim} inconsistent with use_sigma={self.use_sigma} "
                f"(expected {expected_in})"
            )
        canonical_layout = make_obs_layout(self.use_sigma)
        if self.obs_layout != canonical_layout:
            raise ValueError(
                f"obs_layout {self.obs_layout!r} does not match canonical "
                f"layout {canonical_layout!r} for use_sigma={self.use_sigma}"
            )

        # Unpack genome → numpy weights, in the same order as RNNNet.to_genome():
        # [W_xh (h, d), W_hh (h, h), b_h (h,), W_hy (o, h), b_y (o,)]
        h, d, o = self.hidden_size, self.in_dim, self.out_dim
        g = np.asarray(params["genome"], dtype=np.float32)
        expected_size = h * d + h * h + h + o * h + o
        if g.size != expected_size:
            raise ValueError(
                f"genome size {g.size} does not match expected {expected_size} "
                f"for hidden_size={h}, in_dim={d}, out_dim={o}"
            )
        i = 0
        self.W_xh = g[i:i + h * d].reshape(h, d).copy(); i += h * d
        self.W_hh = g[i:i + h * h].reshape(h, h).copy(); i += h * h
        self.b_h  = g[i:i + h].copy();                   i += h
        self.W_hy = g[i:i + o * h].reshape(o, h).copy(); i += o * h
        self.b_y  = g[i:i + o].copy()
        self.params = params

    @classmethod
    def load(cls, path: str) -> "Policy":
        with open(path) as f:
            return cls(json.load(f))

    def __repr__(self) -> str:
        return (f"Policy(kind=vanilla_rnn, hidden={self.hidden_size}, "
                f"in_dim={self.in_dim}, use_sigma={self.use_sigma}, "
                f"clamps=[{self.min_dist_mm:.0f}, {self.max_dist_mm:.0f}] mm, "
                f"σ_max={self.max_sigma_mm:.0f}, drive={self.fixed_drive_mm:.0f} mm)")

    def initial_hidden(self) -> np.ndarray:
        return np.zeros(self.hidden_size, dtype=np.float32)

    def encode_obs(
        self,
        meas_dict: Dict[str, Union[float, np.floating]],
        prev_rot: float,
    ) -> np.ndarray:
        return encode_obs(
            meas_dict, prev_rot,
            min_dist_mm=self.min_dist_mm, max_dist_mm=self.max_dist_mm,
            max_sigma_mm=self.max_sigma_mm, max_rotate_deg=self.max_rotate_deg,
            use_sigma=self.use_sigma,
        )

    def step(
        self,
        obs: np.ndarray,
        hidden: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """One RNN step. obs shape (in_dim,), hidden shape (hidden_size,).

        Returns (rotation_deg, new_hidden). Rotation is the network's tanh
        output × max_rotate_deg, so already in [-max_rotate_deg, max_rotate_deg].
        """
        x = np.asarray(obs,    dtype=np.float32)
        h = np.asarray(hidden, dtype=np.float32)
        h_new = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
        y = np.tanh(self.W_hy @ h_new + self.b_y) * self.max_rotate_deg
        return float(y[0]), h_new
