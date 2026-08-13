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
# The pole channels. Added because the path-following controller now receives
# the same local feature as Experiment 1, poles included, rather than wall
# distances alone. The class posteriors GATE the geometry: when the sensor is
# not reporting a pole, pole_az and pole_dist carry nothing, so they are zeroed
# and p_pole tells the policy to disregard them. That keeps the vector
# fixed-size, so the network needs no branching on the perceived class.
_OBS_CLASS = [
    "p_wall",
    "p_pole",
    "p_none",
]
_OBS_POLE = [
    "pole_az_deg_norm",
    "pole_dist_mm_norm",
]
_OBS_POLE_SIGMA = [
    "pole_az_sigma_deg_norm",
    "pole_dist_sigma_mm_norm",
]
# Class-agnostic nearest-reflector range. The one channel that stays honest
# past ~1.4 m: distance is monaural time-of-flight and needs no classification,
# so it tracks to 2.5 m at ~11% error, where class and azimuth are at chance
# (Performance notes 2026-08-12 night). The pole range channel above saturates
# at ~750 mm by design, being masked to 1 m as the terminal-stop signal, so
# without this the policy has NO usable distance beyond a metre.
_OBS_AGN = [
    "agn_dist_mm_norm",
]
_OBS_AGN_SIGMA = [
    "agn_dist_sigma_mm_norm",
]

_OBS_TAIL = ["prev_rot_deg_norm"]


def make_obs_layout(use_sigma: bool, blind: bool = False,
                    use_poles: bool = True, use_agn: bool = False) -> List[str]:
    """Canonical obs-vector channel names. Train and deploy must agree.

    `use_poles=False` drops the class and pole channels, leaving the pre-2026-08
    wall-only vector. Kept as an ablation: the pole channel may be worth little
    on a given path, and the ablation is how that gets established rather than
    assumed.

    `use_agn=True` adds the class-agnostic nearest-range channel (+ its σ when
    `use_sigma`). Defaults False so every policy trained before 2026-08-13
    still loads and validates at its recorded width.

    `blind=True` strips all sonar channels (distances and σs) — the policy
    sees only `prev_rot`. Used as a control condition: with motor noise
    injection the blind policy cannot use sonar feedback to compensate, so
    poor performance is evidence that downstream success is sonar-driven
    rather than dead-reckoning. `blind` overrides `use_sigma`.
    """
    if blind:
        return list(_OBS_TAIL)
    obs = list(_OBS_BASE) + (list(_OBS_SIGMA) if use_sigma else [])
    if use_poles:
        obs += list(_OBS_CLASS) + list(_OBS_POLE)
        if use_sigma:
            obs += list(_OBS_POLE_SIGMA)
    if use_agn:
        obs += list(_OBS_AGN)
        if use_sigma:
            obs += list(_OBS_AGN_SIGMA)
    return obs + list(_OBS_TAIL)


# ── Stateless obs encoder ────────────────────────────────────────────────────

def encode_obs(
    meas_dict: Dict[str, Union[float, np.floating]] | None,
    prev_rot: float,
    *,
    min_dist_mm: float,
    max_dist_mm: float,
    max_sigma_mm: float,
    max_rotate_deg: float,
    use_sigma: bool,
    blind: bool = False,
    use_poles: bool = True,
    use_agn: bool = False,
    cone_half_deg: float = 35.0,
) -> np.ndarray:
    """Pack a 6-key sonar measurement + previous rotation into the policy obs.

    Distances are clamped to [min_dist_mm, max_dist_mm] and divided by
    max_dist_mm. σs are clamped to [0, max_sigma_mm] and divided by
    max_sigma_mm. prev_rot is divided by max_rotate_deg.

    Pole azimuth is normalised by the cone half-angle and range by max_dist_mm.
    Both are zeroed when the sensor reports no pole; the class posteriors carry
    that information, so the policy can learn to disregard the geometry rather
    than being handed a sentinel it might read as a measurement.

    Widths: 1 blind; otherwise 3 distances (+3 σs) (+3 class +2 pole
    (+2 pole σs)) (+1 agn (+1 agn σ)) +1 prev_rot. So 4 / 7 wall-only,
    9 / 14 with poles, 10 / 16 with poles and the agnostic range.
    In blind mode `meas_dict` is unused and may be None.
    """
    if blind:
        return np.asarray([float(prev_rot) / max_rotate_deg], dtype=np.float32)

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

    if use_poles:
        # Class posteriors first, then the pole geometry they gate. When no pole
        # is reported the geometry keys are NaN, so they are zeroed rather than
        # passed through: a NaN would poison the whole forward pass, and a
        # sentinel distance would be indistinguishable from a real reading.
        obs += [
            float(meas_dict.get("p_wall", 0.0)),
            float(meas_dict.get("p_pole", 0.0)),
            float(meas_dict.get("p_none", 0.0)),
        ]
        az = float(meas_dict.get("pole_az_deg", float("nan")))
        rg = float(meas_dict.get("pole_dist_mm", float("nan")))
        obs += [
            0.0 if not np.isfinite(az) else max(-1.0, min(az / cone_half_deg, 1.0)),
            0.0 if not np.isfinite(rg) else clamp_d(rg) / max_dist_mm,
        ]
        if use_sigma:
            azs = float(meas_dict.get("pole_az_sigma_deg", float("nan")))
            rgs = float(meas_dict.get("pole_dist_sigma_mm", float("nan")))
            obs += [
                0.0 if not np.isfinite(azs) else min(azs / cone_half_deg, 1.0),
                0.0 if not np.isfinite(rgs) else clamp_s(rgs) / max_sigma_mm,
            ]

    if use_agn:
        # Unlike pole_dist this is meaningful on every ping regardless of the
        # class posterior, so it is NOT gated. A missing key would mean the
        # inverse predates the head, which is a configuration error rather
        # than a "no reading" case -- but zero rather than NaN keeps a stale
        # artifact from poisoning the forward pass.
        ad = float(meas_dict.get("agn_dist_mm", float("nan")))
        obs.append(0.0 if not np.isfinite(ad) else clamp_d(ad) / max_dist_mm)
        if use_sigma:
            ads = float(meas_dict.get("agn_dist_sigma_mm", float("nan")))
            obs.append(0.0 if not np.isfinite(ads)
                       else clamp_s(ads) / max_sigma_mm)

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
    blind: bool = False,
    use_poles: bool = False,
    use_agn: bool = False,
    cone_half_deg: float = 35.0,
) -> dict:
    """Deploy-relevant subset of the saved policy JSON. SCRIPT_TrainPolicy
    .save_policy() adds training metadata (val_loss, epoch, ...) on top."""
    return {
        "policy_kind":    "vanilla_rnn",
        "hidden_size":    int(hidden_size),
        "in_dim":         int(in_dim),
        "out_dim":        int(out_dim),
        "use_sigma":      bool(use_sigma),
        "blind":          bool(blind),
        "use_poles":      bool(use_poles),
        "use_agn":        bool(use_agn),
        "cone_half_deg":  float(cone_half_deg),
        "obs_layout":     make_obs_layout(bool(use_sigma), bool(blind),
                                          bool(use_poles), bool(use_agn)),
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
        self.blind          = bool(params.get("blind", False))
        # Absent in policies trained before the pole channels existed, so the
        # default must be False -- otherwise every artifact on disk fails its
        # own in_dim check on load.
        self.use_poles      = bool(params.get("use_poles", False))
        # Same reasoning as use_poles: absent in every artifact trained before
        # 2026-08-13, so it must default False or those fail their in_dim check.
        self.use_agn        = bool(params.get("use_agn", False))
        self.cone_half_deg  = float(params.get("cone_half_deg", 35.0))
        self.max_rotate_deg = float(params["max_rotate_deg"])
        self.fixed_drive_mm = float(params["fixed_drive_mm"])
        self.min_dist_mm    = float(params["min_dist_mm"])
        self.max_dist_mm    = float(params["max_dist_mm"])
        self.max_sigma_mm   = float(params["max_sigma_mm"])
        self.obs_layout     = list(params["obs_layout"])

        expected_in = len(make_obs_layout(self.use_sigma, self.blind,
                                          self.use_poles, self.use_agn))
        if self.in_dim != expected_in:
            raise ValueError(
                f"in_dim={self.in_dim} inconsistent with "
                f"use_sigma={self.use_sigma}, blind={self.blind}, "
                f"use_poles={self.use_poles}, use_agn={self.use_agn} "
                f"(expected {expected_in})"
            )
        canonical_layout = make_obs_layout(self.use_sigma, self.blind,
                                           self.use_poles, self.use_agn)
        if self.obs_layout != canonical_layout:
            raise ValueError(
                f"obs_layout {self.obs_layout!r} does not match canonical "
                f"layout {canonical_layout!r} for use_sigma={self.use_sigma}, "
                f"blind={self.blind}, use_poles={self.use_poles}, "
                f"use_agn={self.use_agn}"
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
        meas_dict: Dict[str, Union[float, np.floating]] | None,
        prev_rot: float,
    ) -> np.ndarray:
        return encode_obs(
            meas_dict, prev_rot,
            min_dist_mm=self.min_dist_mm, max_dist_mm=self.max_dist_mm,
            max_sigma_mm=self.max_sigma_mm, max_rotate_deg=self.max_rotate_deg,
            use_sigma=self.use_sigma, blind=self.blind,
            use_poles=self.use_poles, use_agn=self.use_agn,
            cone_half_deg=self.cone_half_deg,
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
