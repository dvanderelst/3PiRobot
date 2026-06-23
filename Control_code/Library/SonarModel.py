"""
Library/SonarModel.py

Single source of truth for the sonar feature pipeline.

Loads the trained 3-slice (mean, σ) wall-only model (`slices_*` artifacts
from the retired SCRIPT_TrainSonarModel.py, recoverable from git history)
and exposes two interfaces:

    predict_from_envelope(L, R)        # real robot — sonar -> NN
    predict_from_profile(profile, rng) # simulator — geometry + σ_sim noise

Both return the same 6-key dict per sample:

    {distance_right_mm, distance_center_mm, distance_left_mm,
     sigma_right_mm,    sigma_center_mm,    sigma_left_mm}

Slice labels follow the project azimuth convention (`+az = LEFT`):
"right" covers the negative-azimuth bins (robot's physical right), "left"
covers the positive-azimuth bins (robot's physical left).

All sonar-related parameters (cone width, profile geometry,
profile_method, slice boundaries, normalization, σ_sim noise model)
live in the saved feature_params.json. Retrain the model with different
settings and downstream consumers (EnvironmentSimulator, training and
deployment scripts) automatically pick them up via this class.
"""

import json
import os
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn


SLICE_NAMES = ("right", "center", "left")  # ordered by ascending azimuth bin index: index 0 = most-negative az = robot's physical right


def normalize_envelope_per_ping(env, out_min: float = 0.0, out_max: float = 1.0,
                                ref_window: int = 10):
    """Per-ping min-max normalisation along the time (samples) axis, with the
    *upper* reference taken from the first `ref_window` samples — the emit
    pulse region — rather than from the full envelope.

    Why anchor on the emit pulse: in some pings (close, strongly reflective
    wall) a wall echo in the post-emit region can exceed the emit-pulse peak
    in amplitude. If we normalised by the global max, those pings would scale
    differently from "normal" pings (whose max is the emit pulse), giving the
    conv stack inconsistent feature scales. Using the emit-pulse max as the
    upper reference keeps the scale tied to a signal that is always present
    and (within a session) approximately stable; post-emit echoes then live
    on a scale proportional to the emit drive level, which is exactly what
    we want for cross-session gain invariance. Echoes that exceed the emit
    pulse simply produce normalised values > out_max, which the conv stack
    can handle as long as such pings are also represented at training time.

    The lower reference is the *global* min (the noise floor), which sits
    well below the emit pulse and is independent of `ref_window`.

    Removes absolute amplitude as a cue — the conv stack sees envelope shape
    only — so the model is robust to between-session gain drift (battery
    state, transducer wear, mic gain). Applied identically in training (data
    prep before the z-score) and at inference (`predict_from_envelope`); the
    chosen mode + window are recorded in `slices_feature_params.json` under
    `envelope_norm` so train/deploy can't drift apart.

    Accepts shape (T,), (N, T), or (N, T, C). Normalisation is per-row over
    the T axis: each (sample-stream-by-channel) is rescaled independently.
    `ref_window <= 0` falls back to global-max behaviour."""
    e = np.asarray(env, dtype=np.float32)
    span = float(out_max - out_min)
    if e.ndim == 1:
        T = e.shape[0]
        w = T if (ref_window is None or ref_window <= 0) else min(int(ref_window), T)
        hi = float(e[:w].max())
        lo = float(e.min())
        if hi - lo < 1e-6:
            return np.full_like(e, out_min, dtype=np.float32)
        return ((e - lo) / (hi - lo) * span + out_min).astype(np.float32)
    if e.ndim == 2:
        T = e.shape[1]
        w = T if (ref_window is None or ref_window <= 0) else min(int(ref_window), T)
        hi = e[:, :w].max(axis=1, keepdims=True)
        lo = e.min(axis=1, keepdims=True)
        rng = np.maximum(hi - lo, 1e-6)
        return ((e - lo) / rng * span + out_min).astype(np.float32)
    if e.ndim == 3:
        T = e.shape[1]
        w = T if (ref_window is None or ref_window <= 0) else min(int(ref_window), T)
        hi = e[:, :w, :].max(axis=1, keepdims=True)
        lo = e.min(axis=1, keepdims=True)
        rng = np.maximum(hi - lo, 1e-6)
        return ((e - lo) / rng * span + out_min).astype(np.float32)
    raise ValueError(f"normalize_envelope_per_ping: unexpected shape {e.shape}")


def _maybe_normalize_envelope(env, params: dict):
    """Apply the envelope normalisation specified in `params['envelope_norm']`,
    or return the input unchanged if no spec is present (legacy compatibility
    with feature_params.json files written before this field existed)."""
    spec = params.get("envelope_norm")
    if not spec:
        return env
    kind = spec.get("kind", "")
    if kind == "per_ping_minmax":
        return normalize_envelope_per_ping(
            env,
            out_min=float(spec.get("out_min", 0.0)),
            out_max=float(spec.get("out_max", 1.0)),
            ref_window=int(spec.get("ref_window", 10)),
        )
    if kind == "" or kind == "none":
        return env
    raise ValueError(f"unknown envelope_norm kind {kind!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Model architecture (wall-only; SonarSlicesUQ_TwoHeaded in this module
# extends it for the inverse trainer in SCRIPT_TrainInverseModel.py)
# ══════════════════════════════════════════════════════════════════════════════

class SonarSlicesUQ(nn.Module):
    """
    Heteroscedastic 3-slice head with built-in L/R symmetry.

      center_mean    = head((z_L + z_R) / 2)               (symmetric)
      center_log_var = head((z_L + z_R) / 2)               (symmetric)
      right_mean     = side_mean_head(concat(z_L, z_R))    (swap-equivariant)
      left_mean      = side_mean_head(concat(z_R, z_L))      with right
      right_log_var  = side_log_var_head(concat(z_L, z_R))
      left_log_var   = side_log_var_head(concat(z_R, z_L))

    Swapping the L and R sonar channels exactly swaps the left and right
    predictions and leaves the center prediction unchanged.

    Output dict keys follow SLICE_NAMES (project +az = LEFT convention):
    the "right" head is fed by concat(z_L, z_R) and is trained against
    the negative-azimuth slice (mask[0]); the "left" head is fed by
    concat(z_R, z_L) and trained against the positive-azimuth slice
    (mask[2]). The shared `side_*_head` modules and saved checkpoints
    are unchanged by this relabel — only the dict-key strings move.
    """
    def __init__(self, samples, conv_channels, conv_kernel, pool_out,
                 fc_hidden, head_hidden):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, conv_kernel, padding=conv_kernel // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.pool    = nn.AdaptiveAvgPool1d(pool_out)
        feat_dim     = conv_channels[-1] * pool_out
        self.fc      = nn.Sequential(nn.Linear(feat_dim, fc_hidden), nn.ReLU())

        def make_head(in_dim, hidden):
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )
        self.center_mean_head    = make_head(fc_hidden, head_hidden)
        self.center_log_var_head = make_head(fc_hidden, head_hidden)
        self.side_mean_head      = make_head(2 * fc_hidden, head_hidden)
        self.side_log_var_head   = make_head(2 * fc_hidden, head_hidden)

    def _embed(self, ch):
        z = self.encoder(ch.unsqueeze(1))
        z = self.pool(z).flatten(1)
        return self.fc(z)

    def forward(self, left, right):
        zL = self._embed(left)
        zR = self._embed(right)
        z_sym = 0.5 * (zL + zR)
        L_in = torch.cat([zL, zR], dim=-1)
        R_in = torch.cat([zR, zL], dim=-1)
        return {
            "right_mean":     self.side_mean_head(L_in),
            "right_log_var":  self.side_log_var_head(L_in),
            "center_mean":    self.center_mean_head(z_sym),
            "center_log_var": self.center_log_var_head(z_sym),
            "left_mean":      self.side_mean_head(R_in),
            "left_log_var":   self.side_log_var_head(R_in),
        }


class SonarSlicesUQ_TwoHeaded(nn.Module):
    """
    Two-headed inverse: SonarSlicesUQ's wall 3-slice (mean, log_var) heads
    plus a wall/pole class head and a pole-azimuth (mean, log_var) head.

    Shares the same shared trunk (encoder + pool + fc) as SonarSlicesUQ, so
    when poles aren't present in training data this reduces to the same
    expressive power as the wall-only model.

    Symmetry under L↔R sonar swap, enforced by construction (not by data
    augmentation):

      Wall heads (same as SonarSlicesUQ):
        center is symmetric (uses (zL+zR)/2),
        left/right share weights with channels swapped → exact L↔R flip.

      Class head: symmetric — wall vs pole label is invariant to which side
      the object is on. We achieve invariance by averaging the head's output
      on (L,R) and (R,L) inputs. The head can still read the asymmetry that
      discriminates pole from wall (a small pole gives strong L/R imbalance,
      a fronto-parallel wall does not) because it sees both orderings.

      Pole azimuth mean: antisymmetric — swapping L↔R flips the pole's side
      and therefore the sign of its bearing. Constructed as
      0.5·(head(L,R) − head(R,L)).

      Pole azimuth log_var: symmetric — uncertainty about bearing should not
      depend on which side. Constructed as 0.5·(head(L,R) + head(R,L)).

    `class_logits` has shape (B, 2); apply softmax/CE downstream.
    `pole_az_mean` / `pole_az_log_var` are in the same normalised space the
    trainer was set up with; trainers usually divide pole azimuth by
    cone_half_deg so the model outputs roughly [-1, 1].
    """
    def __init__(self, samples, conv_channels, conv_kernel, pool_out,
                 fc_hidden, head_hidden, n_classes: int = 2):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, conv_kernel, padding=conv_kernel // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.pool    = nn.AdaptiveAvgPool1d(pool_out)
        feat_dim     = conv_channels[-1] * pool_out
        self.fc      = nn.Sequential(nn.Linear(feat_dim, fc_hidden), nn.ReLU())

        def make_head(in_dim, hidden, out_dim=1):
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
            )

        # Wall slice heads (mirror SonarSlicesUQ)
        self.center_mean_head    = make_head(fc_hidden, head_hidden)
        self.center_log_var_head = make_head(fc_hidden, head_hidden)
        self.side_mean_head      = make_head(2 * fc_hidden, head_hidden)
        self.side_log_var_head   = make_head(2 * fc_hidden, head_hidden)

        # Class head (wall vs pole)
        self.class_head          = make_head(2 * fc_hidden, head_hidden, out_dim=n_classes)

        # Pole-azimuth heads
        self.pole_az_mean_head    = make_head(2 * fc_hidden, head_hidden)
        self.pole_az_log_var_head = make_head(2 * fc_hidden, head_hidden)

        self.n_classes = n_classes

    def _embed(self, ch):
        z = self.encoder(ch.unsqueeze(1))
        z = self.pool(z).flatten(1)
        return self.fc(z)

    def forward(self, left, right):
        zL = self._embed(left)
        zR = self._embed(right)
        z_sym = 0.5 * (zL + zR)
        LR = torch.cat([zL, zR], dim=-1)
        RL = torch.cat([zR, zL], dim=-1)

        # Class — symmetric averaging
        class_logits = 0.5 * (self.class_head(LR) + self.class_head(RL))

        # Pole azimuth — antisymmetric mean, symmetric log_var
        pole_az_mean    = 0.5 * (self.pole_az_mean_head(LR)    - self.pole_az_mean_head(RL))
        pole_az_log_var = 0.5 * (self.pole_az_log_var_head(LR) + self.pole_az_log_var_head(RL))

        return {
            "right_mean":      self.side_mean_head(LR),
            "right_log_var":   self.side_log_var_head(LR),
            "center_mean":     self.center_mean_head(z_sym),
            "center_log_var":  self.center_log_var_head(z_sym),
            "left_mean":       self.side_mean_head(RL),
            "left_log_var":    self.side_log_var_head(RL),
            "class_logits":    class_logits,       # (B, n_classes)
            "pole_az_mean":    pole_az_mean,       # (B, 1) in normalised space
            "pole_az_log_var": pole_az_log_var,    # (B, 1)
        }


class SonarSlicesUQ_Wall3(nn.Module):
    """Architecture experiment: one 3-output wall head instead of the
    side-head + center-head (+ z_sym) arrangement of SonarSlicesUQ_TwoHeaded.
    Emits the same output dict, so it is a drop-in for the trainer.

    symmetric=True  (B): the wall head is run on both ear orderings (LR, RL) and
                         the outputs combined -- left/right swap, center averaged
                         -- so mirror symmetry is enforced and all three
                         distances use the full binaural signal (no z_sym).
    symmetric=False (A): all outputs are read from LR only; no symmetry is
                         enforced, so the model can learn left/right asymmetries.
    """

    def __init__(self, samples, conv_channels, conv_kernel, pool_out,
                 fc_hidden, head_hidden, n_classes: int = 2, symmetric: bool = True):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, conv_kernel, padding=conv_kernel // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(pool_out)
        feat_dim = conv_channels[-1] * pool_out
        self.fc = nn.Sequential(nn.Linear(feat_dim, fc_hidden), nn.ReLU())

        def make_head(in_dim, hidden, out_dim=1):
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
            )

        self.wall_mean_head = make_head(2 * fc_hidden, head_hidden, out_dim=3)
        self.wall_log_var_head = make_head(2 * fc_hidden, head_hidden, out_dim=3)
        self.class_head = make_head(2 * fc_hidden, head_hidden, out_dim=n_classes)
        self.pole_az_mean_head = make_head(2 * fc_hidden, head_hidden)
        self.pole_az_log_var_head = make_head(2 * fc_hidden, head_hidden)
        self.n_classes = n_classes
        self.symmetric = symmetric

    def _embed(self, ch):
        z = self.encoder(ch.unsqueeze(1))
        z = self.pool(z).flatten(1)
        return self.fc(z)

    def forward(self, left, right):
        zL = self._embed(left)
        zR = self._embed(right)
        LR = torch.cat([zL, zR], dim=-1)
        RL = torch.cat([zR, zL], dim=-1)

        def col(t, i):
            return t[:, i:i + 1]

        if self.symmetric:
            mLR, mRL = self.wall_mean_head(LR), self.wall_mean_head(RL)
            vLR, vRL = self.wall_log_var_head(LR), self.wall_log_var_head(RL)
            # head columns are [left, center, right]; under the LR<->RL swap the
            # mirror's left is the true right and vice versa.
            left_mean = 0.5 * (col(mLR, 0) + col(mRL, 2))
            left_log_var = 0.5 * (col(vLR, 0) + col(vRL, 2))
            center_mean = 0.5 * (col(mLR, 1) + col(mRL, 1))
            center_log_var = 0.5 * (col(vLR, 1) + col(vRL, 1))
            right_mean = 0.5 * (col(mLR, 2) + col(mRL, 0))
            right_log_var = 0.5 * (col(vLR, 2) + col(vRL, 0))
            class_logits = 0.5 * (self.class_head(LR) + self.class_head(RL))
            pole_az_mean = 0.5 * (self.pole_az_mean_head(LR) - self.pole_az_mean_head(RL))
            pole_az_log_var = 0.5 * (self.pole_az_log_var_head(LR) + self.pole_az_log_var_head(RL))
        else:
            m, v = self.wall_mean_head(LR), self.wall_log_var_head(LR)
            left_mean, center_mean, right_mean = col(m, 0), col(m, 1), col(m, 2)
            left_log_var, center_log_var, right_log_var = col(v, 0), col(v, 1), col(v, 2)
            class_logits = self.class_head(LR)
            pole_az_mean = self.pole_az_mean_head(LR)
            pole_az_log_var = self.pole_az_log_var_head(LR)

        return {
            "right_mean": right_mean, "right_log_var": right_log_var,
            "center_mean": center_mean, "center_log_var": center_log_var,
            "left_mean": left_mean, "left_log_var": left_log_var,
            "class_logits": class_logits,
            "pole_az_mean": pole_az_mean,
            "pole_az_log_var": pole_az_log_var,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Loadable wrapper with sim and deploy interfaces
# ══════════════════════════════════════════════════════════════════════════════

class SonarModel:
    SLICE_NAMES = SLICE_NAMES

    def __init__(self, model: nn.Module, params: dict, device: torch.device):
        self.model  = model.to(device).eval()
        self.params = params
        self.device = device

        # Cache geometry derived from saved profile params
        self.bin_centers = self._compute_bin_centers()
        self.slice_masks = self._compute_slice_masks()

        # Cache normalisation stats for fast inference
        self._s_mean = float(params["sonar_norm"]["mean"])
        self._s_std  = float(params["sonar_norm"]["std"])
        self._t_mean = float(params["target_norm"]["mean"])
        self._t_std  = float(params["target_norm"]["std"])
        self._lv_min, self._lv_max = params["log_var_clamp"]

        # Cache empirical σ_sim lookup (per-slice bin centres + bin σs from val).
        # Used by sigma_sim() for the simulator's noise model — see method docstring.
        self._sigma_bin_centers = {
            name: np.asarray(
                params["sigma_sim_per_slice"][name]["empirical"]["bin_centers_mm"],
                dtype=np.float64)
            for name in SLICE_NAMES
        }
        self._sigma_bin_sigmas = {
            name: np.asarray(
                params["sigma_sim_per_slice"][name]["empirical"]["bin_sigmas_mm"],
                dtype=np.float64)
            for name in SLICE_NAMES
        }

    def __repr__(self):
        pp = self.params["profile"]
        return (f"SonarModel(cone=±{self.params['cone_half_deg']:.0f}°, "
                f"profile={pp['opening_angle']:.0f}°/{pp['profile_steps']} bins"
                f" [{pp['profile_method']}], device={self.device})")

    @classmethod
    def load(cls, model_dir: str = "SonarModel",
             device: Optional[str] = None) -> "SonarModel":
        params_path  = os.path.join(model_dir, "slices_feature_params.json")
        weights_path = os.path.join(model_dir, "slices_best_model.pth")
        if not os.path.exists(params_path):
            raise FileNotFoundError(
                f"feature_params not found at {params_path}. "
                "SonarModel.load expects wall-only `slices_*` artifacts; "
                "the wall-only trainer SCRIPT_TrainSonarModel.py was retired "
                "(recoverable from git). The current SCRIPT_TrainInverseModel.py "
                "writes `inverse_*` artifacts in a different shape that this "
                "loader does not yet consume."
            )
        with open(params_path) as f:
            params = json.load(f)

        arch = params["architecture"]
        model = SonarSlicesUQ(
            samples=int(arch["samples"]),
            conv_channels=list(arch["conv_channels"]),
            conv_kernel=int(arch["conv_kernel"]),
            pool_out=int(arch["pool_out"]),
            fc_hidden=int(arch["fc_hidden"]),
            head_hidden=int(arch["head_hidden"]),
        )
        target_device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, params, target_device)

    # ── Parameter accessors (consumed by the simulator) ──────────────────────

    def get_profile_params(self) -> dict:
        """Profile geometry the model was trained with. The simulator should
        compute its profiles using exactly these settings."""
        pp = self.params["profile"]
        return {
            "opening_angle":  float(pp["opening_angle"]),
            "profile_steps":  int(pp["profile_steps"]),
            "profile_method": pp["profile_method"],
        }

    def get_cone_half_deg(self) -> float:
        return float(self.params["cone_half_deg"])

    def get_slice_definitions(self) -> dict:
        return dict(self.params["slice_definitions"])

    # ── Geometry helpers ─────────────────────────────────────────────────────

    def _compute_bin_centers(self) -> np.ndarray:
        pp = self.params["profile"]
        edges = np.linspace(-pp["opening_angle"] / 2.0, pp["opening_angle"] / 2.0,
                            pp["profile_steps"] + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    def _compute_slice_masks(self):
        cone  = float(self.params["cone_half_deg"])
        third = 2.0 * cone / 3.0
        bc = self.bin_centers
        return [
            (bc >= -cone)              & (bc < -cone + third),
            (bc >= -cone + third)      & (bc < -cone + 2.0 * third),
            (bc >= -cone + 2.0 * third) & (bc <= cone),
        ]

    def sigma_sim(self, slice_name: str,
                  d_mm: Union[float, np.ndarray]) -> np.ndarray:
        """σ_sim(d) for one slice — linear interp through the per-bin empirical
        σs measured on the val split. Flat extrapolation outside the bin range
        (np.interp's default). Used by the simulator to add noise to geometric
        truth in predict_from_profile.

        Replaces an earlier parametric `floor + slope·max(0, d−knee)` fit; that
        fit landed on different shapes per slice (a real hinge for right, a flat
        ramp for left/center) and undershot the rightmost empirical σ in left
        and center by ~80–90 mm. Interpolating the bin σs directly removes that
        compromise — the noise model passes through the data points.
        """
        bc = self._sigma_bin_centers[slice_name]
        bs = self._sigma_bin_sigmas[slice_name]
        return np.interp(np.asarray(d_mm, dtype=np.float64), bc, bs)

    # ── Real-robot interface: sonar envelope -> (mean, σ) per slice ──────────

    def predict_from_envelope(self, left, right) -> Dict[str, Union[float, np.ndarray]]:
        """
        Args:
            left, right: numpy arrays of envelope samples, shape (T,) or (N, T).

        Returns dict with arrays of shape (N,) (or scalars if input was 1D):
            distance_{right,center,left}_mm  — model's mean prediction
            sigma_{right,center,left}_mm     — model's per-ping σ prediction
        """
        L = np.asarray(left,  dtype=np.float32)
        R = np.asarray(right, dtype=np.float32)
        squeeze = (L.ndim == 1)
        if squeeze:
            L = L[None, :]; R = R[None, :]

        # Per-ping per-channel envelope normalisation (no-op if the model's
        # feature_params.json has no `envelope_norm` block — legacy models).
        L = _maybe_normalize_envelope(L, self.params)
        R = _maybe_normalize_envelope(R, self.params)

        Ln = (L - self._s_mean) / self._s_std
        Rn = (R - self._s_mean) / self._s_std
        Lt = torch.as_tensor(Ln, dtype=torch.float32).to(self.device)
        Rt = torch.as_tensor(Rn, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out = self.model(Lt, Rt)

        result: Dict[str, Union[float, np.ndarray]] = {}
        for name in SLICE_NAMES:
            mean_n    = out[f"{name}_mean"].cpu().squeeze(1).numpy()
            log_var_n = out[f"{name}_log_var"].cpu().squeeze(1).numpy()
            log_var_n = np.clip(log_var_n, self._lv_min, self._lv_max)
            mean_mm   = mean_n * self._t_std + self._t_mean
            sigma_mm  = np.exp(log_var_n / 2.0) * self._t_std
            if squeeze:
                mean_mm  = float(mean_mm[0])
                sigma_mm = float(sigma_mm[0])
            result[f"distance_{name}_mm"] = mean_mm
            result[f"sigma_{name}_mm"]    = sigma_mm
        return result

    # ── Simulator interface: profile -> noisy observation per slice ──────────

    def predict_from_profile(self, profile,
                             rng: Optional[np.random.Generator] = None
                            ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Args:
            profile: numpy array of wall distances (mm), shape (K,) or (N, K).
                     K must equal the model's profile_steps.
            rng: numpy Generator. If None, uses np.random.default_rng().

        For each slice s, returns:
            distance_{s}_mm = true_min_in_slice + N(0, σ_sim_s(true_min_in_slice))
            sigma_{s}_mm    = σ_sim_s(true_min_in_slice)   (parametric)

        The σ here is the *marginal* σ_sim(d) (depends only on the geometric
        truth) — at deploy time the model's per-ping σ is more informative
        but at sim time we don't have a sonar input to condition on.
        """
        p = np.asarray(profile, dtype=np.float32)
        squeeze = (p.ndim == 1)
        if squeeze:
            p = p[None, :]

        K_expected = len(self.bin_centers)
        if p.shape[1] != K_expected:
            pp = self.params["profile"]
            raise ValueError(
                f"profile has {p.shape[1]} bins but model expects {K_expected} "
                f"(opening_angle={pp['opening_angle']}, "
                f"profile_steps={pp['profile_steps']})"
            )

        if rng is None:
            rng = np.random.default_rng()

        result: Dict[str, Union[float, np.ndarray]] = {}
        for i, name in enumerate(SLICE_NAMES):
            mask     = self.slice_masks[i]
            true_min = p[:, mask].min(axis=1).astype(np.float32)   # (N,)
            sigma    = self.sigma_sim(name, true_min).astype(np.float32)
            noise    = rng.normal(0.0, sigma).astype(np.float32)
            obs      = true_min + noise
            if squeeze:
                obs   = float(obs[0])
                sigma = float(sigma[0])
            result[f"distance_{name}_mm"] = obs
            result[f"sigma_{name}_mm"]    = sigma
        return result

    # ── Convenience: flatten dict to canonical policy-obs vector ─────────────

    @staticmethod
    def to_policy_obs(result: Dict[str, Union[float, np.ndarray]]) -> np.ndarray:
        """
        Pack a result dict into the canonical 6-element policy observation:
            [d_right, d_center, d_left, σ_right, σ_center, σ_left]
        Order matches SLICE_NAMES (ascending azimuth bin: physical right →
        center → physical left).
        Returns shape (6,) for scalar dicts, (N, 6) for batched dicts.
        """
        d = result["distance_right_mm"]
        if isinstance(d, (int, float)):
            return np.array([
                result["distance_right_mm"],  result["distance_center_mm"],
                result["distance_left_mm"],
                result["sigma_right_mm"],     result["sigma_center_mm"],
                result["sigma_left_mm"],
            ], dtype=np.float32)
        return np.stack([
            result["distance_right_mm"],  result["distance_center_mm"],
            result["distance_left_mm"],
            result["sigma_right_mm"],     result["sigma_center_mm"],
            result["sigma_left_mm"],
        ], axis=-1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Two-headed inverse: loadable inference wrapper
# ══════════════════════════════════════════════════════════════════════════════

class InverseModel:
    """Inference wrapper for the two-headed cross-modal inverse
    (`SonarSlicesUQ_TwoHeaded`), trained by SCRIPT_TrainInverseModel.py.

    Parallel to `SonarModel` (wall-only) but for the `inverse_*` artifacts.
    Exposes one entry point —

        predict_from_envelope(L, R)  →  unified local feature dict

    returning the class label, the wall 3-slice (mean, σ), and the pole
    azimuth (mean, σ) in one call. This is the sonar-side producer of the
    local feature the direct-/vicarious-learning policies consume; the
    vision-side producer is geometry (`nearest_reflector_in_cone` +
    `compute_profile`), and both emit the same {class; wall→slices; pole→az}
    shape by construction.

    Artifact layout (SCRIPT_TrainInverseModel.py writes per-CV-fold, not a
    single production model):
      - inverse_feature_params.json   shared: architecture, cone, pole-az
                                      normalisation, envelope_norm, log_var clamp
      - inverse_<fold>_results.json   per-fold: sonar_norm + target_norm stats
      - inverse_<fold>_best_model.pth per-fold weights

    `fold` selects which CV fold's weights + normalisation to load. There is
    no all-data production model yet (see the deploy-model TODO in handoff.md);
    any fold generalises at ~90% class accuracy, which is adequate for the
    direct-learning demo, but a dedicated no-holdout model is the right thing
    to deploy long-term.
    """

    SLICE_NAMES = SLICE_NAMES

    def __init__(self, model: nn.Module, params: dict, fold_stats: dict,
                 device: torch.device, fold: str):
        self.model  = model.to(device).eval()
        self.params = params
        self.device = device
        self.fold   = fold

        self.cone_half_deg   = float(params["cone_half_deg"])
        self.pole_az_divisor = float(params["pole_az_norm"]["divide_by_deg"])
        self.class_names     = list(params.get("class_names", ["wall", "pole"]))
        self._lv_min, self._lv_max = params["log_var_clamp"]

        # Per-fold normalisation (sonar z-score + wall-distance de-norm).
        self._s_mean = float(fold_stats["sonar_norm"]["mean"])
        self._s_std  = float(fold_stats["sonar_norm"]["std"])
        self._t_mean = float(fold_stats["target_norm"]["mean"])
        self._t_std  = float(fold_stats["target_norm"]["std"])

    def __repr__(self):
        return (f"InverseModel(fold={self.fold}, cone=±{self.cone_half_deg:.0f}°, "
                f"classes={self.class_names}, device={self.device})")

    @classmethod
    def load(cls, model_dir: str = "SonarModel", fold: str = "q0",
             device: Optional[str] = None) -> "InverseModel":
        params_path  = os.path.join(model_dir, "inverse_feature_params.json")
        fold_path    = os.path.join(model_dir, f"inverse_{fold}_results.json")
        weights_path = os.path.join(model_dir, f"inverse_{fold}_best_model.pth")
        for p in (params_path, fold_path, weights_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"InverseModel.load: missing {p}. Expected two-headed "
                    f"`inverse_*` artifacts (fold={fold!r}) from "
                    "SCRIPT_TrainInverseModel.py. Run that trainer first, or "
                    "pick a fold that exists (q0..q3 by default)."
                )
        with open(params_path) as f:
            params = json.load(f)
        with open(fold_path) as f:
            fold_stats = json.load(f)

        arch = params["architecture"]
        # Dispatch on the recorded architecture so B-trained (Wall3) and the
        # older base (TwoHeaded) checkpoints both load correctly. Default to
        # TwoHeaded for feature_params written before model_class existed.
        common = dict(
            samples=int(arch["samples"]),
            conv_channels=list(arch["conv_channels"]),
            conv_kernel=int(arch["conv_kernel"]),
            pool_out=int(arch["pool_out"]),
            fc_hidden=int(arch["fc_hidden"]),
            head_hidden=int(arch["head_hidden"]),
            n_classes=int(arch.get("n_classes", 2)),
        )
        model_class = arch.get("model_class", "SonarSlicesUQ_TwoHeaded")
        if model_class == "SonarSlicesUQ_Wall3":
            model = SonarSlicesUQ_Wall3(
                **common, symmetric=bool(arch.get("wall3_symmetric", True))
            )
        elif model_class == "SonarSlicesUQ_TwoHeaded":
            model = SonarSlicesUQ_TwoHeaded(**common)
        else:
            raise ValueError(
                f"InverseModel.load: unknown model_class {model_class!r} in "
                f"{params_path}"
            )
        target_device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, params, fold_stats, target_device, fold)

    def predict_from_envelope(self, left, right) -> Dict[str, Union[float, np.ndarray]]:
        """
        Args:
            left, right: numpy arrays of envelope samples, shape (T,) or (N, T).

        Returns dict (scalars if input was 1D, else (N,) arrays):
            class_label                int   0=wall, 1=pole, 2=none (argmax)
            p_pole                     float P(pole)
            p_none                     float P(none); present only for 3-class
                                       models (nothing within trained range)
            distance_{right,center,left}_mm   wall slice means (always emitted;
                                              only meaningful when class==wall)
            sigma_{right,center,left}_mm      wall slice σ
            pole_az_deg                signed bearing (+ccw = LEFT); only
                                       meaningful when class==pole
            pole_az_sigma_deg          bearing σ
        De-normalisation mirrors SCRIPT_TrainInverseModel.predict exactly.
        """
        L = np.asarray(left,  dtype=np.float32)
        R = np.asarray(right, dtype=np.float32)
        squeeze = (L.ndim == 1)
        if squeeze:
            L = L[None, :]; R = R[None, :]

        L = _maybe_normalize_envelope(L, self.params)
        R = _maybe_normalize_envelope(R, self.params)
        Ln = (L - self._s_mean) / self._s_std
        Rn = (R - self._s_mean) / self._s_std
        Lt = torch.as_tensor(Ln, dtype=torch.float32).to(self.device)
        Rt = torch.as_tensor(Rn, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out = self.model(Lt, Rt)

        result: Dict[str, Union[float, np.ndarray]] = {}

        # Wall slices (de-normalise mean + σ to mm).
        for name in SLICE_NAMES:
            mean_n    = out[f"{name}_mean"].cpu().squeeze(1).numpy()
            log_var_n = np.clip(out[f"{name}_log_var"].cpu().squeeze(1).numpy(),
                                self._lv_min, self._lv_max)
            mean_mm  = mean_n * self._t_std + self._t_mean
            sigma_mm = np.exp(log_var_n / 2.0) * self._t_std
            result[f"distance_{name}_mm"] = mean_mm
            result[f"sigma_{name}_mm"]    = sigma_mm

        # Class (softmax over logits).
        logits = out["class_logits"].cpu().numpy()
        probs  = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs  = probs / probs.sum(axis=1, keepdims=True)
        result["p_pole"]      = probs[:, 1]
        if probs.shape[1] >= 3:
            result["p_none"]  = probs[:, 2]
        result["class_label"] = probs.argmax(axis=1).astype(np.int64)

        # Pole azimuth (de-normalise by cone half-angle).
        az_n   = out["pole_az_mean"].cpu().squeeze(1).numpy()
        az_lvn = np.clip(out["pole_az_log_var"].cpu().squeeze(1).numpy(),
                         self._lv_min, self._lv_max)
        result["pole_az_deg"]       = az_n * self.pole_az_divisor
        result["pole_az_sigma_deg"] = np.exp(az_lvn / 2.0) * self.pole_az_divisor

        if squeeze:
            for k, v in result.items():
                arr = np.asarray(v)
                result[k] = int(arr[0]) if k == "class_label" else float(arr[0])
        return result
