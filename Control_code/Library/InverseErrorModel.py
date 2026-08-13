"""
InverseErrorModel — corrupt geometric truth the way the deployed inverse does.

The path-following simulator does not simulate echoes. It computes what the
robot would truly see at a pose and then degrades it, so a policy trained in
simulation meets the errors the real inverse makes. This class is that
degradation, driven by the tables `SCRIPT_FitInverseErrorModel.py` measures from
2135 labelled echoes.

It replaces the wall-only `SonarModel.predict_from_profile`, which added
`N(0, sigma_sim(d))` to three wall distances and knew nothing about classes or
poles. That model was retired without an archive, which is why the simulator
currently cannot start at all.

What it models
--------------
    class          full 3x3 confusion, conditioned on true class AND true range.
                   The sensor does not merely get noisy with distance -- it gets
                   the CATEGORY wrong, and does so most at the edge of its
                   operating range.
    wall slices    bias and sigma per slice, conditioned on true distance.
    pole azimuth   bias and sigma, conditioned on true range.
    pole range     bias and sigma, conditioned on true range.
    phantom poles  when the confusion model draws `pole` on a cone that holds
                   none, an azimuth and range are invented from the marginal
                   distribution of real false positives.

Why phantoms are generated rather than suppressed
-------------------------------------------------
15.7% of true-`none` echoes come back `pole`, and they are reported at
611 +/- 149 mm -- squarely inside the band where genuine poles are most
reliable. They cannot be filtered on geometry. A policy trained without them
would learn to trust the pole channels in a way deployment does not justify;
one trained with them can only reject them by their inconsistency across
successive looks, which is what the recurrent state is for.

Known simplification
--------------------
Draws are INDEPENDENT at each step. Real error is very likely correlated with
pose -- a particular corner seen from a particular angle will be misread the
same way each time. So simulated recovery from error is probably easier than
the robot's. This is a limitation of the simulation, not of the claim it
supports: the point being made is that a noisy inverse suffices, and the
training must therefore be done against a noisy one.
"""

import json
import os
from typing import Dict, Optional

import numpy as np

CLASS_NAMES = ["wall", "pole", "none"]
SLICE_NAMES = ["right", "center", "left"]


def _pick_bin(bins, value):
    """The bin covering `value`, else the nearest usable one.

    Falling back to the nearest bin rather than returning nothing matters: the
    simulator visits ranges the acquisition never sampled (the arena is metres
    across, the model was trained to 1 m), and a silent None there would
    propagate as a NaN observation into training.
    """
    usable = [b for b in bins if b.get("n", 0) > 0 and b.get("sigma") is not None]
    if not usable:
        return None
    for b in usable:
        if b["lo"] <= value < b["hi"]:
            return b
    return min(usable, key=lambda b: min(abs(value - b["lo"]), abs(value - b["hi"])))


def _pick_probs(rows, value):
    usable = [r for r in rows if r.get("n", 0) > 0 and r.get("p_pred")]
    if not usable:
        return None
    for r in usable:
        if r["lo"] <= value < r["hi"]:
            return r["p_pred"]
    return min(usable,
               key=lambda r: min(abs(value - r["lo"]), abs(value - r["hi"])))["p_pred"]


def _pick_samples(rows, value):
    """The stored per-ping posteriors for the bin containing `value`.

    Same nearest-bin fallback as `_pick_probs`, but the bins now span the full
    data range, so the fallback is a genuine edge case rather than the routine
    path it used to be for anything beyond 1 m.
    """
    usable = [r for r in rows if r.get("samples")]
    if not usable:
        return None
    for r in usable:
        if r["lo"] <= value < r["hi"]:
            return r["samples"]
    return min(usable,
               key=lambda r: min(abs(value - r["lo"]), abs(value - r["hi"])))["samples"]


class InverseErrorModel:
    """Turn the true local feature at a pose into what the inverse would report."""

    def __init__(self, tables: Dict):
        self.t = tables
        self.max_range_mm = float(tables["provenance"]["max_range_mm"])
        # The horizon the simulator should gate ground truth on. Distinct from
        # max_range_mm, which is the trainer's LABELLING cap and stopped
        # meaning "how far the sensor sees" when FAR_LABEL_MODE became
        # "true_class". Older error-model files carry only max_range_mm; using
        # it as a horizon made the simulated sensor blind past 1 m and made it
        # emit a clean p_none = 1.0 that the real inverse never produces.
        self.sensor_horizon_mm = float(
            tables["provenance"].get("sensor_horizon_mm", self.max_range_mm))
        self.cone_half_deg = float(tables["provenance"]["cone_half_deg"])

    @classmethod
    def load(cls, path: str = "SonarModel/inverse_error_model.json"
             ) -> "InverseErrorModel":
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Inverse error model not found at {path}. Run "
                f"SCRIPT_FitInverseErrorModel.py to measure it from the "
                f"acquisition echoes.")
        with open(path) as fh:
            return cls(json.load(fh))

    def __repr__(self) -> str:
        p = self.t["provenance"]
        return (f"InverseErrorModel(fold={p['inverse_fold']}, "
                f"n={p['n_echoes']}, horizon={self.sensor_horizon_mm:.0f}mm)")

    # ── the observation ─────────────────────────────────────────────────────

    def observe(self, true_cls: Optional[int], true_range_mm: float,
                true_pole_az_deg: float, true_slices_mm: Dict[str, float],
                rng: np.random.Generator) -> Dict:
        """One noisy reading of the local feature.

        Args:
            true_cls: 0 wall, 1 pole, 2 none (or None for an empty cone)
            true_range_mm: range to whichever reflector won the cone
            true_pole_az_deg: bearing to the pole, when one won
            true_slices_mm: geometric wall depth per slice
            rng: the simulator's generator, so runs stay reproducible

        Returns the same keys the deployed inverse emits, so the observation
        encoder does not care which produced it.
        """
        if true_cls is None or not np.isfinite(true_range_mm):
            true_cls, true_range_mm = 2, self.sensor_horizon_mm

        # --- class, from the range-conditioned posteriors -------------------
        # Draw one of the model's ACTUAL per-ping posteriors for this (true
        # class, range), then take its argmax as the reported label. Emitting
        # the bin's mean confusion row instead -- what this did until
        # 2026-08-12 -- gave the policy a constant that was a deterministic
        # function of the true class, i.e. an oracle with no per-ping
        # information. See POSTERIOR SAMPLING in SCRIPT_FitInverseErrorModel.
        name = CLASS_NAMES[int(true_cls)]
        samples = (_pick_samples(self.t["class_posterior"][name],
                                 float(true_range_mm))
                   if "class_posterior" in self.t else None)
        if samples:
            probs = list(samples[int(rng.integers(len(samples)))])
            obs_cls = int(np.argmax(probs))
        else:
            # Legacy table (no posterior samples): fall back to the old
            # behaviour so an old error-model JSON still runs.
            rows = self.t["class_confusion"][name]
            probs = _pick_probs(rows, float(true_range_mm))
            if probs is None:
                obs_cls = int(true_cls)
                probs = [0.0, 0.0, 0.0]; probs[obs_cls] = 1.0
            else:
                obs_cls = int(rng.choice(3, p=np.asarray(probs) / np.sum(probs)))

        out: Dict = {
            "class_label": obs_cls,
            "p_wall": float(probs[0]), "p_pole": float(probs[1]),
            "p_none": float(probs[2]),
        }

        # --- class-agnostic nearest range, always emitted ------------------
        # The deployed inverse emits agn_dist_mm on every ping regardless of
        # class, so the simulator must too, or the observation encoder sees a
        # key on the robot that it never saw in training.
        agn_rows = self.t.get("agn_range")
        if agn_rows:
            b = _pick_bin(agn_rows, float(true_range_mm))
            if b is not None and b.get("bias") is not None and b.get("sigma") is not None:
                out["agn_dist_mm"] = float(max(
                    true_range_mm + b["bias"] + rng.normal(0.0, b["sigma"]), 0.0))
                out["agn_dist_sigma_mm"] = float(b.get("pred_sigma") or b["sigma"])

        # --- wall slices, always emitted (the real head does too) ----------
        for nm in SLICE_NAMES:
            tr = float(true_slices_mm.get(nm, np.nan))
            b = _pick_bin(self.t["wall_slices"][nm], tr if np.isfinite(tr) else 0.0)
            if not np.isfinite(tr) or b is None:
                # No wall in this slice: report the horizon, not the old
                # labelling cap. With the encoder's clamp raised, 1000 mm would
                # read as "a wall at a metre" rather than "nothing in range".
                out[f"distance_{nm}_mm"] = float(self.sensor_horizon_mm)
                out[f"sigma_{nm}_mm"] = float(self.sensor_horizon_mm)
                continue
            out[f"distance_{nm}_mm"] = float(tr + b["bias"] + rng.normal(0.0, b["sigma"]))
            out[f"sigma_{nm}_mm"] = float(b["sigma"])

        # --- pole geometry -------------------------------------------------
        if obs_cls == 1 and int(true_cls) == 1:
            ba = _pick_bin(self.t["pole_azimuth"], float(true_range_mm))
            br = _pick_bin(self.t["pole_range"], float(true_range_mm))
            az = float(true_pole_az_deg + ba["bias"] + rng.normal(0.0, ba["sigma"]))
            rg = float(true_range_mm + br["bias"] + rng.normal(0.0, br["sigma"]))
            out.update(pole_az_deg=az, pole_az_sigma_deg=float(ba["sigma"]),
                       pole_dist_mm=max(rg, 0.0),
                       pole_dist_sigma_mm=float(br["sigma"]))
        elif obs_cls == 1:
            # Phantom: the model says pole where there is none, so it must also
            # invent the geometry. Sampled from what real false positives look
            # like -- believable ranges, wide bearings.
            ph = self.t["phantom_pole"]
            out.update(
                pole_az_deg=float(rng.normal(ph["azimuth_deg"]["mean"],
                                             ph["azimuth_deg"]["sd"])),
                pole_az_sigma_deg=float(ph["azimuth_deg"]["sd"]),
                pole_dist_mm=float(max(rng.normal(ph["range_mm"]["mean"],
                                                  ph["range_mm"]["sd"]), 0.0)),
                pole_dist_sigma_mm=float(ph["range_mm"]["sd"]),
                phantom=True)
        else:
            out.update(pole_az_deg=float("nan"), pole_az_sigma_deg=float("nan"),
                       pole_dist_mm=float("nan"), pole_dist_sigma_mm=float("nan"))
        out.setdefault("phantom", False)
        return out
