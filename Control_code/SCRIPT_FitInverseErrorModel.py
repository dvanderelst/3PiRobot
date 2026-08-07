#!/usr/bin/env python3
"""
SCRIPT_FitInverseErrorModel.py

Fit the error model the path-following simulator needs, by running the labelled
acquisition echoes through the deployed inverse and measuring how its output
departs from geometric truth.

Why this exists
---------------
`EnvironmentSimulator` does not simulate echoes. It computes the geometric truth
at the robot's pose and then corrupts it, so that a policy trained in simulation
meets the same errors the real inverse makes. Until now that corruption came
from a wall-only model whose `predict_from_profile` added `N(0, sigma_sim(d))`
per slice; the network itself was loaded and never used on that path.

That model was retired without an archive, which is what currently stops
`SCRIPT_TrainPolicy.py` from starting at all. It also only ever described walls.
The deployed inverse classifies (wall / pole / none) and reports pole azimuth and
range, and the decision recorded in the paper is that the path-following
controller receives that whole local feature. So the error model has to cover:

  - wall slice distance      bias and sigma, per slice, conditioned on truth
  - class confusion          full 3x3, conditioned on true class and range
  - pole azimuth             bias and sigma, conditioned on true range
  - pole range               bias and sigma, conditioned on true range
  - phantom-pole geometry    what the model reports when it invents a pole

Three findings drove those choices, none of which a simpler model would capture:

  1. Range regresses toward the middle of the trained interval: bias +41 mm at
     200-500, +10 mm at 500-800, -36 mm at 800-1000. Overall bias is +9 mm,
     which looks unbiased and is not. Bias must be conditioned on truth.
  2. Azimuth sigma is non-monotonic -- 9.6 deg at 200-500, 7.4 at 500-800, 13.2
     beyond 800 -- mirroring detection, which peaks at 600-700 mm.
  3. 15.7% of true-`none` echoes are classified `pole`. Those phantoms are the
     failure the abstain class was added to reduce, and they survive at one in
     six. A path-following policy meets far more empty cone than pole, so it
     will meet many of them; the simulator has to produce them, and therefore
     has to invent a plausible azimuth and range when it does.

Method
------
Load every labelled echo (the same `load_and_filter` the trainer uses, so the
`none` relabelling is identical), predict with the deployed inverse, and bin the
residuals against geometric truth. Bins are reported with their counts so thin
ones are visible rather than silently trusted.

Output
------
    SonarModel/inverse_error_model.json    the tables, plus provenance
    SonarModel/inverse_error_model.png     diagnostic figure

This script only reads the inverse and the acquisition sessions. It does not
retrain, and does not touch the deployed model.
"""

import json
import os
from datetime import datetime
from typing import Dict, List

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import SCRIPT_TrainInverseModel as T
from Library.SonarModel import InverseModel


# ══════════════════════════════════════════════════════════════════════════════
# Settings  ← change these before running
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR   = "SonarModel"
INVERSE_FOLD = "deploy"          # the model the robot actually deploys with
OUT_JSON    = "SonarModel/inverse_error_model.json"
OUT_PNG     = "SonarModel/inverse_error_model.png"

CLASS_NAMES = ["wall", "pole", "none"]

# Range bins for conditioning. In-range reflectors (wall, pole) live in
# [0, MAX_RANGE]; `none` is by construction beyond it, so it gets its own edges
# -- a reflector just past the horizon is far more confusable than a distant one.
# Matched to the paper figure's panel E. 150 mm below 500 where poles are
# sparse, 100 mm from 500 through the class boundary and just beyond it, where
# the interesting collapse happens and the counts support the resolution.
# Coarser bins smeared it badly: at 200 mm width, pole recall in 800-1000 read
# as 50%, when it is 66% in 800-900 and 26% in 900-1000. The simulator would
# then have told the policy that poles stay half-detectable right up to the
# horizon, which is where a path-following robot spends much of its time.
IN_RANGE_EDGES  = [200., 350., 500., 600., 700., 800., 900., 1000.]
OUT_RANGE_EDGES = [1000., 1100., 1200., 1400., 1700., 2000., 1e9]

# Wall slice bins, over true slice distance.
WALL_EDGES = [0., 250., 500., 750., 1000., 1500., 2000., 1e9]

# Bins with fewer than this are still written, but flagged in the JSON and the
# console so a thin bin is never mistaken for a measured one.
MIN_BIN_N = 15

# ══════════════════════════════════════════════════════════════════════════════


def _binned_residual(true_vals, resid, edges, min_n=MIN_BIN_N) -> List[dict]:
    """bias and sigma of `resid` in bins of `true_vals`."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (true_vals >= lo) & (true_vals < hi) & np.isfinite(resid)
        n = int(m.sum())
        out.append({
            "lo": float(lo), "hi": float(hi), "n": n,
            "centre": float((lo + min(hi, lo * 3 + 1000)) / 2.0) if np.isinf(hi)
                      else float((lo + hi) / 2.0),
            "bias": float(resid[m].mean()) if n else None,
            "sigma": float(resid[m].std()) if n > 1 else None,
            "thin": n < min_n,
        })
    return out


def _confusion(true_cls, pred_cls, true_rng, cls_idx, edges) -> List[dict]:
    """P(predicted | true class, range bin)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (true_cls == cls_idx) & (true_rng >= lo) & (true_rng < hi)
        n = int(m.sum())
        probs = ([float((pred_cls[m] == j).mean()) for j in range(3)] if n
                 else [None, None, None])
        out.append({"lo": float(lo), "hi": float(hi), "n": n,
                    "p_pred": probs, "thin": n < MIN_BIN_N})
    return out


def main() -> None:
    print(f"Loading acquisition echoes (same filter as the trainer)…")
    sonar, slice_t, classes, pole_az, pole_dist, quads, sess, bin_centers = \
        T.load_and_filter()
    print(f"\nPredicting with the deployed inverse (fold={INVERSE_FOLD!r})…")
    inv = InverseModel.load(model_dir=MODEL_DIR, fold=INVERSE_FOLD, device="cpu")
    if inv.pole_dist_divisor is None:
        raise SystemExit("This inverse has no pole-range head; nothing to fit "
                         "for the pole channels. Retrain with pole_dist_head=True.")
    pred = inv.predict_from_envelope(sonar[..., 0], sonar[..., 1])
    pred_cls = np.asarray(pred["class_label"])

    # Truth for conditioning: distance to whichever reflector won the cone.
    # For pole-class echoes that is the pole surface; for wall-class it is the
    # nearest wall in the cone; for `none` it is whatever sat beyond the horizon.
    true_rng = np.asarray(pole_dist, dtype=np.float64)
    wall_near = np.nanmin(np.where(np.isfinite(slice_t), slice_t, np.nan), axis=1)
    true_rng = np.where(classes == 0, wall_near, true_rng)

    model: Dict = {
        "provenance": {
            "fitted": datetime.now().isoformat(timespec="seconds"),
            "inverse_fold": INVERSE_FOLD,
            "sessions": list(T.ACQUISITION_SESSIONS),
            "n_echoes": int(len(classes)),
            "n_by_true_class": {n: int((classes == i).sum())
                                for i, n in enumerate(CLASS_NAMES)},
            "max_range_mm": float(T.MAX_RANGE_MM),
            "cone_half_deg": float(T.CONE_HALF_DEG),
            "note": ("Residuals of the deployed inverse against geometric truth, "
                     "for the training simulator's error model. The simulator "
                     "corrupts geometric truth with these statistics; it does "
                     "not simulate echoes."),
        },
        "class_names": CLASS_NAMES,
    }

    # ── wall slices ──────────────────────────────────────────────────────────
    print("\n=== wall slice residuals (predicted - true), by true distance ===")
    model["wall_slices"] = {}
    for i, nm in enumerate(T.SLICE_NAMES if hasattr(T, "SLICE_NAMES")
                           else ["right", "center", "left"]):
        tr = slice_t[:, i].astype(np.float64)
        pr = np.asarray(pred[f"distance_{nm}_mm"], dtype=np.float64)
        ok = (classes == 0) & np.isfinite(tr)
        bins = _binned_residual(tr[ok], (pr - tr)[ok], WALL_EDGES)
        model["wall_slices"][nm] = bins
        print(f"  {nm}:")
        for b in bins:
            if not b["n"]:
                continue
            flag = "  <-- thin" if b["thin"] else ""
            print(f"    {b['lo']:6.0f}-{b['hi'] if b['hi']<1e8 else float('inf'):>6} "
                  f"n={b['n']:4d}  bias {b['bias']:+7.0f}  sigma {b['sigma']:6.0f}{flag}")

    # ── class confusion ──────────────────────────────────────────────────────
    print("\n=== class confusion P(pred | true, range) ===")
    model["class_confusion"] = {}
    for ci, nm in enumerate(CLASS_NAMES):
        edges = OUT_RANGE_EDGES if nm == "none" else IN_RANGE_EDGES
        rows = _confusion(classes, pred_cls, true_rng, ci, edges)
        model["class_confusion"][nm] = rows
        print(f"  true={nm}:")
        for b in rows:
            if not b["n"]:
                continue
            p = b["p_pred"]
            flag = "  <-- thin" if b["thin"] else ""
            hi = "inf" if b["hi"] > 1e8 else f"{b['hi']:.0f}"
            print(f"    {b['lo']:6.0f}-{hi:>6} n={b['n']:4d}  "
                  f"wall {p[0]*100:5.1f}%  pole {p[1]*100:5.1f}%  "
                  f"none {p[2]*100:5.1f}%{flag}")

    # ── pole geometry, conditional on the model calling it a pole ────────────
    det = (classes == 1) & (pred_cls == 1)
    print(f"\n=== pole geometry when correctly detected (n={int(det.sum())}) ===")
    az_res = np.asarray(pred["pole_az_deg"])[det] - pole_az[det]
    rg_res = np.asarray(pred["pole_dist_mm"])[det] - pole_dist[det]
    model["pole_azimuth"] = _binned_residual(pole_dist[det], az_res, IN_RANGE_EDGES)
    model["pole_range"]   = _binned_residual(pole_dist[det], rg_res, IN_RANGE_EDGES)
    for label, bins, unit in (("azimuth", model["pole_azimuth"], "deg"),
                              ("range", model["pole_range"], "mm")):
        print(f"  {label}:")
        for b in bins:
            if not b["n"]:
                continue
            flag = "  <-- thin" if b["thin"] else ""
            print(f"    {b['lo']:6.0f}-{b['hi']:6.0f} n={b['n']:4d}  "
                  f"bias {b['bias']:+7.1f} {unit}  sigma {b['sigma']:6.1f} {unit}{flag}")

    # ── phantom poles ────────────────────────────────────────────────────────
    ph = (classes != 1) & (pred_cls == 1)
    pz = np.asarray(pred["pole_az_deg"])[ph]
    pr_ = np.asarray(pred["pole_dist_mm"])[ph]
    model["phantom_pole"] = {
        "n": int(ph.sum()),
        "rate_from_wall": float(((classes == 0) & (pred_cls == 1)).sum()
                                / max((classes == 0).sum(), 1)),
        "rate_from_none": float(((classes == 2) & (pred_cls == 1)).sum()
                                / max((classes == 2).sum(), 1)),
        "azimuth_deg": {"mean": float(pz.mean()), "sd": float(pz.std()),
                        "quantiles": [float(q) for q in
                                      np.percentile(pz, [5, 25, 50, 75, 95])]},
        "range_mm": {"mean": float(pr_.mean()), "sd": float(pr_.std()),
                     "quantiles": [float(q) for q in
                                   np.percentile(pr_, [5, 25, 50, 75, 95])]},
        "note": ("Geometry the model reports for poles that are not there. The "
                 "simulator samples from this when the confusion model draws "
                 "`pole` on a cone that holds none, so the policy learns to "
                 "distrust isolated pole readings rather than to trust a "
                 "channel that is clean only in simulation."),
    }
    print(f"\n=== phantom poles (predicted pole, truth is not) ===")
    print(f"  n={model['phantom_pole']['n']}   "
          f"from wall {model['phantom_pole']['rate_from_wall']*100:.1f}%   "
          f"from none {model['phantom_pole']['rate_from_none']*100:.1f}%")
    print(f"  reported azimuth  {pz.mean():+.1f} +/- {pz.std():.1f} deg")
    print(f"  reported range    {pr_.mean():.0f} +/- {pr_.std():.0f} mm")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(model, fh, indent=2)
    print(f"\nWrote {OUT_JSON}")
    _plot(model, pole_dist[det], az_res, rg_res, OUT_PNG)
    print(f"Wrote {OUT_PNG}")


def _plot(model, pd_det, az_res, rg_res, out_png) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    for nm, c in zip(["right", "center", "left"], ["#e41a1c", "#377eb8", "#4daf4a"]):
        b = [x for x in model["wall_slices"][nm] if x["n"]]
        ax.errorbar([x["centre"] for x in b], [x["bias"] for x in b],
                    yerr=[x["sigma"] for x in b], marker="o", capsize=3,
                    color=c, label=nm, alpha=.85)
    ax.axhline(0, color="k", lw=.8); ax.set_title("wall slice residual")
    ax.set_xlabel("true slice distance (mm)"); ax.set_ylabel("pred - true (mm)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for nm, c in zip(model["class_names"], ["#377eb8", "#c05cff", "#999999"]):
        rows = [r for r in model["class_confusion"][nm] if r["n"]]
        ax.plot([(r["lo"] + min(r["hi"], 2500)) / 2 for r in rows],
                [r["p_pred"][model["class_names"].index(nm)] * 100 for r in rows],
                "o-", color=c, label=f"true {nm} -> correct")
    ax.set_ylim(0, 100); ax.set_title("class recall vs true range")
    ax.set_xlabel("true range (mm)"); ax.set_ylabel("% correct"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(pd_det, az_res, s=10, alpha=.4, color="#c05cff")
    b = [x for x in model["pole_azimuth"] if x["n"]]
    ax.errorbar([x["centre"] for x in b], [x["bias"] for x in b],
                yerr=[x["sigma"] for x in b], color="black", marker="s", capsize=3)
    ax.axhline(0, color="k", lw=.8); ax.set_title("pole azimuth residual")
    ax.set_xlabel("true pole range (mm)"); ax.set_ylabel("pred - true (deg)")

    ax = axes[1, 1]
    ax.scatter(pd_det, rg_res, s=10, alpha=.4, color="#ff9f0a")
    b = [x for x in model["pole_range"] if x["n"]]
    ax.errorbar([x["centre"] for x in b], [x["bias"] for x in b],
                yerr=[x["sigma"] for x in b], color="black", marker="s", capsize=3)
    ax.axhline(0, color="k", lw=.8); ax.set_title("pole range residual")
    ax.set_xlabel("true pole range (mm)"); ax.set_ylabel("pred - true (mm)")

    fig.suptitle("Deployed inverse: residuals against geometric truth "
                 "(the simulator's error model)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
