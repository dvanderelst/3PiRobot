"""Should the pole-range head still be capped at 1 m, now that session 6 exists?

The head is trained only on poles within `POLE_DIST_TRAIN_MAX_MM` (1000 mm).
That cap was chosen by EXPT_range_horizon, which measured a widened span
dragging the head's close-range predictions outward: signed bias at 0-500 mm
went +86 mm (span 0-1 m) to +149 mm (span 0-1.9 m). Since the terminal approach
stops on this head's output, a positive bias makes the robot drive nearer than
the protocol says.

BUT that experiment ran on the FIVE-session dataset: 2135 echoes, 222 pole
echoes beyond 1 m, none past 1772 mm. Acquisition06 was collected specifically
to fix that hole and did -- it contributes 207 of the 429 far pole echoes now
available, and carries the furthest pole out to 3196 mm. So the cap rests on a
measurement taken before the data that would test it existed.

The question matters beyond the inverse. Experiment 2 asks whether poles serve
as landmarks; the policy currently receives a pole range that is meaningless
past a metre, so a head that worked at landmark distances could change what that
experiment can claim.

Conditions differ in ONE knob. Labelling stays at the deployed setting
(FAR_LABEL_MODE="true_class"), architecture and seed are shared, and all four
quadrant folds are trained per condition so every echo is scored once by a model
that did not see it.

    cap1000   the deployed setting
    cap1700   out to where the five-session data used to end
    uncapped  every pole echo, to 3.2 m

What decides it, in order:
  1. Close range, 200-500 mm. The stop fires at 400 mm, so bias here is the
     cost side of the trade. Anything that pushes bias positive is a real
     regression regardless of what it buys further out.
  2. Landmark range, 1000-2000 mm. Does a wider cap actually buy usable pole
     ranging, or does the head just fail further out?
  3. Collateral. Class, azimuth, wall slices and the agnostic range head share a
     trunk with this head; a gain here that costs those is not a gain.

Writes TempOutput/PoleRangeCap/summary.json and prints a comparison table. Fold
artifacts go to SonarModel/expt_prc_* so nothing named inverse_* is touched.

    .venv/bin/python3 EXPT_pole_range_cap.py
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json

import numpy as np
import torch

import SCRIPT_TrainInverseModel as T

CONDITIONS = [("cap1000", 1000.0), ("cap1700", 1700.0), ("uncapped", float("inf"))]
BANDS = [(200, 500), (500, 1000), (1000, 1400), (1400, 2000), (2000, 3300)]
OUT_DIR = os.path.join("TempOutput", "PoleRangeCap")


def _oof(sink, n, n_slices):
    out = {"pred": np.full(n, -1, dtype=np.int64),
           "az": np.full(n, np.nan), "pdist": np.full(n, np.nan),
           "pdist_std": np.full(n, np.nan), "agn": np.full(n, np.nan),
           "wall": np.full((n, n_slices), np.nan)}
    for rec in sink:
        idx = rec["val_idx"]
        p = rec["pred"]
        out["pred"][idx] = p["cls_pred"]
        out["az"][idx] = p["pole_pred_az_deg"]
        if p.get("pole_pred_dist_mm") is not None:
            out["pdist"][idx] = p["pole_pred_dist_mm"]
            out["pdist_std"][idx] = p["pole_pred_dist_std"]
        if p.get("agn_pred_dist_mm") is not None:
            out["agn"][idx] = p["agn_pred_dist_mm"]
        out["wall"][idx] = p["wall_pred_mean"]
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for tag, cap in CONDITIONS:
        print(f"\n{'=' * 78}\n  CONDITION {tag}  (POLE_DIST_TRAIN_MAX_MM={cap})\n{'=' * 78}")
        T.FAR_LABEL_MODE = "true_class"
        T.POLE_DIST_TRAIN_MAX_MM = cap
        T.ARTIFACT_PREFIX = f"expt_prc_{tag}"
        np.random.seed(T.SEED)
        torch.manual_seed(T.SEED)

        sonar, slice_t, classes, az, near, quads, sess, _ = T.load_and_filter()
        az_n = np.where(np.isnan(az / T.CONE_HALF_DEG), 0.0,
                        az / T.CONE_HALF_DEG).astype(np.float32)
        pd_n = np.where(np.isnan(near / T.POLE_DIST_NORM_MM), 0.0,
                        near / T.POLE_DIST_NORM_MM).astype(np.float32)
        agn_n = np.where(np.isnan(near / T.AGN_DIST_NORM_MM), 0.0,
                         near / T.AGN_DIST_NORM_MM).astype(np.float32)

        sink = []
        for q in T.CV_QUADRANTS:
            print(f"\n--- {tag} fold q={q} ---")
            T.run_fold(q, sonar, slice_t, classes, az, az_n, quads, sess,
                       device, sub_prefix=f"q{q}",
                       pole_dist_mm=near, pole_dist_n_safe=pd_n,
                       agn_dist_mm=near, agn_dist_n_safe=agn_n,
                       oof_sink=sink)

        o = _oof(sink, len(sonar), slice_t.shape[1])
        pole = (classes == 1) & np.isfinite(near)
        wall = (classes == 0)
        r = {"cap_mm": cap,
             "class_acc": float((o["pred"] == classes).mean()),
             "pole_az_rmse": float(np.sqrt(np.nanmean((o["az"][pole] - az[pole]) ** 2))),
             "agn_rmse": float(np.sqrt(np.nanmean((o["agn"] - near) ** 2))),
             "wall_rmse": float(np.sqrt(np.nanmean(
                 (o["wall"][wall] - slice_t[wall]) ** 2))),
             "pole_range": {}}
        for lo, hi in BANDS:
            m = pole & (near >= lo) & (near < hi)
            if m.sum() < 15:
                continue
            e = o["pdist"][m] - near[m]
            r["pole_range"][f"{lo}-{hi}"] = dict(
                n=int(m.sum()), bias=float(np.nanmean(e)),
                rmse=float(np.sqrt(np.nanmean(e ** 2))),
                pred_mean=float(np.nanmean(o["pdist"][m])),
                sigma=float(np.nanmedian(o["pdist_std"][m])))
        results[tag] = r

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=1)

    print(f"\n{'=' * 78}\n  POLE-RANGE HEAD, by true pole range\n{'=' * 78}")
    hdr = "band".ljust(12) + "".join(t.ljust(22) for t, _ in CONDITIONS)
    print(hdr)
    print(" " * 12 + "".join("bias   rmse   pred ".ljust(22) for _ in CONDITIONS))
    for lo, hi in BANDS:
        key = f"{lo}-{hi}"
        row = f"{lo}-{hi}".ljust(12)
        for tag, _ in CONDITIONS:
            b = results[tag]["pole_range"].get(key)
            row += (f"{b['bias']:+6.0f} {b['rmse']:6.0f} {b['pred_mean']:6.0f} "
                    .ljust(22)) if b else "-".ljust(22)
        print(row)

    print("\ncollateral (shared trunk):")
    print("  metric".ljust(20) + "".join(t.ljust(12) for t, _ in CONDITIONS))
    for k, fmt in (("class_acc", "{:.3f}"), ("pole_az_rmse", "{:.1f}"),
                   ("agn_rmse", "{:.0f}"), ("wall_rmse", "{:.0f}")):
        print(f"  {k}".ljust(20)
              + "".join(fmt.format(results[t][k]).ljust(12) for t, _ in CONDITIONS))
    print(f"\nwrote {os.path.join(OUT_DIR, 'summary.json')}")


if __name__ == "__main__":
    main()
