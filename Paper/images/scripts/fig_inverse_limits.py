"""What the inverse model can and cannot recover, as a function of range.

Replaces the aggregate view of `fig_inverse_results.py`. The claim this figure
has to carry is that *different echo cues survive to different ranges*, and that
the hard part is attribution rather than sensing -- neither of which an
aggregate metrics table can show.

BASIS: out-of-fold cross-validation, not the deployed model's 15% holdout.
Every one of the 2775 echoes is scored exactly once, by the fold that did not
see it (`inverse_q0..q3`, the four spatial quadrants, which partition the
set 656/733/733/653). The deployed model's holdout has 418 echoes in total and
only 75 / 44 / 26 in the 1000-1400 / 1400-2000 / 2000+ bands, which cannot
support a range-resolved claim. The in-sample vs held-out comparison stays in
the table, where it belongs, as the overfitting check.

Fold prefix: the plain `inverse_q*` folds are the ones carrying BOTH range
heads, i.e. the deployed architecture (checked against the checkpoints'
state_dict keys: only `inverse_q*` and `inverse_deploy` have `agn_dist_*`).
Do not reach for `inverse_agn_q*` -- despite the name those are the 2026-08-12
diagnostic in which the existing pole-range head was RETARGETED onto the
agnostic target, so they carry no separate agnostic head at all. The range
table in Performance notes 2026-08-12 (night) comes from that variant, not
from the deployed two-head model, which is one reason to regenerate here
rather than quote it.

Panels:
  (A) class accuracy against range, with the per-band majority baseline
  (B) pole-azimuth error against range, with the zero-predictor baseline
  (C) both range heads against range: the masked pole head and the agnostic one
  (D) reliability of the class posterior

Also writes `fig_inverse_limits_numbers.json` beside this script, so the prose
and the figure quote the same numbers. Regenerate both together.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_inverse_limits.py
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys

from paths import CONTROL, SONAR_MODEL, SCRIPTS

sys.path.insert(0, str(CONTROL))
os.chdir(str(CONTROL))  # so the trainer's relative data paths resolve

import numpy as np  # noqa: E402
import SCRIPT_TrainInverseModel as T  # noqa: E402
from Library.SonarModel import InverseModel  # noqa: E402
import style  # noqa: E402

NAME = "fig_inverse_limits"
FOLD_PREFIX = "q"

# Band edges. Fine where the data are dense and the behaviour changes, coarse
# out in the tail where n is small. 1400 gets its own edge because that is where
# class and azimuth collapse; without an edge there the cliff is smeared across
# a bin and reads as a gentle decline.
BANDS = [(200, 500), (500, 750), (750, 1000), (1000, 1400),
         (1400, 1700), (1700, 2000), (2000, 2500), (2500, 3300)]
MIN_N = 20          # bands thinner than this are dropped, not plotted faint
CLIFF_MM = 1400.0

C_CLASS = "#4C72B0"
C_BASE = "#999999"
C_POLE = "#C44E52"
C_AGN = "#55A868"


def _panel_letter(ax, letter):
    ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


def out_of_fold_predictions(sonar, quads):
    """Predict every echo with the fold that held it out.

    Returns one dict of full-length arrays, in the original ping order, so it
    can be indexed with the same masks as the ground-truth arrays.
    """
    n = len(sonar)
    out = {}
    for q in sorted(set(int(x) for x in quads)):
        sel = np.asarray(quads) == q
        inv = InverseModel.load(model_dir=str(SONAR_MODEL), fold=f"{FOLD_PREFIX}{q}")
        pred = T.predict(inv.model, sonar[sel],
                         (inv._s_mean, inv._s_std), (inv._t_mean, inv._t_std),
                         inv.device)
        for k, v in pred.items():
            if v is None:
                continue
            v = np.asarray(v)
            if k not in out:
                shape = (n,) + v.shape[1:]
                out[k] = np.full(shape, np.nan, dtype=float)
            out[k][sel] = v
        print(f"  fold {FOLD_PREFIX}{q}: {int(sel.sum())} echoes scored out of fold")
    return out


def band_stats(true_range, mask, value_fn):
    """Apply value_fn to each band's members; skip bands with too few."""
    rows = []
    for lo, hi in BANDS:
        m = mask & (true_range >= lo) & (true_range < hi)
        if m.sum() < MIN_N:
            continue
        rows.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                         centre=float(np.mean(true_range[m])), **value_fn(m)))
    return rows


def main():
    style.setup()
    import matplotlib.pyplot as plt

    sonar, slice_t, classes, pole_az_deg, near_dist_mm, quads, sess, bins = \
        T.load_and_filter()
    pred = out_of_fold_predictions(sonar, quads)

    # near_dist is the range to whichever reflector won the cone, for every
    # ping -- the same array the agnostic head is trained on, and the natural
    # x-axis for all four panels.
    rng = np.asarray(near_dist_mm, dtype=float)
    cls_pred = pred["cls_pred"]
    probs = pred["cls_probs"]
    conf = np.nanmax(probs, axis=1)
    is_pole = classes == 1
    finite = np.isfinite(rng)

    numbers = {"basis": "out-of-fold CV over 4 spatial quadrants",
               "n_echoes": int(len(classes)),
               "fold_prefix": FOLD_PREFIX,
               "overall_accuracy": float(np.mean(cls_pred == classes))}

    # ---- (A) class accuracy vs range -------------------------------------
    # The baseline is the per-band majority class, not 50%: the class mix
    # shifts with range (poles are a minority close in, and the far bands are
    # wall-heavy), so a flat chance line would understate what "no better than
    # guessing" means out there.
    def acc_row(m):
        maj = max((classes[m] == ci).mean() for ci in (0, 1))
        return dict(acc=100.0 * float(np.mean(cls_pred[m] == classes[m])),
                    majority=100.0 * float(maj))
    acc_rows = band_stats(rng, finite, acc_row)
    numbers["class_accuracy_by_range"] = acc_rows

    # ---- (B) pole azimuth error vs range ---------------------------------
    # Zero-predictor baseline: always answer "straight ahead". If the head is
    # not beating that, it carries no bearing information at all.
    def az_row(m):
        # Both metrics, because they disagree about where the head stops being
        # useful: the median is robust to the occasional wild bearing, RMSE is
        # not, and the recorded 2026-08-12 claim ("azimuth dies at 1.4 m") was
        # an RMSE reading. Report whichever, but say which.
        err = np.abs(pred["pole_pred_az_deg"][m] - pole_az_deg[m])
        base = np.abs(pole_az_deg[m])
        return dict(mae=float(np.nanmedian(err)),
                    baseline=float(np.nanmedian(base)),
                    rmse=float(np.sqrt(np.nanmean(err ** 2))),
                    baseline_rmse=float(np.sqrt(np.nanmean(base ** 2))))
    az_rows = band_stats(rng, finite & is_pole, az_row)
    numbers["pole_azimuth_by_range"] = az_rows

    # ---- (C) the two range heads -----------------------------------------
    def agn_row(m):
        e = pred["agn_pred_dist_mm"][m] - rng[m]
        return dict(pred_mean=float(np.nanmean(pred["agn_pred_dist_mm"][m])),
                    bias=float(np.nanmean(e)),
                    rmse=float(np.sqrt(np.nanmean(e ** 2))),
                    pct=float(100.0 * np.sqrt(np.nanmean(e ** 2))
                              / max(np.mean(rng[m]), 1.0)))
    agn_rows = band_stats(rng, finite, agn_row)

    def pole_row(m):
        e = pred["pole_pred_dist_mm"][m] - rng[m]
        return dict(pred_mean=float(np.nanmean(pred["pole_pred_dist_mm"][m])),
                    bias=float(np.nanmean(e)),
                    rmse=float(np.sqrt(np.nanmean(e ** 2))))
    pole_rows = band_stats(rng, finite & is_pole, pole_row)
    numbers["agnostic_range_by_range"] = agn_rows
    numbers["pole_range_by_range"] = pole_rows

    # The attribution claim: beyond 2 m, where class is at chance, the agnostic
    # head is equally good on walls and on poles. If those two diverge, "it can
    # locate what it cannot name" is not supported.
    far = finite & (rng >= 2000)
    for ci, nm in ((0, "wall"), (1, "pole")):
        m = far & (classes == ci)
        if m.sum() >= 10:
            e = pred["agn_pred_dist_mm"][m] - rng[m]
            numbers[f"agn_rmse_beyond_2m_{nm}"] = dict(
                n=int(m.sum()), rmse=float(np.sqrt(np.nanmean(e ** 2))),
                pct=float(100.0 * np.sqrt(np.nanmean(e ** 2)) / np.mean(rng[m])))

    # Saturation of the masked pole head: it is trained only on poles within
    # 1 m, so past that it should flatten. This is the number that makes it a
    # stop signal rather than a rangefinder.
    m = finite & is_pole & (rng > 1000)
    if m.sum():
        numbers["pole_head_mean_prediction_beyond_1m"] = dict(
            n=int(m.sum()),
            mean_pred=float(np.nanmean(pred["pole_pred_dist_mm"][m])))

    # ---- (D) reliability of the class posterior ---------------------------
    edges = np.linspace(0.5, 1.0, 11)
    rel = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if m.sum() < MIN_N:
            continue
        acc = float(np.mean(cls_pred[m] == classes[m]))
        rel.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()),
                        conf=float(np.mean(conf[m])), acc=acc))
        ece += (m.sum() / len(conf)) * abs(acc - float(np.mean(conf[m])))
    numbers["reliability"] = rel
    numbers["ece"] = float(ece)
    numbers["confidence_thresholds"] = {
        f"p>={t}": dict(coverage=float(np.mean(conf >= t)),
                        accuracy=float(np.mean(cls_pred[conf >= t]
                                               == classes[conf >= t])))
        for t in (0.7, 0.8, 0.9)}

    # ---- draw -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(style.WIDTH_2COL, 5.6))
    (axa, axb), (axc, axd) = axes
    fig.subplots_adjust(hspace=.42, wspace=.28)

    x = [r["centre"] for r in acc_rows]
    axa.plot(x, [r["acc"] for r in acc_rows], "o-", color=C_CLASS, ms=4,
             label="Model")
    axa.plot(x, [r["majority"] for r in acc_rows], "s--", color=C_BASE, ms=3,
             label="Majority class")
    axa.axvline(CLIFF_MM, color="k", ls=":", lw=.8)
    axa.set_xlabel("Range to nearest reflector (mm)")
    axa.set_ylabel("Correctly classified (%)")
    axa.set_ylim(0, 100)
    axa.set_title("Class")
    axa.legend(frameon=False, fontsize=7, loc="lower left")
    _panel_letter(axa, "A")

    xb = [r["centre"] for r in az_rows]
    axb.plot(xb, [r["mae"] for r in az_rows], "o-", color=C_CLASS, ms=4,
             label="Model")
    axb.plot(xb, [r["baseline"] for r in az_rows], "s--", color=C_BASE, ms=3,
             label="Straight ahead")
    axb.axvline(CLIFF_MM, color="k", ls=":", lw=.8)
    axb.set_xlabel("Range to nearest reflector (mm)")
    axb.set_ylabel("Median |azimuth error| (deg)")
    axb.set_title("Pole azimuth")
    axb.legend(frameon=False, fontsize=7, loc="upper left")
    _panel_letter(axb, "B")

    hi = max(max(r["centre"] for r in agn_rows), 2600)
    axc.plot([0, hi], [0, hi], "--", color="0.4", lw=.8, zorder=0)
    axc.errorbar([r["centre"] for r in agn_rows],
                 [r["pred_mean"] for r in agn_rows],
                 yerr=[r["rmse"] for r in agn_rows], fmt="o-", color=C_AGN,
                 ms=4, lw=1.2, capsize=2, label="Nearest reflector (any class)")
    axc.errorbar([r["centre"] for r in pole_rows],
                 [r["pred_mean"] for r in pole_rows],
                 yerr=[r["rmse"] for r in pole_rows], fmt="s-", color=C_POLE,
                 ms=3.5, lw=1.2, capsize=2, label="The pole (masked head)")
    axc.axvline(CLIFF_MM, color="k", ls=":", lw=.8)
    # Bars are +-RMSE and the masked head's are large enough to run negative,
    # which is a plotting artefact of a symmetric bar on a positive quantity.
    # Clip at zero rather than let the panel imply negative distances.
    axc.set_ylim(0, None)
    axc.set_xlabel("True range (mm)")
    axc.set_ylabel("Predicted range (mm)")
    axc.set_title("Range")
    axc.legend(frameon=False, fontsize=7, loc="upper left")
    _panel_letter(axc, "C")

    axd.plot([0.5, 1.0], [50, 100], "--", color="0.4", lw=.8, zorder=0)
    axd.plot([r["conf"] for r in rel], [100 * r["acc"] for r in rel], "o-",
             color=C_CLASS, ms=4)
    axd.set_xlabel("Predicted probability of the reported class")
    axd.set_ylabel("Correct (%)")
    axd.set_xlim(0.5, 1.0)
    axd.set_ylim(40, 100)
    axd.set_title(f"Reliability (ECE {ece:.3f})")
    _panel_letter(axd, "D")

    style.save(fig, NAME)

    with open(SCRIPTS / f"{NAME}_numbers.json", "w") as f:
        json.dump(numbers, f, indent=1)

    print(f"[fig] {numbers['n_echoes']} echoes, out-of-fold accuracy "
          f"{100 * numbers['overall_accuracy']:.1f}%, ECE {ece:.3f}")
    print(f"[fig] numbers -> {SCRIPTS / (NAME + '_numbers.json')}")


if __name__ == "__main__":
    main()
