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
WINDOW = 300        # echoes per sliding window for the crossing estimate
WINDOW_POLE = 150   # ditto for azimuth, where only pole echoes count
N_BOOT = 300

# Colour carries WHAT THE DATA IS ABOUT; line style carries WHICH MODEL
# (solid/filled = quadrant, dashed/open = deployed, grey dotted = baseline).
# The two are orthogonal, so a reader never has to ask which of the two a
# colour change is signalling.
C_BOTH = "#4C72B0"   # walls and poles together: class, nearest-reflector range
C_POLE = "#C44E52"   # pole only: azimuth, the masked pole-range head
C_WALL = "#937860"   # wall only: the three depth slices, as shades of one hue
C_BASE = "#999999"
C_CLASS = C_BOTH     # kept as an alias so the class panels read naturally
C_AGN = C_BOTH
SLICE_COLORS = {"left": "#BFA48E", "center": "#937860", "right": "#6B5340"}
# One scheme, applied in every panel, so that "which model" and "what kind of
# line" are the same question everywhere:
#   quadrant models -> solid line, filled markers
#   deployed model  -> dashed line, open markers, SAME colour as the quadrant
#                      series it should be compared against
#   baselines       -> thin grey dotted, no markers, drawn behind: these are
#                      not models and must not read as a third one
QUAD = dict(ls="-", marker="o", ms=4, mfc=None)
DEP = dict(ls="--", marker="o", ms=4.5, mfc="none", lw=1.0)
BASE = dict(ls=":", marker="", lw=1.2, color=C_BASE, zorder=0)


def _mark_crossing(ax, c):
    """Vertical mark at the measured crossing, with its bootstrap interval.

    Panel C gets none on purpose: nothing crosses there, which is the point.
    """
    ax.axvspan(c["ci_lo"], c["ci_hi"], color="0.6", alpha=.16, lw=0, zorder=0)
    ax.axvline(c["mm"], color="0.35", ls=":", lw=1.0, zorder=0)


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


def crossing(rng_sorted, margin_fn, window, n_boot=N_BOOT, seed=0):
    """Range at which the model stops beating its baseline, without binning.

    Slides a fixed-count window along range and finds the last window in which
    the model is still ahead. Bins cannot answer this: a bin edge placed where
    the crossing is believed to be will reproduce that belief, which is how
    1400 mm survived in this project for as long as it did -- it was a bin edge
    inherited from an earlier binned analysis, and the model is in fact still
    ahead of the base rate there.

    margin_fn(slice) must return a positive number while the model is winning.
    """
    def _once(order):
        r = rng_sorted[order]
        xs, d = [], []
        for i in range(0, len(r) - window):
            w = order[i:i + window]
            xs.append(float(np.median(rng_sorted[w])))
            d.append(margin_fn(w))
        xs = np.asarray(xs); d = np.asarray(d)
        ahead = np.where(d[:-1] > 0)[0]
        return xs[ahead.max() + 1] if len(ahead) else float("nan")

    base_order = np.arange(len(rng_sorted))
    point = _once(base_order)
    rs = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = np.sort(rs.integers(0, len(rng_sorted), len(rng_sorted)))
        boot.append(_once(idx))
    boot = np.asarray(boot); boot = boot[np.isfinite(boot)]
    return dict(mm=float(point), boot_median=float(np.median(boot)),
                ci_lo=float(np.percentile(boot, 2.5)),
                ci_hi=float(np.percentile(boot, 97.5)), window=int(window))


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

    sonar, slice_t, classes, pole_az_deg, near_dist_mm, quads, sess, bins, poses = \
        T.load_and_filter(with_poses=True)
    pred = out_of_fold_predictions(sonar, quads)

    # The deployed model is a FIFTH model, not a combination of the four folds:
    # it trains once on 85% of a different, single spatial split. So the fold
    # curves describe the procedure, not the weights on the robot. Score the
    # deployed model on its own held-out 15% and overlay it, so the reader can
    # see the deployed instance sitting on the same curve rather than having to
    # assume it does. Only the bands where that holdout has members are drawn;
    # past ~1400 mm it is too thin and the fold curve carries the claim alone.
    dep_val = T.spatial_holdout_mask(poses, sess, T.HOLDOUT_FRAC, T.HOLDOUT_SEED)
    dep_inv = InverseModel.load(model_dir=str(SONAR_MODEL), fold="deploy")
    dep_raw = T.predict(dep_inv.model, sonar[dep_val],
                        (dep_inv._s_mean, dep_inv._s_std),
                        (dep_inv._t_mean, dep_inv._t_std), dep_inv.device)
    dep = {}
    for k, v in dep_raw.items():
        if v is None:
            continue
        v = np.asarray(v)
        full = np.full((len(sonar),) + v.shape[1:], np.nan, dtype=float)
        full[dep_val] = v
        dep[k] = full
    print(f"  deployed model: {int(dep_val.sum())} echoes on its own holdout")

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

    # ---- wall depth profile (no panel, but Experiment 2 reads it) ---------
    # The controller in Experiment 2 steers on these three distances, so the
    # section cannot be silent about them even though they get no panel.
    is_wall = classes == 0
    wall_rows = {}
    for i, sname in enumerate(T.SLICE_NAMES):
        t = slice_t[:, i].astype(float)
        pp = pred["wall_pred_mean"][:, i]
        m = is_wall & np.isfinite(t) & np.isfinite(pp)
        e = pp[m] - t[m]
        wall_rows[sname] = dict(n=int(m.sum()),
                                rmse=float(np.sqrt(np.nanmean(e ** 2))),
                                mae=float(np.nanmean(np.abs(e))))
    numbers["wall_slices"] = wall_rows
    # And the same restricted to the far half, since the claim that the wall
    # profile survives where class does not needs its own number.
    wall_far = {}
    for i, sname in enumerate(T.SLICE_NAMES):
        t = slice_t[:, i].astype(float)
        pp = pred["wall_pred_mean"][:, i]
        m = is_wall & np.isfinite(t) & np.isfinite(pp) & (rng >= 1400)
        if m.sum() >= MIN_N:
            e = pp[m] - t[m]
            wall_far[sname] = dict(n=int(m.sum()),
                                   rmse=float(np.sqrt(np.nanmean(e ** 2))))
    numbers["wall_slices_beyond_1400"] = wall_far

    # Band-resolved, for panel E. Baseline is a constant predictor: answer the
    # band's mean true distance for that slice and never listen. It plays the
    # part the majority class plays in A and straight-ahead plays in B.
    def _wall_rows(mask, source, min_n):
        out = {}
        for i, sname in enumerate(T.SLICE_NAMES):
            t = slice_t[:, i].astype(float)
            pp = source["wall_pred_mean"][:, i]
            ok = mask & np.isfinite(t) & np.isfinite(pp)
            rows = []
            for lo, hi in BANDS:
                m = ok & (rng >= lo) & (rng < hi)
                if m.sum() < min_n:
                    continue
                e = pp[m] - t[m]
                rows.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                                 centre=float(np.mean(rng[m])),
                                 rmse=float(np.sqrt(np.nanmean(e ** 2))),
                                 baseline=float(np.nanstd(t[m]))))
            out[sname] = rows
        return out

    wall_by_range = _wall_rows(is_wall & finite, pred, MIN_N)
    numbers["wall_slices_by_range"] = wall_by_range

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

    # ---- where each quantity stops beating its baseline --------------------
    order_all = np.argsort(np.where(finite, rng, np.inf))
    order_all = order_all[finite[order_all]]
    r_all = rng[order_all]
    ok_all = (cls_pred == classes).astype(float)

    def _class_margin(idx):
        f = float(np.mean(classes[idx] == 0))
        return float(np.mean(ok_all[idx])) - max(f, 1.0 - f)

    pm = finite & is_pole
    order_pole = np.argsort(np.where(pm, rng, np.inf))
    order_pole = order_pole[pm[order_pole]]
    az_err = np.abs(pred["pole_pred_az_deg"] - pole_az_deg)

    def _az_margin(idx):
        return float(np.nanmedian(np.abs(pole_az_deg[idx]))
                     - np.nanmedian(az_err[idx]))

    numbers["crossings"] = {
        "class": crossing(rng[order_all], lambda w: _class_margin(order_all[w]),
                          WINDOW),
        "pole_azimuth": crossing(rng[order_pole],
                                 lambda w: _az_margin(order_pole[w]),
                                 WINDOW_POLE),
    }
    xc = numbers["crossings"]["class"]
    xa = numbers["crossings"]["pole_azimuth"]
    print(f"  class crosses its base rate at {xc['mm']:.0f} mm "
          f"(CI {xc['ci_lo']:.0f}-{xc['ci_hi']:.0f}); azimuth at {xa['mm']:.0f} mm "
          f"(CI {xa['ci_lo']:.0f}-{xa['ci_hi']:.0f})")

    # ---- the deployed model on its own holdout, for the overlay -----------
    DEP_MIN_N = 15
    def _dep_rows(mask, fn):
        rows = []
        for lo, hi in BANDS:
            m = mask & (rng >= lo) & (rng < hi)
            if m.sum() < DEP_MIN_N:
                continue
            rows.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                             centre=float(np.mean(rng[m])), **fn(m)))
        return rows

    dep_acc = _dep_rows(finite & dep_val, lambda m: dict(
        acc=100.0 * float(np.mean(dep["cls_pred"][m] == classes[m]))))
    dep_az = _dep_rows(finite & is_pole & dep_val, lambda m: dict(
        mae=float(np.nanmedian(np.abs(dep["pole_pred_az_deg"][m] - pole_az_deg[m])))))
    dep_agn = _dep_rows(finite & dep_val, lambda m: dict(
        pred_mean=float(np.nanmean(dep["agn_pred_dist_mm"][m]))))
    dep_pole = _dep_rows(finite & is_pole & dep_val, lambda m: dict(
        pred_mean=float(np.nanmean(dep["pole_pred_dist_mm"][m]))))
    # The quadrant models restricted to the echoes the deployed model was
    # scored on. Without this, panel B invites a wrong reading: the deployed
    # curve sits below the quadrant curve, which looks like the deployed model
    # being better, when in fact its holdout simply contains easier pole
    # echoes. Drawn only in B, because that is the panel where the artefact is
    # large; for class the holdout is representative (see paired stats below).
    match_az = _dep_rows(finite & is_pole & dep_val, lambda m: dict(
        mae=float(np.nanmedian(np.abs(pred["pole_pred_az_deg"][m]
                                      - pole_az_deg[m])))))

    dep_wall_by_range = _wall_rows(is_wall & finite & dep_val, dep, 15)
    dep_conf = np.nanmax(dep["cls_probs"], axis=1)
    dep_rel = []
    dep_edges = np.linspace(0.5, 1.0, 6)
    for lo, hi in zip(dep_edges[:-1], dep_edges[1:]):
        m = dep_val & (dep_conf >= lo) & (dep_conf < hi if hi < 1.0 else dep_conf <= hi)
        if m.sum() < DEP_MIN_N:
            continue
        dep_rel.append(dict(conf=float(np.mean(dep_conf[m])), n=int(m.sum()),
                            acc=float(np.mean(dep["cls_pred"][m] == classes[m]))))
    numbers["deployed_holdout"] = dict(
        n=int(dep_val.sum()),
        accuracy=float(np.mean(dep["cls_pred"][dep_val] == classes[dep_val])),
        class_accuracy_by_range=dep_acc,
        pole_azimuth_by_range=dep_az,
        agnostic_range_by_range=dep_agn,
        pole_range_by_range=dep_pole,
        reliability=dep_rel)

    # Like-for-like: both model sets on identical echoes, so any residual
    # difference is a model difference rather than a sampling one.
    rs = np.random.default_rng(0)
    pm_val = finite & is_pole & dep_val
    eq = np.abs(pred["pole_pred_az_deg"][pm_val] - pole_az_deg[pm_val])
    ed = np.abs(dep["pole_pred_az_deg"][pm_val] - pole_az_deg[pm_val])
    bs = [np.median(ed[i]) - np.median(eq[i])
          for i in (rs.integers(0, len(eq), len(eq)) for _ in range(2000))]
    okq = (cls_pred == classes)[dep_val]
    okd = (dep["cls_pred"] == classes)[dep_val]
    bsc = [okd[i].mean() - okq[i].mean()
           for i in (rs.integers(0, len(okq), len(okq)) for _ in range(2000))]
    numbers["like_for_like"] = dict(
        n_pole=int(pm_val.sum()),
        azimuth_quadrant_median=float(np.median(eq)),
        azimuth_deployed_median=float(np.median(ed)),
        azimuth_diff_ci=[float(np.percentile(bs, 2.5)),
                         float(np.percentile(bs, 97.5))],
        azimuth_quadrant_on_other_echoes=float(np.nanmedian(
            np.abs(pred["pole_pred_az_deg"][finite & is_pole & ~dep_val]
                   - pole_az_deg[finite & is_pole & ~dep_val]))),
        n_all=int(dep_val.sum()),
        class_quadrant=float(okq.mean()), class_deployed=float(okd.mean()),
        class_diff_ci=[float(np.percentile(bsc, 2.5)),
                       float(np.percentile(bsc, 97.5))],
        class_quadrant_on_other_echoes=float(
            (cls_pred == classes)[~dep_val].mean()))
    lfl = numbers["like_for_like"]
    print(f"  like-for-like on the deployed holdout: azimuth "
          f"{lfl['azimuth_quadrant_median']:.2f} vs "
          f"{lfl['azimuth_deployed_median']:.2f} deg "
          f"(quadrant models score {lfl['azimuth_quadrant_on_other_echoes']:.2f} "
          f"on the other echoes); class {lfl['class_quadrant']:.3f} vs "
          f"{lfl['class_deployed']:.3f}")

    # ---- draw -------------------------------------------------------------
    # One panel per output head, in the order Fig. 3 draws them, then
    # reliability. C and D are the two range heads and share identical axes, so
    # that the masked head's flatness and the agnostic head's diagonal are read
    # at the same scale: that comparison is the section's central claim and it
    # is now made across two panels rather than inside one.
    fig, axes = plt.subplots(2, 3, figsize=(style.WIDTH_2COL, 5.5))
    (axa, axb, axc), (axd, axe, axf) = axes
    fig.subplots_adjust(hspace=.58, wspace=.42, left=.09, right=.99,
                    top=.93, bottom=.10)

    x = [r["centre"] for r in acc_rows]
    axa.plot(x, [r["majority"] for r in acc_rows], label="Majority class", **BASE)
    axa.plot(x, [r["acc"] for r in acc_rows], color=C_CLASS,
             label="Quadrant models", **QUAD)
    axa.plot([r["centre"] for r in dep_acc], [r["acc"] for r in dep_acc],
             color=C_CLASS, label="Deployed model", **DEP)
    _mark_crossing(axa, xc)
    axa.set_xlabel("Range (mm)")
    axa.set_ylabel("Correctly classified (%)")
    axa.set_ylim(0, 100)
    axa.set_title("Class")
    axa.legend(frameon=False, fontsize=6, loc="lower left")
    _panel_letter(axa, "A")

    xb = [r["centre"] for r in az_rows]
    axb.plot(xb, [r["baseline"] for r in az_rows], label="Straight ahead", **BASE)
    axb.plot(xb, [r["mae"] for r in az_rows], color=C_POLE,
             label="Quadrant models", **QUAD)
    axb.plot([r["centre"] for r in match_az], [r["mae"] for r in match_az],
             color=C_POLE, label="Quadrant models, same echoes",
             **{**DEP, "mfc": C_POLE})
    axb.plot([r["centre"] for r in dep_az], [r["mae"] for r in dep_az],
             color=C_POLE, label="Deployed model", **DEP)
    _mark_crossing(axb, xa)
    axb.set_xlabel("Range (mm)")
    axb.set_ylabel("Median |az. error| (deg)")
    axb.set_title("Pole azimuth")
    axb.legend(frameon=False, fontsize=6, loc="upper left",
               bbox_to_anchor=(0.0, 0.95), labelspacing=.3)
    _panel_letter(axb, "B")

    R_LIM = 2900
    for ax, rows, deprows, colour, title, lab in (
            (axc, pole_rows, dep_pole, C_POLE, "Pole range",
             "trained on poles within 1 m"),
            (axd, agn_rows, dep_agn, C_AGN, "Nearest-reflector range",
             "trained on every echo")):
        ax.plot([0, R_LIM], [0, R_LIM], "--", color="0.4", lw=.8, zorder=0)
        ax.errorbar([r["centre"] for r in rows], [r["pred_mean"] for r in rows],
                    yerr=[r["rmse"] for r in rows], fmt="o-", color=colour,
                    ms=4, lw=1.2, capsize=2, label="Quadrant models")
        ax.plot([r["centre"] for r in deprows],
                [r["pred_mean"] for r in deprows], color=colour,
                label="Deployed model", **DEP)
        ax.set_xlim(0, R_LIM); ax.set_ylim(0, R_LIM)
        ax.set_xlabel("True range (mm)")
        ax.set_ylabel("Predicted range (mm)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=6, loc="upper left",
                  bbox_to_anchor=(0.0, 0.95), labelspacing=.3, title=lab,
                  title_fontsize=6)
    _panel_letter(axc, "C")
    _panel_letter(axd, "D")

    # (E) the wall profile, which Experiment 2 steers on. Plotted as error
    # against range rather than predicted-against-true: the claim is that it
    # does NOT degrade where class does, and an error curve shows that directly.
    for sname in T.SLICE_NAMES:
        rows = wall_by_range[sname]
        if not rows:
            continue
        axe.plot([r["centre"] for r in rows], [r["baseline"] for r in rows],
                 **{**BASE, "color": SLICE_COLORS[sname], "alpha": .45})
        axe.plot([r["centre"] for r in rows], [r["rmse"] for r in rows],
                 color=SLICE_COLORS[sname], label=sname.capitalize(), **QUAD)
    axe.set_xlabel("Range (mm)")
    axe.set_ylabel("Wall RMSE (mm)")
    axe.set_title("Wall depth")
    axe.legend(frameon=False, fontsize=6, loc="upper left",
               bbox_to_anchor=(0.0, 0.95), labelspacing=.3,
               title="dotted: constant predictor", title_fontsize=6)
    _panel_letter(axe, "E")

    axf.plot([0.5, 1.0], [50, 100], "--", color="0.4", lw=.8, zorder=0)
    axf.plot([r["conf"] for r in rel], [100 * r["acc"] for r in rel],
             color=C_CLASS, label="Quadrant models", **QUAD)
    if dep_rel:
        axf.plot([r["conf"] for r in dep_rel], [100 * r["acc"] for r in dep_rel],
                 color=C_CLASS, label="Deployed model", **DEP)
    axf.legend(frameon=False, fontsize=6, loc="upper left",
               bbox_to_anchor=(0.0, 0.95))
    axf.set_xlabel("Reported probability")
    axf.set_ylabel("Correct (%)")
    axf.set_xlim(0.5, 1.0)
    axf.set_ylim(40, 100)
    axf.set_title(f"Reliability (ECE {ece:.3f})")
    _panel_letter(axf, "F")

    style.save(fig, NAME)

    with open(SCRIPTS / f"{NAME}_numbers.json", "w") as f:
        json.dump(numbers, f, indent=1)

    print(f"[fig] {numbers['n_echoes']} echoes, out-of-fold accuracy "
          f"{100 * numbers['overall_accuracy']:.1f}%, ECE {ece:.3f}")
    print(f"[fig] numbers -> {SCRIPTS / (NAME + '_numbers.json')}")


if __name__ == "__main__":
    main()
