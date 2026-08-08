"""Does the inverse need its 1 m range cap?

Since 2026-06-12 the inverse has been trained with `MAX_RANGE_MM = 1000`, and
pings whose nearest in-cone reflector lies beyond that are relabelled to the
'none' class. These arenas contain zero genuinely empty cones, so 'none' is not
"nothing there" -- it is "further than 1 m". The sensors do return those echoes;
the label throws the information away.

This experiment asks whether the far pings are better kept at their true class,
letting the classifier express distance as declining confidence instead of as a
hard cut. Two conditions, identical data, folds, architecture and seed, differing
only in how beyond-1 m pings are labelled:

    cap1m      FAR_LABEL_MODE="none"        (the current canonical behaviour)
    fullrange  FAR_LABEL_MODE="true_class"  (the proposal)

4-fold quadrant CV in both, so every ping is scored exactly once by a model that
did not train on it. Aggregate accuracy is NOT the question here -- the two
conditions are answering different questions beyond 1 m, so their headline
numbers are not comparable. The two that matter are:

  1. Below 1 m, does fullrange cost anything? Same task, same pings, directly
     comparable. This is the regression check.
  2. Beyond 1 m, is fullrange above chance, and -- decisive -- is it CALIBRATED?
     Graceful degradation is only useful if the model reports its own
     uncertainty honestly. A confidently wrong classifier at 2 m is worse than
     an abstain, because the controller cannot tell the two apart.

The headline statistic is the overconfidence gap (mean confidence minus
accuracy) per range band. Near zero means the model knows what it does not know.

Writes to TempOutput/RangeHorizon/ (gitignored). Fold artifacts go to
SonarModel/expt_rh_* so nothing named inverse_* is touched -- the deployed model
and its shared feature_params are left alone.

Run:  .venv/bin/python3 EXPT_range_horizon.py
"""

import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import SCRIPT_TrainInverseModel as T


# ── Settings ──────────────────────────────────────────────────────────────────

# Each condition is (tag, FAR_LABEL_MODE, MAX_RANGE_MM). The cap and the
# labelling rule are separate knobs: cap1700 keeps the abstain class but moves
# the cut out to where this dataset actually ends, which is the configuration
# the first two conditions bracket without either one testing it.
CONDITIONS = [
    ("cap1m",     "none",       1000.0),
    ("fullrange", "true_class", 1000.0),
    ("cap1700",   "none",       1700.0),
]

# Subset on which conditions are compared head to head. Fixed at the original
# cap so the comparison does not move when MAX_RANGE_MM does -- otherwise each
# condition would be scored on a different set of pings.
COMPARE_BELOW_MM = 1000.0

# Band edges follow the 2026-05-21 CNN-vs-distance stratification, which is the
# evidence that motivated the 1 m cut: the CNN beat the hand-feature baseline at
# every band below 1700 mm and collapsed to chance beyond it.
RANGE_BANDS = [(0.0, 500.0), (500.0, 1000.0), (1000.0, 1700.0), (1700.0, np.inf)]

CONF_BINS = np.linspace(0.0, 1.0, 11)   # reliability-diagram bins
OUT_DIR   = os.path.join("TempOutput", "RangeHorizon")


# ── Metrics ───────────────────────────────────────────────────────────────────

def band_label(lo, hi):
    return f"{lo:.0f}-{'inf' if not np.isfinite(hi) else f'{hi:.0f}'}"


def class_metrics(true_c, pred_c, probs):
    """Accuracy, balanced accuracy, per-class recall, and the calibration
    summary for one subset of pings."""
    n = len(true_c)
    if n == 0:
        return None
    acc = float((pred_c == true_c).mean())
    recalls = {}
    for ci, name in enumerate(T.CLASS_NAMES):
        m = true_c == ci
        if m.any():
            recalls[name] = float((pred_c[m] == ci).mean())
    bal = float(np.mean(list(recalls.values()))) if recalls else float("nan")

    conf = probs.max(axis=1)
    correct = (pred_c == true_c).astype(float)
    # Expected calibration error: |confidence - accuracy| averaged over
    # confidence bins, weighted by bin occupancy.
    ece, bins = 0.0, []
    idx = np.digitize(conf, CONF_BINS[1:-1], right=False)
    for b in range(len(CONF_BINS) - 1):
        m = idx == b
        if not m.any():
            continue
        c_mean, a_mean = float(conf[m].mean()), float(correct[m].mean())
        ece += (m.sum() / n) * abs(c_mean - a_mean)
        bins.append({"bin": [float(CONF_BINS[b]), float(CONF_BINS[b + 1])],
                     "n": int(m.sum()), "confidence": c_mean, "accuracy": a_mean})
    return {
        "n": int(n),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "per_class_recall": recalls,
        "n_true_per_class": {name: int((true_c == ci).sum())
                             for ci, name in enumerate(T.CLASS_NAMES)},
        "mean_confidence": float(conf.mean()),
        # The headline: positive means the model claims more certainty than it
        # earns. Near zero is what makes graceful degradation usable.
        "overconfidence_gap": float(conf.mean() - acc),
        "ece": float(ece),
        "reliability_bins": bins,
    }


def sigma_metrics(true_v, pred_v, pred_s):
    """Is a regression head's predicted sigma honest? z = residual / sigma
    should have std 1 and 68.3% of its mass inside +-1."""
    v = np.isfinite(true_v) & np.isfinite(pred_v) & np.isfinite(pred_s) & (pred_s > 0)
    if v.sum() < 5:
        return None
    resid = pred_v[v] - true_v[v]
    z = resid / pred_s[v]
    return {
        "n": int(v.sum()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "mae": float(np.abs(resid).mean()),
        "sigma_median": float(np.median(pred_s[v])),
        "z_std": float(z.std()),                       # target 1.0
        "frac_within_1sigma": float((np.abs(z) < 1).mean()),   # target 0.683
    }


# ── One condition ─────────────────────────────────────────────────────────────

def run_condition(tag, far_mode, max_range, device):
    print(f"\n{'=' * 78}\n  CONDITION {tag}  (FAR_LABEL_MODE={far_mode}, "
          f"MAX_RANGE_MM={max_range:.0f})\n{'=' * 78}")
    T.FAR_LABEL_MODE  = far_mode
    T.MAX_RANGE_MM    = max_range
    T.ARTIFACT_PREFIX = f"expt_rh_{tag}"
    np.random.seed(T.SEED)
    torch.manual_seed(T.SEED)

    sonar, slice_t, classes, pole_az_deg, near_dist_mm, quads, sess, _ = T.load_and_filter()
    print(f"  {len(sonar)} pings: "
          + ", ".join(f"{name}={int((classes == ci).sum())}"
                      for ci, name in enumerate(T.CLASS_NAMES)))

    pole_az_n = (pole_az_deg / T.CONE_HALF_DEG).astype(np.float32)
    pole_az_n_safe = np.where(np.isnan(pole_az_n), 0.0, pole_az_n).astype(np.float32)
    pole_dist_n = (near_dist_mm / T.POLE_DIST_NORM_MM).astype(np.float32)
    pole_dist_n_safe = np.where(np.isnan(pole_dist_n), 0.0, pole_dist_n).astype(np.float32)

    sink = []
    for q in T.CV_QUADRANTS:
        print(f"\n--- {tag} fold q={q} ---")
        # NB: main()'s CV path does NOT pass the pole-distance arguments, so it
        # silently leaves the range head untrained. Passed explicitly here.
        T.run_fold(q, sonar, slice_t, classes, pole_az_deg, pole_az_n_safe,
                   quads, sess, device, sub_prefix=f"q{q}",
                   pole_dist_mm=near_dist_mm, pole_dist_n_safe=pole_dist_n_safe,
                   oof_sink=sink)

    n = len(sonar)
    oof = {
        "probs":     np.full((n, len(T.CLASS_NAMES)), np.nan),
        "pred":      np.full(n, -1, dtype=np.int64),
        "az":        np.full(n, np.nan), "az_std":   np.full(n, np.nan),
        "pdist":     np.full(n, np.nan), "pdist_std": np.full(n, np.nan),
        "wall":      np.full((n, len(T.SLICE_NAMES)), np.nan),
        "wall_std":  np.full((n, len(T.SLICE_NAMES)), np.nan),
    }
    seen = np.zeros(n, dtype=int)
    for rec in sink:
        i, p = rec["val_idx"], rec["pred"]
        seen[i] += 1
        oof["probs"][i]    = p["cls_probs"]
        oof["pred"][i]     = p["cls_pred"]
        oof["az"][i]       = p["pole_pred_az_deg"]
        oof["az_std"][i]   = p["pole_pred_az_std"]
        oof["wall"][i]     = p["wall_pred_mean"]
        oof["wall_std"][i] = p["wall_pred_std"]
        if p.get("pole_pred_dist_mm") is not None:
            oof["pdist"][i]     = p["pole_pred_dist_mm"]
            oof["pdist_std"][i] = p["pole_pred_dist_std"]
    # Every ping must be held out exactly once, or the "out-of-fold" framing is
    # a lie and some pings would be scored by a model that trained on them.
    if not (seen == 1).all():
        raise RuntimeError(
            f"OOF coverage broken: {int((seen == 0).sum())} pings never held out, "
            f"{int((seen > 1).sum())} held out more than once")

    np.savez_compressed(
        os.path.join(OUT_DIR, f"oof_{tag}.npz"),
        true_class=classes, near_dist_mm=near_dist_mm, pole_az_deg=pole_az_deg,
        slice_targets=slice_t, quads=quads, sess=sess, **oof)
    return classes, near_dist_mm, pole_az_deg, slice_t, oof


def load_oof(tag):
    """Rebuild a condition's analysis inputs from its saved out-of-fold npz,
    in the same shape run_condition returns, so no retraining is needed to
    re-score a condition under updated metric code."""
    d = np.load(os.path.join(OUT_DIR, f"oof_{tag}.npz"), allow_pickle=True)
    oof = {k: d[k] for k in ("probs", "pred", "az", "az_std",
                             "pdist", "pdist_std", "wall", "wall_std")}
    return (d["true_class"], d["near_dist_mm"], d["pole_az_deg"],
            d["slice_targets"], oof)


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(tag, classes, near_dist_mm, pole_az_deg, slice_t, oof):
    report = {"condition": tag, "overall": None, "by_range_band": {}}
    report["overall"] = class_metrics(classes, oof["pred"], oof["probs"])

    for lo, hi in RANGE_BANDS:
        m = np.isfinite(near_dist_mm) & (near_dist_mm >= lo) & (near_dist_mm < hi)
        if not m.any():
            continue
        entry = {"classification": class_metrics(classes[m], oof["pred"][m],
                                                 oof["probs"][m])}
        # Regression heads, restricted to the pings each head is trained on.
        pm = m & (classes == T.POLE_CLASS)
        entry["pole_azimuth_deg"] = sigma_metrics(
            pole_az_deg[pm], oof["az"][pm], oof["az_std"][pm])
        entry["pole_range_mm"] = sigma_metrics(
            near_dist_mm[pm], oof["pdist"][pm], oof["pdist_std"][pm])
        wm = m & (classes == T.WALL_CLASS)
        entry["wall_slices_mm"] = {
            name: sigma_metrics(slice_t[wm, i], oof["wall"][wm, i],
                                oof["wall_std"][wm, i])
            for i, name in enumerate(T.SLICE_NAMES)}
        report["by_range_band"][band_label(lo, hi)] = entry

    # The comparable subset: below COMPARE_BELOW_MM every condition faces the
    # same wall-vs-pole problem on the same pings, so this is where "did moving
    # the horizon cost anything?" can actually be answered.
    #
    # Raw accuracy here is NOT a like-for-like measure of discrimination: a
    # condition that can predict 'none' on a close ping is charged for an error
    # a condition without a 'none' class cannot commit. So the abstention is
    # separated out. `accuracy_given_committed` is the discrimination quality;
    # `abstain_rate` is the self-inflicted blindness. The first run showed those
    # two pulling in opposite directions, which the headline number hid.
    m = np.isfinite(near_dist_mm) & (near_dist_mm < COMPARE_BELOW_MM)
    c_m, p_m = classes[m], oof["pred"][m]
    committed = p_m != T.NONE_CLASS
    abstained_truth = {name: int((c_m[~committed] == ci).sum())
                       for ci, name in enumerate(T.CLASS_NAMES)}
    report["below_cap"] = {
        "compare_below_mm": COMPARE_BELOW_MM,
        "classification": class_metrics(c_m, p_m, oof["probs"][m]),
        "abstain_rate": float((~committed).mean()),
        "n_abstained": int((~committed).sum()),
        "abstained_true_class": abstained_truth,
        "abstained_true_range_median_mm": (float(np.median(near_dist_mm[m][~committed]))
                                           if (~committed).any() else None),
        "accuracy_given_committed": (float((p_m[committed] == c_m[committed]).mean())
                                     if committed.any() else None),
        "recall_given_committed": {
            name: float((p_m[committed][c_m[committed] == ci] == ci).mean())
            for ci, name in enumerate(T.CLASS_NAMES)
            if (c_m[committed] == ci).any()},
        "pole_azimuth_deg": sigma_metrics(
            pole_az_deg[m & (classes == T.POLE_CLASS)],
            oof["az"][m & (classes == T.POLE_CLASS)],
            oof["az_std"][m & (classes == T.POLE_CLASS)]),
        "pole_range_mm": sigma_metrics(
            near_dist_mm[m & (classes == T.POLE_CLASS)],
            oof["pdist"][m & (classes == T.POLE_CLASS)],
            oof["pdist_std"][m & (classes == T.POLE_CLASS)]),
    }
    with open(os.path.join(OUT_DIR, f"report_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def print_report(tag, rep):
    print(f"\n{'-' * 78}\n  {tag}\n{'-' * 78}")
    o = rep["overall"]
    print(f"  overall (all ranges): acc={o['accuracy']*100:.1f}%  "
          f"bal={o['balanced_accuracy']*100:.1f}%  n={o['n']}")
    bc = rep["below_cap"]
    b = bc["classification"]
    print(f"  below {bc['compare_below_mm']:.0f} mm:      acc={b['accuracy']*100:.1f}%  "
          f"bal={b['balanced_accuracy']*100:.1f}%  n={b['n']}  "
          f"gap={b['overconfidence_gap']*+100:+.1f} pp")
    if bc["accuracy_given_committed"] is not None:
        rec = "  ".join(f"{k} rec={v*100:.1f}"
                        for k, v in bc["recall_given_committed"].items())
        truth = ", ".join(f"{k}={v}" for k, v in bc["abstained_true_class"].items() if v)
        print(f"    abstained on {bc['n_abstained']} "
              f"({bc['abstain_rate']*100:.1f}%)"
              + (f" -- truly {truth}, median true range "
                 f"{bc['abstained_true_range_median_mm']:.0f} mm"
                 if bc["n_abstained"] else ""))
        print(f"    accuracy given it committed: "
              f"{bc['accuracy_given_committed']*100:.1f}%   {rec}")
    print(f"\n  {'band (mm)':>12} {'n':>5} {'acc':>7} {'bal':>7} {'conf':>7} "
          f"{'gap':>8} {'ECE':>6}   recalls")
    for band, e in rep["by_range_band"].items():
        c = e["classification"]
        rec = "  ".join(f"{k}={v*100:.0f}" for k, v in c["per_class_recall"].items())
        print(f"  {band:>12} {c['n']:>5} {c['accuracy']*100:>6.1f}% "
              f"{c['balanced_accuracy']*100:>6.1f}% {c['mean_confidence']*100:>6.1f}% "
              f"{c['overconfidence_gap']*100:>+7.1f} {c['ece']:>6.3f}   {rec}")
    print(f"\n  {'band (mm)':>12}   pole-az RMSE / z_std    pole-range RMSE / z_std")
    for band, e in rep["by_range_band"].items():
        az, pd_ = e["pole_azimuth_deg"], e["pole_range_mm"]
        az_s = f"{az['rmse']:6.2f}deg / {az['z_std']:.2f} (n={az['n']})" if az else "        --"
        pd_s = f"{pd_['rmse']:6.0f}mm / {pd_['z_std']:.2f} (n={pd_['n']})" if pd_ else "        --"
        print(f"  {band:>12}   {az_s:<24} {pd_s}")


def plot_summary(reports, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    bands = [band_label(lo, hi) for lo, hi in RANGE_BANDS]
    x = np.arange(len(bands))
    colors = {"cap1m": "#377eb8", "fullrange": "#e41a1c", "cap1700": "#4daf4a"}

    ax = axes[0]
    for tag, rep in reports.items():
        y = [rep["by_range_band"].get(b, {}).get("classification", {}).get(
                "balanced_accuracy", np.nan) for b in bands]
        ax.plot(x, np.array(y) * 100, "o-", color=colors.get(tag), label=tag)
    ax.axhline(50, ls=":", c="grey", lw=1)
    ax.text(len(bands) - 1, 51, "chance (2-class)", ha="right", fontsize=8, color="grey")
    ax.set_xticks(x); ax.set_xticklabels(bands, rotation=20)
    ax.set_xlabel("true range to nearest in-cone reflector (mm)")
    ax.set_ylabel("balanced accuracy (%)")
    ax.set_title("Does discrimination survive range?")
    ax.legend(frameon=False)

    ax = axes[1]
    for tag, rep in reports.items():
        y = [rep["by_range_band"].get(b, {}).get("classification", {}).get(
                "overconfidence_gap", np.nan) for b in bands]
        ax.plot(x, np.array(y) * 100, "o-", color=colors.get(tag), label=tag)
    ax.axhline(0, ls="-", c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(bands, rotation=20)
    ax.set_xlabel("true range (mm)")
    ax.set_ylabel("mean confidence - accuracy (pp)")
    ax.set_title("Overconfidence gap (0 = honest)")
    ax.legend(frameon=False)

    ax = axes[2]
    rel_tag = "cap1700" if "cap1700" in reports else "fullrange"
    rep = reports.get(rel_tag)
    if rep:
        for (lo, hi) in RANGE_BANDS:
            b = band_label(lo, hi)
            bins = rep["by_range_band"].get(b, {}).get(
                "classification", {}).get("reliability_bins", [])
            if not bins:
                continue
            ax.plot([d["confidence"] for d in bins], [d["accuracy"] for d in bins],
                    "o-", ms=4, label=f"{b} mm")
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Reliability, {rel_tag}, by range band")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\n  wrote {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(only=None):
    """Run the conditions and summarise. `only` restricts training to the named
    tags; any other condition with a report already on disk is reused, so adding
    a condition costs one condition's compute rather than all of them."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(T.OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   sessions: {T.ACQUISITION_SESSIONS}")

    saved = (T.FAR_LABEL_MODE, T.MAX_RANGE_MM, T.ARTIFACT_PREFIX)
    reports = {}
    try:
        for tag, far_mode, max_range in CONDITIONS:
            npz = os.path.join(OUT_DIR, f"oof_{tag}.npz")
            if only and tag not in only:
                if not os.path.exists(npz):
                    print(f"  skipping {tag} (not selected, no saved predictions)")
                    continue
                # Re-analyse from the saved out-of-fold predictions rather than
                # reloading the old report JSON: the metric code moves, and a
                # cached report silently mixes old definitions with new ones.
                # Re-analysis is seconds and needs no training.
                print(f"  re-analysing {tag} from saved predictions")
                out = load_oof(tag)
            else:
                out = run_condition(tag, far_mode, max_range, device)
            reports[tag] = analyse(tag, *out)
    finally:
        # Never leave the imported module mutated: a later import in the same
        # process would silently inherit the experimental labelling and cap.
        T.FAR_LABEL_MODE, T.MAX_RANGE_MM, T.ARTIFACT_PREFIX = saved

    print(f"\n\n{'=' * 78}\n  SUMMARY\n{'=' * 78}")
    for tag in [c[0] for c in CONDITIONS]:
        if tag in reports:
            print_report(tag, reports[tag])

    if len(reports) > 1:
        print(f"\n{'=' * 78}\n  HEAD TO HEAD (pings closer than "
              f"{COMPARE_BELOW_MM:.0f} mm, identical in every condition)\n{'=' * 78}")
        print(f"  {'condition':>10} {'acc':>7} {'bal':>7} {'abstain':>9} "
              f"{'acc|committed':>14} {'pole rec|comm':>14}")
        for tag in [c[0] for c in CONDITIONS]:
            if tag not in reports:
                continue
            bc = reports[tag]["below_cap"]
            c = bc["classification"]
            agc = bc["accuracy_given_committed"]
            prc = bc["recall_given_committed"].get("pole")
            print(f"  {tag:>10} {c['accuracy']*100:>6.1f}% {c['balanced_accuracy']*100:>6.1f}% "
                  f"{bc['abstain_rate']*100:>8.1f}% "
                  f"{(agc*100 if agc else float('nan')):>13.1f}% "
                  f"{(prc*100 if prc else float('nan')):>13.1f}%")
        print("\n  Raw accuracy conflates two things. 'acc|committed' is discrimination\n"
              "  quality; 'abstain' is how often the cap refuses to answer at all on a\n"
              "  ping that does have a reflector. A horizon change can improve one and\n"
              "  worsen the other, so read both before concluding anything.")

    plot_summary(reports, os.path.join(OUT_DIR, "range_horizon_summary.png"))
    print(f"\nArtifacts in {OUT_DIR}/ (oof_*.npz, report_*.json, summary png)")


if __name__ == "__main__":
    import sys
    # e.g.  EXPT_range_horizon.py --only cap1700
    args = sys.argv[1:]
    sel = args[args.index("--only") + 1:] if "--only" in args else None
    main(only=sel)
