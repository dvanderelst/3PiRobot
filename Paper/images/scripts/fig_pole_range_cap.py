"""Supplementary: why the pole-range head is trained only on poles within 1 m.

The main text states the cap in one clause. This figure is the evidence, so the
argument does not have to be made there.

Two models, identical but for one knob: the range span the pole-range head is
trained on. Both are 4-fold quadrant cross-validations over the same 2775
echoes, so every pole echo is scored once by a model that did not see it.

  capped     poles within 1000 mm  (POLE_DIST_TRAIN_MAX_MM, the deployed setting)
  uncapped   every pole echo, out to 3196 mm

The point the figure has to make is that this is a TRADE, not an oversight. The
uncapped head tracks the pole to 3 m, which the capped one cannot do at all --
past its training span it answers the same ~780 mm whatever the true range. But
it pays for that at close range, where the terminal approach decides to stop at
400 mm, and where an outward bias makes the robot drive nearer than the protocol
intends.

Panels A and B are the predictions; C and D are the cost, as signed bias and as
RMSE against range. The shaded strip in C and D is the band containing the stop.

Requires the fold artifacts from EXPT_pole_range_cap.py
(SonarModel/expt_prc_{capped,uncapped}_q*). Regenerate those first if the data
or the architecture changes.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_pole_range_cap.py
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json
import sys

from paths import CONTROL, SONAR_MODEL, SCRIPTS

sys.path.insert(0, str(CONTROL))
os.chdir(str(CONTROL))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import SCRIPT_TrainInverseModel as T  # noqa: E402
import style  # noqa: E402

NAME = "fig_pole_range_cap"
CONDITIONS = [("expt_prc_cap1000", "Trained on poles within 1 m", "#C44E52"),
              ("expt_prc_uncapped", "Trained on every pole", "#4C72B0")]
BANDS = [(200, 500), (500, 750), (750, 1000), (1000, 1400),
         (1400, 2000), (2000, 3300)]
STOP_MM = 400.0          # the terminal approach threshold this head decides
R_LIM = 3300


def _panel_letter(ax, letter):
    ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


def out_of_fold(prefix, sonar, quads):
    """Predictions from the fold that held each echo out.

    The checkpoints are named expt_prc_*, which InverseModel.load cannot
    address (it builds inverse_{fold}_* paths), so the model is rebuilt from
    the trainer's own architecture constants and the state dict loaded onto it.
    """
    pd = np.full(len(sonar), np.nan)
    for q in T.CV_QUADRANTS:
        ck = torch.load(f"{SONAR_MODEL}/{prefix}_q{q}_best_model.pth",
                        map_location="cpu")
        st = json.load(open(f"{SONAR_MODEL}/{prefix}_q{q}_results.json"))
        model = T.MODEL_CLASS(samples=sonar.shape[1],
                              conv_channels=T.SONAR_CONV_CHANNELS,
                              conv_kernel=T.SONAR_CONV_KERNEL,
                              pool_out=T.SONAR_POOL_OUT,
                              fc_hidden=T.SONAR_FC_HIDDEN,
                              head_hidden=T.SONAR_HEAD_HIDDEN,
                              n_classes=len(T.CLASS_NAMES), **T.MODEL_KWARGS)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        sel = quads == q
        p = T.predict(model, sonar[sel],
                      (st["sonar_norm"]["mean"], st["sonar_norm"]["std"]),
                      (st["target_norm"]["mean"], st["target_norm"]["std"]),
                      torch.device("cpu"))
        pd[sel] = p["pole_pred_dist_mm"]
    return pd


def main():
    style.setup()
    import matplotlib.pyplot as plt

    sonar, slice_t, classes, az, near, quads, sess, _ = T.load_and_filter()
    pole = (classes == 1) & np.isfinite(near)
    truth = near[pole]

    preds, numbers = {}, {}
    for prefix, label, _ in CONDITIONS:
        preds[prefix] = out_of_fold(prefix, sonar, quads)[pole]
        rows = []
        for lo, hi in BANDS:
            m = (truth >= lo) & (truth < hi)
            if m.sum() < 15:
                continue
            e = preds[prefix][m] - truth[m]
            rows.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                             centre=float(truth[m].mean()),
                             bias=float(np.nanmean(e)),
                             rmse=float(np.sqrt(np.nanmean(e ** 2)))))
        numbers[prefix] = dict(label=label, bands=rows)

    fig, axes = plt.subplots(2, 2, figsize=(style.WIDTH_2COL, 5.8))
    (axa, axb), (axc, axd) = axes
    fig.subplots_adjust(hspace=.42, wspace=.30)

    for ax, (prefix, label, colour), letter in zip((axa, axb), CONDITIONS,
                                                   ("A", "B")):
        ax.plot([0, R_LIM], [0, R_LIM], "--", color="0.4", lw=.8, zorder=0)
        ax.scatter(truth, preds[prefix], s=6, alpha=.28, edgecolor="none",
                   color=colour)
        rows = numbers[prefix]["bands"]
        ax.plot([r["centre"] for r in rows], [r["bias"] + r["centre"]
                                              for r in rows],
                "o-", color="k", ms=3.5, lw=1.3, label="Band mean")
        ax.axvline(1000, color="0.3", ls=":", lw=1)
        ax.set_xlim(0, R_LIM); ax.set_ylim(0, R_LIM)
        ax.set_aspect("equal")
        ax.set_xlabel("True pole range (mm)")
        ax.set_ylabel("Predicted (mm)")
        ax.set_title(label, fontsize=8)
        ax.legend(frameon=False, fontsize=6, loc="upper left",
                  bbox_to_anchor=(0.0, 0.94))
        _panel_letter(ax, letter)

    # C and D would both be swamped by the capped head's far-range failure
    # (bias -1687 mm, RMSE 1717) which buries the close-range difference the
    # cap actually exists to protect. C gets an inset over the stop region; D
    # uses a log axis, on which 103 against 279 mm is as legible as 1717.
    for ax, key, ylab, letter in ((axc, "bias", "Signed bias (mm)", "C"),
                                  (axd, "rmse", "RMSE (mm)", "D")):
        ax.axvspan(200, 500, color="0.6", alpha=.16, lw=0, zorder=0)
        if key == "bias":
            ax.axhline(0, color="0.4", lw=.8, zorder=0)
        for prefix, label, colour in CONDITIONS:
            rows = numbers[prefix]["bands"]
            ax.plot([r["centre"] for r in rows], [r[key] for r in rows],
                    "o-", color=colour, ms=4, label=label)
        ax.axvline(1000, color="0.3", ls=":", lw=1)
        ax.set_xlabel("True pole range (mm)")
        ax.set_ylabel(ylab)
        # One legend for the pair: C and D show the same two conditions in the
        # same colours, and C needs its lower-right corner for the inset.
        if key == "rmse":
            ax.legend(frameon=False, fontsize=6, loc="upper left")
            ax.set_yscale("log")
        _panel_letter(ax, letter)

    # Inset on C: the same signed bias over the training span only, which is
    # where the two conditions differ by the amount that matters -- +36 mm
    # against +167 mm in the band holding the stop.
    ins = axc.inset_axes([0.46, 0.13, 0.5, 0.42])
    ins.axhline(0, color="0.4", lw=.7)
    ins.axvspan(200, 500, color="0.6", alpha=.16, lw=0, zorder=0)
    for prefix, label, colour in CONDITIONS:
        rows = [r for r in numbers[prefix]["bands"] if r["hi"] <= 1000]
        ins.plot([r["centre"] for r in rows], [r["bias"] for r in rows],
                 "o-", color=colour, ms=3, lw=1.1)
    ins.set_xlim(150, 1050)
    ins.tick_params(labelsize=5, length=2, pad=1)
    ins.set_title("within the training span", fontsize=5.5, pad=2)
    axc.set_title(f"Shaded: band holding the {STOP_MM:.0f} mm stop", fontsize=7)

    style.save(fig, NAME)
    with open(SCRIPTS / f"{NAME}_numbers.json", "w") as f:
        json.dump(numbers, f, indent=1)
    for prefix, _, _ in CONDITIONS:
        near_row = numbers[prefix]["bands"][0]
        far_row = numbers[prefix]["bands"][-1]
        print(f"[fig] {prefix}: {near_row['lo']}-{near_row['hi']} mm "
              f"bias {near_row['bias']:+.0f} rmse {near_row['rmse']:.0f}   |   "
              f"{far_row['lo']}-{far_row['hi']} mm bias {far_row['bias']:+.0f} "
              f"rmse {far_row['rmse']:.0f}")


if __name__ == "__main__":
    main()
