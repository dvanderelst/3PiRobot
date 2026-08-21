#!/usr/bin/env python3
"""
EXPT_replay_exp1_inverse.py

Replay the CURRENT inverse over the stored envelopes of the Experiment 1 sonar
runs, and compare it per-step against the model those runs actually flew.

Why this exists. Experiment 1 (2026-07-30, 20 runs, 20/20 success) ran on the
1 m-capped, pre-Acquisition06 deploy model. The inverse has since been uncapped
(2026-08-08) and retrained on Acq01A-06A with a class-agnostic range head
(2026-08-12), so the paper currently characterises one model and reports
behaviour obtained with another. Re-running the sonar arm fixes that, but two
of the current model's known properties cut the wrong way in an open arena:

  - class and azimuth die beyond ~1.4 m while the model is no longer capped, so
    it can assert "pole" out where the call is close to a coin flip;
  - 1400-1700 mm is the worst band anywhere.

Exp 1's controller *turns toward* perceived poles, so the exposed number is
pole PRECISION (97.9% in the runs as flown: 2 false poles in 97 calls). This
script answers "does precision fall" before any robot time is spent, because
every step of those runs stored its raw envelope.

What it CANNOT answer. The poses come from the old runs. A different class call
means a different action means a different pose from that step onward, so this
is per-step perception only -- not a prediction of trajectories, step counts or
outcomes. Treat a good result here as "the re-run is not obviously risky", not
as "the re-run will succeed".

Ground truth is recomputed from each run's own arena_features.npz via
`nearest_reflector_in_cone`, exactly as Paper/images/scripts/exp1_stats.py does
it, and for the same reason: the logged referee columns are nearest-over-all-
directions (they exist for collision scoring), not nearest-in-cone.

Usage:
    python3 EXPT_replay_exp1_inverse.py                 # all 10 sonar runs
    python3 EXPT_replay_exp1_inverse.py P1_S3           # one run
"""

import glob
import os
import sys
from collections import Counter

import dill
import numpy as np

from Library.AcquisitionSessionLoader import nearest_reflector_in_cone
from Library.SonarModel import InverseModel
from SCRIPT_RunDirectPolicy import POLE_RANGE_VETO_MM, feature_from_inverse

# ── Config ────────────────────────────────────────────────────────────────────
INVERSE_FOLD  = "deploy"      # the model the re-run would fly
MODEL_DIR     = "SonarModel"
CONE_HALF_DEG = 35.0
OLD_HORIZON_MM = 1000.0       # abstain horizon of the model the runs flew
STOP_MM       = 400.0         # APPROACH_STOP_MM as the runs were flown (HEAD carries 500)
FAR_MM        = 1400.0        # beyond here class/azimuth are unreliable (2026-08-12)
GATE_TAU      = 0.70          # confidence gate on the pole call; see the sweep below

POLES  = ("P1", "P2")
STARTS = (1, 2, 3, 4, 5)
BANDS  = ((0, 400), (400, 600), (600, 800), (800, 1000),
          (1000, 1400), (1400, 2000), (2000, np.inf))

RUN_ROOTS = ("PolicyRuns", os.path.join("PolicyRuns", "old_stuff"))


def find_run(pole, start):
    """The 2026-07-30 runs have since been moved under PolicyRuns/old_stuff/."""
    name = f"direct_{pole}_S{start}_sonar_repeat01"
    for root in RUN_ROOTS:
        d = os.path.join(root, name)
        if os.path.isdir(d):
            return d
    raise SystemExit(f"run {name} not found under {' or '.join(RUN_ROOTS)}")


def load_geometry(run_dir):
    z = np.load(os.path.join(run_dir, "arena_features.npz"))
    xy = np.column_stack([z["x_mm"], z["y_mm"]])
    kind = z["kind"]
    return xy[kind == 0], xy[kind == 1], float(z["pole_radius_mm"])


def truth(walls, poles, prad, x, y, yaw, horizon=None):
    """(class, distance) of the nearest reflector in the forward cone.

    With `horizon`, a reflector beyond it reads "empty" -- the abstain class
    the capped model was trained to emit, so a correct abstention scores as
    correct. Without it, the raw geometry.
    """
    cls, _, dist = nearest_reflector_in_cone(
        walls, poles, prad, x, y, yaw, CONE_HALF_DEG)
    if not np.isfinite(cls) or not np.isfinite(dist):
        return "empty", np.nan
    if horizon is not None and dist > horizon:
        return "empty", dist
    return ("wall" if cls == 0 else "pole"), dist


def replay_run(run_dir, inverse):
    """Per-step records: pose, truth, the flown model's call, the current one's."""
    walls, poles, prad = load_geometry(run_dir)
    steps = []
    for path in sorted(glob.glob(os.path.join(run_dir, "data*.dill"))):
        with open(path, "rb") as fh:
            d = dill.load(fh)["data"]
        pos = d["position"]
        x, y, yaw = pos["x"], pos["y"], pos["yaw_deg"]
        if not all(np.isfinite(v) for v in (x, y, yaw)):
            continue

        sd = np.asarray(d["sonar_package"]["sonar_data"], dtype=np.float32)
        pred = inverse.predict_from_envelope(sd[:, 1], sd[:, 2])
        new = feature_from_inverse(pred)
        p_pole = float(pred.get("p_pole", np.nan))
        p_none = float(pred.get("p_none", np.nan))

        t_raw, dist = truth(walls, poles, prad, x, y, yaw, None)
        t_cap, _    = truth(walls, poles, prad, x, y, yaw, OLD_HORIZON_MM)
        steps.append(dict(
            step=int(d["step"]),
            true_cls=t_raw, true_cls_capped=t_cap, true_dist=dist,
            old_cls=d["feature"]["cls"] or "empty",
            new_cls=new.cls,
            new_pole_dist=float(new.pole_dist_mm) if new.cls == "pole" else np.nan,
            p_pole=p_pole, p_none=p_none, p_wall=1.0 - p_pole - p_none,
            agn_dist=float(pred.get("agn_dist_mm", np.nan)),
        ))
    return steps


def gated(s, tau, agn_max=np.inf):
    """The pole call under a confidence gate instead of a bare argmax.

    Gate only the POLE call and let a rejected one fall back to the runner-up
    over {wall, none}, so the robot keeps wall-following rather than dropping
    into the scan path -- wall-following is what keeps it off the walls.

    `agn_max` additionally refuses a pole call when the class-agnostic range
    head says the nearest reflector is further out than the range at which
    class is trustworthy. The range head is the one output validated past
    1.4 m (to 2579 mm at ~11%), so it can police the class head.
    """
    if s["p_pole"] >= tau and not (s["agn_dist"] > agn_max):
        return "pole"
    return "wall" if s["p_wall"] >= s["p_none"] else "empty"


def band_of(dist):
    if not np.isfinite(dist):
        return None
    for lo, hi in BANDS:
        if lo <= dist < hi:
            return (lo, hi)
    return None


def report(all_steps, per_run):
    n = len(all_steps)
    print(f"\n=== {n} sonar steps over {len(per_run)} runs "
          f"({', '.join(sorted(per_run))}) ===")
    print(f"  'current' = the production feature_from_inverse, i.e. argmax with "
          f"POLE_RANGE_VETO_MM = {POLE_RANGE_VETO_MM}.")
    print("  The two sweeps below explore rules independently of it: their tau")
    print("  rows are argmax + tau, their cap rows argmax + veto. Compare them to")
    print("  'flown', not to 'current', which is already one of them.")

    # ── Agreement, on both truth conventions ──────────────────────────────────
    print("\n-- class agreement with ground truth --")
    print(f"{'':22s} {'flown (capped)':>16s} {'current':>10s}")
    for label, key in (("truth capped at 1 m", "true_cls_capped"),
                       ("raw truth (no cap)",  "true_cls")):
        old = 100 * np.mean([s["old_cls"] == s[key] for s in all_steps])
        new = 100 * np.mean([s["new_cls"] == s[key] for s in all_steps])
        print(f"  {label:20s} {old:15.1f}% {new:9.1f}%")
    print("  (the capped row is the paper's convention and flatters the flown")
    print("   model, which was built to abstain past 1 m; the raw row is what")
    print("   an uncapped model is actually being asked to do.)")

    # ── Pole precision: the exposed number ────────────────────────────────────
    print("\n-- pole calls and precision (raw truth: was a pole really nearest?) --")
    for tag, key in (("flown  ", "old_cls"), ("current", "new_cls")):
        calls = [s for s in all_steps if s[key] == "pole"]
        good = [s for s in calls if s["true_cls"] == "pole"]
        far = [s for s in calls if s["true_cls"] != "pole"
               and (not np.isfinite(s["true_dist"]) or s["true_dist"] > FAR_MM)]
        prec = 100 * len(good) / len(calls) if calls else float("nan")
        print(f"  {tag}: {len(calls):4d} pole calls, {prec:5.1f}% precision, "
              f"{len(calls) - len(good):3d} false "
              f"({len(far)} of them past {FAR_MM:.0f} mm or on an empty cone)")

    # ── Recall by true range ──────────────────────────────────────────────────
    print("\n-- pole recall by true range (steps where a pole IS nearest in cone) --")
    print(f"{'band (mm)':>14s} {'n':>5s} {'flown':>8s} {'current':>9s} "
          f"{f'gated {GATE_TAU:.2f}':>12s}")
    for lo, hi in BANDS:
        sel = [s for s in all_steps
               if s["true_cls"] == "pole" and band_of(s["true_dist"]) == (lo, hi)]
        if not sel:
            continue
        o = 100 * np.mean([s["old_cls"] == "pole" for s in sel])
        c = 100 * np.mean([s["new_cls"] == "pole" for s in sel])
        g = 100 * np.mean([gated(s, GATE_TAU) == "pole" for s in sel])
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:.0f}"
        print(f"  {lo:5.0f}-{hi_s:>5s} {len(sel):5d} {o:7.1f}% {c:8.1f}% {g:11.1f}%")

    # ── Wall/empty confusion, current model only ──────────────────────────────
    print("\n-- current model, raw-truth confusion (rows = truth) --")
    order = ("wall", "pole", "empty")
    print(f"{'':10s}" + "".join(f"{c:>9s}" for c in order))
    for t in order:
        row = Counter(s["new_cls"] for s in all_steps if s["true_cls"] == t)
        print(f"  {t:8s}" + "".join(f"{row.get(c, 0):9d}" for c in order))

    # ── First detection, per run ──────────────────────────────────────────────
    print("\n-- first 'pole' call per run, and the true pole range at that step --")
    print(f"{'run':10s} {'flown step':>11s} {'range':>8s}   {'current step':>13s} "
          f"{'range':>8s}   {f'gated {GATE_TAU:.2f}':>12s} {'range':>8s}")
    for name in sorted(per_run):
        steps = per_run[name]
        out = []
        for call in (lambda s: s["old_cls"], lambda s: s["new_cls"],
                     lambda s: gated(s, GATE_TAU)):
            hit = next((s for s in steps
                        if call(s) == "pole" and s["true_cls"] == "pole"), None)
            out.append((f"{hit['step']:d}", f"{hit['true_dist']:.0f}")
                       if hit else ("none", "-"))
        print(f"  {name:8s} {out[0][0]:>11s} {out[0][1]:>8s}   "
              f"{out[1][0]:>13s} {out[1][1]:>8s}   {out[2][0]:>12s} {out[2][1]:>8s}")

    # ── Spurious terminal stops ───────────────────────────────────────────────
    spurious = [s for s in all_steps
                if s["new_cls"] == "pole" and np.isfinite(s["new_pole_dist"])
                and s["new_pole_dist"] <= STOP_MM
                and (s["true_cls"] != "pole" or s["true_dist"] > 600)]
    print(f"\n-- terminal stop: steps where the current model would fire the "
          f"{STOP_MM:.0f} mm stop --")
    print(f"   with no pole truly within 600 mm: {len(spurious)} of {n} steps")
    print("   (the pole-range head is masked at 1 m and saturates near 740 mm,")
    print("    so a distant phantom should not be able to reach the stop.)")

    # ── Confidence gate on the pole call ──────────────────────────────────────
    # The fix the handoff has carried since 2026-08-13: replace the bare argmax
    # in feature_from_inverse with a threshold. See `gated` above.
    print("\n-- confidence gate: call 'pole' only when p_pole >= tau --")
    print(f"{'tau':>6s} {'calls':>6s} {'precision':>10s} {'recall<1m':>10s} "
          f"{'recall<1.4m':>12s} {'false past 1.4m':>16s}")
    near = [s for s in all_steps if s["true_cls"] == "pole" and s["true_dist"] <= 1000]
    mid  = [s for s in all_steps if s["true_cls"] == "pole" and s["true_dist"] <= 1400]
    for tau in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        calls = [s for s in all_steps if gated(s, tau) == "pole"]
        good = [s for s in calls if s["true_cls"] == "pole"]
        far = [s for s in calls if s["true_cls"] != "pole"
               and (not np.isfinite(s["true_dist"]) or s["true_dist"] > FAR_MM)]
        prec = 100 * len(good) / len(calls) if calls else float("nan")
        r1 = 100 * np.mean([gated(s, tau) == "pole" for s in near]) if near else float("nan")
        r14 = 100 * np.mean([gated(s, tau) == "pole" for s in mid]) if mid else float("nan")
        print(f"  {tau:4.2f} {len(calls):6d} {prec:9.1f}% {r1:9.1f}% {r14:11.1f}% "
              f"{len(far):15d}")
    print("  (tau = 0 is the current argmax behaviour. The flown model's numbers")
    print("   to beat: 77 calls, 97.4% precision, and recall of 25% in 800-1000")
    print("   with nothing at all beyond 1 m.)")

    # ── Gate + agnostic-range veto ────────────────────────────────────────────
    # A pure confidence gate buys precision by discarding exactly the long-range
    # detections that motivate the re-run. The agnostic range head is the one
    # output validated past 1.4 m, so let it veto the class head instead.
    print("\n-- gate + agnostic-range veto: 'pole' only when p_pole >= tau AND "
          "agn_dist <= cap --")
    print(f"{'tau':>6s} {'cap':>6s} {'calls':>6s} {'precision':>10s} {'recall<1m':>10s} "
          f"{'recall<1.4m':>12s} {'false past 1.4m':>16s}")
    # tau 0.50 IS the argmax: every argmax-pole call in these runs scores
    # p_pole >= 0.5, so that row is 'argmax + veto' and isolates the veto.
    for tau in (0.50, 0.60, 0.70):
        for cap in (900.0, 1000.0, 1200.0, 1400.0):
            calls = [s for s in all_steps if gated(s, tau, cap) == "pole"]
            good = [s for s in calls if s["true_cls"] == "pole"]
            far = [s for s in calls if s["true_cls"] != "pole"
                   and (not np.isfinite(s["true_dist"]) or s["true_dist"] > FAR_MM)]
            prec = 100 * len(good) / len(calls) if calls else float("nan")
            r1 = 100 * np.mean([gated(s, tau, cap) == "pole" for s in near]) if near else np.nan
            r14 = 100 * np.mean([gated(s, tau, cap) == "pole" for s in mid]) if mid else np.nan
            print(f"  {tau:4.2f} {cap:6.0f} {len(calls):6d} {prec:9.1f}% {r1:9.1f}% "
                  f"{r14:11.1f}% {len(far):15d}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    calls = [s for s in all_steps if s["new_cls"] == "pole"]
    prec = 100 * np.mean([s["true_cls"] == "pole" for s in calls]) if calls else 0.0
    old_calls = [s for s in all_steps if s["old_cls"] == "pole"]
    old_prec = (100 * np.mean([s["true_cls"] == "pole" for s in old_calls])
                if old_calls else 0.0)
    far_now = [s for s in all_steps if s["new_cls"] == "pole"
               and s["true_cls"] != "pole"
               and (not np.isfinite(s["true_dist"]) or s["true_dist"] > FAR_MM)]
    print("\n=== verdict ===")
    print(f"  pole precision {old_prec:.1f}% (flown) -> {prec:.1f}% (current), "
          f"{len(far_now)} false calls past {FAR_MM:.0f} mm")
    if POLE_RANGE_VETO_MM is not None and len(far_now) <= 5 and not spurious:
        print(f"  GO. The {POLE_RANGE_VETO_MM:.0f} mm veto holds the far-range")
        print("  phantoms down and nothing can reach the terminal stop. Precision")
        print("  sits below the flown model's by design -- that is the price of")
        print("  reopening the 800-1000 mm band, where recall goes 25% -> 81%.")
    elif prec < old_prec - 5:
        print("  Do NOT re-run on the bare argmax: the controller turns toward")
        print("  perceived poles, and this many false calls would send it after")
        print("  phantoms. Gate the pole call first. From the two sweeps above:")
        print("    argmax + veto 1200  -> the knee. 88.2% precision, 800-1000 mm")
        print("                           recall 25% -> 81%, median first detection")
        print("                           798 -> 995 mm, and only 2 false calls past")
        print("                           1.4 m against 55 with no veto.")
        print("    argmax + veto 1000  -> conservative. 95.3% precision, but gives")
        print("                           back half the 800-1000 mm gain (43.8%).")
        print("    veto 1400           -> +6 points of recall for 6x the far")
        print("                           phantoms (12 vs 2). Not worth it.")
        print("    tau alone (0.70)    -> restores precision by discarding exactly")
        print("                           the range gain that motivates the re-run.")
        print("  The gate lives in feature_from_inverse, which only the direct")
        print("  policy uses -- SCRIPT_RunPolicy feeds encode_obs the raw dict, so")
        print("  none of this touches the Experiment 2 artifacts.")
    elif spurious:
        print("  CAUTION: a phantom can reach the terminal stop; check those steps.")
    else:
        print("  GO: precision holds and no phantom can reach the terminal stop.")
    print("\n  Per-step only. Poses are from the flown runs, so closed-loop")
    print("  behaviour would diverge from step one; this is not a prediction of")
    print("  outcomes, path length or step count.")


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else None
    inverse = InverseModel.load(model_dir=MODEL_DIR, fold=INVERSE_FOLD, device="cpu")
    print(f"Current inverse: {inverse}")
    if inverse.pole_dist_divisor is None:
        raise SystemExit("This fold has no pole-range head; the terminal stop "
                         "could never fire and the comparison is meaningless.")

    per_run, all_steps = {}, []
    for pole in POLES:
        for start in STARTS:
            name = f"{pole}_S{start}"
            if want and want not in (name, f"direct_{name}_sonar_repeat01"):
                continue
            d = find_run(pole, start)
            steps = replay_run(d, inverse)
            per_run[name] = steps
            all_steps += steps
            # Every run stores exactly two fewer dills than trajectory.tsv has
            # rows -- systematically, in all ten. So the replay covers 761 of
            # the 781 steps exp1_stats.py scores. Reported, not silently
            # absorbed, because the two counts otherwise look like a bug.
            with open(os.path.join(d, "trajectory.tsv")) as fh:
                n_tsv = sum(1 for _ in fh) - 1
            print(f"  {name}: {len(steps)} envelopes of {n_tsv} logged steps "
                  f"from {d}")
    if not all_steps:
        raise SystemExit(f"no runs matched {want!r}")
    report(all_steps, per_run)


if __name__ == "__main__":
    main()
