"""Experiment 1 aggregate statistics: every number quoted in the Results.

Reads the 20 runs of Experiment 1 (5 start poses x 2 pole placements x 2
modalities) from PolicyRuns/Poles/ and recomputes the aggregate measures
reported in the "Experiment 1" Results subsection and in
Table~\\ref{tab:direct-results}. Run it to re-verify the prose against the data
after any re-run.

The sonar arm read here is the 2026-08-24 re-run on the current (uncapped)
inverse with the restored 75 mm approach step; the vision arm is the original
2026-07-30 one, never re-flown. Distinguish generations by the `drive_mm`
column on `approach` rows: 75.0 is current, 150.0 is the void 2026-08-21
generation parked in PolicyRuns/old_stuff/.

Ground truth is recomputed here from each run's own arena_features.npz rather
than read from the trajectory's logged referee columns, because the logged
columns carry the nearest wall and pole over ALL directions (they exist for
collision and success scoring), not the nearest reflector inside the forward
cone that the class comparison needs.

Three conventions worth knowing before reading the numbers:

  - Per-step class agreement is scored against the raw truth, with no horizon
    applied. The inverse used to abstain past 1 m by construction and an older
    version of this script scored it that way; the deployed model is uncapped
    and never emitted the abstain class in any of these runs, so a horizon on
    the ground truth would now only manufacture disagreement.
  - Pole recall is quoted over the range the controller is allowed to act on,
    that is, out to the POLE_RANGE_VETO_MM veto: a pole further than that is
    one the controller refuses by design, not one the model missed.
  - Aggregates over steps are pooled across runs, not averaged over per-run
    fractions. The two differ because runs vary in length by an order of
    magnitude, and one sonar run is the 200-step failure.

Vision reads its feature from the same digitized geometry the referee uses, so
it reproduces the raw truth by construction; `check_vision_identity` asserts
exactly that, and doubles as a test of the ground-truth recomputation.

    Control_code/.venv/bin/python3 Paper/images/scripts/exp1_stats.py
"""

import csv
import json
import sys

import numpy as np

from paths import CONTROL, POLICY_RUNS

sys.path.insert(0, str(CONTROL))

from Library.AcquisitionSessionLoader import nearest_reflector_in_cone  # noqa: E402

# Must match SCRIPT_RunDirectPolicy.py as it stood for these runs; the values
# are also recorded per run in run_summary.json (approach_stop_mm,
# pole_range_veto_mm) and in Table~\ref{tab:controller-params}.
CONE_HALF_DEG = 35.0
VETO_MM = 1200.0         # range beyond which a pole call is refused
STOP_MM = 400.0          # perceived pole range that ends the approach
COLLISION_MM = 20.0      # true wall clearance scored as a collision

POLES = ("P1", "P2")
STARTS = (1, 2, 3, 4, 5)
SOURCES = ("sonar", "vision")
CLASSES = ("wall", "pole", "empty")
RANGE_BANDS = (0, 400, 800, 1200, 1600, 2000, 10000)


def run_dir(pole, start, source):
    """Finished runs are filed under PolicyRuns/Poles/.

    A fresh run writes to PolicyRuns/ and is moved there afterwards, so the
    root is checked as a fallback and a mid-session re-run resolves without
    editing this file.
    """
    name = f"direct_{pole}_S{start}_{source}_repeat01"
    filed = POLICY_RUNS / "Poles" / name
    return filed if filed.is_dir() else POLICY_RUNS / name


def _num(s):
    return float(s) if s not in ("", None) else np.nan


def load_geometry(d):
    """Walls, poles and pole radius as the referee sees them."""
    z = np.load(d / "arena_features.npz")
    xy = np.column_stack([z["x_mm"], z["y_mm"]])
    kind = z["kind"]
    return xy[kind == 0], xy[kind == 1], float(z["pole_radius_mm"])


def true_class(walls, poles, prad, x, y, yaw, horizon=None):
    """Class and distance of the nearest reflector in the forward cone.

    With `horizon`, a reflector beyond it is reported as "empty". The deployed
    inverse no longer abstains, so the class comparison passes horizon=None;
    the argument survives for callers that want the actionable subset.
    """
    cls, _, dist = nearest_reflector_in_cone(
        walls, poles, prad, x, y, yaw, CONE_HALF_DEG)
    if not np.isfinite(cls) or not np.isfinite(dist):
        return "empty", np.nan
    if horizon is not None and dist > horizon:
        return "empty", dist
    return ("wall" if cls == 0 else "pole"), dist


def load_run(pole, start, source):
    """One run: its summary, its per-step record, and the derived measures."""
    d = run_dir(pole, start, source)
    with open(d / "run_summary.json") as fh:
        summary = json.load(fh)
    with open(d / "trajectory.tsv") as fh:
        traj = [r for r in csv.DictReader(fh, delimiter="\t") if r["x_mm"]]
    walls, poles, prad = load_geometry(d)

    conf = np.zeros((3, 3), dtype=int)     # truth x perceived, raw truth
    first_pole_step = None
    first_pole_true_range = np.nan
    pole_detected, pole_missed = [], []    # true range of in-cone poles
    false_pole = []                        # true pole range at a false call
    xy = []
    for r in traj:
        x, y, yaw = _num(r["x_mm"]), _num(r["y_mm"]), _num(r["yaw_deg"])
        xy.append((x, y))
        perceived = r["feat_cls"] or "empty"
        truth, dist = true_class(walls, poles, prad, x, y, yaw)
        conf[CLASSES.index(truth), CLASSES.index(perceived)] += 1
        if truth == "pole":
            (pole_detected if perceived == "pole" else pole_missed).append(dist)
        elif perceived == "pole":
            false_pole.append(_num(r["pole_near_mm"]))
        if perceived == "pole" and first_pole_step is None:
            first_pole_step = int(r["step"])
            first_pole_true_range = _num(r["pole_near_mm"])

    xy = np.asarray(xy)
    path_mm = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0
    wall_clear = [_num(r["min_wall_mm"]) for r in traj]
    approach = [_num(r["drive_mm"]) for r in traj if r["tag"] == "approach"]

    return dict(
        pole=pole, start=start, source=source, summary=summary,
        n_steps=int(summary["n_steps"]),
        outcome=summary["outcome"],
        reached=bool(summary["reached"]) and bool(summary["aligned"]),
        seed=summary["controller_seed"],
        corrections=int(summary["n_align_corrections"]),
        true_dist_on_arrival=summary.get("final_true_pole_dist_mm"),
        approach_drive_mm=sorted(set(approach)),
        path_mm=path_mm,
        min_wall_mm=float(np.nanmin(wall_clear)),
        conf=conf,
        n_logged=len(traj),
        agreement=100.0 * np.trace(conf) / conf.sum(),
        first_pole_step=first_pole_step,
        first_pole_true_range=first_pole_true_range,
        pole_detected=pole_detected,
        pole_missed=pole_missed,
        false_pole=false_pole,
    )


def load_all():
    runs = []
    for pole in POLES:
        for start in STARTS:
            for source in SOURCES:
                d = run_dir(pole, start, source)
                if not d.is_dir():
                    raise FileNotFoundError(
                        f"missing run {d.name}: looked in PolicyRuns/Poles/ "
                        f"and in PolicyRuns/.")
                runs.append(load_run(pole, start, source))
    return runs


def check_vision_identity(runs):
    """Vision must reproduce the referee's truth on every step.

    Vision reads its feature from the geometry the referee also uses, so any
    disagreement means the ground-truth recomputation here is wrong rather
    than that vision is inaccurate. This is the correctness check on the whole
    module.
    """
    bad = [(r["pole"], r["start"], r["agreement"])
           for r in runs if r["source"] == "vision" and r["agreement"] < 100.0]
    if bad:
        raise AssertionError(
            f"vision disagrees with the recomputed truth in {len(bad)} run(s): "
            f"{bad}. The ground-truth recomputation is wrong.")
    return True


def check_design_matched(runs):
    """The two arms must differ in modality and in nothing else.

    Both faults this catches have happened: the 75 mm approach step lived only
    in the working tree and was lost (so the 2026-08-21 sonar arm drove 150 mm
    against vision's 75), and one pair was flown from the wrong start index and
    so did not share its controller seed.
    """
    drives = {tuple(r["approach_drive_mm"]) for r in runs}
    if drives != {(75.0,)}:
        raise AssertionError(
            f"approach steps are not matched across the 20 runs: {drives}. "
            f"150.0 is the void 2026-08-21 generation.")
    unmatched = []
    for pole in POLES:
        for start in STARTS:
            pair = [r for r in runs if r["pole"] == pole and r["start"] == start]
            if len({r["seed"] for r in pair}) != 1:
                unmatched.append(f"{pole}_S{start}")
    if unmatched:
        raise AssertionError(f"controller seeds not shared in: {unmatched}")
    return True


def _fmt_range(values, digits=0):
    lo, hi = min(values), max(values)
    return f"{np.median(values):.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def report(runs):
    check_vision_identity(runs)
    check_design_matched(runs)
    print("vision reproduces the recomputed ground truth on every step: OK")
    print("75 mm approach step and shared controller seed in all 20 runs: OK\n")

    for source in SOURCES:
        sub = [r for r in runs if r["source"] == source]
        won = [r for r in sub if r["reached"]]
        steps = [r["n_steps"] for r in sub]
        path_m = [r["path_mm"] / 1000.0 for r in sub]
        arrival = [r["true_dist_on_arrival"] for r in won
                   if r["true_dist_on_arrival"] is not None]
        searching = [r["first_pole_step"] for r in won
                     if r["first_pole_step"] is not None]
        approaching = [r["n_steps"] - r["first_pole_step"] for r in won
                       if r["first_pole_step"] is not None]
        first_rng = [r["first_pole_true_range"] for r in won
                     if np.isfinite(r["first_pole_true_range"])]
        conf = sum(r["conf"] for r in sub)

        print(f"=== {source} (n={len(sub)} runs, {conf.sum()} steps) ===")
        outcomes = {}
        for r in sub:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        print(f"  outcomes                     {outcomes}")
        print(f"  reached and aligned          {len(won)} / {len(sub)}")
        print(f"  bearing corrections          {sum(r['corrections'] for r in sub)}")
        print(f"  steps, median (range)        {_fmt_range(steps)}")
        print(f"  path length m, median (range) {_fmt_range(path_m, 2)}")
        if len(won) < len(sub):
            print(f"  successes only: steps        "
                  f"{_fmt_range([r['n_steps'] for r in won])}")
            print(f"  successes only: path m       "
                  f"{_fmt_range([r['path_mm'] / 1000.0 for r in won], 2)}")
        print(f"  steps searching, median      {np.median(searching):.0f}")
        print(f"  steps approaching, median    {np.median(approaching):.0f}")
        print(f"  true pole range at first perception, median  "
              f"{np.median(first_rng):.0f} mm   (range "
              f"{min(first_rng):.0f} to {max(first_rng):.0f})")
        print(f"  true pole distance on arrival {_fmt_range(arrival)} mm")
        print(f"  smallest wall clearance      "
              f"{min(r['min_wall_mm'] for r in sub):.0f} mm "
              f"(collision at {COLLISION_MM:.0f})")
        perceived_empty = conf[:, CLASSES.index("empty")].sum()
        perceived_pole = conf[:, CLASSES.index("pole")].sum()
        print(f"  steps with an empty feature  {perceived_empty} / {conf.sum()}")
        print(f"  steps perceiving a pole      {perceived_pole} / {conf.sum()}")
        # Steps whose nearest in-cone reflector was in fact the pole, and the
        # subset of those the controller was allowed to act on.
        det = np.array([d for r in sub for d in r["pole_detected"]])
        mis = np.array([d for r in sub for d in r["pole_missed"]])
        allp = np.concatenate([det, mis]) if len(det) + len(mis) else np.array([])
        print(f"  steps with the pole nearest  {len(allp)} / {conf.sum()} "
              f"({int((allp <= VETO_MM).sum())} within {VETO_MM:.0f} mm)")

        if source == "sonar":
            print(f"\n  per-step class against the raw truth")
            print(f"    {'truth/perceived':>16} "
                  + " ".join(f"{c:>7}" for c in CLASSES)
                  + f" {'n':>6} {'recall':>8}")
            for i, c in enumerate(CLASSES):
                n = conf[i].sum()
                rec = 100.0 * conf[i, i] / n if n else float("nan")
                print(f"    {c:>16} " + " ".join(f"{v:>7}" for v in conf[i])
                      + f" {n:>6} {rec:>7.1f}%")
            for j, c in enumerate(CLASSES):
                n = conf[:, j].sum()
                prec = 100.0 * conf[j, j] / n if n else float("nan")
                print(f"    {'precision ' + c:>16} {prec:>7.1f}%")
            print(f"    {'overall':>16} "
                  f"{100.0 * np.trace(conf) / conf.sum():>7.1f}%")

            in_veto = (det <= VETO_MM).sum() + (mis <= VETO_MM).sum()
            print(f"\n  within the {VETO_MM:.0f} mm veto: pole recall "
                  f"{100.0 * (det <= VETO_MM).sum() / in_veto:.1f}% "
                  f"({int((det <= VETO_MM).sum())} of {int(in_veto)})")
            false_pole = [d for r in sub for d in r["false_pole"]]
            print(f"  false pole calls             {len(false_pole)} of "
                  f"{perceived_pole} pole calls "
                  f"(precision {100.0 * (1 - len(false_pole) / perceived_pole):.1f}%)")
            if false_pole:
                print(f"    true pole range at those calls: "
                      + ", ".join(f"{d:.0f}" for d in sorted(false_pole)) + " mm")

            print(f"\n  pole recall by true range (pole nearest in cone)")
            for lo, hi in zip(RANGE_BANDS[:-1], RANGE_BANDS[1:]):
                d = int(((det >= lo) & (det < hi)).sum())
                m = int(((mis >= lo) & (mis < hi)).sum())
                rec = 100.0 * d / (d + m) if (d + m) else float("nan")
                print(f"    {f'{lo}-{hi} mm':>12} detected {d:>3}, missed {m:>3}"
                      f"   {rec:>6.1f}%")
        print()

    print("=== paired by start and placement ===")
    print(f"{'trial':>8} {'sonar steps':>12} {'vision steps':>13}"
          f" {'sonar mm':>10} {'vision mm':>10} {'path ratio':>11}"
          f" {'seed shared':>12}")
    slower, ratios, ratios_won = 0, [], []
    for pole in POLES:
        for start in STARTS:
            s = next(r for r in runs if r["source"] == "sonar"
                     and r["pole"] == pole and r["start"] == start)
            v = next(r for r in runs if r["source"] == "vision"
                     and r["pole"] == pole and r["start"] == start)
            slower += s["n_steps"] > v["n_steps"]
            ratio = s["path_mm"] / v["path_mm"]
            ratios.append(ratio)
            if s["reached"]:
                ratios_won.append(ratio)
            print(f"{pole}_S{start:d}".rjust(8)
                  + f"{s['n_steps']:>12}{v['n_steps']:>13}"
                  + f"{s['path_mm']:>10.0f}{v['path_mm']:>10.0f}"
                  + f"{'x' + format(ratio, '.2f'):>11}"
                  + f"{str(s['seed'] == v['seed']):>12}")
    print(f"\n  sonar took more steps in {slower} of {len(runs) // 2} pairs")
    print(f"  paired per-trial path ratio, median  x{np.median(ratios):.2f} "
          f"(x{min(ratios):.2f} to x{max(ratios):.2f})")
    print(f"  the same over the 9 sonar successes  x{np.median(ratios_won):.2f} "
          f"(x{min(ratios_won):.2f} to x{max(ratios_won):.2f})")


if __name__ == "__main__":
    report(load_all())
