"""Experiment 1 aggregate statistics: every number quoted in the Results.

Reads the 20 runs of Experiment 1 (5 start poses x 2 pole placements x 2
modalities) from PolicyRuns/ and recomputes the aggregate measures reported in
the "Experiment 1: obstacle avoidance and target approach" Results subsection
and in Table~\\ref{tab:direct-results}. Run it to re-verify the prose against
the data after any re-run.

Ground truth is recomputed here from each run's own arena_features.npz rather
than read from the trajectory's logged referee columns, because the logged
columns carry the nearest wall and pole over ALL directions (they exist for
collision and success scoring), not the nearest reflector inside the forward
cone that the class comparison needs.

Two conventions worth knowing before reading the numbers:

  - Per-step class agreement is scored against ground truth with the model's
    1 m horizon applied, so that a correct abstention counts as correct. This
    is meaningful for sonar only. Vision reads its feature from the same
    digitized geometry the referee uses, so it reproduces the raw truth by
    construction; `check_vision_identity` asserts exactly that, and doubles as
    a test of the ground-truth recomputation.
  - Aggregates over steps are pooled across runs, not averaged over per-run
    fractions. The two differ because runs vary in length by an order of
    magnitude.

Known data caveat: direct_P2_S5_sonar_repeat01 was launched with START left at
4, so its run_summary.json records session/start of S4 and carries S4's
controller seed. The run itself is a valid S5 run (its step-0 pose is the S5
mark). Everything here keys off the folder name, never the summary's `start`.

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
# are also recorded per run in run_summary.json (approach_stop_mm) and in
# Table~\ref{tab:controller-params}.
CONE_HALF_DEG = 35.0
HORIZON_MM = 1000.0      # range beyond which the inverse abstains
STOP_MM = 400.0          # perceived pole range that ends the approach
COLLISION_MM = 20.0      # true wall clearance scored as a collision

POLES = ("P1", "P2")
STARTS = (1, 2, 3, 4, 5)
SOURCES = ("sonar", "vision")
CLASSES = ("wall", "pole", "empty")
RANGE_BANDS = (0, 400, 600, 800, 1000)


def run_dir(pole, start, source):
    return POLICY_RUNS / f"direct_{pole}_S{start}_{source}_repeat01"


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

    With `horizon`, a reflector beyond it is reported as "empty", matching the
    abstain class the inverse was trained to emit.
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

    conf = np.zeros((3, 3), dtype=int)     # truth (horizon applied) x perceived
    raw_agree = 0
    first_pole_step = None
    first_pole_true_range = np.nan
    pole_detected, pole_missed = [], []    # true range of in-cone poles <= 1 m
    xy = []
    for r in traj:
        x, y, yaw = _num(r["x_mm"]), _num(r["y_mm"]), _num(r["yaw_deg"])
        xy.append((x, y))
        perceived = r["feat_cls"] or "empty"
        t_hor, dist = true_class(walls, poles, prad, x, y, yaw, HORIZON_MM)
        t_raw, _ = true_class(walls, poles, prad, x, y, yaw, None)
        conf[CLASSES.index(t_hor), CLASSES.index(perceived)] += 1
        raw_agree += (perceived == t_raw)
        if t_hor == "pole":
            (pole_detected if perceived == "pole" else pole_missed).append(dist)
        if perceived == "pole" and first_pole_step is None:
            first_pole_step = int(r["step"])
            first_pole_true_range = _num(r["pole_near_mm"])

    xy = np.asarray(xy)
    path_mm = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0
    wall_clear = [_num(r["min_wall_mm"]) for r in traj]

    return dict(
        pole=pole, start=start, source=source, summary=summary,
        n_steps=int(summary["n_steps"]),
        outcome=summary["outcome"],
        seed=summary["controller_seed"],
        corrections=int(summary["n_align_corrections"]),
        true_dist_on_arrival=summary.get("final_true_pole_dist_mm"),
        path_mm=path_mm,
        min_wall_mm=float(np.nanmin(wall_clear)),
        conf=conf,
        n_logged=len(traj),
        raw_agreement=100.0 * raw_agree / len(traj),
        first_pole_step=first_pole_step,
        first_pole_true_range=first_pole_true_range,
        pole_detected=pole_detected,
        pole_missed=pole_missed,
    )


def load_all():
    runs = []
    for pole in POLES:
        for start in STARTS:
            for source in SOURCES:
                d = run_dir(pole, start, source)
                if not d.is_dir():
                    raise FileNotFoundError(
                        f"missing run {d.name}. The P2/S4 sonar run lives only "
                        f"inside PolicyRuns/backupP2.zip; unpack it there.")
                runs.append(load_run(pole, start, source))
    return runs


def check_vision_identity(runs):
    """Vision must reproduce the referee's raw truth on every step.

    Vision reads its feature from the geometry the referee also uses, so any
    disagreement means the ground-truth recomputation here is wrong rather
    than that vision is inaccurate. This is the correctness check on the whole
    module.
    """
    bad = [(r["pole"], r["start"], r["raw_agreement"])
           for r in runs if r["source"] == "vision" and r["raw_agreement"] < 100.0]
    if bad:
        raise AssertionError(
            f"vision disagrees with the recomputed truth in {len(bad)} run(s): "
            f"{bad}. The ground-truth recomputation is wrong.")
    return True


def _fmt_range(values, digits=0):
    lo, hi = min(values), max(values)
    return f"{np.median(values):.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def report(runs):
    check_vision_identity(runs)
    print("vision reproduces the recomputed ground truth on every step: OK\n")

    for source in SOURCES:
        sub = [r for r in runs if r["source"] == source]
        steps = [r["n_steps"] for r in sub]
        path_m = [r["path_mm"] / 1000.0 for r in sub]
        arrival = [r["true_dist_on_arrival"] for r in sub
                   if r["true_dist_on_arrival"] is not None]
        searching = [r["first_pole_step"] for r in sub
                     if r["first_pole_step"] is not None]
        approaching = [r["n_steps"] - r["first_pole_step"] for r in sub
                       if r["first_pole_step"] is not None]
        first_rng = [r["first_pole_true_range"] for r in sub
                     if np.isfinite(r["first_pole_true_range"])]
        conf = sum(r["conf"] for r in sub)

        print(f"=== {source} (n={len(sub)} runs, {conf.sum()} steps) ===")
        outcomes = {}
        for r in sub:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        print(f"  outcomes                     {outcomes}")
        print(f"  bearing corrections          {sum(r['corrections'] for r in sub)}")
        print(f"  steps, median (range)        {_fmt_range(steps)}")
        print(f"  path length m, median (range) {_fmt_range(path_m, 2)}")
        print(f"  steps searching, median      {np.median(searching):.0f}")
        print(f"  steps approaching, median    {np.median(approaching):.0f}")
        print(f"  true pole range at first perception, median  "
              f"{np.median(first_rng):.0f} mm")
        print(f"  true pole distance on arrival {_fmt_range(arrival)} mm")
        print(f"  smallest wall clearance      "
              f"{min(r['min_wall_mm'] for r in sub):.0f} mm "
              f"(collision at {COLLISION_MM:.0f})")
        perceived_empty = conf[:, CLASSES.index("empty")].sum()
        perceived_pole = conf[:, CLASSES.index("pole")].sum()
        print(f"  steps with an empty feature  {perceived_empty} / {conf.sum()}")
        print(f"  steps perceiving a pole      {perceived_pole} / {conf.sum()}")
        # Steps whose nearest reflector lay beyond the horizon. Under sonar the
        # model abstains on these; under vision they are reported regardless,
        # which is the range asymmetry between the two conditions.
        print(f"  steps with nothing within {HORIZON_MM:.0f} mm  "
              f"{conf[CLASSES.index('empty')].sum()} / {conf.sum()}")

        if source == "sonar":
            print(f"\n  per-step class, truth under the {HORIZON_MM:.0f} mm horizon")
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

            det = np.array([d for r in sub for d in r["pole_detected"]])
            mis = np.array([d for r in sub for d in r["pole_missed"]])
            print(f"\n  pole recall by true range (in cone, within the horizon)")
            for lo, hi in zip(RANGE_BANDS[:-1], RANGE_BANDS[1:]):
                d = int(((det >= lo) & (det < hi)).sum())
                m = int(((mis >= lo) & (mis < hi)).sum())
                rec = 100.0 * d / (d + m) if (d + m) else float("nan")
                print(f"    {f'{lo}-{hi} mm':>12} detected {d:>3}, missed {m:>3}"
                      f"   {rec:>6.1f}%")
        print()

    print("=== paired by start and placement ===")
    print(f"{'trial':>8} {'sonar steps':>12} {'vision steps':>13} {'seed shared':>12}")
    slower = 0
    for pole in POLES:
        for start in STARTS:
            s = next(r for r in runs if r["source"] == "sonar"
                     and r["pole"] == pole and r["start"] == start)
            v = next(r for r in runs if r["source"] == "vision"
                     and r["pole"] == pole and r["start"] == start)
            slower += s["n_steps"] > v["n_steps"]
            print(f"{pole}_S{start:d}".rjust(8)
                  + f"{s['n_steps']:>12}{v['n_steps']:>13}"
                  + f"{str(s['seed'] == v['seed']):>12}")
    print(f"\n  sonar took more steps in {slower} of {len(runs) // 2} pairs")


if __name__ == "__main__":
    report(load_all())
