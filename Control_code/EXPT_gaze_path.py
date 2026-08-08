"""Does decoupling gaze from heading buy usable perception along a path?

Experiment 2's robot currently looks where it drives, so what it perceives is
whatever the path's heading happens to point at. Path02 was chosen over Path01
because it hugged obstacles and therefore saw more -- at the cost of the
clearance that later crashed run02.

The idea tested here (Dieter, 2026-08-08) is to specify a LOOKING DIRECTION per
path pose independently of the driving direction, aiming the sonar at nearby
objects instead of past them. The sonar is rigidly mounted, so in practice a
gaze offset costs a rotate-out / rotate-back per step (or a pan servo); this
script does not model that cost, it only asks what the offset would buy.

Everything here is TRUE GEOMETRY -- "is a reflector inside the +-35 deg cone
within the horizon". That is an upper bound on perception: it says the object is
there to be seen, not that the inverse classifies it correctly. A second,
clearly-labelled estimate weights each pose by the deployed model's measured
recall in that range band.

Gaze policies compared:
  along_path      drive heading, i.e. today's behaviour (baseline)
  nearest_object  aim at the nearest reflector within the horizon
  pole_first      aim at the nearest pole within the horizon, else along path
  oracle          sweep headings, keep the best -- the ceiling any gaze rule
                  could reach, useful for knowing how much is left on the table

All gaze is clamped to +-MAX_GAZE_DEG of the drive heading; looking backwards is
not a sensible instruction to give the controller.

Run:  .venv/bin/python3 EXPT_gaze_path.py
"""

import os

import numpy as np

import SCRIPT_AnalysePathRun as A
from Library.LocalFeature import nearest_reflector_in_cone

ARENAS        = ["Path01", "Path02"]
HORIZONS_MM   = [1000.0, 1400.0]
MAX_GAZE_DEG  = 90.0
SWEEP_STEP    = 5.0
CONE_HALF_DEG = A.CONE_HALF_DEG
POLE_LANDMARK_MM = 1400.0   # how far a pole may be and still be worth aiming at

# Deployed-model recall by true range band, measured on the 4-fold out-of-fold
# predictions (Performance notes 2026-08-08). Used only for the weighted
# estimate. NOTE the asymmetry: beyond ~1400 mm wall recall collapses while pole
# recall rises, because the model is biased toward calling distant things poles.
# The high far-range pole recall is that bias, not competence -- do not read the
# weighted numbers as evidence for a longer horizon.
RECALL = [(0, 500, 0.976, 0.826), (500, 1000, 0.944, 0.826),
          (1000, 1400, 0.723, 0.881), (1400, 1e9, 0.262, 0.950)]


def recall_for(cls, rng):
    for lo, hi, wr, pr in RECALL:
        if lo <= rng < hi:
            return wr if cls == 0 else pr
    return 0.0


def perceive(geom, x, y, yaw, horizon):
    """(cls, range) at a pose. cls 0 wall, 1 pole, 2 nothing within horizon."""
    cls, _az, near = nearest_reflector_in_cone(
        geom["walls"], geom["poles"], geom["pole_radius_mm"],
        x, y, yaw, CONE_HALF_DEG)
    if not np.isfinite(cls) or not np.isfinite(near) or near > horizon:
        return 2, near
    return int(cls), float(near)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def clamp_gaze(target_yaw, path_yaw):
    return path_yaw + np.clip(wrap(target_yaw - path_yaw), -MAX_GAZE_DEG, MAX_GAZE_DEG)


def gaze_headings(policy, geom, x, y, path_yaw, horizon):
    """Yaw the robot would look along under `policy`."""
    if policy == "along_path":
        return path_yaw

    walls, poles = geom["walls"], geom["poles"]
    if policy in ("nearest_object", "pole_first"):
        cands = []
        if len(poles):
            rp = np.hypot(poles[:, 0] - x, poles[:, 1] - y) - geom["pole_radius_mm"]
            j = int(np.argmin(rp))
            if rp[j] <= (POLE_LANDMARK_MM if policy == "pole_first" else horizon):
                cands.append((rp[j], np.degrees(np.arctan2(poles[j, 1] - y,
                                                           poles[j, 0] - x))))
        if policy == "nearest_object" or not cands:
            rw = np.hypot(walls[:, 0] - x, walls[:, 1] - y)
            j = int(np.argmin(rw))
            if rw[j] <= horizon:
                cands.append((rw[j], np.degrees(np.arctan2(walls[j, 1] - y,
                                                           walls[j, 0] - x))))
        if not cands:
            return path_yaw
        return clamp_gaze(min(cands)[1], path_yaw)

    if policy == "oracle":
        # Prefer seeing a pole, then any reflector, then the smallest offset --
        # poles are the landmarks the path-integration story rests on.
        best = (2, 0.0, path_yaw)
        for off in np.arange(-MAX_GAZE_DEG, MAX_GAZE_DEG + SWEEP_STEP, SWEEP_STEP):
            yaw = path_yaw + off
            cls, _ = perceive(geom, x, y, yaw, horizon)
            score = {1: 2, 0: 1, 2: 0}[cls]
            if score > {1: 2, 0: 1, 2: 0}[best[0]] or (
                    cls == best[0] and abs(off) < abs(best[1])):
                best = (cls, off, yaw)
        return best[2]

    raise ValueError(policy)


def run_path(arena):
    pts, walls, poles, pole_r, _src = A.load_arena(arena)
    wp = A.load_path(arena)
    seg, hdg = A.densify(wp)
    geom = {"walls": walls, "poles": poles, "pole_radius_mm": pole_r}
    clr = np.array([np.min(np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])) for p in seg])

    print(f"\n{'=' * 92}\n  {arena}   {len(seg)} poses, min clearance {clr.min():.0f} mm, "
          f"{len(poles)} poles\n{'=' * 92}")

    for horizon in HORIZONS_MM:
        print(f"\n  horizon {horizon:.0f} mm     "
              f"{'informative':>12} {'wall':>7} {'pole':>7} "
              f"{'poles seen':>11} {'mean|gaze|':>11} {'weighted':>9}")
        for policy in ("along_path", "nearest_object", "pole_first", "oracle"):
            cls_out, weights, offs, seen = [], [], [], set()
            for (x, y), ph in zip(seg, hdg):
                yaw = gaze_headings(policy, geom, x, y, ph, horizon)
                cls, rng = perceive(geom, x, y, yaw, horizon)
                cls_out.append(cls)
                offs.append(abs(wrap(yaw - ph)))
                # Expected chance the inverse actually reports this correctly;
                # a pose with nothing in range contributes 0.
                weights.append(0.0 if cls == 2 else recall_for(cls, rng))
                if cls == 1 and len(poles):
                    rp = np.hypot(poles[:, 0] - x, poles[:, 1] - y)
                    seen.add(int(np.argmin(rp)))
            c = np.array(cls_out)
            w = float(np.mean(weights))
            print(f"  {policy:>18} {100 * (c != 2).mean():>11.1f}% "
                  f"{100 * (c == 0).mean():>6.1f}% {100 * (c == 1).mean():>6.1f}% "
                  f"{len(seen):>7}/{len(poles):<3} {np.mean(offs):>10.1f}d "
                  f"{100 * w:>8.1f}%")


# ── Distinctiveness ───────────────────────────────────────────────────────────
#
# Detection asks "is anything in view". Self-localisation asks something harder:
# does what the robot perceives tell it WHERE it is? A long featureless wall
# returns much the same profile from many positions -- perceived, but useless as
# a place code. Two poses are called CONFUSABLE when their perceived features sit
# within one standard deviation of the deployed model's own measurement error, so
# the robot could not tell them apart even in principle.
#
# Sigmas are the deployed model's held-out errors (Performance notes 2026-08-08).
SIGMA_SLICE_MM = 300.0     # held-out wall RMSE 292 / 392 / 321
SIGMA_POLE_DEG = 15.0      # held-out pole-az RMSE 15.01
CLASS_PENALTY  = 2.0       # cost of a class mismatch, in sigma units
CONFUSE_THRESH = 1.0       # normalised feature distance below which poses alias
MIN_SEP_MM     = 500.0     # ignore confusions with nearby poses: being unsure
                           # between two points 200 mm apart is not a failure
SEQ_LENS       = (1, 3, 5) # an RNN integrates, so also test short sequences


def pose_features(geom, seg, hdg, policy, horizon):
    """Normalised feature vector per pose, in units of the model's own error."""
    rows = []
    for (x, y), ph in zip(seg, hdg):
        yaw = gaze_headings(policy, geom, x, y, ph, horizon)
        cls, _rng, az, slices = A.true_local_feature(
            x, y, yaw, geom, CONE_HALF_DEG, max_range_mm=horizon)
        s = [slices.get(k, float("nan")) for k in ("left", "center", "right")]
        # No wall in a bin, or one beyond the horizon, both read as "far".
        s = [horizon if not np.isfinite(v) else min(float(v), horizon) for v in s]
        rows.append([s[0] / SIGMA_SLICE_MM, s[1] / SIGMA_SLICE_MM,
                     s[2] / SIGMA_SLICE_MM,
                     CLASS_PENALTY * (1.0 if cls == 1 else 0.0),
                     (az / SIGMA_POLE_DEG) if (cls == 1 and np.isfinite(az)) else 0.0,
                     CLASS_PENALTY * (1.0 if cls == 2 else 0.0)])
    return np.asarray(rows, dtype=float)


def arc_length(seg):
    d = np.hypot(*np.diff(np.vstack([seg, seg[:1]]), axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)[:-1]]), float(d.sum())


def aliasing(feats, s_along, total_len, k):
    """Fraction of poses that alias with a far-away pose, using a k-step window."""
    n = len(feats)
    # Concatenate k consecutive poses (the path is a loop, so wrap).
    idx = (np.arange(n)[:, None] + np.arange(k)[None, :]) % n
    seq = feats[idx].reshape(n, -1) / np.sqrt(k)   # /sqrt(k): keep the threshold
                                                   # comparable across window sizes
    d = np.linalg.norm(seq[:, None, :] - seq[None, :, :], axis=2)
    sep = np.abs(s_along[:, None] - s_along[None, :])
    sep = np.minimum(sep, total_len - sep)         # circular: it is a loop
    far = sep > MIN_SEP_MM
    conf = (d < CONFUSE_THRESH) & far
    aliased = conf.any(axis=1)
    worst = np.where(conf, sep, 0.0).max(axis=1)   # how far off you could be
    return float(aliased.mean()), (float(np.median(worst[aliased]))
                                   if aliased.any() else 0.0)


def local_resolution(feats, s_along, total_len, k, window_mm=1500.0):
    """How precisely does perception pin position down, GIVEN you already know
    roughly where you are?

    Global aliasing is the wrong bar for this policy: the RNN integrates motion,
    so it never has to localise from scratch. The question it actually faces is
    whether perception can refine an estimate that is already within a metre or
    so. This measures the spread of poses that remain confusable inside such a
    window -- i.e. the residual positional uncertainty perception leaves behind.
    """
    n = len(feats)
    idx = (np.arange(n)[:, None] + np.arange(k)[None, :]) % n
    seq = feats[idx].reshape(n, -1) / np.sqrt(k)
    d = np.linalg.norm(seq[:, None, :] - seq[None, :, :], axis=2)
    sep = np.abs(s_along[:, None] - s_along[None, :])
    sep = np.minimum(sep, total_len - sep)
    near = sep <= window_mm
    conf = (d < CONFUSE_THRESH) & near
    spread = np.where(conf, sep, 0.0).max(axis=1)
    return float(np.median(spread)), float(np.percentile(spread, 90))


def run_distinctiveness(arena):
    _pts, walls, poles, pole_r, _src = A.load_arena(arena)
    seg, hdg = A.densify(A.load_path(arena))
    geom = {"walls": walls, "poles": poles, "pole_radius_mm": pole_r}
    s_along, total = arc_length(seg)
    print(f"\n{'=' * 92}\n  DISTINCTIVENESS -- {arena}  "
          f"(loop {total:.0f} mm, {len(seg)} poses)\n{'=' * 92}")
    print("  'aliased' = share of poses indistinguishable from another pose >500 mm")
    print("  away along the loop, given the model's own error. 'worst' = median")
    print("  along-path distance to that confusable pose, i.e. how badly lost.\n")
    for horizon in HORIZONS_MM:
        print(f"  horizon {horizon:.0f} mm" + "".join(
            f"{'k=' + str(k) + ' aliased':>16}{'worst':>9}" for k in SEQ_LENS))
        for policy in ("along_path", "nearest_object", "pole_first", "oracle"):
            f = pose_features(geom, seg, hdg, policy, horizon)
            cells = ""
            for k in SEQ_LENS:
                a, w = aliasing(f, s_along, total, k)
                cells += f"{100 * a:>15.1f}%{w:>8.0f}mm"
            lr_med, lr_90 = local_resolution(f, s_along, total, k=5)
            print(f"  {policy:>16}{cells}   | local k=5: "
                  f"{lr_med:>4.0f} / {lr_90:>4.0f} mm")
        print("  local k=5 = median / 90th pct residual position uncertainty once")
        print("  you already know where you are to within 1.5 m (the RNN does).\n")


def main():
    print(f"Gaze clamped to +-{MAX_GAZE_DEG:.0f} deg of the drive heading; "
          f"cone +-{CONE_HALF_DEG:.0f} deg.")
    print("'informative' is TRUE GEOMETRY (a reflector is in the cone within the "
          "horizon).\n'weighted' multiplies each pose by the deployed model's "
          "measured recall in that\nrange band -- see the RECALL caveat in the "
          "source before quoting it.")
    for arena in ARENAS:
        try:
            run_path(arena)
        except Exception as exc:
            print(f"\n  {arena}: skipped ({type(exc).__name__}: {exc})")
    for arena in ARENAS:
        try:
            run_distinctiveness(arena)
        except Exception as exc:
            print(f"\n  {arena}: distinctiveness skipped "
                  f"({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
