"""
SCRIPT_TestEchoProcessor

Small end-to-end smoke test:
1) Load trained EchoProcessor artifact
2) Load one or more sessions
3) Run distance + IID inference on a batch
4) Print quick agreement metrics
"""

import numpy as np

from Library import DataProcessor
from Library.EchoProcessor import EchoProcessor


# --------------------------------------------
# Config
# --------------------------------------------
artifact_dir = "EchoProcessor"
sessions = ["sessionB01"]
profile_opening_angle = 70
profile_steps = 21
max_samples = 256  # None -> use all


def rankdata(a):
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = r
        i = j + 1
    return ranks


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return np.nan
    xx = x[m] - np.mean(x[m])
    yy = y[m] - np.mean(y[m])
    den = np.sqrt(np.sum(xx * xx) * np.sum(yy * yy))
    if den <= 1e-12:
        return np.nan
    return float(np.sum(xx * yy) / den)


def spearman_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return np.nan
    return pearson_corr(rankdata(x[m]), rankdata(y[m]))


def rmse(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) == 0:
        return np.nan
    return float(np.sqrt(np.mean((x[m] - y[m]) ** 2)))


def main():
    ep = EchoProcessor.load(artifact_dir)

    dc = DataProcessor.DataCollection(sessions)
    sonar = dc.load_sonar(flatten=False)  # (N, samples, 2)
    profiles, _ = dc.load_profiles(opening_angle=profile_opening_angle, steps=profile_steps)
    distance_axis_mm = dc.get_field("sonar_package", "corrected_distance_axis") * 1000.0
    corrected_distance_mm = dc.get_field("sonar_package", "corrected_distance") * 1000.0
    corrected_iid_db = dc.get_field("sonar_package", "corrected_iid")

    y_true_mm = np.min(profiles, axis=1).astype(np.float32)
    half = profiles.shape[1] // 2
    left_min = np.min(profiles[:, :half], axis=1)
    right_min = np.min(profiles[:, half:], axis=1)
    asym_zero_mm = (left_min - right_min).astype(np.float32)

    finite = np.isfinite(sonar).all(axis=(1, 2))
    finite &= np.isfinite(y_true_mm)
    finite &= np.isfinite(distance_axis_mm).all(axis=1)
    finite &= np.isfinite(corrected_distance_mm)
    finite &= np.isfinite(corrected_iid_db)
    finite &= np.isfinite(asym_zero_mm)

    sonar = sonar[finite]
    y_true_mm = y_true_mm[finite]
    distance_axis_mm = distance_axis_mm[finite].astype(np.float32)
    corrected_distance_mm = corrected_distance_mm[finite].astype(np.float32)
    corrected_iid_db = corrected_iid_db[finite].astype(np.float32)
    asym_zero_mm = asym_zero_mm[finite]

    if max_samples is not None:
        n = min(int(max_samples), len(sonar))
        sonar = sonar[:n]
        y_true_mm = y_true_mm[:n]
        distance_axis_mm = distance_axis_mm[:n]
        corrected_distance_mm = corrected_distance_mm[:n]
        corrected_iid_db = corrected_iid_db[:n]
        asym_zero_mm = asym_zero_mm[:n]

    out = ep.predict(sonar, distance_axis_mm)
    pred_distance_mm = out["distance_mm"]
    pred_iid_db = out["iid_db"]

    baseline_iid_sign_inverted = -corrected_iid_db

    print(f"Batch size: {len(sonar)}")
    print("Distance (vs profile min distance):")
    print(
        f"  EchoProcessor: RMSE={rmse(pred_distance_mm, y_true_mm):.1f} mm, "
        f"Bias={np.mean(pred_distance_mm - y_true_mm):.1f} mm, "
        f"Pearson={pearson_corr(pred_distance_mm, y_true_mm):.3f}, "
        f"Spearman={spearman_corr(pred_distance_mm, y_true_mm):.3f}"
    )
    print(
        f"  Acquisition corrected_distance: RMSE={rmse(corrected_distance_mm, y_true_mm):.1f} mm, "
        f"Bias={np.mean(corrected_distance_mm - y_true_mm):.1f} mm, "
        f"Pearson={pearson_corr(corrected_distance_mm, y_true_mm):.3f}, "
        f"Spearman={spearman_corr(corrected_distance_mm, y_true_mm):.3f}"
    )

    print("IID (vs zero-split profile asymmetry left_min-right_min):")
    print(
        f"  EchoProcessor IID: Pearson={pearson_corr(pred_iid_db, asym_zero_mm):.3f}, "
        f"Spearman={spearman_corr(pred_iid_db, asym_zero_mm):.3f}"
    )
    print(
        f"  Baseline corrected IID (-IID): Pearson={pearson_corr(baseline_iid_sign_inverted, asym_zero_mm):.3f}, "
        f"Spearman={spearman_corr(baseline_iid_sign_inverted, asym_zero_mm):.3f}"
    )

    print("\nFirst 5 predictions:")
    k = min(5, len(pred_distance_mm))
    for i in range(k):
        print(
            f"  i={i}: dist_pred={pred_distance_mm[i]:.1f} mm, dist_true={y_true_mm[i]:.1f} mm, "
            f"iid_pred={pred_iid_db[i]:.2f} dB"
        )


if __name__ == "__main__":
    main()
