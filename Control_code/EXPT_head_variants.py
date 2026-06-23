"""Experiment: compare wall-head designs for the inverse model.

Reuses the canonical trainer (same data, loss, CV folds, seed) and only swaps
the model class and the artifact prefix, so the three runs are directly
comparable. Writes inverse_<variant>_* into SonarModel/, leaving the deployed
inverse_* files untouched.

  base : SonarSlicesUQ_TwoHeaded (current: side heads + z_sym center)
  B    : one 3-output wall head, symmetry enforced over LR/RL (no z_sym)
  A    : one 3-output wall head from LR only (no symmetry; can learn asymmetry)

    .venv/bin/python3 EXPT_head_variants.py
"""

import json
import os

import SCRIPT_TrainInverseModel as T
from Library.SonarModel import SonarSlicesUQ_TwoHeaded, SonarSlicesUQ_Wall3

# (class, kwargs, model_class_name) per variant. The trainer constructs via its
# MODEL_CLASS / MODEL_KWARGS globals and records MODEL_CLASS_NAME in
# feature_params, so re-binding those three here selects the variant.
VARIANTS = {
    "base": (SonarSlicesUQ_TwoHeaded, {}, "SonarSlicesUQ_TwoHeaded"),
    "B": (SonarSlicesUQ_Wall3, {"symmetric": True}, "SonarSlicesUQ_Wall3"),
    "A": (SonarSlicesUQ_Wall3, {"symmetric": False}, "SonarSlicesUQ_Wall3"),
}


def run(name, spec):
    cls, kwargs, class_name = spec
    T.MODEL_CLASS = cls
    T.MODEL_KWARGS = kwargs
    T.MODEL_CLASS_NAME = class_name
    T.ARTIFACT_PREFIX = f"inverse_{name}"
    print(f"\n{'#' * 20} VARIANT {name} {'#' * 20}")
    T.main()


def summarize():
    print(f"\n{'=' * 20} COMPARISON {'=' * 20}")
    print(f"{'variant':6} {'class_acc':>11} {'pole_rmse':>11} {'wall L/C/R (mm)':>20}")
    for name in VARIANTS:
        p = os.path.join(T.OUTPUT_DIR, f"inverse_{name}_cv_results.json")
        d = json.load(open(p))["cv_summary"]
        acc = d["class_acc_mean"] * 100
        pol = d["pole_az"]["rmse_deg_mean"]
        w = d["wall_per_slice"]
        lcr = f"{w['left']['rmse_mm_mean']:.0f}/{w['center']['rmse_mm_mean']:.0f}/{w['right']['rmse_mm_mean']:.0f}"
        print(f"{name:6} {acc:9.1f} % {pol:9.2f} ° {lcr:>20}")


if __name__ == "__main__":
    for name, spec in VARIANTS.items():
        run(name, spec)
    summarize()
