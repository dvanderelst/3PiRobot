# Cross-Modal Sonar Calibration Notes

## 1) Summary: Direct Sonar -> Curvature Experiments

### Goal
Test whether a model can learn steering-relevant curvature directly from raw sonar windows, using geometry-derived curvature as supervision.

### Pipeline We Tested
- Build per-sample ground-truth curvature from arena geometry and robot pose.
- Use sonar windows + relative pose history as model input.
- Train regression model (`SCRIPT_TestGroundTruthCurvature.py`) to predict signed curvature.
- Evaluate with both regression metrics and control-oriented metrics.

### What We Implemented
- Added cached curvature target loading via `DataProcessor.load_curvatures(...)`.
- Added reusable profile-to-curvature module (`Library/ProfileCurvature.py`).
- Trained/evaluated a direct sonar-to-curvature network with:
  - Pearson/Spearman correlations,
  - MAE/RMSE,
  - sign accuracy,
  - turn recall/precision/F1,
  - threshold/quantile tradeoff analysis.

### Main Findings
- Direct mapping is possible and beats naive baselines (e.g., always-zero/mean MAE).
- But performance was moderate and sensitive to thresholding when evaluated as control events.
- Compared with the existing staged pipeline (sonar -> profiles -> walliness -> curvature), direct mapping did not show clear practical superiority in our tests.
- Interpretation: the intermediate representation appears to stabilize learning and control relevance.

### Decision
Park direct sonar->curvature work for now and continue with the walliness-based steering pipeline as the primary path.

---

## 2) Reflection: Recent Interpretation and Scientific Framing

### Reading of the Current Evidence
From the recent `Estimated vs GT-Window Curvature` analyses:
- Estimated walliness-derived curvature aligns well with ground-truth-window curvature.
- Agreement strengthens when ambiguous geometric cases are excluded (small GT left/right circle-radius difference).
- This supports that the learned mapping is not just reproducing sonar heuristics; it tracks environment-relevant steering structure.

### Comparison to IID/Distance Heuristics
- IID sign and nearest-distance proxies carry useful information but are weaker and less direct.
- Walliness-derived curvature is more tightly coupled to the steering variable we actually need (signed curvature).

### Conceptual Conclusion
This supports a proof-of-concept for **cross-modal calibration**:
- During exploration, vision supplies high-quality environmental structure.
- That structure supervises sonar-to-steering mapping.
- Later, sonar-alone can produce steering commands that remain grounded in environment-relevant geometry.

### Biological Relevance (Your Framing)
Your stated hypothesis is that bats may learn to use echolocation more efficiently for steering when vision is available. The current computational evidence is consistent with that idea:
- visual information can calibrate sonar-based control mappings,
- learned sonar control signals can reflect world-structure demands,
- and ambiguity appears in predictable regimes rather than random failure.

### Scope / Caveat
Current results are environment-specific; generalization across arenas, trajectories, and sensing/noise conditions remains the next validation step.
