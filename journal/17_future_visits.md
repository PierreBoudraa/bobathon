# 17 — future_visits

## Goal

Exploit the fact that the test file contains *all* visits of each test patient, not just one. This means future visits (later in time) are available at prediction time and can be used as features without any leakage.

## Change

Added 9 new features on top of the `16_honest_final` base:
- **Lead-1 and lead-2** for `off`, `on`, `ledd`: the value at the next visit and the one after, within the same patient (6 features).
- **Patient-level mean** for `off`, `on`, `ledd`: the mean over all visits of the same patient in the dataset (3 features).

All source columns (`off`, `on`, `ledd`) exist in `X_test`. Lead values are computed within-patient so no patient boundary is crossed. Patient means are computed independently on train and test (`groupby('patient_id').transform('mean')`). No target statistics.

Note: for patients whose every visit has a missing value for a given column, the patient mean is `NaN`. `HistGradientBoostingRegressor` handles this natively.

## Result

**Validation RMSE: 4.24** (−0.42 vs 16). Features: 57.

## Decision

Large gain from a simple idea. Future visits are a natural source of information given the dataset structure. Keep and build on top of this feature set.
