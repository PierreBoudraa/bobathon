# 19 — rater_categorical

## Goal

Improve on experiment 18 by (a) telling HGBR that `rater_id_enc` is a true categorical column so it uses the native optimal-partition split instead of an ordinal threshold, and (b) adding explicit per-rater aggregate features from the clinical columns.

## Change

Two additions over experiment 18:

1. **True categorical encoding:** `rater_id_enc` is now stored as `pd.CategoricalDtype` (integer codes 0–49, non-negative). `HistGradientBoostingRegressor` is instantiated with `categorical_features='from_dtype'`, which detects any column with `CategoricalDtype` and applies the dedicated categorical split algorithm.

2. **Per-rater aggregates** (3 new features): `rater_off_mean`, `rater_on_mean`, `rater_ledd_mean` — the mean of each clinical measure per rater, computed independently on train and test with `groupby('rater_id').transform('mean')`. No target statistics.

## Result

**Validation RMSE: 4.04** (−0.04 vs 18). Features: 61.

## Decision

Modest but consistent gain. The marginal benefit of the true categorical split and the per-rater aggregates is small once ordinal encoding is already in place, suggesting rater identity was already captured reasonably well. Keep this as the definitive base feature set.
