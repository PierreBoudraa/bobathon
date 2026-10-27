# 18 — rater_id

## Goal

Test whether the identity of the clinician who performed the rating (`rater_id`) carries predictive signal above and beyond the existing clinical features.

## Change

Added `rater_id` as a single integer-coded feature. The 50 distinct rater IDs are label-encoded using `OrdinalEncoder` with a fixed category list built from the union of train and test raters (so codes are consistent across both DataFrames). The raw string column is excluded from the feature set; only `rater_id_enc` enters the model. No target statistics.

## Result

**Validation RMSE: 4.08** (−0.16 vs 17). Features: 58.

## Decision

Clear gain (+0.16 RMSE reduction). Rater identity carries real signal — each clinician scores differently. In the next experiment, expose this to HGBR as a true categorical feature so the model can find the optimal partition of raters rather than treating codes as ordinal numbers.
