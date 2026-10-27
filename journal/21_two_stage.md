# 21 — two_stage

## Goal

Test whether a second model can improve predictions by exploiting the temporal smoothness of a patient's disease progression — specifically by looking at the first model's predictions for the neighbouring visits.

## Change

**Stage 1:** `HistGradientBoostingRegressor` on the 61 base features (identical to experiment 19). OOF predictions produced by GroupKFold(5) on `patient_id`; test predictions are the average of the 5 fold models.

**Stage 2 features** (7 new, derived from `s1_pred`): within-patient `s1_lag1`–`s1_lag3`, `s1_lead1`–`s1_lead3`, `s1_patient_mean`. No target statistics anywhere.

**Stage 2 evaluation:** same 5 GroupKFold splits. For fold k, the stage-2 model trains on the 4 other folds (using their out-of-fold predictions); validation rows carry `s1_pred` from a model trained on the 4 other folds. Small caveat: the stage-1 models of the other folds have seen the validation patients, so the validation score can be slightly optimistic (usual stacking caveat).

**Known issue in the final fit (fixed in 22):** the final stage-2 model is trained on `s1_full` — the average of all 5 fold models predicting on the data they were trained on (in-sample, too accurate). This creates a distribution mismatch between training and inference.

## Result

**Stage 2 OOF RMSE: 3.87** (−0.17 vs 19). **Kaggle public: 3.677.** Features: 68. Skore Hub: `21_two_stage` (cross-validations/78175).

## Decision

Large gain (+0.17 RMSE). The neighbourhood smoothing is clearly effective. However the final fit used in-sample stage-1 predictions, which is inconsistent with the OOF setup used during CV. Fix this in experiment 22.
