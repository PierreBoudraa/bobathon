# 22 — two_stage_fixed

## Goal

Fix the final-fit inconsistency identified in experiment 21: train the stage-2 model on out-of-sample stage-1 predictions (`s1_oof`) so the distribution of `s1_pred` it sees during training matches what it will see at inference time.

## Change

Single change in the final fit block (CV evaluation and all features are unchanged):

**Removed:** the loop that re-ran all 5 fold models on the full training data and averaged in-sample predictions into `s1_full`.

**Added:** `visits2_full["s1_pred"] = s1_oof` — the same OOF predictions already computed during the CV loop. These are out-of-sample for every row (each visit was predicted by a model that never saw that patient), which is the same kind of prediction that stage 2 receives at test time via `s1_test` (average of 5 fold models on the held-out test set).

Stage-1 test predictions remain the average of the 5 fold models. Small caveat: the stage-1 models of the other folds have seen the validation patients, so the validation score can be slightly optimistic (usual stacking caveat).

Comparison with experiment 21: Pearson correlation of test predictions = 0.9982; RMSE difference = 0.96 — the predictions are very similar overall but the fixed version is meaningfully better on Kaggle.

## Result

**Stage 2 OOF RMSE: 3.87** (same as 21, only the final fit changed). **Kaggle public: 3.623** (vs 3.677 for 21, −0.054). Skore Hub: `22_two_stage_fixed` (cross-validations/80990). Submissions file: `submissions/22_two_stage_fixed.csv`, 11 013 rows, 0 NaN.

## Decision

The fix improved the Kaggle score from 3.677 to 3.623 without changing the validation RMSE. This confirms the diagnosis: experiment 21's final fit was training on a distribution of stage-1 predictions that was too accurate (in-sample), degrading test performance. Experiment 22 is the current best model.
