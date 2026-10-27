# 20 — final_ensemble

## Goal

Reduce variance in the single-model predictions by averaging several independently seeded models (ensemble of seeds), using the exact same features and hyperparameters as experiment 19.

## Change

No feature changes. For each of 5 seeds (0–4):
- Run GroupKFold(5) OOF predictions manually; accumulate into `oof_sum`.
- Fit one full-train model; accumulate test predictions into `test_sum`.

OOF and test predictions are divided by 5 to produce the ensemble averages. The OOF RMSE is computed from the averaged OOF predictions vs `y`.

The Skore Hub report is pushed for seed=0 only (skore evaluates a single model), so the Hub RMSE (4.04) matches experiment 19. The 4.01 figure comes from the manually averaged OOF in the script.

## Result

**Ensemble OOF RMSE: 4.01** (−0.03 vs 19). **Kaggle public: 3.847.** Submissions file: `submissions/20_final_ensemble.csv`, 11 013 rows, 0 NaN.

## Decision

Small but free gain from seed averaging. However, the two-stage approach (experiment 21) is a much larger step. Kept as a simple, interpretable alternative to the two-stage model.
