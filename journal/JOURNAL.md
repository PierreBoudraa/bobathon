# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's disease UPDRS motor score regression (Bobathon ESILV × Probabl × IBM)
- **Goal:** Predict the continuous UPDRS motor score (`target`) from patient clinical visit features; metric is RMSE (lower is better); test patients are entirely disjoint from train patients.
- **Last experiment:** `22_two_stage_fixed` — two-stage HGBR with fixed final fit (stage 2 trained on OOF stage-1 predictions)
- **Last result:** validation RMSE **3.87** (GroupKFold-5) · Kaggle public **3.623**

<!--
Workspace decisions: one-time project-setup choices. Record each when
it is made and treat it as fixed unless you deliberately change one,
updating the recorded date. Reading this block on later sessions avoids
re-deciding what's already settled.
-->

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas — recorded: 2026-10-07
  - env manager: pip+venv — recorded: 2026-10-07
  - CV splitter family: GroupKFold(n_splits=5) on `patient_id` — recorded: 2026-10-07
  - skore mode: hub — recorded: 2026-10-07
  - skore hub workspace: `shaking-data-dia2`, project `bobathon-esilv` — recorded: 2026-10-07
  - skore mlflow tracking uri: n/a
  - student prior: comfortable

## Data understanding (EDA)

<!--
Short index entry - the full analysis lives in data/eda.md. If the
data exploration was skipped, keep just the Status: skipped line.
-->

- **Status:** done — 2026-10-07
- **Summary:** 44 590 rows × 13 columns; continuous regression target (UPDRS motor score, range 0–109, near-symmetric).
  `patient_id` repeats (~8 visits/patient) → **GroupKFold** required.
  `off` (r = 0.871 with target) and `on` (r = 0.669) are dominant predictors; `off`/`ledd`/`time_since_intake_*` have 30–79 % missingness.
  No datetime column; MAE/RMSE are appropriate metrics.
- **Report:** [data/eda.md](../data/eda.md)

## History

<!--
One row per experiment, in chronological order. Newest at the bottom.
Experiments 08–15 are marked INVALID: all features derived from `target`
(target_lag1, target_cumean, bias_correction_mean, disease_slope, target
encodings). Validation RMSE ~2.3 was not reproducible on the test set.
-->

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy_baseline` | Dummy model predicting the global mean | done | RMSE 16.48 | — |
| `02_ridge` | Ridge regression on raw features | done | RMSE 10.60 | — |
| `03_grouped_cv` | Switch to GroupKFold — no patient leaks across folds | done | RMSE 10.60 | — |
| `04_hgbr` | HistGradientBoostingRegressor; handles missing values natively | done | RMSE 7.75 | — |
| `05_tabular` | Add `gene` / `cohort` with skrub pipeline | done | RMSE 7.75 (no gain) | — |
| `06_dataops` | Same model as a skrub DataOps pipeline | done | RMSE 7.76 (no gain) | — |
| `07_best` | Best model of Day 2 | done | RMSE 7.75 | — |
| `08` – `15` | Target-based features (lag/mean/encoding of `target`) | **INVALID — target leakage** | ~~RMSE ~2.3~~ | archived in `experiments/archive_leaky/` |
| `16_honest_final` | First leak-free feature set: lag/delta/cumul/roll for off/on/ledd, time, missingness | done | RMSE **4.66** | [16_honest_final.md](16_honest_final.md) |
| `17_future_visits` | Add lead-1/lead-2 and patient-mean for off/on/ledd | done | RMSE **4.24** | [17_future_visits.md](17_future_visits.md) |
| `18_rater_id` | Add `rater_id` as ordinal-encoded feature | done | RMSE **4.08** | [18_rater_id.md](18_rater_id.md) |
| `19_rater_categorical` | `rater_id` as true CategoricalDtype + per-rater off/on/ledd means | done | RMSE **4.04** | [19_rater_categorical.md](19_rater_categorical.md) |
| `20_final_ensemble` | Average of 5 seeds (seeds 0–4), single stage | done | RMSE **4.01** · Kaggle 3.847 | [20_final_ensemble.md](20_final_ensemble.md) |
| `21_two_stage` | Two-stage HGBR: stage-2 uses lag/lead of stage-1 OOF predictions | done | RMSE **3.87** · Kaggle 3.677 | [21_two_stage.md](21_two_stage.md) |
| `22_two_stage_fixed` | Same two-stage model; final fit uses `s1_oof` instead of in-sample stage-1 preds | done | RMSE **3.87** · Kaggle **3.623** | [22_two_stage_fixed.md](22_two_stage_fixed.md) |

## Backlog

<!--
Ideas not yet committed to a journal/NN_*.md design note.
Source: README.md §9 "What we learned and what we would try next".
-->

| # | Item | Source |
|---|---|---|
| B1 | Extend lag/lead depth beyond 2: try lag3/lead3 for off/on/ledd; add per-visit deviation from patient mean | README §9 |
| B2 | Tune HGBR hyperparameters (max_leaf_nodes, learning_rate, min_samples_leaf) with Optuna or RandomizedSearchCV | README §9 |
| B3 | Try LightGBM or XGBoost as a drop-in replacement or blend with the current HGBR | README §9 |
| B4 | Ensemble the two-stage model over multiple seeds (as in exp 20 but applied to exp 22) | README §9 |
