# 16 — honest_final

## Goal

Establish the first leak-free baseline after discovering that experiments 08–15 derived features from the `target` column and produced invalid validation scores (~2.3 RMSE). The new rule: every feature must be computable the same way on the test set, with no knowledge of any target.

## Change

Built a rich feature set exclusively from columns present in `X_test` (`off`, `on`, `ledd`, age, diagnosis age, sex):
- **Lag/delta:** values at visits −1 and −2, and the change since the last visit.
- **Cumulative statistics:** running mean, min, max, std of each clinical measure (shifted by 1 so the current visit is excluded).
- **Rolling windows:** rolling mean over the last 3 and 5 visits.
- **Time features:** visit number, time since diagnosis, time since the first visit, inter-visit age gap.
- **Missingness flags:** binary indicators for missing `off`, `on`, `ledd`; count of prior `off` observations.
- **Clinical interactions:** `on − off`, `ledd × off`, `ledd × on`.

Model: `HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.01, max_leaf_nodes=63, min_samples_leaf=5, random_state=0)`. Evaluation: GroupKFold(5) on `patient_id`, single seed.

## Result

**Validation RMSE: 4.66** (GroupKFold-5). Skore Hub report: `16_honest_final` (cross-validations/53551).

## Decision

Confirmed that a clean feature set gives RMSE 4.66 — a meaningful but honest number. The gap from the invalid 2.3 (experiments 08–15) to 4.66 is entirely explained by the removed target-based features. Continue adding features that are available in `X_test`.
