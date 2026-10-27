# EDA: Parkinson's UPDRS Motor Score

> Full interactive report: [data/eda_train.html](eda_train.html)

---

## Dataset at a glance

| Item | Value |
|---|---|
| Training rows | 44 590 |
| Columns (features + target) | 13 |
| Target | `target` (continuous UPDRS motor score, 0 – 109.5) |
| Test file | `X_test.csv` (no target; predictions submitted via `sample_submission.csv`) |

The dataset records longitudinal clinical assessments of Parkinson's disease patients. Each row is a single visit; patients appear multiple times (`patient_id` is not unique per row).

---

## Per-column findings

| Column | Dtype | Missing % | # Unique | Notes |
|---|---|---|---|---|
| `patient_id` | string | 0 % | 5 576 | High-cardinality group key — each patient has multiple rows (~8 on average) |
| `cohort` | string | 0 % | 2 | Binary study cohort (A / B) |
| `sexM` | int | 0 % | 2 | Binary sex indicator |
| `gene` | string | **32.4 %** | 4 | Gene variant (LRRK2+, GBA+, SNCA+, or IDIOPATHIC); substantial missingness |
| `age_at_diagnosis` | float | 5.2 % | 563 | Age at PD diagnosis |
| `age` | float | 0 % | 1 082 | Current age at visit |
| `ledd` | float | **36.6 %** | 1 320 | Levodopa equivalent daily dose; correlated with disease severity |
| `time_since_intake_on` | float | **46.4 %** | 64 | Hours since last dose (ON state measurement); high missingness |
| `time_since_intake_off` | float | **78.8 %** | 178 | Hours since last dose (OFF state measurement); very high missingness — OFF visits are rarer |
| `rater_id` | string | 0 % | 50 | Clinician who scored the visit; could introduce rater bias |
| `on` | float | **29.6 %** | 83 | UPDRS motor score under medication (ON state) |
| `off` | float | **42.4 %** | 101 | UPDRS motor score without medication (OFF state) |
| `target` | float | 0 % | 867 | **Prediction target** (UPDRS motor score; unspecified ON/OFF state) |

**Key missingness pattern:** the `off` subtree (`off`, `time_since_intake_off`) is missing for ~42–79 % of rows — these are visits that did not include an OFF-state assessment. The imputer / pipeline must handle this; simple mean imputation on `off` would be noisy given the missingness mechanism is not-at-random (only sicker / more complex patients may get an OFF visit).

---

## Target

- **Type:** continuous regression (no stratification needed for CV)
- **Range:** 0 – 109.5
- **Mean:** 37.5, **Median:** 37.3, **SD:** 16.5
- **IQR:** 25.6 – 49.3 (roughly symmetric bell shape)
- **Distribution:** approximately unimodal and near-symmetric; slight right tail (max 109.5 is far above Q3 = 49.3). A handful of extreme values but no sign of a bimodal split.
- **Implication:** MAE or RMSE are natural metrics. No transformation is obviously required; log-transform is not warranted for a near-symmetric distribution.

---

## Structure signals

### Datetime columns
None detected. There is no explicit date/timestamp column in the training data.

**Implication:** `TimeSeriesSplit` is not warranted by explicit time ordering. However, the `age` column increases with follow-up, so visits within a patient are implicitly ordered.

### Group / ID structure
`patient_id` has 5 576 unique values across 44 590 rows → average ~8 visits per patient. This is the dominant structural signal:

- A patient's visits share the same baseline characteristics (`gene`, `age_at_diagnosis`, `sexM`, `cohort`) and underlying disease trajectory.
- Splitting randomly would leak the same patient into both train and validation folds, inflating validation scores.
- **Implication: `GroupKFold` (or `GroupShuffleSplit`) with `groups=patient_id` is the correct CV strategy.**

`rater_id` (50 unique values) is also a potential grouping concern — the same rater grades many patients — but patient-level leakage is more critical.

---

## Associations

### With the target (Pearson correlation / Cramér V)

| Feature | Pearson r | Cramér V | Interpretation |
|---|---|---|---|
| `off` | **0.871** | 0.406 | Strongest predictor — OFF-state motor score nearly is the target |
| `on` | 0.669 | 0.368 | Second strongest — ON-state score |
| `ledd` | 0.298 | 0.236 | Levodopa dose moderately correlated |
| `age` | 0.310 | 0.114 | Older patients tend to have higher UPDRS |
| `age_at_diagnosis` | 0.133 | 0.051 | Weaker than current age |
| `time_since_intake_on` | ~0 | 0.177 | Near-zero Pearson but non-zero Cramér V — non-linear / stepped relationship |
| `cohort`, `gene`, `sexM`, `rater_id` | n/a / low | ≤ 0.09 | Weak categorical associations |

**⚠ Potential near-leakage flag — `off` (r = 0.871):** the `off` column is the patient's UPDRS score measured in the OFF medication state, and the target is likely the same assessment in an unspecified or combined state. The near-perfect correlation should be reviewed: if `target` is `off` for some visits and `on` for others, `off` may partially be the label itself for those rows. This needs domain clarification but does NOT warrant removing the column — it may be a legitimate strong predictor in all cases.

### Strongest overall
`age_at_diagnosis` ↔ `age` (r = 0.94) — expected multicollinearity (both are ages). One of the two could be dropped or regularized. `on` ↔ `off` (r = 0.87) mirrors the target relationships above.

---

## Modelling implications

| Finding | Implication |
|---|---|
| `patient_id` repeats (avg 8 visits / patient) | **Use `GroupKFold` with `groups=patient_id`** — random CV leaks patient info across folds |
| Target near-symmetric, range 0–109 | **MAE or RMSE** as primary metric; no need for target transform |
| No explicit datetime column | `TimeSeriesSplit` not warranted; standard KFold within groups is fine |
| Heavy missingness in `off`, `ledd`, `time_since_intake_*` | Pipeline needs **an imputer that handles high-missingness columns**; consider indicator columns for `off` and `time_since_intake_off` (missingness is informative) |
| `off` ↔ target near-leakage (r = 0.871) | Investigate whether `off` is a sub-component of `target` for some visits; at minimum, track model performance with and without it |
| `age_at_diagnosis` ↔ `age` multicollinearity (r = 0.94) | Regularization (Ridge, tree-based) will handle it; explicit drop is optional |
| `gene` has 32 % missing values | Impute as a dedicated "unknown" category or use a model that handles categorical NaN natively (HistGradientBoosting) |
| `rater_id` (50 raters) | Medium-cardinality; target-encoding or leave-one-out might extract rater bias signal |

---

## Open questions

1. **What exactly is `target`?** Is it the ON-state score, the OFF-state score, or an average? This determines whether `on` / `off` are strong predictors or near-leakage.
2. **Is there a temporal ordering per patient?** If `age` increases monotonically per patient, a leave-future-out strategy within groups might be more faithful to deployment.
3. **Why is `gene` missing 32 % of the time?** If missingness is associated with a specific cohort or disease subtype, an indicator column would be informative.
4. **Are `time_since_intake_on` and `time_since_intake_off` in hours?** The 64 / 178 unique-value counts suggest a grid of measurement points — confirm the unit and range are as expected.
