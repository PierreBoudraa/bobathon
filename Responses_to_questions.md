# Responses to the workbook questions

Team: shaking-data-dia2 (Pierre Boudraa, Sina Balaei, Nayl Saifoudine, Frank Tankhapanya)

Answers to the workbook questions in `docs/DAY1.md` and `docs/DAY2.md` of the
lab repository (https://github.com/probabl-ai/bobathon-esilv).

**Where the numbers come from.** EDA figures are computed from `data/*.csv`.
Model figures come from our Skore Hub reports and from an offline re-run of the
same setups (same features, `GroupKFold(n_splits=5)` on `patient_id`,
`random_state=0`). The re-run reproduces the Hub scores: Ridge 10.60, HGBR
7.752, DataOps 7.762. All RMSE values are grouped-CV unless stated otherwise.

---

## Day 1

### EDA

**1. Visits and patients.**
Train has 44,590 visits from 5,576 patients. Test has 11,013 visits from 1,395
patients, and no patient appears in both. A patient has 8 visits on average
(median 7) and at most 12, in both train and test.

**2. Missing values.**
The three columns with the most missing values are `time_since_intake_off`
(78.8 %), `time_since_intake_on` (46.4 %) and `off` (42.4 %). They are followed
by `ledd` (36.6 %), `gene` (32.4 %) and `on` (29.6 %).
42.4 % of visits have no `off` score and 29.6 % have no `on` score. No visit
has both missing: every visit has at least one motor score.

**3. Distribution of `target`.**
`target` goes from 0 to 109.5, with mean 37.5, median 37.3 and standard
deviation 16.5. It covers the lower part of the 0–132 MDS-UPDRS range: 99 % of
visits are below 73.4. The distribution is close to symmetric (skewness 0.05),
so it is not skewed.

**4. Columns associated with `target`.**
Correlations with `target`: `off` 0.87, `on` 0.67, `age` 0.31, `ledd` 0.30,
`age_at_diagnosis` 0.13. The others are close to 0.
`off` is strongly related to `target` but is not a copy of it. On average it
is 5.4 points below the target, with a standard deviation of 8.7 points
(mean absolute gap 7.5). `on` is much lower than the target (21.8 points on
average), as expected, since it is measured under medication.

**5. Is `ledd` missing at random?**
No. `ledd` is missing in 35.9 % of cohort A visits and 42.6 % of cohort B
visits. The missingness depends on the cohort, so it carries information.
Clinically, a missing dose can mean that the patient is not yet on levodopa
(early or untreated disease), or that the cohorts recorded medication
differently. Either way, it should be kept as a signal and not filled with a
median.

**6. Columns not to feed to a model.**
- `Index` is a row identifier with no clinical meaning.
- `patient_id` is an identifier. The model could memorise patients, and test
  patients are all new.
- `target` is the label.
- `rater_id` was flagged in the EDA as a possible leak, so we left it out on
  Day 2. We later checked that every test rater also appears in train and used
  it as a feature in experiments 18–22. A clinician's scoring bias is a
  legitimate signal, not a leak.

The leak we actually ran into came from engineered features, not from raw
columns. Features computed from the target of earlier visits (scripts 08–15)
cannot be computed on the test set (see section 4 of the project `README.md`).

### Dummy baseline

**1. Dummy RMSE vs. standard deviation.**
The dummy RMSE is 16.48 on the default random 20 % holdout, and 16.50 with
grouped CV. The standard deviation of `target` is 16.50. They are almost
equal by construction: the dummy always predicts the training mean, so its
RMSE is the square root of the mean squared distance to the mean. That is the
standard deviation of the target.

**2. Shape of the prediction-error plot.**
All predictions take the same value (about 37.5), so the points form a single
flat line across the whole range of actual values. Visits with a low true
score are over-predicted, and those with a high score are under-predicted. A
perfect model would put every point on the diagonal (predicted = actual).

**3. When is an improvement real?**
The fold-to-fold standard deviation of the RMSE is about 0.3 for the dummy,
0.24 for Ridge and 0.14 for HGBR. Across random 20 % holdouts, the dummy
RMSE varies by about 0.08. We treat a gain as real when it is clearly larger
than this spread (roughly 0.3 RMSE) and when the new model wins on every fold
of the same split. Smaller gains are noise.

---

## Day 2

### Step 1: Ridge

**1. Gain over the dummy.**
Ridge lowers the grouped-CV RMSE from 16.50 to 10.60, a 36 % reduction. That is
a big win for a linear model on 8 numeric columns, and it comes mostly from
`off` and `on`.

**2. Coefficients.**
Raw coefficients (alpha = 1): `time_since_intake_on` 1.70, `age` 1.07,
`age_at_diagnosis` −0.96, `off` 0.59, `on` 0.48, `sexM` −0.20,
`time_since_intake_off` −0.15, `ledd` −0.0005.
`time_since_intake_on` has the largest absolute weight. The signs of `off` and
`on` are positive, as expected: a higher measured motor score means a more
severe true OFF score. `age` (+) and `age_at_diagnosis` (−) have opposite
signs and similar sizes. Together they act like time since diagnosis: the
longer the disease has lasted, the higher the score.

**3. Why raw coefficients cannot be compared.**
A coefficient is the effect of one unit of its feature, and the units are very
different. `ledd` is in mg with a standard deviation of 174, so its tiny
coefficient still moves the prediction. `time_since_intake_on` has a standard
deviation below 1, so its large coefficient has a small total effect.
Multiplying each coefficient by the feature's standard deviation gives a
fairer ranking: `age` 12.4, `age_at_diagnosis` −10.1, `off` 7.7, `on` 4.4,
`time_since_intake_on` 1.5. `age` and `age_at_diagnosis` are also strongly
correlated, so their individual weights are unstable. To compare coefficients
directly, the features must be scaled first (for example with
`StandardScaler`).

### Step 2: Tuning alpha

**1. How much does the RMSE move?**
Almost not at all. The RMSE is 10.597 for alpha = 0.1, 1, 10 and 100. With
44,590 rows and 8 features, the data term of the Ridge loss is huge compared
with the penalty `alpha × ||w||²`, so the penalty has almost no effect. There
is also very little overfitting to correct: 8 coefficients cannot memorise
44,590 rows.

**2. Do the coefficients change?**
Barely. From alpha = 0.1 to 100, `sexM` goes from −0.1997 to −0.1978 and
`time_since_intake_on` from 1.704 to 1.699. The other coefficients change only
in the fourth decimal. A visible shrinkage needs alpha around 10,000.

### Step 3: Patient-grouped CV

**1. Random split vs. grouped CV.**
The dummy goes from 16.48 to 16.50 and Ridge from 10.49 to 10.60. Ridge
changed more, because the dummy uses no features and cannot exploit the leak.
Both changes are small: these simple models see one visit at a time and have
too little capacity to memorise patients. The gap becomes large for models
that use a patient's history. That is exactly where our leaky scripts 08–15
went wrong.

**2. Which estimate is closer to Kaggle?**
Grouped CV. We compare on the models whose grouped-CV and Kaggle public scores
are both recorded in this project (experiments 20–22). Their grouped-CV scores
(4.01, 3.87, 3.87) are close to their Kaggle scores (3.847, 3.677, 3.623), and
slightly pessimistic. The Kaggle test set holds out whole
patients, which is what grouped CV simulates. From step 3 on, we used grouped
CV only.

### Step 4: HistGradientBoosting

**1. Does HGBR beat Ridge on every fold?**
Yes. HGBR has a mean RMSE of 7.75 against 10.60 for Ridge (−27 %). It wins on
all 5 folds, by 2.7 to 3.1 points each time.

**2. Fit and predict times.**
Average per fold: HGBR fits in 0.65 s and predicts in 0.015 s, while Ridge
fits in 0.026 s and predicts in 0.002 s. HGBR is about 25 times slower to fit
and 7 times slower to predict. It still takes less than a second, so the
27 % gain is clearly worth it.

**3. Feature importance.**
Permutation importance (increase in RMSE when the feature is shuffled,
averaged over the 5 validation folds): `off` 7.5, `on` 5.2, `age` 2.3,
`age_at_diagnosis` 1.9, `time_since_intake_on` 0.3, `ledd` 0.06. `sexM` and
`time_since_intake_off` are close to 0.
The ranking differs from Ridge. The raw Ridge coefficients put
`time_since_intake_on` first only because of units. Even after scaling, Ridge
leaned more on `age` and `age_at_diagnosis` than on `off`. HGBR relies mostly
on the measured scores. It can also use whether `off` is missing, and it can
model non-linear relations that a linear model cannot.

### Step 5: skrub `tabular_pipeline`

**1. Did `gene` and `cohort` help?**
No. The RMSE is 7.752 on the Hub (7.754 in our re-run) against 7.752 for the
numeric HGBR. The difference is far smaller than the fold-to-fold spread
(0.14). `gene` and `cohort` are only weakly related to the target once `off`
and `on` are known.

**2. Which encoder was chosen?**
`gene` (4 labels plus missing values) and `cohort` (2 labels) were both
detected as low-cardinality columns. `tabular_pipeline("regressor")` uses
`ToCategorical` for them, so they are passed to HGBR as native categorical
features. The numeric columns pass through unchanged.
This matches the cardinalities in the EDA: neither column is
high-cardinality, so no `StringEncoder` was needed. With a linear model, the
same pipeline would use one-hot encoding instead, which is what the lab
instructions describe.

### Step 6: skrub DataOps

**1. Are the scores the same as step 5?**
Almost: 7.762 against 7.752. The difference has two sources:
- The DataOps graph applies a default `TableVectorizer()`, which one-hot
  encodes `gene` and `cohort` (`gene_GBA+`, `gene_LRRK2+`, …, `cohort_B`).
  `tabular_pipeline` instead passes them as native categoricals.
- The HGBR in the graph has `random_state=0`, while the one inside
  `tabular_pipeline` has none. HGBR uses a random validation split for early
  stopping on large datasets, so runs can differ slightly.

The 0.01 gap is well inside the fold-to-fold spread, so the two approaches are
equivalent.

**2. Which columns does `X_op` contain?**
`cohort`, `sexM`, `gene`, `age_at_diagnosis`, `age`, `ledd`,
`time_since_intake_on`, `time_since_intake_off`, `on`, `off`. These are the
same columns as in step 5: `Index`, `patient_id`, `target` and `rater_id`
are dropped in both. `Index` and `patient_id` therefore cannot appear in the
feature importance. If they did, the model would be using identifiers. That
would be leakage on a random split, and useless on the test set, where every
patient is new.

### Step 7: Wrap-up

**1. Which change gave the biggest improvement?**
Within Day 2, the biggest gain came from switching from Ridge to HGBR, which
also keeps missing values as they are: 10.60 → 7.75 (−27 %). We did not run a
median-imputed HGBR, so we cannot split that gain between the model and the
missing values.
- Grouped CV did not improve the score. It is a change of evaluation, and it
  made our estimates slightly worse but honest.
- `gene` and `cohort`, and the DataOps rewrite, gave nothing.

After Day 2, the largest honest gains came from:
- per-patient history features (7.75 → 4.66);
- future visits and the clinician (`rater_id`) (→ 4.04);
- the two-stage model (→ 3.87 in CV, 3.623 on Kaggle).

**2. Where does the best Day 2 model fail most?**
We looked at the 1 % largest errors of the Day 2 HGBR (out-of-fold, absolute
error above 21.5 points), compared with all visits:
- **Severe patients.** The mean true score is 60.6, against 37.5 overall.
  The model under-predicts them by 20.4 points on average, regressing them
  towards the mean.
- **Missing `off`.** `off` is missing in 54.7 % of these visits, against
  42.4 % overall, so the model loses its best clue.
- **Later in the disease.** The median visit number is 6 (against 4), and the
  median time since diagnosis is 6.3 years (against 5.3).

Cohort B is not over-represented (6.5 %, against 11.1 % overall), and the
intake delays look similar to the other visits. The typical failure is an
advanced patient seen without an OFF measurement.

**3. What has not been addressed yet?**
At the end of Day 2, none of the three problems in `CONTEXT.md` was really
handled. Each visit was seen alone (no temporal progression), drug timing was
only a raw column, and missingness was handled only implicitly by HGBR. From
experiment 16 on we addressed:
- **temporal progression**, with lags, rolling means, future visits and the
  stage-2 smoothing;
- **missingness**, with explicit missing flags and the count of past `off`
  measures.

**Drug timing** is still not modelled explicitly. The generative model in
`CONTEXT.md` says that the levodopa effect follows the blood concentration (fast
absorption, then decay). The next feature to build is therefore a
pharmacokinetic curve of `time_since_intake_on` and `time_since_intake_off`
(for example an absorption–decay function), combined with `ledd`. Our earlier
attempt (`09_pk_features.py`) was part of the leaky series, so it was never
evaluated on honest features.
