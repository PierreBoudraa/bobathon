# Predicting Parkinson's motor scores (Bobathon ESILV x Probabl x IBM)

Team: shaking-data-dia2 (Pierre Boudraa, Sina Balaei, Nayl Saifoudine, Frank Tankhapanya)

This repo contains our work for the Bobathon Kaggle competition. The goal is to predict a patient's motor score (UPDRS, between 0 and 132) from clinic visits. We built everything step by step with Bob, the IBM coding assistant, and tracked each experiment on the Skore Hub.

Our final model is a two-step gradient boosting model. Its error is about 3.9 points on our own validation, and it scored **3.623** on the Kaggle public leaderboard.

## 1. The task

- **Data:** one row per clinic visit. About 44,600 visits from 5,600 patients for training, and 11,013 visits for testing.
- **Inputs:** age, sex, age at diagnosis, the medication dose (`ledd`), the time since the last intake, and two raw motor scores measured at the visit: `off` (without medication) and `on` (with medication). Many values are missing.
- **Target:** the "true" OFF motor score. The raw `off` measure is noisy and biased (for example, each clinician scores a bit differently), so the target is a cleaned-up version of it.
- **Metric:** RMSE. It is the typical size of our error, in score points. Lower is better.
- **Important detail:** test patients are different from train patients. The model never sees the same person in training and in testing.

## 2. How we measure progress

We use `GroupKFold` with 5 folds, grouped by `patient_id`. All the visits of one patient stay together, either in training or in validation. This copies what happens on the test set, where the patients are new.

On Day 1 we used a random split, which mixes visits of the same patient and looks better than reality. We switched to the grouped split in `03_grouped_cv.py`.

## 3. Results

| Step | What we did | Validation RMSE |
|---|---|---|
| 01 | Dummy model (always predicts the mean) | 16.48 |
| 02, 03 | Ridge regression | 10.60 |
| 04 | Gradient boosting (HGBR), handles missing values by itself | 7.75 |
| 05, 06 | Extra columns (gene, cohort) and the skrub pipeline | 7.75 and 7.76 (no gain) |
| 16 | Features from past visits of the same patient | 4.66 |
| 17 | Added features from the next visits of the same patient | 4.24 |
| 18 | Added the clinician (`rater_id`) | 4.08 |
| 19 | Clinician as a real category, plus per-clinician averages | 4.04 |
| 20 | Average of 5 models with different seeds | 4.01 |
| 21 | Two-step model: a second model uses the predictions for the neighbouring visits | 3.87 |
| 22 | Same two-step model, with the final fit trained on out-of-sample step 1 predictions | 3.87 |

Kaggle public score: **3.847** for step 20, **3.677** for step 21 and **3.623** for step 22. They are all a bit better than our validation, which tells us the validation was honest. Steps 21 and 22 have the same validation score because only the final training changed.

Skore Hub reports, in the workspace `shaking-data-dia2`, project `bobathon-esilv`: `16_honest_final` (cross-validations/53551), `20_final_ensemble` (cross-validations/57118), `21_two_stage` (cross-validations/78175) and `22_two_stage_fixed` (cross-validations/80990). The report for step 20 shows one single model (seed 0, RMSE 4.04), because Skore evaluates one model at a time. The 4.01 comes from averaging the 5 seeds in the script.

## 4. The mistake we found (and why it matters)

Our first improvement scripts (08 to 15) reported a RMSE of about 2.3. That number was wrong.

Those scripts built features from the **target of earlier visits** (for example `target_lag1`, `target_cumean`, `bias_correction_mean`, `disease_slope`, and target encodings). During validation, the earlier targets are known, so the model looked great. On the test set, nobody knows the targets, so these columns were empty and the model lost its main clue.

We noticed it for three reasons:
1. The predictions on the test set had a mean of about 31, while the training target has a mean of about 37.5.
2. The RMSE dropped from 7.75 to 2.5 in a single step, which was too good to be true.
3. Grouping by patient does not protect against this. It stops us from mixing patients, but not from using a patient's past targets as inputs.

**The rule we followed from step 16 on:** a feature is allowed only if we can compute it the same way on the test set, without knowing any target.

The old scripts are kept in `experiments/archive_leaky/` so the story stays visible, but their numbers should not be trusted.

## 5. What the final model does

Everything comes from columns that exist in the test set (`off`, `on`, `ledd`, age, and so on). Nothing comes from the target.

- **Past visits:** the value at the previous 1 or 2 visits, the change since the last visit, running mean, min, max and standard deviation, and averages over the last 3 and 5 visits.
- **Future visits:** the value at the next 1 or 2 visits, and the average over all visits of the patient. This is allowed because the test file contains all visits of each test patient.
- **Missing values:** flags that tell if `off`, `on` or `ledd` is missing, and how many `off` measures the patient had before. A missing `off` often means something about the visit, so we keep that signal.
- **Time:** visit number, time since diagnosis, time since the first visit.
- **Clinician:** `rater_id` as a true category, plus the average `off`, `on` and `ledd` of each clinician.
- **Model, step 1:** a `HistGradientBoostingRegressor` predicts the score of each visit from the features above.
- **Model, step 2:** a second gradient boosting model gets the same inputs, plus the predictions of the first model for the 3 previous and the 3 next visits of the same patient, and the average prediction of the patient. A patient's score changes slowly from one visit to the next, so this helps to smooth out the noise of single visits.
- **Out-of-fold predictions:** step 2 is trained on out-of-fold step 1 predictions, which means each visit is predicted by a model that never saw that patient. This is the same kind of prediction that step 2 receives on the test set, where step 1 is the average of the 5 fold models. In step 21 the final training used in-sample step 1 predictions instead, which are too accurate. Step 22 fixes this and scores better on Kaggle (3.623 instead of 3.677). One small caveat: the step 1 models of the other folds have seen the validation patients, so our validation score may be slightly optimistic. The Kaggle score is better than the validation score (3.87), so we do not think this is a problem.

Step 20 is the same first model, averaged over 5 seeds. It is kept as a simpler alternative.

## 6. Why our approach is original

- **We use the whole history of each patient,** past and future visits, instead of looking at one visit alone. The test file gives us all visits of each test patient, so this is allowed, and it was our biggest honest gain.
- **We model the clinician.** The task is about scoring bias, so we give the model the identity of the clinician and the average scores of each clinician.
- **We smooth the predictions with a second model,** which looks at the predictions for the neighbouring visits. It uses the idea that the disease evolves gradually.
- **We checked our own results.** We found and removed a data leak, and we explain it in section 4, so the scores in this repo can be trusted.

## 7. How to run it

You need Python 3.12 or newer. On Windows, from the project folder:

```powershell
python -m venv --prompt bobathon .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/skore-agent
python -m pytest tests/test_setup.py -v
```

`python scripts/skore-agent` creates your own `.skore` file with your personal API key. Never share it and never commit it to Git. A virtual environment cannot be copied from another computer, so each person has to create their own.

Put `X_train.csv`, `y_train.csv` and `X_test.csv` in `data/`, then run the final model:

```powershell
python experiments/22_two_stage_fixed.py
```

It pushes a report to the Skore Hub and writes `submissions/22_two_stage_fixed.csv` (11,013 rows, no missing values). The 5-seed version of step 20 is `python experiments/20_final_ensemble.py`. It trains 60 models, so it can take 15 to 30 minutes on a normal laptop. Every script from `16` to `22` is independent, so you can run any of them on its own.

## 8. Project layout

```
data/                         raw data and the EDA report
experiments/
  01_dummy_baseline.py        dummy baseline
  01_dummy_fallback.py        backup version of step 01
  02_ridge.py                 ridge regression
  03_grouped_cv.py            grouped validation by patient
  04_hgbr.py                  gradient boosting
  05_tabular.py               categorical columns with skrub
  06_dataops.py               same model as a skrub DataOps pipeline
  07_best.py                  best model of Day 2
  16_honest_final.py          first honest feature set (past visits)
  17_future_visits.py         adds next visits
  18_rater_id.py              adds the clinician
  19_rater_categorical.py     clinician as a category, per-clinician averages
  20_final_ensemble.py        5 seeds averaged
  21_two_stage.py             two steps, first version
  22_two_stage_fixed.py       final model, two steps, with the fixed final training
  archive_leaky/              scripts 08 to 15 (target-based features, wrong scores)
  archive_probes/             small test scripts from the first setup
  eda/                        notes from the exploratory analysis
submissions/                  CSV files sent to Kaggle (17 to 22), not part of the hand-in zip
journal/                      work log: JOURNAL.md and one design note per experiment (16 to 22)
Responses_to_questions.md     answers to the Day 1 and Day 2 workbook questions
```

## 9. What we learned and what we would try next

- A very good validation score is a reason to check, not to celebrate. If it jumps suddenly, look for a leak first.
- Future visits, the clinician identity and the second smoothing step brought the biggest honest gains.
- A second model must be trained on the same kind of inputs it will see at test time. Using out-of-sample step 1 predictions in the final training improved our Kaggle score from 3.677 to 3.623.
- The best teams on the leaderboard are around 2.1 to 2.5, so there is still room. Ideas we have not tried yet:
  - more past and future visits (lag 3, lead 3) and each visit compared with the patient's average;
  - LightGBM or XGBoost mixed with our current model, and tuning of the parameters;
  - averaging several seeds of the two-step model.
