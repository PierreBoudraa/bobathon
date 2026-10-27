"""
Two-stage model (fixed final fit) — experiments/22_two_stage_fixed.py

Same base features as experiments/19_rater_categorical.py. No column derived
from `target` at any point.

Stage 1
  Fit HGBR on the base features. Produce OOF predictions via GroupKFold(5)
  on patient_id. Because GroupKFold puts every visit of a patient in the
  same fold, the OOF prediction for every visit is computed by a model that
  has never seen any visit of that patient.
  For the test set: average the 5 fold-models' predictions.

Stage 2 features (added on top of base features)
  For each visit, the within-patient lag1–3 and lead1–3 of the stage-1
  prediction, plus the patient's mean stage-1 prediction.
  All derived from the stage-1 OOF column, never from `target`.

  For the test set: the same transformations applied to `s1_test`.

Stage 2 evaluation
  Same 5 GroupKFold splits. For fold k:
    - The stage-2 model trains on rows NOT in fold k; the s2 features of
      those rows are the OOF s1 preds from the 4 other folds.
    - Validation rows (fold k) carry s1 OOF preds produced by a model
      trained on the 4 other folds.
  Small caveat: the stage-1 models of the other folds have seen the
  validation patients, so the validation score can be slightly optimistic
  (usual stacking caveat).

Single seed=0, no ensemble.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Load & sort
# ---------------------------------------------------------------------------
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test  = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index").sort_values(["patient_id", "age"]).reset_index(drop=True)
X_test  = X_test.sort_values(["patient_id", "age"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Base feature engineering (identical to 19_rater_categorical)
# ---------------------------------------------------------------------------
def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("patient_id", sort=False)

    for col in ["off", "on", "ledd"]:
        df[f"{col}_lag1"] = g[col].shift(1)
        df[f"{col}_lag2"] = g[col].shift(2)
        df[f"{col}_delta"] = df[col] - df[f"{col}_lag1"]
        shifted = g[col].shift(1)
        df[f"{col}_cumean"] = shifted.groupby(df["patient_id"]).expanding().mean().values
        df[f"{col}_cumax"]  = shifted.groupby(df["patient_id"]).expanding().max().values
        df[f"{col}_cumin"]  = shifted.groupby(df["patient_id"]).expanding().min().values
        df[f"{col}_custd"]  = shifted.groupby(df["patient_id"]).expanding().std().values
        df[f"{col}_roll3"]  = shifted.groupby(df["patient_id"]).transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df[f"{col}_roll5"]  = shifted.groupby(df["patient_id"]).transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

    df["off_lag2_delta"] = df["off_lag1"] - df["off_lag2"]
    df["on_lag2_delta"]  = df["on_lag1"]  - df["on_lag2"]

    df["age_delta"]              = g["age"].diff()
    df["visit_number"]           = g.cumcount() + 1
    df["time_since_diagnosis"]   = df["age"] - df["age_at_diagnosis"]
    df["age_first_visit"]        = g["age"].transform("first")
    df["time_since_first_visit"] = df["age"] - df["age_first_visit"]

    df["off_missing"]  = df["off"].isna().astype(int)
    df["on_missing"]   = df["on"].isna().astype(int)
    df["ledd_missing"] = df["ledd"].isna().astype(int)
    df["off_count"] = (
        g["off_missing"]
        .transform(lambda x: (1 - x).shift(1).expanding().sum())
        .fillna(0)
    )

    df["on_off_diff"]    = df["on"]   - df["off"]
    df["ledd_times_off"] = df["ledd"] * df["off"]
    df["ledd_times_on"]  = df["ledd"] * df["on"]

    return df


def add_future_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("patient_id", sort=False)
    for col in ["off", "on", "ledd"]:
        df[f"{col}_lead1"] = g[col].shift(-1)
        df[f"{col}_lead2"] = g[col].shift(-2)
        df[f"{col}_patient_mean"] = g[col].transform("mean")
    return df


visits = build_lag_features(visits)
visits = add_future_features(visits)
X_test  = build_lag_features(X_test)
X_test  = add_future_features(X_test)

# rater_id encoding
RATER_CATS = sorted(set(visits["rater_id"]) | set(X_test["rater_id"]))
cat_dtype = pd.CategoricalDtype(categories=list(range(len(RATER_CATS))), ordered=False)

visits["rater_id_enc"] = pd.Categorical(
    visits["rater_id"], categories=RATER_CATS
).codes.astype("int8")
X_test["rater_id_enc"] = pd.Categorical(
    X_test["rater_id"], categories=RATER_CATS
).codes.astype("int8")
visits["rater_id_enc"] = visits["rater_id_enc"].astype(cat_dtype)
X_test["rater_id_enc"] = X_test["rater_id_enc"].astype(cat_dtype)

for col in ["off", "on", "ledd"]:
    visits[f"rater_{col}_mean"] = visits.groupby("rater_id")[col].transform("mean")
    X_test[f"rater_{col}_mean"] = X_test.groupby("rater_id")[col].transform("mean")

# ---------------------------------------------------------------------------
# Base feature set (identical to 19)
# ---------------------------------------------------------------------------
EXCLUDE = {"Index", "patient_id", "cohort", "gene", "rater_id",
           "target", "age_first_visit"}

feature_cols = [
    c for c in visits.columns
    if c not in EXCLUDE and not c.startswith("target")
]

X = visits[feature_cols].copy()
y = visits["target"].copy()
print(f"Base features : {len(feature_cols)}")

# ---------------------------------------------------------------------------
# Skore credentials
# ---------------------------------------------------------------------------
def load_skore_credentials():
    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen, dest = set(), None
    for start in starts:
        for folder in [start, *start.parents]:
            if folder in seen:
                continue
            seen.add(folder)
            candidate = folder / ".skore"
            if candidate.is_file():
                dest = candidate
                break
            if (folder / ".git").exists():
                break
        if dest:
            break
    if dest is None:
        raise FileNotFoundError("Missing .skore")
    cfg = json.loads(dest.read_text(encoding="utf-8"))
    os.environ["SKORE_HUB_URI"]     = cfg["hub_url"]
    os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
    return {k: v for k, v in cfg.items() if k != "api_key"}


from skore import evaluate as skore_evaluate, login, Project

cfg     = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

# ---------------------------------------------------------------------------
# CV splits (GroupKFold — every patient's visits land entirely in one fold)
# ---------------------------------------------------------------------------
groups    = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

BEST_CFG = dict(
    max_iter=1000,
    learning_rate=0.01,
    max_leaf_nodes=63,
    min_samples_leaf=5,
)

# ---------------------------------------------------------------------------
# Stage 1 — OOF predictions on train, averaged predictions on test
# ---------------------------------------------------------------------------
print("\nStage 1 OOF ...")
s1_oof  = np.full(len(y), np.nan)
s1_test = np.zeros(len(X_test))

s1_fold_models = []
for train_idx, val_idx in cv_splits:
    m1 = HistGradientBoostingRegressor(
        random_state=0, categorical_features="from_dtype", **BEST_CFG
    )
    m1.fit(X.iloc[train_idx], y.iloc[train_idx])
    s1_oof[val_idx] = m1.predict(X.iloc[val_idx])
    s1_test += m1.predict(X_test[feature_cols])
    s1_fold_models.append(m1)

s1_test /= len(cv_splits)

rmse_s1 = float(np.sqrt(np.mean((s1_oof - y.values) ** 2)))
print(f"Stage 1 OOF RMSE : {rmse_s1:.4f}")

# ---------------------------------------------------------------------------
# Stage 2 features — lag/lead/patient-mean of s1 predictions
# ---------------------------------------------------------------------------
def add_s1_features(df: pd.DataFrame, s1_col: str = "s1_pred") -> pd.DataFrame:
    """
    Add within-patient lag1–3, lead1–3 and patient-mean of `s1_col`.
    df must be sorted by (patient_id, age) — guaranteed by construction.
    """
    g = df.groupby("patient_id", sort=False)
    for k in range(1, 4):
        df[f"s1_lag{k}"]  = g[s1_col].shift(k)
        df[f"s1_lead{k}"] = g[s1_col].shift(-k)
    df["s1_patient_mean"] = g[s1_col].transform("mean")
    return df


# Build the augmented train DataFrame (visits sorted by patient_id, age)
visits2 = visits.copy()
visits2["s1_pred"] = s1_oof
add_s1_features(visits2)

# Build the augmented test DataFrame
X_test2 = X_test.copy()
X_test2["s1_pred"] = s1_test
add_s1_features(X_test2)

# Stage-2 feature set = base features + all s1_* columns
s2_extra = [c for c in visits2.columns if c.startswith("s1_") and c != "s1_pred"]
feature_cols2 = feature_cols + s2_extra
print(f"Stage 2 features: {len(feature_cols2)} ({len(s2_extra)} new s1 features)")

X2      = visits2[feature_cols2].copy()
X2_test = X_test2[feature_cols2].copy()

# ---------------------------------------------------------------------------
# Stage 2 evaluation — same GroupKFold splits
#
# For fold k:
#   - train rows: s1 OOF preds come from models that never saw fold k patients
#   - val rows:   s1 OOF preds come from a model trained on the 4 other folds
#   => within-patient lag/lead of s1 within fold k are safe (all visits of a
#      patient land in the same fold, so no cross-patient contamination)
# Small caveat: the stage-1 models of the other folds have seen the
# validation patients, so the validation score can be slightly optimistic
# (usual stacking caveat).
# ---------------------------------------------------------------------------
print("\nStage 2 CV ...")
s2_oof = np.full(len(y), np.nan)

for train_idx, val_idx in cv_splits:
    m2 = HistGradientBoostingRegressor(
        random_state=0, categorical_features="from_dtype", **BEST_CFG
    )
    m2.fit(X2.iloc[train_idx], y.iloc[train_idx])
    s2_oof[val_idx] = m2.predict(X2.iloc[val_idx])

rmse_s2 = float(np.sqrt(np.mean((s2_oof - y.values) ** 2)))
print(f"Stage 2 OOF RMSE : {rmse_s2:.4f}")
print(f"Delta vs 19_rater_categorical (4.0364) : {rmse_s2 - 4.0364:+.4f}")

# ---------------------------------------------------------------------------
# Push skore report — stage-2 model evaluated on the same splits
# ---------------------------------------------------------------------------
hgbr_s2 = HistGradientBoostingRegressor(
    random_state=0, categorical_features="from_dtype", **BEST_CFG
)
report = skore_evaluate(hgbr_s2, X2, y, splitter=cv_splits)
rmse_report = float(report.metrics.rmse().iloc[0].iloc[0])
print(f"skore RMSE (stage 2, seed=0)    : {rmse_report:.4f}")

project.put("22_two_stage_fixed", report)
print("Report '22_two_stage_fixed' pushed to Skore Hub")

# ---------------------------------------------------------------------------
# Final stage-2 model: fit on all train data
# Stage 2 must be trained on out-of-sample stage-1 predictions — the same
# kind it will see at test time via s1_test. Using s1_oof here (produced by
# models that never saw each patient) is consistent with that requirement.
# ---------------------------------------------------------------------------
visits2_full = visits.copy()
visits2_full["s1_pred"] = s1_oof
add_s1_features(visits2_full)
X2_full = visits2_full[feature_cols2].copy()

m2_full = HistGradientBoostingRegressor(
    random_state=0, categorical_features="from_dtype", **BEST_CFG
)
m2_full.fit(X2_full, y)
test_preds = m2_full.predict(X2_test)

# ---------------------------------------------------------------------------
# Test prediction diagnostics
# ---------------------------------------------------------------------------
n_nan = int(np.isnan(test_preds).sum())
print(f"\nTest prediction distribution :")
print(f"  mean={test_preds.mean():.3f}  std={test_preds.std():.3f}  "
      f"min={test_preds.min():.1f}  max={test_preds.max():.1f}  NaN={n_nan}")
print(f"Target train distribution :")
print(f"  mean={y.mean():.3f}  std={y.std():.3f}  "
      f"min={y.min():.1f}  max={y.max():.1f}")

# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
Path("submissions").mkdir(exist_ok=True)
submission = X_test[["Index"]].copy()
submission["target"] = test_preds
submission.to_csv("submissions/22_two_stage_fixed.csv", index=False)
print(f"\nsubmissions/22_two_stage_fixed.csv written ({len(submission)} rows)")

# ---------------------------------------------------------------------------
# Comparison with submissions/21_two_stage.csv
# ---------------------------------------------------------------------------
sub21 = pd.read_csv("submissions/21_two_stage.csv")["target"].values
corr  = float(np.corrcoef(test_preds, sub21)[0, 1])
rmse_diff = float(np.sqrt(np.mean((test_preds - sub21) ** 2)))
print(f"\nComparison vs submissions/21_two_stage.csv :")
print(f"  Pearson correlation : {corr:.6f}")
print(f"  RMSE difference     : {rmse_diff:.4f}")
