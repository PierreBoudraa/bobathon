"""
Rater-ID categorical feature — experiments/18_rater_id.py

Adds `rater_id` as a raw integer-coded categorical on top of
experiments/17_future_visits.py (lead1/lead2/patient-mean for off/on/ledd).

Key constraints
  - rater_id is label-encoded using the union of train+test categories so
    every code is consistent across both DataFrames.
  - No target encoding, no column derived from `target`.
  - Lead and patient-mean features unchanged from experiment 17.
  - GroupKFold(5) on patient_id, single seed=0, no ensemble.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------------------------
# Load & sort
# ---------------------------------------------------------------------------
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test  = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index").sort_values(["patient_id", "age"]).reset_index(drop=True)
X_test  = X_test.sort_values(["patient_id", "age"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Feature engineering — same as 16_honest_final
# ---------------------------------------------------------------------------
def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical (lag/cumulative) features for off, on, ledd.
    Safe to call on any slice: all features reference past values only.
    """
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
    """
    Lead-1 and lead-2 values for off / on / ledd (future visits of the same
    patient), plus patient-level mean over all visits for each of those three
    columns.

    All source columns (off, on, ledd) are present in X_test — no target leakage.
    """
    g = df.groupby("patient_id", sort=False)

    for col in ["off", "on", "ledd"]:
        # Next visit value (lead-1) and the one after (lead-2)
        df[f"{col}_lead1"] = g[col].shift(-1)
        df[f"{col}_lead2"] = g[col].shift(-2)
        # Patient-level mean across ALL visits in the dataset
        df[f"{col}_patient_mean"] = g[col].transform("mean")

    return df


visits = build_lag_features(visits)
visits = add_future_features(visits)
X_test  = build_lag_features(X_test)
X_test  = add_future_features(X_test)

# ---------------------------------------------------------------------------
# rater_id — label-encode using joint category set (train ∪ test)
# OrdinalEncoder with handle_unknown='use_encoded_value' keeps future-proof.
# No NaNs in either split (verified).
# ---------------------------------------------------------------------------
enc = OrdinalEncoder(
    categories=[sorted(set(visits["rater_id"]) | set(X_test["rater_id"]))],
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    dtype=np.float64,
)
visits["rater_id_enc"] = enc.fit_transform(visits[["rater_id"]])
X_test["rater_id_enc"] = enc.transform(X_test[["rater_id"]])

# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------
EXCLUDE = {"Index", "patient_id", "cohort", "gene", "rater_id",
           "target", "age_first_visit"}

feature_cols = [
    c for c in visits.columns
    if c not in EXCLUDE and not c.startswith("target")
]

X = visits[feature_cols]
y = visits["target"]
print(f"Features : {len(feature_cols)}")

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
# Evaluation — GroupKFold n=5, single seed=0
# ---------------------------------------------------------------------------
groups    = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

BEST_CFG = dict(
    max_iter=1000,
    learning_rate=0.01,
    max_leaf_nodes=63,
    min_samples_leaf=5,
)

hgbr   = HistGradientBoostingRegressor(random_state=0, **BEST_CFG)
report = skore_evaluate(hgbr, X, y, splitter=cv_splits)
rmse   = float(report.metrics.rmse().iloc[0].iloc[0])
print(f"\nRMSE GroupKFold n=5 (single seed=0) : {rmse:.4f}")
print(f"Delta vs 17_future_visits (4.2366)  : {rmse - 4.2366:+.4f}")

project.put("18_rater_id", report)
print("Report '18_rater_id' pushed to Skore Hub")

# ---------------------------------------------------------------------------
# Test predictions (single model, full train)
# ---------------------------------------------------------------------------
hgbr_full = HistGradientBoostingRegressor(random_state=0, **BEST_CFG)
hgbr_full.fit(X, y)
test_preds = hgbr_full.predict(X_test[feature_cols])

print(f"\nTest prediction distribution :")
print(f"  mean={test_preds.mean():.3f}  std={test_preds.std():.3f}  "
      f"min={test_preds.min():.1f}  max={test_preds.max():.1f}")
print(f"Target train distribution :")
print(f"  mean={y.mean():.3f}  std={y.std():.3f}  "
      f"min={y.min():.1f}  max={y.max():.1f}")

Path("submissions").mkdir(exist_ok=True)
submission = X_test[["Index"]].copy()
submission["target"] = test_preds
submission.to_csv("submissions/18_rater_id.csv", index=False)
print(f"\nsubmissions/18_rater_id.csv written ({len(submission)} rows)")
