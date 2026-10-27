"""
Day 2 — ST5 : skrub tabular_pipeline with categorical columns (gene, cohort).

Adds gene and cohort to the feature set through skrub.tabular_pipeline("regressor"),
which encodes them automatically: one-hot for cohort (2 values), StringEncoder
for gene (many labels). Compared with the numeric HGBR on the same grouped
cv_splits. Writes submissions/05_tabular.csv.
"""
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skrub import tabular_pipeline

# -- Loading the data ---------------------------------------------------
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")

# Numeric features only (ST4 baseline)
feature_cols = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
X_numeric = visits[feature_cols]

# Full features: everything except identifiers and target
# rater_id excluded (possible leakage, see the EDA notes; experiment 18 later
# showed that it helps when used as a raw category)
drop_cols = ["Index", "patient_id", "target", "rater_id"]
X_full = visits.drop(columns=drop_cols)

y = visits["target"]

# -- Credentials Skore Hub ----------------------------------------------------
def load_skore_credentials():
    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen = set()
    dest = None
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
    os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
    os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
    return {k: v for k, v in cfg.items() if k != "api_key"}

from skore import evaluate as skore_evaluate, login, Project

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

# -- GroupKFold (same splits as ST3/ST4) -------------------------------------
groups = visits["patient_id"]
cv_splits_numeric = list(GroupKFold(n_splits=5).split(X_numeric, y, groups=groups))
cv_splits_full = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# -- Models ------------------------------------------------------------------
hgbr = HistGradientBoostingRegressor(random_state=0)
model_tabular = tabular_pipeline("regressor")

print("Columns of X_full:", list(X_full.columns))
print("Columns of X_numeric:", list(X_numeric.columns))

# -- Evaluation of the numeric HGBR (reminder of ST4) -----------------------------------
report_hgbr = skore_evaluate(hgbr, X_numeric, y, splitter=cv_splits_numeric)
print("\nST5a — numeric HGBR (reminder):")
print("  HGBR RMSE =", report_hgbr.metrics.rmse())
project.put("05_hgbr_numeric", report_hgbr)
print("Report '05_hgbr_numeric' pushed to Skore Hub")

# -- Evaluation of tabular_pipeline with categorical columns ---------------------------
report_tabular = skore_evaluate(model_tabular, X_full, y, splitter=cv_splits_full)
print("\nST5b — tabular_pipeline (gene + cohort included):")
print("  tabular RMSE =", report_tabular.metrics.rmse())
project.put("05_tabular", report_tabular)
print("Report '05_tabular' pushed to Skore Hub")

# -- Final fit and Kaggle submission -------------------------------------------
X_test_full = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns and c != "target"])
final = clone(model_tabular).fit(X_full, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test_full)
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/05_tabular.csv", index=False)
print(f"\nsubmissions/05_tabular.csv written ({len(submission)} rows)")
