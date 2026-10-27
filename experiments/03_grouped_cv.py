"""
Day 2 — ST3 : Evaluation with a grouped CV on patient_id (GroupKFold n=5).

Replaces the random 0.2 holdout with an honest split: a whole patient goes
either to train or to validation, never to both. Compares dummy vs ridge on
this splitter to see the impact of the Day 1 data leak.
"""
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

# -- Loading the data ---------------------------------------------------
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

visits = X_train.merge(y_train, on="Index")

feature_cols = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
X = visits[feature_cols]
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

# -- GroupKFold : 5 folds, split by patient_id -------------------------------
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
print(f"cv_splits : {len(cv_splits)} folds")
print(f"Example fold 0: train={len(cv_splits[0][0])} visits, val={len(cv_splits[0][1])} visits")

# -- Evaluation of dummy and ridge with grouped CV --------------------------------
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))

report_dummy_cv = skore_evaluate(dummy, X, y, splitter=cv_splits)
report_ridge_cv = skore_evaluate(ridge, X, y, splitter=cv_splits)

print("\nST3 — grouped CV by patient:")
print("  dummy RMSE grouped-CV =", report_dummy_cv.metrics.rmse())
print("  ridge RMSE grouped-CV =", report_ridge_cv.metrics.rmse())
print("  (reminder, random holdout: dummy=16.48, ridge=10.49)")

project.put("03_dummy_grouped_cv", report_dummy_cv)
project.put("03_ridge_grouped_cv", report_ridge_cv)
print("Reports '03_dummy_grouped_cv' and '03_ridge_grouped_cv' pushed to Skore Hub")
