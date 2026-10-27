"""
Day 2 — ST7 : Best model of Day 2 and final Kaggle submission.

Comparison of the grouped-CV RMSE of all models:
  dummy          : 16.48
  ridge          : 10.60
  hgbr           :  7.752  <-- best
  tabular        :  7.752  (tie, categorical columns add nothing)
  dataops        :  7.762

The best one is the numeric HGBR (ST4): simpler, same RMSE as the tabular model.
This script refits it cleanly on the whole train set, pushes the final report to the Hub,
and writes submissions/07_best.csv (copy of 04_hgbr with an updated key).

Report URL to paste in the description of the Kaggle submission.
"""
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

# -- Loading the data ---------------------------------------------------
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

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

# -- Final report: HGBR with grouped CV -------------------------------------
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

hgbr = HistGradientBoostingRegressor(random_state=0)
report = skore_evaluate(hgbr, X, y, splitter=cv_splits)
print("ST7 — Best model (HGBR):")
print("  RMSE grouped-CV =", report.metrics.rmse())

project.put("07_best", report)
print("Report '07_best' pushed to Skore Hub")

# -- Final fit on the whole train set and Kaggle submission -------------------------
final = clone(hgbr).fit(X, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/07_best.csv", index=False)
print(f"submissions/07_best.csv written ({len(submission)} rows)")
print()
print("=== KAGGLE SUBMISSION ===")
print("File     : submissions/07_best.csv")
print("Hub URL  : see 'Consult your report at' above")
print("Action   : upload the CSV to Kaggle, paste the URL in the description.")
