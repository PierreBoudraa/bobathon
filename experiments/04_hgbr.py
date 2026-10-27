"""
Day 2 — ST4 : HistGradientBoostingRegressor with native missing-value handling.

HGBR handles NaN directly: a missing "off" is a clinical signal
(visit in the ON state only, OFF exam skipped), not noise to impute.
No SimpleImputer here. Compared with Ridge using grouped CV.
Writes submissions/04_hgbr.csv.
"""
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

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

# -- GroupKFold (same splits as ST3) -----------------------------------------
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# -- Models ------------------------------------------------------------------
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
hgbr = HistGradientBoostingRegressor(random_state=0)  # no imputation

# -- Evaluation with grouped CV -----------------------------------------------
report_hgbr = skore_evaluate(hgbr, X, y, splitter=cv_splits)

print("ST4 — HGBR vs Ridge (grouped CV) :")
print("  HGBR RMSE =", report_hgbr.metrics.rmse())
print("  (reminder: ridge grouped-CV mean ~ 10.60)")

project.put("04_hgbr", report_hgbr)
print("Report '04_hgbr' pushed to Skore Hub")

# -- Final fit and Kaggle submission -------------------------------------------
final = clone(hgbr).fit(X, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/04_hgbr.csv", index=False)
print(f"submissions/04_hgbr.csv written ({len(submission)} rows)")
