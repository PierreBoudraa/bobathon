"""
Day 2 — ST1 + ST2 : Ridge with median imputation, alpha tuning, Kaggle submission.

ST1: evaluates dummy vs ridge (alpha=1) with the default splitter (random 0.2 holdout).
ST2: looks for the best alpha in [0.1, 1.0, 10.0, 100.0], pushes the best report,
      writes submissions/02_ridge_tuned.csv.
"""
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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

# -- ST1 : dummy vs ridge alpha=1 (random 0.2 holdout) -----------------------
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))

report_dummy = skore_evaluate(dummy, X, y)
report_ridge = skore_evaluate(ridge, X, y)
print("ST1 — dummy vs ridge (alpha=1):")
print(f"  dummy RMSE = {report_dummy.metrics.rmse():.4f}")
print(f"  ridge RMSE = {report_ridge.metrics.rmse():.4f}")
project.put("02_ridge", report_ridge)
print("Report '02_ridge' pushed to Skore Hub")

# -- ST2: alpha tuning -------------------------------------------------------
best_alpha, best_rmse, best_report = None, float("inf"), None
for alpha in [0.1, 1.0, 10.0, 100.0]:
    r = skore_evaluate(make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=alpha)), X, y)
    rmse = r.metrics.rmse()
    print(f"  alpha={alpha:6.1f}  RMSE={rmse:.4f}")
    if rmse < best_rmse:
        best_alpha, best_rmse, best_report = alpha, rmse, r

print(f"Best alpha: {best_alpha}  RMSE={best_rmse:.4f}")
project.put("02_ridge_tuned", best_report)
print("Report '02_ridge_tuned' pushed to Skore Hub")

# -- Kaggle submission --------------------------------------------------------
best_ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=best_alpha))
final = clone(best_ridge).fit(X, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/02_ridge_tuned.csv", index=False)
print(f"submissions/02_ridge_tuned.csv written ({len(submission)} rows, constant prediction ~{submission['target'].mean():.2f})")
