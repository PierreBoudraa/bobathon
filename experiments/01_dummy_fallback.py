"""
Dummy baseline — DLL workaround:
Uses sklearn.model_selection.train_test_split instead of skore.evaluate
when the Cython DLLs of sklearn are blocked.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

warnings.filterwarnings("ignore")

# ── Check that skore.evaluate is available ───────────────────────────────────
SKORE_EVALUATE_OK = False
try:
    from skore import evaluate as skore_evaluate
    SKORE_EVALUATE_OK = True
    print("skore.evaluate available")
except ImportError as e:
    print(f"skore.evaluate not available ({e})")
    print("Using sklearn.model_selection.train_test_split instead")

# ── Loading the data ────────────────────────────────────────────────────
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

# ── Evaluation ────────────────────────────────────────────────────────────────
if SKORE_EVALUATE_OK:
    dummy = DummyRegressor(strategy="mean")
    report = skore_evaluate(dummy, X, y)
    print("Dummy RMSE (skore):", report.metrics.rmse())
else:
    # Fallback: manual 80/20 evaluation
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_tr, y_tr)
    preds = dummy.predict(X_val)
    rmse = np.sqrt(np.mean((preds - y_val.values) ** 2))
    print(f"Dummy RMSE (manual 80/20): {rmse:.4f}")

    # skore project in local mode to store the report
    try:
        from skore import Project
        project = Project(name="bobathon-esilv", mode="local",
                          workspace=str(Path.cwd() / "reports"))
        # project.put cannot be used without an EstimatorReport,
        # so we only note the RMSE
        print("Note: cannot push to the Hub because the DLLs are blocked")
    except Exception as e:
        print(f"Skore Project: {e}")

# ── Submission Kaggle ─────────────────────────────────────────────────────────
dummy_final = DummyRegressor(strategy="mean")
dummy_final.fit(X, y)
X_test_features = X_test[feature_cols]
preds = dummy_final.predict(X_test_features)

submission = pd.DataFrame({"Index": X_test["Index"], "target": preds})
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/01_dummy.csv", index=False)
print(f"Submission saved: submissions/01_dummy.csv ({len(submission)} rows)")
print(f"Predicted value (constant): {preds[0]:.4f}")
