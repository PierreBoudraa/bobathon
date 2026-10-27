"""
Dummy baseline — Day 1
Manual evaluation (no sklearn.metrics or skore.evaluate, because the DLLs were blocked).
Pushes the report to the Skore Hub through skore.Project when possible.
Writes submissions/01_dummy.csv for Kaggle.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

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

# ── Manual evaluation (80/20, without sklearn.metrics) ─────────────────────────
n = len(X)
n_val = int(0.2 * n)
rng = np.random.default_rng(0)
val_idx = rng.choice(n, size=n_val, replace=False)
train_idx = np.setdiff1d(np.arange(n), val_idx)

X_tr = X.iloc[train_idx]
X_val = X.iloc[val_idx]
y_tr = y.iloc[train_idx]
y_val = y.iloc[val_idx]

dummy_eval = DummyRegressor(strategy="mean")
dummy_eval.fit(X_tr, y_tr)
preds_val = dummy_eval.predict(X_val)

rmse = float(np.sqrt(np.mean((preds_val - y_val.values) ** 2)))
mae = float(np.mean(np.abs(preds_val - y_val.values)))
print(f"Dummy RMSE (80/20 holdout): {rmse:.4f}")
print(f"Dummy MAE  (80/20 holdout): {mae:.4f}")
print(f"Training mean (= every prediction): {dummy_eval.predict([[0]*8])[0]:.4f}")
print(f"Std of target: {y.std():.4f}  (should be about the RMSE)")

# ── Push to Skore Hub ────────────────────────────────────────────────────────
def load_skore_credentials():
    """Load .skore and export the env vars used by skore.login."""
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


try:
    from skore import evaluate as skore_evaluate, login, Project

    print("\nPushing the report to Skore Hub with skore.evaluate...")
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

    dummy_skore = DummyRegressor(strategy="mean")
    report = skore_evaluate(dummy_skore, X, y)
    print("skore RMSE:", report.metrics.rmse())
    project.put("01_dummy", report)
    print("Report '01_dummy' pushed to Skore Hub ✓")

except ImportError as e:
    print(f"\nCannot use skore (import error): {e}")
    print("The manual RMSE above is valid for Day 1.")

# ── Kaggle submission ─────────────────────────────────────────────────────────
dummy_final = DummyRegressor(strategy="mean")
dummy_final.fit(X, y)
X_test_features = X_test[feature_cols]
preds_test = dummy_final.predict(X_test_features)

submission = pd.DataFrame({"Index": X_test["Index"], "target": preds_test})
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/01_dummy.csv", index=False)
print(f"\nSubmission saved: submissions/01_dummy.csv ({len(submission)} rows)")
print(f"Constant value predicted: {preds_test[0]:.4f}")
