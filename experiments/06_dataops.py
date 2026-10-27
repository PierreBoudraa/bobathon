"""
Day 2 — ST6 : skrub DataOps with GroupKFold baked into the graph.

The grouped CV and the patient_id groups are attached directly to the
mark_as_X node: skore.evaluate reads the CV from the DataOp, so the groups=
argument cannot be forgotten after a copy-paste. SkrubLearner.fit/predict takes
an environment dict {"visits": df} to handle train and test with the same graph.
Writes submissions/06_dataops.csv.
"""
import json
import os
from pathlib import Path

import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skrub import TableVectorizer

# -- Loading the data ---------------------------------------------------
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")

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

# -- DataOps graph ------------------------------------------------------------
data = skrub.var("visits", visits)
groups = data["patient_id"]

X_op = (
    data.drop(["Index", "patient_id", "target", "rater_id"], axis=1, errors="ignore")
    .skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": groups},
    )
)
y_op = data["target"].skb.mark_as_y()

pred = (
    X_op
    .skb.apply(TableVectorizer())
    .skb.apply(HistGradientBoostingRegressor(random_state=0), y=y_op)
)

print("DataOps graph built.")
print("Columns of X_op:", list(data.drop(["Index", "patient_id", "target", "rater_id"], axis=1).skb.eval().columns))

# -- Evaluation: skore reads the CV from the DataOp, data= for the named variable --
report = skore_evaluate(pred, data={"visits": visits})
print("\nST6 — DataOps grouped CV:")
print("  RMSE =", report.metrics.rmse())

project.put("06_dataops", report)
print("Report '06_dataops' pushed to Skore Hub")

# -- Final fit and Kaggle submission through SkrubLearner --------------------------
learner = pred.skb.make_learner()
learner.fit({"visits": visits})
pred_test = learner.predict({"visits": X_test})

submission = X_test[["Index"]].copy()
submission["target"] = pred_test
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/06_dataops.csv", index=False)
print(f"\nsubmissions/06_dataops.csv written ({len(submission)} rows)")
