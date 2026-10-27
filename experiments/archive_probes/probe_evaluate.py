import pandas as pd
from sklearn.dummy import DummyRegressor
from skore import evaluate

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
visits = X_train.merge(y_train, on="Index")

feature_cols = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
X = visits[feature_cols]
y = visits["target"]

print("Running skore.evaluate...")
dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)
print("Done!")
print("RMSE:", report.metrics.rmse())
