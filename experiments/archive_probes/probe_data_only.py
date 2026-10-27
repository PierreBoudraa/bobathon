import pandas as pd

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
visits = X_train.merge(y_train, on="Index")

feature_cols = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
X = visits[feature_cols]
y = visits["target"]

print("Data loaded:", X.shape)

from sklearn.dummy import DummyRegressor
dummy = DummyRegressor(strategy="mean")
dummy.fit(X, y)
mean_pred = dummy.predict(X[:5])
print("Dummy predictions (first 5):", mean_pred)
print("Training mean:", y.mean())
