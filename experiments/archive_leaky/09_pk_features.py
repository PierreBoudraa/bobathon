"""
Improvement ST2 — Features pharmacocinetiques PK + indicateurs de missingness.

Construit sur ST1. Ajoute :
- Courbe PK levodopa : concentration estimee au moment de l'evaluation
- Correction PK des scores ON/OFF mesures
- Indicateurs binaires de missingness comme features explicites
- Interactions ledd x scores, ratio ON/OFF
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Chargement + tri
# ---------------------------------------------------------------------------
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test  = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index").sort_values(["patient_id", "age"]).reset_index(drop=True)
X_test  = X_test.sort_values(["patient_id", "age"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Feature engineering ST1 (temporal)
# ---------------------------------------------------------------------------
def build_temporal_features(df, is_test=False):
    g = df.groupby("patient_id", sort=False)
    for col in ["off", "on", "ledd"]:
        df[f"{col}_lag1"] = g[col].shift(1)
    df["target_lag1"] = g["target"].shift(1) if not is_test else np.nan
    df["off_delta"]  = df["off"]  - df["off_lag1"]
    df["on_delta"]   = df["on"]   - df["on_lag1"]
    df["ledd_delta"] = df["ledd"] - df["ledd_lag1"]
    df["age_delta"]  = g["age"].diff()
    for col in ["off", "on"]:
        shifted = g[col].shift(1)
        df[f"{col}_cumean"] = shifted.groupby(df["patient_id"]).expanding().mean().values
        df[f"{col}_cumax"]  = shifted.groupby(df["patient_id"]).expanding().max().values
        df[f"{col}_cumin"]  = shifted.groupby(df["patient_id"]).expanding().min().values
    if not is_test:
        shifted_target = g["target"].shift(1)
        df["target_cumean"] = shifted_target.groupby(df["patient_id"]).expanding().mean().values
        bias = g.apply(lambda x: (x["target"] - x["off"]).shift(1).expanding().mean(), include_groups=False)
        df["bias_correction_mean"] = bias.values
    else:
        df["target_cumean"]        = np.nan
        df["bias_correction_mean"] = np.nan
    df["visit_number"]         = g.cumcount() + 1
    df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    return df


def build_pk_features(df):
    """Ajoute les features PK et de missingness."""
    # Demi-vie levodopa ~2.3h -> k = ln(2)/2.3 ≈ 0.30
    K = 0.30
    df["levo_conc_on"]  = df["ledd"] * np.exp(-K * df["time_since_intake_on"])
    df["levo_conc_off"] = df["ledd"] * np.exp(-K * df["time_since_intake_off"])

    # Scores corriges par la concentration PK
    df["on_pk_corrected"]  = df["on"]  / (1 + df["levo_conc_on"].fillna(0))
    df["off_pk_corrected"] = df["off"] / (1 + df["levo_conc_off"].fillna(0))

    # Indicateurs binaires de missingness
    df["off_missing"]         = df["off"].isna().astype(int)
    df["on_missing"]          = df["on"].isna().astype(int)
    df["timing_off_missing"]  = df["time_since_intake_off"].isna().astype(int)
    df["ledd_missing"]        = df["ledd"].isna().astype(int)

    # Interactions
    df["ledd_times_off"] = df["ledd"] * df["off"]
    df["ledd_times_on"]  = df["ledd"] * df["on"]
    df["on_off_ratio"]   = df["on"] / (df["off"] + 1)
    df["on_off_diff"]    = df["on"] - df["off"]

    return df


visits = build_temporal_features(visits, is_test=False)
visits = build_pk_features(visits)
X_test = build_temporal_features(X_test,  is_test=True)
X_test = build_pk_features(X_test)

# ---------------------------------------------------------------------------
# Feature set complet ST1 + ST2
# ---------------------------------------------------------------------------
temporal_cols = [
    "off_lag1", "on_lag1", "ledd_lag1", "target_lag1",
    "off_delta", "on_delta", "ledd_delta", "age_delta",
    "off_cumean", "off_cumax", "off_cumin",
    "on_cumean", "on_cumax", "on_cumin",
    "target_cumean", "bias_correction_mean",
    "visit_number", "time_since_diagnosis",
]
pk_cols = [
    "levo_conc_on", "levo_conc_off",
    "on_pk_corrected", "off_pk_corrected",
    "off_missing", "on_missing", "timing_off_missing", "ledd_missing",
    "ledd_times_off", "ledd_times_on", "on_off_ratio", "on_off_diff",
]
base_cols = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
feature_cols = base_cols + temporal_cols + pk_cols

X = visits[feature_cols]
y = visits["target"]
print(f"Feature set : {len(feature_cols)} features")

# ---------------------------------------------------------------------------
# Credentials + login
# ---------------------------------------------------------------------------
def load_skore_credentials():
    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen, dest = set(), None
    for start in starts:
        for folder in [start, *start.parents]:
            if folder in seen: continue
            seen.add(folder)
            candidate = folder / ".skore"
            if candidate.is_file(): dest = candidate; break
            if (folder / ".git").exists(): break
        if dest: break
    if dest is None: raise FileNotFoundError("Missing .skore")
    cfg = json.loads(dest.read_text(encoding="utf-8"))
    os.environ["SKORE_HUB_URI"]     = cfg["hub_url"]
    os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
    return {k: v for k, v in cfg.items() if k != "api_key"}

from skore import evaluate as skore_evaluate, login, Project

cfg     = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

# ---------------------------------------------------------------------------
# Evaluation GroupKFold
# ---------------------------------------------------------------------------
groups    = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

hgbr = HistGradientBoostingRegressor(random_state=0)
report = skore_evaluate(hgbr, X, y, splitter=cv_splits)
print("\nIMP-ST2 — HGBR + temporal + PK features :")
print("  RMSE grouped-CV =", report.metrics.rmse())

project.put("09_pk_features", report)
print("Report '09_pk_features' pousse sur Skore Hub")

# ---------------------------------------------------------------------------
# Fit final + soumission
# ---------------------------------------------------------------------------
final = clone(hgbr).fit(X, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/09_pk_features.csv", index=False)
print(f"submissions/09_pk_features.csv ecrit ({len(submission)} lignes)")
