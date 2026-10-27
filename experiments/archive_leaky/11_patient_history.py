"""
Improvement ST4 — Agrégats patient avancés : pente de progression, rolling windows,
biais de correction détaillé par patient.

Construit sur ST1. Ajoute des features qui résument toute l'histoire clinique du patient :
- Pente de régression target vs age (vitesse de dégradation)
- Rolling window 3 visites (off, on, target)
- Std historique de off/on (variabilité individuelle)
- target_lag2 et target_lag3
- Temps depuis la première visite
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
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
# Feature engineering ST1
# ---------------------------------------------------------------------------
def build_temporal_features(df, is_test=False):
    g = df.groupby("patient_id", sort=False)
    for col in ["off", "on", "ledd"]:
        df[f"{col}_lag1"] = g[col].shift(1)
    df["target_lag1"] = g["target"].shift(1) if not is_test else np.nan
    df["target_lag2"] = g["target"].shift(2) if not is_test else np.nan
    df["target_lag3"] = g["target"].shift(3) if not is_test else np.nan
    df["off_delta"]  = df["off"]  - df["off_lag1"]
    df["on_delta"]   = df["on"]   - df["on_lag1"]
    df["ledd_delta"] = df["ledd"] - df["ledd_lag1"]
    df["age_delta"]  = g["age"].diff()
    for col in ["off", "on"]:
        shifted = g[col].shift(1)
        df[f"{col}_cumean"] = shifted.groupby(df["patient_id"]).expanding().mean().values
        df[f"{col}_cumax"]  = shifted.groupby(df["patient_id"]).expanding().max().values
        df[f"{col}_cumin"]  = shifted.groupby(df["patient_id"]).expanding().min().values
        df[f"{col}_custd"]  = shifted.groupby(df["patient_id"]).expanding().std().values
    if not is_test:
        shifted_target = g["target"].shift(1)
        df["target_cumean"] = shifted_target.groupby(df["patient_id"]).expanding().mean().values
        bias = g.apply(lambda x: (x["target"] - x["off"]).shift(1).expanding().mean(), include_groups=False)
        df["bias_correction_mean"] = bias.values
        # Std du biais de correction
        bias_std = g.apply(lambda x: (x["target"] - x["off"]).shift(1).expanding().std(), include_groups=False)
        df["bias_correction_std"] = bias_std.values
    else:
        df["target_cumean"]        = np.nan
        df["bias_correction_mean"] = np.nan
        df["bias_correction_std"]  = np.nan
    df["visit_number"]         = g.cumcount() + 1
    df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    # Temps depuis la première visite du patient
    df["age_first_visit"]       = g["age"].transform("first")
    df["time_since_first_visit"] = df["age"] - df["age_first_visit"]
    return df


# ---------------------------------------------------------------------------
# Feature engineering ST4 : rolling 3 visites + pente de progression
# ---------------------------------------------------------------------------
def build_history_features(df, is_test=False):
    g = df.groupby("patient_id", sort=False)

    # Rolling 3 visites (exclut la visite courante via shift)
    for col in ["off", "on"]:
        shifted = g[col].shift(1)
        df[f"{col}_roll3_mean"] = shifted.groupby(df["patient_id"]).transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

    if not is_test:
        shifted_target = g["target"].shift(1)
        df["target_roll3_mean"] = shifted_target.groupby(df["patient_id"]).transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        # Pente de progression : slope de target vs age sur les visites precedentes
        def calc_slope(subdf):
            slopes = [np.nan]  # premiere visite : pas de pente
            for i in range(1, len(subdf)):
                x = subdf["age"].iloc[:i].values
                y_vals = subdf["target"].iloc[:i].values
                if len(x) >= 2 and np.std(x) > 0:
                    slope, *_ = scipy_stats.linregress(x, y_vals)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
            return pd.Series(slopes, index=subdf.index)

        slopes_all = g.apply(calc_slope, include_groups=False)
        df["disease_slope"] = slopes_all.values
    else:
        df["target_roll3_mean"] = np.nan
        df["disease_slope"]     = np.nan

    return df


visits = build_temporal_features(visits, is_test=False)
visits = build_history_features(visits,  is_test=False)
X_test = build_temporal_features(X_test,  is_test=True)
X_test = build_history_features(X_test,   is_test=True)

# ---------------------------------------------------------------------------
# Feature set ST1 + ST4
# ---------------------------------------------------------------------------
base_cols = ["sexM", "age_at_diagnosis", "age", "ledd",
             "time_since_intake_on", "time_since_intake_off", "on", "off"]
temporal_cols = [
    "off_lag1", "on_lag1", "ledd_lag1",
    "target_lag1", "target_lag2", "target_lag3",
    "off_delta", "on_delta", "ledd_delta", "age_delta",
    "off_cumean", "off_cumax", "off_cumin", "off_custd",
    "on_cumean",  "on_cumax",  "on_cumin",  "on_custd",
    "target_cumean", "bias_correction_mean", "bias_correction_std",
    "visit_number", "time_since_diagnosis", "time_since_first_visit",
]
history_cols = [
    "off_roll3_mean", "on_roll3_mean",
    "target_roll3_mean", "disease_slope",
]
feature_cols = base_cols + temporal_cols + history_cols
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
# Evaluation GroupKFold avec meilleure config HGBR (de ST3)
# ---------------------------------------------------------------------------
groups    = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

best_cfg = dict(max_iter=500, learning_rate=0.02, max_leaf_nodes=63, min_samples_leaf=10)
hgbr = HistGradientBoostingRegressor(random_state=0, **best_cfg)

report = skore_evaluate(hgbr, X, y, splitter=cv_splits)
print("\nIMP-ST4 — HGBR + temporal + history features :")
print("  RMSE grouped-CV =", report.metrics.rmse())

project.put("11_patient_history", report)
print("Report '11_patient_history' pousse sur Skore Hub")

# ---------------------------------------------------------------------------
# Fit final + soumission
# ---------------------------------------------------------------------------
final = clone(hgbr).fit(X, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/11_patient_history.csv", index=False)
print(f"submissions/11_patient_history.csv ecrit ({len(submission)} lignes)")
