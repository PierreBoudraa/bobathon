"""
Improvement ST1 — Features temporelles : lags, deltas, agrégats historiques par patient.

Chaque visite est triee par (patient_id, age). Les features de lag et expanding
donnent au modele acces a l'historique du patient sans fuiter le CV
(GroupKFold garantit que toutes les visites d'un patient restent dans le meme fold).
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
# Chargement
# ---------------------------------------------------------------------------
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test  = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index").sort_values(["patient_id", "age"]).reset_index(drop=True)
X_test  = X_test.sort_values(["patient_id", "age"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Fonction de feature engineering
# ---------------------------------------------------------------------------
def build_features(df, is_test=False):
    """Ajoute les features temporelles. df doit etre trie par (patient_id, age)."""
    g = df.groupby("patient_id", sort=False)

    # -- Lags (visite precedente du meme patient) --
    for col in ["off", "on", "ledd"]:
        df[f"{col}_lag1"] = g[col].shift(1)

    # target_lag1 disponible seulement sur le train
    if not is_test:
        df["target_lag1"] = g["target"].shift(1)
    else:
        df["target_lag1"] = np.nan

    # -- Deltas (variation visite-à-visite) --
    df["off_delta"]  = df["off"]  - df["off_lag1"]
    df["on_delta"]   = df["on"]   - df["on_lag1"]
    df["ledd_delta"] = df["ledd"] - df["ledd_lag1"]
    df["age_delta"]  = g["age"].diff()   # intervalle entre visites

    # -- Agrégats expanding (exclut la visite courante) --
    for col in ["off", "on"]:
        shifted = g[col].shift(1)                           # exclure courante
        df[f"{col}_cumean"] = shifted.groupby(df["patient_id"]).expanding().mean().values
        df[f"{col}_cumax"]  = shifted.groupby(df["patient_id"]).expanding().max().values
        df[f"{col}_cumin"]  = shifted.groupby(df["patient_id"]).expanding().min().values

    if not is_test:
        shifted_target = g["target"].shift(1)
        df["target_cumean"] = shifted_target.groupby(df["patient_id"]).expanding().mean().values

        # Biais de correction historique : mean(target - off) sur visites precedentes
        bias = g.apply(lambda x: (x["target"] - x["off"]).shift(1).expanding().mean(), include_groups=False)
        df["bias_correction_mean"] = bias.values
    else:
        df["target_cumean"]       = np.nan
        df["bias_correction_mean"] = np.nan

    # -- Numéro de visite et temps depuis diagnostic --
    df["visit_number"]       = g.cumcount() + 1
    df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]

    return df


visits   = build_features(visits, is_test=False)
X_test   = build_features(X_test,  is_test=True)

# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------
base_cols = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
extra_cols = [
    "off_lag1", "on_lag1", "ledd_lag1", "target_lag1",
    "off_delta", "on_delta", "ledd_delta", "age_delta",
    "off_cumean", "off_cumax", "off_cumin",
    "on_cumean", "on_cumax", "on_cumin",
    "target_cumean", "bias_correction_mean",
    "visit_number", "time_since_diagnosis",
]
feature_cols = base_cols + extra_cols

X = visits[feature_cols]
y = visits["target"]

print(f"Feature set : {len(feature_cols)} features")
print(f"Dont NaN target_lag1 (1eres visites) : {X['target_lag1'].isna().sum()}")

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
print("\nIMP-ST1 — HGBR + temporal features :")
print("  RMSE grouped-CV =", report.metrics.rmse())

project.put("08_temporal_features", report)
print("Report '08_temporal_features' pousse sur Skore Hub")

# ---------------------------------------------------------------------------
# Fit final + soumission
# ---------------------------------------------------------------------------
X_test_feat = X_test[feature_cols]
final = clone(hgbr).fit(X, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test_feat)
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/08_temporal_features.csv", index=False)
print(f"submissions/08_temporal_features.csv ecrit ({len(submission)} lignes)")
