"""
Improvement ST3 — Tuning HGBR sur le feature set ST1 (meilleur jusqu'ici).

Teste plusieurs combinaisons de max_iter / learning_rate / max_leaf_nodes
et un ensemble de 5 seeds pour viser RMSE < 2.196.
Base : features temporelles de ST1 (RMSE 2.531).
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
# Chargement + features ST1 (copie de build_temporal_features)
# ---------------------------------------------------------------------------
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test  = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index").sort_values(["patient_id", "age"]).reset_index(drop=True)
X_test  = X_test.sort_values(["patient_id", "age"]).reset_index(drop=True)

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

visits = build_temporal_features(visits, is_test=False)
X_test = build_temporal_features(X_test,  is_test=True)

base_cols = ["sexM", "age_at_diagnosis", "age", "ledd",
             "time_since_intake_on", "time_since_intake_off", "on", "off"]
temporal_cols = [
    "off_lag1", "on_lag1", "ledd_lag1", "target_lag1",
    "off_delta", "on_delta", "ledd_delta", "age_delta",
    "off_cumean", "off_cumax", "off_cumin",
    "on_cumean", "on_cumax", "on_cumin",
    "target_cumean", "bias_correction_mean",
    "visit_number", "time_since_diagnosis",
]
feature_cols = base_cols + temporal_cols
X = visits[feature_cols]
y = visits["target"]

groups    = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

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
# Grid tuning HGBR (manuel, 6 combinaisons ciblees)
# ---------------------------------------------------------------------------
configs = [
    dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,  min_samples_leaf=20),
    dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,  min_samples_leaf=20),
    dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=20),
    dict(max_iter=500, learning_rate=0.02, max_leaf_nodes=63,  min_samples_leaf=10),
    dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,  min_samples_leaf=10),
    dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,  min_samples_leaf=10),
]

best_rmse, best_cfg, best_report = float("inf"), None, None
print("Tuning HGBR :")
for cfg_params in configs:
    hgbr = HistGradientBoostingRegressor(random_state=0, **cfg_params)
    r = skore_evaluate(hgbr, X, y, splitter=cv_splits)
    rmse_val = float(r.metrics.rmse().iloc[0].iloc[0])
    print(f"  {cfg_params}  ->  RMSE={rmse_val:.4f}")
    if rmse_val < best_rmse:
        best_rmse, best_cfg, best_report = rmse_val, cfg_params, r

print(f"\nMeilleur config : {best_cfg}  RMSE={best_rmse:.4f}")

# ---------------------------------------------------------------------------
# Ensemble de 5 seeds avec la meilleure config
# ---------------------------------------------------------------------------
print("\nEnsemble 5 seeds :")
seed_preds_cv  = []   # pour calculer le RMSE CV de l'ensemble
seed_preds_test = []

for seed in range(5):
    hgbr_s = HistGradientBoostingRegressor(random_state=seed, **best_cfg)
    # Fit sur tout le train pour les predictions test
    hgbr_s.fit(X, y)
    seed_preds_test.append(hgbr_s.predict(X_test[feature_cols]))

ensemble_test = np.mean(seed_preds_test, axis=0)

# Estimation RMSE CV de l'ensemble par OOF
oof_preds = np.full(len(y), np.nan)
for train_idx, val_idx in cv_splits:
    fold_preds = []
    for seed in range(5):
        hgbr_s = HistGradientBoostingRegressor(random_state=seed, **best_cfg)
        hgbr_s.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_preds.append(hgbr_s.predict(X.iloc[val_idx]))
    oof_preds[val_idx] = np.mean(fold_preds, axis=0)

ensemble_rmse = float(np.sqrt(np.mean((oof_preds - y.values) ** 2)))
print(f"  Ensemble 5 seeds OOF RMSE = {ensemble_rmse:.4f}")

# Choisir le meilleur entre single best et ensemble
use_ensemble = ensemble_rmse < best_rmse
final_rmse = ensemble_rmse if use_ensemble else best_rmse
print(f"\nChosen : {'ensemble' if use_ensemble else 'single best'}  RMSE={final_rmse:.4f}")

# ---------------------------------------------------------------------------
# Push sur Skore Hub (rapport du meilleur modele single)
# ---------------------------------------------------------------------------
project.put("10_model_tuning", best_report)
print("Report '10_model_tuning' pousse sur Skore Hub")

# ---------------------------------------------------------------------------
# Fit final + soumission
# ---------------------------------------------------------------------------
Path("submissions").mkdir(exist_ok=True)
submission = X_test[["Index"]].copy()
if use_ensemble:
    submission["target"] = ensemble_test
else:
    final_model = HistGradientBoostingRegressor(random_state=0, **best_cfg)
    final_model.fit(X, y)
    submission["target"] = final_model.predict(X_test[feature_cols])

submission.to_csv("submissions/10_model_tuning.csv", index=False)
print(f"submissions/10_model_tuning.csv ecrit ({len(submission)} lignes)")
print(f"RMSE final retenu : {final_rmse:.4f}")
