"""
Modele honnete final — experiments/16_honest_final.py

Features exclusivement construites depuis des colonnes disponibles dans X_test
(off, on, ledd, age, sexM, etc.) : lags, deltas, agrégats historiques patients.
Aucune feature dérivée de `target` (fuite confirmée sur scripts 08-15).

RMSE GroupKFold n=5 : ~4.67
Ensemble 10 seeds   : ~4.65 (attendu)
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
# Feature engineering — uniquement des colonnes présentes dans X_test
# Aucune feature dérivée de `target`
# ---------------------------------------------------------------------------
def build_features(df):
    """
    Construit les features à partir des mesures cliniques disponibles en test.
    Toutes les features de lag/agrégat utilisent off/on/ledd, jamais target.
    """
    g = df.groupby("patient_id", sort=False)

    # Lags, deltas, agrégats pour off, on, ledd
    for col in ["off", "on", "ledd"]:
        df[f"{col}_lag1"] = g[col].shift(1)
        df[f"{col}_lag2"] = g[col].shift(2)
        df[f"{col}_delta"] = df[col] - df[f"{col}_lag1"]
        shifted = g[col].shift(1)
        df[f"{col}_cumean"] = shifted.groupby(df["patient_id"]).expanding().mean().values
        df[f"{col}_cumax"]  = shifted.groupby(df["patient_id"]).expanding().max().values
        df[f"{col}_cumin"]  = shifted.groupby(df["patient_id"]).expanding().min().values
        df[f"{col}_custd"]  = shifted.groupby(df["patient_id"]).expanding().std().values
        df[f"{col}_roll3"]  = shifted.groupby(df["patient_id"]).transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df[f"{col}_roll5"]  = shifted.groupby(df["patient_id"]).transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

    # Ecart entre lag1 et lag2 (acceleration du changement)
    df["off_lag2_delta"] = df["off_lag1"] - df["off_lag2"]
    df["on_lag2_delta"]  = df["on_lag1"]  - df["on_lag2"]

    # Features temporelles de visite
    df["age_delta"]              = g["age"].diff()
    df["visit_number"]           = g.cumcount() + 1
    df["time_since_diagnosis"]   = df["age"] - df["age_at_diagnosis"]
    df["age_first_visit"]        = g["age"].transform("first")
    df["time_since_first_visit"] = df["age"] - df["age_first_visit"]

    # Indicateurs de missingness (signal clinique : off manquant = visite ON-only)
    df["off_missing"]  = df["off"].isna().astype(int)
    df["on_missing"]   = df["on"].isna().astype(int)
    df["ledd_missing"] = df["ledd"].isna().astype(int)
    # Nombre de visites avec off disponible jusqu'ici
    df["off_count"] = (
        g["off_missing"]
        .transform(lambda x: (1 - x).shift(1).expanding().sum())
        .fillna(0)
    )

    # Interactions cliniques
    df["on_off_diff"]    = df["on"]   - df["off"]
    df["ledd_times_off"] = df["ledd"] * df["off"]
    df["ledd_times_on"]  = df["ledd"] * df["on"]

    return df


visits = build_features(visits)
X_test  = build_features(X_test)

# ---------------------------------------------------------------------------
# Feature set final (toutes disponibles en test, aucune dérivée de target)
# ---------------------------------------------------------------------------
EXCLUDE = {"Index", "patient_id", "cohort", "gene", "rater_id",
           "target", "age_first_visit"}

feature_cols = [
    c for c in visits.columns
    if c not in EXCLUDE and not c.startswith("target")
]

X = visits[feature_cols]
y = visits["target"]
print(f"Features : {len(feature_cols)}")
print(f"Colonnes : {feature_cols}")

# ---------------------------------------------------------------------------
# Credentials Skore Hub
# ---------------------------------------------------------------------------
def load_skore_credentials():
    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen, dest = set(), None
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
    os.environ["SKORE_HUB_URI"]     = cfg["hub_url"]
    os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
    return {k: v for k, v in cfg.items() if k != "api_key"}

from skore import evaluate as skore_evaluate, login, Project

cfg     = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

# ---------------------------------------------------------------------------
# Evaluation GroupKFold n=5
# ---------------------------------------------------------------------------
groups    = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

BEST_CFG = dict(
    max_iter=1000,
    learning_rate=0.01,
    max_leaf_nodes=63,
    min_samples_leaf=5,
)

hgbr   = HistGradientBoostingRegressor(random_state=0, **BEST_CFG)
report = skore_evaluate(hgbr, X, y, splitter=cv_splits)
rmse_single = float(report.metrics.rmse().iloc[0].iloc[0])
print(f"\nRMSE GroupKFold n=5 (single seed) : {rmse_single:.4f}")

project.put("16_honest_final", report)
print("Report '16_honest_final' pousse sur Skore Hub")

# ---------------------------------------------------------------------------
# Ensemble 10 seeds (OOF RMSE + predictions test)
# ---------------------------------------------------------------------------
print("\nEnsemble 10 seeds ...")
N_SEEDS = 10
oof_preds  = np.zeros(len(y))
test_preds = []

for seed in range(N_SEEDS):
    fold_oofs = np.full(len(y), np.nan)
    for train_idx, val_idx in cv_splits:
        m = HistGradientBoostingRegressor(random_state=seed, **BEST_CFG)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_oofs[val_idx] = m.predict(X.iloc[val_idx])
    oof_preds += fold_oofs
    m_full = HistGradientBoostingRegressor(random_state=seed, **BEST_CFG)
    m_full.fit(X, y)
    test_preds.append(m_full.predict(X_test[feature_cols]))

oof_preds     /= N_SEEDS
ensemble_test  = np.mean(test_preds, axis=0)
rmse_ensemble  = float(np.sqrt(np.mean((oof_preds - y.values) ** 2)))
print(f"RMSE ensemble {N_SEEDS} seeds : {rmse_ensemble:.4f}")

# ---------------------------------------------------------------------------
# Verification de la distribution des predictions (ne doit pas diverger de train)
# ---------------------------------------------------------------------------
print(f"\nDistribution predictions test :")
print(f"  mean={ensemble_test.mean():.3f}  std={ensemble_test.std():.3f}  "
      f"min={ensemble_test.min():.1f}  max={ensemble_test.max():.1f}")
print(f"Distribution target train :")
print(f"  mean={y.mean():.3f}  std={y.std():.3f}  "
      f"min={y.min():.1f}  max={y.max():.1f}")

# ---------------------------------------------------------------------------
# Soumission Kaggle
# ---------------------------------------------------------------------------
Path("submissions").mkdir(exist_ok=True)
submission = X_test[["Index"]].copy()
submission["target"] = ensemble_test
submission.to_csv("submissions/16_honest_final.csv", index=False)

print(f"\nsubmissions/16_honest_final.csv ecrit ({len(submission)} lignes)")
print(f"\nRMSE single seed : {rmse_single:.4f}")
print(f"RMSE ensemble    : {rmse_ensemble:.4f}")
print(f"\nSoumission Kaggle : submissions/16_honest_final.csv")
print("URL Hub : voir 'Consult your report at' ci-dessus")
