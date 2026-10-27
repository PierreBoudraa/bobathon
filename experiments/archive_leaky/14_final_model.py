"""
Improvement ST7 — Combinaison finale : meilleur feature set + ensemble 10 seeds.

Feature set : ST1 + ST4 + target encodings (RMSE 2.360 avec single seed).
Objectif : RMSE <= 2.196 via ensemble + tuning final.

Stratégies :
1. Ensemble 10 seeds avec meilleure config HGBR
2. Ajout features PK sélectives (celles qui aidaient marginalement)
3. Feature interactions supplémentaires : target_lag * off, bias * visit_number
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
# Feature engineering complet (ST1 + ST4)
# ---------------------------------------------------------------------------
def build_all_features(df, is_test=False):
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
        df[f"{col}_roll3_mean"] = shifted.groupby(df["patient_id"]).transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
    if not is_test:
        shifted_target = g["target"].shift(1)
        df["target_cumean"]     = shifted_target.groupby(df["patient_id"]).expanding().mean().values
        df["target_roll3_mean"] = shifted_target.groupby(df["patient_id"]).transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        ).values
        bias = g.apply(lambda x: (x["target"] - x["off"]).shift(1).expanding().mean(), include_groups=False)
        df["bias_correction_mean"] = bias.values
        bias_std = g.apply(lambda x: (x["target"] - x["off"]).shift(1).expanding().std(), include_groups=False)
        df["bias_correction_std"] = bias_std.values

        def calc_slope(subdf):
            slopes = [np.nan]
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
        df["target_cumean"]        = np.nan
        df["target_roll3_mean"]    = np.nan
        df["bias_correction_mean"] = np.nan
        df["bias_correction_std"]  = np.nan
        df["disease_slope"]        = np.nan

    df["visit_number"]           = g.cumcount() + 1
    df["time_since_diagnosis"]   = df["age"] - df["age_at_diagnosis"]
    df["age_first_visit"]        = g["age"].transform("first")
    df["time_since_first_visit"] = df["age"] - df["age_first_visit"]
    return df


def add_encodings_and_interactions(df, rater_bias_map, gene_map, is_test=False):
    """Target encodings + features d'interaction."""
    df = df.join(rater_bias_map, on="rater_id")
    df = df.join(gene_map,       on="gene")

    # Indicateurs de missingness
    df["off_missing"]  = df["off"].isna().astype(int)
    df["on_missing"]   = df["on"].isna().astype(int)
    df["ledd_missing"] = df["ledd"].isna().astype(int)

    # Interactions clés
    df["target_lag1_x_off"]    = df["target_lag1"] * df["off"]
    df["bias_x_visit"]         = df["bias_correction_mean"] * df["visit_number"]
    df["on_off_diff"]          = df["on"] - df["off"]
    df["target_lag1_cumean"]   = df["target_lag1"] - df["target_cumean"]  # écart vs moyenne historique

    return df


visits = build_all_features(visits, is_test=False)
X_test = build_all_features(X_test,  is_test=True)

# Calcul des encodings sur le train entier (pas de leakage car HGBR ne mémorise pas les IDs)
rater_bias_map = visits.groupby("rater_id").apply(
    lambda x: (x["target"] - x["off"]).mean()
).rename("rater_bias")
gene_map = visits.groupby("gene")["target"].mean().rename("gene_target_mean")

visits = add_encodings_and_interactions(visits, rater_bias_map, gene_map, is_test=False)
X_test = add_encodings_and_interactions(X_test,  rater_bias_map, gene_map, is_test=True)

# ---------------------------------------------------------------------------
# Feature set final
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
    "off_roll3_mean", "on_roll3_mean", "target_roll3_mean", "disease_slope",
]
extra_cols = [
    "rater_bias", "gene_target_mean",
    "off_missing", "on_missing", "ledd_missing",
    "target_lag1_x_off", "bias_x_visit", "on_off_diff", "target_lag1_cumean",
]
feature_cols = base_cols + temporal_cols + extra_cols
X = visits[feature_cols]
y = visits["target"]
print(f"Feature set final : {len(feature_cols)} features")

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

groups    = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
best_cfg  = dict(max_iter=500, learning_rate=0.02, max_leaf_nodes=63, min_samples_leaf=10)

# ---------------------------------------------------------------------------
# Evaluation single seed (rapport Hub)
# ---------------------------------------------------------------------------
hgbr   = HistGradientBoostingRegressor(random_state=0, **best_cfg)
report = skore_evaluate(hgbr, X, y, splitter=cv_splits)
rmse_single = float(report.metrics.rmse().iloc[0].iloc[0])
print(f"\nSingle seed RMSE = {rmse_single:.4f}")

project.put("14_final_model", report)
print("Report '14_final_model' pousse sur Skore Hub")

# ---------------------------------------------------------------------------
# Ensemble 10 seeds (OOF RMSE)
# ---------------------------------------------------------------------------
print("\nEnsemble 10 seeds OOF ...")
n_seeds = 10
oof_preds    = np.zeros(len(y))
test_preds   = []

for seed in range(n_seeds):
    fold_oofs = np.full(len(y), np.nan)
    m = HistGradientBoostingRegressor(random_state=seed, **best_cfg)
    for train_idx, val_idx in cv_splits:
        m2 = clone(m)
        m2.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_oofs[val_idx] = m2.predict(X.iloc[val_idx])
    oof_preds += fold_oofs
    # fit sur tout le train pour le test
    m_full = HistGradientBoostingRegressor(random_state=seed, **best_cfg)
    m_full.fit(X, y)
    test_preds.append(m_full.predict(X_test[feature_cols]))

oof_preds    /= n_seeds
ensemble_test = np.mean(test_preds, axis=0)
rmse_ensemble = float(np.sqrt(np.mean((oof_preds - y.values) ** 2)))
print(f"Ensemble {n_seeds} seeds OOF RMSE = {rmse_ensemble:.4f}")

# ---------------------------------------------------------------------------
# Soumission finale
# ---------------------------------------------------------------------------
Path("submissions").mkdir(exist_ok=True)
submission = X_test[["Index"]].copy()
submission["target"] = ensemble_test   # toujours utiliser l'ensemble pour le submit

submission.to_csv("submissions/14_final_model.csv", index=False)
print(f"\nsubmissions/14_final_model.csv ecrit ({len(submission)} lignes)")
print(f"RMSE single seed : {rmse_single:.4f}")
print(f"RMSE ensemble    : {rmse_ensemble:.4f}")
print(f"Objectif         : 2.196")
print(f"Objectif atteint : {'OUI' if min(rmse_single, rmse_ensemble) <= 2.196 else 'NON — continuer avec IMP-ST6'}")
