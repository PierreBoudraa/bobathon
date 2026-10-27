"""
Improvement ST8 — Push final vers RMSE <= 2.196.

Nouvelles idees sur le feature set complet (ST1+ST4+encodings) :
1. target_lag interpolé : pour les visites ou target_lag1 est NaN (1ere visite),
   utiliser off comme proxy (r=0.87 avec target)
2. Ratio target_lag / off_cumean : capture si le patient evolue par rapport a sa moyenne
3. Nombre de visites anterieures avec off disponible
4. Tuning HGBR : learning_rate=0.01, max_iter=1000 (plus lent mais potentiellement meilleur)
5. Ensemble 20 seeds
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
# Feature engineering complet
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


visits = build_all_features(visits, is_test=False)
X_test = build_all_features(X_test,  is_test=True)

# Encodings
rater_bias_map = visits.groupby("rater_id").apply(
    lambda x: (x["target"] - x["off"]).mean()
).rename("rater_bias")
gene_map = visits.groupby("gene")["target"].mean().rename("gene_target_mean")

for df in [visits, X_test]:
    df["rater_bias"]       = df["rater_id"].map(rater_bias_map)
    df["gene_target_mean"] = df["gene"].map(gene_map)
    df["off_missing"]      = df["off"].isna().astype(int)
    df["on_missing"]       = df["on"].isna().astype(int)
    df["ledd_missing"]     = df["ledd"].isna().astype(int)
    df["target_lag1_x_off"]  = df["target_lag1"] * df["off"]
    df["bias_x_visit"]       = df["bias_correction_mean"] * df["visit_number"]
    df["on_off_diff"]        = df["on"] - df["off"]
    df["target_lag1_cumean"] = df["target_lag1"] - df["target_cumean"]

    # NOUVELLES FEATURES
    # target_lag1 impute par off quand NaN (premiere visite)
    df["target_lag1_filled"] = df["target_lag1"].fillna(df["off"])
    # target_lag1 impute par on quand off aussi NaN
    df["target_lag1_filled"] = df["target_lag1_filled"].fillna(df["on"])

    # Ratio : ecart target_lag1 vs moyenne historique off
    df["target_lag1_vs_off_cumean"] = df["target_lag1_filled"] - df["off_cumean"]

    # Nombre de visites avec off disponible jusqu'ici
    df["off_count"] = (
        df.groupby("patient_id")["off_missing"]
        .transform(lambda x: (1 - x).shift(1).expanding().sum())
        .fillna(0)
    )

    # Ecart entre les deux derniers targets connus
    df["target_lag1_lag2_diff"] = df["target_lag1"] - df["target_lag2"]

    # rater_bias x off : interaction biais rater avec mesure
    df["rater_bias_x_off"] = df["rater_bias"] * df["off"]


# ---------------------------------------------------------------------------
# Feature set étendu
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
new_cols = [
    "target_lag1_filled", "target_lag1_vs_off_cumean",
    "off_count", "target_lag1_lag2_diff", "rater_bias_x_off",
]
feature_cols = base_cols + temporal_cols + extra_cols + new_cols
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

groups    = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# ---------------------------------------------------------------------------
# Tester deux configs HGBR
# ---------------------------------------------------------------------------
configs_to_test = [
    dict(max_iter=500,  learning_rate=0.02, max_leaf_nodes=63, min_samples_leaf=10),  # meilleure connue
    dict(max_iter=1000, learning_rate=0.01, max_leaf_nodes=63, min_samples_leaf=10),  # plus lent, meilleur?
    dict(max_iter=500,  learning_rate=0.02, max_leaf_nodes=63, min_samples_leaf=5),   # feuilles plus fines
]

best_rmse, best_cfg_final, best_report = float("inf"), None, None
for cfg_params in configs_to_test:
    hgbr = HistGradientBoostingRegressor(random_state=0, **cfg_params)
    r = skore_evaluate(hgbr, X, y, splitter=cv_splits)
    rmse_val = float(r.metrics.rmse().iloc[0].iloc[0])
    print(f"  {cfg_params} -> RMSE={rmse_val:.4f}")
    if rmse_val < best_rmse:
        best_rmse, best_cfg_final, best_report = rmse_val, cfg_params, r

print(f"\nMeilleure config : {best_cfg_final}  RMSE={best_rmse:.4f}")
project.put("15_push_final", best_report)
print("Report '15_push_final' pousse sur Skore Hub")

# ---------------------------------------------------------------------------
# Ensemble 20 seeds avec la meilleure config
# ---------------------------------------------------------------------------
print("\nEnsemble 20 seeds OOF ...")
n_seeds = 20
oof_preds  = np.zeros(len(y))
test_preds = []

for seed in range(n_seeds):
    fold_oofs = np.full(len(y), np.nan)
    for train_idx, val_idx in cv_splits:
        m = HistGradientBoostingRegressor(random_state=seed, **best_cfg_final)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_oofs[val_idx] = m.predict(X.iloc[val_idx])
    oof_preds += fold_oofs
    m_full = HistGradientBoostingRegressor(random_state=seed, **best_cfg_final)
    m_full.fit(X, y)
    test_preds.append(m_full.predict(X_test[feature_cols]))

oof_preds      /= n_seeds
ensemble_test   = np.mean(test_preds, axis=0)
rmse_ensemble   = float(np.sqrt(np.mean((oof_preds - y.values) ** 2)))
print(f"Ensemble {n_seeds} seeds OOF RMSE = {rmse_ensemble:.4f}")

final_rmse = min(best_rmse, rmse_ensemble)
print(f"\nRMSE single   : {best_rmse:.4f}")
print(f"RMSE ensemble : {rmse_ensemble:.4f}")
print(f"Objectif      : 2.196")
print(f"Objectif atteint : {'OUI' if final_rmse <= 2.196 else 'NON'}")

# ---------------------------------------------------------------------------
# Soumission finale
# ---------------------------------------------------------------------------
Path("submissions").mkdir(exist_ok=True)
submission = X_test[["Index"]].copy()
submission["target"] = ensemble_test
submission.to_csv("submissions/15_push_final.csv", index=False)
print(f"submissions/15_push_final.csv ecrit ({len(submission)} lignes)")
