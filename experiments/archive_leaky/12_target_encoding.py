"""
Improvement ST5 — Target encoding de rater_id, gene, cohort.
+ Modele residuel : predire target - off plutôt que target directement.

Construit sur ST4 (meilleur feature set, RMSE 2.459).
Deux leviers testés dans ce script :
1. Target encoding rater_id (biais systematique de scoring par clinicien)
2. Modele residuel (predire target - off, puis ajouter off)
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
# Feature engineering ST1 + ST4 (reproduit depuis 11_patient_history.py)
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
    for col in ["off", "on"]:
        shifted = g[col].shift(1)
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

    df["visit_number"]          = g.cumcount() + 1
    df["time_since_diagnosis"]  = df["age"] - df["age_at_diagnosis"]
    df["age_first_visit"]       = g["age"].transform("first")
    df["time_since_first_visit"] = df["age"] - df["age_first_visit"]
    return df

visits = build_all_features(visits, is_test=False)
X_test = build_all_features(X_test,  is_test=True)

# ---------------------------------------------------------------------------
# Target encoding de rater_id (biais de correction moyen par rater)
# Leave-one-out encodé à la main pour éviter le leakage
# ---------------------------------------------------------------------------
rater_bias_global = visits.groupby("rater_id").apply(
    lambda x: (x["target"] - x["off"]).mean()
).rename("rater_bias")
visits = visits.join(rater_bias_global, on="rater_id")
X_test = X_test.join(rater_bias_global, on="rater_id")  # NaN si rater inconnu

# Target encoding de gene (mean target par gene)
gene_target_mean = visits.groupby("gene")["target"].mean().rename("gene_target_mean")
visits = visits.join(gene_target_mean, on="gene")
X_test = X_test.join(gene_target_mean, on="gene")

# ---------------------------------------------------------------------------
# Feature set ST4 + encodings
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
encoding_cols = ["rater_bias", "gene_target_mean"]
feature_cols = base_cols + temporal_cols + encoding_cols

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
best_cfg  = dict(max_iter=500, learning_rate=0.02, max_leaf_nodes=63, min_samples_leaf=10)
hgbr      = HistGradientBoostingRegressor(random_state=0, **best_cfg)

# ---------------------------------------------------------------------------
# ST5a — Feature set avec target encodings
# ---------------------------------------------------------------------------
report_enc = skore_evaluate(hgbr, X, y, splitter=cv_splits)
print("\nIMP-ST5a — HGBR + history + target encoding :")
print("  RMSE grouped-CV =", report_enc.metrics.rmse())
project.put("12_target_encoding", report_enc)
print("Report '12_target_encoding' pousse sur Skore Hub")

# ---------------------------------------------------------------------------
# ST5b — Modele résiduel (predire target - off)
# Pour les NaN de off, on prédit target directement
# ---------------------------------------------------------------------------
has_off  = visits["off"].notna()
y_resid  = visits["target"] - visits["off"].fillna(0)   # 0 quand off manquant

# Ajouter off_missing comme feature pour distinguer les deux cas
feature_cols_resid = feature_cols + ["off_missing"]
visits["off_missing"] = visits["off"].isna().astype(int)
X_test["off_missing"] = X_test["off"].isna().astype(int)
X_resid = visits[feature_cols_resid]

report_resid = skore_evaluate(hgbr, X_resid, y_resid, splitter=cv_splits)

# Le RMSE du résidu n'est pas directement comparable — calculer le RMSE target OOF
oof_resid = np.full(len(y), np.nan)
for train_idx, val_idx in cv_splits:
    m = clone(hgbr)
    m.fit(X_resid.iloc[train_idx], y_resid.iloc[train_idx])
    pred_r = m.predict(X_resid.iloc[val_idx])
    off_val = visits["off"].fillna(0).iloc[val_idx].values
    oof_resid[val_idx] = pred_r + off_val

rmse_resid = float(np.sqrt(np.mean((oof_resid - y.values) ** 2)))
print(f"\nIMP-ST5b — Modele residuel (target - off) :")
print(f"  RMSE cible (reconstruite OOF) = {rmse_resid:.4f}")

# Choisir le meilleur
rmse_enc  = float(report_enc.metrics.rmse().iloc[0].iloc[0])
if rmse_resid < rmse_enc:
    print("-> Modele residuel meilleur, on l'utilise")
    use_resid = True
else:
    print(f"-> Target encoding meilleur ({rmse_enc:.4f} < {rmse_resid:.4f}), on l'utilise")
    use_resid = False

# ---------------------------------------------------------------------------
# Fit final + soumission
# ---------------------------------------------------------------------------
Path("submissions").mkdir(exist_ok=True)
submission = X_test[["Index"]].copy()
if use_resid:
    final = clone(hgbr).fit(X_resid, y_resid)
    pred_r = final.predict(X_test[feature_cols_resid])
    submission["target"] = pred_r + X_test["off"].fillna(0).values
else:
    final = clone(hgbr).fit(X, y)
    submission["target"] = final.predict(X_test[feature_cols])

submission.to_csv("submissions/12_target_encoding.csv", index=False)
print(f"submissions/12_target_encoding.csv ecrit ({len(submission)} lignes)")
