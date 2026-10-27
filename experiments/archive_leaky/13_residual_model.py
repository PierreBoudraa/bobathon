"""
Improvement ST6 — Modele residuel avance avec feature set complet (ST1+ST4+encodings).

Deux modeles en cascade :
- Modele A (has_off=True)  : predit (target - off), puis pred_target = off + pred_residual
- Modele B (has_off=False) : predit target directement (pas d'off disponible)

Aussi teste le modele residuel avec ensemble 10 seeds.
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
# Feature engineering complet (ST1 + ST4 + encodings)
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

# ---------------------------------------------------------------------------
# Modele residuel en cascade : 2 modèles séparés selon has_off
# ---------------------------------------------------------------------------
has_off_train = visits["off"].notna().values
has_off_test  = X_test["off"].notna().values

# Residus sur les visites avec off
y_resid_full = (visits["target"] - visits["off"]).fillna(0).values  # 0 quand off manquant

print("Modele residuel OOF :")
oof_resid_target = np.full(len(y), np.nan)

for train_idx, val_idx in cv_splits:
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr         = y.iloc[train_idx].values
    has_off_val  = has_off_train[val_idx]
    off_val      = visits["off"].iloc[val_idx].fillna(0).values

    # Modele A : residuel (entraine sur tout le fold, pas seulement has_off)
    y_resid_tr = y_resid_full[train_idx]
    mA = HistGradientBoostingRegressor(random_state=0, **best_cfg)
    mA.fit(X_tr, y_resid_tr)
    pred_resid = mA.predict(X_val)
    pred_target_A = pred_resid + off_val  # reconstruit target = residuel + off (0 si NaN)

    # Pour les visites sans off : utiliser modele direct sur target
    mB = HistGradientBoostingRegressor(random_state=0, **best_cfg)
    mB.fit(X_tr, y_tr)
    pred_target_B = mB.predict(X_val)

    # Combine : modele A pour has_off, modele B pour no_off
    oof_resid_target[val_idx] = np.where(has_off_val, pred_target_A, pred_target_B)

rmse_resid = float(np.sqrt(np.mean((oof_resid_target - y.values) ** 2)))
print(f"  Residuel cascade OOF RMSE = {rmse_resid:.4f}")

# Comparer avec direct (référence : single seed = 2.3468)
rmse_ref = 2.3468
if rmse_resid < rmse_ref:
    print(f"-> Residuel meilleur ({rmse_resid:.4f} < {rmse_ref:.4f})")
    use_resid = True
else:
    print(f"-> Direct meilleur ({rmse_ref:.4f} < {rmse_resid:.4f})")
    use_resid = False

# ---------------------------------------------------------------------------
# Push rapport (modele residuel evalué via skore)
# ---------------------------------------------------------------------------
y_resid_series = pd.Series(y_resid_full, index=y.index)
hgbr_resid = HistGradientBoostingRegressor(random_state=0, **best_cfg)
report_resid = skore_evaluate(hgbr_resid, X, y_resid_series, splitter=cv_splits)
project.put("13_residual_model", report_resid)
print("Report '13_residual_model' pousse (RMSE du residuel, pas du target reconstitué)")

# ---------------------------------------------------------------------------
# Fit final + soumission du meilleur
# ---------------------------------------------------------------------------
Path("submissions").mkdir(exist_ok=True)
submission = X_test[["Index"]].copy()
off_test = X_test["off"].fillna(0).values

if use_resid:
    # Ensemble 5 seeds residuel
    test_preds_resid = []
    test_preds_direct = []
    for seed in range(5):
        mA = HistGradientBoostingRegressor(random_state=seed, **best_cfg)
        mA.fit(X, y_resid_series)
        test_preds_resid.append(mA.predict(X_test[feature_cols]) + off_test)
        mB = HistGradientBoostingRegressor(random_state=seed, **best_cfg)
        mB.fit(X, y)
        test_preds_direct.append(mB.predict(X_test[feature_cols]))
    pred_resid_test  = np.mean(test_preds_resid,  axis=0)
    pred_direct_test = np.mean(test_preds_direct, axis=0)
    submission["target"] = np.where(has_off_test, pred_resid_test, pred_direct_test)
    out_rmse = rmse_resid
else:
    # Ensemble 10 seeds direct (depuis ST7)
    test_preds = []
    for seed in range(10):
        m = HistGradientBoostingRegressor(random_state=seed, **best_cfg)
        m.fit(X, y)
        test_preds.append(m.predict(X_test[feature_cols]))
    submission["target"] = np.mean(test_preds, axis=0)
    out_rmse = 2.3343  # OOF ensemble 10 seeds de ST7

submission.to_csv("submissions/13_residual_model.csv", index=False)
print(f"\nsubmissions/13_residual_model.csv ecrit ({len(submission)} lignes)")
print(f"RMSE retenu : {out_rmse:.4f}")
print(f"Objectif atteint : {'OUI' if out_rmse <= 2.196 else 'NON'}")
