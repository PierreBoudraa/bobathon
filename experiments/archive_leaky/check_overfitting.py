"""
Verification d'overfitting — 5 tests independants sur le modele final (15_push_final).

Tests :
1. RMSE train vs RMSE CV (ratio train/val)
2. Stabilite inter-folds (ecart-type des RMSE par fold)
3. GroupKFold n=10 vs n=5 (generalisation sur plus de folds)
4. Temporal holdout : train sur les premieres visites, valider sur les dernieres
5. Patient holdout manuel : 20 % des patients ecartés avant toute ingénierie de features
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# ---------------------------------------------------------------------------
# Chargement + feature engineering (copie exacte de 15_push_final)
# ---------------------------------------------------------------------------
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

visits = X_train.merge(y_train, on="Index").sort_values(["patient_id", "age"]).reset_index(drop=True)


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
                y_v = subdf["target"].iloc[:i].values
                if len(x) >= 2 and np.std(x) > 0:
                    slope, *_ = scipy_stats.linregress(x, y_v)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
            return pd.Series(slopes, index=subdf.index)

        df["disease_slope"] = g.apply(calc_slope, include_groups=False).values
    else:
        for c in ["target_cumean", "target_roll3_mean", "bias_correction_mean",
                  "bias_correction_std", "disease_slope"]:
            df[c] = np.nan

    df["visit_number"]           = g.cumcount() + 1
    df["time_since_diagnosis"]   = df["age"] - df["age_at_diagnosis"]
    df["age_first_visit"]        = g["age"].transform("first")
    df["time_since_first_visit"] = df["age"] - df["age_first_visit"]
    return df


visits = build_all_features(visits, is_test=False)

rater_bias_map = visits.groupby("rater_id").apply(
    lambda x: (x["target"] - x["off"]).mean()
).rename("rater_bias")
gene_map = visits.groupby("gene")["target"].mean().rename("gene_target_mean")

visits["rater_bias"]       = visits["rater_id"].map(rater_bias_map)
visits["gene_target_mean"] = visits["gene"].map(gene_map)
visits["off_missing"]      = visits["off"].isna().astype(int)
visits["on_missing"]       = visits["on"].isna().astype(int)
visits["ledd_missing"]     = visits["ledd"].isna().astype(int)
visits["target_lag1_x_off"]       = visits["target_lag1"] * visits["off"]
visits["bias_x_visit"]            = visits["bias_correction_mean"] * visits["visit_number"]
visits["on_off_diff"]             = visits["on"] - visits["off"]
visits["target_lag1_cumean"]      = visits["target_lag1"] - visits["target_cumean"]
visits["target_lag1_filled"]      = visits["target_lag1"].fillna(visits["off"]).fillna(visits["on"])
visits["target_lag1_vs_off_cumean"] = visits["target_lag1_filled"] - visits["off_cumean"]
visits["off_count"] = (
    visits.groupby("patient_id")["off_missing"]
    .transform(lambda x: (1 - x).shift(1).expanding().sum()).fillna(0)
)
visits["target_lag1_lag2_diff"] = visits["target_lag1"] - visits["target_lag2"]
visits["rater_bias_x_off"]      = visits["rater_bias"] * visits["off"]

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
groups = visits["patient_id"]

BEST_CFG = dict(max_iter=1000, learning_rate=0.01, max_leaf_nodes=63, min_samples_leaf=10)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

sep = "=" * 60
print(sep)
print("VERIFICATION D'OVERFITTING — 5 TESTS INDEPENDANTS")
print(sep)

# ---------------------------------------------------------------------------
# TEST 1 : RMSE train vs RMSE CV (ratio)
# Si ratio train_rmse/cv_rmse << 1 => overfit
# ---------------------------------------------------------------------------
print("\n[TEST 1] RMSE train vs RMSE validation (5-fold grouped CV)")
cv5 = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
oof = np.full(len(y), np.nan)
train_rmses = []
for fold_i, (train_idx, val_idx) in enumerate(cv5):
    m = HistGradientBoostingRegressor(random_state=0, **BEST_CFG)
    m.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof[val_idx] = m.predict(X.iloc[val_idx])
    train_rmse_fold = rmse(y.iloc[train_idx], m.predict(X.iloc[train_idx]))
    val_rmse_fold   = rmse(y.iloc[val_idx],   m.predict(X.iloc[val_idx]))
    train_rmses.append(train_rmse_fold)
    print(f"  Fold {fold_i+1}: train_RMSE={train_rmse_fold:.4f}  val_RMSE={val_rmse_fold:.4f}  ratio={train_rmse_fold/val_rmse_fold:.3f}")

cv_rmse_global = rmse(y, oof)
mean_train_rmse = np.mean(train_rmses)
ratio = mean_train_rmse / cv_rmse_global
print(f"  --> train moyen={mean_train_rmse:.4f}  CV global={cv_rmse_global:.4f}  ratio={ratio:.3f}")
if ratio > 0.95:
    print("  VERDICT: OK — pas de sur-apprentissage significatif (ratio proche de 1)")
elif ratio > 0.80:
    print("  VERDICT: ATTENTION — léger overfit possible (ratio entre 0.80 et 0.95)")
else:
    print("  VERDICT: OVERFIT PROBABLE — ratio train/val < 0.80")

# ---------------------------------------------------------------------------
# TEST 2 : Stabilité inter-folds (écart-type des RMSE par fold)
# Un std élevé indique que certains folds sont très faciles/difficiles -> sur-spécialisation
# ---------------------------------------------------------------------------
print(f"\n[TEST 2] Stabilite inter-folds (GroupKFold n=5)")
fold_val_rmses = []
for train_idx, val_idx in cv5:
    m = HistGradientBoostingRegressor(random_state=0, **BEST_CFG)
    m.fit(X.iloc[train_idx], y.iloc[train_idx])
    fold_val_rmses.append(rmse(y.iloc[val_idx], m.predict(X.iloc[val_idx])))

print(f"  RMSE par fold : {[f'{r:.4f}' for r in fold_val_rmses]}")
print(f"  Moyenne : {np.mean(fold_val_rmses):.4f}  Std : {np.std(fold_val_rmses):.4f}  CV% : {100*np.std(fold_val_rmses)/np.mean(fold_val_rmses):.1f}%")
if np.std(fold_val_rmses) / np.mean(fold_val_rmses) < 0.05:
    print("  VERDICT: OK — folds stables (CV% < 5%)")
elif np.std(fold_val_rmses) / np.mean(fold_val_rmses) < 0.10:
    print("  VERDICT: ATTENTION — variabilite moderee entre folds (CV% 5-10%)")
else:
    print("  VERDICT: INSTABLE — grande variabilite entre folds (CV% > 10%)")

# ---------------------------------------------------------------------------
# TEST 3 : GroupKFold n=10 vs n=5 (plus de folds = estimation plus robuste)
# Si RMSE 10-fold >> RMSE 5-fold => le modele était optimiste sur 5 folds
# ---------------------------------------------------------------------------
print(f"\n[TEST 3] GroupKFold n=10 vs n=5")
cv10 = list(GroupKFold(n_splits=10).split(X, y, groups=groups))
oof10 = np.full(len(y), np.nan)
for train_idx, val_idx in cv10:
    m = HistGradientBoostingRegressor(random_state=0, **BEST_CFG)
    m.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof10[val_idx] = m.predict(X.iloc[val_idx])

cv10_rmse = rmse(y, oof10)
diff = cv10_rmse - cv_rmse_global
print(f"  RMSE 5-fold  : {cv_rmse_global:.4f}")
print(f"  RMSE 10-fold : {cv10_rmse:.4f}  (ecart = {diff:+.4f})")
if abs(diff) < 0.05:
    print("  VERDICT: OK — RMSE stable entre 5 et 10 folds")
elif diff < 0.15:
    print("  VERDICT: ATTENTION — legere hausse avec plus de folds")
else:
    print("  VERDICT: OVERFIT POSSIBLE — RMSE monte avec plus de folds")

# ---------------------------------------------------------------------------
# TEST 4 : Temporal holdout — train sur visites 1..N-2, val sur visite N-1 et N
# Simule la prediction de futures visites non vues
# ---------------------------------------------------------------------------
print(f"\n[TEST 4] Temporal holdout (dernieres 2 visites par patient en validation)")
visits_sorted = visits.sort_values(["patient_id", "age"]).copy()
visits_sorted["visit_rank"]    = visits_sorted.groupby("patient_id").cumcount() + 1
visits_sorted["n_visits"]      = visits_sorted.groupby("patient_id")["patient_id"].transform("count")
visits_sorted["is_last2"]      = visits_sorted["visit_rank"] >= visits_sorted["n_visits"] - 1

train_mask = ~visits_sorted["is_last2"]
val_mask   = visits_sorted["is_last2"]

X_tr_temp  = X.loc[visits_sorted.index[train_mask]]
y_tr_temp  = y.loc[visits_sorted.index[train_mask]]
X_val_temp = X.loc[visits_sorted.index[val_mask]]
y_val_temp = y.loc[visits_sorted.index[val_mask]]

m_temp = HistGradientBoostingRegressor(random_state=0, **BEST_CFG)
m_temp.fit(X_tr_temp, y_tr_temp)
rmse_temp_train = rmse(y_tr_temp, m_temp.predict(X_tr_temp))
rmse_temp_val   = rmse(y_val_temp, m_temp.predict(X_val_temp))

print(f"  Train visits (visites 1..N-2) : {train_mask.sum()} rows, RMSE={rmse_temp_train:.4f}")
print(f"  Val   visits (dernieres 2)    : {val_mask.sum()} rows, RMSE={rmse_temp_val:.4f}")
print(f"  Ecart val-train : {rmse_temp_val - rmse_temp_train:+.4f}")
if rmse_temp_val < cv_rmse_global * 1.20:
    print("  VERDICT: OK — RMSE sur futures visites proche du CV groupé")
elif rmse_temp_val < cv_rmse_global * 1.50:
    print("  VERDICT: ATTENTION — RMSE temporal > 20% au-dessus du CV groupé")
else:
    print("  VERDICT: OVERFIT PROBABLE — RMSE temporal >> CV groupé")

# ---------------------------------------------------------------------------
# TEST 5 : Patient holdout manuel (20% patients ecartés avant ingénierie)
# Simule exactement le split Kaggle : patients entiers hors train
# Note : les lags des patients val sont NaN (premiere visite simulée)
# ---------------------------------------------------------------------------
print(f"\n[TEST 5] Patient holdout manuel 80/20 (patients entiers)")
rng = np.random.default_rng(42)
all_patients = visits["patient_id"].unique()
n_val_patients = int(0.20 * len(all_patients))
val_patients = set(rng.choice(all_patients, size=n_val_patients, replace=False))
train_patients = set(all_patients) - val_patients

mask_train = visits["patient_id"].isin(train_patients)
mask_val   = visits["patient_id"].isin(val_patients)

X_tr_ph, y_tr_ph = X[mask_train], y[mask_train]
X_val_ph, y_val_ph = X[mask_val], y[mask_val]

m_ph = HistGradientBoostingRegressor(random_state=0, **BEST_CFG)
m_ph.fit(X_tr_ph, y_tr_ph)
rmse_ph_train = rmse(y_tr_ph, m_ph.predict(X_tr_ph))
rmse_ph_val   = rmse(y_val_ph, m_ph.predict(X_val_ph))

print(f"  Train : {mask_train.sum()} visites / {len(train_patients)} patients, RMSE={rmse_ph_train:.4f}")
print(f"  Val   : {mask_val.sum()} visites / {len(val_patients)} patients, RMSE={rmse_ph_val:.4f}")
print(f"  Ecart val-train : {rmse_ph_val - rmse_ph_train:+.4f}")
if rmse_ph_val < cv_rmse_global * 1.20:
    print("  VERDICT: OK — RMSE patient holdout cohérent avec le CV groupé")
elif rmse_ph_val < cv_rmse_global * 1.50:
    print("  VERDICT: ATTENTION — RMSE > 20% au-dessus du CV groupé")
else:
    print("  VERDICT: OVERFIT PROBABLE — RMSE >> CV groupé")

# ---------------------------------------------------------------------------
# SYNTHESE
# ---------------------------------------------------------------------------
print(f"\n{sep}")
print("SYNTHESE")
print(sep)
print(f"  RMSE GroupKFold n=5   : {cv_rmse_global:.4f}")
print(f"  RMSE GroupKFold n=10  : {cv10_rmse:.4f}")
print(f"  RMSE temporal holdout : {rmse_temp_val:.4f}")
print(f"  RMSE patient holdout  : {rmse_ph_val:.4f}")
print(f"  Ratio train/val (T1)  : {ratio:.3f}")
print(f"  Std inter-folds (T2)  : {np.std(fold_val_rmses):.4f} ({100*np.std(fold_val_rmses)/np.mean(fold_val_rmses):.1f}%)")
print(sep)
