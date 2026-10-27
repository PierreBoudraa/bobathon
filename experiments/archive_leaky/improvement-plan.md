> **Archived plan (written in French).** It led to experiments 08 to 15, which were later found to use target-based features (data leakage), so the RMSE targets below are not valid. See section 4 of the project README.md.

# Plan d'amélioration du score RMSE — Objectif ≤ 2.196

## Contexte et état des lieux

**Baseline Day 2 :** HGBR sur 8 features numériques → RMSE grouped-CV **7.75**
**Objectif :** RMSE ≤ 2.196 (réduction de 72 %)
**Évaluation de référence :** GroupKFold(n_splits=5) sur patient_id — à utiliser pour toutes les étapes

### Pourquoi le gap est si grand

Le CONTEXT.md identifie trois problèmes non adressés :

1. **Progression temporelle** : `off` corrèle à 0.87 avec `target`, mais c'est *la valeur de la visite courante*. La trajectoire du patient dans le temps (comment son score évolue d'une visite à l'autre) n'est pas du tout exploitée. Chaque visite est traitée comme indépendante des précédentes.

2. **Drug timing / pharmacocinétique** : `time_since_intake_on` et `time_since_intake_off` ont une corrélation quasi nulle avec target brut (r ≈ 0), mais leur interaction avec `ledd` et `off` devrait encoder la courbe PK de la lévodopa (absorption rapide, déclin exponentiel). Le signal est dans la *combinaison*, pas dans la variable seule.

3. **Missingness structurelle** : OFF manquant = examen skipped = patient en état ON uniquement. C'est un signal clinique fort sur l'état du patient que le modèle perçoit comme NaN mais ne lit pas comme indicateur binaire explicite.

### Colonnes disponibles

Train/Test : `Index`, `patient_id`, `cohort`, `sexM`, `gene`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, `rater_id`, `on`, `off`
Target (train seulement) : `target`
Ordre des visites : `age` est croissant par patient (proxy du temps)

---

## Sous-tâche 1 — Feature engineering : historique patient et progression temporelle

**Status : [ ] pending**

### Intent

Construire des features qui capturent la *trajectoire* de chaque patient dans le temps.
Le modèle actuel voit chaque visite isolément ; ici on lui donne accès aux visites précédentes
du même patient. C'est le levier le plus fort identifié par CONTEXT.md (progression temporelle).

### Expected Outcomes

- Script `experiments/archive_leaky/08_temporal_features.py`
- Rapport `08_temporal_features` sur Skore Hub
- RMSE significativement < 7.75 (attendu : 4–6 selon la richesse des features)
- Fichier `submissions/08_temporal_features.csv`

### Todo List

1. Créer `experiments/archive_leaky/08_temporal_features.py`
2. Charger `X_train`, `y_train`, `X_test`, construire `visits` avec merge
3. Trier par `patient_id` puis `age` (age = proxy du temps de visite)
4. **Features de lag par patient** (groupby patient_id, shift) :
   - `off_lag1` : valeur `off` de la visite précédente
   - `on_lag1` : valeur `on` de la visite précédente
   - `target_lag1` : valeur `target` de la visite précédente (train only — voir note)
   - `ledd_lag1` : ledd de la visite précédente
5. **Features de tendance** :
   - `off_delta` : `off` - `off_lag1` (variation du score d'une visite à l'autre)
   - `on_delta` : `on` - `on_lag1`
   - `age_delta` : `age` - `age_lag1` (intervalle entre visites)
   - `ledd_delta` : `ledd` - `ledd_lag1` (changement de dose)
6. **Features d'agrégat historique par patient** (expanding mean sur les visites précédentes) :
   - `off_cumean` : moyenne glissante de `off` jusqu'à la visite courante (exclue)
   - `on_cumean` : idem pour `on`
   - `target_cumean` : moyenne glissante de `target` (train only)
   - `off_cumax` : max historique de `off`
   - `off_cumin` : min historique de `off`
7. **Numéro de visite** : `visit_number` = rang de la visite dans la séquence patient (1, 2, 3…)
8. **Temps depuis le diagnostic** : `time_since_diagnosis` = `age` - `age_at_diagnosis`
9. **Note sur target_lag** : pour le train, `target_lag1` est disponible. Pour le test, il faut la
   prédire ou l'imputer — stratégie : pour la première visite d'un patient test, utiliser `off` comme
   proxy ; pour les suivantes, utiliser la prédiction précédente (pas nécessaire si on utilise
   HGBR qui gère les NaN)
10. Construire `X_full_temporal` = features originales + toutes les nouvelles features
11. Évaluer avec `skore.evaluate(hgbr, X_full_temporal, y, splitter=cv_splits)` (GroupKFold)
12. Pusher `08_temporal_features` sur Skore Hub
13. Fit final, prédire `X_test` (recalculer les lag features sur le test avec les dernières visites train connues), écrire `submissions/08_temporal_features.csv`

### Relevant Context

- `age` est croissant par patient → trier par `(patient_id, age)` avant les shifts
- Pour le test : les lags doivent être calculés en chaînant train + test (les dernières visites train
  servent de contexte pour les premières visites test du même patient — mais les patients sont
  disjoints, donc les lags test seront NaN pour la 1ère visite de chaque patient test)
- `target_lag1` est disponible en train mais pas en test → HGBR gère le NaN nativement
- Garder `GroupKFold` : le CV groupé est critique ici car les lags croisent les folds si on n'y prend pas garde (calculer les lags AVANT le split, pas dans un pipeline)

---

## Sous-tâche 2 — Feature engineering : pharmacocinétique de la lévodopa

**Status : [ ] pending**

### Intent

Modéliser explicitement l'effet pharmacocinétique (PK) de la lévodopa pour débiaiser les scores.
Selon CONTEXT.md, "levodopa effect is proportional to blood concentration: fast absorption, then
exponential decline". Le timing de l'évaluation par rapport à la prise du médicament encode
directement le niveau de lévodopa au moment de la mesure.

### Expected Outcomes

- Script `experiments/archive_leaky/09_pk_features.py`
- Rapport `09_pk_features` sur Skore Hub
- RMSE < résultat ST1 (attendu : 3–5)
- Fichier `submissions/09_pk_features.csv`

### Todo List

1. Créer `experiments/archive_leaky/09_pk_features.py`
2. Repartir de `X_full_temporal` de ST1 (réutiliser le même code de feature engineering)
3. **Modèle PK simplifié** (courbe biexponentielle lévodopa) :
   - `levo_conc_on` = `ledd * exp(-k * time_since_intake_on)` avec k ≈ 0.3 (demi-vie ~2h)
   - `levo_conc_off` = `ledd * exp(-k * time_since_intake_off)` idem
   - Ces features sont NaN quand le délai est manquant → HGBR gère
4. **Correction pharmacodynamique du score ON** :
   - `on_pk_corrected` = `on / (1 + levo_conc_on)` — score ON normalisé par le niveau de médicament
   - `off_pk_corrected` = `off / (1 + levo_conc_off)` idem pour OFF
5. **Indicateurs binaires de missingness** (signal structurel) :
   - `off_missing` = 1 si `off` est NaN, 0 sinon (visite ON-only → patient en état ON)
   - `on_missing` = 1 si `on` est NaN
   - `timing_off_missing` = 1 si `time_since_intake_off` est NaN
   - `ledd_missing` = 1 si `ledd` est NaN
6. **Interaction ledd × scores** :
   - `ledd_times_off` = `ledd * off` (dose × score)
   - `ledd_times_on` = `ledd * on`
   - `on_off_ratio` = `on / (off + 1)` (ratio ON/OFF comme proxy de la fenêtre thérapeutique)
   - `on_off_diff` = `on - off`
7. Combiner toutes les features (ST1 + PK) dans `X_full_pk`
8. Évaluer avec GroupKFold, pusher `09_pk_features`
9. Fit final + `submissions/09_pk_features.csv`

### Relevant Context

- k ≈ 0.3 h⁻¹ correspond à une demi-vie de ~2.3h, cohérente avec la lévodopa standard
- Les features PK sont NaN quand `time_since_intake_*` est manquant (78 % pour off) → important
  que HGBR les traite comme signal, pas comme bruit
- `on_off_diff` est probablement très prédictif : la différence mesurée ON/OFF encode le degré de
  fluctuation qui est lié au niveau de débiaisage nécessaire

---

## Sous-tâche 3 — Tuning HGBR et exploration d'autres estimateurs

**Status : [ ] pending**

### Intent

Avec le meilleur feature set trouvé (ST1 + ST2), optimiser l'estimateur lui-même.
Tester HGBR avec hyperparamètres ajustés, puis tester d'autres estimateurs plus puissants
(ExtraTreesRegressor, RandomForest, et une combinaison en stacking/voting).

### Expected Outcomes

- Script `experiments/archive_leaky/10_model_tuning.py`
- Rapport `10_model_tuning` sur Skore Hub
- RMSE < meilleur résultat ST1/ST2
- Fichier `submissions/10_model_tuning.csv`

### Todo List

1. Créer `experiments/archive_leaky/10_model_tuning.py`
2. Repartir du meilleur feature set des ST1+ST2
3. **Tuning HGBR** (grid manuel sur les paramètres les plus impactants) :
   - `max_iter` : [100, 300, 500] (plus d'arbres = meilleur fit)
   - `max_leaf_nodes` : [31, 63, 127] (profondeur des arbres)
   - `learning_rate` : [0.05, 0.1, 0.2]
   - `min_samples_leaf` : [10, 20, 50] (régularisation)
   - Utiliser `skore.evaluate` avec `splitter=cv_splits` pour chaque combinaison candidate
   - Garder la combinaison avec le RMSE moyen le plus bas
4. **ExtraTreesRegressor** (plus rapide que RF, souvent comparable) :
   - `ExtraTreesRegressor(n_estimators=500, random_state=0)`
   - Évaluer avec GroupKFold
5. **Stacking** : combiner HGBR tuné + Ridge comme meta-learner :
   - `StackingRegressor(estimators=[("hgbr", hgbr_tuned), ("etr", etr)], final_estimator=Ridge())`
   - Évaluer avec GroupKFold
6. Comparer tous les rapports, pusher le meilleur sous `10_model_tuning`
7. Fit final + `submissions/10_model_tuning.csv`

### Relevant Context

- `HistGradientBoostingRegressor` : `max_iter=300`, `max_leaf_nodes=63`, `learning_rate=0.05`
  est souvent un bon point de départ pour des datasets de ~50k lignes
- Éviter le tuning par grid search exhaustif avec CV → trop lent ; préférer 4–5 combinaisons
  manuelles ciblées basées sur la connaissance du dataset

---

## Sous-tâche 4 — Features d'agrégats patient cross-visites avancées

**Status : [ ] pending**

### Intent

Construire des features qui résument toute l'histoire connue du patient, pas seulement la visite
précédente. L'idée clé : le "vrai OFF" d'un patient à une visite dépend de sa trajectoire globale
de dégénérescence, que les visites précédentes révèlent progressivement.

### Expected Outcomes

- Script `experiments/archive_leaky/11_patient_history.py`
- Rapport `11_patient_history` sur Skore Hub
- RMSE ≤ 3.0 (ou meilleur que ST3)
- Fichier `submissions/11_patient_history.csv`

### Todo List

1. Créer `experiments/archive_leaky/11_patient_history.py`
2. Repartir du meilleur feature set précédent
3. **Agrégats statistiques sur toute l'histoire du patient** (groupby patient_id, expanding) :
   - `off_patient_mean` : moyenne de tous les `off` du patient jusqu'à la visite courante
   - `off_patient_std` : écart-type de `off` (mesure de la variabilité du patient)
   - `on_patient_mean` : idem pour `on`
   - `target_patient_mean` : moyenne des targets précédents (train only, NaN pour 1ère visite)
   - `target_patient_trend` : pente de régression linéaire de target vs age sur les visites précédentes
     (capture la progression de la maladie : est-ce que ça empire vite ou lentement ?)
4. **Score de débiaisage moyen historique** :
   - `bias_correction_mean` = mean(`target` - `off`) sur les visites précédentes du patient
   - Interprétation : combien le vrai OFF diffère du OFF mesuré en moyenne pour ce patient
   - C'est exactement la quantité à prédire — si on peut estimer l'historique du biais, on peut
     corriger la visite courante
5. **Fenêtre glissante (3 visites)** :
   - `off_roll3_mean` : moyenne des 3 derniers `off` du patient
   - `target_roll3_mean` : moyenne des 3 derniers `target` du patient (si disponibles)
6. **Pente de progression globale** :
   - `disease_slope` : (last_off - first_off) / (last_age - first_age) par patient
   - Capture la vitesse de dégradation du patient sur toute son histoire
7. Évaluer avec GroupKFold, pusher `11_patient_history`
8. Fit final + `submissions/11_patient_history.csv`

### Relevant Context

- `target_patient_mean` et `bias_correction_mean` sont les features les plus prometteuses :
  elles donnent au modèle la réponse historique du patient, pas juste ses mesures brutes
- Ces features ne fuient pas le CV si calculées en mode expanding (visite courante exclue)
  et si le split est par patient (GroupKFold garantit ça)
- Pour le test : recalculer les agrégats en utilisant TOUTES les visites train du patient
  (pas d'expanding, juste la moyenne globale train) comme proxy de l'histoire

---

## Sous-tâche 5 — Target encoding et features de rater

**Status : [ ] pending**

### Intent

Exploiter `rater_id` de façon contrôlée (sans leakage) et encoder la cible avec un
target encoding pour les variables catégorielles. Le rater peut introduire un biais
systématique de scoring qui, s'il est stable, est prédictible.

### Expected Outcomes

- Script `experiments/archive_leaky/12_target_encoding.py`
- Rapport `12_target_encoding` sur Skore Hub
- RMSE potentiellement réduit si le rater a un biais stable
- Fichier `submissions/12_target_encoding.csv`

### Todo List

1. Créer `experiments/archive_leaky/12_target_encoding.py`
2. **Analyse préliminaire rater_id** :
   - Calculer `(target - off).groupby(rater_id).mean()` — est-ce que certains raters
     sur-évaluent ou sous-évaluent systématiquement ?
   - Si l'écart-type inter-rater est > 1 RMSE point, `rater_id` est utile
3. **Target encoding de rater_id** :
   - `rater_bias` = mean(`target` - `off`) par rater (calculé sur le train, appliqué au test)
   - Utiliser `skrub.TargetEncoder` ou calcul manuel avec leave-one-out pour éviter le leakage
4. **Target encoding de gene** :
   - `gene_target_mean` = mean(`target`) par valeur de gene
   - Peut capturer des différences génétiques sur la progression
5. **Target encoding de cohort** :
   - `cohort_target_mean` = mean(`target`) par cohort
6. Combiner avec le meilleur feature set précédent
7. Évaluer avec GroupKFold, pusher `12_target_encoding`
8. Fit final + `submissions/12_target_encoding.csv`

### Relevant Context

- `rater_id` est dans la liste "à exclure" de l'EDA (leakage potentiel) mais ce leakage
  est sur les scores mesurés (`on`, `off`), pas sur `target` — le biais de rater est justement
  ce qu'on veut modéliser
- Utiliser un target encoding leave-one-out (ou cross-fold) pour ne pas leaker le target
  dans le feature du même exemple
- `skrub.TargetEncoder` gère ça nativement

---

## Sous-tâche 6 — Modèle séquentiel : prédire le résidu (target - off)

**Status : [ ] pending**

### Intent

Reformuler le problème : au lieu de prédire `target` directement, prédire la correction
`residual = target - off` (le biais de débiaisage). C'est une transformation du problème qui
tire parti du fait que `off` ≈ `target` à 87 %. Le modèle n'a plus qu'à apprendre une correction
de ±quelques points, pas un score absolu de 0–109.

### Expected Outcomes

- Script `experiments/archive_leaky/13_residual_model.py`
- Rapport `13_residual_model` sur Skore Hub
- RMSE potentiellement < 3.0 (le résidu est plus facile à prédire)
- Fichier `submissions/13_residual_model.csv`

### Todo List

1. Créer `experiments/archive_leaky/13_residual_model.py`
2. Construire `y_residual = visits["target"] - visits["off"]`
   (NaN pour les visites sans `off` → deux stratégies ci-dessous)
3. **Stratégie A : prédire le résidu quand off est disponible, prédire target quand off est NaN** :
   - Séparer les visites en deux groupes : `has_off` (off non-NaN) et `no_off`
   - Pour `has_off` : entraîner un modèle sur `y_residual`, prédire `residual`, then `pred_target = off + pred_residual`
   - Pour `no_off` : entraîner un modèle séparé sur `y_target` directement
4. **Stratégie B : résidu global** :
   - Remplacer les NaN de `off` par 0 dans `y_residual` (le modèle prédit la correction depuis 0)
   - Un seul modèle, feature `off_missing` indique le contexte
5. Tester les deux stratégies avec GroupKFold
6. Pusher le meilleur sous `13_residual_model`
7. Pour la prédiction finale : `pred_target = off_test + pred_residual` (ou 0 + pred si off manquant)
8. Écrire `submissions/13_residual_model.csv`

### Relevant Context

- Corrélation `off` vs `target` = 0.87 → résidu std ≈ 8 (beaucoup plus petit que std target = 16)
- Le modèle de résidu doit apprendre : timing des médicaments, biais du rater, progression
- Pour le test : off est parfois NaN → prévoir le fallback

---

## Sous-tâche 7 — Combinaison finale et optimisation globale

**Status : [ ] pending**

### Intent

Assembler le meilleur feature set de toutes les sous-tâches précédentes, lancer un dernier
cycle de tuning, et produire la soumission finale optimisée.

### Expected Outcomes

- Script `experiments/archive_leaky/14_final_model.py`
- Rapport `14_final_model` sur Skore Hub avec RMSE ≤ 2.196
- Fichier `submissions/14_final_model.csv` — soumission Kaggle finale
- URL du rapport Hub notée pour la description Kaggle

### Todo List

1. Créer `experiments/archive_leaky/14_final_model.py`
2. **Sélectionner les features** : partir du meilleur feature set identifié dans ST1–ST6,
   retirer les features qui n'ont pas amélioré le RMSE
3. **Feature importance** : afficher `report.feature_importance.mean_decrease_impurity()`
   pour identifier les features les plus utiles et supprimer les <5 % d'importance
4. **Tuning final HGBR** :
   - Fixer `max_iter=500`, `learning_rate=0.05`, `max_leaf_nodes=63`
   - `l2_regularization` : tester 0.0, 0.1, 1.0
5. **Ensemble : moyenne de 5 seeds** :
   - Entraîner 5 HGBR avec `random_state` différents (0, 1, 2, 3, 4)
   - Moyenne des prédictions → réduit la variance sans changer le biais
6. Évaluer l'ensemble avec GroupKFold, comparer vs modèle single seed
7. Pusher `14_final_model` sur Skore Hub
8. Fit final de l'ensemble sur tout le train, prédire le test
9. Écrire `submissions/14_final_model.csv`
10. Afficher l'URL du rapport Hub pour la soumission Kaggle

### Relevant Context

- L'objectif est RMSE ≤ 2.196 ; si ce seuil n'est pas atteint après ST1–ST6, c'est que
  les features temporelles manquent encore de signal → revisiter `target_lag1` et `bias_correction_mean`
- L'ensemble de seeds est le levier de dernière chance : gain typique 0.1–0.3 RMSE point sans
  changer l'architecture

---

## Ordre d'exécution et dépendances

```
ST1 (temporal features)
  └── ST2 (PK features, construit sur ST1)
        └── ST3 (tuning estimateur, sur ST1+ST2)
              └── ST4 (patient history, sur meilleur feature set)
                    └── ST5 (target encoding, sur meilleur feature set)
                          └── ST6 (résidu model, reformulation du problème)
                                └── ST7 (combinaison finale)
```

Chaque sous-tâche évalue avec GroupKFold et pousse son rapport sur Skore Hub.
Si une sous-tâche ne bat pas la précédente, on conserve le feature set précédent et on passe à la suivante.

## Critère de succès

RMSE grouped-CV ≤ 2.196 sur le meilleur rapport Hub.
