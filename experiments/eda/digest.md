# Cells: `data\eda.py`

## Cell 0: `# %% [markdown]`

# EDA: Parkinson's UPDRS Motor Score
#
Exploratory data analysis of the training set (`X_train.csv` + `y_train.csv`),
run before designing a model.
#
- **Raw data** is read-only — `X_train.csv`, `y_train.csv` and `X_test.csv`
  live in `data/`. This file never cleans or modifies the raw data.
- **Outputs** go under `EDA_DIR` (the repo's `data/`): an
  `eda_<table>.html` report per table, summarized in `eda.md`.

## Cell 1: `# %%`

```python
import json
from pathlib import Path

import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data may
# live elsewhere.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)
```

## Cell 2: `# %% [markdown]`

## Load the raw data
#
Merge features and target into a single training table. Both files
share an `Index` column that is dropped after joining.

## Cell 3: `# %%`

```python
import pandas as pd

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
RAW = X_train.join(y_train)
RAW.shape  # noqa: B018
```

**stdout:**
```
Out[0]: (44590, 13)
```

## Cell 4: `# %% [markdown]`

## Table overview
#
Per-table report (column types, distributions, associations) saved to
`data/eda_train.html`, plus a compact per-column summary: dtype,
fraction missing, and number of unique values.

## Cell 5: `# %%`

```python
report = skrub.TableReport(RAW, title="Parkinson's train set", verbose=0)
report.write_html(EDA_DIR / "eda_train.html")

summary = json.loads(report.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}  # noqa: B018
```

**stdout:**
```
Out[0]: 
{'n_rows': 44590,
 'n_columns': 13,
 'columns': [{'column': 'patient_id',
   'dtype': 'StringDtype',
   'null_pct': 0.0,
   'n_unique': 5576},
  {'column': 'cohort', 'dtype': 'StringDtype', 'null_pct': 0.0, 'n_unique': 2},
  {'column': 'sexM', 'dtype': 'Int64DType', 'null_pct': 0.0, 'n_unique': 2},
  {'column': 'gene',
   'dtype': 'StringDtype',
   'null_pct': 0.3236600134559318,
   'n_unique': 4},
  {'column': 'age_at_diagnosis',
   'dtype': 'Float64DType',
   'null_pct': 0.05200717649697242,
   'n_unique': 563},
  {'column': 'age',
   'dtype': 'Float64DType',
   'null_pct': 0.0,
   'n_unique': 1082},
  {'column': 'ledd',
   'dtype': 'Float64DType',
   'null_pct': 0.3664050235478807,
   'n_unique': 1320},
  {'column': 'time_since_intake_on',
   'dtype': 'Float64DType',
   'null_pct': 0.46373626373626375,
   'n_unique': 64},
  {'column': 'time_since_intake_off',
   'dtype': 'Float64DType',
   'null_pct': 0.7875308365104283,
   'n_unique': 178},
  {'column': 'rater_id',
   'dtype': 'StringDtype',
   'null_pct': 0.0,
   'n_unique': 50},
  {'column': 'on',
   'dtype': 'Float64DType',
   'null_pct': 0.29647903117290875,
   'n_unique': 83},
  {'column': 'off',
   'dtype': 'Float64DType',
   'null_pct': 0.42415339762278537,
   'n_unique': 101},
  {'column': 'target',
   'dtype': 'Float64DType',
   'null_pct': 0.0,
   'n_unique': 867}]}
```

## Cell 6: `# %% [markdown]`

## Target
#
The target's distribution — spread / skew for this regression problem.
This shapes the metric and whether cross-validation should stratify.

## Cell 7: `# %%`

```python
TARGET = "target"
next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)  # noqa: B018
```

**stdout:**
```
Out[0]: 
{'position': 12,
 'idx': 12,
 'name': 'target',
 'dtype': 'Float64DType',
 'value_is_constant': False,
 'is_ordered': False,
 'null_count': 0,
 'null_proportion': 0.0,
 'nulls_level': 'ok',
 'n_unique': 867,
 'unique_proportion': 0.019443821484637813,
 'is_high_cardinality': True,
 'is_duration': False,
 'duration_unit': None,
 'standard_deviation': 16.49967897803254,
 'mean': 37.47152724826194,
 'inter_quartile_range': 23.699999999999996,
 'quantiles': {'0.0': 0.0,
  '0.25': 25.6,
  '0.5': 37.3,
  '0.75': 49.3,
  '1.0': 109.5},
 'histogram_plot': '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n<svg xmlns:xlink="http://www.w3.org/1999/xlink" width="16.43em" height="8.62em" viewBox="0 0 197.2125 103.4" xmlns="http://www.w3.org/2000/svg" version="1.1">\n <metadata>\n  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n   <cc:Work>\n    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>\n    <dc:date>2026-10-07T15:41:46.706832</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.11.2, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>\n </defs>\n <g id="figure_1">\n  <g id="axes_1">\n   <g id="patch_1">\n    <path d="M 46.0125 79.2 \nL 190.0125 79.2 \nL 190.0125 7.2 \nL 46.0125 7.2 \nL 46.0125 79.2 \nz\n" style="fill: none"/>\n   </g>\n   <g id="patch_2">\n    <path d="M 52.557955 79.2 \nL 65.648863 79.2 \nL 65.648863 63.330484 \nL 52.557955 63.330484 \nz\n" clip-path="url(#pcd21c4f037)" style="fill: #1f77b4"/>\n   </g>\n   <g id="patch_3">\n    <path d="M 65.648863 79.2 \nL 78.739772 79.2 \nL 78.739772 43.059532 \nL 65.648863 43.059532 \nz\n" clip-path="url(#pcd21c4f037)" style="fill: #1f77b4"/>\n   </g>\n   <g id="patch_4">\n    <path d="M 78.739772 79.2 \nL 91.83068 79.2 \nL 91.83068 16.644508 \nL 78.739772 16.644508 \nz\n" clip-path="url(#pcd21c4f037)" style="fill: #1f77b4"/>\n   </g>\n   <g id="patch_5">\n    <path d="M 91.83068 79.2 \nL 104.92159 79.2 \nL 104.92159 10.628571 \nL 91.83068 10.628571 \nz\n" clip-path="url(#pcd21c4f037)" style="fill: #1f77b4"/>\n   </g>\n   <g id="patch_6">\n    <path d="M 104.92159 79.2 \nL 118.0125 79.2 \nL 118.0125 22.538717 \nL 104.92159 22.538717 \nz\n" clip-path="url(#pcd21c4f037)" style="fill: #1f77b4"/>\n   </g>\n   <g id="patch_7">\n    <path d="M 118.0125 79.2 \nL 131.103405 79.2 \nL 131.103405 45.865688 \nL 118.0125 45.865688 \nz\n" clip-path="url(#pcd21c4f037)" style="fill: #1f77b4"/>\n   </g>\n   <g id="patch_8">\n    <path d="M 131.103405 79.2 \nL 144.19432 79.2 \nL 144.19432 67.943343 \nL 131.103405 67.943343 \nz\n" clip-path="url(#pcd21c4f037)" style="fill: #1f77b4"/>\n   </g>\n   <g id="patch_9">\n    <path d="M 144.19432 79.2 \nL 157.285225 79.2 \nL 157.285225 78.046785 \nL 144.19432 78.046785 \nz\n" clip-path="url(#pcd21c4f037)" style="fill: #1f77b4"/>\n   </g>\n   <g id="patch_10">\n    <path d="M 157.285225 79.2 \nL 170.376131 79.2 \nL 170.376131 79.116712 \nL 157.285225 79.116712 \nz\n" clip-path="url(#pcd21c4f037)" style="fill: #1f77b4"/>\n   </g>\n   <g id="patch_11">\n    <path d="M 170.376131 79.2 \nL 183.467045 79.2 \nL 183.467045 79.148746 \nL 170.376131 79.148746 \nz\n" clip-path="url(#pcd21c4f037)" style="fill: #1f77b4"/>\n   </g>\n   <g id="matplotlib.axis_1">\n    <g id="xtick_1">\n     <g id="line2d_1">\n      <defs>\n       <path id="m056721cbd2" d="M 0 0 \nL 0 3.5 \n" style="stroke: var(--color-text-primary); stroke-width: 0.8"/>\n      </defs>\n      <g>\n       <use xlink:href="#m056721cbd2" x="52.557955" y="79.2" style="fill: var(--color-text-primary); stroke: var(--color-text-primary); stroke-width: 0.8"/>\n      </g>\n     </g>\n     <g id="text_1">\n      <text style="font-size: 10px; font-family: \'DejaVu Sans\', \'Bitstream Vera Sans\', \'Computer Modern Sans Serif\', \'Lucida Grande\', \'Verdana\', \'Geneva\', \'Lucid\', \'Arial\', \'Helvetica\', \'Avant Garde\', sans-serif; text-anchor: middle; fill: var(--color-text-primary)" x="52.557955" y="93.797656" transform="rotate(-0 52.557955 93.797656)">0</text>\n     </g>\n    </g>\n    <g id="xtick_2">\n     <g id="line2d_2">\n      <g>\n       <use xlink:href="#m056721cbd2" x="112.333795" y="79.2" style="fill: var(--color-text-primary); stroke: var(--color-text-primary); stroke-width: 0.8"/>\n      </g>\n     </g>\n     <g id="text_2">\n      <text style="font-size: 10px; font-family: \'DejaVu Sans\', \'Bitstream Vera Sans\', \'Computer Modern Sans Serif\', \'Lucida Grande\', \'Verdana\', \'Geneva\', \'Lucid\', \'Arial\', \'Helvetica\', \'Avant Garde\', sans-serif; text-anchor: middle; fill: var(--color-text-primary)" x="112.333795" y="93.797656" transform="rotate(-0 112.333795 93.797656)">50</text>\n     </g>\n    </g>\n    <g id="xtick_3">\n     <g id="line2d_3">\n      <g>\n       <use xlink:href="#m056721cbd2" x="172.109636" y="79.2" style="fill: var(--color-text-primary); stroke: var(--color-text-primary); stroke-width: 0.8"/>\n      </g>\n     </g>\n     <g id="text_3">\n      <text style="font-size: 10px; font-family: \'DejaVu Sans\', \'Bitstream Vera Sans\', \'Computer Modern Sans Serif\', \'Lucida Grande\', \'Verdana\', \'Geneva\', \'Lucid\', \'Arial\', \'Helvetica\', \'Avant Garde\', sans-serif; text-anchor: middle; fill: var(--color-text-primary)" x="172.109636" y="93.797656" transform="rotate(-0 172.109636 93.797656)">100</text>\n     </g>\n    </g>\n   </g>\n   <g id="matplotlib.axis_2">\n    <g id="ytick_1">\n     <g id="line2d_4">\n      <defs>\n       <path id="mda018db470" d="M 0 0 \nL -3.5 0 \n" style="stroke: var(--color-text-primary); stroke-width: 0.8"/>\n      </defs>\n      <g>\n       <use xlink:href="#mda018db470" x="46.0125" y="79.2" style="fill: var(--color-text-primary); stroke: var(--color-text-primary); stroke-width: 0.8"/>\n      </g>\n     </g>\n     <g id="text_4">\n      <text style="font-size: 10px; font-family: \'DejaVu Sans\', \'Bitstream Vera Sans\', \'Computer Modern Sans Serif\', \'Lucida Grande\', \'Verdana\', \'Geneva\', \'Lucid\', \'Arial\', \'Helvetica\', \'Avant Garde\', sans-serif; text-anchor: end; fill: var(--color-text-primary)" x="39.0125" y="82.998828" transform="rotate(-0 39.0125 82.998828)">0</text>\n     </g>\n    </g>\n    <g id="ytick_2">\n     <g id="line2d_5">\n      <g>\n       <use xlink:href="#mda018db470" x="46.0125" y="47.166258" style="fill: var(--color-text-primary); stroke: var(--color-text-primary); stroke-width: 0.8"/>\n      </g>\n     </g>\n     <g id="text_5">\n      <text style="font-size: 10px; font-family: \'DejaVu Sans\', \'Bitstream Vera Sans\', \'Computer Modern Sans Serif\', \'Lucida Grande\', \'Verdana\', \'Geneva\', \'Lucid\', \'Arial\', \'Helvetica\', \'Avant Garde\', sans-serif; text-anchor: end; fill: var(--color-text-primary)" x="39.0125" y="50.965086" transform="rotate(-0 39.0125 50.965086)">5000</text>\n     </g>\n    </g>\n    <g id="ytick_3">\n     <g id="line2d_6">\n      <g>\n       <use xlink:href="#mda018db470" x="46.0125" y="15.132516" style="fill: var(--color-text-primary); stroke: var(--color-text-primary); stroke-width: 0.8"/>\n      </g>\n     </g>\n     <g id="text_6">\n      <text style="font-size: 10px; font-family: \'DejaVu Sans\', \'Bitstream Vera Sans\', \'Computer Modern Sans Serif\', \'Lucida Grande\', \'Verdana\', \'Geneva\', \'Lucid\', \'Arial\', \'Helvetica\', \'Avant Garde\', sans-serif; text-anchor: end; fill: var(--color-text-primary)" x="39.0125" y="18.931344" transform="rotate(-0 39.0125 18.931344)">10000</text>\n     </g>\n    </g>\n   </g>\n   <g id="patch_12">\n    <path d="M 46.0125 79.2 \nL 46.0125 7.2 \n" style="fill: none; stroke: var(--color-text-primary); stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>\n   </g>\n   <g id="patch_13">\n    <path d="M 46.0125 79.2 \nL 190.0125 79.2 \n" style="fill: none; stroke: var(--color-text-primary); stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id="pcd21c4f037">\n   <rect x="46.0125" y="7.2" width="144" height="72"/>\n  </clipPath>\n </defs>\n</svg>\n',
 'histogram_data': {'n_low_outliers': 0,
  'n_high_outliers': 0,
  'bin_counts': [2477, 5641, 9764, 10703, 8844, 5203, 1757, 180, 13, 8],
  'bin_edges': [0.0,
   10.949999809265137,
   21.899999618530273,
   32.849998474121094,
   43.79999923706055,
   54.75,
   65.69999694824219,
   76.6500015258789,
   87.5999984741211,
   98.54999542236328,
   109.5]},
 'plot_names': ['histogram_plot']}
```

## Cell 8: `# %% [markdown]`

## Structure signals
#
Datetime columns (which point to time-based validation) and
high-cardinality id / group-like columns (which point to grouped
validation, to avoid leaking an entity across folds).

## Cell 9: `# %%`

```python
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows if n_rows else None,
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}  # noqa: B018
```

**stdout:**
```
Out[0]: 
{'datetime_cols': [],
 'top_unique_ratio': [{'column': 'patient_id',
   'unique_ratio': 0.1250504597443373},
  {'column': 'ledd', 'unique_ratio': 0.029603050011213276},
  {'column': 'age', 'unique_ratio': 0.024265530387979368},
  {'column': 'target', 'unique_ratio': 0.019443821484637813},
  {'column': 'age_at_diagnosis', 'unique_ratio': 0.012626149360843239},
  {'column': 'time_since_intake_off', 'unique_ratio': 0.003991926440906033},
  {'column': 'off', 'unique_ratio': 0.0022650818569185916},
  {'column': 'on', 'unique_ratio': 0.0018614039022202288},
  {'column': 'time_since_intake_on', 'unique_ratio': 0.0014352993944830679},
  {'column': 'rater_id', 'unique_ratio': 0.001121327651939897}]}
```

## Cell 10: `# %% [markdown]`

## Associations
#
Strongest pairwise column associations. Strong feature↔target links
are candidate predictors; an implausibly perfect one is a possible
leakage flag to call out explicitly.

## Cell 11: `# %%`

```python
assoc = skrub.column_associations(RAW)
# Same library as RAW: pandas has to_dict, polars has to_dicts.
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018
```

**stdout:**
```
Out[0]: 
{'with_target': [{'left_column_name': 'off',
   'left_column_idx': 11,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.40642285542671464,
   'pearson_corr': 0.8711910795602441},
  {'left_column_name': 'on',
   'left_column_idx': 10,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.36842392300602894,
   'pearson_corr': 0.6687349653701669},
  {'left_column_name': 'ledd',
   'left_column_idx': 6,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.2363247093271769,
   'pearson_corr': 0.2980933570585727},
  {'left_column_name': 'time_since_intake_on',
   'left_column_idx': 7,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.17743225988266575,
   'pearson_corr': 0.00023253136570167227},
  {'left_column_name': 'age',
   'left_column_idx': 5,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.11417739102117506,
   'pearson_corr': 0.3098162273403813},
  {'left_column_name': 'time_since_intake_off',
   'left_column_idx': 8,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.09229728589733409,
   'pearson_corr': 0.008290595475295728},
  {'left_column_name': 'cohort',
   'left_column_idx': 1,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.08974644023471143,
   'pearson_corr': nan},
  {'left_column_name': 'gene',
   'left_column_idx': 3,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.05180418260056274,
   'pearson_corr': nan},
  {'left_column_name': 'age_at_diagnosis',
   'left_column_idx': 4,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.05095673215791486,
   'pearson_corr': 0.133149644145748},
  {'left_column_name': 'patient_id',
   'left_column_idx': 0,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.027888987792137127,
   'pearson_corr': nan},
  {'left_column_name': 'sexM',
   'left_column_idx': 2,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.022862625381608062,
   'pearson_corr': -0.0013944750470912357},
  {'left_column_name': 'rater_id',
   'left_column_idx': 9,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.01859497859367726,
   'pearson_corr': nan}],
 'strongest': [{'left_column_name': 'age_at_diagnosis',
   'left_column_idx': 4,
   'right_column_name': 'age',
   'right_column_idx': 5,
   'cramer_v': 0.5622568056979192,
   'pearson_corr': 0.941552838110688},
  {'left_column_name': 'off',
   'left_column_idx': 11,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.40642285542671464,
   'pearson_corr': 0.8711910795602441},
  {'left_column_name': 'on',
   'left_column_idx': 10,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.36842392300602894,
   'pearson_corr': 0.6687349653701669},
  {'left_column_name': 'on',
   'left_column_idx': 10,
   'right_column_name': 'off',
   'right_column_idx': 11,
   'cramer_v': 0.32520058101123944,
   'pearson_corr': 0.8722753376933569},
  {'left_column_name': 'ledd',
   'left_column_idx': 6,
   'right_column_name': 'on',
   'right_column_idx': 10,
   'cramer_v': 0.29210300445250365,
   'pearson_corr': 0.20381404441098497},
  {'left_column_name': 'time_since_intake_on',
   'left_column_idx': 7,
   'right_column_name': 'on',
   'right_column_idx': 10,
   'cramer_v': 0.2526946675328062,
   'pearson_corr': -0.21802266966960424},
  {'left_column_name': 'ledd',
   'left_column_idx': 6,
   'right_column_name': 'target',
   'right_column_idx': 12,
   'cramer_v': 0.2363247093271769,
   'pearson_corr': 0.2980933570585727},
  {'left_column_name': 'time_since_intake_off',
   'left_column_idx': 8,
   'right_column_name': 'off',
   'right_column_idx': 11,
   'cramer_v': 0.22786771550778992,
   'pearson_corr': 0.03322277745207141},
  {'left_column_name': 'ledd',
   'left_column_idx': 6,
   'right_column_name': 'off',
   'right_column_idx': 11,
   'cramer_v': 0.20981668086103428,
   'pearson_corr': 0.21829405460096893},
  {'left_column_name': 'ledd',
   'left_column_idx': 6,
   'right_column_name': 'time_since_intake_on',
   'right_column_idx': 7,
   'cramer_v': 0.19823080652096933,
   'pearson_corr': -0.0005576334634403543}]}
```

## Cell 12: `# %% [markdown]`

## Summary
#
The findings and their modelling implications are written up in
`data/eda.md`.
