# %% [markdown]
# # EDA: Parkinson's UPDRS Motor Score
#
# Exploratory data analysis of the training set (`X_train.csv` + `y_train.csv`),
# run before designing a model.
#
# - **Raw data** is read-only — `X_train.csv`, `y_train.csv` and `X_test.csv`
#   live in `data/`. This file never cleans or modifies the raw data.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): an
#   `eda_<table>.html` report per table, summarized in `eda.md`.

# %%
import json
from pathlib import Path

import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data may
# live elsewhere.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Merge features and target into a single training table. Both files
# share an `Index` column that is dropped after joining.

# %%
import pandas as pd

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
RAW = X_train.join(y_train)
RAW.shape  # noqa: B018

# %% [markdown]
# ## Table overview
#
# Per-table report (column types, distributions, associations) saved to
# `data/eda_train.html`, plus a compact per-column summary: dtype,
# fraction missing, and number of unique values.

# %%
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

# %% [markdown]
# ## Target
#
# The target's distribution — spread / skew for this regression problem.
# This shapes the metric and whether cross-validation should stratify.

# %%
TARGET = "target"
next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns (which point to time-based validation) and
# high-cardinality id / group-like columns (which point to grouped
# validation, to avoid leaking an entity across folds).

# %%
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

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations. Strong feature↔target links
# are candidate predictors; an implausibly perfect one is a possible
# leakage flag to call out explicitly.

# %%
assoc = skrub.column_associations(RAW)
# Same library as RAW: pandas has to_dict, polars has to_dicts.
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
