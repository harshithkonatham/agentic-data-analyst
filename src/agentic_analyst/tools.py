"""Deterministic analysis tools.

Each tool takes the Workspace, does one well-defined piece of analysis, logs the call as
evidence and returns a plain dict. The Claude agents and the offline pipeline both use
these same functions, so every finding can be traced back to a real computation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display needed in CI or containers

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from .workspace import Finding, Workspace  # noqa: E402

MAX_CATEGORIES_FOR_CLASSIFICATION = 20


def _round(value: float, digits: int = 4) -> float | None:
    return None if pd.isna(value) else round(float(value), digits)


def dataset_overview(ws: Workspace) -> dict:
    df = ws.df
    result = {
        "rows": len(df),
        "columns": len(df.columns),
        "target": ws.target,
        "numeric_columns": df.select_dtypes("number").columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
        "sample": df.head(3).to_dict(orient="records"),
    }
    result["evidence"] = ws.log("dataset_overview", {}, result)
    return result


def missing_values(ws: Workspace) -> dict:
    counts = ws.df.isna().sum()
    missing = {
        col: {"count": int(n), "percent": round(100 * n / len(ws.df), 2)}
        for col, n in counts.items()
        if n > 0
    }
    result = {"columns_with_missing": missing, "total_missing_cells": int(counts.sum())}
    result["evidence"] = ws.log("missing_values", {}, result)
    return result


def describe_column(ws: Workspace, column: str) -> dict:
    ws.require_column(column)
    series = ws.df[column]
    if pd.api.types.is_numeric_dtype(series):
        stats = series.describe()
        result = {
            "column": column,
            "kind": "numeric",
            **{k: _round(v) for k, v in stats.items()},
            "skew": _round(series.skew()),
        }
    else:
        top = series.value_counts().head(10)
        result = {
            "column": column,
            "kind": "categorical",
            "unique": int(series.nunique()),
            "top_values": {str(k): int(v) for k, v in top.items()},
        }
    result["evidence"] = ws.log("describe_column", {"column": column}, result)
    return result


def correlations(ws: Workspace, target: str | None = None, top_n: int = 10) -> dict:
    numeric = ws.df.select_dtypes("number")
    if numeric.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for correlations.")
    corr = numeric.corr()
    if target:
        ws.require_column(target)
        if target not in corr:
            raise ValueError(f"Target {target!r} is not numeric, so it has no correlations.")
        series = corr[target].drop(target).dropna()
        ranked = series.reindex(series.abs().sort_values(ascending=False).index).head(top_n)
        pairs = [{"a": target, "b": col, "r": _round(r, 3)} for col, r in ranked.items()]
    else:
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
        ranked = upper.reindex(upper.abs().sort_values(ascending=False).index).head(top_n)
        pairs = [{"a": a, "b": b, "r": _round(r, 3)} for (a, b), r in ranked.items()]
    result = {"method": "pearson", "strongest_pairs": pairs}
    result["evidence"] = ws.log("correlations", {"target": target, "top_n": top_n}, result)
    return result


def group_summary(ws: Workspace, by: str, column: str, agg: str = "mean") -> dict:
    ws.require_column(by)
    ws.require_column(column)
    if agg not in {"mean", "median", "sum", "count", "min", "max"}:
        raise ValueError("agg must be one of mean, median, sum, count, min, max.")
    grouped = ws.df.groupby(by, dropna=False)[column].agg(agg).sort_values(ascending=False)
    result = {
        "by": by,
        "column": column,
        "agg": agg,
        "groups": {str(k): _round(v) for k, v in grouped.head(20).items()},
    }
    result["evidence"] = ws.log("group_summary", {"by": by, "column": column, "agg": agg}, result)
    return result


def _task_type(series: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    is_small_int_set = series.nunique() <= MAX_CATEGORIES_FOR_CLASSIFICATION and np.allclose(
        series.dropna() % 1, 0
    )
    return "classification" if is_small_int_set else "regression"


def train_baseline_model(ws: Workspace, target: str, folds: int = 5) -> dict:
    """Cross-validated random-forest baseline plus the features it leans on most."""
    ws.require_column(target)
    data = ws.df.dropna(subset=[target])
    X, y = data.drop(columns=[target]), data[target]
    numeric = X.select_dtypes("number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    task = _task_type(y)
    if task == "classification":
        model, scoring = RandomForestClassifier(n_estimators=200, random_state=0), "f1_macro"
    else:
        model, scoring = RandomForestRegressor(n_estimators=200, random_state=0), "r2"
    pipe = Pipeline([("prep", prep), ("model", model)])
    scores = cross_val_score(pipe, X, y, cv=folds, scoring=scoring)
    pipe.fit(X, y)

    # Fold one-hot columns back into their source column so importances read naturally.
    names = pipe.named_steps["prep"].get_feature_names_out()
    importances: dict[str, float] = {}
    for name, value in zip(names, pipe.named_steps["model"].feature_importances_, strict=True):
        source = name.split("__", 1)[1]
        for col in categorical:
            if source.startswith(f"{col}_"):
                source = col
                break
        importances[source] = importances.get(source, 0.0) + float(value)
    top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:8]

    result = {
        "target": target,
        "task": task,
        "model": type(model).__name__,
        "metric": scoring,
        "cv_folds": folds,
        "cv_mean": _round(scores.mean()),
        "cv_std": _round(scores.std()),
        "top_features": [{"feature": f, "importance": _round(v, 3)} for f, v in top],
    }
    result["evidence"] = ws.log("train_baseline_model", {"target": target, "folds": folds}, result)
    return result


def plot_histogram(ws: Workspace, column: str) -> dict:
    ws.require_column(column)
    fig, ax = plt.subplots(figsize=(6, 4))
    series = ws.df[column].dropna()
    if pd.api.types.is_numeric_dtype(series):
        ax.hist(series, bins=30, color="#4C78A8")
    else:
        series.value_counts().head(15).plot.bar(ax=ax, color="#4C78A8")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    path = ws.charts_dir / f"hist_{_slug(column)}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    result = {"chart": str(path.relative_to(ws.out_dir))}
    result["evidence"] = ws.log("plot_histogram", {"column": column}, result)
    return result


def plot_scatter(ws: Workspace, x: str, y: str) -> dict:
    ws.require_column(x)
    ws.require_column(y)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ws.df[x], ws.df[y], s=12, alpha=0.6, color="#4C78A8")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    path = ws.charts_dir / f"scatter_{_slug(x)}_{_slug(y)}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    result = {"chart": str(path.relative_to(ws.out_dir))}
    result["evidence"] = ws.log("plot_scatter", {"x": x, "y": y}, result)
    return result


def record_finding(ws: Workspace, title: str, detail: str, evidence: str) -> dict:
    known = {call.ref for call in ws.calls}
    cited = [ref.strip() for ref in evidence.split(",") if ref.strip()]
    unknown = [ref for ref in cited if ref not in known]
    if not cited or unknown:
        raise ValueError(
            f"Evidence must cite earlier tool calls by ref. Unknown: {unknown or '(none given)'}. "
            f"Known refs: {sorted(known)}"
        )
    ws.findings.append(Finding(title=title, detail=detail, evidence=", ".join(cited)))
    return {"recorded": title, "total_findings": len(ws.findings)}


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:40]
