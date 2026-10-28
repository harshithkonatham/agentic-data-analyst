"""A fixed, rule-based analysis that needs no API key.

It calls the same tools the Claude agents use, so it doubles as the CI smoke test and as
a baseline to compare the agents against.
"""

from __future__ import annotations

import pandas as pd

from . import tools
from .models import AnalysisPlan, AnalysisResult, FindingReview, Review
from .workspace import Workspace

DEFAULT_QUESTION = "What drives the target, and how well can it be predicted?"

OFFLINE_STEPS = [
    "Profile the dataset: size, column types, duplicates and missing values.",
    "Describe the target column and plot its distribution.",
    "Train a cross-validated random-forest baseline on the target.",
    "Check linear correlations with the target (numeric targets) or between features.",
    "Plot the most important feature against the target.",
]


def run_offline(ws: Workspace, question: str = DEFAULT_QUESTION) -> AnalysisResult:
    plan = AnalysisPlan(goal=question, steps=OFFLINE_STEPS)

    overview = tools.dataset_overview(ws)
    tools.record_finding(
        ws,
        "Dataset size",
        f"{overview['rows']} rows and {overview['columns']} columns "
        f"({len(overview['numeric_columns'])} numeric); "
        f"{overview['duplicate_rows']} duplicate rows.",
        overview["evidence"],
    )

    missing = tools.missing_values(ws)
    if missing["columns_with_missing"]:
        worst = max(missing["columns_with_missing"].items(), key=lambda kv: kv[1]["count"])
        detail = (
            f"{len(missing['columns_with_missing'])} columns have gaps; the worst is "
            f"{worst[0]!r} with {worst[1]['percent']}% missing."
        )
    else:
        detail = "No missing values in any column."
    tools.record_finding(ws, "Data completeness", detail, missing["evidence"])

    if ws.target:
        _analyse_target(ws, ws.target)
    else:
        corr = tools.correlations(ws, top_n=5)
        top = corr["strongest_pairs"][0]
        tools.record_finding(
            ws,
            "Strongest relationship",
            f"{top['a']!r} and {top['b']!r} have the strongest linear correlation (r={top['r']}).",
            corr["evidence"],
        )

    review = Review(
        reviews=[
            FindingReview(
                title=f.title,
                verdict="supported",
                comment=f"Computed directly from {f.evidence}.",
            )
            for f in ws.findings
        ],
        summary=" ".join(f.detail for f in ws.findings),
        caveats=[
            "Offline mode runs a fixed recipe; it does not adapt to the question.",
            "Correlations and feature importances show association, not causation.",
            "The random forest is a baseline, not a tuned model.",
        ],
    )
    return AnalysisResult(
        mode="offline", model=None, question=question, plan=plan, review=review, workspace=ws
    )


def _analyse_target(ws: Workspace, target: str) -> None:
    tools.describe_column(ws, target)
    hist = tools.plot_histogram(ws, target)

    model = tools.train_baseline_model(ws, target)
    tools.record_finding(
        ws,
        "Baseline predictability",
        f"A {model['model']} predicts {target!r} ({model['task']}) with "
        f"{model['metric']} = {model['cv_mean']} ± {model['cv_std']} "
        f"over {model['cv_folds']}-fold cross-validation.",
        f"{model['evidence']}, {hist['evidence']}",
    )

    top_features = [f["feature"] for f in model["top_features"][:3]]
    tools.record_finding(
        ws,
        "Most important features",
        f"The model relies most on {', '.join(map(repr, top_features))}.",
        model["evidence"],
    )

    if pd.api.types.is_numeric_dtype(ws.df[target]):
        corr = tools.correlations(ws, target=target, top_n=5)
        if corr["strongest_pairs"]:
            top = corr["strongest_pairs"][0]
            direction = "rises" if top["r"] > 0 else "falls"
            tools.record_finding(
                ws,
                "Strongest linear link to the target",
                f"{target!r} {direction} with {top['b']!r} (r={top['r']}).",
                corr["evidence"],
            )

    best = top_features[0]
    if pd.api.types.is_numeric_dtype(ws.df[best]):
        tools.plot_scatter(ws, best, target)
        if model["task"] == "classification":
            groups = tools.group_summary(ws, by=target, column=best, agg="mean")
            ranked = list(groups["groups"].items())
            tools.record_finding(
                ws,
                f"{best} by {target}",
                f"Mean {best!r} ranges from {ranked[-1][1]} ({target}={ranked[-1][0]}) "
                f"to {ranked[0][1]} ({target}={ranked[0][0]}).",
                groups["evidence"],
            )
