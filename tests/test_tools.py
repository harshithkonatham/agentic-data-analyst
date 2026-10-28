import pytest

from agentic_analyst import tools


def test_overview_and_evidence_refs(small_ws):
    result = tools.dataset_overview(small_ws)
    assert result["rows"] == 6
    assert result["categorical_columns"] == ["group"]
    assert result["evidence"] == "dataset_overview#1"
    assert small_ws.calls[0].name == "dataset_overview"


def test_missing_values(small_ws):
    result = tools.missing_values(small_ws)
    assert result["columns_with_missing"] == {"x": {"count": 1, "percent": 16.67}}


def test_describe_numeric_and_categorical(small_ws):
    assert tools.describe_column(small_ws, "y")["kind"] == "numeric"
    cat = tools.describe_column(small_ws, "group")
    assert cat["kind"] == "categorical"
    assert cat["unique"] == 3


def test_unknown_column_is_rejected(small_ws):
    with pytest.raises(ValueError, match="Unknown column"):
        tools.describe_column(small_ws, "nope")


def test_correlations_with_and_without_target(small_ws):
    pairs = tools.correlations(small_ws)["strongest_pairs"]
    assert pairs[0]["r"] > 0.99
    with_target = tools.correlations(small_ws, target="y")["strongest_pairs"]
    assert with_target[0]["b"] == "x"


def test_correlations_rejects_non_numeric_target(small_ws):
    with pytest.raises(ValueError, match="not numeric"):
        tools.correlations(small_ws, target="group")


def test_group_summary(small_ws):
    result = tools.group_summary(small_ws, by="group", column="y", agg="max")
    assert list(result["groups"]) == ["c", "b", "a"]
    with pytest.raises(ValueError, match="agg must be"):
        tools.group_summary(small_ws, by="group", column="y", agg="mode")


def test_baseline_classification(wine_ws):
    result = tools.train_baseline_model(wine_ws, "cultivar")
    assert result["task"] == "classification"
    assert result["cv_mean"] > 0.9
    assert len(result["top_features"]) == 8


def test_baseline_regression(diabetes_ws):
    result = tools.train_baseline_model(diabetes_ws, "progression", folds=3)
    assert result["task"] == "regression"
    assert result["metric"] == "r2"


def test_plots_write_files(small_ws):
    hist = tools.plot_histogram(small_ws, "group")
    scatter = tools.plot_scatter(small_ws, "x", "y")
    for result in (hist, scatter):
        assert (small_ws.out_dir / result["chart"]).exists()


def test_record_finding_requires_known_evidence(small_ws):
    ref = tools.missing_values(small_ws)["evidence"]
    tools.record_finding(small_ws, "Gaps", "x has one gap", ref)
    assert small_ws.findings[0].evidence == ref
    with pytest.raises(ValueError, match="Evidence must cite"):
        tools.record_finding(small_ws, "Made up", "no proof", "correlations#99")
    with pytest.raises(ValueError, match="Evidence must cite"):
        tools.record_finding(small_ws, "Made up", "no proof", "")
