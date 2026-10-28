"""Agent wiring tests with a fake Anthropic client: no network, no API key."""

import json
from types import SimpleNamespace

import pytest

from agentic_analyst.agent import AgentRefused, AnalysisTeam, build_tools
from agentic_analyst.models import AnalysisPlan, FindingReview, Review


def _message(text="done", stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=None,
        content=[SimpleNamespace(type="text", text=text)],
    )


class FakeRunner:
    """Plays the analyst: calls real tools the way Claude would."""

    def __init__(self, tools):
        self.tools = {t.name: t for t in tools}

    def until_done(self):
        model = json.loads(self.tools["train_baseline_model"].call({"target": "cultivar"}))
        bad = self.tools["describe_column"].call({"column": "nope"})
        assert bad.startswith("Error:")
        self.tools["record_finding"].call(
            {
                "title": "Cultivar is predictable",
                "detail": f"f1_macro={model['cv_mean']}",
                "evidence": model["evidence"],
            }
        )
        return _message("Could not check seasonality.")


class FakeClient:
    def __init__(self, refuse=False):
        self.calls = []
        self.refuse = refuse
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(parse=self._parse, tool_runner=self._tool_runner)
        )

    def _parse(self, **kwargs):
        self.calls.append(("parse", kwargs))
        if self.refuse:
            return _message(stop_reason="refusal")
        if kwargs["output_format"] is AnalysisPlan:
            parsed = AnalysisPlan(goal="Explain cultivar", steps=["Train a baseline"])
        else:
            parsed = Review(
                reviews=[
                    FindingReview(
                        title="Cultivar is predictable", verdict="supported", comment="ok"
                    )
                ],
                summary="Cultivar can be predicted well.",
                caveats=["Small dataset."],
            )
        return SimpleNamespace(stop_reason="end_turn", parsed_output=parsed)

    def _tool_runner(self, **kwargs):
        self.calls.append(("tool_runner", kwargs))
        return FakeRunner(kwargs["tools"])


def test_team_runs_plan_analyse_review(wine_ws):
    client = FakeClient()
    result = AnalysisTeam(client=client).run(wine_ws, "What separates cultivars?")

    assert [name for name, _ in client.calls] == ["parse", "tool_runner", "parse"]
    for _, kwargs in client.calls:
        assert kwargs["model"] == "claude-opus-5-5"
        assert kwargs["fallbacks"] == "default"
        assert kwargs["output_config"] == {"effort": "high"}
    assert result.mode == "claude"
    assert result.plan.steps == ["Train a baseline"]
    assert result.review.reviews[0].verdict == "supported"
    assert result.analyst_notes == "Could not check seasonality."
    assert wine_ws.findings[0].title == "Cultivar is predictable"

    # The reviewer sees the cited tool output, not just the claim.
    review_prompt = client.calls[2][1]["messages"][0]["content"]
    assert '"cv_mean"' in review_prompt


def test_refusal_raises(wine_ws):
    with pytest.raises(AgentRefused):
        AnalysisTeam(client=FakeClient(refuse=True)).run(wine_ws, "q")


def test_tool_schemas(small_ws):
    names = {t.name for t in build_tools(small_ws)}
    assert names == {
        "describe_column",
        "missing_values",
        "correlations",
        "group_summary",
        "train_baseline_model",
        "plot_histogram",
        "plot_scatter",
        "record_finding",
    }
