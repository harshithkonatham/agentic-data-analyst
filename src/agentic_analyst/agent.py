"""Three Claude agents that work as a small analytics team.

1. Planner  - reads a profile of the dataset and writes an analysis plan (structured output).
2. Analyst  - follows the plan using the analysis tools, and records findings with evidence
              (agent loop via the Anthropic SDK tool runner).
3. Reviewer - checks every finding against the tool outputs it cites, flags weak or
              unsupported claims and writes the summary (structured output).
"""

from __future__ import annotations

import json

import anthropic
from anthropic import beta_tool

from . import tools
from .models import AnalysisPlan, AnalysisResult, Review
from .workspace import Workspace

MODEL = "claude-opus-5-5"
# If a safety classifier declines a request, the API re-runs it on a recommended fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

PLANNER_SYSTEM = """You are the planner in a small data-analysis team. Given a profile of a
dataset and a question, write a short, concrete plan the analyst can carry out with these
tools: describe_column, missing_values, correlations, group_summary, train_baseline_model,
plot_histogram, plot_scatter. Prefer steps that answer the question over generic profiling."""

ANALYST_SYSTEM = """You are the analyst in a small data-analysis team. Carry out the plan using
the tools. Each tool result includes an "evidence" ref such as "correlations#3".

Record each conclusion with record_finding, citing the evidence refs it rests on. Only state
what the numbers show: say "associated with", not "causes". Aim for 4-8 findings that answer
the question. When you are done, reply with a short note on anything you could not check."""

REVIEWER_SYSTEM = """You are the reviewer in a small data-analysis team. For each finding,
compare the claim with the tool output it cites and give a verdict:
- supported: the cited numbers back the claim as written
- weak: the direction is right but the claim overstates the evidence (e.g. causal wording,
  a small effect described as strong, a metric quoted without its spread)
- unsupported: the cited output does not show this
Then write a summary that uses only supported findings, and list the caveats."""


class AgentRefused(RuntimeError):
    """Claude declined the request even after the server-side fallback."""


def _check(message) -> None:
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        raise AgentRefused(f"Claude declined the request: {details}")


def build_tools(ws: Workspace) -> list:
    """Wrap the analysis functions as Claude tools bound to this workspace."""

    def run(fn, **kwargs) -> str:
        try:
            return json.dumps(fn(ws, **kwargs), default=str)
        except ValueError as exc:  # bad column names etc. go back to Claude to fix
            return f"Error: {exc}"

    @beta_tool
    def describe_column(column: str) -> str:
        """Summary statistics for one column (numeric stats or top categories).

        Args:
            column: Exact column name.
        """
        return run(tools.describe_column, column=column)

    @beta_tool
    def missing_values() -> str:
        """Count and percentage of missing values per column."""
        return run(tools.missing_values)

    @beta_tool
    def correlations(target: str | None = None, top_n: int = 10) -> str:
        """Strongest Pearson correlations between numeric columns.

        Args:
            target: If given, only correlations with this numeric column.
            top_n: How many pairs to return.
        """
        return run(tools.correlations, target=target, top_n=top_n)

    @beta_tool
    def group_summary(by: str, column: str, agg: str = "mean") -> str:
        """Aggregate a column per group, largest first.

        Args:
            by: Column to group by (usually categorical).
            column: Column to aggregate.
            agg: One of mean, median, sum, count, min, max.
        """
        return run(tools.group_summary, by=by, column=column, agg=agg)

    @beta_tool
    def train_baseline_model(target: str) -> str:
        """5-fold cross-validated random forest on the target; returns score and top features.

        Args:
            target: Column to predict.
        """
        return run(tools.train_baseline_model, target=target)

    @beta_tool
    def plot_histogram(column: str) -> str:
        """Save a histogram (or bar chart for categories) of one column for the report.

        Args:
            column: Column to plot.
        """
        return run(tools.plot_histogram, column=column)

    @beta_tool
    def plot_scatter(x: str, y: str) -> str:
        """Save a scatter plot of two numeric columns for the report.

        Args:
            x: Column for the x axis.
            y: Column for the y axis.
        """
        return run(tools.plot_scatter, x=x, y=y)

    @beta_tool
    def record_finding(title: str, detail: str, evidence: str) -> str:
        """Record one conclusion for the report.

        Args:
            title: Short headline, e.g. "Alcohol separates the cultivars".
            detail: One or two sentences with the key numbers.
            evidence: Comma-separated evidence refs from earlier tool results.
        """
        return run(tools.record_finding, title=title, detail=detail, evidence=evidence)

    return [
        describe_column,
        missing_values,
        correlations,
        group_summary,
        train_baseline_model,
        plot_histogram,
        plot_scatter,
        record_finding,
    ]


class AnalysisTeam:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = MODEL,
        effort: str = "high",
        max_iterations: int = 25,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_iterations = max_iterations

    def _common(self) -> dict:
        return {
            "model": self.model,
            "max_tokens": 16000,
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def run(self, ws: Workspace, question: str) -> AnalysisResult:
        overview = tools.dataset_overview(ws)
        plan = self.plan(overview, question, ws.target)
        notes = self.analyse(ws, overview, plan, question)
        review = self.review(ws, question)
        return AnalysisResult(
            mode="claude",
            model=self.model,
            question=question,
            plan=plan,
            review=review,
            workspace=ws,
            analyst_notes=notes,
        )

    def plan(self, overview: dict, question: str, target: str | None) -> AnalysisPlan:
        prompt = (
            f"Question: {question}\nTarget column: {target or '(none)'}\n\n"
            f"Dataset profile:\n{json.dumps(overview, default=str)}"
        )
        response = self.client.beta.messages.parse(
            **self._common(),
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=AnalysisPlan,
        )
        _check(response)
        return response.parsed_output

    def analyse(self, ws: Workspace, overview: dict, plan: AnalysisPlan, question: str) -> str:
        prompt = (
            f"Question: {question}\nTarget column: {ws.target or '(none)'}\n\n"
            f"Dataset profile (evidence {overview['evidence']}):\n"
            f"{json.dumps(overview, default=str)}\n\n"
            f"Plan:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(plan.steps, 1))
        )
        runner = self.client.beta.messages.tool_runner(
            **self._common(),
            system=ANALYST_SYSTEM,
            tools=build_tools(ws),
            messages=[{"role": "user", "content": prompt}],
            max_iterations=self.max_iterations,
        )
        final = runner.until_done()
        _check(final)
        return "".join(b.text for b in final.content if b.type == "text").strip()

    def review(self, ws: Workspace, question: str) -> Review:
        results = {call.ref: call.result for call in ws.calls}
        findings = [
            {
                "title": f.title,
                "detail": f.detail,
                "evidence": {ref: results.get(ref) for ref in f.evidence.split(", ")},
            }
            for f in ws.findings
        ]
        prompt = f"Question: {question}\n\nFindings:\n{json.dumps(findings, default=str)}"
        response = self.client.beta.messages.parse(
            **self._common(),
            system=REVIEWER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=Review,
        )
        _check(response)
        return response.parsed_output
