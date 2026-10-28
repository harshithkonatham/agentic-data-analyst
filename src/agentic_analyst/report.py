"""Render an AnalysisResult as a Markdown report with charts and an evidence appendix."""

from __future__ import annotations

import json
from pathlib import Path

from .models import AnalysisResult

VERDICT_ICON = {"supported": "✅", "weak": "⚠️", "unsupported": "❌"}


def render(result: AnalysisResult) -> str:
    ws = result.workspace
    verdicts = {r.title: r for r in result.review.reviews}
    engine = f"Claude agents ({result.model})" if result.mode == "claude" else "offline recipe"
    lines = [
        f"# Analysis report: {ws.name}",
        "",
        f"**Question:** {result.question}  ",
        f"**Target:** `{ws.target or '(none)'}` · **Rows:** {len(ws.df)} · **Engine:** {engine}",
        "",
        "## Summary",
        "",
        result.review.summary,
        "",
        "## Plan",
        "",
        f"_{result.plan.goal}_",
        "",
        *[f"{i}. {step}" for i, step in enumerate(result.plan.steps, 1)],
        "",
        "## Findings",
        "",
    ]
    for f in ws.findings:
        review = verdicts.get(f.title)
        icon = VERDICT_ICON.get(review.verdict, "") if review else ""
        lines += [f"### {icon} {f.title}".rstrip(), "", f.detail, ""]
        if review:
            lines += [f"> **Reviewer ({review.verdict}):** {review.comment}", ""]
        lines += [f"<sub>Evidence: {f.evidence}</sub>", ""]

    charts = [c.result["chart"] for c in ws.calls if "chart" in c.result]
    if charts:
        lines += ["## Charts", ""]
        lines += [f"![{Path(c).stem}]({c})" for c in charts]
        lines.append("")

    if result.review.caveats:
        lines += ["## Caveats", "", *[f"- {c}" for c in result.review.caveats], ""]
    if result.analyst_notes:
        lines += ["## Analyst notes", "", result.analyst_notes, ""]

    lines += ["## Evidence log", "", "<details><summary>Every tool call</summary>", ""]
    for call in ws.calls:
        result_json = json.dumps(call.result, default=str)
        if len(result_json) > 600:
            result_json = result_json[:600] + " …"
        lines += [f"- **{call.ref}** `{json.dumps(call.args)}` → `{result_json}`"]
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def write(result: AnalysisResult) -> Path:
    out = result.workspace.out_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.md"
    path.write_text(render(result), encoding="utf-8")
    return path
