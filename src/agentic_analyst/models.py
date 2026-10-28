"""Typed outputs shared by the Claude agents and the offline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from .workspace import Workspace


class AnalysisPlan(BaseModel):
    goal: str = Field(description="One sentence: what this analysis should find out.")
    steps: list[str] = Field(description="3-7 concrete analysis steps, in order.")


class FindingReview(BaseModel):
    title: str
    verdict: Literal["supported", "weak", "unsupported"]
    comment: str = Field(description="Why, citing the numbers in the evidence.")


class Review(BaseModel):
    reviews: list[FindingReview]
    summary: str = Field(description="3-5 sentence executive summary of the supported findings.")
    caveats: list[str] = Field(description="Limits a reader should know about.")


@dataclass
class AnalysisResult:
    mode: str  # "claude" or "offline"
    model: str | None
    question: str
    plan: AnalysisPlan
    review: Review
    workspace: Workspace
    analyst_notes: str = ""
