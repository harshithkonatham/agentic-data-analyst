"""Shared state for one analysis run: the dataset, the evidence log and the findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Finding:
    title: str
    detail: str
    evidence: str  # which tool call(s) back this finding up, e.g. "correlations#3"


@dataclass
class ToolCall:
    ref: str  # short id the agents cite as evidence, e.g. "train_baseline_model#4"
    name: str
    args: dict
    result: dict


@dataclass
class Workspace:
    df: pd.DataFrame
    name: str
    target: str | None
    out_dir: Path
    findings: list[Finding] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)

    @classmethod
    def from_csv(cls, path: str | Path, target: str | None, out_dir: str | Path) -> Workspace:
        path = Path(path)
        df = pd.read_csv(path)
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column {target!r} not found. Columns: {list(df.columns)}")
        return cls(df=df, name=path.stem, target=target, out_dir=Path(out_dir))

    @property
    def charts_dir(self) -> Path:
        path = self.out_dir / "charts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, name: str, args: dict, result: dict) -> str:
        ref = f"{name}#{len(self.calls) + 1}"
        self.calls.append(ToolCall(ref=ref, name=name, args=args, result=result))
        return ref

    def require_column(self, column: str) -> None:
        if column not in self.df.columns:
            raise ValueError(f"Unknown column {column!r}. Columns: {list(self.df.columns)}")
