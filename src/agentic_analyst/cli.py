"""Command line: `analyst analyze data.csv --target col` or `analyst demo`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import report
from .offline import DEFAULT_QUESTION, run_offline
from .workspace import Workspace


def _demo_csv(path: Path) -> Path:
    """Write scikit-learn's bundled wine dataset (no download) as a demo CSV."""
    from sklearn.datasets import load_wine

    frame = load_wine(as_frame=True).frame.rename(columns={"target": "cultivar"})
    frame["cultivar"] = frame["cultivar"].map({0: "A", 1: "B", 2: "C"})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _run(csv: Path, target: str | None, question: str, out: Path, args) -> int:
    try:
        ws = Workspace.from_csv(csv, target, out)
    except (OSError, ValueError) as exc:
        print(f"Cannot load {csv}: {exc}")
        return 2
    if args.offline:
        result = run_offline(ws, question)
    else:
        import anthropic

        from .agent import AgentRefused, AnalysisTeam

        team = AnalysisTeam(model=args.model, effort=args.effort)
        try:
            result = team.run(ws, question)
        except anthropic.AuthenticationError:
            print("No valid Anthropic credentials. Set ANTHROPIC_API_KEY or use --offline.")
            return 2
        except AgentRefused as exc:
            print(exc)
            return 3
    path = report.write(result)
    supported = sum(r.verdict == "supported" for r in result.review.reviews)
    print(f"{len(ws.findings)} findings ({supported} supported by the reviewer) → {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analyst", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--question", default=DEFAULT_QUESTION)
        p.add_argument("--out", type=Path, default=Path("reports"))
        p.add_argument("--offline", action="store_true", help="Rule-based run, no API key")
        p.add_argument("--model", default="claude-opus-5-5")
        p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh"])

    analyze = sub.add_parser("analyze", help="Analyse a CSV file")
    analyze.add_argument("csv", type=Path)
    analyze.add_argument("--target", help="Column to explain or predict")
    common(analyze)

    demo = sub.add_parser("demo", help="Analyse the bundled wine dataset")
    common(demo)

    args = parser.parse_args(argv)
    if args.command == "demo":
        csv = _demo_csv(args.out / "wine.csv")
        return _run(csv, "cultivar", args.question, args.out, args)
    return _run(args.csv, args.target, args.question, args.out, args)


if __name__ == "__main__":
    sys.exit(main())
