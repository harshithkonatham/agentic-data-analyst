from agentic_analyst import report
from agentic_analyst.cli import main
from agentic_analyst.offline import run_offline


def test_offline_classification_report(wine_ws):
    result = run_offline(wine_ws)
    titles = [f.title for f in wine_ws.findings]
    assert "Baseline predictability" in titles
    assert any(t.endswith("by cultivar") for t in titles)
    path = report.write(result)
    text = path.read_text()
    assert "## Findings" in text and "## Evidence log" in text
    assert "![" in text  # charts embedded


def test_offline_regression_has_correlation_finding(diabetes_ws):
    run_offline(diabetes_ws)
    assert "Strongest linear link to the target" in [f.title for f in diabetes_ws.findings]


def test_offline_without_target(small_ws):
    run_offline(small_ws)
    assert small_ws.findings[-1].title == "Strongest relationship"


def test_cli_demo_offline(tmp_path, capsys):
    assert main(["demo", "--offline", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").exists()
    assert "findings" in capsys.readouterr().out


def test_cli_bad_target(tmp_path, capsys):
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    assert main(["analyze", str(csv), "--target", "zzz", "--offline", "--out", str(tmp_path)]) == 2
    assert "not found" in capsys.readouterr().out


def test_cli_reports_refusal(tmp_path, monkeypatch, capsys):
    from agentic_analyst import agent

    def refuse(self, ws, question):
        raise agent.AgentRefused("declined")

    monkeypatch.setattr(agent.AnalysisTeam, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(agent.AnalysisTeam, "run", refuse)
    assert main(["demo", "--out", str(tmp_path)]) == 3
    assert "declined" in capsys.readouterr().out
