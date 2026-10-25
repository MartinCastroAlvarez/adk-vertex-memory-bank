"""The CLI and the scripted demo, offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from travel_concierge.cli import main
from travel_concierge.demo import STEPS


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    db = tmp_path / "memory.sqlite3"
    monkeypatch.setenv("CONCIERGE_PROVIDER", "stub")
    monkeypatch.setenv("CONCIERGE_MEMORY_DB", str(db))
    return db


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_chat_then_list_then_forget(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--user", "alice", "I always want an aisle seat.")
    assert code == 0 and "aisle" in out
    code, out, _ = run(capsys, "--user", "alice", "--memories")
    assert out.strip() == "- The user prefers aisle seats."
    code, out, _ = run(capsys, "--user", "alice", "--memories", "--json")
    assert json.loads(out) == {"user": "alice", "memories": ["The user prefers aisle seats."]}
    code, out, _ = run(capsys, "--user", "alice", "--json", "Plan Lisbon")
    payload = json.loads(out)
    assert "aisle" in payload["reply"] and payload["tools"] == []
    code, out, _ = run(capsys, "--user", "alice", "--forget", "--json")
    assert json.loads(out) == {"user": "alice", "deleted": 1}
    code, out, _ = run(capsys, "--user", "alice", "--memories")
    assert "nothing remembered" in out
    code, out, _ = run(capsys, "--user", "alice", "--forget")
    assert out.strip() == "deleted 0"


def test_demo_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--demo", "--json")
    assert code == 0
    result = json.loads(out)
    assert len(result) == len(STEPS)
    assert "vegetarian" in result["step2"]["reply"]
    assert result["step3"]["memories"] == []
    assert result["step5"]["memories"].count("The user prefers window seats.") == 1
    assert result["step5"]["revisions"] == ["CREATED", "CREATED", "CREATED", "UPDATED"]
    assert result["step6"]["memories"] == [] and result["step6"]["revisions"] == []


def test_demo_text(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--demo")
    assert code == 0
    assert (
        "memory UPDATED  The user prefers window seats.  (was: The user prefers aisle seats.)"
        in out
    )
    assert "tools: forget_me" in out


def test_mermaid(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--mermaid", "--memory", "vertex")
    assert code == 2  # vertex needs its settings, even for a diagram
    code, out, _ = run(capsys, "--mermaid")
    assert code == 0 and out.startswith("flowchart LR") and "LocalMemoryBank" in out
    assert "preload_memory" in out and "load_memory, forget_me" in out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ((), "give a message"),
        (("--user", 'a"b', "hi"), "user_id must match"),
        (("--memory", "inmemory", "--forget"), "cannot forget"),
        (("--provider", "ollama", "--memory", "vertex", "hi"), "CONCIERGE_VERTEX_PROJECT"),
    ],
)
def test_errors_exit_2(
    capsys: pytest.CaptureFixture[str], argv: tuple[str, ...], message: str
) -> None:
    code, _, err = run(capsys, *argv)
    assert code == 2 and message in err


def test_provider_switch_picks_that_providers_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    from travel_concierge.cli import parser
    from travel_concierge.cli import settings_from

    s = settings_from(parser().parse_args(["--provider", "ollama", "x"]))
    assert (s.provider, s.model, s.extractor) == ("ollama", "qwen3:4b", "llm")
    s = settings_from(parser().parse_args(["--model", "llama3.2", "--db", "x.sqlite3", "x"]))
    assert s.model == "llama3.2" and str(s.db_path) == "x.sqlite3"
