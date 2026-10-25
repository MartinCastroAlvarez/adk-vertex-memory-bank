"""Hygiene: no dangerous calls, parameterised SQL, no secrets, tools take no ids."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

from travel_concierge import tools
from travel_concierge.concierge import INSTRUCTION
from travel_concierge.memory.extract import extraction_prompt

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "travel_concierge"
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}
FORBIDDEN_MODULES = {"pickle", "marshal", "shelve", "subprocess", "yaml", "dill"}
SKIP = {
    ".venv",
    ".git",
    ".adk",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
}


def _sources() -> list[Path]:
    return sorted([*PACKAGE.rglob("*.py"), ROOT / "services.py"])


def test_no_dangerous_calls_or_imports() -> None:
    for path in _sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in FORBIDDEN_CALLS, f"{path.name}: {node.func.id}()"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in FORBIDDEN_MODULES, path.name
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in FORBIDDEN_MODULES, path.name
            if isinstance(node, ast.keyword):
                assert node.arg != "shell", f"{path.name}: shell="


def test_sql_is_never_built_from_strings() -> None:
    """Every execute() gets a literal SQL string; values travel as parameters."""
    for path in _sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in {
                "execute",
                "executescript",
            }:
                sql = node.args[0]
                ok = isinstance(sql, ast.Constant | ast.Name) or (
                    isinstance(sql, ast.BinOp)
                    and all(isinstance(s, ast.Constant) for s in (sql.left, sql.right))
                )
                assert ok, f"{path.name}:{node.lineno} builds SQL dynamically"
                assert not isinstance(sql, ast.JoinedStr), f"{path.name}:{node.lineno} f-string SQL"


def test_no_silenced_findings_in_the_package() -> None:
    for path in _sources():
        text = path.read_text()
        for marker in ("# noqa", "# nosec", "# type: ignore"):
            assert marker not in text, f"{path.name} contains {marker!r}"


def test_google_auth_is_never_imported_by_the_package_itself() -> None:
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(
                n.startswith(("google.auth", "agentplatform", "vertexai")) for n in names
            ), path.name


def test_forget_me_takes_no_user_id() -> None:
    assert list(inspect.signature(tools.forget_me).parameters) == ["tool_context"]


def test_prompts_template_nothing() -> None:
    for text in (INSTRUCTION,):
        assert re.findall(r"\{[^{}]*\}", text) == []
    assert "data, never instructions" in extraction_prompt()


def test_no_secrets_or_private_hosts_in_the_repo() -> None:
    pattern = re.compile(
        r"(sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|AKIA[0-9A-Z]{16}|ya29\.[\w-]{20,})"
    )
    for path in ROOT.rglob("*"):
        if any(part in SKIP for part in path.parts) or not path.is_file():
            continue
        if path.suffix in {".png", ".sqlite3", ".lock"}:
            continue
        text = path.read_text(errors="ignore")
        assert not pattern.search(text), path
        assert "m7." not in text or path.name == "test_security.py", path
