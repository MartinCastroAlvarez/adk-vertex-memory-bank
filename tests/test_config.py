"""Settings, model selection, the localbank:// URI and services.py."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from travel_concierge.config import DEFAULT_GEMINI_MODEL
from travel_concierge.config import ConfigError
from travel_concierge.config import Settings
from travel_concierge.config import is_loopback
from travel_concierge.memory.extract import LlmExtractor
from travel_concierge.memory.extract import RuleExtractor
from travel_concierge.memory.factory import build_memory_service
from travel_concierge.memory.factory import default_extractor
from travel_concierge.memory.local import LocalMemoryBank
from travel_concierge.memory.local_uri import local_bank_from_uri
from travel_concierge.models import concierge_model
from travel_concierge.models import extraction_model
from travel_concierge.models import ollama_model_id
from travel_concierge.offline import OfflineConcierge

ROOT = Path(__file__).resolve().parent.parent


def test_defaults_are_local_and_free() -> None:
    s = Settings.from_env({})
    assert (s.provider, s.model, s.memory, s.extractor) == ("ollama", "qwen3:4b", "local", "llm")
    assert s.ttl_seconds == 365 * 86400
    assert s.db_path.name == "memory.sqlite3"


def test_env_overrides() -> None:
    s = Settings.from_env(
        {
            "CONCIERGE_PROVIDER": "Gemini",
            "CONCIERGE_MEMORY": "inmemory",
            "CONCIERGE_MEMORY_DB": "~/x.sqlite3",
            "CONCIERGE_MEMORY_TTL_DAYS": "0",
            "CONCIERGE_MEMORY_TOP_K": "3",
        }
    )
    assert s.provider == "gemini" and s.model == DEFAULT_GEMINI_MODEL
    assert s.memory == "inmemory" and s.ttl_seconds is None and s.top_k == 3
    assert "~" not in str(s.db_path)
    assert Settings.from_env({"CONCIERGE_PROVIDER": "stub"}).extractor == "rules"


@pytest.mark.parametrize(
    "env",
    [
        {"CONCIERGE_PROVIDER": "openai"},
        {"CONCIERGE_MEMORY": "redis"},
        {"CONCIERGE_EXTRACTOR": "magic"},
        {"CONCIERGE_MEMORY_TTL_DAYS": "forever"},
        {"CONCIERGE_MEMORY_TTL_DAYS": "-1"},
        {"CONCIERGE_MEMORY_TOP_K": "500"},
        {"OLLAMA_HOST": "http://10.0.0.5:11434"},
        {"CONCIERGE_MEMORY": "vertex"},
    ],
)
def test_bad_settings_are_refused(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_remote_ollama_needs_an_explicit_opt_in() -> None:
    env = {"OLLAMA_HOST": "http://10.0.0.5:11434", "CONCIERGE_ALLOW_REMOTE_OLLAMA": "1"}
    assert Settings.from_env(env).allow_remote_ollama
    assert is_loopback("http://localhost:1") and is_loopback("http://[::1]:1")
    assert not is_loopback("http://example.com")


def test_direct_construction_is_validated() -> None:
    for kwargs in ({"provider": "x"}, {"memory": "x"}, {"extractor": "x"}, {"top_k": 0}):
        with pytest.raises(ConfigError):
            Settings(**kwargs)  # type: ignore[arg-type]


def test_models_per_provider() -> None:
    assert ollama_model_id("qwen3:4b") == "ollama_chat/qwen3:4b"
    assert ollama_model_id("ollama/llama3.2") == "ollama/llama3.2"
    assert isinstance(concierge_model(Settings(provider="stub")), OfflineConcierge)
    llm = concierge_model(Settings())
    assert llm.model == "ollama_chat/qwen3:4b"
    assert extraction_model(Settings()).model == "ollama_chat/qwen3:4b"
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        concierge_model(Settings(provider="gemini", model=DEFAULT_GEMINI_MODEL), env={})
    gemini = concierge_model(
        Settings(provider="gemini", model=DEFAULT_GEMINI_MODEL), env={"GOOGLE_API_KEY": "x"}
    )
    assert gemini.model == DEFAULT_GEMINI_MODEL
    assert extraction_model(Settings(provider="gemini"), env={"GOOGLE_API_KEY": "x"})
    with pytest.raises(ConfigError, match="CONCIERGE_EXTRACTOR=rules"):
        extraction_model(Settings(provider="stub"))


def test_extractors_and_backends_from_settings(tmp_path: Path) -> None:
    assert isinstance(default_extractor(Settings(extractor="rules")), RuleExtractor)
    assert isinstance(default_extractor(Settings()), LlmExtractor)
    from google.adk.memory import InMemoryMemoryService

    assert isinstance(build_memory_service(Settings(memory="inmemory")), InMemoryMemoryService)
    bank = build_memory_service(Settings(db_path=tmp_path / "m.sqlite3"))
    assert isinstance(bank, LocalMemoryBank) and isinstance(bank.extractor, LlmExtractor)


def test_localbank_uri(tmp_path: Path) -> None:
    settings = Settings(provider="stub", extractor="rules", db_path=tmp_path / "default.sqlite3")
    assert local_bank_from_uri("localbank://", settings).path == str(tmp_path / "default.sqlite3")
    other = local_bank_from_uri(f"localbank://{tmp_path}/other.sqlite3", settings)
    assert other.path == str(tmp_path / "other.sqlite3")
    with pytest.raises(ConfigError):
        local_bank_from_uri("sqlite:///x.db", settings)


def test_services_py_registers_the_localbank_scheme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from google.adk.cli.service_registry import get_service_registry

    monkeypatch.setenv("CONCIERGE_PROVIDER", "stub")
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("services", None)
    importlib.import_module("services")
    service = get_service_registry().create_memory_service(
        f"localbank://{tmp_path}/adk.sqlite3", agents_dir=str(ROOT)
    )
    assert isinstance(service, LocalMemoryBank)
    assert isinstance(service.extractor, RuleExtractor)
