"""Settings for the concierge, read from the environment (``CONCIERGE_*``)."""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Literal
from typing import cast
from urllib.parse import urlparse

Provider = Literal["ollama", "gemini", "stub"]
PROVIDERS: tuple[Provider, ...] = ("ollama", "gemini", "stub")
Backend = Literal["local", "vertex", "inmemory"]
BACKENDS: tuple[Backend, ...] = ("local", "vertex", "inmemory")
ExtractorKind = Literal["llm", "rules"]
EXTRACTORS: tuple[ExtractorKind, ...] = ("llm", "rules")

DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_DB = Path("~/.cache/travel-concierge/memory.sqlite3")
APP_NAME = "travel_concierge"

_ENGINE_ID = re.compile(r"^[0-9]{1,32}$")
_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_LOCATION = re.compile(r"^[a-z]+(-[a-z0-9]+)+$")


class ConfigError(ValueError):
    """A setting is missing or invalid."""


def _int(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _choice(env: Mapping[str, str], name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = env.get(name, "").strip().lower() or default
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def is_loopback(url: str) -> bool:
    """True when ``url`` points at this machine (127.0.0.0/8, ::1 or localhost)."""
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class VertexSettings:
    """Where the Memory Bank lives. Only read when ``memory == "vertex"``."""

    project: str = ""
    location: str = ""
    agent_engine_id: str = ""

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("CONCIERGE_VERTEX_PROJECT", self.project),
                ("CONCIERGE_VERTEX_LOCATION", self.location),
                ("CONCIERGE_VERTEX_AGENT_ENGINE_ID", self.agent_engine_id),
            )
            if not value
        ]
        if missing:
            raise ConfigError(
                "CONCIERGE_MEMORY=vertex needs "
                + ", ".join(missing)
                + " (see docs/vertex-setup.md;"
                " Memory Bank is a paid Google Cloud service)"
            )
        if not _PROJECT.match(self.project):
            raise ConfigError(f"CONCIERGE_VERTEX_PROJECT {self.project!r} is not a project id")
        if not _LOCATION.match(self.location):
            raise ConfigError(f"CONCIERGE_VERTEX_LOCATION {self.location!r} is not a region")
        if not _ENGINE_ID.match(self.agent_engine_id):
            raise ConfigError(
                "CONCIERGE_VERTEX_AGENT_ENGINE_ID is the numeric id at the end of "
                "projects/.../reasoningEngines/<id>, not the full resource name"
            )


@dataclass(frozen=True)
class Settings:
    """Everything the concierge needs. Build it with :meth:`from_env` or directly."""

    provider: Provider = "ollama"
    model: str = DEFAULT_OLLAMA_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    allow_remote_ollama: bool = False
    memory: Backend = "local"
    db_path: Path = field(default=DEFAULT_DB)
    extractor: ExtractorKind = "llm"
    ttl_days: int = 365
    top_k: int = 10
    vertex: VertexSettings = field(default_factory=VertexSettings)

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ConfigError(f"provider must be one of {', '.join(PROVIDERS)}")
        if self.memory not in BACKENDS:
            raise ConfigError(f"memory must be one of {', '.join(BACKENDS)}")
        if self.extractor not in EXTRACTORS:
            raise ConfigError(f"extractor must be one of {', '.join(EXTRACTORS)}")
        remote = not is_loopback(self.ollama_host)
        if self.provider == "ollama" and remote and not self.allow_remote_ollama:
            raise ConfigError(
                f"OLLAMA_HOST {self.ollama_host!r} is not loopback; set "
                "CONCIERGE_ALLOW_REMOTE_OLLAMA=1 to send conversations to another machine"
            )
        if not 0 <= self.ttl_days <= 3650:
            raise ConfigError("ttl_days must be between 0 (never expire) and 3650")
        if not 1 <= self.top_k <= 50:
            raise ConfigError("top_k must be between 1 and 50")
        if self.memory == "vertex":
            self.vertex.validate()

    @property
    def ttl_seconds(self) -> float | None:
        return self.ttl_days * 86400.0 if self.ttl_days else None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read ``CONCIERGE_*`` variables (and ``OLLAMA_HOST``) from ``env`` or ``os.environ``."""
        env = os.environ if env is None else env
        provider = _choice(env, "CONCIERGE_PROVIDER", "ollama", PROVIDERS)
        default_model = DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_OLLAMA_MODEL
        # The stub has no LLM to extract facts with, so it uses the rule-based extractor.
        default_extractor = "rules" if provider == "stub" else "llm"
        db = env.get("CONCIERGE_MEMORY_DB", "").strip()
        return cls(
            provider=cast(Provider, provider),
            model=env.get("CONCIERGE_MODEL", "").strip() or default_model,
            ollama_host=env.get("OLLAMA_HOST", "").strip() or DEFAULT_OLLAMA_HOST,
            allow_remote_ollama=env.get("CONCIERGE_ALLOW_REMOTE_OLLAMA", "") == "1",
            memory=cast(Backend, _choice(env, "CONCIERGE_MEMORY", "local", BACKENDS)),
            db_path=Path(db).expanduser() if db else DEFAULT_DB.expanduser(),
            extractor=cast(
                ExtractorKind,
                _choice(env, "CONCIERGE_EXTRACTOR", default_extractor, EXTRACTORS),
            ),
            ttl_days=_int(env, "CONCIERGE_MEMORY_TTL_DAYS", 365, 0, 3650),
            top_k=_int(env, "CONCIERGE_MEMORY_TOP_K", 10, 1, 50),
            vertex=VertexSettings(
                project=env.get("CONCIERGE_VERTEX_PROJECT", "").strip(),
                location=env.get("CONCIERGE_VERTEX_LOCATION", "").strip(),
                agent_engine_id=env.get("CONCIERGE_VERTEX_AGENT_ENGINE_ID", "").strip(),
            ),
        )
