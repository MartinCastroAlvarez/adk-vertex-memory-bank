"""``localbank://<path>`` -> :class:`LocalMemoryBank` (for ADK's ``--memory_service_uri``)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from travel_concierge.config import ConfigError
from travel_concierge.config import Settings
from travel_concierge.memory.factory import default_extractor
from travel_concierge.memory.local import LocalMemoryBank

SCHEME = "localbank://"


def local_bank_from_uri(uri: str, settings: Settings | None = None) -> LocalMemoryBank:
    """``localbank://`` uses ``CONCIERGE_MEMORY_DB``; ``localbank://x.sqlite3`` that file."""
    if not uri.startswith(SCHEME):
        raise ConfigError(f"expected a {SCHEME}<path> URI, got {uri!r}")
    settings = settings or Settings.from_env()
    raw = uri[len(SCHEME) :]
    if raw:
        settings = replace(settings, db_path=Path(raw).expanduser())
    return LocalMemoryBank(
        settings.db_path,
        extractor=default_extractor(settings),
        default_ttl=settings.ttl_seconds,
        top_k=settings.top_k,
    )
