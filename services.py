"""ADK service registration, loaded by ``adk web`` / ``adk run`` from this folder.

Registers the ``localbank://`` memory URI so ADK's own CLIs can use the local Memory
Bank look-alike::

    adk web . --memory_service_uri=localbank://.adk/memory.sqlite3
    adk run travel_concierge --memory_service_uri=localbank://.adk/memory.sqlite3

``agentengine://<agent_engine_id>`` (ADK's built-in Vertex AI Memory Bank) needs no
registration - and a Google Cloud project that bills you.
"""

from __future__ import annotations

from typing import Any

from google.adk.cli.service_registry import get_service_registry

from travel_concierge.memory.local_uri import local_bank_from_uri


def _localbank(uri: str, **_: Any) -> Any:
    return local_bank_from_uri(uri)


get_service_registry().register_memory_service("localbank", _localbank)
