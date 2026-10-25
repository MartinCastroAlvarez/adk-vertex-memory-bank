"""ADK entry point: ``root_agent`` for ``adk web`` and ``adk run``.

The model comes from ``CONCIERGE_PROVIDER`` (``ollama`` by default, ``stub`` offline,
``gemini`` with your own key). The memory service is the runner's: pass
``--memory_service_uri=localbank://<path>`` to use this repo's local Memory Bank
(registered in the repo root's ``services.py``), or ``agentengine://<id>`` for Vertex.
"""

from __future__ import annotations

from travel_concierge.concierge import build_agent
from travel_concierge.config import Settings

root_agent = build_agent(Settings.from_env())
