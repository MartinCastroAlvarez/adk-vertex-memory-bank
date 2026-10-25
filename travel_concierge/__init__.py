"""travel_concierge: a Google ADK agent with long-term memory, built for Vertex AI Memory Bank.

``import travel_concierge`` builds nothing and imports no Google Cloud SDK. ADK's
loaders import ``travel_concierge.agent`` (which builds ``root_agent`` from the
environment); the ``agent`` attribute is resolved lazily so they find it.
"""

from __future__ import annotations

import importlib
from typing import Any

from travel_concierge.concierge import build_agent
from travel_concierge.config import Settings
from travel_concierge.memory.factory import build_memory_service
from travel_concierge.runner import ChatTurn
from travel_concierge.runner import Concierge

__all__ = ["ChatTurn", "Concierge", "Settings", "build_agent", "build_memory_service"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "agent":
        return importlib.import_module("travel_concierge.agent")
    raise AttributeError(f"module 'travel_concierge' has no attribute {name!r}")
