"""Long-term memory backends behind ADK's ``BaseMemoryService``.

``local`` (``LocalMemoryBank``) and ``vertex`` (``VertexMemoryBank``) are imported
from their modules on demand, so importing this package never loads the Vertex SDK.
"""

from __future__ import annotations

from travel_concierge.memory.base import MemoryAdmin
from travel_concierge.memory.factory import build_memory_service

__all__ = ["MemoryAdmin", "build_memory_service"]
