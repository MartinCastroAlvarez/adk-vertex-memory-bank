"""``CONCIERGE_MEMORY`` -> a ``BaseMemoryService``. The agent never sees which one."""

from __future__ import annotations

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.models.base_llm import BaseLlm

from travel_concierge.config import Settings
from travel_concierge.memory import topics as t
from travel_concierge.memory.extract import Extractor
from travel_concierge.memory.extract import LlmExtractor
from travel_concierge.memory.extract import RuleExtractor


def default_extractor(settings: Settings, llm: BaseLlm | None = None) -> Extractor:
    if settings.extractor == "rules":
        return RuleExtractor()
    if llm is None:
        from travel_concierge.models import extraction_model

        llm = extraction_model(settings)
    return LlmExtractor(llm)


def build_memory_service(
    settings: Settings, *, extractor: Extractor | None = None
) -> BaseMemoryService:
    """The backend ``settings.memory`` names.

    * ``local`` (default): :class:`~travel_concierge.memory.local.LocalMemoryBank` on
      ``settings.db_path``, extraction by the local model (or rules for the stub).
    * ``vertex``: :class:`~travel_concierge.memory.vertex.VertexMemoryBank` - a paid
      Google Cloud service; needs ``pip install "adk-vertex-memory-bank[vertex]"``,
      the three ``CONCIERGE_VERTEX_*`` settings and Application Default Credentials.
    * ``inmemory``: ADK's ``InMemoryMemoryService`` (raw events, keyword match, no
      extraction, nothing persisted), for comparison.
    """
    if settings.memory == "vertex":
        settings.vertex.validate()
        from travel_concierge.memory.vertex import VertexMemoryBank

        return VertexMemoryBank(
            project=settings.vertex.project,
            location=settings.vertex.location,
            agent_engine_id=settings.vertex.agent_engine_id,
            allowed_topics=t.TOPICS,
        )
    if settings.memory == "inmemory":
        from google.adk.memory import InMemoryMemoryService

        return InMemoryMemoryService()
    from travel_concierge.memory.local import LocalMemoryBank

    return LocalMemoryBank(
        settings.db_path,
        extractor=extractor or default_extractor(settings),
        default_ttl=settings.ttl_seconds,
        top_k=settings.top_k,
    )
