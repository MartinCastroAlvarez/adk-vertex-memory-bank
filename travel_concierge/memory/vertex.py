"""``VertexMemoryBank``: ADK's ``VertexAiMemoryBankService`` plus "forget me".

**This talks to a paid Google Cloud service.** Nothing in this repository's tests,
demo or CI constructs it against Google Cloud: the tests inject a fake client
(``client=``), and this module never imports ``agentplatform`` or ``google.auth``
until a real client is built on the first memory call.

What it adds to ADK's service (whose own code does the Memory Bank calls):

* ``add_session_to_memory`` and ``add_events_to_memory`` go through
  ``memories.generate`` (extraction + consolidation, the documented path) rather
  than the fire-and-forget ``ingest_events`` task, so a write either finishes or
  raises inside the callback that asked for it; ``allowed_topics`` narrows what the
  bank extracts. Both are plain ``custom_metadata`` keys ADK forwards.
* ``forget_user`` purges the ``{app_name, user_id}`` scope with
  ``memories.purge(filter='scope.app_name="..." AND scope.user_id="..."', force=True)``.
  Scope values are validated first, so a user id can't rewrite the filter.
* ``list_memories`` reads the whole scope with ``memories.retrieve(scope=...)``.

TTL lives on the Memory Bank itself (``context_spec.memory_bank_config.ttl_config``,
see docs/vertex-setup.md): ``memories.generate`` only takes a *revision* TTL.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any

from google.adk.memory.memory_entry import MemoryEntry
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.genai import types

from travel_concierge.memory import topics as t
from travel_concierge.memory.base import check_scope

if TYPE_CHECKING:
    from google.adk.events.event import Event
    from google.adk.sessions.session import Session


def topic_ids(topics: Sequence[str]) -> list[dict[str, str]]:
    """Topic names -> Memory Bank ``MemoryTopicId`` dicts (managed enum or custom label)."""
    return [
        {"managed_memory_topic": name}
        if name in t.MANAGED_TOPICS
        else {"custom_memory_topic_label": name}
        for name in topics
    ]


def purge_filter(app_name: str, user_id: str) -> str:
    check_scope(app_name, user_id)
    return f'scope.app_name="{app_name}" AND scope.user_id="{user_id}"'


class VertexMemoryBank(VertexAiMemoryBankService):
    """Vertex AI Agent Engine Memory Bank behind ``BaseMemoryService``."""

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        agent_engine_id: str | None = None,
        *,
        allowed_topics: Sequence[str] = (),
        wait_for_completion: bool = True,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(project, location, agent_engine_id, **kwargs)
        self.allowed_topics = tuple(allowed_topics)
        self.wait_for_completion = wait_for_completion
        self._client = client

    @property
    def bank_name(self) -> str:
        return f"reasoningEngines/{self._agent_engine_id}"

    def _build_api_client(self) -> Any:
        if self._client is not None:
            return self._client
        return super()._build_api_client()  # pragma: no cover - real Google Cloud only

    def _memories(self) -> Any:
        client = self._get_api_client()
        banks = getattr(client, "memory_banks", None)
        return banks.memories if banks is not None else client.agent_engines.memories

    async def add_session_to_memory(self, session: Session) -> None:
        await self.add_events_to_memory(
            app_name=session.app_name,
            user_id=session.user_id,
            events=session.events,
            session_id=session.id,
        )

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        check_scope(app_name, user_id)
        # Any GenerateMemories-only key routes ADK to memories.generate.
        metadata: dict[str, object] = {"wait_for_completion": self.wait_for_completion}
        if self.allowed_topics:
            metadata["allowed_topics"] = topic_ids(self.allowed_topics)
        metadata.update(custom_metadata or {})
        await super().add_events_to_memory(
            app_name=app_name,
            user_id=user_id,
            events=events,
            session_id=session_id,
            custom_metadata=metadata,
        )

    async def forget_user(self, *, app_name: str, user_id: str) -> int | None:
        operation = await self._memories().purge(
            name=self.bank_name,
            filter=purge_filter(app_name, user_id),
            force=True,
            config={"wait_for_completion": True},
        )
        response = getattr(operation, "response", None)
        count = getattr(response, "purge_count", None)
        return int(count) if count is not None else None

    async def list_memories(self, *, app_name: str, user_id: str) -> list[MemoryEntry]:
        check_scope(app_name, user_id)
        pager = await self._memories().retrieve(
            name=self.bank_name, scope={"app_name": app_name, "user_id": user_id}
        )
        out: list[MemoryEntry] = []
        async for item in pager:
            memory = getattr(item, "memory", None)
            if memory is None or not memory.fact:
                continue
            labels = [
                topic.managed_memory_topic or topic.custom_memory_topic_label
                for topic in memory.topics or []
            ]
            out.append(
                MemoryEntry(
                    id=(memory.name or "").rsplit("/", 1)[-1] or None,
                    author="user",
                    content=types.Content(role="user", parts=[types.Part(text=memory.fact)]),
                    timestamp=memory.update_time.isoformat() if memory.update_time else None,
                    custom_metadata={
                        "topic": ",".join(str(getattr(x, "value", x)) for x in labels if x)
                    },
                )
            )
        return out
