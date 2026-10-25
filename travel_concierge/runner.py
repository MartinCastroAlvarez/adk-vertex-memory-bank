"""Run the concierge: ADK ``Runner`` + a session service + the chosen memory service.

Sessions are short-term memory (one conversation, ``InMemorySessionService`` here);
the memory service is long-term memory (every conversation a user ever had). A new
``session_id`` is a new conversation; what carries over is only what the memory
service kept.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from dataclasses import field

from google.adk.agents.base_agent import BaseAgent
from google.adk.events.event import Event
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import BaseSessionService
from google.genai import types

from travel_concierge.config import APP_NAME
from travel_concierge.config import Settings
from travel_concierge.memory.base import MemoryAdmin
from travel_concierge.memory.base import check_scope
from travel_concierge.memory.base import entry_text


class MemoryUnsupportedError(RuntimeError):
    """The configured memory service can't do what was asked (list or forget a user)."""


@dataclass
class ChatTurn:
    """What one message produced."""

    reply: str
    session_id: str
    events: list[Event] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[str]:
        return [
            part.function_call.name or ""
            for event in self.events
            for part in (event.content.parts or [] if event.content else [])
            if part.function_call
        ]


class Concierge:
    """A chat front-end over one agent, one session service and one memory service."""

    def __init__(
        self,
        agent: BaseAgent | None = None,
        memory: BaseMemoryService | None = None,
        settings: Settings | None = None,
        sessions: BaseSessionService | None = None,
    ) -> None:
        settings = settings or Settings.from_env()
        if agent is None:
            from travel_concierge.concierge import build_agent

            agent = build_agent(settings)
        if memory is None:
            from travel_concierge.memory.factory import build_memory_service

            memory = build_memory_service(settings)
        self.agent = agent
        self.memory = memory
        self.sessions = sessions or InMemorySessionService()
        self.runner = Runner(
            agent=agent,
            app_name=APP_NAME,
            session_service=self.sessions,
            memory_service=memory,
        )

    def close(self) -> None:
        """Release the memory service's resources (the local SQLite file), if it has any."""
        close = getattr(self.memory, "close", None)
        if callable(close):
            close()

    async def chat(self, user_id: str, message: str, session_id: str | None = None) -> ChatTurn:
        """Send one message; a new ``session_id`` (the default) starts a new conversation."""
        check_scope(APP_NAME, user_id)
        session_id = session_id or uuid.uuid4().hex
        existing = await self.sessions.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        if existing is None:
            await self.sessions.create_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id
            )
        content = types.Content(role="user", parts=[types.Part(text=message)])
        events = [
            event
            async for event in self.runner.run_async(
                user_id=user_id, session_id=session_id, new_message=content
            )
        ]
        reply = ""
        for event in events:
            if event.author != "user" and event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts if not p.thought)
                reply = text or reply
        return ChatTurn(reply=reply.strip(), session_id=session_id, events=events)

    async def memories(self, user_id: str) -> list[str]:
        """Every live memory for ``user_id`` (needs a backend with ``list_memories``)."""
        entries = await self._list(user_id)
        return [entry_text(e) for e in entries]

    async def _list(self, user_id: str) -> list[MemoryEntry]:
        if not isinstance(self.memory, MemoryAdmin):
            found = await self.memory.search_memory(app_name=APP_NAME, user_id=user_id, query="")
            return list(found.memories)
        return await self.memory.list_memories(app_name=APP_NAME, user_id=user_id)

    async def forget(self, user_id: str) -> int | None:
        if not isinstance(self.memory, MemoryAdmin):
            raise MemoryUnsupportedError(f"{type(self.memory).__name__} cannot forget a whole user")
        return await self.memory.forget_user(app_name=APP_NAME, user_id=user_id)
