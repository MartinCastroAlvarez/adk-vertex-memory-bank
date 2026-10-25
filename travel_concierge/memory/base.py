"""What the concierge needs from a memory backend beyond ADK's ``BaseMemoryService``.

The agent itself only talks to ``BaseMemoryService`` (through ``preload_memory``,
``load_memory`` and ``callback_context.add_events_to_memory``). Two operations are not
part of that interface and both backends add them:

* ``forget_user`` - delete everything stored for one user in one app ("forget me");
* ``list_memories`` - every live memory for one user, for the CLI and the tests.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol
from typing import runtime_checkable

from google.adk.events.event import Event
from google.adk.memory.memory_entry import MemoryEntry

from travel_concierge.memory.extract import Turn

#: The characters a scope value may use: it ends up inside a Memory Bank filter string.
SCOPE_VALUE = re.compile(r"^[A-Za-z0-9._@:+-]{1,128}$")


class ScopeError(ValueError):
    """An app name or user id that cannot be used as a memory scope."""


def check_scope(app_name: str, user_id: str) -> None:
    for label, value in (("app_name", app_name), ("user_id", user_id)):
        if not SCOPE_VALUE.match(value or ""):
            raise ScopeError(f"{label} must match {SCOPE_VALUE.pattern}, got {value!r}")


@runtime_checkable
class MemoryAdmin(Protocol):
    async def forget_user(self, *, app_name: str, user_id: str) -> int | None: ...

    async def list_memories(self, *, app_name: str, user_id: str) -> list[MemoryEntry]: ...


def entry_text(entry: MemoryEntry) -> str:
    """The text of a memory entry (its text parts joined)."""
    parts = entry.content.parts or []
    return " ".join(p.text for p in parts if p.text).strip()


def is_conversation_event(event: Event) -> bool:
    """User and agent text only: tool calls, tool results and empty events are not memory.

    Re-ingesting a ``load_memory`` result would store recalled memories a second time,
    and a ``forget_me`` call is not a fact about the user.
    """
    if not event.content or not event.content.parts:
        return False
    if any(p.function_call or p.function_response for p in event.content.parts):
        return False
    return any(p.text and not p.thought for p in event.content.parts)


def turns_from_events(events: Iterable[Event]) -> list[Turn]:
    turns = []
    for event in events:
        if not is_conversation_event(event) or event.content is None:
            continue
        text = " ".join(p.text for p in event.content.parts or [] if p.text and not p.thought)
        role = "user" if event.author == "user" else "model"
        turns.append(Turn(role=role, text=text))
    return turns


_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s\s*$")


def parse_duration(value: object) -> float | None:
    """``"3600s"`` (the protobuf Duration JSON form Memory Bank uses), or a number."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("a TTL is a duration like '3600s', not a bool")
    if isinstance(value, int | float):
        seconds = float(value)
    else:
        m = _DURATION.match(str(value))
        if not m:
            raise ValueError(f"TTL must look like '3600s', got {value!r}")
        seconds = float(m.group(1))
    if seconds <= 0:
        raise ValueError("TTL must be positive")
    return seconds
