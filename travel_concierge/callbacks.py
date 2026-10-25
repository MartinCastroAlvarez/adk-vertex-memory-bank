"""The two callbacks that make memory work, and safe.

* :func:`save_to_memory` (``after_agent_callback``) hands the turn's new conversation
  events to whatever ``BaseMemoryService`` the runner has, via
  ``callback_context.add_events_to_memory``. A cursor in session state remembers how
  far the session was saved, so every event is sent once (Memory Bank and the local
  backend consolidate, but re-sending a whole session each turn is wasted work and,
  on Vertex, money). Right after a "forget me" nothing from the session is saved.
* :func:`memory_firewall` (``before_model_callback``) rewrites every recalled memory in
  the request - the ``<PAST_CONVERSATIONS>`` block ``preload_memory`` inserts and the
  results ``load_memory`` returns - through :func:`~travel_concierge.memory.sanitize.neutralise`,
  whichever backend produced them.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from travel_concierge.memory.base import entry_text
from travel_concierge.memory.base import is_conversation_event
from travel_concierge.memory.sanitize import neutralise

CURSOR_KEY = "memory_saved_events"
FORGET_KEY = "memory_forget_requested"
OPEN_TAG = "<PAST_CONVERSATIONS>"
CLOSE_TAG = "</PAST_CONVERSATIONS>"


async def save_to_memory(callback_context: CallbackContext) -> types.Content | None:
    """Write this turn's conversation to long-term memory (after every agent run)."""
    session = callback_context.session
    state = callback_context.state
    end = len(session.events)
    if state.get(FORGET_KEY):
        state[FORGET_KEY] = False
        state[CURSOR_KEY] = end
        return None
    start = int(state.get(CURSOR_KEY) or 0)
    fresh = [e for e in session.events[start:end] if is_conversation_event(e)]
    if fresh:
        await callback_context.add_events_to_memory(events=fresh)
    state[CURSOR_KEY] = end
    return None


def _neutralise_block(text: str) -> str:
    start = text.find(OPEN_TAG)
    # rfind: a stored memory may itself contain a fake closing tag.
    end = text.rfind(CLOSE_TAG)
    if start < 0 or end < start:
        return text
    body = text[start + len(OPEN_TAG) : end]
    lines = []
    for line in body.splitlines():
        if not line.strip():
            continue
        prefix, sep, rest = line.partition(": ")
        if prefix in {"user", "Time"} and sep:
            lines.append(f"{prefix}: {rest if prefix == 'Time' else neutralise(rest)}")
        else:
            lines.append(neutralise(line))
    return text[:start] + OPEN_TAG + "\n" + "\n".join(lines) + "\n" + text[end:]


def _memory_text(memory: Any) -> tuple[str, str | None]:
    """(text, timestamp) of a recalled memory, whether a ``MemoryEntry`` or its dict form."""
    if isinstance(memory, MemoryEntry):
        return entry_text(memory), memory.timestamp
    if isinstance(memory, dict):
        parts = (memory.get("content") or {}).get("parts") or []
        text = " ".join(p.get("text") or "" for p in parts if isinstance(p, dict))
        return text.strip(), memory.get("timestamp")
    return str(memory), None


def _neutralise_response(response: dict[str, Any]) -> dict[str, Any]:
    """``load_memory``'s result (``{"result": LoadMemoryResponse}``) as plain, safe data."""
    result = response.get("result", response)
    memories = getattr(result, "memories", None)
    if memories is None and isinstance(result, dict):
        memories = result.get("memories")
    if not isinstance(memories, list):
        return response
    clean = []
    for memory in memories:
        text, when = _memory_text(memory)
        clean.append({"memory": neutralise(text), **({"time": when} if when else {})})
    return {"memories": clean}


def memory_firewall(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Recalled memories reach the model as neutralised data, never as instructions."""
    for content in llm_request.contents:
        for part in content.parts or []:
            if part.text and OPEN_TAG in part.text:
                part.text = _neutralise_block(part.text)
            response = part.function_response
            if response and response.name == "load_memory" and response.response:
                response.response = _neutralise_response(dict(response.response))
    return None
