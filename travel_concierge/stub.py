"""Stub models: ``BaseLlm`` subclasses that never touch a network.

* :class:`ScriptedLlm` plays a fixed list of replies, one per model call, and records
  every request it got. Tests give each agent its own script.
* :class:`travel_concierge.offline.OfflineConcierge` (built on :class:`StubLlm`) computes
  its replies from the request, so the whole concierge runs offline.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel
from pydantic import Field


@dataclass(frozen=True)
class ToolCall:
    """One function call in a scripted reply."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


#: A scripted reply: text, a JSON-able dict/model (sent as text), or tool calls.
Reply = str | dict[str, Any] | BaseModel | ToolCall | list[ToolCall]


def system_text(request: LlmRequest) -> str:
    system = request.config.system_instruction if request.config else None
    return system if isinstance(system, str) else ""


def last_function_responses(request: LlmRequest) -> list[tuple[str, dict[str, Any]]]:
    """``(name, response)`` pairs in the newest content, if it is a tool result."""
    if not request.contents:
        return []
    found = []
    for part in request.contents[-1].parts or []:
        if part.function_response and part.function_response.name:
            found.append((part.function_response.name, dict(part.function_response.response or {})))
    return found


def to_parts(reply: Reply) -> list[types.Part]:
    if isinstance(reply, ToolCall):
        reply = [reply]
    if isinstance(reply, list):
        return [
            types.Part(function_call=types.FunctionCall(name=c.name, args=dict(c.args)))
            for c in reply
        ]
    if isinstance(reply, BaseModel):
        return [types.Part(text=reply.model_dump_json())]
    if isinstance(reply, dict):
        return [types.Part(text=json.dumps(reply))]
    return [types.Part(text=reply)]


class StubLlm(BaseLlm):
    """Base for offline models: declares its capabilities and builds responses."""

    @property
    def capabilities(self) -> LlmCapabilities:
        # Like LiteLlm on ollama: output_schema + tools go through set_model_response.
        return LlmCapabilities(output_schema_and_tools=False)

    def reply(self, request: LlmRequest) -> Reply:  # pragma: no cover - abstract
        raise NotImplementedError

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        parts = to_parts(self.reply(llm_request))
        prompt = len(system_text(llm_request)) // 4
        output = sum(len(p.text or "") for p in parts) // 4
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=prompt,
                candidates_token_count=output,
                total_token_count=prompt + output,
            ),
        )


class ScriptedLlm(StubLlm):
    """Replies from ``script`` in order, then repeats the last one if ``repeat_last``.

    ``requests`` keeps every :class:`LlmRequest` it received, so a test can assert
    what an agent was shown (its rendered instruction, its tools, its contents).
    """

    model: str = "scripted"
    script: list[Reply] = Field(default_factory=list)
    repeat_last: bool = False
    requests: list[LlmRequest] = Field(default_factory=list)

    @property
    def calls(self) -> int:
        return len(self.requests)

    def reply(self, request: LlmRequest) -> Reply:
        self.requests.append(request.model_copy(deep=True))
        index = len(self.requests) - 1
        if index < len(self.script):
            return self.script[index]
        if self.repeat_last and self.script:
            return self.script[-1]
        raise AssertionError(f"{self.model}: script exhausted after {len(self.script)} replies")
