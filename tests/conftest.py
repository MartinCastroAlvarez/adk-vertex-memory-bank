"""Shared fixtures: two memory backends behind one interface, a fake clock, stub models."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import pytest
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.models.llm_request import LlmRequest
from pydantic import Field

from tests.fake_memory_bank import ENGINE_ID
from tests.fake_memory_bank import FakeMemoryBank
from travel_concierge.concierge import build_agent
from travel_concierge.config import Settings
from travel_concierge.memory.local import LocalMemoryBank
from travel_concierge.offline import OfflineConcierge
from travel_concierge.runner import Concierge
from travel_concierge.stub import Reply

T0 = 1_800_000_000.0
DAY = 86400.0


@pytest.fixture(autouse=True)
def _no_google_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to find Google credentials fails the test: nothing here may authenticate."""
    import google.auth
    import google.auth._default

    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("google.auth.default() was called: tests must never authenticate")

    monkeypatch.setattr(google.auth, "default", refuse)
    monkeypatch.setattr(google.auth._default, "default", refuse)
    for name in ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_AGENT_ENGINE_ID"):
        monkeypatch.delenv(name, raising=False)


@dataclass
class Clock:
    now: float = T0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingConcierge(OfflineConcierge):
    """The offline concierge, keeping every request it was sent."""

    requests: list[LlmRequest] = Field(default_factory=list)

    def reply(self, request: LlmRequest) -> Reply:
        self.requests.append(request.model_copy(deep=True))
        return super().reply(request)

    def prompt_text(self, index: int = -1) -> str:
        """Everything the model saw in one request, as text (contents + tool results)."""
        request = self.requests[index]
        chunks = [str(request.config.system_instruction or "")] if request.config else []
        for content in request.contents:
            for part in content.parts or []:
                chunks.append(part.text or "")
                if part.function_response:
                    chunks.append(str(part.function_response.response))
        return "\n".join(chunks)


@dataclass
class Backend:
    """One memory backend under test plus the knobs a contract test needs."""

    kind: str
    service: BaseMemoryService
    clock: Clock
    fake: FakeMemoryBank | None = None
    model: RecordingConcierge = field(default_factory=RecordingConcierge)

    def concierge(self) -> Concierge:
        settings = Settings(provider="stub", extractor="rules")
        agent = build_agent(settings, model=self.model)
        return Concierge(agent=agent, memory=self.service, settings=settings)


def make_backend(kind: str, ttl: float | None = 365 * DAY) -> Backend:
    clock = Clock()
    if kind == "local":
        return Backend(kind, LocalMemoryBank(":memory:", default_ttl=ttl, clock=clock), clock)
    from travel_concierge.memory.vertex import VertexMemoryBank

    fake = FakeMemoryBank(default_ttl=ttl, clock=clock)
    service = VertexMemoryBank(
        project="demo-project",
        location="us-central1",
        agent_engine_id=ENGINE_ID,
        allowed_topics=(
            "USER_PREFERENCES",
            "USER_PERSONAL_INFO",
            "EXPLICIT_INSTRUCTIONS",
            "KEY_CONVERSATION_DETAILS",
            "travel_preferences",
        ),
        client=fake,
    )
    return Backend(kind, service, clock, fake)


@pytest.fixture(params=["local", "vertex"])
def backend(request: pytest.FixtureRequest) -> Backend:
    """Every contract test runs twice: LocalMemoryBank, and VertexMemoryBank on a fake client."""
    return make_backend(request.param)


@pytest.fixture
def settings() -> Settings:
    return Settings(provider="stub", extractor="rules")
