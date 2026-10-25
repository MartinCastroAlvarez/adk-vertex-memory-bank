"""VertexMemoryBank: the exact Memory Bank requests ADK sends, against the fake client."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest
from google.adk.events.event import Event
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types

from tests.conftest import Backend
from tests.conftest import make_backend
from tests.fake_memory_bank import ENGINE_ID
from tests.fake_memory_bank import FakeMemoryBank
from travel_concierge.config import APP_NAME
from travel_concierge.config import ConfigError
from travel_concierge.config import Settings
from travel_concierge.config import VertexSettings
from travel_concierge.memory.factory import build_memory_service
from travel_concierge.memory.vertex import VertexMemoryBank
from travel_concierge.memory.vertex import purge_filter
from travel_concierge.memory.vertex import topic_ids

BANK = f"reasoningEngines/{ENGINE_ID}"


@pytest.fixture
def vertex() -> Backend:
    return make_backend("vertex")


def fake_of(backend: Backend) -> FakeMemoryBank:
    assert backend.fake is not None
    return backend.fake


async def test_a_turn_is_one_generate_call_with_scope_topics_and_text_only(vertex: Backend) -> None:
    await vertex.concierge().chat("alice", "I always want an aisle seat.")
    fake = fake_of(vertex)
    generate = [kw for name, kw in fake.calls if name == "generate"]
    assert len(generate) == 1
    call = generate[0]
    assert call["name"] == BANK
    assert call["scope"] == {"app_name": APP_NAME, "user_id": "alice"}
    assert call["config"]["wait_for_completion"] is True
    labels = [
        t.get("managed_memory_topic") or t.get("custom_memory_topic_label")
        for t in call["config"]["allowed_topics"]
    ]
    assert "travel_preferences" in labels and "USER_PREFERENCES" in [str(x) for x in labels]
    events = call["direct_contents_source"]["events"]
    roles = [e["content"]["role"] for e in events]
    assert roles == ["user", "model"]
    assert events[0]["content"]["parts"][0]["text"] == "I always want an aisle seat."
    assert "ingest_events" not in fake.call_names()


async def test_recall_is_a_similarity_retrieve_in_the_users_scope(vertex: Backend) -> None:
    concierge = vertex.concierge()
    await concierge.chat("alice", "I always want an aisle seat.")
    await concierge.chat("alice", "Plan Lisbon.")
    retrieve = [kw for name, kw in fake_of(vertex).calls if name == "retrieve"]
    assert retrieve[-1] == {
        "name": BANK,
        "scope": {"app_name": APP_NAME, "user_id": "alice"},
        "similarity_search_params": {"search_query": "Plan Lisbon."},
    }


async def test_only_new_events_are_sent_each_turn(vertex: Backend) -> None:
    concierge = vertex.concierge()
    first = await concierge.chat("alice", "I always want an aisle seat.")
    await concierge.chat("alice", "My home airport is SFO.", session_id=first.session_id)
    generate = [kw for name, kw in fake_of(vertex).calls if name == "generate"]
    texts = [
        [e["content"]["parts"][0]["text"] for e in kw["direct_contents_source"]["events"]]
        for kw in generate
    ]
    assert texts[1][0] == "My home airport is SFO."
    assert "I always want an aisle seat." not in texts[1]


async def test_forget_user_is_a_forced_scoped_purge(vertex: Backend) -> None:
    service = vertex.service
    assert isinstance(service, VertexMemoryBank)
    await vertex.concierge().chat("alice", "I always want an aisle seat.")
    count = await service.forget_user(app_name=APP_NAME, user_id="alice")
    assert count == 1
    purge = [kw for name, kw in fake_of(vertex).calls if name == "purge"][-1]
    assert purge == {
        "name": BANK,
        "filter": f'scope.app_name="{APP_NAME}" AND scope.user_id="alice"',
        "force": True,
        "config": {"wait_for_completion": True},
    }


async def test_list_memories_maps_ids_topics_and_times(vertex: Backend) -> None:
    service = vertex.service
    assert isinstance(service, VertexMemoryBank)
    await vertex.concierge().chat("alice", "I always want an aisle seat.")
    [entry] = await service.list_memories(app_name=APP_NAME, user_id="alice")
    assert entry.id and len(entry.id) == 32
    assert entry.custom_metadata["topic"] == "travel_preferences"
    assert entry.timestamp and entry.timestamp.startswith("2027-")
    assert entry.author == "user"


async def test_add_memory_uses_memories_create_with_ttl(vertex: Backend) -> None:
    memory = MemoryEntry(
        content=types.Content(role="user", parts=[types.Part(text="Likes trains")])
    )
    await vertex.service.add_memory(
        app_name=APP_NAME, user_id="alice", memories=[memory], custom_metadata={"ttl": "60s"}
    )
    create = [kw for name, kw in fake_of(vertex).calls if name == "create"][-1]
    assert create["fact"] == "Likes trains"
    assert create["config"]["ttl"] == "60s"
    vertex.clock.advance(61)
    found = await vertex.service.search_memory(app_name=APP_NAME, user_id="alice", query="trains")
    assert found.memories == []


async def test_events_without_text_send_nothing(vertex: Backend) -> None:
    event = Event(author="user", content=types.Content(role="user", parts=[]))
    await vertex.service.add_events_to_memory(app_name=APP_NAME, user_id="alice", events=[event])
    assert "generate" not in fake_of(vertex).call_names()


def test_topic_ids_and_purge_filter() -> None:
    assert topic_ids(["USER_PREFERENCES", "travel_preferences"]) == [
        {"managed_memory_topic": "USER_PREFERENCES"},
        {"custom_memory_topic_label": "travel_preferences"},
    ]
    assert purge_filter("app", "u-1") == 'scope.app_name="app" AND scope.user_id="u-1"'


def test_the_injected_client_is_used_and_no_credentials_are_looked_up() -> None:
    fake = FakeMemoryBank()
    service = VertexMemoryBank("demo-project", "us-central1", ENGINE_ID, client=fake)
    assert service._build_api_client() is fake
    assert service.bank_name == BANK


def test_factory_builds_vertex_only_with_complete_valid_settings() -> None:
    vertex = VertexSettings("demo-project", "us-central1", ENGINE_ID)
    service = build_memory_service(
        Settings(provider="stub", extractor="rules", memory="vertex", vertex=vertex)
    )
    assert isinstance(service, VertexMemoryBank)
    assert service.allowed_topics[-1] == "travel_preferences"
    with pytest.raises(ConfigError, match="CONCIERGE_VERTEX_PROJECT"):
        Settings(memory="vertex")
    for bad in (
        replace(vertex, agent_engine_id=f"projects/p/locations/l/reasoningEngines/{ENGINE_ID}"),
        replace(vertex, project="Not A Project"),
        replace(vertex, location="moon"),
    ):
        with pytest.raises(ConfigError):
            Settings(memory="vertex", vertex=bad)


def test_importing_the_package_loads_no_cloud_sdk_and_no_credentials() -> None:
    code = (
        "import sys, travel_concierge, travel_concierge.memory, travel_concierge.agent;"
        "assert 'agentplatform' not in sys.modules, 'agentplatform imported';"
        "assert 'travel_concierge.memory.vertex' not in sys.modules;"
        "print('ok')"
    )
    env = {"CONCIERGE_PROVIDER": "stub", "PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
