"""An in-process fake of the Vertex AI Memory Bank client. No network, no credentials.

It has the shape ADK's ``VertexAiMemoryBankService`` talks to -
``client.memory_banks.memories.{generate, create, retrieve, purge, delete}`` - and
returns the SDK's own ``agentplatform.types`` objects, so the real ADK code that
builds requests and parses responses runs unchanged against it. It behaves like the
service the docs describe, closely enough for a contract test:

* ``generate`` extracts facts from user turns (its "server-side model" is the rule
  extractor) and consolidates per scope: a new value for a slot updates the memory,
  "forget my ..." deletes it; ``allowed_topics`` filters what is kept;
* scopes match exactly, like Memory Bank's;
* the bank's ``default_ttl`` (``ttl_config``) sets ``expire_time``; expired memories
  are not returned;
* ``purge`` understands the ``scope.<key>="..." AND ...`` filter and honours ``force``.

Every call is recorded in ``calls`` so tests can assert the exact request.
"""

from __future__ import annotations

import datetime
import re
import uuid
from collections.abc import AsyncIterator
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from agentplatform import types as vt

from travel_concierge.memory import topics as t
from travel_concierge.memory.extract import rule_candidates

ENGINE_ID = "1234567890"
_FILTER_TERM = re.compile(r'^scope\.([a-z_]+)="([^"\\]*)"$')


def _dump(value: Any) -> Any:
    return value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else value


def _topic_label(topic: Any) -> str:
    topic = _dump(topic)
    label = topic.get("managed_memory_topic") or topic.get("custom_memory_topic_label") or ""
    return str(getattr(label, "value", label))


@dataclass
class _Stored:
    memory: vt.Memory
    slot: str | None


class _Pager:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        for item in self._items:
            yield item


@dataclass
class FakeMemoryBank:
    """The fake client. Pass it as ``VertexMemoryBank(..., client=FakeMemoryBank())``."""

    engine_id: str = ENGINE_ID
    default_ttl: float | None = None
    clock: Callable[[], float] = field(
        default=lambda: datetime.datetime.now(datetime.UTC).timestamp()
    )
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    store: dict[str, _Stored] = field(default_factory=dict)

    # The client surface ADK reads: client.memory_banks.memories
    @property
    def memory_banks(self) -> FakeMemoryBank:
        return self

    @property
    def memories(self) -> FakeMemoryBank:
        return self

    # ------------------------------------------------------------------ helpers
    @property
    def bank(self) -> str:
        return f"reasoningEngines/{self.engine_id}"

    def _now(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.clock(), tz=datetime.UTC)

    def _check_name(self, name: str) -> None:
        if not name.startswith(self.bank):
            raise ValueError(f"unknown Memory Bank {name!r}")

    def _expire(self, ttl: float | None) -> datetime.datetime | None:
        return self._now() + datetime.timedelta(seconds=ttl) if ttl else None

    def _live(self, scope: dict[str, str]) -> list[_Stored]:
        now = self._now()
        return [
            s
            for s in self.store.values()
            if s.memory.scope == scope
            and (s.memory.expire_time is None or s.memory.expire_time > now)
        ]

    def _new(
        self,
        fact: str,
        scope: dict[str, str],
        topic: str,
        slot: str | None,
        ttl: float | None,
        memory_id: str | None = None,
    ) -> vt.Memory:
        now = self._now()
        memory_id = memory_id or uuid.uuid4().hex
        topic_id = (
            vt.MemoryTopicId(managed_memory_topic=topic)
            if topic in t.MANAGED_TOPICS
            else vt.MemoryTopicId(custom_memory_topic_label=topic)
        )
        memory = vt.Memory(
            name=f"projects/demo/locations/us-central1/{self.bank}/memories/{memory_id}",
            fact=fact,
            scope=dict(scope),
            topics=[topic_id],
            create_time=now,
            update_time=now,
            expire_time=self._expire(ttl),
        )
        self.store[memory_id] = _Stored(memory, slot)
        return memory

    # ------------------------------------------------------------------ API
    async def generate(
        self,
        *,
        name: str,
        scope: dict[str, str] | None = None,
        config: Any = None,
        direct_contents_source: Any = None,
        direct_memories_source: Any = None,
        vertex_session_source: Any = None,
    ) -> vt.GenerateMemoriesOperation:
        self.calls.append(
            (
                "generate",
                {
                    "name": name,
                    "scope": scope,
                    "config": _dump(config),
                    "direct_contents_source": _dump(direct_contents_source),
                    "direct_memories_source": _dump(direct_memories_source),
                },
            )
        )
        self._check_name(name)
        if scope is None or vertex_session_source is not None:
            raise ValueError("this fake supports direct sources with an explicit scope only")
        config = _dump(config) or {}
        allowed = {_topic_label(x) for x in config.get("allowed_topics") or []}
        texts: list[str] = []
        for event in (_dump(direct_contents_source) or {}).get("events", []):
            content = event.get("content") or {}
            if content.get("role", "user") == "user":
                texts.extend(p["text"] for p in content.get("parts", []) if p.get("text"))
        for direct in (_dump(direct_memories_source) or {}).get("direct_memories", []):
            texts.append(direct["fact"])
        generated: list[vt.GenerateMemoriesResponseGeneratedMemory] = []
        for text in texts:
            for c in rule_candidates(text):
                if allowed and c.topic not in allowed and not c.forget:
                    continue
                live = self._live(scope)
                same = [s for s in live if c.slot and s.slot == c.slot] or [
                    s for s in live if s.memory.fact == c.fact
                ]
                if c.forget:
                    for s in same:
                        del self.store[s.memory.name.rsplit("/", 1)[-1]]
                        generated.append(
                            vt.GenerateMemoriesResponseGeneratedMemory(
                                memory=s.memory, action="DELETED"
                            )
                        )
                    continue
                if same:
                    s = same[0]
                    s.memory.fact, s.memory.update_time = c.fact, self._now()
                    s.memory.expire_time = self._expire(self.default_ttl)
                    generated.append(
                        vt.GenerateMemoriesResponseGeneratedMemory(
                            memory=s.memory, action="UPDATED"
                        )
                    )
                else:
                    memory = self._new(c.fact, scope, c.topic, c.slot, self.default_ttl)
                    generated.append(
                        vt.GenerateMemoriesResponseGeneratedMemory(memory=memory, action="CREATED")
                    )
        return vt.GenerateMemoriesOperation(
            name="operations/generate",
            done=True,
            response=vt.GenerateMemoriesResponse(generated_memories=generated),
        )

    async def create(
        self, *, name: str, fact: str, scope: dict[str, str], config: Any = None
    ) -> vt.MemoryOperation:
        config = _dump(config) or {}
        self.calls.append(
            ("create", {"name": name, "fact": fact, "scope": scope, "config": config})
        )
        self._check_name(name)
        ttl = config.get("ttl")
        seconds = float(str(ttl).rstrip("s")) if ttl else self.default_ttl
        topics = config.get("topics") or []
        topic = _topic_label(topics[0]) if topics else t.USER_PREFERENCES
        memory = self._new(fact, scope, topic, None, seconds, config.get("memory_id"))
        return vt.MemoryOperation(name="operations/create", done=True, response=memory)

    async def retrieve(
        self,
        *,
        name: str,
        scope: dict[str, str],
        similarity_search_params: Any = None,
        simple_retrieval_params: Any = None,
        config: Any = None,
    ) -> _Pager:
        params = _dump(similarity_search_params) or {}
        self.calls.append(
            ("retrieve", {"name": name, "scope": scope, "similarity_search_params": params})
        )
        self._check_name(name)
        live = self._live(scope)
        query = set(re.findall(r"[a-z]+", str(params.get("search_query", "")).lower()))

        def distance(s: _Stored) -> float:
            words = set(re.findall(r"[a-z]+", (s.memory.fact or "").lower()))
            return 1.0 - len(words & query) / (len(words | query) or 1)

        ranked = sorted(live, key=distance) if params else live
        top_k = int(params.get("top_k") or 10) if params else len(ranked)
        return _Pager(
            [
                vt.RetrieveMemoriesResponseRetrievedMemory(memory=s.memory, distance=distance(s))
                for s in ranked[:top_k]
            ]
        )

    async def purge(
        self,
        *,
        name: str,
        filter: str | None = None,
        filter_groups: Any = None,
        force: bool = False,
        config: Any = None,
    ) -> vt.PurgeMemoriesOperation:
        self.calls.append(
            ("purge", {"name": name, "filter": filter, "force": force, "config": _dump(config)})
        )
        self._check_name(name)
        wanted: dict[str, str] = {}
        for term in (filter or "").split(" AND "):
            m = _FILTER_TERM.match(term.strip())
            if not m:
                raise ValueError(f"unsupported purge filter term {term!r}")
            wanted[m.group(1)] = m.group(2)
        doomed = [
            k
            for k, s in self.store.items()
            if all((s.memory.scope or {}).get(k2) == v for k2, v in wanted.items())
        ]
        if force:
            for key in doomed:
                del self.store[key]
        return vt.PurgeMemoriesOperation(
            name="operations/purge",
            done=True,
            response=vt.PurgeMemoriesResponse(purge_count=len(doomed)),
        )

    async def delete(self, *, name: str, config: Any = None) -> vt.DeleteMemoryOperation:
        self.calls.append(("delete", {"name": name}))
        self._check_name(name)
        self.store.pop(name.rsplit("/", 1)[-1], None)
        return vt.DeleteMemoryOperation(name="operations/delete", done=True)

    async def ingest_events(self, **kwargs: Any) -> None:
        self.calls.append(("ingest_events", kwargs))
        raise AssertionError("VertexMemoryBank must use memories.generate, not ingest_events")

    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]
