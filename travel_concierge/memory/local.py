"""``LocalMemoryBank``: Memory Bank semantics on SQLite, for free, on your machine.

What it copies from Vertex AI Agent Engine Memory Bank:

* **Memory generation** - ``add_session_to_memory`` / ``add_events_to_memory`` run an
  extractor (an LLM on ollama, or rules offline) over the conversation and keep
  facts under topics (managed ``USER_PREFERENCES`` etc. plus a custom
  ``travel_preferences`` topic).
* **Consolidation** - each candidate is merged with what the user already has: a
  new value for a slot *updates* the old fact (seat: aisle -> window), a
  near-duplicate updates instead of duplicating, an explicit "forget my seat
  preference" or "I'm no longer vegan" *deletes*. Every change writes a revision.
* **Scopes** - memories belong to ``{app_name, user_id}`` and a search only ever sees
  its own scope, exactly like Memory Bank's exact-match scope.
* **Similarity retrieval** - ``search_memory`` returns the ``top_k`` memories of the
  scope ranked by BM25 then recency (see :mod:`travel_concierge.memory.search`).
* **TTL** - every create or update sets ``expire_at`` (``default_ttl``), a direct write
  can pass its own ``ttl``; expired memories are invisible and purged on write.
* **Direct writes** - ``add_memory`` stores facts as given (Memory Bank's
  ``memories.create``) or consolidates them with ``enable_consolidation=True``.

Plus ``forget_user`` (memories *and* revisions), ``delete_memory`` and
``list_memories``. SQL is parameterised; the file is created with mode 0600.
"""

from __future__ import annotations

import datetime
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types

from travel_concierge.memory import topics as t
from travel_concierge.memory.base import check_scope
from travel_concierge.memory.base import entry_text
from travel_concierge.memory.base import parse_duration
from travel_concierge.memory.base import turns_from_events
from travel_concierge.memory.extract import Candidate
from travel_concierge.memory.extract import Extractor
from travel_concierge.memory.extract import RuleExtractor
from travel_concierge.memory.sanitize import clean_fact
from travel_concierge.memory.sanitize import looks_like_instruction
from travel_concierge.memory.search import jaccard
from travel_concierge.memory.search import rank

if TYPE_CHECKING:
    from google.adk.events.event import Event
    from google.adk.sessions.session import Session

Action = Literal["CREATED", "UPDATED", "DELETED", "SKIPPED", "REJECTED"]
DUPLICATE_THRESHOLD = 0.7
FORGET_THRESHOLD = 0.4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    topic TEXT NOT NULL,
    slot TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    expire_at REAL
);
CREATE INDEX IF NOT EXISTS memories_scope ON memories (app_name, user_id);
CREATE TABLE IF NOT EXISTS revisions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    fact TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS revisions_scope ON revisions (app_name, user_id);
"""


@dataclass(frozen=True)
class StoredMemory:
    id: str
    app_name: str
    user_id: str
    fact: str
    topic: str
    slot: str | None
    created_at: float
    updated_at: float
    expire_at: float | None

    def to_entry(self) -> MemoryEntry:
        when = datetime.datetime.fromtimestamp(self.updated_at, tz=datetime.UTC)
        return MemoryEntry(
            id=self.id,
            author="user",
            content=types.Content(role="user", parts=[types.Part(text=self.fact)]),
            timestamp=when.isoformat(),
            custom_metadata={"topic": self.topic, "slot": self.slot or ""},
        )


@dataclass(frozen=True)
class Generated:
    """One consolidation decision, like Memory Bank's ``GeneratedMemory``."""

    action: Action
    fact: str
    memory_id: str | None = None
    previous_fact: str | None = None


def _open(path: str | Path) -> sqlite3.Connection:
    if str(path) != ":memory:":
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.touch(mode=0o600)
        os.chmod(target, 0o600)
        path = target
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    return conn


class LocalMemoryBank(BaseMemoryService):
    """A Memory Bank look-alike persisted to SQLite. See the module docstring."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        extractor: Extractor | None = None,
        default_ttl: float | None = 365 * 86400.0,
        top_k: int = 10,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = str(path)
        self.extractor: Extractor = extractor or RuleExtractor()
        self.default_ttl = default_ttl
        self.top_k = top_k
        self.clock = clock
        self._conn = _open(path)
        self._lock = threading.RLock()
        self.last_generated: list[Generated] = []

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    # BaseMemoryService
    # ------------------------------------------------------------------ #

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
        turns = turns_from_events(events)
        if not turns:
            self.last_generated = []
            return
        candidates = await self.extractor.extract(turns)
        ttl = parse_duration((custom_metadata or {}).get("ttl")) or self.default_ttl
        self.last_generated = self._consolidate(app_name, user_id, candidates, ttl, True)

    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        check_scope(app_name, user_id)
        meta = dict(custom_metadata or {})
        ttl = parse_duration(meta.get("ttl")) or self.default_ttl
        consolidate = meta.get("enable_consolidation") is True
        candidates = []
        for memory in memories:
            text = entry_text(memory)
            if not text:
                raise ValueError("a memory needs text")
            slot = memory.custom_metadata.get("slot") or None
            slot = slot if slot in t.SLOTS else None
            topic = str(memory.custom_metadata.get("topic") or t.topic_for(slot))
            candidates.append(Candidate(text, t.topic_for(slot, topic), slot))
        if consolidate:
            result = self._consolidate(app_name, user_id, candidates, ttl, False)
        else:
            result = [self._create(app_name, user_id, c, ttl) for c in candidates]
        self.last_generated = result

    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        check_scope(app_name, user_id)
        live = self._live(app_name, user_id)
        ranked = rank(query, live, [m.fact for m in live], [m.updated_at for m in live])
        return SearchMemoryResponse(memories=[m.to_entry() for m in ranked[: self.top_k]])

    async def delete_memory(
        self, *, app_name: str, user_id: str, memory_id: str | None = None
    ) -> None:
        check_scope(app_name, user_id)
        if not memory_id:
            raise ValueError("memory_id is required")
        with self._lock:
            self._conn.execute(
                "DELETE FROM memories WHERE id = ? AND app_name = ? AND user_id = ?",
                (memory_id, app_name, user_id),
            )

    # ------------------------------------------------------------------ #
    # MemoryAdmin
    # ------------------------------------------------------------------ #

    async def forget_user(self, *, app_name: str, user_id: str) -> int:
        """Delete every memory *and revision* in the scope; returns the memories removed."""
        check_scope(app_name, user_id)
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE app_name = ? AND user_id = ?", (app_name, user_id)
            )
            self._conn.execute(
                "DELETE FROM revisions WHERE app_name = ? AND user_id = ?", (app_name, user_id)
            )
            return int(cur.rowcount)

    async def list_memories(self, *, app_name: str, user_id: str) -> list[MemoryEntry]:
        check_scope(app_name, user_id)
        live = sorted(self._live(app_name, user_id), key=lambda m: -m.updated_at)
        return [m.to_entry() for m in live]

    def revisions(self, app_name: str, user_id: str) -> list[dict[str, Any]]:
        """The history of a scope, oldest first (what consolidation did)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id, action, fact, created_at FROM revisions "
                "WHERE app_name = ? AND user_id = ? ORDER BY seq",
                (app_name, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def purge_expired(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE expire_at IS NOT NULL AND expire_at <= ?",
                (self.clock(),),
            )
            return int(cur.rowcount)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _rows(self, sql: str, params: Iterable[object]) -> list[StoredMemory]:
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [StoredMemory(**dict(row)) for row in rows]

    def _live(self, app_name: str, user_id: str) -> list[StoredMemory]:
        return self._rows(
            "SELECT * FROM memories WHERE app_name = ? AND user_id = ? "
            "AND (expire_at IS NULL OR expire_at > ?) ORDER BY created_at",
            (app_name, user_id, self.clock()),
        )

    def _revision(
        self, memory_id: str, app_name: str, user_id: str, action: str, fact: str
    ) -> None:
        self._conn.execute(
            "INSERT INTO revisions (memory_id, app_name, user_id, action, fact, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (memory_id, app_name, user_id, action, fact, self.clock()),
        )

    def _expiry(self, ttl: float | None) -> float | None:
        return self.clock() + ttl if ttl else None

    def _create(self, app_name: str, user_id: str, c: Candidate, ttl: float | None) -> Generated:
        now, memory_id, fact = self.clock(), uuid.uuid4().hex, clean_fact(c.fact)
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (memory_id, app_name, user_id, fact, c.topic, c.slot, now, now, self._expiry(ttl)),
            )
            self._revision(memory_id, app_name, user_id, "CREATED", fact)
        return Generated("CREATED", fact, memory_id)

    def _update(self, old: StoredMemory, fact: str, ttl: float | None) -> Generated:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET fact = ?, updated_at = ?, expire_at = ? WHERE id = ?",
                (fact, self.clock(), self._expiry(ttl), old.id),
            )
            self._revision(old.id, old.app_name, old.user_id, "UPDATED", fact)
        return Generated("UPDATED", fact, old.id, old.fact)

    def _delete(self, old: StoredMemory) -> Generated:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (old.id,))
            self._revision(old.id, old.app_name, old.user_id, "DELETED", old.fact)
        return Generated("DELETED", old.fact, old.id)

    def _consolidate(
        self,
        app_name: str,
        user_id: str,
        candidates: Sequence[Candidate],
        ttl: float | None,
        extracted: bool,
    ) -> list[Generated]:
        """Merge candidates into the scope, one decision each (see module docstring)."""
        with self._lock:
            self.purge_expired()
            out: list[Generated] = []
            for c in candidates:
                fact = clean_fact(c.fact)
                if extracted and looks_like_instruction(fact):
                    out.append(Generated("REJECTED", fact))
                    continue
                live = self._live(app_name, user_id)
                if c.forget:
                    targets = [m for m in live if c.slot and m.slot == c.slot] or [
                        m for m in live if jaccard(m.fact, fact) >= FORGET_THRESHOLD
                    ]
                    out.extend(self._delete(m) for m in targets)
                    if not targets:
                        out.append(Generated("SKIPPED", fact))
                    continue
                same = [m for m in live if c.slot and m.slot == c.slot] or [
                    m
                    for m in live
                    if m.topic == c.topic and jaccard(m.fact, fact) >= DUPLICATE_THRESHOLD
                ]
                if not same:
                    out.append(
                        self._create(app_name, user_id, Candidate(fact, c.topic, c.slot), ttl)
                    )
                    continue
                keep, *extra = sorted(same, key=lambda m: -m.updated_at)
                out.extend(self._delete(m) for m in extra)
                if keep.fact == fact:
                    with self._lock:
                        self._conn.execute(
                            "UPDATE memories SET expire_at = ? WHERE id = ?",
                            (self._expiry(ttl), keep.id),
                        )
                    out.append(Generated("SKIPPED", fact, keep.id, keep.fact))
                else:
                    out.append(self._update(keep, fact, ttl))
            return out
