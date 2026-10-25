"""The scripted demo: four conversations, two users, one memory store.

Each step is a *new session*, so anything the concierge knows in step 2 came from
long-term memory, not from the chat history.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from travel_concierge.config import APP_NAME
from travel_concierge.memory.local import LocalMemoryBank
from travel_concierge.runner import Concierge


@dataclass(frozen=True)
class Step:
    user: str
    message: str
    note: str


STEPS = (
    Step(
        "alice",
        "Hi! I'm vegetarian and I always want an aisle seat. My home airport is SFO.",
        "session 1: alice states preferences",
    ),
    Step("alice", "Plan a long weekend in Lisbon for me.", "session 2: recall, no chat history"),
    Step("bob", "Plan a long weekend in Lisbon for me.", "session 3: bob sees none of it"),
    Step("alice", "Actually, I prefer window seats now.", "session 4: a changed preference"),
    Step("alice", "What do you remember about my seat?", "session 5: updated, not duplicated"),
    Step("alice", "Please forget everything about me.", "session 6: forget me"),
)


async def run_demo(concierge: Concierge, echo: Callable[[str], None] = print) -> dict[str, object]:
    """Play :data:`STEPS`; returns what a test (or CI) can check."""
    snapshots: dict[str, object] = {}
    for i, step in enumerate(STEPS, 1):
        echo(f"\n=== {step.note} ({step.user})")
        echo(f"> {step.message}")
        memory = concierge.memory
        if isinstance(memory, LocalMemoryBank):
            memory.last_generated = []
        turn = await concierge.chat(step.user, step.message)
        echo(turn.reply)
        if isinstance(memory, LocalMemoryBank):
            for g in memory.last_generated:
                was = f"  (was: {g.previous_fact})" if g.previous_fact else ""
                echo(f"  memory {g.action:<8} {g.fact}{was}")
        if turn.tool_calls:
            echo(f"  tools: {', '.join(turn.tool_calls)}")
        snapshots[f"step{i}"] = {
            "reply": turn.reply,
            "memories": await concierge.memories(step.user),
            "revisions": [r["action"] for r in memory.revisions(APP_NAME, step.user)]
            if isinstance(memory, LocalMemoryBank)
            else [],
        }
    return snapshots
