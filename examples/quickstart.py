"""Six conversations, two users, one local memory store - offline, with checks.

    python -m examples.quickstart

Uses the stub model and the rule extractor (no ollama, no key, no network) and a
throwaway SQLite file. Drop the ``Settings(...)`` overrides to run it on ollama.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from travel_concierge import Concierge
from travel_concierge import Settings
from travel_concierge.demo import run_demo


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            provider="stub", extractor="rules", db_path=Path(tmp) / "memory.sqlite3"
        )
        result = await run_demo(Concierge(settings=settings))

    step = {k: v for k, v in result.items() if isinstance(v, dict)}
    assert "vegetarian" in step["step2"]["reply"], "session 2 recalls session 1"
    assert step["step3"]["memories"] == [], "bob sees nothing of alice's"
    seats = [m for m in step["step5"]["memories"] if "seat" in m]
    assert seats == ["The user prefers window seats."], "consolidation updated, not duplicated"
    assert step["step6"]["memories"] == [], "forget me"
    print("\nquickstart ok: recall, isolation, consolidation and forget-me all checked")


if __name__ == "__main__":
    asyncio.run(main())
