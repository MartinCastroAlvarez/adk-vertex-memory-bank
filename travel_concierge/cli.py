"""``python -m travel_concierge "<message>"``: talk to the concierge, inspect or erase memory."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from travel_concierge.config import BACKENDS
from travel_concierge.config import PROVIDERS
from travel_concierge.config import Backend
from travel_concierge.config import ConfigError
from travel_concierge.config import Provider
from travel_concierge.config import Settings
from travel_concierge.memory.base import ScopeError
from travel_concierge.mermaid import to_mermaid
from travel_concierge.runner import Concierge
from travel_concierge.runner import MemoryUnsupportedError


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="travel-concierge",
        description="A Google ADK travel concierge with long-term memory "
        "(local Memory Bank look-alike by default, Vertex AI Memory Bank optional).",
    )
    p.add_argument("message", nargs="?", help='e.g. "I always want an aisle seat"')
    p.add_argument("--user", default="local", help="user id: the memory scope (default: local)")
    p.add_argument("--session", help="continue a session id (default: a new conversation)")
    p.add_argument("--provider", choices=PROVIDERS, help="default: CONCIERGE_PROVIDER or ollama")
    p.add_argument("--model", help="model id for the provider (default: qwen3:4b on ollama)")
    p.add_argument("--memory", choices=BACKENDS, help="default: CONCIERGE_MEMORY or local")
    p.add_argument("--db", type=Path, help="SQLite file for the local backend")
    p.add_argument("--memories", action="store_true", help="list what is remembered and exit")
    p.add_argument("--forget", action="store_true", help="forget --user entirely and exit")
    p.add_argument(
        "--demo",
        action="store_true",
        help="play the six-session scripted demo (in-memory store unless --db)",
    )
    p.add_argument("--json", action="store_true", help="print results as JSON")
    p.add_argument("--mermaid", action="store_true", help="print the agent graph and exit")
    return p


def settings_from(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    provider = cast(Provider, args.provider or settings.provider)
    model = args.model or settings.model
    extractor = settings.extractor
    if args.provider and args.provider != settings.provider:
        defaults = Settings.from_env({"CONCIERGE_PROVIDER": args.provider})
        model = args.model or defaults.model
        extractor = defaults.extractor
    return replace(
        settings,
        provider=provider,
        model=model,
        extractor=extractor,
        memory=cast(Backend, args.memory or settings.memory),
        # The demo gets a throwaway store unless --db says otherwise.
        db_path=args.db or (Path(":memory:") if args.demo else settings.db_path),
    )


async def _run(args: argparse.Namespace, concierge: Concierge) -> int:
    if args.memories:
        memories = await concierge.memories(args.user)
        if args.json:
            print(json.dumps({"user": args.user, "memories": memories}, indent=2))
        else:
            print("\n".join(f"- {m}" for m in memories) or f"(nothing remembered for {args.user})")
        return 0
    if args.forget:
        count = await concierge.forget(args.user)
        print(
            json.dumps({"user": args.user, "deleted": count}) if args.json else f"deleted {count}"
        )
        return 0
    if args.demo:
        from travel_concierge.demo import run_demo

        result = await run_demo(concierge, echo=(lambda _: None) if args.json else print)
        if args.json:
            print(json.dumps(result, indent=2))
        return 0
    turn = await concierge.chat(args.user, args.message, session_id=args.session)
    if args.json:
        payload = {"reply": turn.reply, "session_id": turn.session_id, "tools": turn.tool_calls}
        print(json.dumps(payload, indent=2))
    else:
        print(turn.reply)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = settings_from(args)
        if args.mermaid:
            print(to_mermaid(settings))
            return 0
        if not (args.message or args.memories or args.forget or args.demo):
            print(
                'error: give a message, e.g. python -m travel_concierge "Plan Lisbon"',
                file=sys.stderr,
            )
            return 2
        concierge = Concierge(settings=settings)
        return asyncio.run(_run(args, concierge))
    except (ConfigError, ScopeError, MemoryUnsupportedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
