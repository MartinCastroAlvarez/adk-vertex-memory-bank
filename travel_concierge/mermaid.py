"""A Mermaid flowchart generated from the real agent and the configured backend."""

from __future__ import annotations

from google.adk.agents import LlmAgent

from travel_concierge.concierge import build_agent
from travel_concierge.config import Settings
from travel_concierge.stub import ScriptedLlm

_BACKENDS = {
    "local": "LocalMemoryBank<br/>SQLite · extract → consolidate · BM25 · TTL",
    "vertex": "VertexMemoryBank<br/>Vertex AI Agent Engine Memory Bank",
    "inmemory": "InMemoryMemoryService<br/>raw events · keyword match",
}


def _name(callback: object) -> str:
    callbacks = callback if isinstance(callback, list) else [callback]
    return ", ".join(getattr(c, "__name__", type(c).__name__) for c in callbacks if c)


def to_mermaid(settings: Settings, agent: LlmAgent | None = None) -> str:
    agent = agent or build_agent(settings, model=ScriptedLlm(model="graph"))
    tools = [str(getattr(t, "name", None) or getattr(t, "__name__", "tool")) for t in agent.tools]
    auto = [t for t in tools if t == "preload_memory"]
    callable_tools = [t for t in tools if t not in auto]
    backend = _BACKENDS[settings.memory]
    return "\n".join(
        [
            "flowchart LR",
            '  user(["user message"])',
            f'  subgraph agent ["LlmAgent {agent.name}"]',
            "    direction TB",
            f'    preload["<b>{", ".join(auto)}</b><br/>search_memory(user message)<br/>'
            '→ PAST_CONVERSATIONS block"]',
            f'    firewall["<b>before_model: {_name(agent.before_model_callback)}</b><br/>'
            'neutralise recalled memories"]',
            '    llm["model"]',
            f'    tools["<b>tools</b><br/>{", ".join(callable_tools)}"]',
            f'    save["<b>after_agent: {_name(agent.after_agent_callback)}</b><br/>'
            'add_events_to_memory(new events)"]',
            "    preload --> firewall --> llm",
            "    llm <--> tools",
            "    llm --> save",
            "  end",
            f'  memory[("BaseMemoryService<br/>{backend}")]',
            "  user --> preload",
            "  preload -. search_memory .-> memory",
            "  tools -. load_memory / forget_user .-> memory",
            "  save -. generate: extract + consolidate .-> memory",
            '  reply(["reply"])',
            "  llm --> reply",
        ]
    )
