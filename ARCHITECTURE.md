# Architecture

```
travel_concierge/
  agent.py        root_agent = build_agent(Settings.from_env())       <- adk run / adk web
  concierge.py    the LlmAgent: preload_memory, load_memory, forget_me, two callbacks
  callbacks.py    save_to_memory (after_agent), memory_firewall (before_model)
  tools.py        forget_me (no arguments: the session's own scope)
  runner.py       Concierge: Runner + InMemorySessionService + any BaseMemoryService
  config.py       Settings from CONCIERGE_* (provider, backend, TTL, Vertex ids)
  models.py       provider -> BaseLlm (LiteLlm on ollama, Gemini, OfflineConcierge)
  offline.py      OfflineConcierge: a template stub that reads the preloaded memories
  stub.py         StubLlm, ScriptedLlm (tests), ToolCall
  demo.py         the six-session scripted demo
  mermaid.py      Mermaid graph generated from the agent and backend
  cli.py          python -m travel_concierge
  memory/
    base.py       MemoryAdmin protocol (forget_user, list_memories), scope checks, helpers
    topics.py     managed + custom topics, slots
    extract.py    Candidate; RuleExtractor; LlmExtractor (JSON from any BaseLlm)
    local.py      LocalMemoryBank: SQLite, consolidation, revisions, BM25, TTL
    search.py     BM25 + Jaccard
    sanitize.py   clean_fact, looks_like_instruction, neutralise
    vertex.py     VertexMemoryBank(VertexAiMemoryBankService) + forget_user/list_memories
    factory.py    CONCIERGE_MEMORY -> BaseMemoryService
    local_uri.py  localbank://<path> for ADK's --memory_service_uri
services.py       registers localbank:// with ADK's service registry
tests/fake_memory_bank.py   in-process fake of the agentplatform Memory Bank client
```

## One turn

1. `Runner.run_async` appends the user's message to the session and starts the agent.
2. Before the model call, ADK runs each tool's `process_llm_request`.
   `preload_memory` calls `memory_service.search_memory(app_name, user_id, query=<the
   message>)` and inserts the hits as a `<PAST_CONVERSATIONS>` block; `load_memory`
   adds its declaration and a "you have memory" instruction.
3. `memory_firewall` (`before_model_callback`) rewrites the block (and any earlier
   `load_memory` results in the request) through `neutralise()`.
4. The model answers, possibly calling `load_memory(query)` or `forget_me()` first;
   each tool call starts another model call, which passes the firewall again.
5. `save_to_memory` (`after_agent_callback`) collects the session's events since the
   cursor `state["memory_saved_events"]`, keeps user/agent text only, and calls
   `callback_context.add_events_to_memory(events=...)`. If `forget_me` ran, it saves
   nothing and moves the cursor to the end.

## The memory backends

| | `LocalMemoryBank` | `VertexMemoryBank` |
| --- | --- | --- |
| write | `add_events_to_memory` → extractor → `_consolidate` | ADK's `add_events_to_memory` → `memories.generate(direct_contents_source, scope, config)` |
| consolidation | slot match → UPDATE, near-duplicate → UPDATE, forget → DELETE, repeat → SKIP, instruction → REJECT; revision row per change | server-side (Gemini) |
| recall | BM25 over live facts in scope, recency tie-break, top-k | `memories.retrieve(scope, similarity_search_params)` |
| TTL | `expire_at` per memory, `default_ttl` or per-write `ttl` | the bank's `ttl_config` |
| forget a user | `DELETE` memories + revisions in scope | `memories.purge(filter='scope.app_name=".." AND scope.user_id=".."', force=True)` |
| list | live rows, newest first | `memories.retrieve(scope)` |

Both implement ADK's `BaseMemoryService` and this repo's `MemoryAdmin` protocol, and the
same contract tests run on both.

### Local schema

```sql
memories(id, app_name, user_id, fact, topic, slot, created_at, updated_at, expire_at)
revisions(seq, memory_id, app_name, user_id, action, fact, created_at)
```

Every query filters on `app_name` and `user_id`; all SQL is literal with `?` parameters.
The file is opened in WAL mode and created with permissions `0600`.

## Why these choices

- **Deltas, not sessions.** `add_session_to_memory` on every turn re-sends the whole
  conversation; Memory Bank consolidates duplicates but bills each generation. A cursor
  in session state makes every event go once.
- **`memories.generate` over `ingest_events` for Vertex.** ADK 2.13 defaults to
  `ingest_events` in a background task, which drops failures into a log. Routing through
  `generate` with `wait_for_completion` keeps writes synchronous and testable, and adds
  `allowed_topics`.
- **Slots for consolidation.** A deterministic rule ("one current seat preference")
  makes the core Memory Bank behaviour, *update instead of duplicate*, testable without
  an LLM judge. The LLM decides the slot; code decides the merge.
- **Firewall on read, filter on write.** The local backend can refuse a bad fact; Vertex
  (or anything else writing to the bank) can't be trusted to, so the guarantee lives in a
  callback every backend's output passes through.
- **A fake client, not mocks of ADK.** `FakeMemoryBank` sits below ADK's
  `VertexAiMemoryBankService`, so ADK's own request building and response parsing run in
  the tests; only the network is replaced.
