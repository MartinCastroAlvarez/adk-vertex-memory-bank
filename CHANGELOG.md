# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-25

### Added

- A travel concierge on google-adk 2.13 (`LlmAgent` with `preload_memory`, `load_memory`
  and `forget_me`), exposed as `root_agent` in `travel_concierge/agent.py` for `adk run`
  and `adk web`.
- `save_to_memory` (`after_agent_callback`, deltas via `add_events_to_memory`, a write
  cursor in session state) and `memory_firewall` (`before_model_callback` that
  neutralises recalled memories in the preload block and `load_memory` results).
- `LocalMemoryBank`: Memory Bank semantics on SQLite - LLM or rule-based fact extraction
  under managed and custom topics, slot and similarity consolidation with revisions,
  `{app_name, user_id}` scopes, BM25 recall, TTL, direct writes, `forget_user`.
- `VertexMemoryBank`: ADK's `VertexAiMemoryBankService` routed through
  `memories.generate` with `allowed_topics`, plus `forget_user` (scoped forced purge) and
  `list_memories`; never called by tests or CI.
- `CONCIERGE_MEMORY=local|vertex|inmemory`, `services.py` registering `localbank://` for
  ADK's `--memory_service_uri`, `docs/vertex-setup.md` (gcloud steps, Memory Bank config,
  clean-up; paid).
- A contract test suite run on both backends against a fake Memory Bank client;
  `OfflineConcierge` stub; `python -m travel_concierge` with `--demo`, `--memories`,
  `--forget`, `--json` and a generated `--mermaid` graph.
