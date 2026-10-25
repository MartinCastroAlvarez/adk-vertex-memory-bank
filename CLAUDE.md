# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md). 5. [`docs/vertex-setup.md`](docs/vertex-setup.md).

## Mission

`adk-vertex-memory-bank` shows long-term memory in current Google ADK (Python): an agent
that depends only on `BaseMemoryService`, recalls with `preload_memory` / `load_memory`,
writes with an `after_agent_callback`, and runs on Vertex AI Memory Bank or on a local,
free look-alike with the same behaviour, as small, correct and tested code.

## The rules

1. **Zero cost.** Never call, deploy to or authenticate against Google Cloud. Tests, CI
   and the demo use the stub model, the rule extractor, SQLite and `FakeMemoryBank`.
   `google.auth.default()` fails any test that calls it; keep it that way.
2. **The agent knows only `BaseMemoryService`.** Backend-specific code lives in
   `travel_concierge/memory/`. A new behaviour goes into the contract suite
   (`tests/test_contract.py`) and must pass on both backends.
3. **Memories are untrusted.** Recalled text reaches the model only through
   `memory_firewall`; extracted facts go through `clean_fact` and `looks_like_instruction`.
4. **Scopes are sacred.** Every read, write, list and purge takes `app_name` and
   `user_id`, validated by `check_scope`. Tools never take a user id.
5. **ADK conventions.** `root_agent` lives in `travel_concierge/agent.py`; `adk run` and
   `adk web` with `--memory_service_uri=localbank://...` must keep working (CI runs both).
6. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a test, or
   loosen lint/type/coverage config. SQL stays literal with `?` parameters.

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`, `mypy`
(strict), `bandit` and `pip-audit` all green; `python -m examples.quickstart` passes;
README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG are updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets, `.env`,
`.adk/` or SQLite files.
