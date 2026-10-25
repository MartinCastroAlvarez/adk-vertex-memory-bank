# Contributing to adk-vertex-memory-bank

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install
poetry run pytest
poetry run ruff check travel_concierge tests examples services.py
poetry run ruff format --check travel_concierge tests examples services.py
poetry run mypy travel_concierge services.py
poetry run bandit -c pyproject.toml -q -r travel_concierge services.py
poetry run pip-audit --ignore-vuln CVE-2026-114603   # see SECURITY.md
poetry run python -m examples.quickstart
```

Open a PR against `main` from a `feat/...`, `fix/...`, `docs/...` or `chore/...` branch.

## What this project will not accept

- Tests or CI that call a real model, Google Cloud or the network, or that need credentials.
- A memory behaviour tested on one backend only: add it to `tests/test_contract.py`, and
  to `tests/fake_memory_bank.py` if the fake needs to learn it (following the Memory Bank
  docs, with a link in the PR).
- A tool that takes a user id, a path or a URL.
- Hosted providers or Vertex turned on by default.

## What it does accept

- Local embeddings for recall (e.g. a small sentence-transformer or ollama embeddings)
  behind `memory/search.py`, keeping BM25 as the dependency-free default.
- An LLM-judged consolidation step (Memory Bank asks a model to merge), as long as the slot
  rules stay as the deterministic fallback.
- More slots and topics, Memory Bank features such as revisions rollback or structured
  profiles (`retrieve_profiles`), and hardening, each with a hostile test.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds defaults only.
