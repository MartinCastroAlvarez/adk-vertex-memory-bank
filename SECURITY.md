# Security

adk-vertex-memory-bank stores what users tell an agent and feeds it back into later
prompts. That makes long-term memory both a privacy store and a delayed prompt-injection
channel: whatever was written months ago reaches the model today. Design goals: one
user can never see or erase another's memories, a stored memory can never act as an
instruction, a user can erase everything about themselves, nothing leaves the machine
unless you opt in, and the paid Vertex backend is never touched by this repo's tests or
CI.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/adk-vertex-memory-bank/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory
unless you ask otherwise. A memory that crosses a user scope, survives `forget_me`, or
reaches a prompt un-neutralised is in scope and very welcome.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** The memory store (SQLite file or Memory Bank), session state, environment
variables (optional `GOOGLE_API_KEY`, Google Cloud credentials), the model endpoint.

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| A user | their messages | read or erase another user's memories, plant instructions the agent later follows, keep data after asking to be forgotten |
| The model (concierge or extractor) | text, tool calls, JSON memories | write instruction-like "facts", erase another scope, store malformed records |
| Another writer to Memory Bank | raw facts in the bank (Vertex) | have them obeyed when recalled |
| Whoever controls `OLLAMA_HOST` | where conversations are sent | receive them without your explicit opt-in |

| Threat | Control |
| --- | --- |
| Cross-user leakage | Every memory carries the scope `{app_name, user_id}`; the local backend filters every query by both (parameterised SQL), Vertex retrieval passes the scope and Memory Bank matches it exactly. Contract tests run on both backends: bob never sees alice, another `app_name` never sees either. |
| Scope / filter injection | `app_name` and `user_id` must match `^[A-Za-z0-9._@:+-]{1,128}$` before any read, write, list or purge (tested with `alice" OR scope.user_id="bob`, empty, spaces, 200 chars, non-ASCII). That is what makes the Vertex purge filter string (`scope.user_id="..."`) safe to build. |
| Erasing someone else | `forget_me` takes **no arguments**: the scope is the session's own. Neither the model nor an injected instruction can name another user. |
| Delayed prompt injection | Two layers. (1) Write time: the local backend rejects extracted facts that read like instructions (`REJECTED`, tested). (2) Read time, for every backend: the `memory_firewall` `before_model_callback` rewrites each recalled memory in the `<PAST_CONVERSATIONS>` block and in `load_memory` results through `neutralise()`: instruction-like text becomes `[withheld: ...]`, `<`, `>`, `{`, `}` and backticks are defused, control characters dropped, length capped. A fake `</PAST_CONVERSATIONS>` inside a memory can't close the block (the parser uses the last closing tag; tested). The instruction tells the model memories are data. Pattern matching reduces injection; it cannot prove its absence. |
| Untrusted extractor output | `LlmExtractor` parses JSON out of the answer (dropping `<think>` blocks and fences), validates it with pydantic, maps unknown topics and slots to safe defaults, caps candidates (20) and fact length (300), and drops anything malformed. |
| Re-ingesting recalled memories | `save_to_memory` sends only user/agent text events; function calls and results (`load_memory`, `forget_me`) are never written back, and each event is sent once (a cursor in session state). |
| Right to be forgotten | `forget_me` / `--forget` delete every memory in the scope; the local backend deletes the revision history too; Vertex purges with `force=True`. The turn that asked to be forgotten is not saved. Tested on both backends, including that other users are untouched. |
| Data at rest | The SQLite file is created with mode `0600` in `~/.cache/travel-concierge/`. It is not encrypted; use disk encryption where that matters. TTL (365 days by default) expires memories; expired rows are invisible immediately and deleted on the next write. |
| Data leaving the machine | ollama on loopback by default; `OLLAMA_HOST` must be loopback unless `CONCIERGE_ALLOW_REMOTE_OLLAMA=1`. Gemini only with `CONCIERGE_PROVIDER=gemini` and your key; Vertex only with `CONCIERGE_MEMORY=vertex` and all three `CONCIERGE_VERTEX_*` settings. |
| Accidental cloud calls / cost | Tests replace `google.auth.default` with a function that fails the test, use a fake Memory Bank client, and a subprocess test proves `import travel_concierge` loads neither the Vertex SDK nor the Vertex adapter. CI has no credentials. |
| Secrets in prompts or logs | The package reads no secret. The extraction prompt forbids storing passwords and card numbers, and `neutralise` withholds memories mentioning API keys, passwords or card numbers. `.env` is git-ignored. |
| Code execution | No `eval`/`exec`/`compile`, `pickle`/`yaml`, `subprocess`/`shell=`; SQL is literal with `?` parameters. AST tests enforce this and forbid `# noqa`, `# nosec`, `# type: ignore` in the package. |

## 2. Operator notes

- **`adk web` is a development server.** Keep it on `127.0.0.1`: it has no authentication,
  and anyone who reaches it can chat as any `user_id`, which means reading that user's
  memories. Put real authentication in front of any deployment and derive `user_id` from it.
- **Vertex AI Memory Bank is paid** and stores data in your Google Cloud project; see
  [docs/vertex-setup.md](docs/vertex-setup.md). Give the serving identity
  `roles/aiplatform.memoryUser` only.
- **Rule extraction is a demo.** The stub provider's regex extractor understands a handful
  of travel phrasings; use the LLM extractor for real conversations.

## 3. Dependencies

CI runs `bandit` on the package and `pip-audit` on the locked environment including the
`vertex` extra. Dependabot alerts, security updates and secret scanning are enabled on the
repository. `filelock>=4.2.1` is pinned as a floor for a transitive dependency
(CVE-2026-112907, -112913, -112940).

Known and accepted: **CVE-2026-114603** in `opentelemetry-api` 1.44 (quadratic parsing of
an oversized W3C `tracestate` header, fixed in 1.45.1). `google-adk` 2.13 pins
`opentelemetry-api<=1.44`, so the fix can't be installed until ADK relaxes the pin. It is
reachable only through a server that extracts trace context from untrusted requests; the
CLI, tests and `adk run` serve nothing, and `adk web` should stay on loopback. CI ignores
exactly this ID; remove the ignore when ADK allows 1.45.1.
