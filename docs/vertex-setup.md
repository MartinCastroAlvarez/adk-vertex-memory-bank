# Running the concierge on Vertex AI Agent Engine Memory Bank

> **This costs money.** Memory Bank is a paid Google Cloud service: memory generation
> calls a Gemini model, retrieval calls an embedding model, and stored memories are
> billed. Nothing in this repository - tests, demo, CI - ever calls it; the default
> backend is the free local one. Follow this page only with **your own** project and
> billing account, and delete the resources when you are done (last section).
> Check current prices on the Vertex AI / Agent Platform pricing pages first.

No Terraform: four `gcloud` commands, one Python snippet, three environment variables.
Google has been renaming this product (Vertex AI Agent Engine → Gemini Enterprise Agent
Platform, "reasoning engine" → "agent runtime"); the API resource is still
`projects/<p>/locations/<l>/reasoningEngines/<id>`, and that `<id>` is what ADK's
`VertexAiMemoryBankService(agent_engine_id=...)` wants. Names below are as of
`google-adk` 2.13 and `google-cloud-agentplatform` 2.6.

## 1. Project, API, roles

```bash
export PROJECT_ID=my-project          # yours; billing must be enabled
export LOCATION=us-central1
gcloud auth login
gcloud config set project "$PROJECT_ID"
gcloud services enable aiplatform.googleapis.com
# the account that creates the instance:   roles/aiplatform.user
# the account the agent runs as (memories): roles/aiplatform.memoryUser
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:you@example.com" --role="roles/aiplatform.user"
gcloud auth application-default login   # Application Default Credentials for the SDK
```

Grant the narrowest role to the identity that serves the agent: it only needs
`roles/aiplatform.memoryUser` (read, write and generate memories), not admin rights.

## 2. Install the Vertex extra

```bash
pip install "adk-vertex-memory-bank[vertex] @ git+https://github.com/MartinCastroAlvarez/adk-vertex-memory-bank"
```

## 3. Create an Agent Engine instance with a Memory Bank

The Memory Bank is configured on the instance: the topics to extract (the four managed
topics plus this repo's custom `travel_preferences`), the scope keys ADK uses
(`app_name`, `user_id`), third-person facts (like the local backend writes), the TTL,
and the models it may use. This dict validates against
`agentplatform.types.AgentRuntimeConfig`:

```python
import agentplatform

from travel_concierge.memory.topics import TRAVEL_PREFERENCES_DESCRIPTION

PROJECT, LOCATION = "my-project", "us-central1"
client = agentplatform.Client(project=PROJECT, location=LOCATION)

managed = ["USER_PERSONAL_INFO", "USER_PREFERENCES", "KEY_CONVERSATION_DETAILS",
           "EXPLICIT_INSTRUCTIONS"]
runtime = client.runtimes.create(config={
    "display_name": "travel-concierge-memory",
    "context_spec": {"memory_bank_config": {
        "customization_configs": [{
            "scope_keys": ["app_name", "user_id"],
            "memory_topics": [
                *({"managed_memory_topic": {"managed_topic_enum": t}} for t in managed),
                {"custom_memory_topic": {"label": "travel_preferences",
                                         "description": TRAVEL_PREFERENCES_DESCRIPTION}},
            ],
            "enable_third_person_memories": True,
        }],
        # Memories expire a year after they were last created or updated.
        "ttl_config": {"default_ttl": f"{365 * 24 * 3600}s"},
        "generation_config": {"model":
            f"projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/gemini-3.5-flash"},
        "similarity_search_config": {"embedding_model":
            f"projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/text-embedding-005"},
    }},
})
print(runtime.api_resource.name.split("/")[-1])   # the agent_engine_id
```

No agent code is deployed by this call: the instance only hosts the Memory Bank. The
concierge keeps running on your machine (or wherever you run ADK).

## 4. Point the concierge at it

```bash
export CONCIERGE_MEMORY=vertex
export CONCIERGE_VERTEX_PROJECT=$PROJECT_ID
export CONCIERGE_VERTEX_LOCATION=$LOCATION
export CONCIERGE_VERTEX_AGENT_ENGINE_ID=1234567890    # the number printed above

python -m travel_concierge --user alice "I'm vegetarian and I always want an aisle seat."
python -m travel_concierge --user alice "Plan a long weekend in Lisbon."   # new session
python -m travel_concierge --user alice --memories
python -m travel_concierge --user alice --forget
```

That one variable swaps `LocalMemoryBank` for `VertexMemoryBank`
(`travel_concierge/memory/vertex.py`), a thin subclass of ADK's
`VertexAiMemoryBankService(project, location, agent_engine_id)`. The agent does not
change. What each call becomes:

| Concierge | Memory Bank API |
| --- | --- |
| `save_to_memory` after each turn | `memories.generate(direct_contents_source=..., scope={app_name, user_id}, config={wait_for_completion, allowed_topics})`: extraction + consolidation server-side |
| `preload_memory` / `load_memory` | `memories.retrieve(scope=..., similarity_search_params={search_query})` |
| `forget_me`, `--forget` | `memories.purge(filter='scope.app_name="..." AND scope.user_id="..."', force=True)` |
| `--memories` | `memories.retrieve(scope=...)` |

The local model still writes the replies (ollama); only memory moves to Google Cloud.
Set `CONCIERGE_PROVIDER=gemini` and `GOOGLE_API_KEY` too if you want Gemini replies.

With ADK's own CLIs, use ADK's built-in URI instead of the env vars:

```bash
GOOGLE_CLOUD_PROJECT=$PROJECT_ID GOOGLE_CLOUD_LOCATION=$LOCATION \
  adk web . --memory_service_uri="agentengine://1234567890"
```

## 5. Clean up

```bash
python - <<'PY'
import agentplatform
client = agentplatform.Client(project="my-project", location="us-central1")
client.runtimes.delete(name="projects/my-project/locations/us-central1/reasoningEngines/1234567890",
                       force=True)   # force also deletes its memories and sessions
PY
```

Deleting the instance deletes the memories in it. If you only want to remove one
person's data, `python -m travel_concierge --user <id> --forget` purges their scope.
