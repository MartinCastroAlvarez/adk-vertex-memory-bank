"""Which model the concierge (and the local memory extractor) uses.

* ``ollama`` (default): ``LiteLlm(model="ollama_chat/<model>")`` on a loopback ollama.
* ``gemini`` (optional): ADK's ``Gemini`` with your own ``GOOGLE_API_KEY``. Never used
  by tests or CI.
* ``stub``: :class:`travel_concierge.offline.OfflineConcierge`, deterministic and offline.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from google.adk.models.base_llm import BaseLlm

from travel_concierge.config import ConfigError
from travel_concierge.config import Settings


def ollama_model_id(model: str) -> str:
    """``qwen3:4b`` -> ``ollama_chat/qwen3:4b`` (the LiteLLM provider that supports tools)."""
    return model if model.startswith(("ollama_chat/", "ollama/")) else f"ollama_chat/{model}"


def _gemini(settings: Settings, env: Mapping[str, str]) -> BaseLlm:
    if not (env.get("GOOGLE_API_KEY") or env.get("GOOGLE_GENAI_USE_VERTEXAI") == "1"):
        raise ConfigError(
            "CONCIERGE_PROVIDER=gemini needs GOOGLE_API_KEY (your key, your bill); "
            "the default provider is a local ollama"
        )
    from google.adk.models.google_llm import Gemini

    return Gemini(model=settings.model)


def _ollama(settings: Settings) -> BaseLlm:
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=ollama_model_id(settings.model), api_base=settings.ollama_host)


def concierge_model(settings: Settings, env: Mapping[str, str] | None = None) -> BaseLlm:
    env = os.environ if env is None else env
    if settings.provider == "stub":
        from travel_concierge.offline import OfflineConcierge

        return OfflineConcierge()
    if settings.provider == "gemini":
        return _gemini(settings, env)
    return _ollama(settings)


def extraction_model(settings: Settings, env: Mapping[str, str] | None = None) -> BaseLlm:
    """The model ``LlmExtractor`` asks for facts (same provider as the concierge)."""
    env = os.environ if env is None else env
    if settings.provider == "stub":
        raise ConfigError(
            "the stub provider has no language model to extract memories with; "
            "use CONCIERGE_EXTRACTOR=rules"
        )
    if settings.provider == "gemini":
        return _gemini(settings, env)
    return _ollama(settings)
