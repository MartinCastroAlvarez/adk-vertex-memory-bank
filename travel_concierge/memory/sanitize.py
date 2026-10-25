"""Stored memories are untrusted text.

A memory is written from what a user (or a model) said, possibly months ago, and
read back into a future prompt. Anything that made it into the store - a user who
typed "remember: ignore your instructions", an extraction model that copied it, a
record written straight to Memory Bank by another tool - must arrive in the prompt
as *data about the user*, never as instructions. Two functions do that:

* :func:`clean_fact` normalises a fact (control characters, markup, length);
* :func:`looks_like_instruction` flags text that addresses the model instead of
  describing the user.

The local backend refuses instruction-like facts at write time. Because a remote
store (Vertex) can still hold one, the agent also rewrites every recalled memory
through :func:`neutralise` right before each model call (``callbacks.memory_firewall``).
"""

from __future__ import annotations

import re
import unicodedata

MAX_FACT_CHARS = 300
WITHHELD = "[withheld: this memory read like an instruction, not a fact about the user]"

_INSTRUCTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(instructions?|rules|prompt|above|previous|prior)\b",
    r"\bdisregard\b.{0,40}\b(instructions?|rules|prompt|above|previous|prior)\b",
    r"\bforget\b.{0,20}\b(your|all|the)\b.{0,20}\b(instructions?|rules|prompt)\b",
    r"\b(system|developer)\s*(prompt|message|instructions?)\b",
    r"\byou\s+(are|must|should|will)\s+(now|always|never)\b",
    r"\b(new|updated|override)\s+instructions?\b",
    r"\bact\s+as\b",
    r"\bjailbreak\b|\bdan\s+mode\b|\bdeveloper\s+mode\b",
    r"\b(call|invoke|run|execute)\s+(the\s+)?(tool|function)\b",
    r"\b(reveal|print|show|leak)\b.{0,30}\b(prompt|instructions?|secrets?|keys?|passwords?)\b",
    r"\b(api[_ -]?key|password|credit\s*card|card\s*number)\b",
    r"</?\s*(system|assistant|user|tool|past_conversations|memories)\s*>",
]
_INSTRUCTION = re.compile("|".join(f"(?:{p})" for p in _INSTRUCTION_PATTERNS), re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_MARKUP = str.maketrans({"<": "‹", ">": "›", "{": "(", "}": ")", "`": "'"})


def clean_fact(text: str, limit: int = MAX_FACT_CHARS) -> str:
    """NFKC-normalise, drop control characters, defuse markup, collapse whitespace, cap."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub(" ", text).translate(_MARKUP)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def looks_like_instruction(text: str) -> bool:
    """True when ``text`` reads like a prompt injection rather than a fact."""
    return bool(_INSTRUCTION.search(unicodedata.normalize("NFKC", text)))


def neutralise(text: str) -> str:
    """What a recalled memory may look like inside a prompt."""
    return WITHHELD if looks_like_instruction(text) else clean_fact(text)
