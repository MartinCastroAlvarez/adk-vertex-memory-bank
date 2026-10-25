"""Recall ranking for the local backend: Okapi BM25 over facts, recency as tie-break.

Memory Bank's ``retrieve(similarity_search_params=...)`` is a nearest-neighbour
search over embeddings: it always returns the ``top_k`` closest memories in the
scope, even when none of them shares a word with the query. This ranker keeps that
contract with a lexical score, so "plan a weekend in Lisbon" still recalls "prefers
aisle seats" - just ranked after anything that mentions Lisbon.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import TypeVar

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "user",
        "users",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)
T = TypeVar("T")


def tokens(text: str) -> list[str]:
    """Lowercase word tokens without stopwords, with a light plural stem."""
    out = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.append(word)
    return out


def jaccard(a: str, b: str) -> float:
    """Token-set overlap, 0..1: how the local consolidator spots near-duplicates."""
    sa, sb = set(tokens(a)), set(tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def bm25_scores(query: str, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """One BM25 score per document."""
    q = tokens(query)
    tokenised = [tokens(d) for d in docs]
    if not q or not tokenised:
        return [0.0] * len(docs)
    n = len(tokenised)
    avg = sum(len(d) for d in tokenised) / n or 1.0
    df = Counter(term for d in tokenised for term in set(d))
    scores = []
    for doc in tokenised:
        tf = Counter(doc)
        score = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(doc) / avg))
        scores.append(score)
    return scores


def rank(query: str, items: Sequence[T], texts: Sequence[str], recency: Sequence[float]) -> list[T]:
    """``items`` ordered by BM25 score, then by ``recency`` (newest first)."""
    scores = bm25_scores(query, texts)
    order = sorted(range(len(items)), key=lambda i: (-scores[i], -recency[i]))
    return [items[i] for i in order]
