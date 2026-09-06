"""acme.search.hybrid — a compact hybrid retriever.

Included in the sample corpus as a realistic code file to ask questions about.
Combines a lexical BM25 ranking with a dense vector ranking and fuses the two
with Reciprocal Rank Fusion. Pure standard library.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the of to in and or is are for on with as at by from this that it".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on word characters, drop very common words."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class Document:
    doc_id: str
    text: str
    embedding: list[float] = field(default_factory=list)


class BM25Index:
    """Okapi BM25 over an in-memory document set."""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = docs
        self._tokens = [tokenize(d.text) for d in docs]
        self._len = [len(t) for t in self._tokens]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0
        self._df: Counter[str] = Counter()
        for toks in self._tokens:
            self._df.update(set(toks))
        self._n = len(docs)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_terms = tokenize(query)
        scored: list[tuple[str, float]] = []
        for doc, toks, length in zip(self.docs, self._tokens, self._len, strict=True):
            tf = Counter(toks)
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                denom = tf[term] + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_len or 1.0)
                )
                score += self._idf(term) * (tf[term] * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((doc.doc_id, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def dense_search(
    query_embedding: list[float], docs: list[Document], top_k: int = 10
) -> list[tuple[str, float]]:
    scored = [
        (d.doc_id, cosine_similarity(query_embedding, d.embedding))
        for d in docs
        if d.embedding
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, top_k: int | None = None
) -> list[tuple[str, float]]:
    """Fuse several ranked ID lists into one using Reciprocal Rank Fusion.

    Each list contributes ``1 / (k + rank)`` to a document's score, where ``rank``
    is 1-indexed. The constant ``k`` (default ``60``, the value from the original
    Cormack et al. paper) damps the influence of the very top ranks so that a
    document must do well across *several* rankings to rise to the top.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return fused[:top_k] if top_k else fused


def hybrid_search(
    query: str,
    query_embedding: list[float],
    docs: list[Document],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Run BM25 and dense search, then fuse the two rankings with RRF."""
    bm25 = BM25Index(docs)
    lexical = [doc_id for doc_id, _ in bm25.search(query, top_k=50)]
    dense = [doc_id for doc_id, _ in dense_search(query_embedding, docs, top_k=50)]
    return reciprocal_rank_fusion([lexical, dense], top_k=top_k)
