"""Dense embeddings via the local Ollama ``nomic-embed-text`` model.

Chosen because it is already pulled, adds ~0.5 GB rather than the ~2 GB a
PyTorch sentence-transformer would, and needs no separate runtime. Sparse
retrieval is handled lexically by BM25 in :mod:`grounded.index.store`, so this
module is dense-only.
"""

from __future__ import annotations

import ollama

from grounded.config import settings

_PREFIX_DOC = "search_document: "
_PREFIX_QUERY = "search_query: "


def _client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_host)


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    resp = _client().embed(
        model=settings.embed_model, input=[_PREFIX_DOC + t for t in texts]
    )
    return [list(v) for v in resp["embeddings"]]


def embed_query(text: str) -> list[float]:
    resp = _client().embed(model=settings.embed_model, input=_PREFIX_QUERY + text)
    return list(resp["embeddings"][0])


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Prefix-free embeddings for similarity comparisons (e.g. verification)."""
    if not texts:
        return []
    resp = _client().embed(model=settings.embed_model, input=texts)
    return [list(v) for v in resp["embeddings"]]
