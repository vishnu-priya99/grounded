"""Hybrid retrieval over an embedded Qdrant collection.

- **Dense**: every chunk's ``nomic-embed-text`` vector lives in a local Qdrant
  collection (embedded/local mode -- no server process, storage is a folder on
  disk). Qdrant does the similarity search natively.
- **Sparse**: an in-process BM25 index over the same chunks (Qdrant's sparse
  vectors would need a second index configured per collection; plain BM25 is
  simpler and this corpus is small).
- **Fusion**: Reciprocal Rank Fusion of the two ranked lists, same as before.

NOTE -- Qdrant's local/embedded mode takes an **exclusive lock** on its storage
directory for as long as a client has it open. Unlike the DuckDB table store
(opened read-only for queries), only *one* process can have this index open at
a time: don't run the CLI or eval harness while the Streamlit app is running --
close the app first. This is the tradeoff embedded mode was originally dropped
for; kept here anyway per an explicit choice to use Qdrant embedded mode.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from grounded.config import settings
from grounded.index.embeddings import embed_documents, embed_query
from grounded.ingest.models import Chunk, SourceKind

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LOCK = threading.Lock()
_COLLECTION = "chunks"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _point_id(chunk_id: str) -> int:
    """Qdrant point ids must be an unsigned int or a UUID; ``chunk_id`` is a
    16-hex-char (64-bit) stable hash, so it fits directly into a u64 int."""
    return int(chunk_id, 16)


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    components: dict[str, float] = field(default_factory=dict)

    @property
    def context_text(self) -> str:
        if self.chunk.parent_text and len(self.chunk.parent_text) > len(self.chunk.body):
            return self.chunk.parent_text
        return self.chunk.body


class HybridStore:
    def __init__(self) -> None:
        settings.ensure_dirs()
        if settings.qdrant_url:
            self._client = QdrantClient(url=settings.qdrant_url)
        else:
            self._dir = str(settings.storage_dir / "qdrant")
            self._client = QdrantClient(path=self._dir)
        self._ensure_collection()

        self.chunks: list[Chunk] = []
        self._bm25 = None
        self._load()

    # ---- setup / persistence ------------------------------------------
    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(_COLLECTION):
            self._client.create_collection(
                _COLLECTION,
                vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
            )

    def _load(self) -> None:
        self.chunks = self._scroll_all()
        self._build_bm25()

    def _scroll_all(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                _COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False,
            )
            chunks.extend(Chunk.model_validate(p.payload) for p in points)
            if offset is None:
                break
        return chunks

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        corpus = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def close(self) -> None:
        self._client.close()

    # ---- write ------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk], *, batch: int = 64) -> None:
        if not chunks:
            return
        with _LOCK:
            existing = {c.chunk_id for c in self.chunks}
            fresh = [c for c in chunks if c.chunk_id not in existing]
            if not fresh:
                return
            for i in range(0, len(fresh), batch):
                part = fresh[i : i + batch]
                vecs = embed_documents([c.text for c in part])
                points = [
                    PointStruct(
                        id=_point_id(c.chunk_id),
                        vector=np.asarray(v, dtype=np.float32).tolist(),
                        payload=c.model_dump(mode="json"),
                    )
                    for c, v in zip(part, vecs, strict=True)
                ]
                self._client.upsert(_COLLECTION, points=points)
            self.chunks.extend(fresh)
            self._build_bm25()

    def delete_doc(self, doc_id: str) -> None:
        with _LOCK:
            self._client.delete(
                _COLLECTION,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchAny(any=[doc_id]))]
                ),
            )
            self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
            self._build_bm25()

    # ---- read -----------------------------------------------------
    def stats(self) -> dict[str, int]:
        return {"chunks": len(self.chunks), "documents": len(self.documents())}

    def documents(self) -> list[str]:
        return sorted({c.source_file for c in self.chunks})

    def doc_ids_for(self, source_file: str) -> list[str]:
        """``doc_id``s behind one displayed filename — normally one, but a
        source_file could map to more than one doc_id if it was re-ingested
        after actually changing (doc_id includes file size)."""
        return sorted({c.doc_id for c in self.chunks if c.source_file == source_file})

    def chunks_by_ordinal(
        self, ordinal: int, *, source_files: list[str] | None = None
    ) -> list[Chunk]:
        """Enumerated-item chunks matching a positional reference. ``ordinal=-1``
        means "the last item" — resolved per document."""
        pool = [
            c for c in self.chunks
            if c.ordinal is not None
            and (not source_files or c.source_file in source_files)
        ]
        if ordinal >= 0:
            return [c for c in pool if c.ordinal == ordinal]
        out: list[Chunk] = []
        for src in {c.source_file for c in pool}:
            in_doc = [c for c in pool if c.source_file == src]
            top = max(c.ordinal for c in in_doc)  # type: ignore[type-var]
            out.extend(c for c in in_doc if c.ordinal == top)
        return out

    def chunks_for_documents(self, source_files: list[str]) -> list[Chunk]:
        """Every chunk belonging to the given document(s), in reading order —
        for a "broad" question ('what topics does this cover') that needs to
        survey a whole document rather than the usual top-k similarity search,
        which only ever samples a handful of chunks regardless of how many
        the document actually has."""
        pool = [c for c in self.chunks if c.source_file in source_files]
        return sorted(pool, key=lambda c: (c.page or 0, c.ordinal or 0))

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source_files: list[str] | None = None,
        kinds: list[SourceKind] | None = None,
    ) -> list[Retrieved]:
        if not self.chunks:
            return []
        top_k = top_k or settings.top_k
        n = settings.retrieve_candidates

        allow_ids = {
            c.chunk_id
            for c in self.chunks
            if (not source_files or c.source_file in source_files)
            and (not kinds or c.source_kind in kinds)
        }
        if not allow_ids:
            return []

        # dense -- Qdrant applies the same source/kind filter natively
        q = np.asarray(embed_query(query), dtype=np.float32).tolist()
        qfilter = _build_filter(source_files, kinds)
        hits = self._client.query_points(
            _COLLECTION, query=q, limit=n, query_filter=qfilter, with_payload=True,
        ).points
        dense_order = [h.payload["chunk_id"] for h in hits]

        # sparse -- BM25 stays in-process, post-filtered like before
        sparse_order: list[str] = []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(query))
            for i in np.argsort(-scores):
                if scores[i] <= 0 or len(sparse_order) >= n:
                    break
                cid = self.chunks[i].chunk_id
                if cid in allow_ids:
                    sparse_order.append(cid)

        return self._fuse(query, dense_order, sparse_order, top_k)

    # ---- fusion --------------------------------------------------
    def _fuse(
        self, query: str, dense: list[str], sparse: list[str], top_k: int
    ) -> list[Retrieved]:
        k = settings.rrf_k
        by_id = {c.chunk_id: c for c in self.chunks}
        agg: dict[str, Retrieved] = {}

        def bump(cid: str, comp: str, rank: int) -> None:
            c = by_id.get(cid)
            if c is None:
                return
            r = agg.setdefault(cid, Retrieved(chunk=c, score=0.0))
            r.score += 1.0 / (k + rank)
            r.components[comp] = 1.0 / (k + rank)
            if comp == "dense":
                r.dense_rank = rank
            else:
                r.sparse_rank = rank

        for rank, cid in enumerate(dense, start=1):
            bump(cid, "dense", rank)
        for rank, cid in enumerate(sparse, start=1):
            bump(cid, "sparse", rank)

        ordered = sorted(agg.values(), key=lambda r: r.score, reverse=True)
        if settings.enable_rerank and not settings.fast_mode:
            ordered = _rerank(query, ordered[: max(top_k * 4, 20)])
        return ordered[:top_k]


def _build_filter(
    source_files: list[str] | None, kinds: list[SourceKind] | None
) -> Filter | None:
    must = []
    if source_files:
        must.append(FieldCondition(key="source_file", match=MatchAny(any=source_files)))
    if kinds:
        must.append(FieldCondition(key="source_kind", match=MatchAny(any=[k.value for k in kinds])))
    return Filter(must=must) if must else None


def _rerank(query: str, items: list[Retrieved]) -> list[Retrieved]:  # pragma: no cover
    """Optional ONNX cross-encoder rerank. Disabled by default; the accuracy
    upgrade for machines with a GPU."""
    try:
        from grounded.index.rerank import cross_encoder_scores

        scores = cross_encoder_scores(query, [r.chunk.body for r in items])
        for r, s in zip(items, scores, strict=True):
            r.score = float(s)
        return sorted(items, key=lambda r: r.score, reverse=True)
    except Exception as exc:
        print(f"[retrieval] rerank unavailable, keeping RRF order: {exc}")
        return items
