"""Ingestion pipeline: path in, indexed chunks out."""

from __future__ import annotations

import atexit
import time
from dataclasses import dataclass
from pathlib import Path

from grounded.config import settings
from grounded.index.store import HybridStore
from grounded.ingest import parsers
from grounded.ingest.chunking import chunk_document
from grounded.ingest.enumerate import split_enumerated
from grounded.ingest.tables import TableStore


@dataclass
class IngestResult:
    source_file: str
    ok: bool
    doc_id: str = ""
    n_chunks: int = 0
    n_tables: int = 0
    kind: str = ""
    seconds: float = 0.0
    error: str = ""


@dataclass
class _Stores:
    vector: HybridStore
    tables: TableStore

    def close(self) -> None:
        self.vector.close()
        self.tables.close()


_stores: _Stores | None = None


def get_stores() -> _Stores:
    global _stores
    if _stores is None:
        _stores = _Stores(vector=HybridStore(), tables=TableStore())
        atexit.register(_stores.close)
    return _stores


def ingest_file(path: str | Path, *, stores: _Stores | None = None) -> IngestResult:
    path = Path(path)
    stores = stores or get_stores()
    start = time.perf_counter()

    try:
        parsers.guard(path)
        if not parsers.supported(path):
            return IngestResult(
                path.name, ok=False,
                error=f"unsupported file type ({path.suffix or 'no extension'})",
            )

        if path.suffix.lower() in parsers.SPREADSHEET_EXT:
            doc_id, cards = stores.tables.register_spreadsheet(path)
            stores.tables.reload()  # drop back to read-only
            stores.vector.add_chunks(cards)
            return IngestResult(
                path.name, ok=True, doc_id=doc_id, n_tables=len(cards),
                n_chunks=len(cards), kind="spreadsheet",
                seconds=time.perf_counter() - start,
            )

        doc = parsers.parse(path)
        doc = split_enumerated(doc)
        chunks = chunk_document(doc)
        if not chunks:
            return IngestResult(
                path.name, ok=False, error="no text could be extracted",
                seconds=time.perf_counter() - start,
            )
        stores.vector.delete_doc(doc.doc_id)  # idempotent: re-ingest replaces
        stores.vector.add_chunks(chunks)
        return IngestResult(
            path.name, ok=True, doc_id=doc.doc_id, n_chunks=len(chunks),
            kind=doc.source_kind.value, seconds=time.perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001 - report, don't crash the batch
        return IngestResult(
            path.name, ok=False, error=f"{type(exc).__name__}: {exc}",
            seconds=time.perf_counter() - start,
        )


def ingest_paths(paths: list[str | Path]) -> list[IngestResult]:
    stores = get_stores()
    return [ingest_file(p, stores=stores) for p in paths]


def delete_source_file(source_file: str, *, stores: _Stores | None = None) -> int:
    """Remove one displayed document from every store it touched — the vector
    index, its SQL tables if it was a spreadsheet, and its file under
    ``data/uploads/`` if that's where it came from (a bundled sample under
    ``data/samples/`` is left on disk; only the index entry goes). Returns
    how many chunks were removed."""
    stores = stores or get_stores()
    doc_ids = stores.vector.doc_ids_for(source_file)
    n_chunks = sum(1 for c in stores.vector.chunks if c.source_file == source_file)
    for doc_id in doc_ids:
        stores.vector.delete_doc(doc_id)
        stores.tables.delete_doc(doc_id)
    stores.tables.reload()  # drop back to read-only, same as after ingest

    upload_path = settings.data_dir / "uploads" / source_file
    if upload_path.exists():
        upload_path.unlink()
    return n_chunks
