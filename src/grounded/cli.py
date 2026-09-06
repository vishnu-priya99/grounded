"""Developer CLI — smoke-test ingestion and retrieval without the UI.

    python -m grounded.cli ingest data/samples/*
    python -m grounded.cli search "what was Q3 revenue"
    python -m grounded.cli ask "summarise the refund policy"
    python -m grounded.cli stats
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


def _expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        hits = [Path(p) for p in glob.glob(pat)]
        out.extend(hits or [Path(pat)])
    return [p for p in out if p.is_file()]


def cmd_ingest(args: list[str]) -> int:
    from grounded.ingest.pipeline import ingest_paths

    paths = _expand(args)
    if not paths:
        print("no files matched")
        return 1
    print(f"ingesting {len(paths)} file(s)...\n")
    for r in ingest_paths(paths):
        status = "OK " if r.ok else "FAIL"
        detail = (
            f"{r.kind:11s} {r.n_chunks:3d} chunks"
            + (f", {r.n_tables} tables" if r.n_tables else "")
            + f"  ({r.seconds:.1f}s)"
            if r.ok
            else r.error
        )
        print(f"  [{status}] {r.source_file:40s} {detail}")
    return 0


def cmd_search(args: list[str]) -> int:
    from grounded.ingest.pipeline import get_stores

    query = " ".join(args)
    hits = get_stores().vector.search(query)
    print(f'\nquery: "{query}"  ->  {len(hits)} hits\n')
    for i, h in enumerate(hits, start=1):
        tag = f"d{h.dense_rank or '-'}/s{h.sparse_rank or '-'}"
        print(f"[{i}] {h.chunk.citation_label()}   rrf={h.score:.4f} ({tag})")
        print("    " + h.chunk.body[:220].replace("\n", " ") + "\n")
    return 0


def cmd_ask(args: list[str]) -> int:
    from grounded.agents.graph import answer

    result = answer(" ".join(args))
    print("\n" + result.answer + "\n")
    if result.fallback_used:
        print("[AI-computed — not from your documents, no citations]")
    else:
        print(f"confidence: {result.confidence:.2f}   intent: {result.intent}")
    for c in result.citations:
        sup = "" if c.support is None else f"  support={c.support}"
        print(f"  [{c.n}] {c.label}{sup} — {c.snippet[:110]}")
    for t in result.tables:
        if t.sql:
            print(f"\n  SQL: {t.sql}")
    print("\ntrace:")
    for note in result.notes:
        print(f"  - {note}")
    return 0


def cmd_stats(_: list[str]) -> int:
    from grounded.ingest.pipeline import get_stores

    s = get_stores()
    print("vector store:", s.vector.stats())
    print("documents:", ", ".join(s.vector.documents()) or "(none)")
    print("SQL tables:", ", ".join(m.table_name for m in s.tables.list_tables()) or "(none)")
    return 0


_COMMANDS = {
    "ingest": cmd_ingest,
    "search": cmd_search,
    "ask": cmd_ask,
    "stats": cmd_stats,
}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] not in _COMMANDS:
        print(__doc__)
        return 1
    return _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
