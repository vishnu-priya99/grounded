"""Ingestion must survive arbitrary, messy, or hostile uploads — one bad file
never crashes a batch, and the failure reason is legible."""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded.ingest import parsers
from grounded.ingest.tables import _clean_columns, _read_frames


def test_empty_file_is_rejected_cleanly(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("")
    with pytest.raises(ValueError, match="empty"):
        parsers.guard(f)


def test_oversize_file_is_rejected(tmp_path: Path, monkeypatch) -> None:
    from grounded.config import settings

    monkeypatch.setattr(settings, "max_file_mb", 0)
    f = tmp_path / "big.txt"
    f.write_text("x" * 1024)
    with pytest.raises(ValueError, match="limit"):
        parsers.guard(f)


def test_pdf_content_wins_over_wrong_extension(tmp_path: Path) -> None:
    f = tmp_path / "actually_a_pdf.txt"
    f.write_bytes(b"%PDF-1.4\n% fake but the header is what routing checks\n")
    assert parsers._resolve_kind(f) == "pdf"


def test_binary_garbage_is_unrecognised(tmp_path: Path) -> None:
    f = tmp_path / "mystery.bin"
    f.write_bytes(bytes(range(256)) * 4)
    assert parsers._sniff(f) is None
    with pytest.raises(ValueError):
        parsers._resolve_kind(f)


def test_csv_with_semicolons_and_bom(tmp_path: Path) -> None:
    f = tmp_path / "euro.csv"
    f.write_bytes("﻿name;score;note\nalice;10;ok\nbob;7;\n".encode())
    frames = _read_frames(f)
    df = next(iter(frames.values()))
    assert list(df.columns) == ["name", "score", "note"]
    assert len(df) == 2


def test_csv_with_ragged_rows_skips_bad_lines(tmp_path: Path) -> None:
    f = tmp_path / "ragged.csv"
    f.write_text("a,b,c\n1,2,3\n4,5,6,7,8\n9,10,11\n")
    df = next(iter(_read_frames(f).values()))
    assert len(df) >= 2  # the good rows survived


def test_clean_columns_snake_cases_and_dedupes() -> None:
    import pandas as pd

    df = pd.DataFrame([[1, 2, 3, 4]], columns=["Q3 Budget ($)", "Q3 Budget ($)",
                                               "Unnamed: 2", "2026"])
    out = _clean_columns(df)
    assert out.columns[0] == "q3_budget"
    assert out.columns[1] == "q3_budget_1"
    assert out.columns[2] == "col_3"
    assert out.columns[3] == "c_2026"


def test_batch_continues_past_a_bad_file(tmp_path: Path, monkeypatch) -> None:
    from grounded.config import settings
    from grounded.ingest import pipeline
    from grounded.ingest.pipeline import _Stores

    monkeypatch.setattr(settings, "storage_dir", tmp_path / "store")

    class _FakeVec:
        def add_chunks(self, chunks): ...
        def delete_doc(self, doc_id): ...
        def close(self): ...

    class _FakeTables:
        def close(self): ...

    monkeypatch.setattr(pipeline, "_stores",
                        _Stores(vector=_FakeVec(), tables=_FakeTables()))

    good = tmp_path / "ok.md"
    good.write_text("# Title\n\nSome content here that is long enough to chunk.")
    bad = tmp_path / "bad.txt"
    bad.write_text("")

    results = pipeline.ingest_paths([bad, good])
    assert results[0].ok is False and "empty" in results[0].error
    assert results[1].ok is True
