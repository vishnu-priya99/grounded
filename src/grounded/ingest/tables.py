"""Spreadsheets as data, not text.

Each sheet becomes a read-only DuckDB table. A natural-language *table card*
(schema + sample rows + basic stats) is emitted so the retrieval layer can
discover that a table can answer a question — without the raw rows ever entering
a prompt. The Table/Data agent later writes SQL against these tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from grounded.config import settings
from grounded.ingest.models import Chunk, SourceKind
from grounded.util import stable_id

_SAFE = re.compile(r"[^a-z0-9_]+")
_READONLY = re.compile(r"^\s*(with|select)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|copy|pragma|install|load)\b",
    re.IGNORECASE,
)


@dataclass
class TableMeta:
    table_name: str
    source_file: str
    sheet: str | None
    columns: list[tuple[str, str]]
    row_count: int
    sample_markdown: str

    def card(self) -> str:
        cols = ", ".join(f"{n} ({t})" for n, t in self.columns)
        where = f"sheet '{self.sheet}' of " if self.sheet else ""
        return (
            f"SQL table `{self.table_name}` — from {where}{self.source_file}. "
            f"{self.row_count} rows. Columns: {cols}.\n\n"
            f"Sample rows (preview only — the full table has {self.row_count} "
            f"rows; not every value or category appears below):\n"
            f"{self.sample_markdown}"
        )


def _sanitize(name: str) -> str:
    return _SAFE.sub("_", name.lower()).strip("_") or "t"


_META_DDL = (
    "CREATE TABLE IF NOT EXISTS _grounded_meta ("
    "table_name VARCHAR PRIMARY KEY, source_file VARCHAR, sheet VARCHAR, "
    "row_count BIGINT, columns_json VARCHAR, sample_md VARCHAR, doc_id VARCHAR)"
)


class TableStore:
    """DuckDB-backed registry of spreadsheet data.

    Opened **read-only by default** so the app, the eval harness and a CLI can all
    query the same tables at once. Writes (ingestion) briefly take a read-write
    connection via :meth:`register_spreadsheet` / :meth:`delete_doc`, then drop
    back to read-only. Callers reconnect with :meth:`reload` after an ingest.
    """

    def __init__(self, *, writable: bool = False) -> None:
        settings.ensure_dirs()
        self._path = settings.duckdb_path
        self.con: duckdb.DuckDBPyConnection | None = None
        self._connect(writable=writable)

    def _connect(self, *, writable: bool) -> None:
        from pathlib import Path as _P

        if self.con is not None:
            self.con.close()
            self.con = None
        if writable:
            self.con = duckdb.connect(self._path)
            self.con.execute(_META_DDL)
        elif _P(self._path).exists():
            self.con = duckdb.connect(self._path, read_only=True)

    def reload(self) -> None:
        """Re-open read-only to pick up tables written by another connection."""
        self._connect(writable=False)

    def close(self) -> None:
        if self.con is not None:
            self.con.close()
            self.con = None

    # ---- ingest -------------------------------------------------------
    def register_spreadsheet(self, path: Path) -> tuple[str, list[Chunk]]:
        self._connect(writable=True)
        doc_id = stable_id(path.name, str(path.stat().st_size))
        frames = _read_frames(path)
        if not frames:
            raise ValueError("spreadsheet has no readable rows")

        chunks: list[Chunk] = []
        base = _sanitize(path.stem)
        used: set[str] = set()
        for sheet, df in frames.items():
            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if df.empty:
                continue
            if len(df) > settings.max_table_rows:
                df = df.head(settings.max_table_rows)
            df = _clean_columns(df)
            tname = base if len(frames) == 1 else f"{base}__{_sanitize(str(sheet))}"
            while tname in used:
                tname += "_x"
            used.add(tname)
            self.con.register("_tmp_df", df)
            self.con.execute(f'CREATE OR REPLACE TABLE "{tname}" AS SELECT * FROM _tmp_df')
            self.con.unregister("_tmp_df")

            meta = self._describe(tname, path.name, sheet if len(frames) > 1 else None)
            self.con.execute(
                "INSERT OR REPLACE INTO _grounded_meta VALUES (?,?,?,?,?,?,?)",
                [meta.table_name, meta.source_file, meta.sheet, meta.row_count,
                 _json_cols(meta.columns), meta.sample_markdown, doc_id],
            )
            chunks.append(
                Chunk(
                    chunk_id=stable_id(doc_id, tname, "card"),
                    doc_id=doc_id,
                    source_file=path.name,
                    source_kind=SourceKind.SPREADSHEET,
                    text=meta.card(),
                    body=meta.card(),
                    section_path=[tname],
                    element_kind="table",
                )
            )
        return doc_id, chunks

    def _describe(self, tname: str, source_file: str, sheet: str | None) -> TableMeta:
        info = self.con.execute(f'PRAGMA table_info("{tname}")').fetchall()
        columns = [(row[1], row[2]) for row in info]
        row_count = self.con.execute(f'SELECT count(*) FROM "{tname}"').fetchone()[0]
        sample = self.con.execute(f'SELECT * FROM "{tname}" LIMIT 15').fetch_df()
        return TableMeta(
            table_name=tname,
            source_file=source_file,
            sheet=sheet,
            columns=columns,
            row_count=int(row_count),
            sample_markdown=sample.to_markdown(index=False),
        )

    def delete_doc(self, doc_id: str) -> None:
        self._connect(writable=True)
        rows = self.con.execute(
            "SELECT table_name FROM _grounded_meta WHERE doc_id = ?", [doc_id]
        ).fetchall()
        for (tname,) in rows:
            self.con.execute(f'DROP TABLE IF EXISTS "{tname}"')
        self.con.execute("DELETE FROM _grounded_meta WHERE doc_id = ?", [doc_id])

    # ---- query -------------------------------------------------------
    def list_tables(self) -> list[TableMeta]:
        if self.con is None:
            return []
        rows = self.con.execute(
            "SELECT table_name, source_file, sheet, row_count, columns_json, sample_md "
            "FROM _grounded_meta"
        ).fetchall()
        return [
            TableMeta(r[0], r[1], r[2], _load_cols(r[4]), r[3], r[5]) for r in rows
        ]

    def schema_text(self, table_names: list[str] | None = None) -> str:
        metas = self.list_tables()
        if table_names:
            metas = [m for m in metas if m.table_name in table_names]
        return "\n\n".join(m.card() for m in metas)

    def run_sql(
        self, sql: str, *, max_rows: int = 200,
        allowed_tables: set[str] | None = None,
    ) -> tuple[list[str], list[list]]:
        if self.con is None:
            raise ValueError("No spreadsheet data has been ingested.")
        if not _READONLY.match(sql) or _FORBIDDEN.search(sql):
            raise ValueError("Only read-only SELECT / WITH queries are allowed.")
        if allowed_tables is not None:
            lowered = sql.lower()
            off_limits = [
                m.table_name for m in self.list_tables()
                if m.table_name not in allowed_tables
                and re.search(rf"\b{re.escape(m.table_name.lower())}\b", lowered)
            ]
            if off_limits:
                raise ValueError(
                    "Query references tables outside the current scope: "
                    + ", ".join(off_limits)
                )
        if " limit " not in sql.lower():
            sql = sql.rstrip().rstrip(";") + f" LIMIT {max_rows}"
        rel = self.con.execute(sql)
        cols = [d[0] for d in rel.description]
        rows = [list(r) for r in rel.fetchall()]
        return cols, rows

    def has_tables(self) -> bool:
        return bool(self.list_tables())


def _read_frames(path: Path) -> dict[str, pd.DataFrame]:
    """Load a spreadsheet into {sheet_name: DataFrame}, tolerating messy input."""
    ext = path.suffix.lower()
    if ext in {".csv", ".tsv", ".txt"}:
        sep = "\t" if ext == ".tsv" else None
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(
                    path, sep=sep, engine="python", encoding=enc,
                    on_bad_lines="skip", skip_blank_lines=True,
                )
                return {path.stem: df}
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError("could not parse CSV (encoding or delimiter)")
    try:
        return dict(pd.read_excel(path, sheet_name=None))
    except Exception as exc:
        raise ValueError(f"could not read spreadsheet: {exc}") from exc


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Snake-case column names so LLM-written SQL doesn't need quoting, and
    de-duplicate collisions. ``Unnamed: 0`` etc. become ``col_1``."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for i, c in enumerate(df.columns):
        raw = str(c).strip()
        if not raw or raw.lower().startswith("unnamed"):
            raw = f"col_{i + 1}"
        name = _SAFE.sub("_", raw.lower()).strip("_") or f"col_{i + 1}"
        if name[0].isdigit():
            name = f"c_{name}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    df = df.copy()
    df.columns = out
    return df


def _json_cols(cols: list[tuple[str, str]]) -> str:
    import json

    return json.dumps(cols)


def _load_cols(raw: str) -> list[tuple[str, str]]:
    import json

    return [tuple(x) for x in json.loads(raw)]
