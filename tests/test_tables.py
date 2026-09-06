"""TableStore tests — real DuckDB, no LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded.ingest.tables import TableStore

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.fixture
def store(tmp_path, monkeypatch) -> TableStore:
    from grounded.config import settings

    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    s = TableStore()
    s.register_spreadsheet(SAMPLES / "q3_sales.csv")
    return s


def test_csv_becomes_queryable_table(store: TableStore) -> None:
    cols, rows = store.run_sql("SELECT SUM(revenue_usd) AS t FROM q3_sales")
    assert rows[0][0] == 2_460_000


def test_run_sql_rejects_tables_outside_the_allowed_scope(store: TableStore) -> None:
    store.register_spreadsheet(SAMPLES / "company_metrics.xlsx")
    store.reload()
    # only q3_sales is in scope
    with pytest.raises(ValueError, match="outside the current scope"):
        store.run_sql(
            "SELECT SUM(headcount) FROM company_metrics__headcount",
            allowed_tables={"q3_sales"},
        )
    # a query that stays in scope still runs
    _, rows = store.run_sql(
        "SELECT SUM(units_sold) AS t FROM q3_sales", allowed_tables={"q3_sales"}
    )
    assert rows[0][0] > 0


def test_multi_sheet_xlsx_registers_one_table_per_sheet(store: TableStore) -> None:
    store.register_spreadsheet(SAMPLES / "company_metrics.xlsx")
    names = {m.table_name for m in store.list_tables()}
    assert "company_metrics__headcount" in names
    assert "company_metrics__budget" in names


@pytest.mark.parametrize(
    "sql",
    ["DROP TABLE q3_sales", "DELETE FROM q3_sales", "UPDATE q3_sales SET returns = 0",
     "INSERT INTO q3_sales VALUES (1)", "ATTACH 'x.db'"],
)
def test_write_queries_are_rejected(store: TableStore, sql: str) -> None:
    with pytest.raises(ValueError):
        store.run_sql(sql)


def test_limit_is_injected(store: TableStore) -> None:
    _, rows = store.run_sql("SELECT * FROM q3_sales", max_rows=3)
    assert len(rows) == 3


def test_table_card_mentions_columns(store: TableStore) -> None:
    card = store.schema_text(["q3_sales"])
    assert "revenue_usd" in card and "region" in card
