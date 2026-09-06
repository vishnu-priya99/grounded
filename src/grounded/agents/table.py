"""Table / Data agent.

Turns a question into one read-only DuckDB query, executes it, and self-corrects
on error. The SQL string is returned as provenance — the answer can show exactly
how a number was computed. No arbitrary Python is ever run.
"""

from __future__ import annotations

from pydantic import BaseModel

from grounded.agents.state import AgentState, TableAnswer
from grounded.config import settings
from grounded.ingest.pipeline import get_stores
from grounded.llm import chat_structured

_SYSTEM = """You write a single DuckDB SQL query to answer the user's question \
from the tables below. Rules:
- SELECT / WITH only. No INSERT/UPDATE/DELETE/DDL.
- Use exact table and column names as given (quote if they contain spaces).
- Prefer explicit aggregations; add ORDER BY for "top"/"most"/"least".
- The "Sample rows" shown for each table are a partial preview, never the full
  table. Do NOT conclude that a value, category or row is absent because it is
  not in the sample — write the query and let it run. Only return sql="" when
  the tables genuinely lack the needed columns.
- If the question asks about a metric, entity or concept that no table has a
  column for, return sql="" — do NOT substitute a loosely-related column (e.g.
  do not answer an "ARPP" or "revenue per user" question by dividing two
  unrelated columns that happen to be present).
- If the question cannot be answered from these tables, return sql="" and \
explain in `note`."""


class _SQL(BaseModel):
    sql: str
    note: str = ""


def run_table(state: AgentState) -> AgentState:
    decision = state["decision"]
    if not decision.needs_tables:
        return {"tables": [], "notes": ["table: skipped"]}

    store = get_stores().tables
    if not store.has_tables():
        return {"tables": [], "notes": ["table: no spreadsheets ingested"]}

    # Honour "Restrict answers to": the table agent must only see — and only be
    # able to query — spreadsheets in the active scope. Without this it happily
    # answers a Meta-10-K question off an unrelated Acme sheet that the user
    # explicitly excluded.
    files = state.get("source_files")
    metas = store.list_tables()
    if files:
        metas = [m for m in metas if m.source_file in files]
    if not metas:
        return {"tables": [], "notes": ["table: no spreadsheets in scope"]}
    allowed = {m.table_name for m in metas}

    schema = store.schema_text(sorted(allowed))
    question = decision.rewritten_question or state["question"]
    result = TableAnswer(question=question)
    error = ""

    for attempt in range(1, settings.max_sql_retries + 2):
        result.attempts = attempt
        user = f"Tables:\n{schema}\n\nQuestion: {question}"
        if error:
            user += f"\n\nYour previous query failed with:\n{error}\nFix it."
        gen = chat_structured(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            _SQL,
            fallback=_SQL(sql="", note="could not generate SQL"),
        )

        if not gen.sql.strip():
            result.error = gen.note or "question not answerable from the tables"
            break

        result.sql = gen.sql.strip()
        try:
            cols, rows = store.run_sql(result.sql, allowed_tables=allowed)
            result.columns, result.rows, result.error = cols, rows, ""
            break
        except Exception as exc:  # noqa: BLE001 - feed back to the model
            error = str(exc)
            result.error = error

    note = (
        f"table: {'ok' if not result.error else 'failed'} "
        f"after {result.attempts} attempt(s)"
    )
    return {"tables": [result], "notes": [note]}
