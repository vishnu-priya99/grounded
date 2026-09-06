# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Grounded is a multi-agent document-chat system with a local-first data layer
(embedded Qdrant, DuckDB, local embeddings, the index and the files never leave
the machine): upload PDFs/DOCX/PPTX/HTML/spreadsheets/images/code, ask questions
across all of them, get answers with inline `[n]` citations and a confidence
score, and an explicit refusal when the answer isn't in the documents. The LLM is
provider-swappable (`llm.py`): ships on Claude, `LLM_PROVIDER=ollama` for local.
Full narrative in `README.md`; this file is the operating manual for the code.

**Hardware constraint that shapes every dependency choice:** built for a 16 GB RAM
laptop, no discrete GPU, **no Docker/WSL2**. Every library swap in this repo (see
"Key design decisions" in README) traces back to that budget. Don't add a
PyTorch/CUDA dependency or assume a container is available.

## Commands

```bash
# setup (first time)
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
cp .env.example .env

# generate the demo corpus (PDF/DOCX/XLSX/PPTX/PNG incl. bar/column/line/pie
# charts; the .md/.txt/.csv/.html samples are checked in), then ingest it
.venv/Scripts/python.exe scripts/make_samples.py
.venv/Scripts/python.exe -m grounded.cli ingest "data/samples/*"

# run the app
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
# or: .\run.ps1   (Windows one-command launcher; .\run.ps1 -Ingest to also (re)ingest)

# CLI (no UI), fastest way to sanity-check a change
.venv/Scripts/python.exe -m grounded.cli ask "question here"
.venv/Scripts/python.exe -m grounded.cli search "query"
.venv/Scripts/python.exe -m grounded.cli stats

# tests
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pytest -q tests/test_agents.py::test_numbers_normalises_scale_suffixes  # single test

# lint / typecheck
.venv/Scripts/python.exe -m ruff check src/ tests/ app/ eval/ scripts/
.venv/Scripts/python.exe -m mypy src/

# gold-set evaluation (32 items, ~30 min on CPU)
.venv/Scripts/python.exe eval/run_eval.py
```

A `Makefile` mirrors these (`make setup|ingest|run|test|lint|eval|clean`) for a
Unix-like shell; on Windows, `run.ps1` and the raw commands above are the primary
path; `make` isn't assumed to be installed.

Tests that touch storage isolate themselves with
`monkeypatch.setattr(settings, "storage_dir", tmp_path)` (see
`test_reliability.py`, `test_tables.py`) rather than writing into the real
`.grounded/`. Follow that pattern for any new test that ingests or indexes.

After any ingest/router/synthesis change, the discipline is: run `pytest`
(83 tests), then re-run the hand battery in **`test_questions.md`** in the UI
(PART A = 10 questions covering every format; PART B = 25 covering the exam +
Meta 10-K, with expected answers and which rows should refuse) and back out any
regression. Several regressions here only showed up in the full battery, not the
unit tests (a loose regex once deleted two exam questions; growing the corpus to
14 docs broke the exam chart).

## Architecture

### Two pipelines, one index

**Ingestion** (`ingest/pipeline.py` → format router) branches on file type:
- Prose (PDF/DOCX/PPTX/HTML/TXT/MD/images/code) → `parsers.py` (format-specific,
  tags every element `heading`/`table`/`code`/`paragraph`/`figure` using each
  format's own structure, font size for PDF, styles for DOCX; PDF also strips
  running headers/footers so page furniture never becomes a bogus heading).
  **HTML** (`_parse_html`) is `html2text`'d to Markdown and run through the
  Markdown path, extension-routed only (no magic-byte sniff), so a `.txt` that
  opens with `<html>` is unaffected. **Native PPTX charts** (`_pptx_chart_element`)
  are read straight from the chart XML (categories × series → markdown table +
  caption); they were invisible before. PDF images are OCR'd (multi-scale, small
  images only) and, when a vision provider is configured, read by a VLM; a page
  whose drawings form a **bar-chart shape** (`_has_bar_chart`: ≥4 same-width,
  evenly-spaced, bottom-aligned rects **whose heights vary**: a row of equal
  boxes is labels, not bars) or is mostly-graphic is rendered and vision-read as
  a `figure` (gated + capped 4/doc). The chart-vision prompt asks for a
  plain-language caption sentence ("This chart shows Family of Apps (FoA) revenue
  by quarter…") **before** the data table, so the chunk retrieves for the words a
  question actually uses.
  Then `ingest/enumerate.py` (`split_enumerated`): a run of ≥4 sequentially-
  numbered items is re-expanded into one `list_item` per item, each with an
  `ordinal`; plus `_reorder_scrambled_options` (2-col MCQ option order),
  `_relocate_shared_preambles` ("Answer Questions 24-28 based on…" moves to the
  item it introduces), `_group_shared_stimulus` (for a set introduced by "Answer
  Questions N-M based on the following {chart/figure/table/passage}": folds the
  stimulus BODY, the "Figure 1: …" caption **and** the chart's OCR/vision data,
  matched by an `image_ref`, `_STIMULUS_BODY_RE`, or a bare-number
  `_is_data_dump`, plus every sibling's stem into each member's `shared_context`,
  so the questions carry the chart independent of retrieval luck),
  `_number_unlabelled_items` (image puzzles get synthetic ordinals),
  `reading_order` (number parts by page position). Conservative: every pass
  no-ops when its pattern is absent.
  → `chunking.py` (splits **only** on those pre-existing tags, never re-analyzes
  layout itself; tables, code, figures, and enumerated items are atomic) →
  embedded (`nomic-embed-text`) → stored. Vision-read image files are kept under
  `.grounded/images/<doc_id>/` for a later re-read.
- Spreadsheets (CSV/XLSX/TSV) → `ingest/tables.py`: real rows go into DuckDB as a
  queryable table; a small text "card" (schema + a **15-row preview**, labelled
  "preview only, N rows") is the *only* thing that also enters the search index,
  so a plain-English question can discover the table exists. The card is
  discovery-only; the Table agent later queries the real DuckDB table directly.

Both paths converge on `index/store.py`'s `HybridStore`, which holds every chunk
regardless of origin.

### Retrieval is hybrid, fused with RRF

`HybridStore.search()` runs two independent rankings per query and merges them.
This is the retrieval mental model to keep in mind for any change here:
- **Dense**: embedded Qdrant (`qdrant_client`, local/`path=` mode, no server
  process). `QDRANT_URL` in `.env` switches to a standalone server instead
  (`localhost:6333`, for the dashboard); empty (default) is embedded.
- **Sparse**: in-process `rank_bm25.BM25Okapi`, rebuilt from all chunks (scrolled
  out of Qdrant) whenever the store loads or a chunk is added.
- **Fusion**: Reciprocal Rank Fusion, `score = Σ 1/(rrf_k + rank)` per list a
  chunk appears in (`store.py::_fuse`, `rrf_k=60` in `config.py`). A chunk ranked
  decently in both lists beats one ranked #1 in only one.

**Embedding gotcha:** `embeddings.py` prefixes text asymmetrically:
`"search_document: "` when indexing, `"search_query: "` when embedding a
question. This is required by `nomic-embed-text` for good retrieval quality.
It's silent, not enforced: a new embedding call site that skips the right
prefix won't error, it'll just quietly degrade dense search relevance.

**Operational gotcha:** embedded Qdrant takes an exclusive lock on
`.grounded/qdrant/` for as long as one process has it open, the same constraint
DuckDB has on `.grounded/tables.duckdb` while writable. Don't run the Streamlit
app, the CLI, and the eval harness against the same `.grounded/` simultaneously;
close one before starting another (DuckDB is opened read-only for queries and
briefly read-write only during ingest, so DuckDB *reads* can overlap, Qdrant
embedded mode cannot, for either reads or writes).

`tools/qdrant/` (gitignored) is a standalone `qdrant.exe` from an earlier
experiment with server mode, for the web dashboard at `localhost:6333`. It is
**not** part of the default running path, `QDRANT_URL` is empty in `.env`, so
the app runs embedded, no server needed. Don't assume it needs to be started;
only relevant if `QDRANT_URL` gets set again.

### Six-agent LangGraph pipeline (the sixth is conditional)

`agents/graph.py` wires a mostly-linear graph (`router → retrieval → table →
synthesis → verification`), then a **conditional edge to `fallback`** and END.
Every node reads/writes a shared typed `AgentState` dict and no-ops when the
router's decision makes it irrelevant (e.g. retrieval and verification both skip
on `intent="chitchat"`). Read `graph.py` first when tracing a question
end-to-end; each agent file is small and single-purpose.

Key behaviors worth knowing before touching an agent:
- **Router** (`router.py`) sets `intent`, rewrites the question to stand alone
  using chat history, and produces `sub_queries` + `needs_tables`. It also
  produces `ordinals: list[int]` for positional questions ("the third question",
  "question 2 and 3", "the last one" → `-1`): **LLM-extracted via the structured
  output**, no regex (empirically Haiku 4.5 handles multi-item lists reliably; an
  earlier regex was removed). `run_router` reads the store for the live document
  list and any synthetic puzzle ordinals, and remaps "the Nth pattern question"
  to the Nth item of that set (it would otherwise collide with real item N).
- **Retrieval** (`retrieval.py`) has a safety net: if a spreadsheet "card" gets
  matched by search even though the router didn't set `needs_tables`, it promotes
  the question to `table_qa` anyway. When the router set `ordinals`, it also pins
  the exact enumerated-item chunk(s) (`store.chunks_by_ordinal`) to the front of
  `passages` and marks each `r.components["pinned"] = 1.0`, so synthesis can't
  answer "question 3" from a lexically-similar neighbour and the fallback can
  identify exactly which passages were the targets.
- **Fallback** (`fallback.py`): the ONE deliberate exception to "never guess".
  Fires only when verification refused AND `decision.ordinals` is non-empty AND a
  pinned passage is a multiple-choice question (≥3 lettered options via
  `_looks_like_mcq`) or an image puzzle (`_is_image_task`: has `image_ref`, no
  lettered options). Solves each item from the model's own knowledge
  `FALLBACK_VOTES` times (default 3) and takes the majority; all-disagree → an
  honest "different every time" decline; a "Final answer: cannot determine
  confidently" line is also respected. Uses `FALLBACK_MODEL` (e.g.
  `claude-sonnet-5`) and re-reads the pinned chunk's `image_ref` via
  `chat_with_image` when there is one. Output is always labelled
  `🧮 Not in your documents`.
- **Table agent** (`table.py`) only writes `SELECT`/`WITH` SQL (regex-enforced in
  `tables.py`, no arbitrary code execution) and self-corrects up to
  `max_sql_retries` on a DuckDB error. It **honours "Restrict answers to"**:
  filters `store.list_tables()` by `state["source_files"]`, only shows the model
  the scoped schema, and passes `run_sql(allowed_tables=…)` which rejects a query
  that names an out-of-scope table, without this it answered a scoped-out
  question off an unrelated sheet. Prompt also bars substituting a loosely-related
  column for a metric no table has (an "ARPP" question must not become "divide two
  present columns"), and treats sample rows as a preview (don't refuse for a value
  not shown, run the query).
- **Verification** (`verification.py`) is the faithfulness gate: computes
  `confidence = 0.55*support_ratio + 0.20*cite_coverage + 0.25*retrieval_signal`
  per sentence-to-citation check. Two independent conditions, checked in order:
  (1) refuses outright (`_REFUSAL`, `refused=True`) if nothing was cited/supported
  at all; (2) otherwise, if `confidence < settings.confidence_threshold`, the
  answer still goes through, it's only prepended with a low-confidence warning,
  never blocked. This is *not* an LLM-judge call; it's similarity/keyword-recall
  based, cheaper and more reliable than a second 7B call on this hardware.
- **Synthesis** (`synthesis.py`) returns a structured `SynthesisResult(found:
  bool, answer: str)`, `found=False` emits `_REFUSAL`, which is what the
  verification/graph refusal path and the fallback routing key on (replaced a
  fragile exact-magic-string check). The prompt bars two specific guesses: an
  unmarked MCQ/puzzle in the context is NOT an answer, and a figure must not be
  **back-calculated** from a percentage change / growth rate / ratio (a real,
  observed failure: retrieval surfaces "revenue grew 22%" prose instead of the
  segment table, and the model would invent the base figure with verification
  rubber-stamping it). Context assembly, before the model call:
  - a **successful table result leads** the context, labelled the authoritative
    answer (a lone SQL result buried under prose from other files got treated as
    noise and refused);
  - when the router's `intent == "table_qa"`, the retrieved passages (all
    cross-document, spreadsheet chunks are stripped upstream) are **dropped
    entirely**; for a promoted/`mixed` question a ≥2-shared-keyword filter runs
    instead;
  - when the question says **chart / graph / plot / axis**, the top passage's
    `source_file` fixes which document's chart it means, and **other documents'
    chart/figure/table chunks are dropped** (`_CHART_QUESTION_RE`,
    `_CHART_CAPTION_RE`): "the chart" is one chart, and a 10-K ARPP question was
    mixing in an Acme revenue chart. Prose from other files always stays.

### Structured LLM output is schema-driven, not per-prompt

`llm.chat_structured()` is the one place every agent gets typed JSON back from
the model. It introspects the Pydantic model's `model_fields` to build the "reply
with exactly these keys" instruction and a shape example, **the expected output
schema is never hand-written into an agent's own system prompt**; a new
structured-output agent just needs a Pydantic model and `chat_structured(...,
schema=YourModel)`. It also retries once with a repair message and accepts a
`fallback` value, because small local models are inconsistent JSON emitters.

`llm.chat()` picks the backend from `settings.llm_provider` (`ollama` default,
or `anthropic` / `openai`, `pip install -e ".[cloud]"` + a key). On the
`anthropic` path: `_resolve_model` swaps a stale Ollama tag for a `claude-*`
default, `temperature` is **not** sent (the Claude 5 family rejects it), and
`json_mode` becomes a system-prompt instruction. `llm.chat_with_image()` /
`vision_available()` add the vision path (Anthropic, or Ollama+`VLM_MODEL`);
`_track_anthropic_usage` accumulates real per-call cost (`usage_summary()`,
shown in the Streamlit sidebar). Nothing above `llm.py` knows which provider is
live. This project currently runs on `anthropic` (`claude-haiku-4-5` for
router/synthesis, `FALLBACK_MODEL=claude-sonnet-5` for the fallback solve),
the follow-up brief explicitly allowed paid keys.

### Conversation memory lives in the UI layer only

`memory.py`'s `SessionMemory` keeps the last 4 turns verbatim plus an
LLM-folded running summary of everything older (`context()` feeds the Router's
history-aware rewrite). It's only instantiated in `app/streamlit_app.py`,
`cli.py`'s `ask` never passes `history=`, so every CLI question is stateless,
answered as if it were the first turn. Keep this in mind before assuming CLI and
UI behavior match on a multi-turn/follow-up question, they intentionally don't.

### One config surface, three entry points

`config.py`'s `settings` singleton (env/`.env`-backed) is the only place that
reads environment variables, nothing else calls `os.environ` directly. The UI
(`app/streamlit_app.py`), CLI (`cli.py`), and eval harness (`eval/run_eval.py`)
are three separate front doors that all call the same
`agents.graph.answer(question, ...)`, behavior differences between them almost
always mean divergence should be fixed in `graph.py`/agents, not per-entry-point
(conversation memory, above, is the one deliberate exception).
