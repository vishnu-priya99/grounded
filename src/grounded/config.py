"""Central configuration, loaded from environment / .env.

Every tunable lives here so the rest of the codebase never reads ``os.environ``
directly. Import the module-level ``settings`` singleton.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Absolute path — a relative ".env" is resolved against the process's
        # current working directory, not the project root, so launching from
        # one folder up (or anywhere but the repo root) silently finds no
        # file and falls back to every hardcoded default (ollama/qwen2.5:7b)
        # with no error. This makes the config load the same regardless of
        # where `streamlit run` / `python -m grounded.cli` was invoked from.
        env_file=str(_REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore",
    )

    # --- paths ---
    data_dir: Path = _REPO_ROOT / "data"
    storage_dir: Path = _REPO_ROOT / ".grounded"
    """Where the embedded Qdrant collection, DuckDB and the session DB are persisted."""

    # --- models (all served by a local Ollama unless llm_provider says otherwise) ---
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    router_model: str = "qwen2.5:7b"
    """Set to ``qwen2.5:1.5b`` for a faster (weaker) planner."""
    fallback_model: str = ""
    """Model for the fallback agent's actual solve step (chart reading, pattern
    puzzles, tricky MCQs) — where model quality matters most. Empty = use
    ``llm_model``. Set to e.g. ``claude-sonnet-5`` to spend more only here."""
    fallback_votes: int = 3
    """Self-consistency: how many times the fallback solves each targeted item,
    taking the majority answer. A single call is flaky on reasoning/puzzle
    questions. 1 disables voting (cheapest); 3 is a good reliability/cost
    balance; higher for hard sets."""
    embed_model: str = "nomic-embed-text"
    vlm_model: str = ""
    """e.g. ``moondream`` or ``qwen2.5vl:3b``. Empty disables image captioning."""

    llm_provider: str = "ollama"
    """``ollama`` | ``openai`` | ``anthropic`` — the latter two need an API key."""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- retrieval ---
    qdrant_url: str = ""
    """If set (e.g. http://localhost:6333), connect to a standalone Qdrant
    server instead of embedded/local mode. Needed for the web dashboard.
    Empty = embedded mode, no extra process required."""
    embed_dim: int = 768
    retrieve_candidates: int = 40
    """How many hits each of dense/sparse contributes before fusion."""
    top_k: int = 8
    """Passages handed to synthesis after fusion (+ rerank, if enabled)."""
    enable_rerank: bool = False
    rrf_k: int = 60
    broad_max_tokens: int = 6000
    """Token budget for a "broad" question ('what topics does this cover',
    'list every X') — these need to survey a whole document, not just the
    usual top-k passages. Capped independently of document size: a document
    bigger than this budget gets an honestly-partial sample, not silently
    truncated context passed off as complete."""

    # --- chunking ---
    chunk_target_tokens: int = 480
    chunk_overlap_tokens: int = 64
    max_context_expansion_tokens: int = 1600
    """Ceiling on ``parent_text`` (the enclosing-section context a chunk expands
    to at answer time). Independent of document size: a report with sparse
    headings can have one section spanning many pages — without this cap,
    every chunk retrieved from it would drag that whole section into every
    prompt, regardless of how large the source document is. ~3-4x a normal
    chunk, generous enough to still add real context."""

    # --- ingestion guardrails (for arbitrary user uploads) ---
    max_file_mb: int = 50
    max_pdf_pages: int = 300
    max_table_rows: int = 100_000
    max_image_px: int = 4000
    """Longest edge; larger images are downscaled before OCR."""

    # --- agents ---
    confidence_threshold: float = 0.55
    max_sql_retries: int = 2
    fast_mode: bool = False
    """Skip rerank + verification for latency."""
    llm_num_ctx: int = 8192
    llm_temperature: float = 0.1

    def ensure_dirs(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "index").mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "images").mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def duckdb_path(self) -> str:
        return str(self.storage_dir / "tables.duckdb")


settings = Settings()
