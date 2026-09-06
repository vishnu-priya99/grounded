"""Small shared helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from grounded.config import settings


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — good enough for chunk sizing
    without pulling in a tokenizer dependency."""
    return max(1, len(text) // 4)


def stable_id(*parts: str) -> str:
    """Deterministic short id from arbitrary string parts."""
    h = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def image_store_path(ref: str) -> Path:
    """Resolve an ``Element``/``Chunk.image_ref`` (a ``"<doc_id>/<id>.png"``
    relative path) to its file on disk under ``storage_dir/images``. Shared by
    ingestion (which writes here — see ``ingest/parsers.py``) and the fallback
    agent (which reads back the raw bytes for a vision re-read at answer time
    — see ``agents/fallback.py``) so both sides agree on layout in one place."""
    return settings.storage_dir / "images" / ref


_WS = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")

_SENT_BOUND = re.compile(r"(?<=[.!?])\s+")
_LEAD_CITES = re.compile(r"^((?:\[\d+\]\s*)+)")
_ONLY_CITES = re.compile(r"^(?:\[\d+\]\s*)+$")


def split_sentences(text: str) -> list[str]:
    """Split into sentences while keeping inline ``[n]`` citation markers attached
    to the sentence they belong to (the one *before* them)."""
    parts = [p.strip() for p in _SENT_BOUND.split(text.strip()) if p.strip()]
    out: list[str] = []
    for part in parts:
        lead = _LEAD_CITES.match(part)
        if out and lead:
            out[-1] = f"{out[-1]} {lead.group(1).strip()}".strip()
            part = part[lead.end():].strip()
        if not part:
            continue
        if out and _ONLY_CITES.match(part):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()
