"""Normalized document model.

Parsers for every file type converge on :class:`DocModel` — an ordered list of
:class:`Element` objects carrying enough metadata (page, section path, kind) to
support layout-aware chunking and precise citations. Nothing downstream of the
parsers branches on file type again.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

ElementKind = Literal[
    "title",
    "heading",
    "paragraph",
    "list_item",
    "table",
    "figure",
    "code",
    "caption",
]


class SourceKind(StrEnum):
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"
    CODE = "code"
    WEB = "web"


class Element(BaseModel):
    """One structural unit of a document."""

    kind: ElementKind
    text: str
    page: int | None = None
    """1-indexed page (PDF/PPTX slide). ``None`` for formats without pages."""
    heading_level: int | None = None
    """1..6 for ``heading``/``title`` elements."""
    section_path: list[str] = Field(default_factory=list)
    """Breadcrumb of enclosing headings, outermost first."""
    bbox: tuple[float, float, float, float] | None = None
    """(x0, y0, x1, y1) in PDF points, when the parser provides it."""
    table_markdown: str | None = None
    """For ``table`` elements: the table rendered as GitHub-flavored markdown."""
    ordinal: int | None = None
    """Position of an enumerated item ("3." in a numbered list / "Question 3"),
    as written in the document. Set by :mod:`grounded.ingest.enumerate`; lets a
    query like "the third question" resolve to an exact chunk."""
    image_ref: str | None = None
    """``"<doc_id>/<id>.png"`` — set when this element IS a vision-read image
    (a rendered chart page, an OCR'd embedded figure) or has been linked to one
    on the same page by :func:`grounded.ingest.enumerate._attach_page_figures`.
    Resolve with :func:`grounded.util.image_store_path`. Lets an agent (the
    fallback agent, notably) re-read the actual source image instead of
    trusting only its text description."""
    shared_context: str | None = None
    """For a question that belongs to a shared-stimulus set ("Answer Questions
    24-28 based on the following graph"): the stimulus plus every sibling's
    stem, set by :func:`grounded.ingest.enumerate._group_shared_stimulus`.
    Becomes the chunk's ``parent_text`` so answering any one member carries the
    context the whole set depends on (e.g. what an unlabelled axis represents,
    only inferable across the set)."""


class DocModel(BaseModel):
    """A parsed document, format-agnostic."""

    doc_id: str
    source_file: str
    source_kind: SourceKind
    title: str
    elements: list[Element] = Field(default_factory=list)
    page_count: int | None = None
    extra: dict[str, str] = Field(default_factory=dict)
    """Free-form parser notes, e.g. ``{"language": "python"}``."""


class Chunk(BaseModel):
    """A retrievable unit with everything a citation needs."""

    chunk_id: str
    doc_id: str
    source_file: str
    source_kind: SourceKind
    text: str
    """Embedded text, breadcrumb prefix included."""
    body: str
    """The chunk content without the breadcrumb prefix (shown to the user)."""
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    element_kind: ElementKind = "paragraph"
    ordinal: int | None = None
    """The item number this chunk holds, when it is one entry of an enumerated
    list ("Question 3"). Used to answer positional questions exactly."""
    image_ref: str | None = None
    """See ``Element.image_ref`` — carried through by ``chunking.py`` when the
    chunk is a single element (a figure's own chunk, or a linked enumerated
    item)."""
    parent_text: str | None = None
    """Enclosing section, used for context expansion at answer time."""

    def citation_label(self) -> str:
        loc = f"p.{self.page}" if self.page is not None else (
            " > ".join(self.section_path) or "start"
        )
        return f"{self.source_file} ({loc})"
