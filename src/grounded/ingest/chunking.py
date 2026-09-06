"""Layout-aware chunking.

Rules:
- Split on heading boundaries, not fixed windows.
- Never split a table or a code block.
- Prepend the section breadcrumb to every chunk so an isolated passage still
  carries context (and so lexical search can match on section titles).
- Carry page + section metadata through untouched — it becomes the citation.
- Keep the enclosing section as ``parent_text`` for context expansion.
"""

from __future__ import annotations

import re

from grounded.config import settings
from grounded.ingest.models import Chunk, DocModel, Element
from grounded.util import estimate_tokens, stable_id


def chunk_document(doc: DocModel) -> list[Chunk]:
    chunks: list[Chunk] = []
    # Group elements into sections delimited by headings.
    sections: list[tuple[list[str], list[Element]]] = []
    current_path: list[str] = []
    current: list[Element] = []

    for el in doc.elements:
        if el.kind in {"heading", "title"}:
            if current:
                sections.append((list(current_path), current))
                current = []
            current_path = el.section_path or [el.text]
            current.append(el)
        else:
            current.append(el)
    if current:
        sections.append((list(current_path), current))

    for path, els in sections:
        parent_text = _section_text(els)
        breadcrumb = _breadcrumb(doc.title, path)
        # Split the section into runs of flowing text separated by atomic
        # elements (tables, code) that must each stand alone.
        runs: list[list[Element]] = [[]]
        for el in els:
            if el.kind in {"heading", "title"}:
                continue
            if el.kind in {"table", "code", "figure"} or _is_enumerated_item(el):
                runs.append([el])
                runs.append([])
            else:
                runs[-1].append(el)

        for run in runs:
            if not run:
                continue
            atomic = run[0].kind in {"table", "code", "figure"} or _is_enumerated_item(run[0])
            pieces = [run] if atomic else _pack(run)
            for piece in pieces:
                chunks.append(_make_chunk(doc, piece, path, breadcrumb, parent_text))

    return chunks


def _is_enumerated_item(el: Element) -> bool:
    """An entry of a numbered list, split out by ``ingest.enumerate`` — kept
    whole in its own chunk, exactly like a table or code block."""
    return el.kind == "list_item" and el.ordinal is not None


def _breadcrumb(title: str, path: list[str]) -> str:
    parts = [title, *[p for p in path if p and p != title]]
    return " > ".join(dict.fromkeys(parts))


def _section_text(els: list[Element]) -> str:
    """The enclosing section, for context expansion at answer time — capped so
    a section's size tracks the document's heading density, not its page
    count. A sparsely-headed 200-page report must not turn one retrieved
    chunk into a 50,000-token prompt."""
    text = "\n\n".join(el.text for el in els if el.text).strip()
    limit = settings.max_context_expansion_tokens
    if estimate_tokens(text) > limit:
        text = text[: limit * 4].rstrip() + " …[section truncated for length]"
    return text


_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _explode(el: Element, limit: int) -> list[Element]:
    """Break one over-long text element into paragraph- then sentence-sized pieces."""
    if estimate_tokens(el.text) <= limit:
        return [el]
    pieces: list[str] = []
    for para in el.text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if estimate_tokens(para) <= limit:
            pieces.append(para)
            continue
        buf = ""
        for sent in _SENT_BOUNDARY.split(para):
            if buf and estimate_tokens(buf + " " + sent) > limit:
                pieces.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            pieces.append(buf)
    return [el.model_copy(update={"text": p}) for p in pieces] or [el]


def _pack(els: list[Element]) -> list[list[Element]]:
    """Greedily pack paragraphs up to the target token budget with overlap."""
    target = settings.chunk_target_tokens
    overlap = settings.chunk_overlap_tokens
    els = [piece for el in els for piece in _explode(el, target)]
    out: list[list[Element]] = []
    cur: list[Element] = []
    cur_tok = 0
    for el in els:
        tok = estimate_tokens(el.text)
        if cur and cur_tok + tok > target:
            out.append(cur)
            # carry the tail for overlap
            carry: list[Element] = []
            carry_tok = 0
            for prev in reversed(cur):
                carry_tok += estimate_tokens(prev.text)
                carry.insert(0, prev)
                if carry_tok >= overlap:
                    break
            cur = list(carry)
            cur_tok = carry_tok
        cur.append(el)
        cur_tok += tok
    if cur:
        out.append(cur)
    return out


def _make_chunk(
    doc: DocModel,
    els: list[Element],
    path: list[str],
    breadcrumb: str,
    parent_text: str,
) -> Chunk:
    body = "\n\n".join(el.text for el in els if el.text).strip()
    page = next((el.page for el in els if el.page is not None), None)
    kind = els[0].kind if els else "paragraph"
    ordinal = els[0].ordinal if len(els) == 1 else None
    image_ref = els[0].image_ref if len(els) == 1 else None
    shared_context = els[0].shared_context if len(els) == 1 else None

    label = breadcrumb
    if ordinal is not None:
        tag = f"item {ordinal}"
        label = f"{breadcrumb} · {tag}" if breadcrumb else tag
    text = f"[{label}]\n{body}" if label else body

    cid = stable_id(doc.doc_id, breadcrumb, body[:120], str(page), str(ordinal))
    return Chunk(
        chunk_id=cid,
        doc_id=doc.doc_id,
        source_file=doc.source_file,
        source_kind=doc.source_kind,
        text=text,
        body=body,
        page=page,
        section_path=path,
        element_kind=kind,
        ordinal=ordinal,
        image_ref=image_ref,
        # An enumerated item is meant to stand alone; don't let context
        # expansion pull in every sibling item in the list — the one
        # exception is a shared-stimulus set (a chart + the questions about
        # it), where the stimulus IS the context the item can't be answered
        # without (see ingest/enumerate._group_shared_stimulus).
        parent_text=_shared_or_section(shared_context, parent_text, body, ordinal),
    )


def _shared_or_section(
    shared_context: str | None, parent_text: str, body: str, ordinal: int | None
) -> str | None:
    if shared_context:
        limit = settings.max_context_expansion_tokens
        return (
            shared_context if estimate_tokens(shared_context) <= limit
            else shared_context[: limit * 4].rstrip() + " …[set truncated for length]"
        )
    if ordinal is not None:
        return None
    return parent_text if parent_text != body else None
