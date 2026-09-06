"""Rebuild enumerated lists ("1. … 2. … 3. …", exam questions, numbered steps)
into one :class:`Element` per item, each carrying an ``ordinal``.

Why this pass exists: the format parsers tag structure by font size / paragraph
style, so a run of numbered questions written in body text stays as ordinary
paragraphs — frequently several items to one block, or one item split across a
page break. Nothing downstream then corresponds to "question 3", and a
positional query ("what's the answer to the third question?") has no chunk to
land on. Splitting the run here lets :mod:`grounded.ingest.chunking` keep each
item whole (like it already does for tables and code) and lets the retrieval
agent target an item by its number.

Conservative by design: it only fires on a genuine sequential run, and leaves
every other element untouched.
"""

from __future__ import annotations

import re

from grounded.ingest.models import DocModel, Element

# Start-of-line "3." / "3)" followed by a letter or quote/bracket. A letter (not
# a digit) after the delimiter rules out decimals ("1.20") and page-mangled
# runs; the space between "3." and the text is often lost in PDF extraction, so
# it's optional.
_ITEM_RE = re.compile(r"(?m)^[ \t]*(\d{1,3})[.)][ \t]*(?=[A-Za-z\"'(\[])")

_FLOWING = {"paragraph", "list_item"}

# A short instruction that introduces a BATCH of sub-parts to work through —
#   "For each of the following, which number replaces the question mark?"
#   "Complete each of the following sequences:"
#   "Find the missing value in each of these:"
# The "for each of the following"-style construction is the discriminator: a
# single numbered question's stem, or form-filling boilerplate ("fill in the
# following details"), doesn't have it. Anchors _number_unlabelled_items
# below and, in split_enumerated, ends the preceding numbered run so the
# batch's image parts aren't absorbed into the last question's chunk.
_BATCH_CUE_RE = re.compile(
    r"\bfor (?:each|all) of the following\b"
    r"|\beach of the following\b"
    r"|\bin each of (?:the following|these)\b"
    r"|\bcomplete (?:the|each) (?:following|sequence)",
    re.I,
)
_OPTION_MARK_RE = re.compile(r"(?:^|\s)[a-d][.)]\s", re.M)
_NUMBERED_STEM_RE = re.compile(r"\b\d{1,3}\.\s*[A-Z]")


def _is_batch_prompt(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 160 or t[0].isdigit():
        return False
    if not t.endswith(("?", ":")):
        return False
    if not _BATCH_CUE_RE.search(t):
        return False
    # Not question content that merely happens to say "the following".
    return not (_OPTION_MARK_RE.search(t) or _NUMBERED_STEM_RE.search(t))


_MIN_ITEMS = 4  # fewer than this isn't worth restructuring

# "a. text  c. text" / "b. text  d. text" — a 2-column MCQ option layout
# extracts in visual row order (a, c, b, d), not answer order, because the
# parser has no notion of columns, only a left-to-right/top-to-bottom scan.
# That reordering is invisible to a human skimming the PDF (the columns are
# obviously separate) but is just a flat, confusingly-ordered token sequence
# to a model — and was traced to a real wrong-answer case (a fabricated
# option that blended text from two different real ones).
# Requires the letter be preceded by whitespace/line-start, not just "not a
# letter or digit" — a question stem is full of things like "R(a,b)" or
# "R.a)" (a relation schema, a qualified column reference) where the char
# right before the letter is "(", "," or "." — none of them alphanumeric, so
# a blocklist of excluded characters kept missing cases. Requiring
# whitespace before the letter is simpler and actually correct: a real
# option always starts a new line or follows a space after the previous
# option's text.
_OPTION_SCAN_RE = re.compile(r"(?:(?<=\s)|^)([a-d])[.)]\s*", re.M | re.I)


def _reorder_scrambled_options(text: str) -> str:
    """If ``text`` contains a-d option markers out of order, restore a/b/c/d
    order — each option's own text kept verbatim, just relocated. A no-op
    when options are already in order or too few to be confident.

    A stem sentence ending in a single capital letter ("...of Company B.",
    "A -> B and B -> C.") reads exactly like an option marker and will
    collide with that same letter's real option later in the item. Real
    options are always the last, unbroken run in an item's text, so an
    earlier occurrence of a letter is the false one — keep only each
    letter's *last* match before judging whether what's left is a clean,
    non-repeating option sequence."""
    matches = list(_OPTION_SCAN_RE.finditer(text))
    if len(matches) < 3:
        return text
    last_by_letter: dict[str, int] = {}
    for idx, m in enumerate(matches):
        last_by_letter[m.group(1).lower()] = idx
    matches = [matches[i] for i in sorted(last_by_letter.values())]
    if len(matches) < 3:
        return text
    letters = [m.group(1).lower() for m in matches]
    if letters == sorted(letters):
        return text
    bounds = [m.start() for m in matches] + [len(text)]
    spans = [
        (letters[i], text[bounds[i] : bounds[i + 1]].strip())
        for i in range(len(matches))
    ]
    preamble = text[: matches[0].start()].rstrip()
    body = "\n".join(t for _, t in sorted(spans, key=lambda x: x[0]))
    return f"{preamble}\n{body}" if preamble else body


# "For Questions 29 and 30, find..." / "Answer Questions 24-28 based on the
# following line graph." — a shared preamble that belongs to the item(s) it
# INTRODUCES, not the one it happens to trail. Common in real exams (a
# passage or figure shared by several questions) and, since a run only ever
# assigns trailing text to the PRECEDING item, always lands in the wrong
# place without this — including any data that follows it in the same run
# (a chart's OCR'd values, for one, land in the wrong question entirely
# otherwise: verified against this exact document).
_SHARED_PREAMBLE_RE = re.compile(
    r"\b(?:For|Answer)\s+Questions?\s+(\d{1,3})"
    r"(?:\s*(?:and|-|to|through)\s*(\d{1,3}))?\b[^\n]*",
    re.I,
)

# The un-numbered element that *is* the shared stimulus: a figure/table caption,
# or a chart-data transcription block.
_STIMULUS_BODY_RE = re.compile(
    r"^\s*#*\s*(figure|table|exhibit|chart|graph|diagram|passage|"
    r"chart data|data (?:points|table)|(?:x|y)-axis)\b",
    re.I,
)


def _is_data_dump(text: str) -> bool:
    """A loose OCR blob of a chart's axis ticks and data labels — mostly bare
    numbers, one per line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    numeric = sum(1 for ln in lines if re.fullmatch(r"[\d,]+(?:\.\d+)?%?", ln))
    return numeric / len(lines) > 0.6


def _relocate_shared_preambles(elements: list[Element]) -> list[Element]:
    """Move a trailing shared-preamble sentence — and everything after it in
    the same item's text — to lead the item it actually introduces, crossing
    any heading (a Figure/Table caption) that split the enumeration into
    separate runs in between. A no-op when no such sentence is found, when
    its target item isn't found, or when it isn't strictly ahead (never move
    text backward)."""
    by_ordinal = {el.ordinal: idx for idx, el in enumerate(elements) if el.ordinal is not None}
    out = list(elements)
    for idx, el in enumerate(elements):
        if el.ordinal is None:
            continue
        m = _SHARED_PREAMBLE_RE.search(el.text)
        if not m:
            continue
        target_idx = by_ordinal.get(int(m.group(1)))
        if target_idx is None or target_idx <= idx:
            continue
        head = el.text[: m.start()].rstrip()
        tail = el.text[m.start() :].strip()
        if len(head) < 5:  # would gut the item's own content — too risky
            continue
        # A vision-read image absorbed into this element's text by _expand()
        # (see image_ref_in there) physically sits after the match point in
        # every real case seen so far (a chart's data trails the "Answer
        # Questions N-M..." sentence that introduces it) — move the ref along
        # with the tail it belongs to, not left behind on the head.
        out[idx] = out[idx].model_copy(update={"text": head, "image_ref": None})
        target = out[target_idx]
        out[target_idx] = target.model_copy(update={
            "text": f"{tail}\n\n{target.text}",
            "image_ref": el.image_ref or target.image_ref,
        })
    return out


def _attach_page_figures(elements: list[Element]) -> list[Element]:
    """A chart/diagram is rendered as its own ``figure`` element (a vision
    read of the actual image — see ``ingest/parsers.py``), but it isn't itself
    a numbered item, so the ordinal-pin mechanism (``agents/retrieval.py``)
    would never surface it for "what's the answer to question 24" even when
    the chart IS that question's actual content. Tag every enumerated item on
    the SAME page with the figure's ``image_ref``, so the fallback agent can
    re-read the real image instead of guessing from text alone.

    Conservative: only when a page has exactly one figure. A page with two+
    figures is genuinely ambiguous — which question goes with which — and
    guessing wrong would be worse than not linking either."""
    by_page: dict[int, list[Element]] = {}
    for el in elements:
        if el.page is not None:
            by_page.setdefault(el.page, []).append(el)
    out = list(elements)
    for pno, page_els in by_page.items():
        figures = [el for el in page_els if el.kind == "figure" and el.image_ref]
        if len(figures) != 1:
            continue
        ref = figures[0].image_ref
        for idx, el in enumerate(out):
            if el.page == pno and el.ordinal is not None and not el.image_ref:
                out[idx] = el.model_copy(update={"image_ref": ref})
    return out


def split_enumerated(doc: DocModel) -> DocModel:
    """Return ``doc`` with any enumerated runs re-expanded into per-item
    elements. Returns the same object when nothing matched."""
    out: list[Element] = []
    run: list[Element] = []
    changed = False

    def flush_run() -> None:
        nonlocal changed
        if not run:
            return
        expanded = _expand(run)
        if expanded is None:
            out.extend(run)
        else:
            out.extend(expanded)
            changed = True
        run.clear()

    for el in doc.elements:
        # A batch prompt ("For each of the following, ... ?") ends the
        # numbered run — without this its trailing image parts get absorbed
        # into the last question's chunk and _number_unlabelled_items below
        # has nothing left to tag.
        if el.kind in _FLOWING and not _is_batch_prompt(el.text):
            run.append(el)
        else:
            flush_run()
            out.append(el)
    flush_run()

    if not changed:
        return doc
    out = _relocate_shared_preambles(out)
    out = _attach_page_figures(out)
    out = _number_unlabelled_items(out)
    out = _group_shared_stimulus(out)
    return doc.model_copy(update={"elements": out})


def _number_unlabelled_items(elements: list[Element]) -> list[Element]:
    """Give image-only sub-parts that carry no numbers of their own a synthetic
    ordinal continuing the document's main run, so each becomes a targetable
    atomic item like any numbered question and can reach the fallback solver.
    Each keeps its own ``image_ref`` (the fallback reads the actual image).

    Two ways in, because a batch's introducing prompt doesn't always survive
    PDF text extraction:
    1. a batch prompt ("For each of the following, ... ?" — see
       ``_is_batch_prompt``) with image parts right after it;
    2. structurally — a run of >=2 consecutive image-only unnumbered
       paragraphs that each contain a "?" (the missing-value slot), i.e. a
       puzzle set whether or not its prompt was extracted.

    Conservative: only in a document that already has a numbered run (so it's
    quiz-like, not a photo essay), only >=2 parts, only ``paragraph`` kind (a
    rendered ``figure`` — an answer-key grid — is excluded)."""
    ords = [el.ordinal for el in elements if el.ordinal is not None]
    if not ords:
        return elements
    counter = [max(ords) + 1]
    out = list(elements)
    tagged: set[int] = set()

    def reading_order(indices: list[int]) -> list[int]:
        """Number the parts by where they sit on the page — reading order,
        top rows first then left-to-right within a row — not by the order the
        parser emitted the images (which does NOT match the layout: "the
        first puzzle" must mean the leftmost one). A "row" is a cluster of
        tops within half a median box-height of each other, so items visually
        side by side (whose tops differ by a few points) stay one row."""
        boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        for i in indices:
            bb = elements[i].bbox
            if bb is None:
                return indices  # can't order without every position
            boxes.append((i, bb))
        heights = sorted(y1 - y0 for _, (_x0, y0, _x1, y1) in boxes)
        tol = max(20.0, heights[len(heights) // 2] * 0.5)
        by_top = sorted(boxes, key=lambda t: t[1][1])
        ranked: list[tuple[int, float, int]] = []
        band, prev_top = 0, by_top[0][1][1]
        for i, (x0, y0, _x1, _y1) in by_top:
            if y0 - prev_top > tol:
                band += 1
            prev_top = y0
            ranked.append((band, x0, i))
        return [i for _, _, i in sorted(ranked)]

    def tag(indices: list[int], label: str) -> None:
        for n, idx in enumerate(reading_order(indices), start=1):
            src = elements[idx]
            out[idx] = src.model_copy(update={
                "kind": "list_item",
                "ordinal": counter[0],
                "text": f"Sub-question {n} of {len(indices)} ({label}): {src.text}",
            })
            counter[0] += 1
            tagged.add(idx)

    def is_part(idx: int) -> bool:
        el = elements[idx]
        return (
            idx not in tagged and el.kind == "paragraph"
            and bool(el.image_ref) and el.ordinal is None
        )

    # 1) anchored on a batch prompt
    i = 0
    while i < len(elements):
        if elements[i].ordinal is None and _is_batch_prompt(elements[i].text):
            prompt = elements[i].text.strip()
            parts: list[int] = []
            j = i + 1
            while j < len(elements):
                if elements[j].kind in {"heading", "title"}:
                    break
                if is_part(j):
                    parts.append(j)
                elif len(elements[j].text.strip()) > 10:
                    break
                j += 1
            if len(parts) >= 2:
                tag(parts, f"instruction: {prompt[:100]}")
            i = max(j, i + 1)
        else:
            i += 1

    # 2) structural fallback — a run of >=2 image-only "?" paragraphs, allowing
    #    tiny separators ("? =") between them but broken by any real text or a
    #    heading (same leniency as the prompt-anchored scan above).
    run: list[int] = []

    def flush() -> None:
        if len(run) >= 2 and any("?" in elements[k].text for k in run):
            tag(run, "an image sub-question set")
        run.clear()

    for idx in range(len(elements)):
        el = elements[idx]
        if is_part(idx):
            run.append(idx)
        elif el.kind in {"heading", "title"} or len(el.text.strip()) > 10:
            flush()
    flush()

    return out if tagged else elements


def _group_shared_stimulus(elements: list[Element]) -> list[Element]:
    """Questions sharing one stimulus — a chart, a passage, a code listing
    introduced by "Answer Questions 24-28 based on ..." — are a single
    semantic unit even though each stays its own atomic chunk (so "question
    26" still pins exactly). Give every member, as ``shared_context``, the
    stimulus plus every sibling's stem, so answering any one of them carries
    the context the set depends on — e.g. what an unlabelled chart axis
    represents, which is only inferable across the whole set."""
    idx_by_ordinal = {
        el.ordinal: i for i, el in enumerate(elements) if el.ordinal is not None
    }
    text_by_ordinal = {
        el.ordinal: el.text for el in elements if el.ordinal is not None
    }
    out = list(elements)
    for pi, el in enumerate(elements):
        if el.ordinal is None:
            continue
        m = _SHARED_PREAMBLE_RE.search(el.text)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        members = [o for o in range(lo, hi + 1) if o in idx_by_ordinal]
        if len(members) < 2:
            continue
        own = re.search(rf"(?m)^[ \t]*{el.ordinal}[.)]", el.text)
        stimulus = (el.text[: own.start()] if own else el.text).rstrip()
        # The stimulus body itself — a "Figure 1: ..." caption, a chart's OCR'd
        # / vision-transcribed values — is usually its own un-numbered element
        # sitting just before the preamble (a heading between two runs) or
        # right after it. Without folding it in, members only carry the "Answer
        # Questions N-M based on the graph" sentence and the sibling stems, not
        # the data the questions actually need — and whether retrieval happens
        # to surface that loose chunk is luck that runs out on a big corpus.
        first_i = min(idx_by_ordinal[o] for o in members)
        extra = [
            e.text.strip()
            for e in elements[max(0, pi - 3): max(first_i, pi + 1)]
            if e.ordinal is None
            and e.text.strip()
            and not _SHARED_PREAMBLE_RE.search(e.text)
            and (e.image_ref or _STIMULUS_BODY_RE.match(e.text.strip())
                 or _is_data_dump(e.text))
        ]
        roster = "\n\n".join(
            f"[Q{o}] {text_by_ordinal[o].strip()}" for o in members
        )
        parts = [p for p in [stimulus, *extra] if p]
        shared = (
            "\n\n".join([*parts, f"All questions in this set:\n{roster}"])
            if parts else roster
        )
        # If the stimulus is an image (a chart), every member depends on it —
        # share the ref so the fallback can re-read the real image for any of
        # them, not just whichever item absorbed it during text-merging.
        set_ref = next(
            (elements[idx_by_ordinal[o]].image_ref for o in members
             if elements[idx_by_ordinal[o]].image_ref),
            None,
        )
        for o in members:
            i = idx_by_ordinal[o]
            update = {"shared_context": shared}
            if set_ref and not out[i].image_ref:
                update["image_ref"] = set_ref
            out[i] = out[i].model_copy(update=update)
    return out


def _expand(run: list[Element]) -> list[Element] | None:
    """Turn one run of flowing elements into [preamble?, item, item, …] when it
    holds a sequential numbered list; otherwise ``None``."""
    # Concatenate the run, remembering which element each character came from so
    # every item keeps the right page and section breadcrumb.
    spans: list[tuple[int, int, Element]] = []
    parts: list[str] = []
    cursor = 0
    for el in run:
        text = el.text or ""
        parts.append(text)
        spans.append((cursor, cursor + len(text), el))
        cursor += len(text) + 1  # +1 for the "\n" join below
    joined = "\n".join(parts)

    matches = list(_ITEM_RE.finditer(joined))
    if len(matches) < _MIN_ITEMS:
        return None

    numbers = [int(m.group(1)) for m in matches]
    seg = _main_run(numbers)
    if seg is None:
        return None
    i, j = seg  # inclusive indices into `matches`

    def loc_at(offset: int) -> Element:
        for start, end, el in spans:
            if start <= offset < end:
                return el
        return spans[-1][2]

    def image_ref_in(lo: int, hi: int) -> str | None:
        """A vision-read image (an OCR'd embedded figure — see
        ``ingest/parsers.py``) can end up as ordinary flowing text absorbed
        into this span, same as any other paragraph — the chart's page-level
        ``figure`` element never reaches here at all (it breaks the run
        instead; see ``_attach_page_figures``). Carry its ``image_ref``
        forward so the item keeps the link to its real source image, not just
        the text description. Conservative like that same sibling function:
        only when exactly one distinct ref overlaps this item's span."""
        refs = list(
            dict.fromkeys(
                el.image_ref for start, end, el in spans
                if el.image_ref and start < hi and end > lo
            )
        )
        return refs[0] if len(refs) == 1 else None

    def para(text: str, offset: int) -> Element | None:
        text = text.strip()
        if len(text) < 15:
            return None
        src = loc_at(offset)
        return Element(
            kind="paragraph", text=text, page=src.page,
            section_path=list(src.section_path),
        )

    lo = matches[i].start()
    hi = matches[j + 1].start() if j + 1 < len(matches) else len(joined)

    # Trailing image-only elements after the last numbered item — an unnumbered
    # puzzle set — must NOT be merged into that item (or collapsed into one
    # tail paragraph): keep each as its own element so _number_unlabelled_items
    # can tag it with its own image_ref.
    trailing: list[Element] = []
    if j + 1 >= len(matches):
        last_start = matches[j].start()
        seen: set[int] = set()
        for s_start, _s_end, el in spans:
            if s_start > last_start and el.image_ref and el.text and id(el) not in seen:
                seen.add(id(el))
                trailing.append(el)
        if trailing:
            hi = min(
                s_start for s_start, _e, el in spans if id(el) in seen
            )

    items: list[Element] = []
    if pre := para(joined[:lo], 0):
        items.append(pre)

    bounds = [matches[k].start() for k in range(i, j + 1)] + [hi]
    for idx, k in enumerate(range(i, j + 1)):
        body = joined[bounds[idx] : bounds[idx + 1]].strip()
        if not body:
            continue
        body = _reorder_scrambled_options(body)
        src = loc_at(matches[k].start())
        items.append(
            Element(
                kind="list_item", text=body, ordinal=numbers[k], page=src.page,
                section_path=list(src.section_path),
                image_ref=image_ref_in(bounds[idx], bounds[idx + 1]),
            )
        )

    # When `trailing` was split off, joined[hi:] IS that content — don't also
    # merge it into a tail paragraph (it would duplicate).
    if not trailing and (tail := para(joined[hi:], max(hi - 1, 0))):
        items.append(tail)
    items.extend(trailing)
    return items or None


def _main_run(numbers: list[int]) -> tuple[int, int] | None:
    """The longest stretch that counts up without restarting — the document's
    principal list. A short front-matter list ("1.–5." of exam instructions)
    ahead of the real "1.–30." is thereby left as ordinary prose, not competing
    for ordinal 3. Returns inclusive (start, end) indices, or ``None`` when no
    stretch is a convincing enumeration."""
    best: tuple[int, int] | None = None
    start = 0
    for k in range(1, len(numbers) + 1):
        broke = k == len(numbers) or not (0 <= numbers[k] - numbers[k - 1] <= 4)
        if broke:
            if best is None or (k - 1 - start) > (best[1] - best[0]):
                best = (start, k - 1)
            start = k
    if best is None:
        return None
    lo, hi = best
    run = numbers[lo : hi + 1]
    if len(run) < _MIN_ITEMS:
        return None
    steps = [b - a for a, b in zip(run, run[1:])]  # noqa: B905
    if sum(s == 1 for s in steps) < 0.6 * len(steps):
        return None
    return best
