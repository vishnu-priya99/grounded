"""Retrieval agent.

Runs hybrid search once per sub-query, merges by best RRF score, and keeps the
top passages. Table cards are dropped here — the Table agent handles structured
data — but their presence tells us tables are relevant.
"""

from __future__ import annotations

from collections import Counter

from grounded.agents.state import AgentState
from grounded.config import settings
from grounded.index.store import Retrieved
from grounded.ingest.models import SourceKind
from grounded.ingest.pipeline import get_stores
from grounded.util import estimate_tokens


def run_retrieval(state: AgentState) -> AgentState:
    decision = state["decision"]
    if decision.intent == "chitchat":
        return {"passages": [], "notes": ["retrieval: skipped (chitchat)"]}

    store = get_stores().vector
    files = state.get("source_files")
    best: dict[str, Retrieved] = {}
    saw_table_card = False

    for sub in decision.sub_queries:
        for hit in store.search(sub, top_k=settings.top_k, source_files=files):
            if hit.chunk.source_kind == SourceKind.SPREADSHEET:
                saw_table_card = True
                continue
            cur = best.get(hit.chunk.chunk_id)
            if cur is None or hit.score > cur.score:
                best[hit.chunk.chunk_id] = hit

    passages = sorted(best.values(), key=lambda r: r.score, reverse=True)[: settings.top_k]
    notes = [f"retrieval: {len(passages)} passages from {len(decision.sub_queries)} sub-queries"]

    # A positional question ("the third question", "question 2 and 3") — force
    # every pinned item to the *front*, whether or not normal search also
    # happened to find it, then pad with only a couple more passages, not the
    # full top_k. A chunk normal search already ranked outside the top few
    # must still be reordered to lead — leaving it at its original (low) rank
    # was the bug in an earlier version of this: the "pinned" note printed,
    # but the chunk still got cut by the smaller slice below because it was
    # never actually moved up. Padding beyond that bought nothing but cost
    # and noise — a mostly-irrelevant filler chunk riding along in every
    # prompt for no accuracy gain, now that the answer is guaranteed present.
    if decision.ordinals:
        seen_ids: set[str] = set()
        pinned = []
        for n in decision.ordinals:
            for c in store.chunks_by_ordinal(n, source_files=files):
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    pinned.append(c)
        if pinned:
            by_id = {p.chunk.chunk_id: p for p in passages}
            lead = max((p.score for p in passages), default=1.0) + 1.0
            forced = []
            for c in pinned:
                r = by_id.get(c.chunk_id) or Retrieved(chunk=c, score=lead)
                # Marked explicitly, even when reusing a passage normal search
                # already found — the fallback agent needs to tell "this is
                # one of the targeted items" apart from generic padding, and
                # a reused Retrieved otherwise carries no such signal.
                r.components = {**r.components, "pinned": 1.0}
                forced.append(r)
            pinned_ids = {c.chunk_id for c in pinned}
            # Padding must have actually ranked well in *some* signal — not
            # just be whatever was left after removing the pinned chunk(s).
            # Without this floor, "top 2 of what's left" can and did pull in
            # a completely unrelated chunk (e.g. a different question that
            # merely shares a stray keyword) just to fill the slot.
            _PAD_RANK_FLOOR = 5
            rest = [
                p for p in passages
                if p.chunk.chunk_id not in pinned_ids
                and (
                    (p.dense_rank is not None and p.dense_rank <= _PAD_RANK_FLOOR)
                    or (p.sparse_rank is not None and p.sparse_rank <= _PAD_RANK_FLOOR)
                )
            ]
            passages = (forced + rest)[: len(forced) + 2]
            notes.append(
                f"retrieval: pinned {len(pinned)} chunk(s) for item(s) {decision.ordinals}"
                f", {len(rest)} supporting passage(s) cleared the relevance floor"
            )

    # A broad question ("what topics does this cover", "list every question")
    # can't be answered from a handful of similarity-matched passages — it
    # needs the whole document. Similarity search alone would just sample an
    # arbitrary subset and let the model answer as if that subset were
    # everything, with no signal that 24 of 30 items were never shown to it.
    if decision.broad:
        targets = files
        if not targets:
            # No explicit scope — infer the document in question from what
            # normal search above already leaned towards, rather than
            # sweeping every document in a multi-document index.
            counts = Counter(p.chunk.source_file for p in passages)
            if counts:
                targets = [counts.most_common(1)[0][0]]
        if targets:
            pool = store.chunks_for_documents(targets)
            kept, tok = [], 0
            for c in pool:
                t = estimate_tokens(c.body)
                if kept and tok + t > settings.broad_max_tokens:
                    break
                kept.append(c)
                tok += t
            passages = [Retrieved(chunk=c, score=1.0) for c in kept]
            truncated = (
                " (truncated — document is larger than the budget)"
                if len(kept) < len(pool) else ""
            )
            notes.append(
                f"retrieval: broad — surveying {len(kept)}/{len(pool)} chunks "
                f"from {', '.join(targets)}{truncated}"
            )

    out: AgentState = {"passages": passages, "notes": notes}
    if saw_table_card and not decision.needs_tables:
        # retrieval surfaced a spreadsheet — let the table agent take a look
        state["decision"].needs_tables = True
        out["notes"].append("retrieval: promoted to table_qa (table card matched)")
    return out
