"""Citation / Verification agent.

Checks each sentence of the draft against the sources it cites, computes a
confidence score, prunes unused citations, and decides whether to ship the
answer, hedge it, or fall back to "I don't know".
"""

from __future__ import annotations

import re

import numpy as np

from grounded.agents.state import AgentState, Citation, SentenceCheck
from grounded.config import settings
from grounded.index.embeddings import embed_texts
from grounded.util import split_sentences

_REFUSAL = "I couldn't find this in your documents."
_CITE_RE = re.compile(r"\[(\d+)\]")
_MD = re.compile(r"[*_`#]+")
_WORD = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOP = frozenset(
    "the a an of to in on for and or is are was were be been being by with as at from "
    "this that it its into per about over under you your our their has have had will "
    "may can could would should".split()
)
# RRF score when one chunk tops both the dense and the sparse list.
_STRONG_RRF = 2.0 / (settings.rrf_k + 1)
_SUPPORT_THRESHOLD = 0.52


def _sentences(text: str) -> list[str]:
    return split_sentences(text)


def _kw(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


_NUM_RE = re.compile(r"([\d,]+\.?\d*)\s*(million|billion|thousand|[mkb])?\b", re.I)
_MULT = {"million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9, "thousand": 1e3, "k": 1e3}


def _numbers(text: str) -> set[str]:
    out: set[str] = set()
    for num_s, suf in _NUM_RE.findall(text):
        try:
            val = float(num_s.replace(",", ""))
        except ValueError:
            continue
        out.add(f"{val:g}")
        out.add(f"{val * _MULT.get(suf.lower(), 1.0):.0f}")
    return out


def _full_text_by_n(state: AgentState) -> dict[int, str]:
    return {
        c.n: _MD.sub("", c.source_text or c.snippet)
        for c in state.get("citations", [])
    }


def run_verification(state: AgentState) -> AgentState:
    draft = state.get("draft_answer", "").strip()
    citations = state.get("citations", [])
    passage_scores = [r.score for r in state.get("passages", [])]

    if not draft or draft.rstrip(".") == _REFUSAL.rstrip("."):
        return {
            "answer": _REFUSAL, "citations": [], "checks": [],
            "confidence": 0.0, "refused": True,
            "notes": ["verification: grounded refusal"],
        }
    if state["decision"].intent == "chitchat":
        return {
            "answer": draft, "citations": [], "checks": [],
            "confidence": 1.0, "refused": False,
            "notes": ["verification: skipped (chitchat)"],
        }

    sents = _sentences(draft)
    cited_by_sentence = [[int(m) for m in _CITE_RE.findall(s)] for s in sents]
    by_n = {c.n: c for c in citations}
    full_text = _full_text_by_n(state)

    trusted = {c.n for c in citations if c.source_file == "spreadsheet query"}
    checks, scores = _check_support(sents, cited_by_sentence, full_text, trusted)

    grounded = [c for c in checks if _needs_support(c.sentence)]
    supported = [c for c in grounded if c.supported]
    support_ratio = len(supported) / len(grounded) if grounded else 1.0
    cite_coverage = (
        sum(1 for c in grounded if c.cited) / len(grounded) if grounded else 0.0
    )
    retrieval_signal = (
        min(1.0, max(passage_scores) / _STRONG_RRF) if passage_scores else 0.3
    )

    confidence = round(
        min(0.95, 0.55 * support_ratio + 0.20 * cite_coverage + 0.25 * retrieval_signal), 2
    )

    used_ns = {n for cs in cited_by_sentence for n in cs}
    final_citations = _attach_support(
        [by_n[n] for n in sorted(used_ns) if n in by_n], checks, scores
    )
    draft, final_citations = _prune_weak(draft, final_citations)

    answer, refused = draft, False
    # Refuse only when the model neither grounded nor cited anything, or when it
    # cited but nothing checks out AND retrieval was also weak.
    nothing_cited = grounded and cite_coverage == 0.0
    unsupported_and_weak = support_ratio == 0.0 and retrieval_signal < 0.5
    if not final_citations and not state.get("tables"):
        answer, refused, final_citations = _REFUSAL, True, []
    elif nothing_cited or unsupported_and_weak:
        answer, refused, final_citations = _REFUSAL, True, []
    elif confidence < settings.confidence_threshold:
        answer = (
            "_Low confidence — the documents only partially support this._\n\n" + draft
        )

    return {
        "answer": answer,
        "citations": final_citations,
        "checks": checks,
        "confidence": confidence,
        "refused": refused,
        "notes": [
            f"verification: {len(supported)}/{len(grounded)} supported, "
            f"cite_cov={cite_coverage:.1f}, retr={retrieval_signal:.2f}, "
            f"confidence={confidence}"
        ],
    }


def _prune_weak(draft: str, cits: list[Citation]) -> tuple[str, list[Citation]]:
    """Drop citations that barely match, and strip their inline markers — unless
    that would leave a sentence with none. When one citation clearly holds up,
    the bar for keeping its partners rises."""
    if not cits:
        return draft, cits
    best = max((c.support or 0.0) for c in cits)
    floor = 0.50 if best >= 0.65 else 0.38
    strong = [c for c in cits if c.support is None or c.support >= floor]
    if not strong:
        strong = [max(cits, key=lambda c: c.support or 0.0)]
    keep = {c.n for c in strong}
    dropped = {c.n for c in cits if c.n not in keep}
    if dropped:
        for sent in split_sentences(draft):
            marks = {int(m) for m in _CITE_RE.findall(sent)}
            if marks and marks.issubset(dropped):
                keep |= {min(marks)}  # keep one so the sentence stays cited
        for n in sorted(dropped - keep):
            draft = draft.replace(f" [{n}]", "").replace(f"[{n}]", "")
    strong = [c for c in cits if c.n in keep]
    return draft.strip(), sorted(strong, key=lambda c: -(c.support or 0.0))


def _needs_support(sentence: str) -> bool:
    s = sentence.strip().lower()
    if len(s) < 15:
        return False
    return not s.startswith(("in summary", "overall", "note that", "this means", "in short"))


def _check_support(
    sents: list[str],
    cited: list[list[int]],
    full_text: dict[int, str],
    trusted: set[int],
) -> tuple[list[SentenceCheck], dict[tuple[int, int], float]]:
    """Score each sentence against its cited sources with a blend of semantic
    similarity (nomic-embed cosine) and content-word recall.

    Deterministic and fast — a better fit for a CPU-only box than asking a 7B
    model to adjudicate entailment, which it does unreliably in both directions.
    """
    clean_sents = [_MD.sub("", s) for s in sents]
    src_ids = sorted(full_text)
    to_embed = clean_sents + [full_text[i] for i in src_ids]
    vecs = [np.asarray(v, dtype=np.float32) for v in embed_texts(to_embed)]
    sent_vecs = vecs[: len(clean_sents)]
    src_vecs = {i: vecs[len(clean_sents) + k] for k, i in enumerate(src_ids)}
    src_kw = {i: _kw(full_text[i]) for i in src_ids}

    checks: list[SentenceCheck] = []
    pair_scores: dict[tuple[int, int], float] = {}
    for idx, (raw, clean, cs) in enumerate(zip(sents, clean_sents, cited, strict=True)):
        s_kw = _kw(clean)
        s_nums = _numbers(clean)
        best = 0.0
        for n in cs:
            if n in trusted:
                # a SQL query that executed is ground truth; boost further if the
                # sentence's numbers appear in the result
                hit = bool(s_nums & _numbers(full_text.get(n, "")))
                score = 0.95 if hit else 0.80
            elif n in src_vecs:
                cos = _cosine(sent_vecs[idx], src_vecs[n])
                recall = len(s_kw & src_kw[n]) / len(s_kw) if s_kw else 0.0
                score = 0.55 * cos + 0.45 * recall
            else:
                continue
            pair_scores[(idx, n)] = round(score, 3)
            best = max(best, score)
        checks.append(
            SentenceCheck(sentence=raw, supported=best >= _SUPPORT_THRESHOLD, cited=cs)
        )
    return checks, pair_scores


def _attach_support(
    cits: list[Citation],
    checks: list[SentenceCheck],
    scores: dict[tuple[int, int], float],
) -> list[Citation]:
    for c in cits:
        vals = [v for (_, n), v in scores.items() if n == c.n]
        c.support = round(max(vals), 2) if vals else None
    return cits
