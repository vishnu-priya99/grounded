"""Synthesis agent.

Combines retrieved passages and table results into one grounded answer with
inline ``[n]`` citation markers. Strictly context-only: if the context does not
contain the answer, it must say so.

Small models are unreliable at emitting citation markers, so after generation we
back-fill a marker onto any uncited sentence by lexical match against the
context blocks. The verification agent then checks whatever ends up cited.
"""

from __future__ import annotations

import re

from grounded.agents.state import AgentState, Citation, SynthesisResult
from grounded.llm import chat, chat_structured
from grounded.util import split_sentences

_SYSTEM = """You answer strictly from the numbered context below.
- Use ONLY facts present in the context. No outside knowledge.
- Answer the EXACT question asked. If it asks "which X", name the X; if it asks
  "how long / how many / how much", lead with the number. Do not answer a related
  but different question.
- A multiple-choice question's OPTIONS are not an answer. If the context shows a
  question with listed options (a/b/c/d, i/ii/iii, etc.) and nothing marks one
  as correct — no answer key, no checkmark, no "correct answer:" label — the
  answer is NOT in the context, even though the option text is right there.
  Restating or picking one of the options is a guess, not an answer: set found
  to false.
- The same applies to ANY unsolved question or puzzle in the context — a
  number pattern, a grid with a "?", a word problem with choices. If the
  context is the question itself and not its solution, set found to false
  (empty answer). Do NOT answer with "the value is not provided" or a
  description of the puzzle — that is not an answer either.
- Do NOT compute the answer by working backward from a percentage change, a
  growth rate, a ratio, or a sum of estimated parts. If the context gives you
  "X grew 22% to reach $Y" you may state $Y; but if it only gives "X grew 22%"
  or "X grew by $Z" and NOT the value asked for, that value is not in the
  context — set found to false. Reconstructing a figure from a change is a
  guess, not a grounded fact, no matter how the arithmetic looks.
  (Simple addition of two figures both stated in the context — e.g. "60 + 20
  points = 80" — is fine.)
- End every sentence that uses a source with its marker, e.g. [1] or [2][3].
- Set found to false, with an empty answer, when the context does not actually
  contain the answer — never guess or fall back on outside knowledge here.
- Be concise. Use the user's own terms.

Example 1:
Context:
[1] (policy.md) Refunds are issued within 14 days of purchase.
[2] (policy.md) Enterprise annual plans have a 30-day window.
Question: what's the refund window for enterprise annual plans?
-> found: true, answer: "Enterprise annual plans have a 30-day refund window. [2]"

Example 2:
Context:
[1] (exam.pdf) 3. Which of the following is not a member of a class in C++?
a. Static function  b. Const function  c. Friend function  d. Virtual function
Question: what is the answer to question 3?
-> found: false, answer: (empty)"""

_BROAD_ADDENDUM = """

This is a BROAD question ("what topics does this cover", "list every X") — the
context below is every chunk that could be gathered for this document, up to a
size budget; it may not be literally 100% of the document. Base counts/lists
only on what you actually see below, never estimate or round up. If your
context looks partial (a document this size plausibly has more), say your
answer may not cover every item — do not present a partial sample as
complete."""

_REFUSAL = "I couldn't find this in your documents."
_CITE_NUM = re.compile(r"\[(\d+)\]")
# A question about a specific chart/graph, and the caption line a chart chunk
# leads with (from the vision-transcription prompt / the pptx chart extractor).
_CHART_QUESTION_RE = re.compile(r"\b(chart|graph|plot|bar chart|line graph|axis)\b", re.I)
_CHART_CAPTION_RE = re.compile(
    r"^\s*#*\s*(this chart shows|chart description|chart data|chart analysis|"
    r"[a-z ]+ chart\b)",
    re.I,
)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an of to in on for and or is are was were be by with as at from this that "
    "it its into per about over under it's you your our their has have had will".split()
)


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def run_synthesis(state: AgentState) -> AgentState:
    decision = state["decision"]
    question = decision.rewritten_question or state["question"]

    if decision.intent == "chitchat":
        reply = chat(
            [
                {"role": "system", "content": "You are a concise assistant for a "
                 "document-chat app. Answer briefly; no citations needed."},
                {"role": "user", "content": state["question"]},
            ]
        )
        return {"draft_answer": reply, "citations": [], "notes": ["synthesis: chitchat"]}

    citations: list[Citation] = []
    blocks: list[str] = []
    block_keywords: dict[int, set[str]] = {}
    n = 0

    # A successful table result is a *computed* answer — SQL run against the
    # actual rows — so it leads, ahead of the retrieved prose. Retrieval drops
    # spreadsheet chunks (the table agent owns that data), so on a multi-document
    # index the passages below are often only loosely-related text from *other*
    # files; emitting the SQL result last and unlabelled let the small synthesis
    # model treat it as noise and refuse a question it had already answered. The
    # passages still follow it and still win when the question is about them.
    table_results = [t for t in state.get("tables", []) if not (t.error and not t.rows)]
    for t in table_results:
        n += 1
        table_md = t.as_markdown(limit=25)
        citations.append(
            Citation(
                n=n, source_file="spreadsheet query",
                snippet=f"SQL: {t.sql}\n{t.as_markdown(limit=8)}"[:400],
                source_text=f"SQL: {t.sql}\n{table_md}",
            )
        )
        blocks.append(
            f"[{n}] Answer computed directly from the spreadsheet by running "
            f"this SQL query:\n{t.sql}\nResult:\n{table_md}\n"
            f"This is the authoritative answer for a question about this "
            f"spreadsheet's data — use it."
        )
        block_keywords[n] = _keywords(table_md + " " + question)

    # Feed the strongest passages to the model; too many dilutes and invites
    # mis-citation. All retrieved passages still reach the verifier via state.
    # A "broad" question is the deliberate exception — retrieval already
    # gathered a whole-document set sized to a token budget, not the usual
    # handful, so every one of those chunks is meant to be seen here.
    candidates = state.get("passages", [])
    if table_results and not decision.broad:
        # A SQL query has computed an answer. Retrieval deliberately drops every
        # spreadsheet chunk (see retrieval.py — the table agent owns structured
        # data), so the passages here are always leftover text from *other*
        # documents.
        if decision.intent == "table_qa":
            # The router judged this a pure structured-data question. Retrieval's
            # passages are then only a by-product of its "are tables relevant?"
            # check, never meant to answer — and unrelated prose (an exam item
            # that happens to share "new"/"three", a 10-K cover page that shares
            # "months") reliably outvotes the computed result and forces a false
            # "not found". Drop them; the SQL block stands on its own. Keyword
            # overlap can't filter this safely — the leaking words are common.
            candidates = []
        else:
            # A doc-QA question that retrieval *promoted* to tables, or a "mixed"
            # question: a passage may genuinely carry part of the answer, so keep
            # the ones that share real wording with the question (>= 2 content
            # words — one common word leaks too much).
            q_kw = _keywords(question)
            if q_kw:
                candidates = [
                    r for r in candidates
                    if len(q_kw & _keywords(r.chunk.body)) >= 2
                ]

    # "the chart" / "the graph" in a question means ONE specific chart, and
    # retrieval's top hit fixes which document it's in. On a multi-document
    # corpus the lower ranks pull in *other* documents' charts (a 10-K ARPP
    # question dragging in an Acme revenue chart), and the small model then
    # mixes their numbers. Drop the foreign chart/figure blocks — never the
    # same chart the question asks about. Prose from other documents stays.
    if not decision.broad and candidates and _CHART_QUESTION_RE.search(question):
        top_src = candidates[0].chunk.source_file
        candidates = [
            r for r in candidates
            if r.chunk.source_file == top_src
            or not (
                r.chunk.element_kind in ("figure", "table")
                or _CHART_CAPTION_RE.match(r.chunk.body.lstrip())
            )
        ]
    # Everything still reaches the verifier via state regardless of the above.
    passage_limit = len(candidates) if decision.broad else 6
    for r in candidates[:passage_limit]:
        n += 1
        c = r.chunk
        citations.append(
            Citation(
                n=n, source_file=c.source_file, page=c.page,
                section_path=c.section_path, snippet=c.body[:280].strip(),
                source_text=r.context_text,
            )
        )
        blocks.append(f"[{n}] ({c.citation_label()}) {r.context_text}")
        block_keywords[n] = _keywords(r.context_text)

    if not blocks:
        return {
            "draft_answer": _REFUSAL, "citations": [],
            "notes": ["synthesis: no context"],
        }

    system = _SYSTEM + _BROAD_ADDENDUM if decision.broad else _SYSTEM
    result = chat_structured(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "Context:\n" + "\n\n".join(blocks)
             + f"\n\nQuestion: {question}"},
        ],
        SynthesisResult,
        fallback=SynthesisResult(found=False, answer=""),
    )

    if not result.found or not result.answer.strip():
        return {
            "draft_answer": _REFUSAL, "citations": [],
            "notes": ["synthesis: model reports not found"],
        }

    draft = _backfill_citations(result.answer.strip(), block_keywords)
    return {
        "draft_answer": draft,
        "citations": citations,
        "notes": [f"synthesis: {len(blocks)} context blocks"],
    }


def _backfill_citations(draft: str, block_keywords: dict[int, set[str]]) -> str:
    """Attach the best lexically-matching context block to each sentence.

    The model's own markers are kept; this only *adds* a marker when a sentence
    has none, or when a clearly-better-matching block was not cited. The verifier
    decides which of the cited blocks actually holds up.
    """
    out: list[str] = []
    for sent in split_sentences(draft):
        s = sent.strip()
        if not s or len(s) < 15:
            out.append(s)
            continue
        existing = set(_CITE_NUM.findall(s))
        kw = _keywords(s)
        if not kw:
            out.append(s)
            continue
        ranked = sorted(
            ((len(kw & bkw) / len(kw), bn) for bn, bkw in block_keywords.items() if bkw),
            reverse=True,
        )
        if not ranked:
            out.append(s)
            continue
        top_overlap, top_n = ranked[0]
        need_marker = not existing
        threshold = 0.30 if need_marker else 0.45
        if top_overlap >= threshold and str(top_n) not in existing:
            s = f"{s} [{top_n}]"
        out.append(s)
    return " ".join(out)
