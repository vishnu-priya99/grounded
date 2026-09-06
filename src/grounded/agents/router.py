"""Router / Planner agent.

Classifies intent, rewrites the question to stand alone using conversation
context, and decomposes it into focused sub-queries. One structured LLM call.
"""

from __future__ import annotations

import re

from grounded.agents.state import AgentState, RouterDecision
from grounded.config import settings
from grounded.ingest.pipeline import get_stores
from grounded.llm import chat_structured

# "what topics does this cover", "list every question", "how many are about X",
# "which questions are about C++", "summarize" — needs the WHOLE document, not
# the usual handful of similarity-matched passages.
_BROAD_RE = re.compile(
    r"\b(what (?:topics|subjects)|which topics|list (?:all|every)|every "
    r"(?:question|item|section)|summar(?:y|ize|ise)|overview|how many\b.{0,25}"
    r"\b(?:are|is)\b|which (?:questions|items|sections|parts)\b.{0,25}"
    r"\b(?:are|about)\b)",
    re.I,
)


def _looks_broad(text: str) -> bool:
    """Deterministic fallback/override for the model's own ``broad`` guess —
    this pattern is worth catching reliably even when the model misses it."""
    return bool(_BROAD_RE.search(text))

_SYSTEM = """You are the planner for a document-chat system. Given the user's \
question and the conversation context, decide how to answer it.

intent:
- "table_qa"  : needs aggregation/filtering/lookup over spreadsheet data (sums, \
counts, averages, "by region", "how many", "total").
- "doc_qa"    : answerable from prose in documents (policies, reports, notes).
- "code_qa"   : about the contents or behaviour of a source-code file.
- "mixed"     : needs both prose and spreadsheet data.
- "chitchat"  : greeting or meta-question needing no documents (e.g. "hello", \
"what can you do", "thanks"). Documents ARE indexed and searchable right now \
(see below) — if any reasonable reading of the question could be about their \
content, even phrased ambiguously ("question 7", "the last question", "the \
answer"), classify it as doc_qa/table_qa/code_qa/mixed, never chitchat. Use \
chitchat only when no plausible reading connects to the available documents.

Rewrite the question so it stands alone (resolve "it", "that", "the second one" \
from context). Produce 1-3 sub_queries for retrieval; a simple question needs \
only one. Set needs_tables true for table_qa or mixed.

Don't just repeat the question's exact wording in every sub_query — search \
also fails when the user's words differ from the document's. Include at least \
one sub_query phrased the way the document likely states it, especially for \
questions about a duration, amount, date, or named value (e.g. "how long did \
it last" -> also try "duration"; "how much does it cost" -> also try "price" \
/ "fee"). Only add variants that still mean the same thing.

If the question points at one or more enumerated items by position — question \
numbers, "Nth", an "and"/comma-separated list — set ordinals to a list of \
those numbers: "the third question" -> [3], "question 2 and 3" -> [2, 3], \
"10th and 11th question" -> [10, 11], "questions 5, 7, and 9" -> [5, 7, 9]. \
Use -1 for "the last one". A named set of items counts as positional too — \
set ordinals to every number in the set (see any hint below). Empty list if \
the question isn't positional.

Set broad true if answering needs to survey the WHOLE document rather than a \
couple of passages — "what topics does this cover", "list every question", \
"summarize this", "how many questions are about X". Otherwise false."""


_PATTERN_SET_RE = re.compile(r"\b(pattern|puzzle|patterns|puzzles)\b", re.I)


def _resolve_subset_ordinals(
    ordinals: list[int], question: str, sub_q_ords: list[int]
) -> list[int]:
    """"The 4th pattern question" collides with real question 4 — the model
    often returns [4] instead of the 4th of the puzzle set. When the question
    is clearly about the puzzle set and the ordinals look like *positions
    within it* (1..N) rather than the set's own item numbers, remap them.
    Leaves everything else untouched."""
    if not (ordinals and sub_q_ords and _PATTERN_SET_RE.search(question)):
        return ordinals
    if any(o in sub_q_ords for o in ordinals):  # already correct
        return ordinals
    n = len(sub_q_ords)
    if all(1 <= o <= n for o in ordinals):
        return [sub_q_ords[o - 1] for o in ordinals]
    return ordinals


def run_router(state: AgentState) -> AgentState:
    ctx = state.get("history") or "(no prior conversation)"
    files = state.get("source_files")
    store = get_stores().vector
    docs = store.documents()
    if files:
        scope = f"\nThe user restricted the search to: {', '.join(files)}"
    elif docs:
        scope = f"\n{len(docs)} document(s) indexed and searchable: {', '.join(docs)}"
    else:
        scope = "\nNo documents are indexed yet."

    sub_q_ords = sorted(
        {
            c.ordinal for c in store.chunks
            if c.ordinal is not None and c.body.lstrip().lower().startswith("sub-question")
        }
    )
    if sub_q_ords:
        instr = next(
            (c.body.split("instruction:", 1)[1].split("):", 1)[0].strip()
             for c in store.chunks
             if c.ordinal in sub_q_ords and "instruction:" in c.body),
            "",
        )
        scope += (
            f"\nHint: items {sub_q_ords} are an image sub-question set"
            + (f' under the instruction "{instr}"' if instr else "")
            + f" — 'the puzzles' / 'the pattern questions' / 'problem 2' means "
            f"all of {sub_q_ords}; 'the Nth pattern question' means the Nth of "
            f"that list (the 2nd pattern question is {sub_q_ords[1]}, NOT item 2)."
        )

    q = state["question"]
    decision = chat_structured(
        [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": f"Conversation context:\n{ctx}{scope}\n\nQuestion: {q}",
            },
        ],
        RouterDecision,
        model=settings.router_model,
        fallback=RouterDecision(
            intent="doc_qa", rewritten_question=q, sub_queries=[q],
            needs_tables=False, reasoning="fallback: router parse failed",
        ),
    )
    if not decision.sub_queries:
        decision.sub_queries = [decision.rewritten_question or state["question"]]

    # A reference to item #N is a document reference by definition — it
    # cannot be chitchat, whatever the model guessed. This is exactly the
    # pattern ("question 7", "the last question") that was getting misrouted
    # to chitchat and skipping retrieval entirely.
    if decision.ordinals and decision.intent == "chitchat" and docs:
        decision.intent = "doc_qa"

    decision.ordinals = _resolve_subset_ordinals(
        decision.ordinals, decision.rewritten_question or q, sub_q_ords
    )

    if not decision.broad and _looks_broad(decision.rewritten_question or q):
        decision.broad = True

    notes = [f"router: intent={decision.intent}, {len(decision.sub_queries)} sub-queries"]
    if decision.ordinals:
        notes.append(f"router: targets item(s) {decision.ordinals}")
    if decision.broad:
        notes.append("router: broad question — will survey the whole document")

    return {
        "decision": decision,
        "intent": decision.intent,
        "notes": notes,
    }
