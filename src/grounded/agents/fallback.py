"""Fallback agent — the one deliberate, narrow exception to "never use outside
knowledge."

Grounded's whole premise is refusing rather than guessing. But a document that
IS a quiz/exam (questions with lettered options, no answer key) is a case
where "I couldn't find this in your documents" is technically correct yet
unhelpful — the user often wants the question solved, not just confirmed
unanswered. This agent only fires when ALL of:

1. Verification already refused (the grounded path found nothing to cite).
2. The question targeted one or more specific numbered items
   (``decision.ordinals``) — never for a vague, undirected question.
3. A targeted item is a self-contained question with a definite answer — a
   multiple-choice question (>=3 lettered options) or an image-based question
   (a pattern puzzle, a read-the-diagram item) — never for, say, a missing
   contract clause or policy section, where fabricating content from outside
   knowledge would be actively wrong, not just unhelpful. Checked per item:
   "question 2 and 3" can solve #2 and skip #3 if only #2 qualifies.

The result is always labelled unmistakably as NOT sourced from the user's
documents — it must never be confused with a grounded, cited answer.

When a targeted item's pinned chunk carries an ``image_ref`` (a chart the
item depends on — see ``ingest/enumerate.py``'s ``_attach_page_figures`` and
``ingest/parsers.py``'s vision-read of chart pages), the model gets the real
source image at answer time, not just its static ingest-time description —
and is allowed to say it genuinely can't tell (e.g. crossing chart lines)
rather than being forced to name a letter it isn't sure of.
"""

from __future__ import annotations

import re
from collections import Counter

from grounded.agents.state import AgentState
from grounded.agents.verification import _REFUSAL
from grounded.config import settings
from grounded.index.store import Retrieved
from grounded.llm import chat, chat_with_image, vision_available
from grounded.util import image_store_path

# "a.Static function", "a.Θ(log n)", "a.2000" — a PDF's 2-column layout
# routinely flattens "a. X b. Y c. Z d. W" onto one continuous line with no
# newlines at all, so anchoring to line-start (an earlier version of this)
# missed everything past the first option. Anchor to a token boundary instead
# (not preceded by a letter/digit). No digit-exclusion after the delimiter —
# an earlier version borrowed one from the numbered-ITEM pattern (there, to
# stop "1.20" being misread as item "1"), but that concern doesn't transfer
# to LETTER options: "a" is never a digit, so "a.2000" can't be confused with
# a decimal the way "1.20" can. The exclusion's only real effect here was
# rejecting genuine options whose answer is a plain number (years, counts,
# percentages — common in quantitative questions) as "not multiple-choice".
_OPTION_RE = re.compile(r"(?<![A-Za-z0-9])[a-d][.)]\s*", re.I)
# Requiring 3 (not 2) keeps this conservative — this heuristic gates letting
# the model use outside knowledge, so a false "yes, this is an MCQ" is worse
# than an occasional missed one.
_MIN_OPTIONS = 3

FALLBACK_LABEL = "🧮 Not in your documents — AI-computed, not a cited answer:\n\n"

_SYSTEM = """The user's document contains this multiple-choice question (with \
no answer key or marked correct choice). Using your own knowledge — not the \
document, it has no more information than what's shown — answer the specific \
question asked. You may be given the shared material (a chart, a passage) and \
the other questions in the same set for context; still answer only the one \
asked. If an image is attached, it IS the actual source material (e.g. the \
chart itself, not a description of it) — read it directly.

Give your direct answer. Most of these questions have one clear, obvious \
correct option — trust your first read rather than manufacturing doubt \
between two options that both sound plausible.

Only reconsider your answer if a specific trap is actually, concretely \
present — not as a routine check on every question:
- A negative number, zero, or a boundary value that could flip a numeric \
  answer.
- Code whose output is asked for — check it actually COMPILES before \
  answering what it prints (e.g. in Java, catching a broader exception type \
  before a more specific one is a compile error, not a runtime branch).
- A chart/graph where lines or series cross or overlap — if you genuinely \
  cannot tell which one is which even looking directly at the image, that's \
  a real reason to say so, not a routine worry.

If neither concretely applies here, go with your direct answer — do not \
second-guess a clearly correct option just to seem thorough.

End with exactly one line, last, in ONE of these two forms:
Final answer: <letter>. <option text>
Final answer: cannot determine confidently — <one short, specific reason>

Only use the second form when reading the actual material (including the \
image, if attached) still leaves it genuinely ambiguous — not as a hedge on \
an otherwise answerable question."""

_PUZZLE_SYSTEM = """The user's document shows this question as an image, with \
no answer given — a number/letter pattern, a diagram to read off, a small \
visual problem. The image is attached: read it directly (any text version \
of it may be misread — trust the image).

Work out what's being asked and answer it. Give it a real try — test the \
obvious rules (row/column sums, differences, ratios, primes, position in a \
grid). Keep your working brief. Only if the obvious rules genuinely don't \
hold should you say you can't determine it.

End with exactly one line, last, in ONE of these two forms:
Final answer: <the answer — usually a number or short value>
Final answer: cannot determine confidently — <one short, specific reason>"""

# Guards against the model picking a letter anyway while still writing
# "cannot determine" earlier in its reasoning — only the FINAL line decides.
_CANT_DETERMINE_RE = re.compile(r"final answer:\s*cannot determine", re.I)


def _looks_like_mcq(text: str) -> bool:
    return len(_OPTION_RE.findall(text)) >= _MIN_OPTIONS


def _is_image_task(p: Retrieved) -> bool:
    """An image-backed item with no lettered options — a pattern puzzle, a
    read-the-diagram question. Gated the same as an MCQ: solving it from
    knowledge is helpful, not a fabrication risk, because it's a
    self-contained question with a definite answer (a missing contract
    clause, by contrast, has no image and no definite answer)."""
    return bool(p.chunk.image_ref) and not _looks_like_mcq(p.chunk.body)


def _label_for(p: Retrieved) -> str:
    # p.chunk.ordinal is always the real resolved number here (e.g. 30 for
    # "the last one" in a 30-item doc) — the router's -1 shorthand never
    # reaches this far, so there's no separate "last question" case to name.
    return f"question {p.chunk.ordinal}" if p.chunk.ordinal is not None else "this item"


_FINAL_RE = re.compile(r"final answer:\s*(.+?)\s*$", re.I | re.M)


def _answer_key(reply: str) -> str | None:
    """A normalised token for the reply's final answer, for voting: the MCQ
    letter, the numeric value, or ``"CANT"`` for a decline. ``None`` if no
    final-answer line at all."""
    m = _FINAL_RE.search(reply)
    if not m:
        return None
    ans = m.group(1).strip().lower()
    if ans.startswith("cannot determine"):
        return "CANT"
    letter = re.match(r"([a-d])[.)]", ans)
    if letter:
        return letter.group(1)
    num = re.match(r"[-+]?\d+(?:\.\d+)?", ans)
    return num.group(0) if num else ans[:40]


def _one_solve(p: Retrieved, messages: list[dict], model: str | None) -> str:
    if p.chunk.image_ref and vision_available():
        path = image_store_path(p.chunk.image_ref)
        if path.exists():
            try:
                return chat_with_image(
                    messages, path.read_bytes(), model=model
                ).strip()
            except Exception as exc:  # pragma: no cover - network/API best-effort
                print(f"[fallback] vision read failed for {_label_for(p)}: {exc}")
    return chat(messages, model=model).strip()


def _solve_one(p: Retrieved, messages: list[dict]) -> str:
    """Answer one target, using self-consistency: the model's sampling makes a
    single call flaky on a reasoning/puzzle question (it flipped 1.25 -> 1.20
    and 4 -> "can't tell" between runs of the same question), so solve it
    ``FALLBACK_VOTES`` times and go with the majority. A concrete answer needs
    >=2 votes to win; otherwise the answers disagreed and we say so rather
    than pick one at random.

    Each attempt uses the real source image when the pinned chunk has one (a
    chart the item depends on, a puzzle that IS an image) and text-only
    otherwise, and ``FALLBACK_MODEL`` when set — this is where model quality
    earns its cost."""
    model = settings.fallback_model or None
    votes = max(1, settings.fallback_votes)
    replies = [r for r in (_one_solve(p, messages, model) for _ in range(votes)) if r]
    if not replies:
        return ""
    if len(replies) == 1:
        return replies[0]

    keyed = [(r, _answer_key(r)) for r in replies]
    tally = Counter(k for _, k in keyed if k and k != "CANT")
    if tally:
        winner, n = tally.most_common(1)[0]
        if n >= 2:
            return next(r for r, k in keyed if k == winner)

    if any(k == "CANT" for _, k in keyed):
        return next(r for r, k in keyed if k == "CANT")

    seen = dict.fromkeys(
        m.group(1).strip() for r, k in keyed if k
        for m in [_FINAL_RE.search(r)] if m
    )
    tried = ", ".join(seen) or "different things"
    return (
        f"Solved this several times and got a different answer each time "
        f"({tried}) — not consistent enough to commit to one.\n"
        "Final answer: cannot determine confidently — inconsistent across attempts"
    )


def run_fallback(state: AgentState) -> AgentState:
    decision = state["decision"]
    # Same signal graph.py's routing already checked (belt-and-braces, in case
    # this is ever invoked directly): the refusal can arrive via the hard
    # "refused" path or the low-confidence hedge path, so check the text.
    if not (decision.ordinals and _REFUSAL in state.get("answer", "")):
        return {"notes": ["fallback: not applicable"]}

    # Retrieval marks exactly the passages it pinned for a targeted item —
    # each carries its own real chunk.ordinal, which is what a multi-item
    # question ("2 and 3") needs: solve each independently, on its own
    # retrieved text.
    targets = [p for p in state.get("passages", []) if p.components.get("pinned")]
    solved: list[str] = []
    n_answered = 0
    for p in targets:
        multi = len(targets) > 1
        image_task = _is_image_task(p)
        # Gate on the item's OWN text — a shared-stimulus set's context would
        # trivially pass _looks_like_mcq (it carries every sibling's options).
        if not (image_task or _looks_like_mcq(p.chunk.body)):
            # Silently dropping a targeted item read as "it just wasn't
            # asked about" — say so instead, so a genuinely-skipped item
            # ("2 and 5" where 5 turns out not to be an MCQ) is visible
            # rather than vanishing from the combined answer.
            if multi:
                solved.append(
                    f"**{_label_for(p).capitalize()}:** doesn't look like a "
                    "multiple-choice question — declining to guess without a source."
                )
            continue
        # For a shared-stimulus set, context_text carries the stimulus (a
        # chart) plus every sibling stem — the framing a single item can't be
        # answered without (e.g. what an unlabelled axis represents).
        passage = p.chunk.body if image_task else p.context_text
        ask = (
            "Question: what is the answer shown in the image?" if image_task
            else f"Question: what is the answer to question {p.chunk.ordinal}?"
            if p.chunk.ordinal is not None
            else "Question: what is the answer to this question?"
        )
        messages = [
            {"role": "system", "content": _PUZZLE_SYSTEM if image_task else _SYSTEM},
            {"role": "user", "content": f"{passage}\n\n{ask}"},
        ]
        reply = _solve_one(p, messages)
        if not reply:
            # An empty reply (e.g. the model spent its whole output budget
            # reasoning and never reached a conclusion) — say so plainly
            # rather than emit a blank answer.
            solved.append(
                f"**{_label_for(p).capitalize()}:** couldn't be worked out in one "
                "pass — no answer produced." if multi
                else "This couldn't be worked out in one pass — no answer produced."
            )
            continue
        # The model looked (at text, or — when p.chunk.image_ref points at a
        # chart — the actual image) and may still decline to commit. That's a
        # more honest, more useful answer than either a cited refusal or a
        # forced, possibly-fabricated letter — surface it either way, but
        # only count an actual answer toward n_answered.
        if not _CANT_DETERMINE_RE.search(reply):
            n_answered += 1
        solved.append(f"**{_label_for(p).capitalize()}:**\n{reply}" if multi else reply)

    if not solved:
        return {"notes": ["fallback: not applicable"]}

    return {
        "answer": FALLBACK_LABEL + "\n\n".join(solved),
        "refused": False,
        "fallback_used": True,
        # Clear whatever the failed grounded attempt left attached — an
        # ungrounded answer must never carry over stale citations/sentence
        # checks from before, or the UI would show "Sources (N)" and
        # "not a cited answer" side by side, contradicting each other.
        "citations": [],
        "checks": [],
        "notes": [
            f"fallback: answered {n_answered}/{len(targets)} targeted item(s) from "
            "outside knowledge — no answer key in the document"
        ],
    }
