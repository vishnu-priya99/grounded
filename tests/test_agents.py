"""Unit tests for the deterministic parts of the agent layer (no LLM calls)."""

from __future__ import annotations

from grounded.agents.fallback import _looks_like_mcq, run_fallback
from grounded.agents.graph import _after_verification
from grounded.agents.state import Citation, RouterDecision
from grounded.agents.synthesis import _backfill_citations, _keywords
from grounded.agents.verification import _numbers, _prune_weak, _sentences
from grounded.index.store import Retrieved
from grounded.ingest.models import Chunk, SourceKind


def test_numbers_normalises_scale_suffixes() -> None:
    assert "2460000" in _numbers("total was USD 2.46M")
    assert _numbers("2.46M") & _numbers("2460000")
    assert _numbers("30 days") & _numbers("a 30-day window")


def test_sentence_split_keeps_citation_marker_attached() -> None:
    sents = _sentences("The window is 30 days. [1] It applies to Enterprise. [2]")
    assert sents[0].endswith("[1]")
    assert sents[1].endswith("[2]")


def test_backfill_adds_marker_to_uncited_sentence() -> None:
    kw = {1: _keywords("Enterprise annual contracts have a 30 day refund window")}
    out = _backfill_citations("Enterprise annual contracts get a 30 day refund window.", kw)
    assert "[1]" in out


def test_backfill_leaves_cited_sentence_alone() -> None:
    kw = {1: _keywords("refund window enterprise"), 2: _keywords("something unrelated")}
    out = _backfill_citations("The refund window is 30 days. [1]", kw)
    assert out.count("[1]") == 1
    assert "[2]" not in out


def test_synthesis_leads_with_the_table_result_and_drops_unrelated_passages(
    monkeypatch,
) -> None:
    """When SQL computed the answer, its block comes first and off-topic
    passages from other documents are kept out of the prompt — they used to
    outvote the result and trip a false 'not found'."""
    from grounded.agents.state import SynthesisResult, TableAnswer
    from grounded.agents.synthesis import run_synthesis

    seen: dict[str, str] = {}

    def capture(messages, model, fallback=None):
        seen["user"] = messages[-1]["content"]
        return SynthesisResult(found=True, answer="The total headcount is 119. [1]")

    monkeypatch.setattr("grounded.agents.synthesis.chat_structured", capture)

    noise = Retrieved(
        chunk=Chunk(
            chunk_id="m1", doc_id="d2", source_file="Meta-10K.pdf",
            source_kind=SourceKind.DOCUMENT,
            text="Meta had a global workforce of 74,067 employees at year end.",
            body="Meta had a global workforce of 74,067 employees at year end.",
        ),
        score=0.4,
    )
    table = TableAnswer(
        question="q", sql="SELECT SUM(headcount) AS total FROM hc",
        columns=["total"], rows=[[119]],
    )
    state = {
        "question": "what is the total headcount across all departments?",
        "decision": _decision(intent="table_qa", needs_tables=True),
        "passages": [noise],
        "tables": [table],
    }
    out = run_synthesis(state)

    assert out["draft_answer"].startswith("The total headcount is 119.")
    body = seen["user"]
    # SQL result is block [1], ahead of any passage.
    assert "[1] Answer computed directly from the spreadsheet" in body
    # The unrelated Meta passage was filtered out.
    assert "74,067" not in body


def test_synthesis_keeps_related_passages_when_not_a_pure_table_question(
    monkeypatch,
) -> None:
    """A doc-QA question that merely *also* touched a table (promoted, or
    'mixed') must still see its prose passages — only a router-declared
    ``table_qa`` drops them wholesale."""
    from grounded.agents.state import SynthesisResult, TableAnswer
    from grounded.agents.synthesis import run_synthesis

    seen: dict[str, str] = {}

    def capture(messages, model, fallback=None):
        seen["user"] = messages[-1]["content"]
        return SynthesisResult(found=True, answer="Meta had 74,067 employees. [2]")

    monkeypatch.setattr("grounded.agents.synthesis.chat_structured", capture)

    related = Retrieved(
        chunk=Chunk(
            chunk_id="m1", doc_id="d2", source_file="Meta-10K.pdf",
            source_kind=SourceKind.DOCUMENT,
            text="Meta had a global workforce of 74,067 employees at year end.",
            body="Meta had a global workforce of 74,067 employees at year end.",
        ),
        score=0.4,
    )
    table = TableAnswer(
        question="q", sql="SELECT SUM(headcount) AS total FROM hc",
        columns=["total"], rows=[[119]],
    )
    q = "how many employees does Meta have in its global workforce?"
    state = {
        "question": q,
        "decision": _decision(
            intent="doc_qa", needs_tables=True, rewritten_question=q
        ),
        "passages": [related],
        "tables": [table],
    }
    run_synthesis(state)

    # doc_qa -> keyword filter, not drop-all: the on-topic Meta passage survives.
    assert "74,067" in seen["user"]


def test_synthesis_chart_question_drops_other_documents_charts(monkeypatch) -> None:
    """'the chart' means one chart; the top hit fixes which document. A chart
    from a different file is never that chart — drop it (prose still stays)."""
    from grounded.agents.state import SynthesisResult
    from grounded.agents.synthesis import run_synthesis

    seen: dict[str, str] = {}

    def capture(messages, model, fallback=None):
        seen["user"] = messages[-1]["content"]
        return SynthesisResult(found=True, answer="Q1 2023 was lowest at $9.47. [1]")

    monkeypatch.setattr("grounded.agents.synthesis.chat_structured", capture)

    def _r(cid, src, body, kind="paragraph", score=0.3):
        return Retrieved(
            chunk=Chunk(
                chunk_id=cid, doc_id=src, source_file=src,
                source_kind=SourceKind.DOCUMENT, text=body, body=body,
                element_kind=kind,
            ),
            score=score,
        )

    passages = [
        _r("a", "Meta-10K.pdf",
           "# Chart Description and Data\nARPP by quarter: Mar 31 2023 $9.47 ...",
           kind="figure", score=0.9),
        _r("b", "Meta-10K.pdf", "Average Revenue Per Person (ARPP) is FoA revenue "
           "divided by average DAP for the quarter.", score=0.8),
        _r("c", "board_deck.pptx",
           "line markers chart - Quarterly revenue trend: Total revenue (USD M) "
           "Q3 2025 1.89 ... Q3 2026 2.66", kind="table", score=0.5),
    ]
    q = "which quarter had the lowest ARPP on the chart?"
    run_synthesis({
        "question": q,
        "decision": _decision(intent="doc_qa", rewritten_question=q),
        "passages": passages, "tables": [],
    })

    body = seen["user"]
    assert "ARPP by quarter" in body            # the Meta chart stays
    assert "Total revenue (USD M)" not in body  # the foreign chart is dropped


def test_prune_weak_drops_low_support_citation_and_marker() -> None:
    cits = [
        Citation(n=1, source_file="a.md", support=0.85),
        Citation(n=2, source_file="b.md", support=0.10),
    ]
    draft, kept = _prune_weak("Claim one. [1][2]", cits)
    assert [c.n for c in kept] == [1]
    assert "[2]" not in draft


def test_prune_weak_keeps_one_when_all_weak() -> None:
    cits = [Citation(n=1, source_file="a.md", support=0.2),
            Citation(n=2, source_file="b.md", support=0.1)]
    _, kept = _prune_weak("Claim. [1][2]", cits)
    assert len(kept) == 1


def test_looks_like_mcq_needs_two_lettered_options() -> None:
    mcq = "3.Which is not a member of class (in C++)?\na.Static\nb.Const\nc.Friend\nd.Virtual"
    assert _looks_like_mcq(mcq)
    assert not _looks_like_mcq("Refunds are issued within 14 days of purchase.")
    assert not _looks_like_mcq("a. one option is not enough on its own")


def test_looks_like_mcq_handles_symbolic_options() -> None:
    # Big-O notation options (Greek letters), not plain Latin-letter text —
    # the exact case that slipped through an earlier, too-strict version.
    mcq = "9.How many bits of memory? \na.Θ(log n) c. Θ(n) b.Θ(n2) d. Θ(2n)"
    assert _looks_like_mcq(mcq)
    # A decimal at line-start must not be mistaken for an option marker.
    assert not _looks_like_mcq("a.1.20\nb.2.40\nthese are prices, not options")


def _decision(**kw) -> RouterDecision:
    base = dict(intent="doc_qa", rewritten_question="q", needs_tables=False)
    base.update(kw)
    return RouterDecision(**base)


def test_after_verification_routes_to_fallback_only_for_refusal_text_and_ordinal() -> None:
    refusal = "I couldn't find this in your documents."
    hedged = "_Low confidence — the documents only partially support this._\n\n" + refusal

    assert _after_verification(
        {"answer": refusal, "decision": _decision(ordinals=[3])}
    ) == "fallback"
    # The hedge path prefixes a warning but keeps the refusal text -> still routes.
    assert _after_verification(
        {"answer": hedged, "decision": _decision(ordinals=[9])}
    ) == "fallback"
    # Multiple targeted items -> still routes.
    assert _after_verification(
        {"answer": refusal, "decision": _decision(ordinals=[2, 3])}
    ) == "fallback"
    # A real (non-refusal) answer -> no fallback, whatever the ordinal.
    assert _after_verification(
        {"answer": "Overloaded functions can accept same number of arguments. [1]",
         "decision": _decision(ordinals=[12])}
    ) == "end"
    # Refusal text but no positional target -> no fallback (nothing safe to solve).
    assert _after_verification(
        {"answer": refusal, "decision": _decision(ordinals=[])}
    ) == "end"


def _pinned_chunk(ordinal: int, body: str, *, image_ref: str | None = None) -> Retrieved:
    chunk = Chunk(
        chunk_id=f"c{ordinal}", doc_id="d", source_file="exam.pdf",
        source_kind=SourceKind.DOCUMENT, text=body, body=body, ordinal=ordinal,
        image_ref=image_ref,
    )
    return Retrieved(chunk=chunk, score=1.0, components={"pinned": 1.0})


def test_fallback_reports_a_non_mcq_target_instead_of_silently_dropping_it(monkeypatch) -> None:
    mcq = "2.Which is correct?\na.one\nb.two\nc.three\nd.four"
    prose = "5.Explain the impact of the incident on customers."  # not an MCQ

    monkeypatch.setattr(
        "grounded.agents.fallback.chat",
        lambda messages, model=None: "Final answer: a. one",
    )

    state = {
        "decision": _decision(ordinals=[2, 5]),
        "answer": "I couldn't find this in your documents.",
        "passages": [_pinned_chunk(2, mcq), _pinned_chunk(5, prose)],
    }
    out = run_fallback(state)

    assert out["fallback_used"] is True
    assert "Question 2" in out["answer"] and "Final answer: a. one" in out["answer"]
    # The non-MCQ target is named, not silently missing from the combined answer.
    assert "Question 5" in out["answer"]
    assert "declining to guess" in out["answer"]
    assert "1/2" in out["notes"][0]


def test_fallback_reads_the_real_image_for_a_chart_backed_item(monkeypatch, tmp_path) -> None:
    # Q24-28 style: the pinned chunk carries an image_ref (see
    # ingest/enumerate.py's _attach_page_figures) pointing at the chart it
    # depends on. When a vision provider is configured, the actual image
    # bytes must be sent instead of (or in addition to) the plain-text chat.
    mcq = "24.In how many years were exports higher?\na.2\nb.3\nc.4\nd.5"
    img_path = tmp_path / "d" / "chart.png"
    img_path.parent.mkdir(parents=True)
    img_path.write_bytes(b"fake-png-bytes")

    monkeypatch.setattr("grounded.agents.fallback.vision_available", lambda: True)
    monkeypatch.setattr("grounded.agents.fallback.image_store_path", lambda ref: tmp_path / ref)
    monkeypatch.setattr(
        "grounded.agents.fallback.chat",
        lambda messages, model=None: (_ for _ in ()).throw(
            AssertionError("should not use text-only chat")
        ),
    )
    seen: dict = {}

    def fake_chat_with_image(messages, image_bytes, model=None):
        seen["image_bytes"] = image_bytes
        seen["messages"] = messages
        return "Final answer: c. 4"

    monkeypatch.setattr("grounded.agents.fallback.chat_with_image", fake_chat_with_image)

    state = {
        "decision": _decision(ordinals=[24]),
        "answer": "I couldn't find this in your documents.",
        "passages": [_pinned_chunk(24, mcq, image_ref="d/chart.png")],
    }
    out = run_fallback(state)

    assert seen["image_bytes"] == b"fake-png-bytes"
    assert "Final answer: c. 4" in out["answer"]
    assert out["notes"][0].startswith("fallback: answered 1/1")


def test_fallback_declines_rather_than_forcing_a_letter_when_told_it_cannot_tell(
    monkeypatch,
) -> None:
    mcq = "24.In how many years were exports higher?\na.2\nb.3\nc.4\nd.5"
    monkeypatch.setattr(
        "grounded.agents.fallback.chat",
        lambda messages, model=None: "The lines cross repeatedly and I can't tell "
        "them apart.\nFinal answer: cannot determine confidently — the two series "
        "overlap.",
    )

    state = {
        "decision": _decision(ordinals=[24]),
        "answer": "I couldn't find this in your documents.",
        "passages": [_pinned_chunk(24, mcq)],
    }
    out = run_fallback(state)

    # Surfaced honestly, but not counted as an actual answer.
    assert "cannot determine confidently" in out["answer"]
    assert out["notes"][0].startswith("fallback: answered 0/1")


def test_resolve_subset_ordinals_remaps_positions_within_the_puzzle_set() -> None:
    from grounded.agents.router import _resolve_subset_ordinals

    subset = [31, 32, 33, 34]
    # "the 4th pattern question" -> model says [4] -> should become [34]
    assert _resolve_subset_ordinals([4], "the fourth number pattern question", subset) == [34]
    assert _resolve_subset_ordinals([2], "answer to 2nd puzzle", subset) == [32]
    # already correct -> untouched
    assert _resolve_subset_ordinals([33], "the third pattern question", subset) == [33]
    # not about the puzzle set -> untouched
    assert _resolve_subset_ordinals([4], "what is the answer to question 4", subset) == [4]
    # position out of range -> untouched (don't guess)
    assert _resolve_subset_ordinals([9], "the 9th pattern question", subset) == [9]
    # no subset in this doc -> untouched
    assert _resolve_subset_ordinals([4], "the fourth pattern question", []) == [4]


def test_fallback_self_consistency_takes_the_majority_answer(monkeypatch) -> None:
    from grounded.config import settings
    monkeypatch.setattr(settings, "fallback_votes", 3)
    mcq = "24.Which is correct?\na.one\nb.two\nc.three\nd.four"
    replies = iter([
        "reasoning A\nFinal answer: c. three",
        "reasoning B\nFinal answer: a. one",
        "reasoning C\nFinal answer: c. three",
    ])
    monkeypatch.setattr(
        "grounded.agents.fallback.chat", lambda messages, model=None: next(replies)
    )
    state = {
        "decision": _decision(ordinals=[24]),
        "answer": "I couldn't find this in your documents.",
        "passages": [_pinned_chunk(24, mcq)],
    }
    out = run_fallback(state)
    assert "Final answer: c. three" in out["answer"]  # 2/3 majority wins
    assert out["notes"][0].startswith("fallback: answered 1/1")


def test_fallback_self_consistency_reports_when_attempts_all_disagree(monkeypatch) -> None:
    from grounded.config import settings
    monkeypatch.setattr(settings, "fallback_votes", 3)
    mcq = "24.Which is correct?\na.one\nb.two\nc.three\nd.four"
    replies = iter([
        "Final answer: a. one",
        "Final answer: b. two",
        "Final answer: d. four",
    ])
    monkeypatch.setattr(
        "grounded.agents.fallback.chat", lambda messages, model=None: next(replies)
    )
    state = {
        "decision": _decision(ordinals=[24]),
        "answer": "I couldn't find this in your documents.",
        "passages": [_pinned_chunk(24, mcq)],
    }
    out = run_fallback(state)
    assert "different answer each time" in out["answer"]
    assert out["notes"][0].startswith("fallback: answered 0/1")


def test_fallback_solves_an_image_task_with_no_lettered_options(monkeypatch, tmp_path) -> None:
    # Problem 2: an image question with no lettered options — _looks_like_mcq is
    # false, but it's still a self-contained question with a definite answer,
    # so the fallback should solve it (via the attached image).
    body = "Sub-question 1 of 4 (instruction: which number replaces the ?): ? ="
    img = tmp_path / "d" / "p1.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"puzzle-png")

    monkeypatch.setattr("grounded.agents.fallback.vision_available", lambda: True)
    monkeypatch.setattr("grounded.agents.fallback.image_store_path", lambda ref: tmp_path / ref)
    seen: dict = {}

    def fake_chat_with_image(messages, image_bytes, model=None):
        seen["system"] = messages[0]["content"]
        return "Each number doubles.\nFinal answer: 32"

    monkeypatch.setattr("grounded.agents.fallback.chat_with_image", fake_chat_with_image)

    state = {
        "decision": _decision(ordinals=[31]),
        "answer": "I couldn't find this in your documents.",
        "passages": [_pinned_chunk(31, body, image_ref="d/p1.png")],
    }
    out = run_fallback(state)

    assert "as an image" in seen["system"].lower()  # image-task prompt, not MCQ prompt
    assert "multiple-choice" not in seen["system"].lower()
    assert "Final answer: 32" in out["answer"]
    assert out["notes"][0].startswith("fallback: answered 1/1")
