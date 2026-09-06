"""Enumerated-list handling: parsing, chunking, and positional-query routing.
No Ollama or vector store needed."""

from __future__ import annotations

from grounded.ingest.chunking import chunk_document
from grounded.ingest.enumerate import (
    _attach_page_figures,
    _group_shared_stimulus,
    _number_unlabelled_items,
    _relocate_shared_preambles,
    _reorder_scrambled_options,
    split_enumerated,
)
from grounded.ingest.models import DocModel, Element, SourceKind

_EXAM = (
    "Questions 1-30 have exactly ONE correct answer.\n"
    "1. For the following code, which of the following is correct:\n"
    "a. option A\nb. option B\nc. option C\nd. option D\n"
    "2. What will be the output of the program?\n"
    "a. Finished\nb. Compilation fails\n"
    "3. Which of the following is not a member of a class in C++?\n"
    "a. Static function\nb. Const function\nc. Friend function\nd. Virtual function\n"
    "4. Which statement is correct?\na. one\nb. two\n"
    "5. Three coins are tossed. What is the probability of at most two heads?\n"
    "a. 3/4\nb. 3/8\nc. 1/4\nd. 7/8\n"
)


def _doc(*elements: Element) -> DocModel:
    return DocModel(
        doc_id="t", source_file="exam.pdf", source_kind=SourceKind.DOCUMENT,
        title="Exam", elements=list(elements),
    )


def test_split_enumerated_makes_one_element_per_question() -> None:
    doc = split_enumerated(_doc(Element(kind="paragraph", text=_EXAM)))
    items = [e for e in doc.elements if e.ordinal is not None]
    assert [e.ordinal for e in items] == [1, 2, 3, 4, 5]
    q3 = next(e for e in items if e.ordinal == 3)
    assert "not a member of a class in C++" in q3.text
    assert "Friend function" in q3.text
    assert "probability" not in q3.text  # didn't bleed into Q5


def test_split_enumerated_keeps_preamble() -> None:
    doc = split_enumerated(_doc(Element(kind="paragraph", text=_EXAM)))
    assert doc.elements[0].ordinal is None
    assert "exactly ONE correct answer" in doc.elements[0].text


def test_questions_split_across_elements_are_recombined() -> None:
    # PDF page breaks routinely cut one item's text across two blocks.
    doc = split_enumerated(_doc(
        Element(kind="paragraph", text="1. First question stem\na. a\nb. b\n2. Second"),
        Element(kind="paragraph", text=" question stem continues\na. a\nb. b\n"
                "3. Third question\na. a\nb. b\n4. Fourth\n5. Fifth\n"),
    ))
    q2 = next(e for e in doc.elements if e.ordinal == 2)
    assert "Second question stem continues" in " ".join(q2.text.split())


def test_non_enumerated_text_is_left_alone() -> None:
    prose = _doc(Element(kind="paragraph", text="Refunds are issued within 14 days. "
                         "Enterprise plans get 30 days. Contact support to start one."))
    assert split_enumerated(prose) is prose


def test_short_numbered_list_is_not_restructured() -> None:
    doc = _doc(Element(kind="paragraph", text="1. one\n2. two\n3. three"))
    assert split_enumerated(doc) is doc  # below the 4-item floor


def test_each_question_becomes_its_own_atomic_chunk() -> None:
    doc = split_enumerated(_doc(
        Element(kind="heading", text="Problem 1", heading_level=1, section_path=["Problem 1"]),
        Element(kind="paragraph", text=_EXAM, section_path=["Problem 1"]),
    ))
    chunks = chunk_document(doc)
    by_ord = {c.ordinal: c for c in chunks if c.ordinal is not None}
    assert set(by_ord) == {1, 2, 3, 4, 5}
    assert "Friend function" in by_ord[3].body
    assert "output of the program" not in by_ord[3].body
    assert "item 3" in by_ord[3].text  # retrieval hint in the embedded text

# Positional-reference extraction ("the 3rd question", "question 2 and 3") is
# now the router's own LLM call (see router.py's _SYSTEM prompt) rather than a
# deterministic function — nothing here to unit-test without a live call.


def test_reorder_scrambled_options_fixes_2column_pdf_layout() -> None:
    # Real case: a 2-column option layout extracts in visual row order
    # (a, c, b, d) instead of answer order — this scrambled a fallback answer
    # by making the model misquote which text belonged to option "c".
    scrambled = (
        "22.The following SQL statement ... privileges?\n"
        "a.SELECT ON R \n"
        "c. UPDATE ON S \n"
        "b.UPDATE ON R(a) \n"
        "d. SELECT ON T"
    )
    fixed = _reorder_scrambled_options(scrambled)
    assert fixed.index("a.SELECT ON R") < fixed.index("b.UPDATE ON R(a)")
    assert fixed.index("b.UPDATE ON R(a)") < fixed.index("c. UPDATE ON S")
    assert fixed.index("c. UPDATE ON S") < fixed.index("d. SELECT ON T")
    assert "22.The following SQL statement" in fixed  # preamble kept


def test_reorder_scrambled_options_ignores_schema_notation_in_stem() -> None:
    # "R(a,b)", "S(b,c)", "R.a)" must not be mistaken for option markers —
    # each contributed a false match in an earlier version of this fix.
    text = (
        "22.Tables R(a,b), S(b,c), and T(a,c). WHERE NOT EXISTS (SELECT a "
        "FROM T WHERE T.a = R.a)\n"
        "a.SELECT ON R \n"
        "c. UPDATE ON S \n"
        "b.UPDATE ON R(a) \n"
        "d. SELECT ON T"
    )
    fixed = _reorder_scrambled_options(text)
    assert fixed.index("a.SELECT ON R") < fixed.index("b.UPDATE ON R(a)")
    assert fixed.index("b.UPDATE ON R(a)") < fixed.index("c. UPDATE ON S")


def test_reorder_scrambled_options_noop_when_already_ordered() -> None:
    ok = "3.Which is not a member?\na.Static\nb.Const\nc.Friend\nd.Virtual"
    assert _reorder_scrambled_options(ok) == ok


def test_reorder_scrambled_options_noop_on_too_few_or_repeated() -> None:
    assert _reorder_scrambled_options("a.only one option") == "a.only one option"
    repeated = "a.one\nb.two\na.repeated letter\nb.also repeated"
    assert _reorder_scrambled_options(repeated) == repeated


def _item(ordinal: int, text: str, *, page: int | None = None) -> Element:
    return Element(kind="list_item", ordinal=ordinal, text=text, page=page)


def _figure(page: int, image_ref: str) -> Element:
    return Element(kind="figure", page=page, text="chart description", image_ref=image_ref)


def test_relocate_shared_preamble_moves_trailing_sentence_and_its_data() -> None:
    # The exact real case: a shared instruction PLUS a chart's OCR'd values
    # trail item 23, but both belong to 24 — and the two items don't even
    # have to be adjacent in the list (a heading can split them into
    # separate enumeration runs; this operates on the flat element list
    # after all runs are done, so it still finds ordinal 24 wherever it is).
    items = [
        _item(23, "23.SQL question.\na.one\nb.two\n\n"
              "Answer Questions 24-28 based on the following line graph.\n"
              "1.75\n1.5\n1995\n1996"),
        _item(24, "24.In how many years were exports higher?\na.2\nb.3"),
        _item(25, "25.Some other question.\na.x\nb.y"),
    ]
    out = _relocate_shared_preambles(items)
    by_ord = {e.ordinal: e for e in out}
    assert "Answer Questions" not in by_ord[23].text
    assert "1.75" not in by_ord[23].text
    assert by_ord[23].text.rstrip().endswith("b.two")
    assert by_ord[24].text.startswith("Answer Questions 24-28")
    assert "1.75" in by_ord[24].text
    assert "24.In how many years" in by_ord[24].text  # own content kept, after the moved tail
    assert "Answer Questions" not in by_ord[25].text  # untouched


def test_relocate_shared_preamble_handles_and_separated_range() -> None:
    items = [
        _item(28, "28.Last question of the set.\na.x\nb.y\n\n"
              "For Questions 29 and 30, find the statement that must be true."),
        _item(29, "29.Erin has a bird.\na.p\nb.q"),
        _item(30, "30.Ten new shows.\na.p\nb.q"),
    ]
    out = _relocate_shared_preambles(items)
    by_ord = {e.ordinal: e for e in out}
    assert "For Questions" not in by_ord[28].text
    assert by_ord[29].text.startswith("For Questions 29 and 30")
    assert "For Questions" not in by_ord[30].text  # only moved to the first target


def test_relocate_shared_preamble_noop_without_a_forward_target() -> None:
    # No item numbered 99 exists -> nothing to relocate to, leave as-is.
    items = [_item(5, "5.Text.\na.x\n\nFor Questions 99 and 100, do the thing.")]
    out = _relocate_shared_preambles(items)
    assert out[0].text == items[0].text


def test_attach_page_figures_links_a_single_chart_to_every_item_on_its_page() -> None:
    # Real case: Q24-28 all share one line-graph chart on the same page — the
    # chart isn't itself a numbered item, so without this, none of Q24-28's
    # chunks would ever carry the image_ref the fallback agent needs for a
    # vision re-read.
    items = [
        _figure(page=5, image_ref="doc1/chart5.png"),
        _item(24, "24.Text.\na.2\nb.3", page=5),
        _item(25, "25.Text.\na.x\nb.y", page=5),
        _item(30, "30.Different page.\na.x\nb.y", page=6),
    ]
    out = _attach_page_figures(items)
    by_ord = {e.ordinal: e for e in out if e.ordinal is not None}
    assert by_ord[24].image_ref == "doc1/chart5.png"
    assert by_ord[25].image_ref == "doc1/chart5.png"
    assert by_ord[30].image_ref is None  # different page, untouched


def test_attach_page_figures_noop_when_a_page_has_more_than_one_figure() -> None:
    # Ambiguous which question goes with which chart -> link neither, rather
    # than guess.
    items = [
        _figure(page=5, image_ref="doc1/a.png"),
        _figure(page=5, image_ref="doc1/b.png"),
        _item(24, "24.Text.\na.x\nb.y", page=5),
    ]
    out = _attach_page_figures(items)
    by_ord = {e.ordinal: e for e in out if e.ordinal is not None}
    assert by_ord[24].image_ref is None


def test_attach_page_figures_does_not_override_an_existing_image_ref() -> None:
    items = [
        _figure(page=5, image_ref="doc1/chart5.png"),
        _item(24, "24.Text.\na.x\nb.y", page=5).model_copy(
            update={"image_ref": "doc1/own.png"}
        ),
    ]
    out = _attach_page_figures(items)
    by_ord = {e.ordinal: e for e in out if e.ordinal is not None}
    assert by_ord[24].image_ref == "doc1/own.png"


def test_group_shared_stimulus_gives_every_member_the_stimulus_and_sibling_stems() -> None:
    items = [
        _item(24, "Answer Questions 24-28 based on the following line graph.\n"
              "Company A: 1.75, 1.5\nCompany B: 0.75, 0.75\n"
              "24.In how many years were exports more than imports for Company A?\n"
              "a.2\nb.3"),
        _item(25, "25.If imports rose 40%, what is the ratio of exports to imports?\na.1.2\nb.1.3"),
        _item(26, "26.Some other question about the graph.\na.x\nb.y"),
        _item(27, "27.Another.\na.x\nb.y"),
        _item(28, "28.Last of the set.\na.x\nb.y"),
        _item(29, "29.Unrelated question.\na.x\nb.y"),
    ]
    out = _group_shared_stimulus(items)
    by_ord = {e.ordinal: e for e in out}

    for o in (24, 25, 26, 27, 28):
        sc = by_ord[o].shared_context
        assert sc is not None
        assert "Company A: 1.75" in sc           # the stimulus
        assert "ratio of exports to imports" in sc  # Q25's framing reaches Q26
        assert "[Q28]" in sc                      # every sibling's stem is listed
    assert by_ord[29].shared_context is None      # outside the set, untouched


def test_group_shared_stimulus_shares_a_chart_image_ref_across_the_set() -> None:
    items = [
        _item(24, "Answer Questions 24-25 based on the following graph.\n"
              "24.First.\na.x\nb.y").model_copy(update={"image_ref": "d/chart.png"}),
        _item(25, "25.Second.\na.x\nb.y"),
    ]
    out = _group_shared_stimulus(items)
    by_ord = {e.ordinal: e for e in out}
    assert by_ord[25].image_ref == "d/chart.png"  # inherited from the set


def test_number_unlabelled_items_assigns_ordinals_continuing_the_run() -> None:
    els = [
        _item(30, "30.Last real question.\na.x\nb.y"),
        Element(kind="paragraph", text="For each of the following, which number "
                "replaces the question mark?"),
        Element(kind="paragraph", text="? =", image_ref="d/p1.png"),
        Element(kind="paragraph", text="? =", image_ref="d/p2.png"),
        Element(kind="paragraph", text="A full sentence of real body text ends the batch."),
    ]
    out = _number_unlabelled_items(els)
    items = [e for e in out if e.ordinal is not None and e.ordinal > 30]
    assert [p.ordinal for p in items] == [31, 32]
    assert all(p.kind == "list_item" for p in items)
    assert "Sub-question 1 of 2" in items[0].text
    assert items[0].image_ref == "d/p1.png"


def test_number_unlabelled_items_triggers_on_other_batch_phrasings() -> None:
    for prompt in (
        "Complete each of the following sequences:",
        "Find the missing value in each of these:",
        "For all of the following, what number comes next?",
    ):
        els = [
            _item(10, "10.q.\na.x\nb.y"),
            Element(kind="paragraph", text=prompt),
            Element(kind="paragraph", text="img", image_ref="d/a.png"),
            Element(kind="paragraph", text="img", image_ref="d/b.png"),
        ]
        out = _number_unlabelled_items(els)
        assert [e.ordinal for e in out if e.ordinal and e.ordinal > 10] == [11, 12]


def test_number_unlabelled_items_noop_on_a_normal_numbered_stem() -> None:
    # A numbered MCQ stem is also a question ending in "?" — must NOT trigger.
    els = [
        _item(3, "3.Which of the following is not a member of a class?\na.x\nb.y"),
        Element(kind="paragraph", text="img", image_ref="d/a.png"),
        Element(kind="paragraph", text="img", image_ref="d/b.png"),
    ]
    out = _number_unlabelled_items(els)
    assert not [e for e in out if e.ordinal and e.ordinal > 3]


def test_number_unlabelled_items_ignores_lookalike_lines() -> None:
    # Real false positives seen in the sample exam: form-filling boilerplate,
    # and a run of options mashed with the next stem — both say "the
    # following" and end with ":" but neither introduces an image batch.
    for junk in (
        "Please fill-in the following details:",
        "a.nR (number of tuples in R)\nc. nS\nb.min(nS, nR)\n\n22.The following "
        "SQL statement requires certain privileges to execute:",
    ):
        els = [
            _item(10, "10.q.\na.x\nb.y"),
            Element(kind="paragraph", text=junk),
            Element(kind="paragraph", text="img", image_ref="d/a.png"),
            Element(kind="paragraph", text="img", image_ref="d/b.png"),
        ]
        out = _number_unlabelled_items(els)
        assert not [e for e in out if e.ordinal and e.ordinal > 10]


def test_number_unlabelled_items_noop_with_only_one_image() -> None:
    els = [
        _item(5, "5.q.\na.x\nb.y"),
        Element(kind="paragraph", text="For each of the following, which is correct?"),
        Element(kind="paragraph", text="img", image_ref="d/a.png"),
    ]
    out = _number_unlabelled_items(els)
    assert not [e for e in out if e.ordinal and e.ordinal > 5]


def test_number_unlabelled_items_structural_fallback_without_a_prompt() -> None:
    # The exam's puzzle prompt doesn't always survive PDF text extraction —
    # a run of image-only "?" paragraphs after the numbered run is enough.
    els = [
        _item(30, "30.Last real question.\na.x\nb.y"),
        Element(kind="paragraph", text="3\n5\n8\n13\n22\n?", image_ref="d/p1.png"),
        Element(kind="paragraph", text="? =", image_ref=None),  # tiny separator, ignored
        Element(kind="paragraph", text="7 3 6 2\n2 8 5 4\n?", image_ref="d/p2.png"),
    ]
    out = _number_unlabelled_items(els)
    items = [e for e in out if e.ordinal and e.ordinal > 30]
    assert [e.ordinal for e in items] == [31, 32]
    assert "image sub-question set" in items[0].text


def test_number_unlabelled_items_numbers_parts_in_page_reading_order() -> None:
    # Parser emits images in a different order than they sit on the page —
    # "the first puzzle" must mean the leftmost one, not the first emitted.
    # Real case: 4 puzzles side by side, tops 548/558/560/557 (within ~12pt) —
    # a naive y-bucket split the first one off; a tolerance keeps them one row.
    def img(x: float, top: float, text: str) -> Element:
        return Element(kind="paragraph", text=text, image_ref="d/x.png",
                       bbox=(x, top, x + 110, top + 200))

    els = [
        _item(30, "30.Last real question.\na.x\nb.y"),
        img(489, 548, "puzzle D on page: ? here"),   # rightmost, emitted first
        img(203, 558, "puzzle B on page: ? here"),
        img(61, 560, "puzzle A on page: ? here"),     # leftmost
        img(345, 557, "puzzle C on page: ? here"),
    ]
    out = _number_unlabelled_items(els)
    by_ord = {e.ordinal: e.text for e in out if e.ordinal and e.ordinal > 30}
    assert "Sub-question 1" in by_ord[31] and "puzzle A" in by_ord[31]
    assert "Sub-question 2" in by_ord[32] and "puzzle B" in by_ord[32]
    assert "Sub-question 3" in by_ord[33] and "puzzle C" in by_ord[33]
    assert "Sub-question 4" in by_ord[34] and "puzzle D" in by_ord[34]


def test_number_unlabelled_items_structural_ignores_images_without_a_question_mark() -> None:
    # Two decorative images after the last question — no "?" — left alone.
    els = [
        _item(9, "9.q.\na.x\nb.y"),
        Element(kind="paragraph", text="Figure: company logo", image_ref="d/a.png"),
        Element(kind="paragraph", text="Figure: office photo", image_ref="d/b.png"),
    ]
    out = _number_unlabelled_items(els)
    assert not [e for e in out if e.ordinal and e.ordinal > 9]
