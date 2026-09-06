"""Ingestion tests that don't need Ollama or a vector store."""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded.ingest import parsers
from grounded.ingest.chunking import chunk_document
from grounded.ingest.models import DocModel, Element, SourceKind

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def _doc(*elements: Element) -> DocModel:
    return DocModel(
        doc_id="t", source_file="t.md", source_kind=SourceKind.DOCUMENT,
        title="Test", elements=list(elements),
    )


def test_markdown_headings_become_section_paths() -> None:
    doc = parsers.parse(SAMPLES / "refund_policy.md")
    sections = {" > ".join(e.section_path) for e in doc.elements if e.section_path}
    assert any("Refund window" in p for p in sections)
    assert any("Cancellation" in p for p in sections)
    assert doc.source_kind is SourceKind.DOCUMENT


def test_code_file_splits_on_definitions() -> None:
    doc = parsers.parse(SAMPLES / "retrieval_demo.py")
    assert doc.source_kind is SourceKind.CODE
    assert doc.extra["language"] == "python"
    names = " ".join(e.text for e in doc.elements)
    assert "def reciprocal_rank_fusion" in names
    assert "class BM25Index" in names
    assert len(doc.elements) >= 3  # split, not one blob


def test_chunker_keeps_tables_atomic_and_prepends_breadcrumb() -> None:
    doc = _doc(
        Element(kind="heading", text="Revenue", heading_level=1, section_path=["Revenue"]),
        Element(kind="paragraph", text="Some prose about revenue.", section_path=["Revenue"]),
        Element(kind="table", text="| a | b |\n| --- | --- |\n| 1 | 2 |",
                table_markdown="| a | b |", section_path=["Revenue"]),
    )
    chunks = chunk_document(doc)
    table_chunks = [c for c in chunks if c.element_kind == "table"]
    assert len(table_chunks) == 1
    assert all(c.text.startswith("[Test") for c in chunks)


def test_csv_detected_as_spreadsheet_extension() -> None:
    assert (SAMPLES / "q3_sales.csv").suffix.lower() in parsers.SPREADSHEET_EXT


@pytest.mark.parametrize("name", ["refund_policy.md", "onboarding_notes.txt", "retrieval_demo.py"])
def test_supported(name: str) -> None:
    assert parsers.supported(SAMPLES / name)


def test_html_parses_via_the_markdown_path(tmp_path: Path) -> None:
    html = tmp_path / "faq.html"
    html.write_text(
        "<!doctype html><html><head><title>FAQ</title>"
        "<style>.x{color:red}</style></head><body>"
        "<nav>Home | Pricing</nav>"
        "<h1>Billing</h1><p>Enterprise plans bill <b>annually</b>.</p>"
        "<h2>Refunds</h2><p>Enterprise contracts have a 30-day window.</p>"
        "<table><tr><th>Plan</th><th>Window</th></tr>"
        "<tr><td>Starter</td><td>14 days</td></tr></table>"
        "<script>bad()</script></body></html>",
        encoding="utf-8",
    )
    assert parsers.supported(html)
    assert parsers._resolve_kind(html) == "html"

    doc = parsers.parse(html)
    kinds = {e.kind for e in doc.elements}
    assert "heading" in kinds
    # headings drive section paths
    refund = next(e for e in doc.elements if "30-day window" in e.text)
    assert refund.section_path == ["Billing", "Refunds"]
    # script/style dropped, emphasis flattened, table content kept
    body = "\n".join(e.text for e in doc.elements)
    assert "bad()" not in body and "color:red" not in body
    assert "annually" in body
    assert "| Starter | 14 days |" in body


class _FakeSeries:
    def __init__(self, name: str, values: tuple) -> None:
        self.name = name
        self.values = values


class _FakePlot:
    def __init__(self, categories: list[str]) -> None:
        self.categories = categories


class _FakeChartType:
    name = "PIE"


class _FakeChart:
    def __init__(self, categories, series) -> None:
        self.plots = [_FakePlot(categories)]
        self.series = series
        self.chart_type = _FakeChartType()


def test_pptx_chart_becomes_a_markdown_table_with_caption() -> None:
    chart = _FakeChart(
        ["Enterprise", "Team", "Starter"],
        [_FakeSeries("Q3 revenue (USD)", (1_276_800.0, 930_000.0, 452_200.0))],
    )
    el = parsers._pptx_chart_element(chart, page=6, title="Revenue share by segment")
    assert el is not None and el.kind == "table"
    assert "pie chart — Revenue share by segment" in el.text
    # big numbers render readably, not in scientific notation
    assert "1,276,800" in el.text and "e+" not in el.text
    assert "| Enterprise | 1,276,800 |" in el.table_markdown


def test_pptx_chart_element_returns_none_on_empty_chart() -> None:
    assert parsers._pptx_chart_element(_FakeChart([], []), page=1, title="") is None


def test_board_deck_sample_exposes_its_charts() -> None:
    doc = parsers.parse(SAMPLES / "q3_board_deck.pptx")
    chart_tables = [
        e.text for e in doc.elements
        if e.kind == "table" and "chart" in e.text.split("\n", 1)[0].lower()
    ]
    assert len(chart_tables) == 3  # column, line, pie
    joined = "\n".join(chart_tables)
    assert "996,000" in joined  # region column chart
    assert "2.66" in joined     # revenue-trend line chart
    assert "1,276,800" in joined  # segment pie chart


class _Rect:
    def __init__(self, x0: float, y0: float, w: float, h: float) -> None:
        self.x0, self.y0, self.width, self.height = x0, y0, w, h
        self.y1 = y0 + h


class _FakePage:
    def __init__(self, drawings: list[dict]) -> None:
        self._drawings = drawings

    def get_drawings(self) -> list[dict]:
        return self._drawings


def _bars(n: int, *, w: float = 15, step: float = 30, base_y: float = 500) -> list[dict]:
    # n same-width, evenly-spaced, bottom-aligned filled rects of varying height
    return [
        {"type": "f", "rect": _Rect(100 + i * step, base_y - (20 + i * 5), w, 20 + i * 5)}
        for i in range(n)
    ]


def test_has_bar_chart_detects_a_row_of_bars() -> None:
    assert parsers._has_bar_chart(_FakePage(_bars(9)))


def test_has_bar_chart_rejects_table_cell_shading() -> None:
    # header/cell fills: same y-band, wildly different widths, not a bar row
    cells = [
        {"type": "f", "rect": _Rect(50, 100, 200, 12)},
        {"type": "f", "rect": _Rect(50, 130, 60, 12)},
        {"type": "f", "rect": _Rect(50, 160, 340, 12)},
        {"type": "f", "rect": _Rect(50, 190, 90, 12)},
    ]
    assert not parsers._has_bar_chart(_FakePage(cells))


def test_has_bar_chart_rejects_too_few_bars() -> None:
    assert not parsers._has_bar_chart(_FakePage(_bars(3)))


def test_has_bar_chart_rejects_a_row_of_equal_height_boxes() -> None:
    # 5 same-width, evenly-spaced, bottom-aligned rects that are all the SAME
    # height — a strip of label/marker boxes (as on the exam's puzzle page),
    # not a data bar chart.
    boxes = [
        {"type": "f", "rect": _Rect(80 + i * 130, 675, 99, 21)} for i in range(5)
    ]
    assert not parsers._has_bar_chart(_FakePage(boxes))
