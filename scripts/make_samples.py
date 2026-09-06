"""Generate the binary sample files (PDF, DOCX, XLSX, PPTX, PNG) for the demo corpus.

Run once:  python scripts/make_samples.py

The text / CSV / code / markdown / HTML samples are checked in directly and not
touched here. All files describe the same fictional company, "Acme Cloud", and the
numbers line up across files so cross-document questions have real answers:

  * Total Q3 revenue: USD 2.66M  (business review PDF)
  * Sales-data total: USD 2,460,000  (q3_sales.csv)  <- product revenue only
  * Q3 revenue by region: NA 996k / EMEA 714k / APAC 540k / LATAM 210k
    (q3_sales.csv, the business review PDF's table AND its page-3 bar chart)
  * EMEA net revenue retention: 112%
  * 2026-03-09 EMEA outage: 47 minutes  (also see the postmortem markdown)
  * Engineering headcount: 42  (company_metrics.xlsx)
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _font(size: int):
    from PIL import ImageFont

    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 1. Q3 business review — multi-page PDF with real tables (reportlab)
# ---------------------------------------------------------------------------
def make_pdf() -> None:
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    story: list = []

    def para(text: str, style: str = "BodyText") -> None:
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 4 * mm))

    def grid(data: list[list[str]], col_widths=None) -> None:
        t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#25324d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#eef1f7")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c6c9da")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
        story.append(Spacer(1, 6 * mm))

    # ---- page 1
    para("Acme Cloud — Q3 2026 Business Review", "Title")
    para("Executive summary", "Heading1")
    para(
        "Total Q3 revenue reached <b>USD 2.66M</b>, up 18% quarter on quarter and "
        "41% year on year. Growth was led by new-logo acquisition in APAC and by "
        "expansion within the existing EMEA base, which delivered the strongest net "
        "revenue retention of any region at <b>112%</b>. The Enterprise segment is "
        "48% of revenue despite the smallest unit volume. Gross margin held at 79%; "
        "cash runway is 21 months. Two items need board attention: LATAM pipeline "
        "coverage is 1.9x against a 3.0x target, and support CSAT dipped to 4.1."
    )
    para("Headline metrics", "Heading2")
    grid([
        ["Metric", "Q3 2026", "Q2 2026", "Q3 2025", "QoQ", "YoY", "Plan"],
        ["Revenue (USD M)", "2.66", "2.25", "1.89", "+18%", "+41%", "2.55"],
        ["Net revenue retention", "106%", "104%", "101%", "+2pp", "+5pp", "105%"],
        ["Gross margin", "79%", "79%", "77%", "flat", "+2pp", "78%"],
        ["New logos", "612", "544", "474", "+13%", "+29%", "600"],
        ["Logo churn (qtr)", "3.1%", "2.8%", "3.4%", "+0.3pp", "-0.3pp", "3.0%"],
        ["Support CSAT", "4.1", "4.5", "4.4", "-0.4", "-0.3", "4.4"],
        ["API availability", "99.94%", "99.97%", "99.91%", "-0.03pp", "+0.03pp", "99.9%"],
    ], col_widths=[38 * mm, 20 * mm, 20 * mm, 20 * mm, 18 * mm, 18 * mm, 18 * mm])
    story.append(PageBreak())

    # ---- page 2
    para("Revenue detail", "Heading1")
    para("Revenue by region — all products (USD)", "Heading2")
    grid([
        ["Region", "Q3 rev", "Q2 rev", "QoQ", "New logos", "Churned", "NRR"],
        ["North America", "996,000", "889,000", "+12%", "184", "41", "104%"],
        ["EMEA", "714,000", "655,000", "+9%", "135", "24", "112%"],
        ["APAC", "540,000", "403,000", "+34%", "251", "58", "108%"],
        ["LATAM", "210,000", "198,000", "+6%", "42", "16", "99%"],
        ["Total (product)", "2,460,000", "2,145,000", "+15%", "612", "139", "106%"],
    ], col_widths=[34 * mm, 24 * mm, 24 * mm, 16 * mm, 22 * mm, 20 * mm, 16 * mm])
    para(
        "Product revenue (2.46M) plus professional services (0.14M) and marketplace "
        "revenue share (0.06M) gives the 2.66M total.", "BodyText",
    )
    para("Revenue by segment", "Heading2")
    grid([
        ["Segment", "Q3 revenue", "Share", "Customers", "ARPA (USD)", "Gross churn"],
        ["Enterprise", "1,276,800", "48%", "99", "12,897", "0.8%"],
        ["Team", "930,000", "35%", "1,240", "750", "1.6%"],
        ["Starter", "452,200", "17%", "8,900", "51", "2.9%"],
    ], col_widths=[28 * mm, 26 * mm, 16 * mm, 22 * mm, 24 * mm, 22 * mm])
    story.append(PageBreak())

    # ---- page 3: revenue-by-region bar chart (drawn as vector bars, so the
    # ingest pipeline's chart detector picks it up and reads it with vision —
    # the numbers match q3_sales.csv and the page-2 table exactly).
    para("Q3 revenue by region (chart)", "Heading1")
    _regions = ["North America", "EMEA", "APAC", "LATAM"]
    _rev = [996_000, 714_000, 540_000, 210_000]
    chart = Drawing(430, 250)
    chart.add(String(0, 232, "Q3 2026 product revenue by region (USD)",
                     fontName="Helvetica-Bold", fontSize=11))
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 35, 25, 360, 185
    bc.data = [_rev]
    bc.barWidth = 34
    bc.groupSpacing = 30
    bc.bars[0].fillColor = colors.HexColor("#2f6fb0")
    bc.bars[0].strokeColor = None
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = 1_100_000
    bc.valueAxis.valueStep = 200_000
    bc.categoryAxis.categoryNames = _regions
    bc.categoryAxis.labels.fontSize = 8
    bc.barLabels.fontSize = 8
    bc.barLabelFormat = "%d"
    bc.barLabels.dy = 6
    chart.add(bc)
    story.append(chart)
    story.append(Spacer(1, 6 * mm))
    para(
        "The chart above plots Q3 2026 product revenue for each region. North "
        "America is the largest at USD 996,000 and LATAM the smallest at USD "
        "210,000; the four regions sum to the USD 2,460,000 product-revenue "
        "total. APAC grew fastest quarter on quarter (+34%).", "BodyText",
    )
    story.append(PageBreak())

    # ---- page 4
    para("Customers and support", "Heading1")
    para(
        "New logos were 612 for the quarter (APAC 41%, NA 30%, EMEA 22%, LATAM 7%). "
        "Gross revenue churn was 1.4%; net was strongly positive on EMEA expansion.",
        "BodyText",
    )
    para("Support metrics", "Heading2")
    grid([
        ["Metric", "Jul", "Aug", "Sep", "Q3", "Target"],
        ["Median first response", "2h40m", "2h18m", "2h14m", "2h24m", "3h00m"],
        ["Median full resolution", "2.1d", "1.9d", "1.8d", "1.9d", "2.0d"],
        ["Tickets created", "3,180", "3,140", "3,092", "9,412", "-"],
        ["Backlog at month end", "78", "71", "63", "63", "< 80"],
        ["CSAT", "3.9", "4.1", "4.2", "4.1", "4.4"],
    ], col_widths=[42 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm])
    story.append(PageBreak())

    # ---- page 5
    para("Reliability, risks and outlook", "Heading1")
    para(
        "API availability was 99.94% against a 99.9% SLO. One Sev-1 incident: on "
        "2026-03-09 an EMEA deploy set the database connection-pool ceiling to 20 "
        "instead of 200, causing a 47-minute partial outage in EMEA only. Root cause "
        "and six action items are in the postmortem.", "BodyText",
    )
    para("Key risks", "Heading2")
    grid([
        ["Risk", "Likelihood", "Impact", "Severity", "Owner", "Due"],
        ["LATAM pipeline coverage 1.9x vs 3.0x", "High", "High", "High", "Sales", "Q4"],
        ["Support CSAT 4.1 below 4.4 floor", "High", "Medium", "High", "Support", "Q4"],
        ["Two senior Eng roles open 90+ days", "Medium", "Medium", "Medium", "Eng", "Nov"],
        ["FX exposure on EUR/GBP contracts", "Medium", "Low", "Low", "Finance", "ongoing"],
    ], col_widths=[52 * mm, 20 * mm, 18 * mm, 18 * mm, 18 * mm, 16 * mm])
    para(
        "Q4 priorities: (1) close the LATAM pipeline gap with a partner motion, "
        "(2) staff support to plan and recover CSAT to 4.4+, (3) ship EU data "
        "residency GA for Enterprise, (4) hold gross margin at 79% or better.",
        "BodyText",
    )

    SimpleDocTemplate(
        str(OUT / "q3_business_review.pdf"), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title="Acme Cloud — Q3 2026 Business Review",
    ).build(story)
    print("wrote q3_business_review.pdf  (5 pages: tables + a revenue-by-region chart)")


# ---------------------------------------------------------------------------
# 2. Data Processing Addendum — DOCX
# ---------------------------------------------------------------------------
def make_docx() -> None:
    import docx

    def add_table(document, rows: list[list[str]]) -> None:
        t = document.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Light Grid Accent 1"
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                t.rows[r].cells[c].text = val
        document.add_paragraph("")

    d = docx.Document()
    d.add_heading("Acme Cloud — Data Processing Addendum", level=0)
    d.add_paragraph(
        "This Data Processing Addendum (\"DPA\") forms part of the Master "
        "Subscription Agreement between Acme Cloud, Inc. (\"Acme\", processor) and "
        "the customer (\"Customer\", controller). Version 2.4, effective 2026-01-15."
    )

    d.add_heading("1. Definitions", level=1)
    d.add_paragraph(
        "\"Customer Personal Data\" means personal data contained in Customer "
        "Content that Acme processes on Customer's behalf. \"Sub-processor\" means "
        "a third party engaged by Acme to process Customer Personal Data. "
        "\"Applicable Data Protection Law\" means the GDPR, the UK GDPR, and any "
        "other law relating to the protection of personal data applicable to the "
        "processing under the Agreement."
    )

    d.add_heading("2. Roles and scope of processing", level=1)
    d.add_paragraph(
        "Customer is the controller and Acme is the processor of Customer "
        "Personal Data. Acme processes Customer Personal Data only on documented "
        "instructions from Customer, including with regard to international "
        "transfers, unless required to do otherwise by law."
    )

    d.add_heading("3. Sub-processors", level=1)
    d.add_paragraph(
        "Customer authorises Acme to engage the Sub-processors listed below. Acme "
        "gives Customer at least 30 days' prior notice of the addition or "
        "replacement of a Sub-processor by updating the Register and notifying "
        "subscribed contacts. Customer may object on reasonable data-protection "
        "grounds within that period; if the objection cannot be resolved, Customer "
        "may terminate the affected Services."
    )
    add_table(d, [
        ["Sub-processor", "Purpose", "Data categories", "Region", "Engaged since"],
        ["CloudHost Inc.", "Compute and object storage", "All Customer Content",
         "EU (Frankfurt), US (Virginia)", "2021-04"],
        ["MailRoute Ltd.", "Transactional email delivery", "Email address, name",
         "US (Oregon)", "2021-04"],
        ["TraceIO GmbH", "Application error monitoring", "IP address, request metadata",
         "EU (Ireland)", "2022-09"],
        ["PayLink S.A.", "Subscription payment processing", "Billing contact, card token",
         "EU (Paris)", "2021-04"],
        ["VoiceScribe Inc.", "Support call transcription (opt-in)",
         "Call audio, transcripts", "US (Virginia)", "2025-02"],
    ])

    d.add_heading("4. Security measures", level=1)
    d.add_paragraph(
        "Acme maintains technical and organisational measures including: "
        "encryption of Customer Content in transit (TLS 1.2+) and at rest "
        "(AES-256); role-based access control with least privilege and quarterly "
        "access reviews; centralised audit logging; annual third-party "
        "penetration testing; and a documented secure development lifecycle. Acme "
        "holds SOC 2 Type II and ISO/IEC 27001 certifications."
    )

    d.add_heading("5. Data retention and deletion", level=1)
    d.add_paragraph(
        "Acme retains Customer Content for the duration of the subscription. On "
        "termination or expiry, Customer Content is deleted from production "
        "systems within 30 days. The retention schedule below applies:"
    )
    add_table(d, [
        ["Data type", "Storage location", "Retention after termination", "Notes"],
        ["Customer Content (production)", "Primary database, object store",
         "Deleted within 30 days", "Account reactivatable during this window"],
        ["Backups of Customer Content", "Encrypted backup store",
         "Purged on a rolling 35-day cycle", "No copies remain after 35 days"],
        ["Audit logs (access to Content)", "Central log store",
         "Retained 12 months, then deleted", "Used for security investigations"],
        ["Billing and tax records", "Finance systems",
         "Retained 7 years", "Legal obligation; not Customer Personal Data"],
        ["Support tickets and transcripts", "Support platform",
         "Deleted within 90 days", "Unless a dispute is open"],
    ])
    d.add_paragraph(
        "On written request made before deletion, Acme will provide Customer a "
        "copy of Customer Content in a structured, machine-readable format."
    )

    d.add_heading("6. International transfers", level=1)
    d.add_paragraph(
        "Where Acme transfers Customer Personal Data out of the EEA, the UK, or "
        "Switzerland to a country without an adequacy decision, the transfer is "
        "governed by the Standard Contractual Clauses, incorporated by reference. "
        "Customers on the Enterprise plan may elect EU-only data residency, in "
        "which case Customer Content is stored and processed solely in EU regions."
    )

    d.add_heading("7. Data subject requests", level=1)
    d.add_paragraph(
        "Acme provides self-service tools that allow Customer to access, correct, "
        "export, and delete personal data. Acme will forward to Customer, without "
        "undue delay, any request received directly from a data subject relating "
        "to Customer Personal Data."
    )

    d.add_heading("8. Personal data breach", level=1)
    d.add_paragraph(
        "Acme notifies Customer without undue delay and in any event within 72 "
        "hours of becoming aware of a personal data breach affecting Customer "
        "Personal Data, with the information Customer needs to meet its own "
        "notification obligations."
    )

    d.add_heading("9. Audit", level=1)
    d.add_paragraph(
        "Acme makes available its most recent SOC 2 Type II report and "
        "certifications on request. Customer may, on 30 days' notice and no more "
        "than once per 12 months, conduct an audit limited to Acme's processing "
        "of Customer Personal Data, subject to confidentiality."
    )

    d.save(OUT / "data_processing_addendum.docx")
    print("wrote data_processing_addendum.docx  (9 sections)")


# ---------------------------------------------------------------------------
# 3. Company metrics — multi-sheet XLSX
# ---------------------------------------------------------------------------
def make_xlsx() -> None:
    import pandas as pd

    headcount = pd.DataFrame({
        "department": ["Engineering", "Sales", "Support", "Product", "G&A", "Marketing"],
        "headcount": [42, 28, 17, 12, 9, 11],
        "open_roles": [6, 3, 4, 1, 0, 1],
        "attrition_ytd": [3, 4, 5, 1, 0, 2],
        "avg_tenure_years": [2.8, 1.9, 1.6, 2.4, 3.4, 1.7],
    })
    budget = pd.DataFrame({
        "department": ["Engineering", "Sales", "Support", "Product", "G&A", "Marketing"],
        "q3_budget_usd": [1_260_000, 940_000, 410_000, 380_000, 320_000, 380_000],
        "q3_actual_usd": [1_205_000, 988_000, 402_000, 366_000, 305_000, 415_000],
    })
    budget["variance_usd"] = budget["q3_actual_usd"] - budget["q3_budget_usd"]
    budget["variance_pct"] = (budget["variance_usd"] / budget["q3_budget_usd"] * 100).round(1)

    monthly = pd.DataFrame({
        "month": ["2026-07", "2026-08", "2026-09"],
        "starting_mrr_usd": [742_000, 771_000, 802_000],
        "new_mrr_usd": [21_000, 22_500, 26_000],
        "expansion_mrr_usd": [14_000, 16_500, 18_000],
        "churned_mrr_usd": [6_000, 8_000, 7_500],
    })
    monthly["ending_mrr_usd"] = (
        monthly["starting_mrr_usd"] + monthly["new_mrr_usd"]
        + monthly["expansion_mrr_usd"] - monthly["churned_mrr_usd"]
    )

    pipeline = pd.DataFrame({
        "stage": ["Discovery", "Evaluation", "Proposal", "Negotiation", "Verbal yes"],
        "deals": [64, 38, 21, 12, 5],
        "value_usd": [3_100_000, 2_050_000, 1_240_000, 760_000, 410_000],
        "win_rate_pct": [8, 18, 34, 55, 85],
    })
    pipeline["weighted_usd"] = (
        pipeline["value_usd"] * pipeline["win_rate_pct"] / 100
    ).round(0).astype(int)

    with pd.ExcelWriter(OUT / "company_metrics.xlsx") as xw:
        headcount.to_excel(xw, sheet_name="headcount", index=False)
        budget.to_excel(xw, sheet_name="budget", index=False)
        monthly.to_excel(xw, sheet_name="monthly_revenue", index=False)
        pipeline.to_excel(xw, sheet_name="pipeline", index=False)
    print("wrote company_metrics.xlsx  (4 sheets)")


# ---------------------------------------------------------------------------
# 4. Q3 board deck — PPTX
# ---------------------------------------------------------------------------
def make_pptx() -> None:
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.util import Inches, Pt

    prs = Presentation()
    title_only = prs.slide_layouts[5]

    def add_chart(slide, chart_type, cats, series):
        data = CategoryChartData()
        data.categories = cats
        for name, values in series:
            data.add_series(name, values)
        chart = slide.shapes.add_chart(
            chart_type, Inches(0.8), Inches(1.4), Inches(8.5), Inches(4.6), data
        ).chart
        plot = chart.plots[0]
        plot.has_data_labels = True
        labels = plot.data_labels
        labels.font.size = Pt(11)
        if chart_type == XL_CHART_TYPE.PIE:
            labels.show_category_name = True
            labels.show_percentage = True
            labels.number_format = "0%"
            labels.number_format_is_linked = False
        else:
            labels.show_value = True
            labels.number_format = "#,##0.##"
            labels.number_format_is_linked = False
        return chart

    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "Acme Cloud — Q3 2026 Board Review"
    s.placeholders[1].text = "Prepared for the board of directors · 2026-10-08"

    s = prs.slides.add_slide(title_only)
    s.shapes.title.text = "Agenda"
    tb = s.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(8), Inches(4)).text_frame
    for item in ["Q3 at a glance", "Revenue by region and segment",
                 "Reliability: the March 9 incident", "Risks", "Q4 priorities"]:
        para = tb.add_paragraph()
        para.text = f"•  {item}"
        para.font.size = Pt(20)

    s = prs.slides.add_slide(title_only)
    s.shapes.title.text = "Q3 at a glance"
    rows = [
        ("Revenue", "USD 2.66M  (+18% QoQ, +41% YoY)"),
        ("Net revenue retention", "106% overall · 112% EMEA"),
        ("New logos", "612  (APAC-led)"),
        ("Gross margin", "79%"),
        ("Support CSAT", "4.1  (below 4.4 floor)"),
        ("API availability", "99.94%  (SLO 99.9%)"),
    ]
    tbl = s.shapes.add_table(len(rows) + 1, 2, Inches(0.7), Inches(1.5),
                             Inches(8.5), Inches(0.4 * (len(rows) + 1))).table
    tbl.cell(0, 0).text = "Metric"
    tbl.cell(0, 1).text = "Q3 2026"
    for i, (k, v) in enumerate(rows, start=1):
        tbl.cell(i, 0).text = k
        tbl.cell(i, 1).text = v

    # clustered column chart — same region figures as q3_sales.csv / the PDF
    s = prs.slides.add_slide(title_only)
    s.shapes.title.text = "Revenue by region"
    add_chart(
        s, XL_CHART_TYPE.COLUMN_CLUSTERED,
        ["North America", "EMEA", "APAC", "LATAM"],
        [("Q3 2026 revenue (USD)", (996_000, 714_000, 540_000, 210_000))],
    )

    # line chart — quarterly revenue trend (USD millions); Q4 2025 and Q1 2026
    # appear only here
    s = prs.slides.add_slide(title_only)
    s.shapes.title.text = "Quarterly revenue trend"
    add_chart(
        s, XL_CHART_TYPE.LINE_MARKERS,
        ["Q3 2025", "Q4 2025", "Q1 2026", "Q2 2026", "Q3 2026"],
        [("Total revenue (USD M)", (1.89, 2.02, 2.16, 2.25, 2.66))],
    )

    # pie chart — revenue share by segment (matches the PDF's segment table)
    s = prs.slides.add_slide(title_only)
    s.shapes.title.text = "Revenue share by segment"
    add_chart(
        s, XL_CHART_TYPE.PIE,
        ["Enterprise", "Team", "Starter"],
        [("Q3 2026 revenue (USD)", (1_276_800, 930_000, 452_200))],
    )

    s = prs.slides.add_slide(title_only)
    s.shapes.title.text = "Reliability: 2026-03-09 EMEA incident"
    tb = s.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(8.5), Inches(4)).text_frame
    for line in [
        "Sev-1, EMEA only, 47 minutes (14:02–14:49 UTC).",
        "Cause: a deploy set the DB connection-pool ceiling to 20 instead of 200.",
        "Detected in 2 min; rolled back via 'make release REF='; recovered by 14:49.",
        "Six action items tracked; schema bounds and EMEA-scale load test are the key two.",
    ]:
        para = tb.add_paragraph()
        para.text = f"•  {line}"
        para.font.size = Pt(18)

    s = prs.slides.add_slide(title_only)
    s.shapes.title.text = "Q4 priorities"
    tb = s.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(8.5), Inches(4)).text_frame
    for line in [
        "Close the LATAM pipeline gap (1.9x → 3.0x) with a partner motion.",
        "Staff support to plan; recover CSAT to 4.4+.",
        "Ship EU data residency GA for Enterprise.",
        "Hold gross margin at 79% or better.",
    ]:
        para = tb.add_paragraph()
        para.text = f"•  {line}"
        para.font.size = Pt(18)

    prs.save(OUT / "q3_board_deck.pptx")
    print(f"wrote q3_board_deck.pptx  ({len(prs.slides)} slides: "
          "column + line + pie charts)")


# ---------------------------------------------------------------------------
# 5. Invoices — PNG (for OCR)
# ---------------------------------------------------------------------------
def _invoice_png(filename: str, number: str, bill_to: str, date: str, due: str,
                 lines: list[tuple[str, int, float]], tax_pct: float) -> None:
    from PIL import Image, ImageDraw

    W, H = 1000, 1300
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    big, mid, sm = _font(40), _font(24), _font(20)

    d.text((60, 50), "Acme Cloud, Inc.", font=big, fill="black")
    d.text((60, 100), "500 Market Street, San Francisco, CA 94105", font=sm, fill="black")
    d.text((60, 128), "billing@acme.example", font=sm, fill="black")

    d.text((60, 210), f"INVOICE  {number}", font=mid, fill="black")
    d.text((60, 250), f"Bill to:  {bill_to}", font=sm, fill="black")
    d.text((60, 280), f"Invoice date:  {date}", font=sm, fill="black")
    d.text((60, 308), f"Due date:  {due}", font=sm, fill="black")
    d.text((60, 336), "Payment terms:  Net 30", font=sm, fill="black")

    y = 410
    d.line((60, y - 12, W - 60, y - 12), fill="black", width=2)
    d.text((60, y), "Description", font=sm, fill="black")
    d.text((640, y), "Qty", font=sm, fill="black")
    d.text((760, y), "Amount (USD)", font=sm, fill="black")
    d.line((60, y + 34, W - 60, y + 34), fill="black", width=1)

    y += 60
    subtotal = 0.0
    for desc, qty, amount in lines:
        d.text((60, y), desc, font=sm, fill="black")
        d.text((640, y), str(qty), font=sm, fill="black")
        d.text((760, y), f"{amount:,.2f}", font=sm, fill="black")
        subtotal += amount
        y += 44

    tax = round(subtotal * tax_pct / 100, 2)
    total = subtotal + tax
    y += 30
    d.line((560, y - 10, W - 60, y - 10), fill="black", width=1)
    d.text((560, y), "Subtotal", font=sm, fill="black")
    d.text((760, y), f"{subtotal:,.2f}", font=sm, fill="black")
    y += 40
    d.text((560, y), f"Tax ({tax_pct:g}%)", font=sm, fill="black")
    d.text((760, y), f"{tax:,.2f}", font=sm, fill="black")
    y += 44
    d.text((560, y), "Total due", font=mid, fill="black")
    d.text((760, y), f"{total:,.2f}", font=mid, fill="black")

    d.text((60, H - 90), "Thank you for your business. Questions: billing@acme.example",
           font=sm, fill="black")
    img.save(OUT / filename)
    print(f"wrote {filename}")


def make_invoices() -> None:
    _invoice_png(
        "invoice_ac_2026_0417.png", "#AC-2026-0417", "Northwind Traders Ltd",
        "2026-07-31", "2026-08-30",
        [("Team plan — annual subscription (25 seats)", 1, 4000.00),
         ("Additional object storage — 500 GB", 1, 600.00)],
        tax_pct=0.0,
    )
    _invoice_png(
        "invoice_ac_2026_0392.png", "#AC-2026-0392", "Globex Corporation",
        "2026-07-15", "2026-08-14",
        [("Enterprise plan — annual subscription", 1, 48000.00),
         ("Premium support add-on", 1, 9600.00),
         ("Professional services — onboarding (40 hrs)", 40, 8000.00)],
        tax_pct=8.5,
    )


if __name__ == "__main__":
    make_pdf()
    make_docx()
    make_xlsx()
    make_pptx()
    make_invoices()
    print("\nAll sample files written to", OUT)
