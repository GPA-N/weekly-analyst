"""Page layout: geometry, the stamp box, fonts, page decoration, document build and the section builders (spec 6.8).

Sections are the front page, findings, risks and next checks. Tables and text primitives come from tables.py.
"""

import io
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import pikepdf
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph, Table
from svglib import fonts as svg_fonts
from svglib.svglib import svg2rlg

from weekly_analyst.report.charts import TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY, ChartSpec, make_chart
from weekly_analyst.report.model import ReportModel, readable_key
from weekly_analyst.report.tables import (
    FONT,
    FONT_BOLD,
    KV_STYLE,
    STYLES,
    _p,
    _section,
    attribution_table,
    kpi_strip,
    next_checks_table,
)

PAGE_SIZES = {"A4": A4, "Letter": LETTER}
AUTHOR = "weekly-analyst"
MARGIN = 50
TOP = 90  # body frame top sits below the header band
BOTTOM = 70  # body frame bottom sits above the footer band
STAMP_W = 240  # wide enough for "Approved by <name> <UTC time>" at the stamp font size
STAMP_H = 26
STAMP_GAP = 20  # stamp box top sits this far below the page top edge


def stamp_box(paper: str) -> tuple[float, float, float, float]:
    """Stamp box in PDF points (bottom-left origin). Same place on every page, whatever the status."""
    width, height = PAGE_SIZES[paper]
    return (width - MARGIN - STAMP_W, height - STAMP_GAP - STAMP_H, width - MARGIN, height - STAMP_GAP)


def stamp_label(status: str, approver: str | None, approved_at: str | None = None) -> str:
    if status == "draft":
        return "Draft, pending approval"
    return f"Approved by {approver}" + (f" {approved_at}" if approved_at else "")


def register_fonts() -> None:
    """DejaVu ships with matplotlib. Fonts are local and embedded, never substituted."""
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont(FONT, str(ttf / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(ttf / "DejaVuSans-Bold.ttf")))
    # Chart text names "DejaVu Sans" in its font-family list. Map it to the embedded font.
    svg_fonts.register_font("DejaVu Sans", str(ttf / "DejaVuSans.ttf"), rlgFontName=FONT)


def _chart(spec: ChartSpec):
    drawing = svg2rlg(io.BytesIO(make_chart(spec)))
    if drawing is None:
        raise ValueError(f"chart {spec.title!r} did not convert to a vector drawing")
    drawing.initialFontName = FONT  # else the drawing opens in Times-Roman, not embedded
    return drawing


def front_section(model: ReportModel, width: float) -> list:
    """Title block, executive summary and KPI strip. Must fit page 1."""
    title_rows = [
        ("Reporting period", model.reporting_period),
        ("Data as of", model.data_as_of),
        ("Prepared for", model.prepared_for),
        ("Report version", model.report_version),
    ]
    story = [
        _p(model.report_name, "title"),
        Table([[_p(k, "small"), _p(v, "body")] for k, v in title_rows], colWidths=[120, width - 120], style=KV_STYLE),
        _p("Executive summary", "h1"),
        _p(model.headline, "bold"),
        Paragraph(f"<b>Recommended action.</b> {escape(model.recommended_action)}", STYLES["body"]),
    ]
    if model.kpis:
        story.append(kpi_strip(model, width))
    story.append(_p("Key findings", "h1"))
    for f in model.findings:
        story.append(_p(f"{readable_key(f.item_key)} ({f.supporting.text}): {f.so_what}", "body"))
    return story


def findings_section(model: ReportModel, width: float) -> list:
    blocks = []
    for n, f in enumerate(model.findings, 1):
        blocks.append(
            KeepTogether(
                [
                    _p(f"Finding {n}: {readable_key(f.item_key)}", "h2"),
                    _chart(f.chart),
                    _p(f"Chart description: {f.chart.alt_text}", "small"),
                    attribution_table(f, width),
                    _p("Interpretation", "label"),
                    _p(f.interpretation, "body"),
                ]
            )
        )
    return _section(_p("Findings", "h1"), blocks)


def risks_section(model: ReportModel) -> list:
    story = [_p("Risks, caveats and data quality", "h1")]
    if not model.risks_caveats:
        story.append(_p("None recorded.", "body"))
    story += [Paragraph(escape(c), STYLES["body"], bulletText="•") for c in model.risks_caveats]
    return story


def next_checks_section(model: ReportModel, width: float) -> list:
    return [
        _p("Suggested next checks", "h1"),
        _p("Owner and due date are left blank for a person to fill in.", "small"),
        next_checks_table(model, width),
    ]


def decorate_page(canv, doc, *, model: ReportModel, paper: str, stamp: str) -> None:
    """Header, footer, page number and the stamp. The stamp box is drawn the same way for both statuses."""
    width, height = PAGE_SIZES[paper]
    x0, y0, x1, y1 = stamp_box(paper)
    canv.saveState()
    canv.setFont(FONT, 8)
    canv.setFillColor(HexColor(TEXT_SECONDARY))
    canv.drawString(MARGIN, height - 36, model.report_name)
    canv.setStrokeColor(HexColor(TEXT_SECONDARY))
    canv.setLineWidth(0.8)
    canv.rect(x0, y0, x1 - x0, y1 - y0, stroke=1, fill=0)
    canv.setFont(FONT_BOLD, 8.5)
    canv.setFillColor(HexColor(TEXT_PRIMARY))
    canv.drawString(x0 + 8, y0 + 9, stamp)
    canv.setFont(FONT, 7)
    canv.setFillColor(HexColor(TEXT_MUTED))
    canv.drawString(MARGIN, 40, f"Run {model.run_id}   Snapshot {model.snapshot_date}   Hash {model.snapshot_hash[:16]}")
    canv.drawRightString(width - MARGIN, 40, f"Page {canv.getPageNumber()}")
    canv.restoreState()


def build_document(story: list, paper: str, on_page, title: str) -> bytes:
    width, height = PAGE_SIZES[paper]
    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf,
        pagesize=(width, height),
        title=title,
        author=AUTHOR,
        subject="Weekly analysis",
        creator=AUTHOR,
        # Without these the canvas starts in Helvetica, a non-embedded standard font.
        initialFontName=FONT,
        initialFontSize=9.5,
        initialLeading=13,
        invariant=1,
    )
    frame = Frame(MARGIN, BOTTOM, width - 2 * MARGIN, height - TOP - BOTTOM, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="body", frames=[frame], onPage=on_page)])
    doc.build(story)
    return buf.getvalue()


def page_count(pdf: bytes) -> int:
    return len(pikepdf.open(io.BytesIO(pdf)).pages)
