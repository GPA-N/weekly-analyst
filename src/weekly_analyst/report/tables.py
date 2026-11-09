"""Text and table primitives, and the table blocks: KPI strip, attribution, next checks and appendices (spec 6.8).

Layout imports this module, so it imports nothing from layout.
"""

from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import KeepTogether, Paragraph, Table

from weekly_analyst.report.charts import BASELINE, GRIDLINE, TEXT_PRIMARY, TEXT_SECONDARY
from weekly_analyst.report.model import Finding, ReportModel, ShownNumber

FONT = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"
METHOD_TEXT = (
    "Code computes every number in this report from the source snapshot. The language model wrote only the text fields.",
    "Each shown number is checked against an independent recompute before the report is issued.",
    "Each chart has a written description and its data table in Appendix B.",
)


def _style(name: str, **kwargs) -> ParagraphStyle:
    base = dict(fontName=FONT, fontSize=9.5, leading=13, textColor=HexColor(TEXT_PRIMARY), bulletFontName=FONT)
    return ParagraphStyle(name, **{**base, **kwargs})


STYLES = {
    "title": _style("title", fontName=FONT_BOLD, fontSize=18, leading=22, spaceAfter=8),
    "h1": _style("h1", fontName=FONT_BOLD, fontSize=12, leading=16, spaceBefore=10, spaceAfter=4),
    "h2": _style("h2", fontName=FONT_BOLD, fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3),
    "body": _style("body"),
    "bold": _style("bold", fontName=FONT_BOLD),
    "label": _style("label", fontName=FONT_BOLD, spaceBefore=6),
    "small": _style("small", fontSize=7.5, leading=9.5, textColor=HexColor(TEXT_SECONDARY)),
    "cell": _style("cell", fontSize=8.5, leading=11),
    "num": _style("num", fontSize=8.5, leading=11, alignment=TA_RIGHT),
    "kpi": _style("kpi", fontName=FONT_BOLD, fontSize=14, leading=18),
}
# FONTNAME on every table: without it, cells default to Helvetica (not embedded).
KV_STYLE = [
    ("FONTNAME", (0, 0), (-1, -1), FONT),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ("LINEBELOW", (0, -1), (-1, -1), 0.6, HexColor(BASELINE)),
]
GRID_STYLE = [
    ("FONTNAME", (0, 0), (-1, -1), FONT),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LINEBELOW", (0, 0), (-1, 0), 0.6, HexColor(BASELINE)),
    ("LINEBELOW", (0, 1), (-1, -1), 0.3, HexColor(GRIDLINE)),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]
KPI_STYLE = [
    ("FONTNAME", (0, 0), (-1, -1), FONT),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("BOX", (0, 0), (-1, -1), 0.6, HexColor(BASELINE)),
    ("INNERGRID", (0, 0), (-1, -1), 0.3, HexColor(GRIDLINE)),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]


def _p(text: str, style: str) -> Paragraph:
    """Escaped paragraph. Labels and model text are never parsed as markup."""
    return Paragraph(escape(text), STYLES[style])


def _cell(value: ShownNumber | str) -> Paragraph:
    if isinstance(value, ShownNumber):
        return _p(value.text, "num")
    return _p(value, "cell")


def _section(heading: Paragraph, blocks: list) -> list:
    """Keep a section heading on the same page as its first block."""
    if not blocks:
        return [heading]
    return [KeepTogether([heading, blocks[0]])] + blocks[1:]


def next_check_rows(checks: list[str]) -> list[list[str]]:
    """Owner and due date stay blank. The agent never invents them (spec 6.8)."""
    return [["#", "Check", "Owner", "Due date"]] + [[str(i), c, "", ""] for i, c in enumerate(checks, 1)]


def kpi_strip(model: ReportModel, width: float) -> Table:
    cols = [
        [
            _p(k.label, "small"),
            _p(k.value.text, "kpi"),
            _p(f"vs prior: {k.vs_prior.text}", "small"),
            _p(f"vs baseline: {k.vs_baseline.text if k.vs_baseline else 'no baseline'}", "small"),
        ]
        for k in model.kpis
    ]
    return Table([list(row) for row in zip(*cols)], colWidths=[width / len(cols)] * len(cols), style=KPI_STYLE)


def attribution_table(f: Finding, width: float) -> Table:
    """Signed change of each segment against the prior period, under one finding's chart."""
    rows = [[_p("Segment", "cell"), _p("Change", "num")]] + [[_p(r.segment, "cell"), _cell(r.change)] for r in f.attribution]
    return Table(rows, colWidths=[width - 120, 120], style=GRID_STYLE)


def next_checks_table(model: ReportModel, width: float) -> Table:
    rows = [[_p(cell, "cell") for cell in row] for row in next_check_rows(model.next_checks)]
    return Table(rows, colWidths=[30, width - 210, 100, 80], style=GRID_STYLE)


def appendix_a(model: ReportModel, width: float) -> list:
    header = ["Metric", "Definition", "Owner", "Includes and excludes", "Version and checksum"]
    rows = [[_p(h, "cell") for h in header]]
    for m in model.metric_definitions:
        rows.append(
            [
                _p(m.name, "cell"),
                _p(m.definition, "cell"),
                _p(m.owner, "cell"),
                _p(f"Includes: {m.includes}. Excludes: {m.excludes}.", "cell"),
                _p(f"Version {m.version}. Checksum {m.checksum}.", "cell"),
            ]
        )
    return [
        _p("Appendix A. Metric definitions", "h1"),
        Table(rows, colWidths=[90, 130, 80, 110, width - 410], style=GRID_STYLE),
    ]


def appendix_b(model: ReportModel, width: float) -> list:
    tables = {t.item_key: t for t in model.data_tables}
    blocks = []
    for n, f in enumerate(model.findings, 1):
        table = tables[f.item_key]
        cols = len(table.columns)
        rows = [[_p(c, "cell") for c in table.columns]] + [[_cell(v) for v in row] for row in table.rows]
        blocks.append(
            KeepTogether(
                [_p(f"Chart {n} data: {f.chart.title}", "h2"), Table(rows, colWidths=[width / cols] * cols, style=GRID_STYLE)]
            )
        )
    return _section(_p("Appendix B. Data tables", "h1"), blocks)


def appendix_c() -> list:
    return [_p("Appendix C. Method and checks", "h1")] + [_p(t, "body") for t in METHOD_TEXT]
