"""PDF report renderer: the Renderer Protocol and its ReportLab implementation (spec 6.8, ticket 020).

Renderer: ReportLab. docs/spike-results.md does not exist, so the rd.mmd fallback applies.
Code supplies every number, chart and table. The model supplies only text fields. All text is
escaped before layout, so markup inside a label is drawn as literal text.
"""

import re
from functools import partial
from pathlib import Path
from typing import Literal, Protocol

from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import PageBreak

from weekly_analyst.core.config import Settings
from weekly_analyst.core.text import numerals
from weekly_analyst.report.charts import CATEGORICAL_LIGHT, make_chart, validate_palette
from weekly_analyst.report.layout import (
    MARGIN,
    PAGE_SIZES,
    STAMP_W,
    build_document,
    decorate_page,
    findings_section,
    front_section,
    next_checks_section,
    page_count,
    register_fonts,
    risks_section,
    stamp_label,
)
from weekly_analyst.report.model import (
    RenderResult,
    ReportModel,
    ReportTooLong,
    ShownNumber,
    readable_key,
)
from weekly_analyst.report.tables import FONT_BOLD, METHOD_TEXT, appendix_a, appendix_b, appendix_c, next_check_rows

PAGE_CAP = Settings().page_cap

__all__ = ["Renderer", "ReportLabRenderer"]


class Renderer(Protocol):
    """Turns a ReportModel into a PDF at out. Callers depend on this, not on the PDF library."""

    def render(
        self,
        model: ReportModel,
        out: Path,
        *,
        status: Literal["draft", "approved"],
        approver: str | None = None,
        paper: Literal["A4", "Letter"] = "A4",
        approved_at: str | None = None,
    ) -> RenderResult: ...


class ReportLabRenderer(Renderer):
    """Fixed ReportLab template: front page, findings, risks, next checks and appendices A to C."""

    def render(
        self,
        model: ReportModel,
        out: Path,
        *,
        status: Literal["draft", "approved"],
        approver: str | None = None,
        paper: Literal["A4", "Letter"] = "A4",
        approved_at: str | None = None,
    ) -> RenderResult:
        register_fonts()
        stamp = stamp_label(status, approver, approved_at)  # computed once, so layout, check and exemptions agree
        _check(model, status, approver, stamp)
        _mark_deltas(model)
        width = PAGE_SIZES[paper][0] - 2 * MARGIN
        on_page = partial(decorate_page, model=model, paper=paper, stamp=stamp)
        if page_count(build_document(front_section(model, width), paper, on_page, model.report_name)) > 1:
            raise ValueError("executive summary does not fit page 1")
        story = (
            front_section(model, width)
            + [PageBreak()]
            + findings_section(model, width)
            + risks_section(model)
            + next_checks_section(model, width)
            + appendix_a(model, width)
            + appendix_b(model, width)
            + appendix_c()
        )
        pdf = build_document(story, paper, on_page, model.report_name)
        pages = page_count(pdf)
        if pages > PAGE_CAP:
            raise ReportTooLong(f"report is {pages} pages, over the cap of {PAGE_CAP}")
        out.write_bytes(pdf)
        warnings = []
        if model.findings and not validate_palette(list(CATEGORICAL_LIGHT)):
            warnings.append("palette not validated: validator failed or node is missing")
        shown = _shown_numbers(model)
        structural = _structural_numbers(model, pages, stamp, {s.text for s in shown})
        return RenderResult(pages=pages, shown_numbers=shown, warnings=warnings, structural_numbers=structural)


def _check(model: ReportModel, status: str, approver: str | None, stamp: str) -> None:
    if status == "approved" and not approver:
        raise ValueError("an approved report needs an approver")
    if pdfmetrics.stringWidth(stamp, FONT_BOLD, 8.5) > STAMP_W - 16:
        raise ValueError("stamp text is too long for the stamp box")
    if sorted(t.item_key for t in model.data_tables) != sorted(f.item_key for f in model.findings):
        raise ValueError("data tables must match findings one to one: every chart needs its data table")


def _mark_deltas(model: ReportModel) -> None:
    """Every displayed change is a signed delta from the prior period, so its check recomputes both periods."""
    for k in model.kpis:
        k.vs_prior.prev_period = model.prior_period
    for f in model.findings:
        for r in f.attribution:
            r.change.prev_period = model.prior_period


def _shown_numbers(model: ReportModel) -> list[ShownNumber]:
    shown = []
    for k in model.kpis:
        shown += [k.value, k.vs_prior] + ([k.vs_baseline] if k.vs_baseline else [])
    for f in model.findings:
        shown.append(f.supporting)
        shown += [r.change for r in f.attribution]
    for t in model.data_tables:
        shown += [c for row in t.rows for c in row if isinstance(c, ShownNumber)]
    return shown


def _svg_text(svg: bytes) -> list[str]:
    """Text nodes of a chart SVG: tick labels, title and description."""
    return re.findall(r">([^<>]+)<", svg.decode("utf-8"))


def _structural_text(model: ReportModel, pages: int, stamp: str) -> list[str]:
    """Text the renderer writes itself, apart from shown numbers: labels, periods, dates, indices, page
    numbers, run and hash, chart text.

    Headline, recommended action and so-whats are model text and are left out, so check_report judges
    their numerals against shown numbers only. risks_caveats mixes model caveats with code-written risks,
    so it is taken whole.
    """
    text = [
        model.report_name,
        model.reporting_period,
        model.data_as_of,
        model.prepared_for,
        model.report_version,
        model.run_id,
        model.snapshot_date,
        model.snapshot_hash,
        model.snapshot_hash[:16],
        stamp,
        *METHOD_TEXT,
        *model.risks_caveats,
        *(cell for row in next_check_rows(model.next_checks) for cell in row),
        *(f"Page {n}" for n in range(1, pages + 1)),
        *(k.label for k in model.kpis),
    ]
    for n, f in enumerate(model.findings, 1):
        text += [
            f"Finding {n}: {readable_key(f.item_key)}",
            readable_key(f.item_key),
            f.item_key,
            f.interpretation,
            f.chart.title,
            f.chart.alt_text,
            f"Chart {n} data: {f.chart.title}",
            *(r.segment for r in f.attribution),
            *_svg_text(make_chart(f.chart)),
        ]
    for t in model.data_tables:
        text += [*t.columns, *(c for row in t.rows for c in row if not isinstance(c, ShownNumber))]
    for m in model.metric_definitions:
        text += [
            m.name,
            m.definition,
            m.owner,
            f"Includes: {m.includes}. Excludes: {m.excludes}.",
            f"Version {m.version}. Checksum {m.checksum}.",
        ]
    return text


def _structural_numbers(model: ReportModel, pages: int, stamp: str, shown_texts: set[str]) -> list[str]:
    found = (token for text in _structural_text(model, pages, stamp) for token in numerals(text))
    return list(dict.fromkeys(token for token in found if token not in shown_texts))
