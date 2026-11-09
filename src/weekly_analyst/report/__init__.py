"""Report: charts and the PDF renderer."""

from pathlib import Path
from typing import Literal

from weekly_analyst.report.model import RenderResult, ReportModel
from weekly_analyst.report.renderer import ReportLabRenderer


def render_pdf(
    model: ReportModel,
    out: Path,
    *,
    status: Literal["draft", "approved"],
    approver: str | None = None,
    paper: Literal["A4", "Letter"] = "A4",
    approved_at: str | None = None,
) -> RenderResult:
    """Callers keep this name. The renderer behind it can change without touching them.

    approved_at is the UTC time printed on an approved stamp, for example "2026-11-09 12:00 UTC".
    """
    return ReportLabRenderer().render(
        model, out, status=status, approver=approver, paper=paper, approved_at=approved_at
    )
