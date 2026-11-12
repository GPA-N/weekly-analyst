"""PDF report render (spec 6.8, ticket 020). Written before report.py."""

import re
import socket

import pikepdf
import pytest
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTChar
from pypdf import PdfReader

from weekly_analyst.report.charts import ChartSpec
from weekly_analyst.core.pdfscan import scan_pdf
from weekly_analyst.report import render_pdf
from weekly_analyst.report.layout import stamp_box
from weekly_analyst.report.model import (
    AttributionRow,
    DataTable,
    Finding,
    Kpi,
    MetricDefinition,
    ReportModel,
    ReportTooLong,
    ShownNumber,
)
from weekly_analyst.report.tables import next_check_rows

HOSTILE = [
    '=HYPERLINK("http://canary")',
    "{{x}}",
    "<script>alert(1)</script>",
    r"C:\Users\data\n\\x",
]


def _num(text, metric="orders", segment=None, period="2026-W45"):
    return ShownNumber(text=text, metric=metric, segment=segment, period=period)


def _chart(title, alt):
    return ChartSpec(
        kind="line",
        title=title,
        x=["W43", "W44", "W45"],
        series={"North": [10.0, 12.0, 15.0], "South": [8.0, 9.0, 7.5]},
        unit="Orders",
        alt_text=alt,
    )


def _finding(key, so_what, segments):
    return Finding(
        item_key=key,
        supporting=_num("+3,000"),
        so_what=so_what,
        chart=_chart(f"Orders by week: {key}", f"Two lines of weekly orders for {key}."),
        attribution=[AttributionRow(seg, _num(delta)) for seg, delta in segments],
        interpretation=f"The change in {key} sits mostly in the first segment.",
    )


def _tables(findings):
    return [
        DataTable(
            item_key=f.item_key,
            columns=["Week", "North", "South"],
            rows=[["W43", _num("10"), _num("8")], ["W44", _num("12"), _num("9")], ["W45", _num("15"), _num("7.5")]],
        )
        for f in findings
    ]


def _model(**overrides) -> ReportModel:
    findings = [
        _finding(
            "orders_north",
            "North orders rose sharply last week.",
            [("North", "+3,000"), (HOSTILE[0], "-500")],
        ),
        _finding(
            "orders_south",
            "South orders fell slightly against the baseline.",
            [("South", "-400"), (HOSTILE[2], "+50")],
        ),
    ]
    base = dict(
        report_name="Weekly orders analysis",
        reporting_period="2026-W44 to 2026-W45",
        prior_period="2026-W44",
        data_as_of="2026-11-08",
        prepared_for="Operations lead",
        report_version="1.0",
        run_id="run-0001",
        snapshot_date="2026-11-08",
        snapshot_hash="sha256:" + "ab12" * 16,
        headline="Orders grew in North last week.",
        recommended_action="Check the North channel before Monday's review.",
        kpis=[
            Kpi("Orders", _num("15,000"), _num("+25%"), _num("+10%")),
            Kpi("Returns", _num("120"), _num("-4%"), None),
        ],
        findings=findings,
        risks_caveats=["Week 45 is a short week.", HOSTILE[1], HOSTILE[3]],
        next_checks=["Confirm the North feed is complete.", HOSTILE[0]],
        metric_definitions=[
            MetricDefinition(
                name=HOSTILE[2],
                definition="Count of paid order rows.",
                owner="Sales analytics",
                includes="all paid orders",
                excludes="test orders",
                version="2",
                checksum="sha256:0f",
            )
        ],
        data_tables=_tables(findings),
    )
    base.update(overrides)
    return ReportModel(**base)


def _render(tmp_path, model=None, status="draft", approver=None, paper="A4", name="r.pdf"):
    out = tmp_path / name
    result = render_pdf(model or _model(), out, status=status, approver=approver, paper=paper)
    return out, result


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _chars_outside_stamp(path, paper):
    x0, y0, x1, y1 = stamp_box(paper)
    kept = []
    for page in extract_pages(str(path)):
        stack = [page]
        while stack:
            node = stack.pop()
            if isinstance(node, LTChar):
                inside = x0 <= node.x0 and node.x1 <= x1 and y0 <= node.y0 and node.y1 <= y1
                if not inside:
                    kept.append((node.get_text(), round(node.x0, 2), round(node.y0, 2)))
            elif hasattr(node, "__iter__"):
                stack.extend(node)
    return sorted(kept)


def test_eight_page_cap_raises_report_too_long(tmp_path):
    model = _model(next_checks=[f"Check item {i} for the long list." for i in range(400)])
    with pytest.raises(ReportTooLong):
        render_pdf(model, tmp_path / "long.pdf", status="draft")
    assert not (tmp_path / "long.pdf").exists()


def test_exec_summary_on_page_1_and_findings_after(tmp_path):
    model = _model()
    out, result = _render(tmp_path, model)
    assert 2 <= result.pages <= 8
    page1 = _squash(extract_text(str(out), page_numbers=[0]))
    for text in [model.headline, model.recommended_action, model.kpis[0].label, model.findings[0].so_what,
                 model.findings[1].so_what]:
        assert _squash(text) in page1
    assert _squash("Appendix A.") not in page1
    assert _squash("Finding 1:") in _squash(extract_text(str(out), page_numbers=[1]))


def test_exec_summary_that_overflows_page_1_is_rejected(tmp_path):
    findings = [_finding(f"item_{i}", "x " * 120, [("Seg", "+1")]) for i in range(30)]
    model = _model(findings=findings, data_tables=_tables(findings))
    with pytest.raises(ValueError, match="page 1"):
        render_pdf(model, tmp_path / "overflow.pdf", status="draft")


def test_fonts_embedded_and_text_extractable_by_pypdf_and_pdfminer(tmp_path):
    model = _model()
    out, _ = _render(tmp_path, model)
    pdf = pikepdf.open(out)
    fonts = [o for o in pdf.objects if isinstance(o, pikepdf.Dictionary) and str(o.get("/Type")) == "/Font"]
    assert fonts

    def embedded(font):
        desc = font.get("/FontDescriptor")
        if desc is None and "/DescendantFonts" in font:
            desc = font.DescendantFonts[0].get("/FontDescriptor")
        return desc is not None and any(k in desc for k in ("/FontFile", "/FontFile2", "/FontFile3"))

    assert all(embedded(f) for f in fonts), [str(f.get("/BaseFont")) for f in fonts if not embedded(f)]
    pypdf_text = "".join(page.extract_text() for page in PdfReader(out).pages)
    assert _squash(model.headline) in _squash(pypdf_text)
    assert _squash(model.headline) in _squash(extract_text(str(out)))


def test_hostile_labels_are_inert_literal_text(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network attempted during render")

    monkeypatch.setattr(socket, "socket", no_network)
    out, _ = _render(tmp_path)
    text = _squash(extract_text(str(out)))
    for label in HOSTILE:
        assert _squash(label) in text
    assert scan_pdf(out) == []


def test_stamp_moves_no_text_outside_its_box(tmp_path):
    draft, draft_result = _render(tmp_path, status="draft", name="draft.pdf")
    approved, approved_result = _render(
        tmp_path, status="approved", approver="Named Reviewer", name="approved.pdf"
    )
    assert draft_result.pages == approved_result.pages
    assert _chars_outside_stamp(draft, "A4") == _chars_outside_stamp(approved, "A4")
    for page in range(draft_result.pages):
        assert _squash("Draft, pending approval") in _squash(extract_text(str(draft), page_numbers=[page]))
    assert _squash("Approved by Named Reviewer") in _squash(extract_text(str(approved)))


def test_approved_report_without_approver_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        render_pdf(_model(), tmp_path / "x.pdf", status="approved")


def test_owner_and_due_date_columns_are_blank(tmp_path):
    rows = next_check_rows(["First check.", "Second check."])
    assert rows[0] == ["#", "Check", "Owner", "Due date"]
    assert [r[1] for r in rows[1:]] == ["First check.", "Second check."]
    assert [r[2] for r in rows[1:]] == ["", ""]
    assert [r[3] for r in rows[1:]] == ["", ""]
    out, _ = _render(tmp_path)
    text = _squash(extract_text(str(out)))
    assert _squash("Owner") in text and _squash("Due date") in text


def test_every_chart_has_alt_text_and_data_table_in_appendix_b(tmp_path):
    model = _model()
    out, _ = _render(tmp_path, model)
    text = _squash(extract_text(str(out)))
    appendix_b = text[text.index(_squash("Appendix B.")) : text.index(_squash("Appendix C."))]
    for finding in model.findings:
        assert _squash(finding.chart.alt_text) in text
        assert _squash(finding.chart.title) in appendix_b
    for table in model.data_tables:
        for row in table.rows:
            for cell in row:
                value = cell.text if isinstance(cell, ShownNumber) else cell
                assert _squash(value) in appendix_b


def test_chart_without_alt_text_is_rejected(tmp_path):
    model = _model()
    model.findings[0].chart = _finding("orders_north", "x", [("North", "+1")]).chart.model_copy(
        update={"alt_text": "   "}
    )
    with pytest.raises(ValueError):
        render_pdf(model, tmp_path / "x.pdf", status="draft")


def test_finding_without_data_table_is_rejected(tmp_path):
    model = _model(data_tables=_model().data_tables[1:])
    with pytest.raises(ValueError):
        render_pdf(model, tmp_path / "x.pdf", status="draft")


def test_clean_report_scans_empty_metadata_set_and_charts_vector(tmp_path):
    out, _ = _render(tmp_path)
    assert scan_pdf(out) == []
    with pikepdf.open(out) as pdf:
        assert str(pdf.docinfo["/Title"]) == "Weekly orders analysis"
        assert str(pdf.docinfo["/Author"]) == "weekly-analyst"
        images = [o for o in pdf.objects if isinstance(o, pikepdf.Stream) and str(o.get("/Subtype")) == "/Image"]
    assert images == []


def test_letter_paper_page_size(tmp_path):
    out, _ = _render(tmp_path, paper="Letter")
    box = PdfReader(out).pages[0].mediabox
    assert round(float(box.width)) == 612 and round(float(box.height)) == 792


def test_delta_numbers_carry_prior_period_and_levels_do_not(tmp_path):
    model = _model()
    _, result = _render(tmp_path, model)
    deltas = {id(k.vs_prior) for k in model.kpis} | {id(r.change) for f in model.findings for r in f.attribution}
    assert deltas
    for shown in result.shown_numbers:
        assert shown.prev_period == ("2026-W44" if id(shown) in deltas else None)


def test_structural_numbers_are_renderer_labels_not_shown_or_model_text(tmp_path):
    _, result = _render(tmp_path, _model(headline="Orders grew 99% in North."))
    structural = set(result.structural_numbers)
    # ISO period and date parts, run id, report version, page and finding indices.
    assert {"2026", "11", "08", "0001", "1"} <= structural
    assert not structural & {s.text for s in result.shown_numbers}
    assert "99" not in structural  # model text is held to the model_text rule, not exempted
