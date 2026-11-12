"""Deterministic report checks (ticket 017, spec 6.3)."""

from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pypdf import PdfReader

from weekly_analyst.report import model as report, render_pdf
from weekly_analyst.report.charts import ChartSpec
from weekly_analyst.analysis.checks import ShownNumber, check_report
from weekly_analyst.data.metrics_file import load_metrics_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "metrics_sales.yaml"
METRICS = load_metrics_file(FIXTURE)

# 80 rows in ISO week 2026-W02, all dated inside it. Revenue sums to 1590.0.
# North has 40 rows, 14 with returned = 1, so its return rate is 0.35.
# East has 5 rows, below the 30-row gate, so it cannot be recomputed.


def _write_snapshot(tmp_path: Path) -> Path:
    start = date(2026, 1, 5)
    regions = ["north"] * 40 + ["south"] * 35 + ["east"] * 5
    rows = [
        (start + timedelta(days=i % 7), region, round(10 + i * 0.25, 2), int(i % 3 == 0))
        for i, region in enumerate(regions)
    ]
    table = pa.table(
        {
            "date": pa.array([r[0] for r in rows], type=pa.date32()),
            "region": [r[1] for r in rows],
            "revenue": [r[2] for r in rows],
            "returned": [r[3] for r in rows],
        }
    )
    path = tmp_path / "snapshot.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture
def snapshot(tmp_path):
    return _write_snapshot(tmp_path)


def _shown(text, metric="revenue", segment=None, period="2026-W02"):
    return ShownNumber(text=text, metric=metric, segment=segment, period=period)


def _kinds(failures):
    return [f.kind for f in failures]


def test_clean_draft_returns_no_failures(snapshot):
    pdf_text = "Revenue reached 1,590. North return rate was 35.0%."
    shown = [_shown("1,590"), _shown("35.0%", "return_rate", {"region": "north"})]
    assert check_report(pdf_text, shown, METRICS, snapshot) == []


def test_wrong_number_rejected(snapshot):
    failures = check_report("Revenue reached 1,500.", [_shown("1,500")], METRICS, snapshot)
    assert _kinds(failures) == ["wrong_number"]
    assert "1,500" in failures[0].detail


@pytest.mark.parametrize("metric", ["", "profit"])
def test_claim_without_known_metric_rejected(snapshot, metric):
    failures = check_report("Revenue reached 1,590.", [_shown("1,590", metric=metric)], METRICS, snapshot)
    assert _kinds(failures) == ["no_metric"]


def test_causal_overstatement_rejected(snapshot):
    pdf_text = "Revenue reached 1,590 because of the spring promotion."
    failures = check_report(pdf_text, [_shown("1,590")], METRICS, snapshot)
    assert _kinds(failures) == ["wording"]
    assert "because of" in failures[0].detail


def test_number_in_pdf_not_in_shown_numbers_rejected(snapshot):
    failures = check_report("Revenue reached 1,590 across 12 orders.", [_shown("1,590")], METRICS, snapshot)
    assert _kinds(failures) == ["unlisted_number"]
    assert failures[0].detail == "12"


def test_number_below_min_rows_gate_fails_closed(snapshot):
    failures = check_report(
        "East revenue was 1,590.",
        [_shown("1,590", segment={"region": "east"})],
        METRICS,
        snapshot,
    )
    assert _kinds(failures) == ["wrong_number"]


def test_unparseable_shown_text_is_wrong_number_not_a_crash(snapshot):
    failures = check_report("", [_shown("about 1,590")], METRICS, snapshot)
    assert _kinds(failures) == ["wrong_number"]


def _write_two_week_snapshot(tmp_path: Path) -> Path:
    # 2026-W01: 40 rows of revenue 10.0, sum 400.0. 2026-W02: the rows from _write_snapshot, sum 1590.0.
    prior = pa.table(
        {
            "date": pa.array([date(2025, 12, 29) + timedelta(days=i % 7) for i in range(40)], type=pa.date32()),
            "region": ["north"] * 40,
            "revenue": [10.0] * 40,
            "returned": [0] * 40,
        }
    )
    current = pq.read_table(_write_snapshot(tmp_path))
    path = tmp_path / "two_weeks.parquet"
    pq.write_table(pa.concat_tables([prior.cast(current.schema), current]), path)
    return path


@pytest.fixture
def two_weeks(tmp_path):
    return _write_two_week_snapshot(tmp_path)


def _delta(text, metric="revenue", segment=None, period="2026-W02", prev_period="2026-W01"):
    return ShownNumber(text=text, metric=metric, segment=segment, period=period, prev_period=prev_period)


def test_correct_signed_delta_passes(two_weeks):
    # 1,590.0 in 2026-W02 less 400.0 in 2026-W01 is +1,190.0.
    assert check_report("Revenue rose +1,190.", [_delta("+1,190")], METRICS, two_weeks) == []


def test_wrong_delta_rejected(two_weeks):
    # The level 1,590 is shown as the change. The signed delta is +1,190.
    failures = check_report("Revenue rose +1,590.", [_delta("+1,590")], METRICS, two_weeks)
    assert _kinds(failures) == ["wrong_number"]
    assert "+1,590" in failures[0].detail


def test_delta_without_prior_data_rejected(snapshot):
    # The snapshot has no 2026-W01 rows, so the prior level is None and the delta cannot be confirmed.
    failures = check_report("Revenue rose +1,190.", [_delta("+1,190")], METRICS, snapshot)
    assert _kinds(failures) == ["wrong_number"]


def _clean_render_model() -> report.ReportModel:
    """A report whose every shown number recomputes from the two_weeks snapshot."""

    def num(text, segment=None, period="2026-W02"):
        return report.ShownNumber(text=text, metric="revenue", segment=segment, period=period)

    north = {"region": "north"}
    finding = report.Finding(
        item_key="revenue_north",
        supporting=num("595.00", north),
        so_what="North added +195.00 this week.",
        chart=ChartSpec(
            kind="line",
            title="Revenue by ISO week: north",
            x=["W01", "W02"],
            series={"value": [400.0, 595.0]},
            unit="Revenue",
            alt_text="Line of revenue by ISO week for north; the latest week is last.",
        ),
        attribution=[report.AttributionRow("north", num("+195.00", north))],
        interpretation="Each row is one region segment and its share of the change in revenue.",
    )
    table = report.DataTable(
        item_key="revenue_north",
        columns=["Week", "Revenue"],
        rows=[["W01", num("400.00", north, "2026-W01")], ["W02", num("595.00", north)]],
    )
    return report.ReportModel(
        report_name="Weekly sales analysis",
        reporting_period="2026-W01 to 2026-W02",
        prior_period="2026-W01",
        data_as_of="2026-01-11",
        prepared_for="Operations lead",
        report_version="1",
        run_id="run-0001",
        snapshot_date="2026-01-11",
        snapshot_hash="sha256:" + "ab12" * 16,
        headline="Revenue rose in the latest week.",
        recommended_action="Check the north region before the review.",
        kpis=[report.Kpi("revenue", num("1,590.00"), num("+1,190.00"))],
        findings=[finding],
        risks_caveats=[],
        next_checks=["Check the source rows behind revenue_north"],
        metric_definitions=[],
        data_tables=[table],
    )


def test_rendered_clean_report_passes_with_zero_failures(tmp_path, two_weeks):
    model = _clean_render_model()
    out = tmp_path / "clean.pdf"
    rendered = render_pdf(model, out, status="draft")
    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(out)).pages)
    model_text = "\n".join(
        [model.headline, model.recommended_action, *(f.so_what for f in model.findings), *model.risks_caveats]
    )
    failures = check_report(
        pdf_text,
        rendered.shown_numbers,
        METRICS,
        two_weeks,
        exempt=rendered.structural_numbers,
        model_text=model_text,
    )
    assert failures == []


def test_structural_numeral_in_body_is_not_flagged(snapshot):
    failures = check_report("Revenue reached 1,590 in 2026.", [_shown("1,590")], METRICS, snapshot, exempt=["2026"])
    assert failures == []


def test_invented_body_numeral_still_fails_with_exempt(snapshot):
    pdf_text = "Revenue reached 1,590 across 12 orders in 2026."
    failures = check_report(pdf_text, [_shown("1,590")], METRICS, snapshot, exempt=["2026"])
    assert _kinds(failures) == ["unlisted_number"]
    assert failures[0].detail == "12"


def test_model_text_numeral_not_shown_is_rejected(snapshot):
    failures = check_report(
        "Revenue reached 1,590.",
        [_shown("1,590")],
        METRICS,
        snapshot,
        model_text="Revenue hit 99 this week.",
    )
    assert _kinds(failures) == ["unlisted_number"]
    assert failures[0].detail == "99"
