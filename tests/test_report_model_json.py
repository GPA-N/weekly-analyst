"""ReportModel JSON round trip: to_dict and from_dict restore the model exactly (spec 6.7, ticket 044).

approve re-renders the stored model, so what goes into the store must come back equal, cell types included.
"""

import json

from weekly_analyst.report.charts import ChartSpec
from weekly_analyst.report.model import (
    AttributionRow,
    DataTable,
    Finding,
    Kpi,
    MetricDefinition,
    ReportModel,
    ShownNumber,
)


def _shown(text="1,234.00", segment=None, prev=None) -> ShownNumber:
    return ShownNumber(text=text, metric="Revenue", segment=segment, period="2026-W44", prev_period=prev)


def _model() -> ReportModel:
    chart = ChartSpec(
        kind="line",
        title="Revenue by ISO week: region north",
        x=["W41", "W42", "W43"],
        series={"value": [1.5, 2.25, 1000.125]},
        unit="Revenue",
        alt_text="Line of revenue by ISO week for region north; the latest week is last.",
    )
    finding = Finding(
        item_key="Revenue | region north",
        supporting=_shown(segment={"region": "north"}),
        so_what="Revenue rose in the north.",
        chart=chart,
        attribution=[AttributionRow(segment="web", change=_shown("+12.00", {"region": "north"}, "2026-W43"))],
        interpretation="Each row is one channel segment and its share of the change in Revenue.",
    )
    return ReportModel(
        report_name="Weekly analysis",
        reporting_period="2026-W44",
        prior_period="2026-W43",
        data_as_of="2026-11-01",
        prepared_for="ops@example.com",
        report_version="1",
        run_id="run-20261109T120000-abcd1234",
        snapshot_date="2026-11-01",
        snapshot_hash="c0ffee" * 10,
        headline="Weekly sales moved",
        recommended_action="Review the flagged segments.",
        kpis=[
            Kpi(label="Revenue", value=_shown(), vs_prior=_shown("+1.00", prev="2026-W43"), vs_baseline=_shown("-3.00")),
            Kpi(label="Units", value=_shown("12"), vs_prior=_shown("+0")),
        ],
        findings=[finding],
        risks_caveats=["insufficient history: fewer than 8 prior weeks"],
        next_checks=["Check the source rows behind Revenue | region north"],
        metric_definitions=[
            MetricDefinition(
                name="Revenue",
                definition="Sum of revenue",
                owner="Finance",
                includes="all rows",
                excludes="returns",
                version="2",
                checksum="sum-1",
            )
        ],
        data_tables=[
            DataTable(
                item_key="Revenue | region north",
                columns=["Week", "Revenue"],
                rows=[["W41", _shown("1.00", {"region": "north"})], ["W42", "n/a"]],
            )
        ],
    )


def test_round_trip_through_json_is_exact():
    model = _model()
    restored = ReportModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored == model


def test_round_trip_keeps_cell_types():
    restored = ReportModel.from_dict(json.loads(json.dumps(_model().to_dict())))
    shown_cell, text_cell = restored.data_tables[0].rows[0][1], restored.data_tables[0].rows[1][1]
    assert isinstance(shown_cell, ShownNumber)
    assert shown_cell.segment == {"region": "north"}
    assert text_cell == "n/a"


def test_round_trip_keeps_none_fields_and_chart_floats():
    restored = ReportModel.from_dict(json.loads(json.dumps(_model().to_dict())))
    assert restored.kpis[1].vs_baseline is None
    assert restored.kpis[1].value.prev_period is None
    assert restored.findings[0].chart.series == {"value": [1.5, 2.25, 1000.125]}
    assert isinstance(restored.findings[0].chart, ChartSpec)


def test_to_dict_is_plain_json_and_second_round_trip_is_stable():
    once = _model().to_dict()
    json_text = json.dumps(once, sort_keys=True)  # raises if a dataclass or pydantic object leaked in
    assert ReportModel.from_dict(json.loads(json_text)).to_dict() == once
