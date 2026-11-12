"""Independent recompute (ticket 008): pyarrow-only path and display matching."""

import ast
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from weekly_analyst.data.metrics_file import load_metrics_file
from weekly_analyst.data.recompute import matches_displayed, recompute

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "metrics_sales.yaml"
RECOMPUTE_SRC = ROOT / "src" / "weekly_analyst" / "data" / "recompute.py"

# 80 rows in ISO week 2026-W02 (Mon 2026-01-05 to Sun 2026-01-11).
# north 40 rows, south 35 rows, east 5 rows (east is below the 30-row gate).
REGIONS = ["north"] * 40 + ["south"] * 35 + ["east"] * 5
IN_PERIOD = [
    {
        "date": date(2026, 1, 5) + timedelta(days=i % 7),
        "region": region,
        "revenue": round(10 + i * 0.25, 2),
        "returned": int(i % 3 == 0),
    }
    for i, region in enumerate(REGIONS)
]
# Must be ignored: 2026-01-04 is ISO 2026-W01, 2026-01-12 is ISO 2026-W03.
OUTSIDE_PERIOD = [
    {"date": date(2026, 1, 4), "region": "north", "revenue": 1000.0, "returned": 1},
    {"date": date(2026, 1, 12), "region": "north", "revenue": 5000.0, "returned": 1},
]


def _metric(name):
    return next(m for m in load_metrics_file(FIXTURE).metrics if m.name == name)


def _write_snapshot(tmp_path, date_type):
    rows = IN_PERIOD + OUTSIDE_PERIOD
    dates = [r["date"] for r in rows]
    if date_type == "string":
        date_column = pa.array([d.isoformat() for d in dates], type=pa.string())
    else:
        date_column = pa.array(dates, type=pa.date32())
    table = pa.table(
        {
            "date": date_column,
            "region": [r["region"] for r in rows],
            "revenue": [r["revenue"] for r in rows],
            "returned": [r["returned"] for r in rows],
        }
    )
    path = tmp_path / f"snapshot_{date_type}.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture(params=["date32", "string"])
def snapshot(request, tmp_path):
    return _write_snapshot(tmp_path, request.param)


def test_sum_over_whole_period_matches_hand_value(snapshot):
    expected = sum(r["revenue"] for r in IN_PERIOD)
    got = recompute(snapshot, _metric("revenue"), None, "2026-W02")
    assert got == pytest.approx(expected, rel=1e-9)


def test_mean_over_segment_matches_hand_value(snapshot):
    north = [r for r in IN_PERIOD if r["region"] == "north"]
    expected = sum(r["returned"] for r in north) / len(north)
    got = recompute(snapshot, _metric("return_rate"), {"region": "north"}, "2026-W02")
    assert got == pytest.approx(expected, rel=1e-9)


def test_count_counts_rows_of_segment(snapshot):
    count_metric = _metric("revenue").model_copy(update={"agg": "count"})
    got = recompute(snapshot, count_metric, {"region": "south"}, "2026-W02")
    assert got == 35.0


def test_returns_none_below_min_rows_gate(snapshot):
    assert recompute(snapshot, _metric("revenue"), {"region": "east"}, "2026-W02") is None


def test_returns_none_when_period_has_no_rows(snapshot):
    assert recompute(snapshot, _metric("revenue"), None, "2026-W30") is None


def test_rejects_malformed_period(snapshot):
    with pytest.raises(ValueError):
        recompute(snapshot, _metric("revenue"), None, "2026-46")


@pytest.mark.parametrize(
    ("value", "shown", "expected"),
    [
        (1_249_999, "1.2M", True),
        (1_250_001, "1.2M", False),
        (12_345.4, "12,345", True),
        (12_345.6, "12,345", False),
        (1.46, "+1.5", True),
        (-1.46, "+1.5", False),
        (-1.46, "-1.5", True),
        (0.034, "3.4%", True),
        (0.0347, "3.4%", False),
        (-0.0204, "-2.0%", True),
        (float("nan"), "1.2M", False),
    ],
)
def test_matches_displayed_rounds_to_shown_precision(value, shown, expected):
    assert matches_displayed(value, shown) is expected


def test_matches_displayed_rejects_unparseable_text():
    with pytest.raises(ValueError):
        matches_displayed(1.0, "about 1")


def test_recompute_imports_no_polars_duckdb_pandas_or_metrics_calc():
    tree = ast.parse(RECOMPUTE_SRC.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    banned = ("polars", "duckdb", "pandas", "weekly_analyst.data.metrics_calc")
    hits = sorted(n for n in names for b in banned if n == b or n.startswith(b + "."))
    assert hits == []
