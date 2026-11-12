"""The min-rows rule for named queries (spec 6.4, ticket 047), on a hand-built snapshot with hand-computed values.

Big has 40 rows a week and small has 10. Settings().min_rows_per_segment is 30, so small is always left out.
"""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import load_metrics_file
from weekly_analyst.data.named_queries import build_sql
from weekly_analyst.data.query import LocalQuery

METRICS = load_metrics_file(Path(__file__).parent / "fixtures" / "metrics_sales.yaml")
PERIOD = "2026-W03"


def run_named(query, name, **params):
    return query.run(build_sql(METRICS, name, params))


def _thin_row(day, region, revenue):
    return {
        "date": day, "region": region, "product": "pro", "channel": "web", "revenue": revenue,
        "units": 1, "discount": 0.0, "returned": 0,
    }


@pytest.fixture
def thin(tmp_path):
    """W02 and W03: big has 40 rows, small has 10. W04: only small, 10 rows. Min rows is 30."""
    n_big, n_small = 40, 10
    assert n_big >= Settings().min_rows_per_segment > n_small
    rows = []
    for week_start in (date(2026, 1, 5), date(2026, 1, 12)):
        rows += [_thin_row(week_start + timedelta(days=i % 7), "big", 10.0) for i in range(n_big)]
        rows += [_thin_row(week_start + timedelta(days=i % 7), "small", 5.0) for i in range(n_small)]
    rows += [_thin_row(date(2026, 1, 19) + timedelta(days=i % 7), "small", 5.0) for i in range(n_small)]
    schema = {
        "date": pl.Date, "region": pl.Utf8, "product": pl.Utf8, "channel": pl.Utf8,
        "revenue": pl.Float64, "units": pl.Int64, "discount": pl.Float64, "returned": pl.Int64,
    }
    path = tmp_path / "thin.parquet"
    pl.DataFrame(rows, schema=schema).write_parquet(path)
    return LocalQuery(path, METRICS.columns(), Settings())


def test_segment_below_min_rows_is_absent_from_by_segment_and_top(thin):
    assert run_named(thin, "by_segment", metric="revenue", segment_col="region", period=PERIOD) == [
        {"segment": "big", "value": 400.0}
    ]
    assert run_named(
        thin, "top_segments", metric="revenue", segment_col="region", period=PERIOD, n=10, direction="desc"
    ) == [{"segment": "big", "value": 400.0}]


def test_share_total_counts_the_thin_segment_but_does_not_list_it(thin):
    rows = run_named(thin, "share_of_total", metric="revenue", segment_col="region", period=PERIOD)
    assert [r["segment"] for r in rows] == ["big"]
    assert rows[0]["share"] == pytest.approx(400.0 / 450.0, rel=1e-9)


def test_thin_filter_slice_and_thin_period_return_nothing(thin):
    assert run_named(
        thin, "filtered_total", metric="revenue", period=PERIOD, filter_col="region", filter_value="small"
    ) == []
    assert run_named(thin, "by_segment", metric="revenue", segment_col="region", period="2026-W04") == []
    assert run_named(thin, "spread", metric="revenue", period="2026-W04", segment_col="region") == []
    assert run_named(thin, "distinct_count", column="region", period="2026-W04") == []


def test_ratio_and_period_compare_need_the_segment_in_both_periods(thin):
    assert run_named(
        thin, "ratio", metric_a="revenue", metric_b="units", period=PERIOD, segment_col="region"
    ) == [{"segment": "big", "value": 10.0}]
    assert run_named(thin, "ratio", metric_a="revenue", metric_b="units", period=PERIOD) == [{"value": 9.0}]
    compare = run_named(
        thin, "period_compare", metric="revenue", period=PERIOD, prev_period="2026-W02", segment_col="region"
    )
    assert compare == [{"segment": "big", "now": 400.0, "before": 400.0, "delta": 0.0}]
    assert run_named(
        thin, "period_compare", metric="revenue", period="2026-W04", prev_period=PERIOD, segment_col="region"
    ) == []


def test_distinct_count_and_spread_count_only_segments_that_pass(thin):
    assert run_named(thin, "distinct_count", column="region", period=PERIOD) == [{"value": 2}]
    spread = run_named(thin, "spread", metric="revenue", period=PERIOD, segment_col="region")
    assert spread == [{"median": 400.0, "stddev": None, "min": 400.0, "max": 400.0}]
