"""Named queries against independent truth on simulator data (spec 6.4, ticket 047).
Level queries: recompute.recompute (pyarrow). Share, ratio, distinct counts and spread: polars.
Each comparison asserts its truth is non-empty first, so an empty match proves nothing.
"""

from pathlib import Path

import polars as pl
import pytest

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import load_metrics_file
from weekly_analyst.data.named_queries import build_sql
from weekly_analyst.data.recompute import recompute

FIXTURE = Path(__file__).parent / "fixtures" / "metrics_sales.yaml"
METRICS = load_metrics_file(FIXTURE)
# The fixture plus one count metric (units, agg count), so the count path is covered by the same truth tests.
EXTENDED = METRICS.model_copy(
    update={
        "metrics": [
            *METRICS.metrics,
            next(m for m in METRICS.metrics if m.name == "units").model_copy(
                update={"name": "unit_rows", "agg": "count"}
            ),
        ]
    }
)
PERIOD = "2026-W03"
PREV = "2026-W02"


def run_named(query, name, **params):
    return query.run(build_sql(EXTENDED, name, params))


def metric(name):
    return next(m for m in EXTENDED.metrics if m.name == name)


def segment_values(snapshot, column):
    return sorted(set(pl.read_parquet(snapshot).get_column(column).drop_nulls().to_list()))


@pytest.mark.parametrize("name", ["revenue", "units", "return_rate", "avg_discount", "unit_rows"])
@pytest.mark.parametrize("segment_col", ["region", "product", "channel"])
def test_by_segment_equals_recompute_for_every_segment(named_query, named_snapshot, name, segment_col):
    expected = {}
    for value in segment_values(named_snapshot, segment_col):
        got = recompute(named_snapshot, metric(name), {segment_col: value}, PERIOD)
        if got is not None:
            expected[value] = got
    assert expected, "the truth must be non-empty, or the comparison proves nothing"
    rows = run_named(named_query, "by_segment", metric=name, segment_col=segment_col, period=PERIOD)
    got = {row["segment"]: row["value"] for row in rows}
    assert set(got) == set(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-9)


@pytest.mark.parametrize("name", ["revenue", "return_rate"])
@pytest.mark.parametrize("segment_col", ["region", "product", "channel"])
@pytest.mark.parametrize("n", [1, 3, 10])
@pytest.mark.parametrize("direction", ["desc", "asc"])
def test_top_segments_is_the_first_n_of_the_recompute_ranking(
    named_query, named_snapshot, name, segment_col, n, direction
):
    values = {}
    for value in segment_values(named_snapshot, segment_col):
        got = recompute(named_snapshot, metric(name), {segment_col: value}, PERIOD)
        if got is not None:
            values[value] = got
    assert values, "the truth must be non-empty, or the comparison proves nothing"
    # Ties (not expected with simulator data) fall back to segment name, ascending, in both directions.
    sign = -1 if direction == "desc" else 1
    ranked = sorted(values.items(), key=lambda kv: (sign * kv[1], kv[0]))[:n]
    rows = run_named(
        named_query, "top_segments", metric=name, segment_col=segment_col, period=PERIOD, n=n, direction=direction
    )
    assert [row["segment"] for row in rows] == [key for key, _ in ranked]
    for row, (_, value) in zip(rows, ranked):
        assert row["value"] == pytest.approx(value, rel=1e-9)


@pytest.mark.parametrize("name", ["revenue", "return_rate", "unit_rows"])
@pytest.mark.parametrize(
    ("filter_col", "value"),
    [("region", "north"), ("region", "south"), ("product", "pro"), ("channel", "web")],
)
def test_filtered_total_equals_recompute_of_the_slice(named_query, named_snapshot, name, filter_col, value):
    expected = recompute(named_snapshot, metric(name), {filter_col: value}, PERIOD)
    assert expected is not None, "the truth must be non-empty, or the comparison proves nothing"
    rows = run_named(
        named_query, "filtered_total", metric=name, period=PERIOD, filter_col=filter_col, filter_value=value
    )
    assert len(rows) == 1
    assert rows[0]["value"] == pytest.approx(expected, rel=1e-9)


def period_frame(snapshot, period):
    df = pl.read_parquet(snapshot)
    return df.filter(pl.col("date").dt.strftime("%G-W%V") == period)


def additive(m):
    """The part of a metric that adds up across segments: non-null count for count, the sum otherwise."""
    column = pl.col(m.column)
    return column.is_not_null().cast(pl.Float64) if m.agg == "count" else column.cast(pl.Float64)


def polars_shares(snapshot, m, segment_col, period):
    rows = period_frame(snapshot, period)
    total = rows.select(additive(m).sum()).item()
    grouped = rows.group_by(segment_col).agg(additive(m).sum().alias("part"), pl.len().alias("rows"))
    return {
        r[segment_col]: r["part"] / total
        for r in grouped.iter_rows(named=True)
        if r["rows"] >= Settings().min_rows_per_segment
    }


@pytest.mark.parametrize("name", ["revenue", "return_rate", "unit_rows"])
@pytest.mark.parametrize("segment_col", ["region", "product", "channel"])
def test_share_of_total_equals_polars_share_of_the_period(named_query, named_snapshot, name, segment_col):
    expected = polars_shares(named_snapshot, metric(name), segment_col, PERIOD)
    assert expected, "the truth must be non-empty, or the comparison proves nothing"
    rows = run_named(named_query, "share_of_total", metric=name, segment_col=segment_col, period=PERIOD)
    got = {row["segment"]: row["share"] for row in rows}
    assert set(got) == set(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-9)


def polars_ratio(snapshot, a, b, segment_col, period):
    """SUM of a over SUM of b for the period (key None) or per segment, segments below the min-rows rule left out."""
    rows = period_frame(snapshot, period)
    aggs = [
        pl.len().alias("rows"),
        pl.col(a.column).cast(pl.Float64).sum().alias("sa"),
        pl.col(b.column).cast(pl.Float64).sum().alias("sb"),
    ]
    if segment_col is None:
        keyed = {None: rows.select(aggs).row(0, named=True)}
    else:
        keyed = {r[segment_col]: r for r in rows.group_by(segment_col).agg(aggs).iter_rows(named=True)}
    min_rows = Settings().min_rows_per_segment
    return {key: r["sa"] / r["sb"] for key, r in keyed.items() if r["rows"] >= min_rows}


@pytest.mark.parametrize(("a", "b"), [("revenue", "units"), ("return_rate", "units")])
@pytest.mark.parametrize("segment_col", [None, "region", "product"])
def test_ratio_equals_polars_sum_over_sum(named_query, named_snapshot, a, b, segment_col):
    expected = polars_ratio(named_snapshot, metric(a), metric(b), segment_col, PERIOD)
    assert expected, "the truth must be non-empty, or the comparison proves nothing"
    params = {"metric_a": a, "metric_b": b, "period": PERIOD}
    if segment_col is not None:
        params["segment_col"] = segment_col
    rows = run_named(named_query, "ratio", **params)
    if segment_col is None:
        assert len(rows) == 1
        assert rows[0]["value"] == pytest.approx(expected[None], rel=1e-9)
        return
    got = {row["segment"]: row["value"] for row in rows}
    assert set(got) == set(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-9)


@pytest.mark.parametrize(
    ("column", "filter_col", "value"),
    [
        ("region", None, None),
        ("product", None, None),
        ("channel", None, None),
        ("region", "product", "pro"),
        ("product", "region", "north"),
    ],
)
def test_distinct_count_equals_polars_unique_of_the_slice(named_query, named_snapshot, column, filter_col, value):
    frame = period_frame(named_snapshot, PERIOD)
    params = {"column": column, "period": PERIOD}
    if filter_col is not None:
        frame = frame.filter(pl.col(filter_col).cast(pl.Utf8) == value)
        params.update(filter_col=filter_col, filter_value=value)
    assert frame.height >= Settings().min_rows_per_segment, "the truth slice must pass the min-rows rule"
    expected = frame.get_column(column).drop_nulls().n_unique()
    assert expected > 1, "the truth must be non-trivial, or the comparison proves nothing"
    assert run_named(named_query, "distinct_count", **params) == [{"value": expected}]


@pytest.mark.parametrize("name", ["revenue", "return_rate"])
@pytest.mark.parametrize("segment_col", [None, "region", "product"])
def test_period_compare_equals_recompute_of_both_periods(named_query, named_snapshot, name, segment_col):
    m = metric(name)
    if segment_col is None:
        expected = {None: (recompute(named_snapshot, m, None, PERIOD), recompute(named_snapshot, m, None, PREV))}
    else:
        expected = {}
        for value in segment_values(named_snapshot, segment_col):
            now = recompute(named_snapshot, m, {segment_col: value}, PERIOD)
            before = recompute(named_snapshot, m, {segment_col: value}, PREV)
            if now is not None and before is not None:
                expected[value] = (now, before)
    assert expected, "the truth must be non-empty, or the comparison proves nothing"
    params = {"metric": name, "period": PERIOD, "prev_period": PREV}
    if segment_col is not None:
        params["segment_col"] = segment_col
    rows = run_named(named_query, "period_compare", **params)
    got = {row.get("segment"): row for row in rows}
    assert set(got) == set(expected)
    for key, (now, before) in expected.items():
        assert got[key]["now"] == pytest.approx(now, rel=1e-9)
        assert got[key]["before"] == pytest.approx(before, rel=1e-9)
        assert got[key]["delta"] == pytest.approx(now - before, rel=1e-9)


def per_segment_value(m):
    c = pl.col(m.column)
    if m.agg == "count":
        return c.is_not_null().sum().cast(pl.Float64)
    return c.cast(pl.Float64).mean() if m.agg == "mean" else c.cast(pl.Float64).sum()


def polars_spread(snapshot, m, segment_col, period):
    """Median, sample stddev, min and max: of the per-segment values if a segment is given, else of the rows."""
    rows = period_frame(snapshot, period)
    min_rows = Settings().min_rows_per_segment
    if segment_col is None:
        if rows.height < min_rows:
            return None
        values = rows.get_column(m.column).cast(pl.Float64).drop_nulls()
    else:
        grouped = rows.group_by(segment_col).agg(per_segment_value(m).alias("v"), pl.len().alias("rows"))
        values = grouped.filter(pl.col("rows") >= min_rows).get_column("v")
    return {"median": values.median(), "stddev": values.std(), "min": values.min(), "max": values.max()}


@pytest.mark.parametrize("name", ["revenue", "return_rate", "unit_rows"])
@pytest.mark.parametrize("segment_col", [None, "region", "product"])
def test_spread_equals_polars_median_stddev_min_max(named_query, named_snapshot, name, segment_col):
    expected = polars_spread(named_snapshot, metric(name), segment_col, PERIOD)
    assert expected, "the truth must be non-empty, or the comparison proves nothing"
    params = {"metric": name, "period": PERIOD}
    if segment_col is not None:
        params["segment_col"] = segment_col
    rows = run_named(named_query, "spread", **params)
    assert len(rows) == 1
    assert set(rows[0]) == {"median", "stddev", "min", "max"}
    for key, value in expected.items():
        assert rows[0][key] == pytest.approx(value, rel=1e-9), key
