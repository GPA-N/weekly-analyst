"""Metric computation against known constructed totals (spec 6.3, ticket 007)."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_calc import compute
from weekly_analyst.data.metrics_file import load_metrics_file

FIXTURE = Path(__file__).parent / "fixtures" / "metrics_sales.yaml"
WEEK_START = date(2026, 11, 9)  # Monday of ISO week 2026-W46
PERIOD = "2026-W46"
REGIONS = ("north", "south", "east", "west")
PRODUCTS = ("basic", "plus", "pro")
CHANNELS = ("web", "store", "partner")
SCHEMA = {
    "date": pl.Date,
    "region": pl.Utf8,
    "product": pl.Utf8,
    "channel": pl.Utf8,
    "units": pl.Int64,
    "revenue": pl.Float64,
    "discount": pl.Float64,
    "returned": pl.Int64,
}


def metrics_by_name() -> dict:
    return {m.name: m for m in load_metrics_file(FIXTURE).metrics}


def line(day, region, product, channel, units=2, revenue=21.0, discount=0.0, returned=0):
    return {
        "date": day,
        "region": region,
        "product": product,
        "channel": channel,
        "units": units,
        "revenue": revenue,
        "discount": discount,
        "returned": returned,
    }


def build_rows() -> list[dict]:
    """Every (region, product, channel) gets exactly 30 rows in PERIOD; two rows fall outside it."""
    rows = []
    combos = [(r, p, c) for r in REGIONS for p in PRODUCTS for c in CHANNELS]
    for ci, (region, product, channel) in enumerate(combos):
        for j in range(30):
            units = (ci + j) % 7 + 1
            rows.append(
                line(
                    WEEK_START + timedelta(days=j % 7),
                    region,
                    product,
                    channel,
                    units=units,
                    revenue=units * 10.5,
                    discount=(j % 4) * 0.05,
                    returned=int((ci + j) % 5 == 0),
                )
            )
    # Neighbouring ISO weeks (Sunday 2026-11-08 is W45, Monday 2026-11-16 is W47): must never count.
    rows.append(line(date(2026, 11, 8), "north", "pro", "web", units=10_000, revenue=1.0, discount=0.9, returned=1))
    rows.append(line(date(2026, 11, 16), "north", "pro", "web", units=10_000, revenue=1.0, discount=0.9, returned=1))
    return rows


def truth(rows, column, agg, segment):
    """Expected value computed in plain Python from the constructed rows, not from polars."""
    selected = [
        r
        for r in rows
        if WEEK_START <= r["date"] <= WEEK_START + timedelta(days=6)
        and all(r[k] == v for k, v in (segment or {}).items())
    ]
    values = [r[column] for r in selected]
    if agg == "count":
        return float(len(values))
    if agg == "sum":
        return float(sum(values))
    return sum(values) / len(values)


@pytest.mark.parametrize(
    "segment",
    [
        None,
        {"region": "north"},
        {"region": "north", "product": "pro"},
        {"region": "south", "product": "basic", "channel": "web"},
    ],
)
def test_compute_equals_truth_for_every_metric(segment):
    rows = build_rows()
    df = pl.DataFrame(rows, schema=SCHEMA)
    for m in load_metrics_file(FIXTURE).metrics:
        expected = truth(rows, m.column, m.agg, segment)
        assert compute(df, m, segment, PERIOD) == pytest.approx(expected, rel=1e-9), m.name


def test_count_agg_counts_rows_in_slice():
    df = pl.DataFrame(build_rows(), schema=SCHEMA)
    m = metrics_by_name()["units"].model_copy(update={"agg": "count"})
    assert compute(df, m, {"region": "north"}, PERIOD) == 270.0


def test_none_below_min_rows_and_value_at_min_rows():
    n = Settings().min_rows_per_segment
    seg = {"region": "east", "product": "plus", "channel": "store"}
    units = metrics_by_name()["units"]
    short = [line(WEEK_START + timedelta(days=i % 7), "east", "plus", "store") for i in range(n - 1)]
    assert compute(pl.DataFrame(short, schema=SCHEMA), units, seg, PERIOD) is None
    enough = short + [line(WEEK_START, "east", "plus", "store")]
    assert compute(pl.DataFrame(enough, schema=SCHEMA), units, seg, PERIOD) == pytest.approx(2.0 * n)


def test_period_is_iso_week_including_year_boundary():
    n = Settings().min_rows_per_segment
    seg = {"region": "east", "product": "plus", "channel": "store"}
    w53 = [line(date(2026, 12, 28) + timedelta(days=i % 7), "east", "plus", "store", units=2) for i in range(n)]
    outside = [
        line(date(2026, 12, 27), "east", "plus", "store", units=1000),  # Sunday, 2026-W52
        line(date(2027, 1, 4), "east", "plus", "store", units=1000),  # Monday, 2027-W01
    ]
    df = pl.DataFrame(w53 + outside, schema=SCHEMA)
    assert compute(df, metrics_by_name()["units"], seg, "2026-W53") == pytest.approx(2.0 * n)


@pytest.mark.parametrize(("metric_name", "null_column"), [("units", "units"), ("avg_discount", "discount")])
def test_all_null_metric_gives_none(metric_name, null_column):
    n = Settings().min_rows_per_segment
    rows = [line(WEEK_START + timedelta(days=i % 7), "west", "pro", "partner") for i in range(n)]
    for r in rows:
        r[null_column] = None
    df = pl.DataFrame(rows, schema=SCHEMA)
    seg = {"region": "west", "product": "pro", "channel": "partner"}
    assert compute(df, metrics_by_name()[metric_name], seg, PERIOD) is None
