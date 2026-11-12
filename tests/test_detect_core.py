"""Detection core: per-segment z-score with shrinkage and CUSUM (spec 6.6 a, c; ticket 010)."""

import math
import random
from datetime import date, timedelta

import polars as pl
import pytest

from weekly_analyst.core.config import Settings
from weekly_analyst.detection import cusum_score, zscore_scores
from weekly_analyst.data.metrics_file import MetricDef, MetricsFile

MONDAY_W40 = date(2026, 9, 28)  # Monday of ISO week 2026-W40
DATE_COLUMN = "date"


def period_of(week: int) -> str:
    iso = (MONDAY_W40 + timedelta(weeks=week)).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def metrics_file() -> MetricsFile:
    metric = MetricDef(
        name="units_mean",
        definition="Average units per sale row.",
        owner="test",
        inclusions="all rows",
        exclusions="none",
        column="units",
        agg="mean",
        segment_columns=["region"],
        min_change=0.0,
        version=1,
    )
    return MetricsFile(metrics=[metric], date_column=DATE_COLUMN, checksum="test")


def block(week: int, region: str, value: float | None, rows: int = 30) -> list[tuple]:
    """rows identical sale rows: the segment-week mean is exactly `value`."""
    return [(week, region, value)] * rows


def frame(*blocks: list[tuple]) -> pl.DataFrame:
    entries = [entry for b in blocks for entry in b]
    return pl.DataFrame(
        {
            DATE_COLUMN: [MONDAY_W40 + timedelta(weeks=w) for w, _, _ in entries],
            "region": [r for _, r, _ in entries],
            "units": [v for _, _, v in entries],
        },
        schema={DATE_COLUMN: pl.Date, "region": pl.Utf8, "units": pl.Float64},
    )


def key(region: str, week: int) -> tuple:
    return ("units_mean", (("region", region),), period_of(week))


def test_segment_under_min_rows_is_dropped_before_scoring():
    history = frame(*[block(w, "north", 100.0 + (5.0 if w % 2 else -5.0)) for w in range(8)])
    too_small = Settings().min_rows_per_segment - 1
    current = frame(block(8, "north", 100.0), block(8, "south", 1000.0, rows=too_small))

    scores = zscore_scores(history, current, metrics_file(), Settings())

    assert key("south", 8) not in scores
    assert key("north", 8) in scores


def test_tiny_segment_cannot_dominate_through_shrinkage():
    # Unshrunk, south's z would be 5 / 0.01 = 500 (its own history sd is 0.01).
    history = frame(
        *[block(w, "north", 100.0 + (5.0 if w % 2 else -5.0)) for w in range(10)],
        *[block(w, "south", v) for w, v in zip(range(3), (50.00, 50.01, 49.99))],
    )
    current = frame(block(10, "north", 100.0), block(10, "south", 55.0))

    scores = zscore_scores(history, current, metrics_file(), Settings())

    assert scores[key("south", 10)] < 5.0


@pytest.mark.parametrize(
    ("history_value", "current_value"),
    [(42.0, 43.0), (None, None)],
    ids=["constant", "all_null"],
)
def test_constant_and_all_null_columns_score_zero_not_nan(history_value, current_value):
    history = frame(
        *[block(w, r, history_value) for w in range(10) for r in ("north", "south")]
    )
    current = frame(block(10, "north", current_value), block(10, "south", current_value))

    scores = zscore_scores(history, current, metrics_file(), Settings())

    assert set(scores) == {key("north", 10), key("south", 10)}
    for value in scores.values():
        assert math.isfinite(value)
        assert value == 0.0


def test_cusum_flags_shifted_mean_and_not_flat_noise():
    rng = random.Random(11)
    flat = [rng.gauss(0.0, 1.0) for _ in range(30)]
    shifted = [rng.gauss(0.0, 1.0) for _ in range(20)] + [rng.gauss(3.0, 1.0) for _ in range(20)]

    assert cusum_score(shifted, k=0.5, h=5.0) >= 1.0
    assert cusum_score(flat, k=0.5, h=5.0) < 1.0


def test_scaling_a_column_by_two_leaves_zscores_unchanged():
    rng = random.Random(3)
    segments = ("north", "south", "east")
    history = frame(*[block(w, r, rng.uniform(50.0, 150.0)) for w in range(10) for r in segments])
    current = frame(*[block(10, r, rng.uniform(50.0, 150.0)) for r in segments])

    base = zscore_scores(history, current, metrics_file(), Settings())
    doubled = zscore_scores(
        history.with_columns(pl.col("units") * 2.0),
        current.with_columns(pl.col("units") * 2.0),
        metrics_file(),
        Settings(),
    )

    assert any(v > 0.0 for v in base.values())
    assert base.keys() == doubled.keys()
    for k in base:
        assert doubled[k] == pytest.approx(base[k], rel=1e-9)
