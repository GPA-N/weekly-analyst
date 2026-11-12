"""Attribution rows show a number the independent check can confirm (found on real retail data: the check halted briefs on these rows)."""

from datetime import date, timedelta

import polars as pl

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_calc import compute
from weekly_analyst.data.metrics_file import MetricDef
from weekly_analyst.systems.report_findings import attribution_rows, change_text

PREV, NOW = "2026-W40", "2026-W41"  # Monday 2026-09-28 and 2026-10-05


def _metric(agg: str) -> MetricDef:
    return MetricDef(name="rate", definition="d", owner="o", inclusions="i", exclusions="e", column="v", agg=agg,
                     segment_columns=["region"], min_change=0.01, version=1)


def _df(counts: dict[str, tuple[int, int]], values: dict[str, tuple[float, float]]) -> pl.DataFrame:
    rows = []
    for seg, (n_prev, n_now) in counts.items():
        for start, n, v in ((date(2026, 9, 28), n_prev, values[seg][0]), (date(2026, 10, 5), n_now, values[seg][1])):
            rows += [{"date": start + timedelta(days=i % 7), "region": seg, "v": v} for i in range(n)]
    return pl.DataFrame(rows)


def test_mean_metric_shows_the_segments_own_change_and_small_segments_are_left_out():
    n = Settings().min_rows_per_segment
    df = _df({"big": (n + 10, n + 10), "other": (n + 30, n + 30), "tiny": (n - 25, n - 25)},
             {"big": (0.10, 0.30), "other": (0.20, 0.20), "tiny": (0.50, 0.90)})
    rows = {r.segment: r.change for r in attribution_rows(df, "date", _metric("mean"), "region", NOW)}
    assert set(rows) == {"big", "other"}  # "tiny" has too few rows for the check to recompute
    assert rows["big"].text == change_text(0.30 - 0.10)  # its own change, not its weighted contribution to the total
    assert rows["big"].segment == {"region": "big"} and rows["big"].period == NOW


def test_every_shown_change_can_be_recomputed_from_the_slice():
    n = Settings().min_rows_per_segment
    df = _df({"a": (n, n + 5), "b": (n + 8, n)}, {"a": (1.0, 4.0), "b": (2.0, 1.5)})
    for row in attribution_rows(df, "date", _metric("sum"), "region", NOW):
        now, before = compute(df, _metric("sum"), row.change.segment, NOW), compute(df, _metric("sum"), row.change.segment, PREV)
        assert row.change.text == change_text(now - before)
