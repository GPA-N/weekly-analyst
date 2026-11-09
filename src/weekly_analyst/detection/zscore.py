"""Per-segment z-score with shrinkage toward the pooled spread (spec 6.6 a)."""

import math

import polars as pl

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import MetricDef, MetricsFile
from weekly_analyst.detection.base import Detector, Key

SHRINKAGE_PRIOR_N = 30  # segment estimates get weight n / (n + 30)


def _segment_period_values(
    df: pl.DataFrame, m: MetricDef, date_column: str, min_rows: int
) -> dict[tuple[tuple[tuple[str, str], ...], str], float | None]:
    """Metric value per (segment, ISO week). Segment-weeks under the row gate are dropped."""
    values = pl.col(m.column).cast(pl.Float64, strict=False).fill_nan(None)
    if m.agg == "count":
        value = values.count().cast(pl.Float64)
    elif m.agg == "sum":
        value = pl.when(values.count() > 0).then(values.sum()).otherwise(None)
    else:
        value = values.mean()
    grouped = (
        df.with_columns(pl.col(date_column).cast(pl.Date).dt.strftime("%G-W%V").alias("period"))
        .group_by(["period", *m.segment_columns])
        .agg(pl.len().alias("rows"), value.alias("value"))
        .filter(pl.col("rows") >= min_rows)
    )
    out: dict[tuple[tuple[tuple[str, str], ...], str], float | None] = {}
    for row in grouped.iter_rows(named=True):
        segment = tuple((col, str(row[col])) for col in m.segment_columns)
        value_out = row["value"]
        if value_out is not None and not math.isfinite(value_out):
            value_out = None
        out[(segment, row["period"])] = value_out
    return out


def _baselines(
    past: dict[tuple[tuple[tuple[str, str], ...], str], float | None],
) -> dict[tuple[tuple[str, str], ...], tuple[int, float, float]]:
    """Per segment: (n, mean, sample sd) over its non-null history periods."""
    by_segment: dict[tuple[tuple[str, str], ...], list[float]] = {}
    for (segment, _period), value in past.items():
        if value is not None:
            by_segment.setdefault(segment, []).append(value)
    stats = {}
    for segment, values in by_segment.items():
        n = len(values)
        mean = sum(values) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
        stats[segment] = (n, mean, sd)
    return stats


def _pooled_cv(stats: dict[tuple[tuple[str, str], ...], tuple[int, float, float]]) -> float:
    """Global baseline for spread: pooled sd over pooled |mean|, so it scales with the data."""
    usable = [(sd, abs(mean)) for n, mean, sd in stats.values() if n > 1 and mean != 0.0]
    level_total = sum(level for _, level in usable)
    if level_total == 0.0:
        return 0.0
    return sum(sd for sd, _ in usable) / level_total


def _segment_z(
    value: float | None, base: tuple[int, float, float] | None, cv: float
) -> float:
    if value is None or base is None:
        return 0.0
    n, mean, sd = base
    if sd <= 0.0:  # zero-variance history: no z-score
        return 0.0
    w = n / (n + SHRINKAGE_PRIOR_N)
    sd_shrunk = w * sd + (1.0 - w) * cv * abs(mean)
    z = abs(value - mean) / sd_shrunk
    return z if math.isfinite(z) else 0.0


def zscore_scores(
    history: pl.DataFrame,
    current: pl.DataFrame,
    metrics: MetricsFile,
    settings: Settings,
) -> dict[tuple, float]:
    """|z| per (metric, segment, ISO week) of `current`, against each segment's history.

    Segment sd is shrunk toward the pooled coefficient of variation times the segment mean,
    with weight n / (n + 30). Scores are 0.0 where no finite score exists.
    """
    scores: dict[tuple, float] = {}
    for m in metrics.metrics:
        past = _segment_period_values(history, m, metrics.date_column, settings.min_rows_per_segment)
        now = _segment_period_values(current, m, metrics.date_column, settings.min_rows_per_segment)
        stats = _baselines(past)
        cv = _pooled_cv(stats)
        for (segment, period), value in now.items():
            scores[(m.name, segment, period)] = _segment_z(value, stats.get(segment), cv)
    return scores


class ZScoreDetector(Detector):
    """Owns the per-segment z-score: |z| of each segment-week against its own history."""

    name = "zscore"

    def score(
        self, history: pl.DataFrame, current: pl.DataFrame, metrics: MetricsFile, settings: Settings
    ) -> dict[Key, float]:
        return zscore_scores(history, current, metrics, settings)
