"""Analyst checks in code, not model judgment (spec 6.6)."""

import math
import re
from collections.abc import Sequence

import polars as pl

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import MetricDef


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a rate. An empty group gives the full range."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def small_group(n: int, settings: Settings) -> bool:
    return n < settings.min_rows_per_segment


def pooled_odds_ratio(strata: list[tuple[int, int, int, int]]) -> float:
    """Mantel-Haenszel pooled odds ratio over 2x2 strata given as (a, b, c, d)."""
    numerator = 0.0
    denominator = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n == 0:
            continue
        numerator += a * d / n
        denominator += b * c / n
    if denominator == 0:
        raise ValueError("no discordant counts in any stratum; odds ratio is undefined")
    return numerator / denominator


def concentration(values: Sequence[float], top_fraction: float = 0.1) -> float:
    """Share of the total held by the largest `top_fraction` of values (at least one value)."""
    total = sum(values)
    if total == 0:
        return 0.0
    k = max(1, math.ceil(top_fraction * len(values)))
    return sum(sorted(values, reverse=True)[:k]) / total


def _contributions(df: pl.DataFrame, segment_col: str, metric: MetricDef) -> pl.DataFrame:
    """Per-segment share of one side's metric total, so the segments add up to the total."""
    value = pl.col(metric.column)
    if metric.agg == "count":
        numerator = value.is_not_null().cast(pl.Float64)
    else:
        numerator = value.cast(pl.Float64)
    if metric.agg == "mean":
        non_null = df[metric.column].count()
        if non_null == 0:
            raise ValueError(f"mean of {metric.column!r} is undefined: no non-null values")
        divisor = float(non_null)
    else:
        divisor = 1.0
    return (
        df.group_by(segment_col)
        .agg(numerator.sum().alias("contribution"))
        .rename({segment_col: "segment"})
        .with_columns(pl.col("contribution") / divisor)
    )


def attribute_change(
    prev: pl.DataFrame, curr: pl.DataFrame, metric: MetricDef, segment_col: str
) -> pl.DataFrame:
    """Additive split of a metric's change by segment. Deltas sum to the total change."""
    before = _contributions(prev, segment_col, metric)
    after = _contributions(curr, segment_col, metric)
    deltas = (
        pl.concat([after, before.with_columns(-pl.col("contribution"))])
        .group_by("segment", maintain_order=True)
        .agg(pl.col("contribution").sum().alias("delta"))
    )
    total_change = deltas["delta"].sum()
    if total_change == 0:
        share = pl.lit(None, dtype=pl.Float64)
    else:
        share = pl.col("delta") / total_change
    return deltas.with_columns(share.alias("share")).sort(pl.col("delta").abs(), descending=True)


_WORDING_FLAGS = {
    "error": re.compile(r"\berrors?\b", re.IGNORECASE),
    "confirmed loss": re.compile(r"\bconfirmed\s+loss\b", re.IGNORECASE),
    "caused": re.compile(r"\bcaused\b", re.IGNORECASE),
    "due to": re.compile(r"\bdue\s+to\b", re.IGNORECASE),
    "because of": re.compile(r"\bbecause\s+of\b", re.IGNORECASE),
    "led to": re.compile(r"\bled\s+to\b", re.IGNORECASE),
}


def wording_check(text: str) -> list[str]:
    return [flag for flag, pattern in _WORDING_FLAGS.items() if pattern.search(text)]
