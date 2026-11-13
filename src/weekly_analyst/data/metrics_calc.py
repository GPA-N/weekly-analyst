"""Model-path metric computation with polars (spec 6.3, 6.8).

Independent of recompute.py by design: no shared code or engine.
"""

import math

import polars as pl

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import MetricDef

DATE_COLUMN = "date"


def compute(
    df: pl.DataFrame, m: MetricDef, segment: dict[str, str] | None, period: str, date_column: str = DATE_COLUMN
) -> float | None:
    """Value of metric m for one ISO week (e.g. "2026-W46") and segment.

    Returns None when the slice has fewer rows than Settings().min_rows_per_segment,
    or when there are no non-null values to aggregate.
    """
    predicate = pl.col(date_column).cast(pl.Date).dt.strftime("%G-W%V") == period
    for key, value in (segment or {}).items():
        predicate = predicate & (pl.col(key) == value)
    rows = df.filter(predicate)
    if rows.height < Settings().min_rows_per_segment:
        return None
    values = rows.get_column(m.column).drop_nulls()
    if m.agg == "count":
        return float(values.len())
    if values.len() == 0:
        return None
    result = values.sum() if m.agg == "sum" else values.mean()
    if result is None or math.isnan(result):
        return None
    return float(result)
