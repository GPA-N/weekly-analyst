"""Parameter validators for named queries (spec 6.4, ticket 047).

Each check returns the value it accepted or raises NamedQueryError. A refusal names the parameter and never
echoes the model's value, so a hostile value cannot reach the halt reasons or the logs.
"""

import re
from datetime import date, timedelta

from weekly_analyst.data.metrics_file import MetricDef, MetricsFile


class NamedQueryError(ValueError):
    """A named query was refused before any SQL was built."""


_PERIOD = re.compile(r"[0-9]{4}-W[0-9]{2}")
_FILTER_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _./&-]{0,39}")
_DIRECTIONS = ("desc", "asc")


def period_range(period: object) -> tuple[date, date]:
    """ISO week "YYYY-Www" as its Monday and Sunday dates."""
    if not isinstance(period, str) or _PERIOD.fullmatch(period) is None:
        raise NamedQueryError("period must look like YYYY-Www")
    try:
        monday = date.fromisocalendar(int(period[:4]), int(period[6:]), 1)
    except ValueError:
        raise NamedQueryError("period is not a valid ISO week") from None
    return monday, monday + timedelta(days=6)


def check_metric(metrics: MetricsFile, name: object) -> MetricDef:
    if isinstance(name, str):
        for metric in metrics.metrics:
            if metric.name == name:
                return metric
    raise NamedQueryError("metric is not in the metrics file")


def check_segment(metric: MetricDef, column: object) -> str:
    if isinstance(column, str) and column in metric.segment_columns:
        return column
    raise NamedQueryError("segment column is not a segment of this metric")


def check_segment_column(metrics: MetricsFile, column: object) -> str:
    """A segment column of any metric in the file (distinct_count has no metric of its own)."""
    names = {name for metric in metrics.metrics for name in metric.segment_columns}
    if isinstance(column, str) and column in names:
        return column
    raise NamedQueryError("column is not a segment column of the metrics file")


def check_n(n: object) -> int:
    if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 10:
        return n
    raise NamedQueryError("n must be an integer from 1 to 10")


def check_direction(direction: object) -> str:
    if direction in _DIRECTIONS:
        return direction
    raise NamedQueryError("direction must be desc or asc")


def check_filter_value(value: object) -> str:
    if isinstance(value, str) and _FILTER_VALUE.fullmatch(value):
        return value
    raise NamedQueryError("filter value does not match the allowed pattern")
