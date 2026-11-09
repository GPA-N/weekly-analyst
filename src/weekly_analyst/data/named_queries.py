"""Named, parameterised queries (spec 6.4): the model picks a name and fills validated parameters, never SQL.

Each builder checks its parameters (named_checks) before it formats any SQL, and the result is one SELECT over
the snapshot table, run through LocalQuery so the lockdown stays. A refusal raises NamedQueryError, a ValueError,
so the investigate state halts on it like any other tool error.
"""

from collections.abc import Callable
from dataclasses import dataclass

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import MetricDef, MetricsFile
from weekly_analyst.data.named_checks import (
    NamedQueryError,
    check_direction,
    check_filter_value,
    check_metric,
    check_n,
    check_segment,
    check_segment_column,
    period_range,
)
from weekly_analyst.data.query import TABLE_NAME

_AGG_SQL = {"sum": "SUM", "mean": "AVG", "count": "COUNT"}


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _agg(m: MetricDef) -> str:
    return f"{_AGG_SQL[m.agg]}({_q(m.column)})"


def _min_rows() -> int:
    return Settings().min_rows_per_segment


def _where_period(metrics: MetricsFile, period: object) -> str:
    start, end = period_range(period)
    col = _q(metrics.date_column)
    return f"{col} >= DATE '{start.isoformat()}' AND {col} <= DATE '{end.isoformat()}'"


def _grouped(metrics: MetricsFile, value: str, seg: str, period: object, tail: str) -> str:
    """A value per segment for one period. Segments below the min-rows rule are left out."""
    return (
        f"SELECT CAST({_q(seg)} AS VARCHAR) AS segment, {value} AS value "
        f"FROM {TABLE_NAME} WHERE {_where_period(metrics, period)} "
        f"GROUP BY {_q(seg)} HAVING COUNT(*) >= {_min_rows()} {tail}"
    )


def _by_segment(metrics: MetricsFile, *, metric: object, segment_col: object, period: object) -> str:
    m = check_metric(metrics, metric)
    seg = check_segment(m, segment_col)
    return _grouped(metrics, _agg(m), seg, period, "ORDER BY 1")


def _top_segments(
    metrics: MetricsFile, *, metric: object, segment_col: object, period: object, n: object, direction: object
) -> str:
    m = check_metric(metrics, metric)
    seg = check_segment(m, segment_col)
    limit = check_n(n)
    order = "DESC" if check_direction(direction) == "desc" else "ASC"
    return _grouped(metrics, _agg(m), seg, period, f"ORDER BY 2 {order}, 1 LIMIT {limit}")


def _ratio(
    metrics: MetricsFile, *, metric_a: object, metric_b: object, period: object, segment_col: object = None
) -> str:
    """SUM of metric_a over SUM of metric_b, for the period or per segment (both metrics must have the segment)."""
    a = check_metric(metrics, metric_a)
    b = check_metric(metrics, metric_b)
    value = f"SUM({_q(a.column)}) / NULLIF(SUM({_q(b.column)}), 0)"
    if segment_col is None:
        return (
            f"SELECT {value} AS value FROM {TABLE_NAME} "
            f"WHERE {_where_period(metrics, period)} HAVING COUNT(*) >= {_min_rows()}"
        )
    seg = check_segment(a, segment_col)
    check_segment(b, segment_col)
    return _grouped(metrics, value, seg, period, "ORDER BY 1")


def _period_compare(
    metrics: MetricsFile, *, metric: object, period: object, prev_period: object, segment_col: object = None
) -> str:
    """The metric now, before and their delta. A segment needs the min-rows count in both periods."""
    m = check_metric(metrics, metric)
    now_in = f"({_where_period(metrics, period)})"
    before_in = f"({_where_period(metrics, prev_period)})"
    agg, col = _AGG_SQL[m.agg], _q(m.column)
    now_value = f"{agg}(CASE WHEN {now_in} THEN {col} END)"
    before_value = f"{agg}(CASE WHEN {before_in} THEN {col} END)"
    values = f'{now_value} AS "now", {before_value} AS "before", ({now_value} - {before_value}) AS "delta"'
    enough = (
        f"COUNT(CASE WHEN {now_in} THEN 1 END) >= {_min_rows()} "
        f"AND COUNT(CASE WHEN {before_in} THEN 1 END) >= {_min_rows()}"
    )
    if segment_col is None:
        return f"SELECT {values} FROM {TABLE_NAME} WHERE {now_in} OR {before_in} HAVING {enough}"
    seg = check_segment(m, segment_col)
    return (
        f"SELECT CAST({_q(seg)} AS VARCHAR) AS segment, {values} FROM {TABLE_NAME} "
        f"WHERE {now_in} OR {before_in} GROUP BY {_q(seg)} HAVING {enough} ORDER BY 1"
    )


def _spread(metrics: MetricsFile, *, metric: object, period: object, segment_col: object = None) -> str:
    """Median, sample stddev, min and max. With a segment: of the per-segment values. Without: of the rows."""
    m = check_metric(metrics, metric)
    col = _q(m.column)
    stats = (
        f'MEDIAN(CAST({col} AS DOUBLE)) AS "median", STDDEV_SAMP(CAST({col} AS DOUBLE)) AS "stddev", '
        f'MIN(CAST({col} AS DOUBLE)) AS "min", MAX(CAST({col} AS DOUBLE)) AS "max"'
    )
    where = _where_period(metrics, period)
    if segment_col is None:
        return f"SELECT {stats} FROM {TABLE_NAME} WHERE {where} HAVING COUNT(*) >= {_min_rows()}"
    seg = check_segment(m, segment_col)
    # The derived column is named as the metric column, the one name the lockdown check lets a model reference.
    per_segment = (
        f"SELECT {_agg(m)} AS {col} FROM {TABLE_NAME} WHERE {where} "
        f"GROUP BY {_q(seg)} HAVING COUNT(*) >= {_min_rows()}"
    )
    return f"SELECT {stats} FROM ({per_segment}) AS per_segment HAVING COUNT(*) >= 1"


def _distinct_count(
    metrics: MetricsFile,
    *,
    column: object,
    period: object,
    filter_col: object = None,
    filter_value: object = None,
) -> str:
    col = check_segment_column(metrics, column)
    where = _where_period(metrics, period)
    if filter_col is not None or filter_value is not None:
        if filter_col is None or filter_value is None:
            raise NamedQueryError("filter_col and filter_value must be given together")
        fcol = check_segment_column(metrics, filter_col)
        where += f" AND CAST({_q(fcol)} AS VARCHAR) = {_literal(check_filter_value(filter_value))}"
    return (
        f"SELECT COUNT(DISTINCT CAST({_q(col)} AS VARCHAR)) AS value FROM {TABLE_NAME} "
        f"WHERE {where} HAVING COUNT(*) >= {_min_rows()}"
    )


def _share_of_total(metrics: MetricsFile, *, metric: object, segment_col: object, period: object) -> str:
    """Each segment's part of the period total. The total counts every row, including segments left out."""
    m = check_metric(metrics, metric)
    seg = check_segment(m, segment_col)
    part = "COUNT" if m.agg == "count" else "SUM"
    col, where = _q(m.column), _where_period(metrics, period)
    total = f"(SELECT {part}({col}) FROM {TABLE_NAME} WHERE {where})"
    return (
        f"SELECT CAST({_q(seg)} AS VARCHAR) AS segment, {part}({col}) / NULLIF({total}, 0) AS share "
        f"FROM {TABLE_NAME} WHERE {where} GROUP BY {_q(seg)} HAVING COUNT(*) >= {_min_rows()} ORDER BY 1"
    )


def _filtered_total(
    metrics: MetricsFile, *, metric: object, period: object, filter_col: object, filter_value: object
) -> str:
    m = check_metric(metrics, metric)
    col = check_segment(m, filter_col)
    value = check_filter_value(filter_value)
    return (
        f"SELECT {_agg(m)} AS value FROM {TABLE_NAME} "
        f"WHERE {_where_period(metrics, period)} AND CAST({_q(col)} AS VARCHAR) = {_literal(value)} "
        f"HAVING COUNT(*) >= {_min_rows()}"
    )


@dataclass(frozen=True)
class NamedQuery:
    name: str
    params: tuple[str, ...]
    build: Callable[..., str]  # build(metrics, **params) -> one SELECT; validates before it formats
    optional: frozenset[str] = frozenset()


NAMED_QUERIES: dict[str, NamedQuery] = {
    "by_segment": NamedQuery("by_segment", ("metric", "segment_col", "period"), _by_segment),
    "top_segments": NamedQuery(
        "top_segments", ("metric", "segment_col", "period", "n", "direction"), _top_segments
    ),
    "filtered_total": NamedQuery(
        "filtered_total", ("metric", "period", "filter_col", "filter_value"), _filtered_total
    ),
    "share_of_total": NamedQuery("share_of_total", ("metric", "segment_col", "period"), _share_of_total),
    "ratio": NamedQuery(
        "ratio", ("metric_a", "metric_b", "period", "segment_col"), _ratio, frozenset({"segment_col"})
    ),
    "distinct_count": NamedQuery(
        "distinct_count",
        ("column", "period", "filter_col", "filter_value"),
        _distinct_count,
        frozenset({"filter_col", "filter_value"}),
    ),
    "period_compare": NamedQuery(
        "period_compare",
        ("metric", "period", "prev_period", "segment_col"),
        _period_compare,
        frozenset({"segment_col"}),
    ),
    "spread": NamedQuery("spread", ("metric", "period", "segment_col"), _spread, frozenset({"segment_col"})),
}


def build_sql(metrics: MetricsFile, name: object, params: object) -> str:
    """The SELECT for one named query. Unknown names and wrong parameter sets are refused here."""
    if not isinstance(name, str) or name not in NAMED_QUERIES:
        raise NamedQueryError("unknown named query")
    if not isinstance(params, dict):
        raise NamedQueryError("params must be an object")
    query = NAMED_QUERIES[name]
    if set(params) - set(query.params):
        raise NamedQueryError("unexpected parameter")
    if set(query.params) - query.optional - set(params):
        raise NamedQueryError("missing parameter")
    return query.build(metrics, **params)
