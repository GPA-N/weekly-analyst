"""Named parameterised queries (spec 6.4, ticket 047): parameter validators, the period range and refusals.

Every validator refuses before any SQL is built. Refusals raise NamedQueryError, a ValueError.
Truth tests for each query: test_named_queries_truth.py. The min-rows rule: test_named_queries_thin.py.
"""

from datetime import date
from pathlib import Path

import pytest

from weekly_analyst.data.metrics_file import load_metrics_file
from weekly_analyst.data.named_checks import (
    NamedQueryError,
    check_direction,
    check_filter_value,
    check_metric,
    check_n,
    check_segment,
    period_range,
)
from weekly_analyst.data.named_queries import build_sql
from weekly_analyst.data.query import LocalQuery

FIXTURE = Path(__file__).parent / "fixtures" / "metrics_sales.yaml"
METRICS = load_metrics_file(FIXTURE)
PERIOD = "2026-W03"
INJECTION = "x'); DROP TABLE snapshot; --"


def test_period_range_is_monday_to_sunday_of_iso_week():
    assert period_range("2026-W46") == (date(2026, 11, 9), date(2026, 11, 15))


def test_period_range_handles_week_53_of_a_53_week_year():
    assert period_range("2026-W53") == (date(2026, 12, 28), date(2027, 1, 3))


@pytest.mark.parametrize(
    "bad",
    [
        "2026-46",
        "2026-W5",
        "2026-W46 ",
        "2026-W00",
        "2026-W54",
        "2025-W53",
        "0000-W01",
        "２０２６-W46",
        "2026-W46\n",
        None,
        46,
        b"2026-W46",
    ],
)
def test_period_range_refuses_malformed_or_impossible_periods(bad):
    with pytest.raises(NamedQueryError):
        period_range(bad)


@pytest.mark.parametrize("n", [1, 10])
def test_check_n_accepts_bounds(n):
    assert check_n(n) == n


@pytest.mark.parametrize("bad", [0, 11, True, 3.0, "3", None])
def test_check_n_refuses_outside_int_1_to_10(bad):
    with pytest.raises(NamedQueryError):
        check_n(bad)


@pytest.mark.parametrize("direction", ["desc", "asc"])
def test_check_direction_accepts_asc_and_desc(direction):
    assert check_direction(direction) == direction


@pytest.mark.parametrize("bad", ["DESC", "up", "", None])
def test_check_direction_refuses_anything_else(bad):
    with pytest.raises(NamedQueryError):
        check_direction(bad)


@pytest.mark.parametrize("value", ["north", "North Region", "a.b/c&d-e_f", "x" * 40])
def test_check_filter_value_accepts_conforming_values(value):
    assert check_filter_value(value) == value


@pytest.mark.parametrize(
    "bad",
    ["x" * 41, "north\n", "north'", "a; DROP TABLE snapshot", "", " north", None, 7],
)
def test_check_filter_value_refuses_nonconforming_values(bad):
    with pytest.raises(NamedQueryError):
        check_filter_value(bad)


def test_check_metric_returns_definition_for_known_name():
    assert check_metric(METRICS, "revenue").column == "revenue"


@pytest.mark.parametrize("bad", ["profit", "", None, 3])
def test_check_metric_refuses_names_not_in_metrics_file(bad):
    with pytest.raises(NamedQueryError):
        check_metric(METRICS, bad)


def test_check_segment_accepts_only_the_metrics_segment_columns():
    revenue = check_metric(METRICS, "revenue")
    assert check_segment(revenue, "region") == "region"
    for bad in ("customer_email", "revenue", "date", None):
        with pytest.raises(NamedQueryError):
            check_segment(revenue, bad)


REFUSED = [
    pytest.param("by_segment", {"metric": "profit", "segment_col": "region", "period": PERIOD}, id="unknown_metric"),
    pytest.param(
        "by_segment", {"metric": "revenue", "segment_col": "customer_email", "period": PERIOD}, id="non_segment_col"
    ),
    pytest.param(
        "top_segments",
        {"metric": "revenue", "segment_col": "region", "period": PERIOD, "n": 11, "direction": "desc"},
        id="n_eleven",
    ),
    pytest.param(
        "top_segments",
        {"metric": "revenue", "segment_col": "region", "period": PERIOD, "n": 3, "direction": "sideways"},
        id="bad_direction",
    ),
    pytest.param(
        "by_segment", {"metric": "revenue", "segment_col": "region", "period": "2026-46"}, id="bad_period"
    ),
    pytest.param(
        "filtered_total",
        {"metric": "revenue", "period": PERIOD, "filter_col": "region", "filter_value": INJECTION},
        id="injection_filter_value",
    ),
    pytest.param(
        "by_segment",
        {"metric": "revenue; DROP TABLE snapshot", "segment_col": "region", "period": PERIOD},
        id="injection_metric_name",
    ),
    pytest.param(
        "by_segment",
        {"metric": "revenue", "segment_col": "region; DROP TABLE snapshot", "period": PERIOD},
        id="injection_segment_col",
    ),
    pytest.param(
        "by_segment",
        {"metric": "revenue", "segment_col": "region", "period": f"{PERIOD}'; DROP TABLE snapshot"},
        id="injection_period",
    ),
    pytest.param("no_such_query", {}, id="unknown_query_name"),
    pytest.param("by_segment", {"metric": "revenue", "segment_col": "region"}, id="missing_period"),
    pytest.param(
        "by_segment",
        {"metric": "revenue", "segment_col": "region", "period": PERIOD, "sql": "SELECT 1"},
        id="extra_parameter",
    ),
]


@pytest.mark.parametrize(("name", "params"), REFUSED)
def test_refused_query_raises_named_query_error(name, params):
    with pytest.raises(NamedQueryError):
        build_sql(METRICS, name, params)


@pytest.mark.parametrize("params", [None, "region=north", ["region"]])
def test_params_must_be_an_object(params):
    with pytest.raises(NamedQueryError):
        build_sql(METRICS, "by_segment", params)


def test_refusal_runs_no_sql_and_the_snapshot_survives(named_query, monkeypatch):
    executed = []
    original = LocalQuery.run

    def spy(self, sql, timeout_s=10.0):
        executed.append(sql)
        return original(self, sql, timeout_s)

    monkeypatch.setattr(LocalQuery, "run", spy)
    params = {"metric": "revenue", "period": PERIOD, "filter_col": "region", "filter_value": INJECTION}
    with pytest.raises(NamedQueryError):
        named_query.run(build_sql(METRICS, "filtered_total", params))
    assert executed == []
    survived = {"metric": "revenue", "segment_col": "region", "period": PERIOD}
    assert named_query.run(build_sql(METRICS, "by_segment", survived))


def test_refusal_message_never_echoes_the_hostile_value():
    params = {"metric": "revenue", "period": PERIOD, "filter_col": "region", "filter_value": INJECTION}
    with pytest.raises(NamedQueryError) as err:
        build_sql(METRICS, "filtered_total", params)
    assert "DROP" not in str(err.value) and "snapshot" not in str(err.value)
