import polars as pl
import pytest

from weekly_analyst.analysis.analyst import (
    attribute_change,
    concentration,
    pooled_odds_ratio,
    small_group,
    wilson,
    wording_check,
)
from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import MetricDef


def _metric(agg: str) -> MetricDef:
    return MetricDef(
        name="amount_metric",
        definition="sum of amount",
        owner="test",
        inclusions="all rows",
        exclusions="none",
        column="amount",
        agg=agg,
        segment_columns=["region"],
        min_change=0.0,
        version=1,
    )


def _total(df: pl.DataFrame, metric: MetricDef) -> float:
    series = df[metric.column]
    if metric.agg == "sum":
        return series.sum()
    if metric.agg == "mean":
        return series.mean()
    return series.count()


def _frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    prev = pl.DataFrame({"region": ["N", "N", "S", "E"], "amount": [10.0, 20.0, 5.0, 4.0]})
    curr = pl.DataFrame(
        {
            "region": ["N", "S", "S", "W", "W", "W"],
            "amount": [15.0, 7.0, 3.0, 4.0, 6.0, None],
        }
    )
    return prev, curr


def test_wilson_zero_of_ten_inside_unit_interval():
    lo, hi = wilson(0, 10)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_ten_of_ten_inside_unit_interval():
    lo, hi = wilson(10, 10)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_five_of_ten_matches_known_interval():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)


def test_small_group_flags_below_min_rows_per_segment():
    settings = Settings()
    assert small_group(29, settings) is True
    assert small_group(30, settings) is False


def test_mantel_haenszel_matches_hand_computed_example():
    # Stratum 1 (a=10, b=10, c=5, d=5, n=30): ad/n = 5/3, bc/n = 5/3
    # Stratum 2 (a=4, b=6, c=2, d=8, n=20): ad/n = 8/5, bc/n = 3/5
    # Sum ad/n = 49/15, sum bc/n = 34/15, so OR = 49/34
    assert pooled_odds_ratio([(10, 10, 5, 5), (4, 6, 2, 8)]) == pytest.approx(49 / 34)


def test_concentration_top_decile_of_uniform_is_0_1():
    assert concentration([1.0] * 100) == pytest.approx(0.1)


def test_wording_check_flags_error_and_confirmed_loss():
    flags = wording_check("This is an error, a confirmed loss of 500.")
    assert "error" in flags
    assert "confirmed loss" in flags


def test_wording_check_flags_causal_words():
    assert wording_check("The spike was caused by a promo.") == ["caused"]
    assert wording_check("Traffic fell due to the outage.") == ["due to"]
    assert wording_check("The dip was because of the outage.") == ["because of"]
    assert wording_check("The change led to a drop.") == ["led to"]


def test_wording_check_passes_clean_text():
    text = "Unusual refund spike in region North. Labeled fraud value 1200, unverified."
    assert wording_check(text) == []


@pytest.mark.parametrize("agg", ["sum", "mean", "count"])
def test_attribution_sums_to_total_change_to_1e9(agg):
    prev, curr = _frames()
    metric = _metric(agg)
    out = attribute_change(prev, curr, metric, "region")
    expected = _total(curr, metric) - _total(prev, metric)
    assert out["delta"].sum() == pytest.approx(expected, abs=1e-9)


def test_attribution_sum_deltas_per_segment_and_shares():
    prev, curr = _frames()
    out = attribute_change(prev, curr, _metric("sum"), "region")
    deltas = dict(zip(out["segment"].to_list(), out["delta"].to_list()))
    assert deltas == pytest.approx({"N": -15.0, "S": 5.0, "E": -4.0, "W": 10.0})
    assert out.columns == ["segment", "delta", "share"]
    assert out["share"].sum() == pytest.approx(1.0)


def test_attribution_share_is_null_when_total_change_is_zero():
    prev, _ = _frames()
    out = attribute_change(prev, prev, _metric("sum"), "region")
    assert out["share"].null_count() == out.height
