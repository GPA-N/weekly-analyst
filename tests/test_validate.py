from datetime import date, timedelta
from pathlib import Path

import polars as pl

from weekly_analyst.data.metrics_file import MetricsFile, load_metrics_file
from weekly_analyst.data.validate import PlausibilityRule, validate

FIXTURE = Path(__file__).parent / "fixtures" / "metrics_sales.yaml"
METRICS = load_metrics_file(FIXTURE)
REGIONS = ("north", "south", "east", "west")
PRODUCTS = ("basic", "plus", "pro")
CHANNELS = ("web", "store", "partner")


class WithRules(MetricsFile):
    plausibility: list[PlausibilityRule] = []


def _frame(n: int = 6) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2026, 1, 5) + timedelta(days=i) for i in range(n)],
            "region": [REGIONS[i % 4] for i in range(n)],
            "product": [PRODUCTS[i % 3] for i in range(n)],
            "channel": [CHANNELS[i % 3] for i in range(n)],
            "revenue": [10.0 + i for i in range(n)],
            "units": [1 + i for i in range(n)],
            "returned": [i % 2 for i in range(n)],
            "discount": [0.01 * i for i in range(n)],
        }
    )


def _schema(df: pl.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in df.schema.items()}


def _kinds_metrics(findings):
    return {(f.kind, f.metric) for f in findings}


def test_clean_snapshot_has_no_findings():
    df = _frame()
    result = validate(df, METRICS, prior_schema=_schema(df), prior_rows=6)
    assert result.findings == []
    assert result.halted_metrics == set()


def test_retyped_column_halts_only_its_metrics():
    base = _frame()
    retyped = base.with_columns(pl.col("units").cast(pl.String))
    result = validate(retyped, METRICS, prior_schema=_schema(base), prior_rows=6)
    assert result.halted_metrics == {"units"}
    flagged = [f for f in result.findings if f.kind == "schema_change"]
    assert [f.metric for f in flagged] == ["units"]
    assert "units" in flagged[0].detail


def test_no_retype_check_without_prior_schema():
    base = _frame()
    retyped = base.with_columns(pl.col("units").cast(pl.String))
    result = validate(retyped, METRICS, prior_schema=None, prior_rows=None)
    assert result.halted_metrics == set()
    assert not [f for f in result.findings if f.kind == "schema_change"]


def test_missing_column_is_schema_change_finding():
    base = _frame()
    result = validate(base.drop("discount"), METRICS, prior_schema=_schema(base), prior_rows=6)
    assert result.halted_metrics == {"avg_discount"}
    assert _kinds_metrics(result.findings) == {("schema_change", "avg_discount")}


def test_all_null_and_constant_column_flagged_without_error():
    base = _frame()
    odd = base.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("discount"),
        pl.lit(0, dtype=pl.Int64).alias("returned"),
    )
    result = validate(odd, METRICS, prior_schema=_schema(base), prior_rows=6)
    kinds = _kinds_metrics(result.findings)
    assert ("nulls", "avg_discount") in kinds
    assert ("constant_column", "return_rate") in kinds
    assert result.halted_metrics == set()


def test_nan_counts_as_missing_in_float_column():
    base = _frame()
    with_nan = base.with_columns(
        discount=pl.Series("discount", [float("nan"), 0.01, 0.02, 0.03, 0.04, 0.05])
    )
    result = validate(with_nan, METRICS, prior_schema=_schema(base), prior_rows=6)
    assert ("nulls", "avg_discount") in _kinds_metrics(result.findings)


def test_row_count_jump_flagged():
    base = _frame(6)
    result = validate(_frame(10), METRICS, prior_schema=_schema(base), prior_rows=6)
    jumps = [f for f in result.findings if f.kind == "row_jump"]
    assert len(jumps) == 1
    assert jumps[0].metric is None


def test_row_count_change_of_exactly_50_percent_is_not_flagged():
    base = _frame(6)
    result = validate(_frame(9), METRICS, prior_schema=_schema(base), prior_rows=6)
    assert not [f for f in result.findings if f.kind == "row_jump"]


def test_duplicate_rows_flagged():
    base = _frame(6)
    duplicated = pl.concat([base, base.head(1)])
    result = validate(duplicated, METRICS, prior_schema=_schema(base), prior_rows=6)
    dups = [f for f in result.findings if f.kind == "duplicates"]
    assert len(dups) == 1
    assert dups[0].metric is None


def test_plausibility_hook_defaults_to_none():
    base = _frame()
    odd = base.with_columns(units=pl.Series("units", [1, 0, 3, 4, 5, 6]))
    result = validate(odd, METRICS, prior_schema=_schema(base), prior_rows=6)
    assert not [f for f in result.findings if f.kind == "implausible_combo"]


def test_plausibility_rule_flags_matching_rows():
    rule = PlausibilityRule(
        name="returned_with_zero_units", column_a="units", value_a=0, column_b="returned", value_b=1
    )
    metrics = WithRules(**METRICS.model_dump(), plausibility=[rule])
    base = _frame()
    odd = base.with_columns(units=pl.Series("units", [1, 0, 3, 4, 5, 6]))
    result = validate(odd, metrics, prior_schema=_schema(base), prior_rows=6)
    hits = [f for f in result.findings if f.kind == "implausible_combo"]
    assert len(hits) == 1
    assert "returned_with_zero_units" in hits[0].detail
