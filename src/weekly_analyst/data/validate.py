"""Deterministic validation of an extracted snapshot (spec 6.2, state 3)."""

from dataclasses import dataclass
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from weekly_analyst.data.metrics_file import MetricDef, MetricsFile

FindingKind = Literal[
    "schema_change", "nulls", "duplicates", "row_jump", "implausible_combo", "constant_column"
]
ROW_JUMP_LIMIT = 0.5


@dataclass(frozen=True)
class Finding:
    kind: FindingKind
    metric: str | None  # None for dataset-level findings
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    findings: list[Finding]
    halted_metrics: set[str]


class PlausibilityRule(BaseModel):
    """Rows where column_a == value_a and column_b == value_b are implausible."""

    model_config = ConfigDict(extra="forbid")

    name: str
    column_a: str
    value_a: int | float | str
    column_b: str
    value_b: int | float | str


def _metric_columns(metric: MetricDef, date_column: str) -> set[str]:
    return {date_column, metric.column, *metric.segment_columns}


def _missing_count(series: pl.Series) -> int:
    count = series.null_count()
    if series.dtype.is_float():
        count += int(series.is_nan().sum())
    return count


def _present_values(series: pl.Series) -> pl.Series:
    values = series.drop_nulls()
    if values.dtype.is_float():
        values = values.filter(values.is_not_nan())
    return values


def _is_row_jump(rows: int, prior: int) -> bool:
    if prior == 0:
        return rows != 0
    return abs(rows - prior) / prior > ROW_JUMP_LIMIT


def validate(
    df: pl.DataFrame,
    metrics: MetricsFile,
    prior_schema: dict[str, str] | None,
    prior_rows: int | None,
) -> ValidationResult:
    findings: list[Finding] = []
    halted: set[str] = set()
    broken: set[str] = set()
    users: dict[str, list[str]] = {}
    for metric in metrics.metrics:
        for col in _metric_columns(metric, metrics.date_column):
            users.setdefault(col, []).append(metric.name)

    for col in sorted(users):
        if col not in df.columns:
            reason = "is missing"
        elif (
            prior_schema is not None
            and col in prior_schema
            and str(df.schema[col]) != prior_schema[col]
        ):
            reason = f"changed type from {prior_schema[col]} to {df.schema[col]}"
        else:
            continue
        broken.add(col)
        for name in users[col]:
            halted.add(name)
            findings.append(Finding("schema_change", name, f"column {col!r} {reason}"))

    for col in sorted(users):
        if col not in df.columns:
            continue
        series = df.get_column(col)
        missing = _missing_count(series)
        present = _present_values(series)
        constant = present.len() > 0 and present.n_unique() == 1
        for name in users[col]:
            if missing:
                detail = f"column {col!r}: {missing} of {df.height} value(s) null or NaN"
                findings.append(Finding("nulls", name, detail))
            if constant:
                detail = f"column {col!r} has one value in every row"
                findings.append(Finding("constant_column", name, detail))

    duplicates = df.height - df.unique().height
    if duplicates > 0:
        findings.append(Finding("duplicates", None, f"{duplicates} duplicate row(s)"))

    if prior_rows is not None and _is_row_jump(df.height, prior_rows):
        detail = f"row count {df.height} vs {prior_rows} in prior snapshot"
        findings.append(Finding("row_jump", None, detail))

    for rule in getattr(metrics, "plausibility", []):
        if rule.column_a in broken or rule.column_b in broken:
            continue
        if rule.column_a not in df.columns or rule.column_b not in df.columns:
            continue
        hits = df.filter(
            (pl.col(rule.column_a) == rule.value_a) & (pl.col(rule.column_b) == rule.value_b)
        ).height
        if hits:
            findings.append(Finding("implausible_combo", None, f"{rule.name}: {hits} row(s)"))

    return ValidationResult(findings=findings, halted_metrics=halted)
