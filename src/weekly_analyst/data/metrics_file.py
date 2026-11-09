"""Metrics file: metric definitions, column map and checksum (spec 6.8)."""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator, model_validator

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Names reach SQL identifiers, plan item keys (which use | , = as separators) and the SQL grammar,
# so they are plain identifiers and nothing else.
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
MAX_FILE_BYTES = 1_000_000


def _identifier(value: str, what: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{what} {value!r} must match [A-Za-z_][A-Za-z0-9_]{{0,63}}")
    return value


class MetricDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    definition: NonBlank
    owner: str
    inclusions: str
    exclusions: str
    column: str
    agg: Literal["sum", "mean", "count"]
    segment_columns: list[str]
    min_change: float
    version: int

    @field_validator("name", "column")
    @classmethod
    def _names_are_identifiers(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("segment_columns")
    @classmethod
    def _segments_are_unique_identifiers(cls, value: list[str]) -> list[str]:
        for item in value:
            _identifier(item, "segment column")
        if len(set(value)) != len(value):
            raise ValueError("segment_columns must not repeat a column")
        return value

    @field_validator("min_change")
    @classmethod
    def _min_change_is_finite_and_not_negative(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("min_change must be a finite number, zero or more")
        return value

    @field_validator("version")
    @classmethod
    def _version_is_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("version must be 1 or more")
        return value


class MetricsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[MetricDef]
    date_column: str
    checksum: str

    @field_validator("date_column")
    @classmethod
    def _date_column_is_identifier(cls, value: str) -> str:
        return _identifier(value, "date_column")

    @model_validator(mode="after")
    def _metrics_are_unique_and_present(self) -> "MetricsFile":
        if not self.metrics:
            raise ValueError("metrics must list at least one metric")
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("metric names must be unique")
        return self

    def columns(self) -> set[str]:
        cols = {self.date_column}
        for metric in self.metrics:
            cols.add(metric.column)
            cols.update(metric.segment_columns)
        return cols


def _checksum(metrics_file: MetricsFile) -> str:
    content = metrics_file.model_dump(mode="json", exclude={"checksum"})
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_metrics_file(path: Path) -> MetricsFile:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"metrics file too large (over {MAX_FILE_BYTES} bytes)")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        raw = json.loads(text)
    elif path.suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported metrics file type {path.suffix!r}: use .yaml, .yml or .json")
    if not isinstance(raw, dict):
        raise ValueError("metrics file must be a mapping with date_column and metrics")
    metrics_file = MetricsFile.model_validate({**raw, "checksum": ""})
    metrics_file.checksum = _checksum(metrics_file)
    return metrics_file
