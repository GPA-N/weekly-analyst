"""Report model: the shown numbers, findings and report data the renderer lays out (spec 6.8, ticket 020).

Code builds these dataclasses. The renderer only draws them and never computes a number.
"""

from dataclasses import asdict, dataclass

from weekly_analyst.report.charts import ChartSpec


class ReportTooLong(RuntimeError):
    """The report is over the page cap. Nothing is written."""


def _opt_dict(value) -> dict | None:
    return None if value is None else value.to_dict()


def _opt(kind, data: dict | None):
    return None if data is None else kind.from_dict(data)


@dataclass
class ShownNumber:
    text: str
    metric: str
    segment: dict | None
    period: str
    prev_period: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ShownNumber":
        return cls(**data)


@dataclass
class Kpi:
    label: str
    value: ShownNumber
    vs_prior: ShownNumber
    vs_baseline: ShownNumber | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "value": self.value.to_dict(),
            "vs_prior": self.vs_prior.to_dict(),
            "vs_baseline": _opt_dict(self.vs_baseline),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Kpi":
        return cls(
            label=data["label"],
            value=ShownNumber.from_dict(data["value"]),
            vs_prior=ShownNumber.from_dict(data["vs_prior"]),
            vs_baseline=_opt(ShownNumber, data["vs_baseline"]),
        )


@dataclass
class AttributionRow:
    segment: str
    change: ShownNumber

    def to_dict(self) -> dict:
        return {"segment": self.segment, "change": self.change.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "AttributionRow":
        return cls(segment=data["segment"], change=ShownNumber.from_dict(data["change"]))


@dataclass
class Finding:
    item_key: str
    supporting: ShownNumber
    so_what: str
    chart: ChartSpec
    attribution: list[AttributionRow]
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "item_key": self.item_key,
            "supporting": self.supporting.to_dict(),
            "so_what": self.so_what,
            "chart": self.chart.model_dump(mode="json"),
            "attribution": [row.to_dict() for row in self.attribution],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Finding":
        return cls(
            item_key=data["item_key"],
            supporting=ShownNumber.from_dict(data["supporting"]),
            so_what=data["so_what"],
            chart=ChartSpec.model_validate(data["chart"]),
            attribution=[AttributionRow.from_dict(row) for row in data["attribution"]],
            interpretation=data["interpretation"],
        )


@dataclass
class MetricDefinition:
    name: str
    definition: str
    owner: str
    includes: str
    excludes: str
    version: str
    checksum: str


@dataclass
class DataTable:
    item_key: str
    columns: list[str]
    rows: list[list[ShownNumber | str]]

    def to_dict(self) -> dict:
        rows = [[cell.to_dict() if isinstance(cell, ShownNumber) else cell for cell in row] for row in self.rows]
        return {"item_key": self.item_key, "columns": list(self.columns), "rows": rows}

    @classmethod
    def from_dict(cls, data: dict) -> "DataTable":
        rows = [[ShownNumber.from_dict(c) if isinstance(c, dict) else c for c in row] for row in data["rows"]]
        return cls(item_key=data["item_key"], columns=list(data["columns"]), rows=rows)


_PLAIN_FIELDS = (
    "report_name", "reporting_period", "prior_period", "data_as_of", "prepared_for", "report_version",
    "run_id", "snapshot_date", "snapshot_hash", "headline", "recommended_action", "risks_caveats", "next_checks",
)


@dataclass
class ReportModel:
    report_name: str
    reporting_period: str
    prior_period: str
    data_as_of: str
    prepared_for: str
    report_version: str
    run_id: str
    snapshot_date: str
    snapshot_hash: str
    headline: str
    recommended_action: str
    kpis: list[Kpi]
    findings: list[Finding]
    risks_caveats: list[str]
    next_checks: list[str]
    metric_definitions: list[MetricDefinition]
    data_tables: list[DataTable]

    def to_dict(self) -> dict:
        return {
            **{name: getattr(self, name) for name in _PLAIN_FIELDS},
            "kpis": [kpi.to_dict() for kpi in self.kpis],
            "findings": [finding.to_dict() for finding in self.findings],
            "metric_definitions": [asdict(m) for m in self.metric_definitions],
            "data_tables": [table.to_dict() for table in self.data_tables],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReportModel":
        return cls(
            **{name: data[name] for name in _PLAIN_FIELDS},
            kpis=[Kpi.from_dict(kpi) for kpi in data["kpis"]],
            findings=[Finding.from_dict(finding) for finding in data["findings"]],
            metric_definitions=[MetricDefinition(**m) for m in data["metric_definitions"]],
            data_tables=[DataTable.from_dict(table) for table in data["data_tables"]],
        )


@dataclass
class RenderResult:
    pages: int
    shown_numbers: list[ShownNumber]
    warnings: list[str]
    structural_numbers: list[str]  # numerals the renderer writes itself, not shown numbers


def readable_key(item_key: str) -> str:
    """The plan item key as a reader sees it: the metric and its segment, no period and no separators."""
    parts = item_key.split("|")
    values = [pair.split("=", 1)[-1] for pair in (parts[2].split(",") if len(parts) > 2 else []) if pair]
    return f"{parts[0]}: {' / '.join(values)}" if values else parts[0]
