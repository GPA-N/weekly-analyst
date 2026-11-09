"""Components: plain frozen dataclasses, one per piece of run data, each JSON-serialisable (architecture.md).

A run's progress is exactly the set of components it holds. Components store plain data only; the
analysis and report packages build them, so this module imports nothing from them.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from weekly_analyst.core.items import Item


class _Component:
    def to_json(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, data: dict[str, Any]):
        return cls(**data)


@dataclass(frozen=True)
class SourceC(_Component):
    """The CSV a run started from and its hash. Written at run start; the guard checks it before each state."""

    csv_path: str
    snapshot_hash: str


@dataclass(frozen=True)
class SnapshotC(_Component):
    """The Parquet snapshot written by extract."""

    parquet: str
    rows_read: int
    rows_in_file: int


@dataclass(frozen=True)
class ValidationC(_Component):
    period: str
    n_periods: int
    cold_start: bool
    findings: list[dict]


@dataclass(frozen=True)
class ItemsC(_Component):
    """Detected items: the alerts this run selected, whether any are material, and the week's count before it."""

    selected: list[Item]
    material: bool
    alerts_before: int

    def to_json(self) -> dict[str, Any]:
        return {
            "selected": [_item_to_json(item) for item in self.selected],
            "material": self.material,
            "alerts_before": self.alerts_before,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ItemsC":
        return cls(
            selected=[_item_from_json(row) for row in data["selected"]],
            material=data["material"],
            alerts_before=data["alerts_before"],
        )


@dataclass(frozen=True)
class PlanC(_Component):
    steps: list[dict]


@dataclass(frozen=True)
class EvidenceC(_Component):
    """Investigate results: one entry per tool call, with its tool, item key and result."""

    results: list[dict]


@dataclass(frozen=True)
class DraftC(_Component):
    draft: dict


@dataclass(frozen=True)
class CheckC(_Component):
    """Check state: the redraft round and the failures that round was asked to fix."""

    round_no: int
    failures: list[str]


@dataclass(frozen=True)
class ReportC(_Component):
    """The checked PDF and its page count, plus what approve re-renders and re-checks: the ReportModel as a
    dict, the shown numbers, the structural numerals, the model-written text and the paper size."""

    pdf: str
    pages: int
    model: dict = field(default_factory=dict)
    shown_numbers: list[dict] = field(default_factory=list)
    structural_numbers: list[str] = field(default_factory=list)
    model_text: str = ""
    paper: str = "A4"


@dataclass(frozen=True)
class OutboxC(_Component):
    report: str


@dataclass(frozen=True)
class HaltC(_Component):
    reason: str


def _item_to_json(item: Item) -> dict[str, Any]:
    return {
        "metric": item.metric,
        "segment": [list(pair) for pair in item.segment],
        "period": item.period,
        "score": item.score,
        "detail": item.detail,
    }


def _item_from_json(data: dict[str, Any]) -> Item:
    return Item(
        metric=data["metric"],
        segment=tuple(tuple(pair) for pair in data["segment"]),
        period=data["period"],
        score=data["score"],
        detail=data["detail"],
    )


COMPONENTS: dict[str, type[_Component]] = {
    cls.__name__: cls
    for cls in (
        SourceC, SnapshotC, ValidationC, ItemsC, PlanC, EvidenceC,
        DraftC, CheckC, ReportC, OutboxC, HaltC,
    )
}
