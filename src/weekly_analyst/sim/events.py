"""Injected-event specs for the replay simulator (spec section 7.1).

An event changes only the weeks and cells it names. ``trend_break`` and ``drift``
stay in effect from ``week`` to the last week; every other kind touches ``week`` only.
For value kinds, ``magnitude`` is a multiplicative factor on the targeted cells.
"""

from dataclasses import dataclass

KINDS = frozenset(
    {"outlier", "trend_break", "schema_change", "drift", "quiet", "mid_run_change", "injection"}
)


@dataclass(frozen=True)
class EventSpec:
    kind: str
    week: int
    metric: str | None
    segment: tuple[str, str, str] | None
    magnitude: float

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown event kind: {self.kind!r}")


@dataclass(frozen=True)
class TruthEvent:
    kind: str
    week: int
    metric: str | None
    segment: tuple[str, str, str] | None
    magnitude: float
