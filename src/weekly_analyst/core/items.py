"""Detection item (metric, segment, period, score, detail). Moved from detect.py so analysis need not import detect."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Item:
    metric: str
    segment: tuple[tuple[str, str], ...]
    period: str
    score: float
    detail: dict[str, Any]
