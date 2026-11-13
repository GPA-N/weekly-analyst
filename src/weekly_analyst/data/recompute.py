"""Independent recompute of a metric from the Parquet snapshot (spec 6.3).

Deliberately separate from metrics_calc: pyarrow and pyarrow.compute only, so the
two implementations share neither engine nor code.
"""

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import MetricDef

# The snapshot's date column. recompute's signature carries no date column name,
# and the simulator and metrics fixture both use "date".
DATE_COLUMN = "date"

_PERIOD = re.compile(r"^(\d{4})-W(\d{2})$")
_SHOWN = re.compile(
    r"^(?P<sign>[+-]?)(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?(?P<suffix>[KMB%]?)$"
)
# Divisor that turns the raw value into the unit shown ("%" shows a fraction as percent).
_SCALE = {
    "": Decimal(1),
    "K": Decimal(10**3),
    "M": Decimal(10**6),
    "B": Decimal(10**9),
    "%": Decimal("0.01"),
}


def _parse_period(period: str) -> tuple[int, int]:
    match = _PERIOD.match(period)
    if match is None:
        raise ValueError(f"period must look like 2026-W46, got {period!r}")
    return int(match.group(1)), int(match.group(2))


def recompute(
    snapshot_path: Path, m: MetricDef, segment: dict[str, str] | None, period: str, date_column: str = DATE_COLUMN
) -> float | None:
    year, week = _parse_period(period)
    segment = segment or {}
    table = pq.read_table(snapshot_path, columns=[date_column, m.column, *segment])

    dates = pc.cast(table[date_column], pa.date32())
    mask = pc.and_(pc.equal(pc.iso_year(dates), year), pc.equal(pc.iso_week(dates), week))
    for key, value in segment.items():
        mask = pc.and_(mask, pc.equal(pc.cast(table[key], pa.string()), value))
    rows = table.filter(mask, null_selection_behavior="drop")

    if rows.num_rows < Settings().min_rows_per_segment:
        return None
    values = rows[m.column]
    if m.agg == "sum":
        result = pc.sum(values)
    elif m.agg == "mean":
        result = pc.mean(values)
    else:
        result = pc.count(values)
    result = result.as_py()
    return None if result is None else float(result)


def matches_displayed(value: float, shown: str) -> bool:
    match = _SHOWN.match(shown.strip())
    if match is None:
        raise ValueError(f"unrecognised displayed number: {shown!r}")
    if not math.isfinite(value):
        return False
    frac = match.group("frac") or ""
    expected = Decimal(match.group("sign") + match.group("int").replace(",", "") + ("." + frac if frac else ""))
    scaled = Decimal(repr(float(value))) / _SCALE[match.group("suffix")]
    rounded = scaled.quantize(Decimal(1).scaleb(-len(frac)), rounding=ROUND_HALF_UP)
    return rounded == expected
