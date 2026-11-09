"""Demo sales data: one CSV with nine weeks, the last week changed by a scenario (used by the hosted demo and the eval).

Eight identical prior weeks give the detectors a history (cold start is fewer than 8 periods); the ninth week is
scaled by the scenario factor. Every segment has 35 rows per week, above the 30-row gate. Synthetic data only.
"""

import random
from datetime import date, timedelta
from pathlib import Path

SCENARIOS = {
    "quiet": 1.0,  # nothing changed: the agent stays silent
    "spike": 1.6,  # units and revenue jump in the newest week
    "drop": 0.55,  # units and revenue fall in the newest week
}
START = date(2026, 8, 3)  # a Monday
WEEKS = 9
SEGMENTS = [(r, p, c) for r in ("north", "south") for p in ("basic", "pro") for c in ("web", "store")]
HEADER = "date,region,product,channel,units,revenue,discount,returned"


def write_demo_csv(path: Path, scenario: str, *, weeks: int = WEEKS, start: date = START, seed: int = 7) -> Path:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}: use one of {sorted(SCENARIOS)}")
    if weeks < 2:
        raise ValueError("weeks must be at least 2")
    bump = SCENARIOS[scenario]
    rng = random.Random(seed)
    base = {
        seg: (rng.randint(5, 25), round(rng.uniform(9.0, 90.0), 2), round(rng.uniform(0.0, 0.2), 2))
        for seg in SEGMENTS
    }
    lines = [HEADER]
    for week in range(weeks):
        scale = bump if week == weeks - 1 else 1.0
        for day in range(7):
            day_iso = (start + timedelta(weeks=week, days=day)).isoformat()
            for seg in SEGMENTS:
                units0, price, discount = base[seg]
                for k in range(5):
                    units = round((units0 + (day + k) % 3) * scale)
                    revenue = round(units * price, 2)
                    returned = (day + k) % 2
                    lines.append(f"{day_iso},{seg[0]},{seg[1]},{seg[2]},{units},{revenue:.2f},{discount:.2f},{returned}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
