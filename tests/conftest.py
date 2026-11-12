"""Shared simulator snapshot for the named query tests (ticket 047).

The snapshot is three weeks of simulator CSV, concatenated and run through the real extract. In it the west region
carries a hostile label, so every named query is exercised on a label that must be sanitised before it leaves the tool.
"""

import csv
from pathlib import Path

import pytest

from weekly_analyst.core.config import Settings
from weekly_analyst.data.extract import extract
from weekly_analyst.data.metrics_file import load_metrics_file
from weekly_analyst.data.query import LocalQuery
from weekly_analyst.sim.generate import generate

FIXTURE = Path(__file__).parent / "fixtures" / "metrics_sales.yaml"
METRICS = load_metrics_file(FIXTURE)
# A region label with a quote, a semicolon and a DROP.
HOSTILE = "west'); DROP TABLE snapshot; --"


@pytest.fixture(scope="module")
def named_snapshot(tmp_path_factory):
    work = tmp_path_factory.mktemp("named")
    weekly = generate(seed=11, weeks=3, events=[], out_dir=work / "weeks").files
    rows = []
    for index, path in enumerate(weekly):
        with path.open(encoding="utf-8", newline="") as fh:
            body = list(csv.reader(fh))
        rows.extend(body if index == 0 else body[1:])
    region = rows[0].index("region")
    for row in rows[1:]:
        if row[region] == "west":
            row[region] = HOSTILE
    combined = work / "combined.csv"
    with combined.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    out = work / "snap"
    out.mkdir()
    return extract(combined, METRICS, out, Settings()).parquet_path


@pytest.fixture(scope="module")
def named_query(named_snapshot):
    return LocalQuery(named_snapshot, METRICS.columns(), Settings())
