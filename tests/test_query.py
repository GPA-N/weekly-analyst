import hashlib
import time
from pathlib import Path

import polars as pl
import pytest

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import load_metrics_file
from weekly_analyst.data.query import LocalQuery, QueryRejected

FIXTURE = Path(__file__).parent / "fixtures" / "metrics_sales.yaml"
ALLOWED = load_metrics_file(FIXTURE).columns()
ROWS = 2000
TIMEOUT = 1.0


def _write_snapshot(path: Path) -> Path:
    pl.DataFrame(
        {
            "date": [f"2026-01-{i % 28 + 1:02d}" for i in range(ROWS)],
            "revenue": [float(i) for i in range(ROWS)],
            "units": [i % 7 + 1 for i in range(ROWS)],
            "returned": [i % 10 == 0 for i in range(ROWS)],
            "discount": [0.05] * ROWS,
            "region": [("north", "south", "east", "west")[i % 4] for i in range(ROWS)],
            "product": ["widget"] * ROWS,
            "channel": ["web"] * ROWS,
            "customer_email": [f"user{i}@example.com" for i in range(ROWS)],
        }
    ).write_parquet(path)
    return path


@pytest.fixture
def snapshot(tmp_path):
    return _write_snapshot(tmp_path / "snapshot.parquet")


@pytest.fixture
def query(snapshot):
    return LocalQuery(snapshot, ALLOWED, Settings())


REJECTED = [
    pytest.param("SELECT * FROM read_csv('/etc/passwd')", id="read_csv_path"),
    pytest.param("SELECT * FROM read_parquet('http://x/y')", id="read_parquet_url"),
    pytest.param("ATTACH 'other.db' AS other", id="attach"),
    pytest.param("INSTALL httpfs", id="install"),
    pytest.param("COPY (SELECT 1 AS a) TO 'out.csv'", id="copy"),
    pytest.param("SET enable_external_access=true", id="set_external_access"),
    pytest.param("SELECT revenue FROM snapshot; SELECT units FROM snapshot", id="two_statements"),
    pytest.param("SELECT customer_email FROM snapshot", id="disallowed_column"),
    pytest.param(
        "SELECT count(*) FROM snapshot a, snapshot b, snapshot c, snapshot d",
        id="runaway_cross_join",
    ),
]


@pytest.mark.parametrize("sql", REJECTED)
def test_rejects(snapshot, sql):
    query = LocalQuery(snapshot, ALLOWED, Settings())
    started = time.monotonic()
    with pytest.raises(QueryRejected):
        query.run(sql, timeout_s=TIMEOUT)
    assert time.monotonic() - started < TIMEOUT + 2


def test_positive_control_allowed_column_works(query):
    rows = query.run("SELECT revenue FROM snapshot LIMIT 5")
    assert len(rows) == 5
    assert set(rows[0]) == {"revenue"}


def test_column_named_only_in_string_literal_is_not_a_reference(query):
    rows = query.run("SELECT count(*) AS n FROM snapshot WHERE region = 'customer_email'")
    assert rows == [{"n": 0}]


def test_source_csv_hash_unchanged(tmp_path, snapshot):
    source = tmp_path / "source.csv"
    pl.read_parquet(snapshot).write_csv(source)
    csv_before = hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot_before = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    query = LocalQuery(snapshot, ALLOWED, Settings())
    query.run("SELECT revenue FROM snapshot LIMIT 5")
    with pytest.raises(QueryRejected):
        query.run("SELECT customer_email FROM snapshot")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == csv_before
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == snapshot_before


def test_injected_valid_select_from_cell_cannot_widen_columns(query):
    injected = "SELECT customer_email FROM snapshot"  # valid SQL that arrived inside a cell value
    with pytest.raises(QueryRejected):
        query.run(injected)
    rows = query.run("SELECT * FROM snapshot LIMIT 3")
    assert set(rows[0]) == ALLOWED


def test_engine_lockdown_in_effect(query):
    row = query.run(
        "SELECT current_setting('enable_external_access') AS external_access, "
        "current_setting('lock_configuration') AS config_locked, "
        "current_setting('autoload_known_extensions') AS autoload, "
        "current_setting('autoinstall_known_extensions') AS autoinstall"
    )[0]
    assert {k: str(v).lower() for k, v in row.items()} == {
        "external_access": "false",
        "config_locked": "true",
        "autoload": "false",
        "autoinstall": "false",
    }
