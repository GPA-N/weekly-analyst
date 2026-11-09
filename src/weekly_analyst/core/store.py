"""SQLite state store: weekly alert counts, reported items, run states (spec 6.6)."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def iso_week_of(ts: datetime) -> str:
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    item_key TEXT NOT NULL,
    iso_week TEXT NOT NULL,
    run_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


class StateStore:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        with self._transaction() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def alerts_sent_in_week(self, iso_week: str) -> int:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE iso_week = ?", (iso_week,)
            ).fetchone()
        return int(row[0])

    def record_alert(self, item_key: str, iso_week: str, run_id: str) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO alerts (item_key, iso_week, run_id) VALUES (?, ?, ?)",
                (item_key, iso_week, run_id),
            )

    def reported_keys(self) -> set[str]:
        with self._transaction() as conn:
            rows = conn.execute("SELECT DISTINCT item_key FROM alerts").fetchall()
        return {row[0] for row in rows}

    def save_state(self, run_id: str, state: str, payload: dict) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, state, payload) VALUES (?, ?, ?)",
                (run_id, state, json.dumps(payload, sort_keys=True)),
            )

    def load_run(self, run_id: str) -> dict | None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT state, payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return {"run_id": run_id, "state": row[0], "payload": json.loads(row[1])}
