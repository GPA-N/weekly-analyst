"""Confined DuckDB query tool over a run's Parquet snapshot (spec 6.4, AC-004)."""

import hashlib
import json
import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb

from weekly_analyst.core.config import Settings

logger = logging.getLogger(__name__)

TABLE_NAME = "snapshot"


class QueryRejected(Exception):
    """A statement was refused, stopped on timeout, or failed in the engine."""


def _column_references(node: Any) -> Iterator[list[str]]:
    if isinstance(node, dict):
        if node.get("type") == "COLUMN_REF":
            yield node["column_names"]
        for value in node.values():
            yield from _column_references(value)
    elif isinstance(node, list):
        for item in node:
            yield from _column_references(item)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class LocalQuery:
    def __init__(self, parquet_path: Path, allowed_columns: set[str], settings: Settings) -> None:
        self._allowed = {name.casefold() for name in allowed_columns}
        self._con = duckdb.connect()
        self._con.execute(f"SET memory_limit = '{settings.duckdb_memory}'")
        self._con.execute(f"SET threads = {settings.duckdb_threads}")
        # A table, not a view: a view over read_parquet re-binds on every query and
        # fails once external access is disabled below.
        columns = ", ".join(_quote(name) for name in sorted(allowed_columns))
        self._con.execute(
            f"CREATE TABLE {TABLE_NAME} AS SELECT {columns} FROM read_parquet(?)",
            [str(parquet_path)],
        )
        self._con.execute("SET enable_external_access = false")
        # Autoload and autoinstall must be disabled before the lock: a locked
        # configuration rejects any later SET.
        self._con.execute("SET autoload_known_extensions = false")
        self._con.execute("SET autoinstall_known_extensions = false")
        self._con.execute("SET lock_configuration = true")
        logger.info(
            "local_query limits memory=%s threads=%d external_access=off config=locked",
            settings.duckdb_memory,
            settings.duckdb_threads,
        )

    def run(self, sql: str, timeout_s: float = 10.0) -> list[dict]:
        digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]
        try:
            self._check(sql)
            names, rows = self._execute(sql, timeout_s)
        except QueryRejected:
            logger.info("local_query rejected hash=%s", digest)
            raise
        logger.info("local_query ok hash=%s rows=%d", digest, len(rows))
        return [dict(zip(names, row)) for row in rows]

    def _check(self, sql: str) -> None:
        try:
            parsed = json.loads(
                self._con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
            )
        except duckdb.Error:
            raise QueryRejected("statement could not be parsed") from None
        statements = [] if parsed.get("error") else parsed["statements"]
        if len(statements) != 1 or statements[0]["node"]["type"] != "SELECT_NODE":
            raise QueryRejected("only a single SELECT statement is allowed")
        for names in _column_references(statements[0]):
            if len(names) > 2 or names[-1].casefold() not in self._allowed:
                raise QueryRejected(f"column {'.'.join(names)!r} is not in the metrics file")

    def _execute(self, sql: str, timeout_s: float) -> tuple[list[str], list[tuple]]:
        timed_out = threading.Event()

        def interrupt() -> None:
            timed_out.set()
            self._con.interrupt()

        timer = threading.Timer(timeout_s, interrupt)
        timer.start()
        try:
            cursor = self._con.execute(sql)
            names = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
        except duckdb.Error:
            if timed_out.is_set():
                raise QueryRejected(f"query stopped after {timeout_s} s") from None
            raise QueryRejected("the engine rejected the query") from None
        finally:
            timer.cancel()
        return names, rows
