"""Read-only CSV extract to a Parquet snapshot (spec 6.2 state 2, 6.5)."""

import codecs
import csv
import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import MetricsFile

log = logging.getLogger(__name__)

ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
CHUNK_BYTES = 1 << 20


class ExtractError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class ExtractResult:
    snapshot_hash: str
    parquet_path: Path
    rows_read: int
    rows_in_file: int


def _halt(code: str, detail: str, rows_read: int, rows_in_file: int | str) -> ExtractError:
    log.error(
        "extract halted code=%s rows_read=%s rows_in_file=%s detail=%s",
        code,
        rows_read,
        rows_in_file,
        detail,
    )
    return ExtractError(code, detail)


def _hash_and_check_utf8(csv_path: Path) -> str:
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")()
    with open(csv_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(chunk)
            try:
                decoder.decode(chunk)
            except UnicodeDecodeError:
                raise _halt("bad_encoding", "file is not valid UTF-8", 0, "unknown") from None
    try:
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        raise _halt("bad_encoding", "file is not valid UTF-8", 0, "unknown") from None
    return digest.hexdigest()


def _is_iso_date(value: str) -> bool:
    if not ISO_DATE.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _scan_source(csv_path: Path, metrics: MetricsFile) -> tuple[list[str], int, int | None]:
    """Stream the file once: check the header, count data rows, find the first bad date."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return [], 0, None
        if not header:
            raise _halt("bad_header", "header row is blank", 0, "unknown")
        if any(";" in cell for cell in header):
            raise _halt("bad_header", "header uses a semicolon delimiter; use commas", 0, "unknown")
        if len(set(header)) != len(header):
            raise _halt("duplicate_columns", "header has duplicate column names", 0, "unknown")
        if metrics.date_column not in header:
            raise _halt("bad_header", f"header has no date column {metrics.date_column!r}", 0, "unknown")
        date_index = header.index(metrics.date_column)
        ncols = len(header)
        rows = 0
        first_bad_date: int | None = None
        for record in reader:
            if len(record) != ncols:
                raise _halt(
                    "bad_header",
                    f"data row {rows + 1} has {len(record)} fields, header has {ncols}",
                    0,
                    "unknown",
                )
            rows += 1
            if first_bad_date is None and not _is_iso_date(record[date_index]):
                first_bad_date = rows
    return header, rows, first_bad_date


def _write_atomic(df: pl.DataFrame, final: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=final.parent, prefix=".extract-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            df.write_parquet(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def extract(csv_path: Path, metrics: MetricsFile, out_dir: Path, settings: Settings) -> ExtractResult:
    size = csv_path.stat().st_size
    if size > settings.max_bytes:
        raise _halt("too_big", f"file is {size} bytes, cap is {settings.max_bytes}", 0, "unknown")

    snapshot_hash = _hash_and_check_utf8(csv_path)
    header, rows, first_bad_date = _scan_source(csv_path, metrics)
    if rows == 0:
        raise _halt("empty", "no data rows after the header", 0, 0)
    if rows > settings.max_rows:
        raise _halt("too_many_rows", f"{rows} rows, cap is {settings.max_rows}", 0, rows)
    cells = rows * len(header)
    if cells > settings.max_cells:
        raise _halt("too_many_cells", f"{cells} cells, cap is {settings.max_cells}", 0, rows)
    if first_bad_date is not None:
        raise _halt("bad_date", f"data row {first_bad_date} is not an ISO date (YYYY-MM-DD)", 0, rows)

    metric_columns = metrics.columns()
    present = [name for name in header if name in metric_columns]
    df = pl.read_csv(
        csv_path,
        columns=present,
        schema_overrides={metrics.date_column: pl.Utf8},
        infer_schema_length=None,
    )
    df = df.with_columns(pl.col(metrics.date_column).str.to_date("%Y-%m-%d", strict=True))

    final = out_dir / f"snapshot-{snapshot_hash}.parquet"
    _write_atomic(df, final)
    return ExtractResult(
        snapshot_hash=snapshot_hash,
        parquet_path=final,
        rows_read=df.height,
        rows_in_file=rows,
    )
