"""Tests for the read-only CSV extract (plan Task 4, spec 6.2 state 2 and 6.5)."""

import hashlib
import logging
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from weekly_analyst.data import extract as extract_module
from weekly_analyst.core.config import Settings
from weekly_analyst.data.extract import ExtractError, extract
from weekly_analyst.data.metrics_file import load_metrics_file

FIXTURES = Path(__file__).parent / "fixtures"
HEADER = "date,region,product,channel,units,revenue,discount,returned"
ROWS = [
    "2026-11-02,North,Widget,Web,3,29.97,0.10,0",
    "2026-11-03,South,Gadget,Store,5,124.95,0.00,1",
    "2026-11-04,North,Gadget,Web,2,49.90,0.05,0",
]


@pytest.fixture
def metrics():
    return load_metrics_file(FIXTURES / "metrics_sales.yaml")


@pytest.fixture
def dirs(tmp_path):
    source = tmp_path / "source"
    out = tmp_path / "out"
    source.mkdir()
    out.mkdir()
    return source, out


def write_source(directory: Path, text: str, bom: bool = False, name: str = "sales.csv") -> Path:
    data = text.encode("utf-8")
    path = directory / name
    path.write_bytes(b"\xef\xbb\xbf" + data if bom else data)
    return path


def sales_text(header: str = HEADER, rows: list[str] = ROWS) -> str:
    return "\n".join([header, *rows]) + "\n"


def test_source_hash_unchanged_after_extract(dirs, metrics):
    source, out = dirs
    path = write_source(source, sales_text())
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    extract(path, metrics, out, Settings())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_hash_is_sha256_of_file_bytes(dirs, metrics):
    source, out = dirs
    path = write_source(source, sales_text(), bom=True)
    result = extract(path, metrics, out, Settings())
    assert result.snapshot_hash == hashlib.sha256(path.read_bytes()).hexdigest()


def test_parquet_has_only_metric_columns(dirs, metrics):
    source, out = dirs
    header = HEADER + ",customer_email"
    rows = [f"{row},someone@example.com" for row in ROWS]
    path = write_source(source, sales_text(header, rows))
    result = extract(path, metrics, out, Settings())
    df = pl.read_parquet(result.parquet_path)
    assert set(df.columns) == metrics.columns()
    assert "customer_email" not in df.columns


def test_rows_read_matches_rows_in_file_on_success(dirs, metrics):
    source, out = dirs
    path = write_source(source, sales_text())
    result = extract(path, metrics, out, Settings())
    assert result.rows_read == 3
    assert result.rows_in_file == 3
    assert result.parquet_path == out / f"snapshot-{result.snapshot_hash}.parquet"
    assert pl.read_parquet(result.parquet_path).height == 3


def test_success_leaves_only_the_final_file(dirs, metrics):
    source, out = dirs
    path = write_source(source, sales_text())
    result = extract(path, metrics, out, Settings())
    assert [p.name for p in out.iterdir()] == [result.parquet_path.name]


@pytest.mark.parametrize("failure", ["write", "rename"])
def test_parquet_written_atomically(dirs, metrics, monkeypatch, failure):
    source, out = dirs
    path = write_source(source, sales_text())

    def crash(*args, **kwargs):
        raise OSError("simulated crash")

    if failure == "write":
        monkeypatch.setattr(pl.DataFrame, "write_parquet", crash)
    else:
        monkeypatch.setattr(extract_module.os, "replace", crash)

    with pytest.raises(OSError):
        extract(path, metrics, out, Settings())
    assert list(out.iterdir()) == []


@pytest.mark.parametrize(
    ("cap_field", "cap_value", "code", "expected_rows_in_file"),
    [
        ("max_bytes", 10, "too_big", "unknown"),
        ("max_rows", 2, "too_many_rows", "3"),
        ("max_cells", 23, "too_many_cells", "3"),
    ],
)
def test_each_cap_halts_with_code_and_logs_rows_read_vs_in_file(
    dirs, metrics, caplog, cap_field, cap_value, code, expected_rows_in_file
):
    source, out = dirs
    path = write_source(source, sales_text())
    settings = replace(Settings(), **{cap_field: cap_value})
    with caplog.at_level(logging.ERROR, logger="weekly_analyst.data.extract"):
        with pytest.raises(ExtractError) as exc:
            extract(path, metrics, out, settings)
    assert exc.value.code == code
    assert f"code={code}" in caplog.text
    assert "rows_read=" in caplog.text
    assert f"rows_in_file={expected_rows_in_file}" in caplog.text
    assert list(out.iterdir()) == []


def test_bom_is_stripped_and_accepted(dirs, metrics):
    source, out = dirs
    path = write_source(source, sales_text(), bom=True)
    result = extract(path, metrics, out, Settings())
    df = pl.read_parquet(result.parquet_path)
    assert df.height == 3
    assert "date" in df.columns


@pytest.mark.parametrize(
    ("text", "code"),
    [
        (sales_text(header=HEADER.replace(",", ";"), rows=[r.replace(",", ";") for r in ROWS]), "bad_header"),
        (sales_text(header=HEADER.replace("region", "units")), "duplicate_columns"),
        (HEADER + "\n", "empty"),
        ("", "empty"),
        (sales_text(rows=[ROWS[0] + ",extra", *ROWS[1:]]), "bad_header"),
        (sales_text(rows=[ROWS[0], "", *ROWS[1:]]), "bad_header"),
        ("\n" + sales_text(), "bad_header"),
    ],
    ids=["semicolon", "duplicate_columns", "header_only", "empty_file", "ragged_row", "blank_data_line", "blank_header"],
)
def test_review_focus_1_rejections(dirs, metrics, text, code):
    source, out = dirs
    path = write_source(source, text)
    with pytest.raises(ExtractError) as exc:
        extract(path, metrics, out, Settings())
    assert exc.value.code == code
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("bad", ["11/09/2026", "2026-02-30", "2026-11-9", ""])
def test_non_iso_date_rejected(dirs, metrics, bad):
    source, out = dirs
    rows = [f"{bad},North,Widget,Web,3,29.97,0.10,0", *ROWS[1:]]
    path = write_source(source, sales_text(rows=rows))
    with pytest.raises(ExtractError) as exc:
        extract(path, metrics, out, Settings())
    assert exc.value.code == "bad_date"


def test_non_utf8_is_rejected_as_bad_encoding(dirs, metrics):
    source, out = dirs
    path = source / "sales.csv"
    path.write_bytes(sales_text().encode("utf-8").replace(b"North", b"No\xffrth"))
    with pytest.raises(ExtractError) as exc:
        extract(path, metrics, out, Settings())
    assert exc.value.code == "bad_encoding"


def test_excel_workbook_is_rejected_with_a_clear_reason(dirs, metrics):
    source, out = dirs
    for magic in (b"PK\x03\x04", b"\xd0\xcf\x11\xe0"):
        path = source / "sales.csv"
        path.write_bytes(magic + b"junk")
        with pytest.raises(ExtractError) as exc:
            extract(path, metrics, out, Settings())
        assert exc.value.code == "not_csv" and "Excel workbook" in str(exc.value)


def test_metric_column_missing_from_source_is_left_out(dirs, metrics):
    source, out = dirs
    header = "date,region,product,channel,units,revenue,discount"
    rows = [row.rsplit(",", 1)[0] for row in ROWS]
    path = write_source(source, sales_text(header, rows))
    result = extract(path, metrics, out, Settings())
    df = pl.read_parquet(result.parquet_path)
    assert set(df.columns) == metrics.columns() - {"returned"}
