"""Cross-check of metric engines on random slices (ticket 008, spec 6.3).

Arms: polars (metrics_calc if present, else inline polars), duckdb, pandas and
recompute. pandas is imported only in this file.
"""

import math
import random
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import load_metrics_file
from weekly_analyst.data.recompute import recompute
from weekly_analyst.sim.generate import generate

try:
    from weekly_analyst.data.metrics_calc import compute as metrics_calc_compute
except ImportError:
    metrics_calc_compute = None

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "metrics_sales.yaml"
SEGMENT_COLUMNS = ("region", "product", "channel")
N_SLICES = 100
REL_TOL = 1e-9
MIN_ROWS = Settings().min_rows_per_segment


def _year_week(period):
    year, week = period.split("-W")
    return int(year), int(week)


def _polars_inline(snapshot, m, segment, period):
    year, week = _year_week(period)
    df = pl.read_parquet(snapshot).with_columns(pl.col("date").cast(pl.Date))
    mask = (pl.col("date").dt.iso_year() == year) & (pl.col("date").dt.week() == week)
    for key, value in (segment or {}).items():
        mask = mask & (pl.col(key) == value)
    sub = df.filter(mask)
    if sub.height < MIN_ROWS:
        return None
    agg = {"sum": pl.col(m.column).sum(), "mean": pl.col(m.column).mean(), "count": pl.col(m.column).count()}
    value = sub.select(agg[m.agg]).item()
    return None if value is None else float(value)


def _polars_arm(snapshot, m, segment, period):
    if metrics_calc_compute is None:
        return _polars_inline(snapshot, m, segment, period)
    df = pl.read_parquet(snapshot).with_columns(pl.col("date").cast(pl.Date))
    return metrics_calc_compute(df, m, segment, period)


def _duckdb_arm(snapshot, m, segment, period):
    year, week = _year_week(period)
    sql_agg = {"sum": "SUM", "mean": "AVG", "count": "COUNT"}[m.agg]
    where = ["isoyear(date) = ?", "week(date) = ?"]
    params = [str(snapshot), year, week]
    for key, value in (segment or {}).items():
        where.append(f'"{key}" = ?')
        params.append(value)
    sql = (
        f'SELECT COUNT(*), {sql_agg}("{m.column}") '
        f"FROM read_parquet(?) WHERE {' AND '.join(where)}"
    )
    rows, value = duckdb.connect().execute(sql, params).fetchone()
    if rows < MIN_ROWS or value is None:
        return None
    return float(value)


def _pandas_arm(snapshot, m, segment, period):
    year, week = _year_week(period)
    df = pd.read_parquet(snapshot)
    iso = pd.to_datetime(df["date"]).dt.isocalendar()
    mask = (iso["year"] == year) & (iso["week"] == week)
    for key, value in (segment or {}).items():
        mask &= df[key] == value
    sub = df.loc[mask, m.column]
    if len(sub) < MIN_ROWS:
        return None
    value = {"sum": sub.sum, "mean": sub.mean, "count": sub.count}[m.agg]()
    return None if pd.isna(value) else float(value)


def _snapshot(tmp_path):
    out_dir = tmp_path / "sim"
    out_dir.mkdir()
    result = generate(seed=7, weeks=8, events=[], out_dir=out_dir)
    string_cols = {c: pa.string() for c in ("date", "region", "product", "channel")}
    tables = [
        pacsv.read_csv(f, convert_options=pacsv.ConvertOptions(column_types=string_cols))
        for f in result.files
    ]
    table = pa.concat_tables(tables)
    date_idx = table.schema.get_field_index("date")
    table = table.set_column(date_idx, "date", pc.cast(table["date"], pa.date32()))
    snapshot = tmp_path / "snapshot.parquet"
    pq.write_table(table, snapshot)
    return snapshot, table


def _periods(table):
    out = set()
    for d in table["date"].to_pylist():
        iso = d.isocalendar()
        out.add(f"{iso.year}-W{iso.week:02d}")
    return sorted(out)


def _agree(results):
    values = list(results.values())
    if all(v is None for v in values):
        return True
    if any(v is None for v in values):
        return False
    ref = results["recompute"]
    return all(math.isclose(v, ref, rel_tol=REL_TOL) for v in values)


def test_four_way_crosscheck_on_100_random_slices(tmp_path):
    snapshot, table = _snapshot(tmp_path)
    metrics = load_metrics_file(FIXTURE).metrics
    periods = _periods(table)
    seg_values = {c: sorted(set(table[c].to_pylist())) for c in SEGMENT_COLUMNS}
    rng = random.Random(20261109)

    with_values = 0
    for _ in range(N_SLICES):
        m = rng.choice(metrics)
        period = rng.choice(periods)
        k = rng.randint(0, len(SEGMENT_COLUMNS))
        cols = rng.sample(SEGMENT_COLUMNS, k)
        segment = {c: rng.choice(seg_values[c]) for c in cols} or None

        results = {
            "polars": _polars_arm(snapshot, m, segment, period),
            "duckdb": _duckdb_arm(snapshot, m, segment, period),
            "pandas": _pandas_arm(snapshot, m, segment, period),
            "recompute": recompute(snapshot, m, segment, period),
        }
        assert _agree(results), (m.name, segment, period, results)
        if results["recompute"] is not None:
            with_values += 1

    assert with_values >= 30, f"only {with_values} slices had enough rows to compare values"
