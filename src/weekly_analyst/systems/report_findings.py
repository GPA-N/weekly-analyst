"""Finding sections of the PDF (spec 6.8): chart, attribution rows and data table for one material item."""

import polars as pl

from weekly_analyst.analysis.analyst import attribute_change
from weekly_analyst.analysis.plan import _item_key as plan_item_key
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.items import Item
from weekly_analyst.core.system import Halt
from weekly_analyst.data import sanitize
from weekly_analyst.data.metrics_calc import compute
from weekly_analyst.report.charts import ChartSpec
from weekly_analyst.report.model import AttributionRow, DataTable, Finding, ShownNumber
from weekly_analyst.systems.common import metric_map, period_expr, prev_period, public_item, week

SAMPLE_WEEKS = 8  # chart and data-table history per finding


def finding_for(ctx: RunContext, df: pl.DataFrame, item: Item, text) -> tuple[Finding, DataTable]:
    metric = metric_map(ctx)[item.metric]
    date_col = ctx.metrics.date_column
    segment = dict(item.segment)  # raw: grouping and recompute
    shown = public_item(item)  # what the report prints
    label = segment_label(dict(shown.segment))
    key = plan_item_key(shown)
    level = compute(df, metric, segment, item.period)
    if level is None:
        raise Halt(f"{key}: too few rows to report")
    points = series(ctx, df, metric, segment)
    col = metric.segment_columns[0]
    shares = attribute_change(
        week(df, date_col, prev_period(item.period)), week(df, date_col, item.period), metric, col
    )
    chart = ChartSpec(
        kind="line",
        title=f"{metric.name} by ISO week: {label}",
        x=[week_label for _, week_label, _ in points],
        series={"value": [value for _, _, value in points]},
        unit=metric.name,
        alt_text=f"Line of {metric.name} by ISO week for {label}; the latest week is last.",
    )
    rows = [
        AttributionRow(
            segment=sanitize.label(str(r["segment"])),
            change=ShownNumber(change_text(r["delta"]), metric.name, {col: str(r["segment"])}, item.period),
        )
        for r in shares.iter_rows(named=True)
    ]
    finding = Finding(
        item_key=key,
        supporting=ShownNumber(level_text(level), metric.name, segment, item.period),
        so_what=text.so_what,
        chart=chart,
        attribution=rows,
        interpretation=f"Each row is one {col} segment and its share of the change in {metric.name}.",
    )
    table = DataTable(
        item_key=key,
        columns=["Week", metric.name],
        rows=[
            [week_label, ShownNumber(level_text(value), metric.name, segment, period)]
            for period, week_label, value in points
        ],
    )
    return finding, table


def series(ctx: RunContext, df: pl.DataFrame, metric, segment: dict) -> list[tuple[str, str, float]]:
    periods = df.select(period_expr(ctx.metrics.date_column).unique().sort()).to_series().to_list()
    points = []
    for period in periods[-SAMPLE_WEEKS:]:
        value = compute(df, metric, segment, period)
        if value is not None:
            points.append((period, "W" + period.split("-W", 1)[1], value))
    return points


def level_text(value: float) -> str:
    return f"{value:,.2f}"


def change_text(value: float) -> str:
    return f"{value:+,.2f}"


def segment_label(segment: dict) -> str:
    return ", ".join(f"{key} {value}" for key, value in segment.items()) or "all segments"
