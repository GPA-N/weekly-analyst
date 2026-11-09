"""Helpers the systems share: ISO-week arithmetic, the snapshot frame, and items as the model sees them."""

from dataclasses import replace
from datetime import date, timedelta

import polars as pl

from weekly_analyst.core.components import ItemsC, SnapshotC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.items import Item
from weekly_analyst.core.world import World
from weekly_analyst.data import sanitize


def period_expr(date_column: str) -> pl.Expr:
    return pl.col(date_column).cast(pl.Date).dt.strftime("%G-W%V")


def split(df: pl.DataFrame, date_column: str) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    """History (weeks before the latest) and current (the latest ISO week) of one snapshot."""
    period = period_expr(date_column)
    latest = df.select(period.max()).item()
    return df.filter(period < latest), df.filter(period == latest), latest


def period_count(df: pl.DataFrame, date_column: str) -> int:
    if df.height == 0:
        return 0
    return int(df.select(period_expr(date_column).n_unique()).item())


def rows_in_last_period(df: pl.DataFrame, date_column: str) -> int:
    period = period_expr(date_column)
    last = df.select(period.max()).item()
    return df.filter(period == last).height


def prev_period(period: str) -> str:
    year, week = (int(part) for part in period.split("-W", 1))
    iso = (date.fromisocalendar(year, week, 1) - timedelta(weeks=1)).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def week(df: pl.DataFrame, date_column: str, period: str) -> pl.DataFrame:
    return df.filter(period_expr(date_column) == period)


def frame(world: World, entity: Entity, ctx: RunContext) -> pl.DataFrame:
    """The run's Parquet snapshot, read once per attempt."""
    if "frame" not in ctx.cache:
        ctx.cache["frame"] = pl.read_parquet(world.get(entity, SnapshotC).parquet, memory_map=False)
    return ctx.cache["frame"]


def metric_map(ctx: RunContext) -> dict:
    if "by_name" not in ctx.cache:
        ctx.cache["by_name"] = {m.name: m for m in ctx.metrics.metrics}
    return ctx.cache["by_name"]


def raw_items(world: World, entity: Entity) -> list[Item]:
    return world.get(entity, ItemsC).selected


def public_item(item: Item) -> Item:
    """The item with segment labels sanitized, for the model and the report. Grouping and checks use the raw item."""
    return replace(item, segment=tuple((col, sanitize.label(value)) for col, value in item.segment))


def public_items(world: World, entity: Entity) -> list[Item]:
    return [public_item(item) for item in raw_items(world, entity)]
