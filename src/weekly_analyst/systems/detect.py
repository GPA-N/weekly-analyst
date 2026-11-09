"""Detect state: score and rank candidate changes, then select alerts within the weekly budget (spec 6.2, 6.6)."""

import math
from dataclasses import replace

from weekly_analyst.core.components import ItemsC, SnapshotC, ValidationC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.items import Item
from weekly_analyst.core.system import System
from weekly_analyst.core.world import World
from weekly_analyst.data.metrics_calc import compute
from weekly_analyst.detection import remaining_budget, score_items, select_alerts
from weekly_analyst.systems.common import frame, metric_map, prev_period, split


class DetectSystem(System):
    name = "detect"
    requires = (ValidationC, SnapshotC)
    produces = (ItemsC,)

    def run(self, world: World, entity: Entity, ctx: RunContext) -> None:
        validation = world.get(entity, ValidationC)
        df = frame(world, entity, ctx)
        history, current, period = split(df, ctx.metrics.date_column)
        selected: list[Item] = []
        if not validation.cold_start:
            items = score_items(history, current, ctx.metrics, ctx.settings)
            prev = prev_period(period)
            priced = [
                replace(item, detail={**item.detail, "abs_change": _abs_change(ctx, df, item, prev)})
                for item in items
            ]
            min_change = {m.name: m.min_change for m in ctx.metrics.metrics}
            budget = remaining_budget(ctx.store, period, ctx.settings)
            selected = select_alerts(priced, budget, min_change, ctx.store.reported_keys())
        schema_change = any(f["kind"] == "schema_change" for f in validation.findings)
        world.add(
            entity,
            ItemsC(
                selected=selected,
                material=bool(selected) or schema_change,
                alerts_before=ctx.store.alerts_sent_in_week(period),
            ),
        )

    def ends_run(self, world: World, entity: Entity) -> bool:
        """A quiet snapshot (no material item, no schema change) ends after detect; nothing is sent."""
        return not world.get(entity, ItemsC).material


def _abs_change(ctx: RunContext, df, item: Item, prev: str) -> float:
    metric = metric_map(ctx)[item.metric]
    segment = dict(item.segment)
    now = compute(df, metric, segment, item.period)
    before = compute(df, metric, segment, prev)
    if now is None or before is None:
        return math.nan  # a NaN change fails the minimum-change gate
    return abs(now - before)
