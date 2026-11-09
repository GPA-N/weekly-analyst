"""Validate state: schema drift, nulls, duplicates and row-count jumps, in deterministic polars (spec 6.2 step 3)."""

from weekly_analyst.core.components import SnapshotC, ValidationC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.system import System
from weekly_analyst.core.world import World
from weekly_analyst.data.validate import validate
from weekly_analyst.detection import insufficient_history
from weekly_analyst.systems.common import frame, period_count, rows_in_last_period, split


class ValidateSystem(System):
    name = "validate"
    requires = (SnapshotC,)
    produces = (ValidationC,)

    def run(self, world: World, entity: Entity, ctx: RunContext) -> None:
        col = ctx.metrics.date_column
        history, current, period = split(frame(world, entity, ctx), col)
        n_periods = period_count(history, col)
        prior_schema = {name: str(dtype) for name, dtype in history.schema.items()} if n_periods else None
        prior_rows = rows_in_last_period(history, col) if n_periods else None
        result = validate(current, ctx.metrics, prior_schema, prior_rows)
        world.add(
            entity,
            ValidationC(
                period=period,
                n_periods=n_periods,
                cold_start=insufficient_history(n_periods, ctx.settings),
                findings=[{"kind": f.kind, "metric": f.metric, "detail": f.detail} for f in result.findings],
            ),
        )
