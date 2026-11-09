"""Remember state: record the sent alerts in the store so they are not repeated (spec 6.2 step 10, 6.6)."""

from weekly_analyst.core.components import ItemsC, OutboxC, ValidationC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.system import Halt, System
from weekly_analyst.core.world import World
from weekly_analyst.detection import item_key


class RememberSystem(System):
    name = "remember"
    requires = (ItemsC, ValidationC, OutboxC)
    produces = ()  # no output component: it always runs, and a resumed run re-counts its alerts below
    state_row = False  # the outbox row must stay for approval; remember's effects live in the alerts table

    def run(self, world: World, entity: Entity, ctx: RunContext) -> None:
        recorded_items = world.get(entity, ItemsC)
        items = recorded_items.selected
        period = world.get(entity, ValidationC).period
        # Count what this run already recorded, so a resumed run never records an alert twice.
        recorded = ctx.store.alerts_sent_in_week(period) - recorded_items.alerts_before
        if not 0 <= recorded <= len(items):
            raise Halt("remember: alert count does not match this run")
        for item in items[recorded:]:
            ctx.store.record_alert(item_key(item), period, entity.run_id)
