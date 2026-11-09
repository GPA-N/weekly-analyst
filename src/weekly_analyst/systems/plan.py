"""Plan state: the model emits a schema-valid plan over the material items; no model call when there are none (spec 6.2 step 5)."""

from weekly_analyst.analysis.plan import PlanFailed, get_plan
from weekly_analyst.analysis.plan import _item_key as plan_item_key
from weekly_analyst.core.components import ItemsC, PlanC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.system import Halt, System
from weekly_analyst.core.world import World
from weekly_analyst.systems.common import public_items


class PlanSystem(System):
    name = "plan"
    requires = (ItemsC,)
    produces = (PlanC,)

    def run(self, world: World, entity: Entity, ctx: RunContext) -> None:
        items = public_items(world, entity)
        if not items:  # schema-change-only brief: no model call
            world.add(entity, PlanC(steps=[]))
            return
        try:
            plan = get_plan(ctx.client, items, ctx.settings)
        except PlanFailed as err:
            raise Halt(f"plan: {err}") from None
        allowed = {plan_item_key(item) for item in items}
        if any(step.item_key not in allowed for step in plan.steps):
            raise Halt("plan: names an item that is not in this run")
        world.add(entity, PlanC(steps=plan.model_dump(mode="json")["steps"]))
