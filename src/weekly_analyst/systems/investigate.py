"""Investigate state: run the plan's tool calls within the budget, from the state's allowlist (spec 6.2 step 6)."""

from weekly_analyst.analysis.plan import STATE_ALLOWLIST
from weekly_analyst.core.components import EvidenceC, PlanC, SnapshotC, ValidationC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.system import Halt, System
from weekly_analyst.core.world import World
from weekly_analyst.data.query import QueryRejected
from weekly_analyst.systems.tools import ToolRunner


class InvestigateSystem(System):
    name = "investigate"
    requires = (PlanC, SnapshotC, ValidationC)
    produces = (EvidenceC,)

    def run(self, world: World, entity: Entity, ctx: RunContext) -> None:
        steps = world.get(entity, PlanC).steps
        budget = ctx.settings.investigate_calls
        if len(steps) > budget:
            raise Halt(f"investigate: plan asks for {len(steps)} tool calls, budget is {budget}")
        tools = ToolRunner(world, entity, ctx)
        results = []
        failures: list[str] = []
        for step in steps:
            ctx.check_time()
            tool = step["tool"]
            if tool not in STATE_ALLOWLIST["investigate"]:
                raise Halt(f"investigate: tool {tool!r} is not allowed in this state")
            try:
                result = tools.call_tool(tool, step["args"])
            except (QueryRejected, KeyError, TypeError, ValueError) as err:
                # One failed check does not sink the report: it is recorded, and the report says so.
                failures.append(f"{tool}: {err}")
                results.append({"tool": tool, "item_key": step["item_key"], "error": type(err).__name__})
                continue
            results.append({"tool": tool, "item_key": step["item_key"], "result": result})
        if steps and len(failures) == len(steps):
            raise Halt(f"tool error: {failures[0]}")
        world.add(entity, EvidenceC(results=results))
