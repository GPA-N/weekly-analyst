"""Draft state: the model fills the fixed text fields, checked against the material items (spec 6.2 step 7)."""

from weekly_analyst.analysis.draft import DraftFailed, DraftFields, get_draft
from weekly_analyst.analysis.plan import _item_key as plan_item_key
from weekly_analyst.core.components import DraftC, EvidenceC, ItemsC, ValidationC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.system import Halt, System
from weekly_analyst.core.world import World
from weekly_analyst.data.metrics_calc import compute
from weekly_analyst.systems.common import frame, metric_map, prev_period, public_items, raw_items


class DraftSystem(System):
    name = "draft"
    requires = (ValidationC, ItemsC, EvidenceC)
    produces = (DraftC,)

    def run(self, world: World, entity: Entity, ctx: RunContext) -> None:
        draft = request_draft(world, entity, ctx, [])
        world.add(entity, DraftC(draft=draft.model_dump(mode="json")))


def request_draft(world: World, entity: Entity, ctx: RunContext, failures: list[str]) -> DraftFields:
    """Ask the model for the draft text. Also used by the check state's redraft rounds."""
    try:
        draft = get_draft(ctx.client, _evidence(world, entity, ctx, failures), ctx.settings)
    except DraftFailed as err:
        raise Halt(f"draft: {err}") from None
    wanted = sorted(plan_item_key(item) for item in public_items(world, entity))
    given = sorted(f.item_key for f in draft.findings)
    if wanted != given:
        raise Halt("draft: findings do not match the material items")
    return draft


def _direction(df, ctx: RunContext, item) -> str:
    """up, down or unchanged for the item's metric against the prior week: words only, no numbers for the model."""
    metric = metric_map(ctx)[item.metric]
    segment = dict(item.segment)
    now = compute(df, metric, segment, item.period)
    before = compute(df, metric, segment, prev_period(item.period))
    if now is None or before is None:
        return "unknown"
    return "up" if now > before else "down" if now < before else "unchanged"


def _largest_segments(result: dict) -> list[str]:
    """Names of the segments that moved most, from an attribution result. Labels only, never numbers."""
    rows = result.get("segments") or []
    ranked = sorted(rows, key=lambda r: abs(r.get("delta") or 0), reverse=True)
    return [str(r["segment"]) for r in ranked[:2]]


def _evidence(world: World, entity: Entity, ctx: RunContext, failures: list[str]) -> dict:
    """What the model sees: words and labels only. Every number in the report comes from code, so none goes in."""
    validation = world.get(entity, ValidationC)
    df = frame(world, entity, ctx)
    items = [
        {
            "item_key": plan_item_key(public),
            "metric": public.metric,
            "segment": dict(public.segment),
            "direction": _direction(df, ctx, raw),
        }
        for public, raw in zip(public_items(world, entity), raw_items(world, entity))
    ]
    investigation = [
        {
            "tool": r["tool"],
            "item_key": r["item_key"],
            "completed": "error" not in r,
            "largest_segments": _largest_segments(r.get("result", {})) if r["tool"] == "attribute_change" else [],
        }
        for r in world.get(entity, EvidenceC).results
    ]
    return {
        "cold_start": validation.cold_start,
        "items": items,
        "validation": [{"kind": f["kind"], "metric": f.get("metric")} for f in validation.findings],
        "investigation": investigation,
        "check_failures": failures,
    }
