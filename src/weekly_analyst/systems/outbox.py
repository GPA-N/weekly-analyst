"""Outbox state: write the checked brief to the outbox, stamped with the snapshot hash (spec 6.2 step 9, 6.7)."""

from pathlib import Path

from weekly_analyst.core.components import OutboxC, ReportC, SourceC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.system import System
from weekly_analyst.core.world import World
from weekly_analyst.delivery.outbox import Outbox


class OutboxSystem(System):
    name = "outbox"
    requires = (ReportC, SourceC)
    produces = (OutboxC,)
    state_row = False  # Outbox.put writes the "outbox" store row itself; approve() reads that row

    def run(self, world: World, entity: Entity, ctx: RunContext) -> None:
        report = world.get(entity, ReportC)
        snapshot_hash = world.get(entity, SourceC).snapshot_hash
        Outbox(ctx.out_dir, ctx.store, ctx.settings).put(
            entity.run_id, Path(report.pdf), snapshot_hash, ctx.metrics.checksum, report=report.to_json()
        )
        world.add(entity, OutboxC(report=str(ctx.out_dir / f"{entity.run_id}.pdf")))
