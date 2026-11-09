"""Scheduler: drives the fixed order of systems over one run (spec 6.2, architecture.md).

Before each state it checks the wall time and the source hash. It runs a system only when its produced
components are missing, saves the world and the store state after each state, and stops on a halt or
when a system says the run ends. The model never chooses the order.
"""

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from weekly_analyst.core.components import HaltC, SourceC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.system import Halt, Restart, System
from weekly_analyst.core.world import World

log = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Scheduler:
    def __init__(self, systems: Sequence[System]) -> None:
        self.systems = tuple(systems)
        self._order = {system.name: index for index, system in enumerate(self.systems)}

    def drive(self, world: World, entity: Entity, ctx: RunContext) -> None:
        for system in self.systems:
            try:
                self._guard(world, entity, ctx)
                if not _done(world, entity, system):
                    _check_requires(world, entity, system)
                    system.run(world, entity, ctx)
            except Halt as halt:
                log.warning("run %s halted at %s: %s", entity.run_id, system.name, halt.reason)
                world.add(entity, HaltC(reason=halt.reason))
                return
            self._checkpoint(world, entity, ctx, system)
            if system.ends_run(world, entity):
                return

    def _guard(self, world: World, entity: Entity, ctx: RunContext) -> None:
        """Wall-time limit, and restart if the source file changed since this run began."""
        ctx.check_time()
        source = world.get(entity, SourceC)
        try:
            digest = file_sha256(Path(source.csv_path))
        except OSError:
            raise Halt("source file is missing or unreadable") from None
        if digest != source.snapshot_hash:
            raise Restart()

    def _checkpoint(self, world: World, entity: Entity, ctx: RunContext, system: System) -> None:
        world.save(entity)
        if not system.state_row:
            return
        # A resumed run replays finished states; the store row only moves forward, never back.
        row = ctx.store.load_run(entity.run_id)
        if row is None or self._order[row["state"]] < self._order[system.name]:
            snapshot_hash = world.get(entity, SourceC).snapshot_hash
            ctx.store.save_state(entity.run_id, system.name, {"snapshot_hash": snapshot_hash})


def _done(world: World, entity: Entity, system: System) -> bool:
    """A system with no outputs (remember) always runs; its effects are idempotent."""
    return bool(system.produces) and all(world.has(entity, kind) for kind in system.produces)


def _check_requires(world: World, entity: Entity, system: System) -> None:
    missing = [kind.__name__ for kind in system.requires if not world.has(entity, kind)]
    if missing:
        raise RuntimeError(f"{system.name} needs {', '.join(missing)}")
