"""System base class and the control signals a system may raise (spec 6.2, architecture.md)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weekly_analyst.core.context import RunContext
    from weekly_analyst.core.entity import Entity
    from weekly_analyst.core.world import World


class Halt(Exception):
    """A system stops the run. The reason goes into the outcome."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Restart(Exception):
    """The source file changed mid-run. The run is abandoned and a new run starts from extract."""


class System(ABC):
    """One state of a run.

    ``requires`` lists the components that must exist before ``run``; ``produces`` lists the ones ``run``
    adds. A system whose produces all exist is skipped on resume. ``state_row`` is False when another
    writer owns the store row (Outbox writes "outbox" itself) or the state leaves none (remember).
    """

    name: str
    requires: tuple[type, ...] = ()
    produces: tuple[type, ...] = ()
    state_row: bool = True

    @abstractmethod
    def run(self, world: World, entity: Entity, ctx: RunContext) -> None:
        """Do the state's work and add its components to the world."""

    def ends_run(self, world: World, entity: Entity) -> bool:
        """True when the run stops after this state, for example a quiet snapshot after detect."""
        return False
