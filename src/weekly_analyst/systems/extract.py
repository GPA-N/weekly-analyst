"""Extract state: read the CSV into a Parquet snapshot, code only (spec 6.2 step 2)."""

from pathlib import Path

from weekly_analyst.core.components import SnapshotC, SourceC
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.system import Halt, Restart, System
from weekly_analyst.core.world import World
from weekly_analyst.data.extract import ExtractError, extract


class ExtractSystem(System):
    name = "extract"
    requires = (SourceC,)
    produces = (SnapshotC,)

    def run(self, world: World, entity: Entity, ctx: RunContext) -> None:
        source = world.get(entity, SourceC)
        try:
            result = extract(Path(source.csv_path), ctx.metrics, world.run_dir(entity), ctx.settings)
        except ExtractError as err:
            raise Halt(f"extract: {err}") from None
        if result.snapshot_hash != source.snapshot_hash:
            raise Restart()  # the file changed between the guard and the read
        world.add(
            entity,
            SnapshotC(
                parquet=str(result.parquet_path),
                rows_read=result.rows_read,
                rows_in_file=result.rows_in_file,
            ),
        )
