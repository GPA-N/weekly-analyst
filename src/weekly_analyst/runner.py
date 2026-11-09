"""Run entry points, run lock and outcome (spec 6.2, 6.5, 6.6, 6.7; ticket 022). Thin facade over core/ and systems/.

A run works in ``work_dir/runs/<run_id>/``: its progress file holds the components of each finished state,
and the Parquet snapshot sits beside it. The states are the systems in SYSTEMS, driven by the scheduler.
A finished or halted run deletes its folder. A source file that changes mid-run starts a fresh run at extract.
"""

import logging
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from weekly_analyst.analysis.model import ModelClient
from weekly_analyst.core.components import HaltC, ItemsC, OutboxC, SourceC
from weekly_analyst.core.config import Settings
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.scheduler import Scheduler, file_sha256
from weekly_analyst.core.store import StateStore
from weekly_analyst.core.system import Restart
from weekly_analyst.core.world import PROGRESS_FILE, World
from weekly_analyst.data.metrics_file import load_metrics_file
from weekly_analyst.systems import SYSTEMS

log = logging.getLogger(__name__)

STATES: tuple[str, ...] = tuple(system.name for system in SYSTEMS)
SCHEDULER = Scheduler(SYSTEMS)
RUNS_DIR = "runs"
LOCK_FILE = "run.lock"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class RunLocked(RuntimeError):
    """Another run holds the run lock. Nothing was started."""


@dataclass(frozen=True)
class RunConfig:
    metrics_path: Path
    work_dir: Path
    out_dir: Path
    delivery_mode: Literal["approval", "auto"]
    recipients: list[str]
    paper: Literal["A4", "Letter"]
    settings: Settings = field(default_factory=Settings)


@dataclass(frozen=True)
class RunOutcome:
    kind: Literal["silent", "brief", "halted"]
    run_id: str
    report_path: Path | None
    reasons: list[str]


def run(csv_path: Path, cfg: RunConfig, client: ModelClient, store: StateStore) -> RunOutcome:
    """Run the pipeline for one snapshot. An unfinished run with the same hash is resumed, never refused."""
    _refuse_auto(cfg)
    with _run_lock(cfg.work_dir):
        digest = file_sha256(Path(csv_path))
        run_id = _same_hash_run(cfg, digest) or _start_run(cfg, Path(csv_path), digest)
        return _drive(run_id, cfg, client, store)


def resume(run_id: str, cfg: RunConfig, client: ModelClient, store: StateStore) -> RunOutcome:
    """Continue an unfinished run from its progress file."""
    _refuse_auto(cfg)
    if not _RUN_ID.match(run_id):
        raise ValueError(f"run id {run_id!r} is not a plain identifier")
    with _run_lock(cfg.work_dir):
        if not (_runs_dir(cfg) / run_id / PROGRESS_FILE).is_file():
            raise ValueError(f"no unfinished run {run_id!r}")
        return _drive(run_id, cfg, client, store)


def _drive(run_id: str, cfg: RunConfig, client: ModelClient, store: StateStore) -> RunOutcome:
    started = time.monotonic()
    while True:
        world = World(_runs_dir(cfg))
        entity = world.load(run_id)
        ctx = RunContext(
            settings=cfg.settings,
            metrics=load_metrics_file(cfg.metrics_path),
            client=client,
            store=store,
            out_dir=cfg.out_dir,
            paper=cfg.paper,
            recipients=cfg.recipients,
            started=started,
        )
        try:
            SCHEDULER.drive(world, entity, ctx)
        except Restart:
            run_id = _restart(cfg, run_id)
            continue
        outcome = _outcome(world, entity)
        shutil.rmtree(world.run_dir(entity))  # run Parquet and drafts are deleted at finish
        return outcome


def _outcome(world: World, entity: Entity) -> RunOutcome:
    if world.has(entity, HaltC):
        return RunOutcome("halted", entity.run_id, None, [world.get(entity, HaltC).reason])
    if not world.get(entity, ItemsC).material:
        return RunOutcome("silent", entity.run_id, None, [])
    return RunOutcome("brief", entity.run_id, Path(world.get(entity, OutboxC).report), [])


def _restart(cfg: RunConfig, run_id: str) -> str:
    world = World(_runs_dir(cfg))
    entity = world.load(run_id)
    if world.has(entity, OutboxC):  # a brief for superseded data is withdrawn; approve refuses it anyway
        Path(world.get(entity, OutboxC).report).unlink(missing_ok=True)
    csv_path = Path(world.get(entity, SourceC).csv_path)
    shutil.rmtree(world.run_dir(entity))
    log.info("source changed mid-run; run %s abandoned, starting a fresh run", run_id)
    return _start_run(cfg, csv_path, file_sha256(csv_path))


def _start_run(cfg: RunConfig, csv_path: Path, digest: str) -> str:
    run_id = f"run-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    world = World(_runs_dir(cfg))
    entity = Entity(run_id)
    world.run_dir(entity).mkdir(parents=True)
    world.add(entity, SourceC(csv_path=str(csv_path.resolve()), snapshot_hash=digest))
    world.save(entity)
    log.info("run %s started", run_id)
    return run_id


def _same_hash_run(cfg: RunConfig, digest: str) -> str | None:
    world = World(_runs_dir(cfg))
    for path in sorted(world.runs_dir.glob(f"*/{PROGRESS_FILE}")):
        entity = world.load(path.parent.name)
        if world.get(entity, SourceC).snapshot_hash == digest:
            return entity.run_id
    return None


def _refuse_auto(cfg: RunConfig) -> None:
    if cfg.delivery_mode != "approval":
        raise NotImplementedError("auto-send is not built in ticket 022; use delivery_mode 'approval'")


def _runs_dir(cfg: RunConfig) -> Path:
    return cfg.work_dir / RUNS_DIR


@contextmanager
def _run_lock(work_dir: Path) -> Iterator[None]:
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / LOCK_FILE, "a+b") as fh:
        if not _try_lock(fh):
            raise RunLocked("another run holds the run lock; nothing started")
        try:
            yield
        finally:
            _unlock(fh)


def _try_lock(fh) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(fh) -> None:
    if os.name == "nt":
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
