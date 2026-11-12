"""Scheduler tests: order, resume skip, checkpoints, halt, ends_run, restart, timeout, requires (ticket 041).

The systems here are small test states (Step), so these tests exercise the scheduler alone. make_ctx is
shared with test_ecs_core.py.
"""

import json

import pytest

from test_ecs_core import make_ctx
from weekly_analyst.core.components import HaltC, ItemsC, OutboxC, PlanC, SnapshotC, SourceC, ValidationC
from weekly_analyst.core.config import Settings
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.scheduler import Scheduler, file_sha256
from weekly_analyst.core.store import StateStore
from weekly_analyst.core.system import Halt, Restart, System
from weekly_analyst.core.world import PROGRESS_FILE, World


# ---------------------------------------------------------------- scheduler


class Step(System):
    """Test state: logs its name, optionally runs an action, adds one component, halts, or ends the run."""

    def __init__(self, name, calls, *, produces=(), requires=(), component=None, halt=None,
                 ends=False, state_row=True, action=None):
        self.name = name
        self.calls = calls
        self.produces = produces
        self.requires = requires
        self.component = component
        self.halt = halt
        self.ends = ends
        self.state_row = state_row
        self.action = action

    def run(self, world, entity, ctx):
        self.calls.append(self.name)
        if self.action is not None:
            self.action()
        if self.halt is not None:
            raise Halt(self.halt)
        if self.component is not None:
            world.add(entity, self.component)

    def ends_run(self, world, entity):
        return self.ends


@pytest.fixture
def harness(tmp_path):
    source = tmp_path / "sales.csv"
    source.write_bytes(b"date,region\n2026-08-03,north\n")
    world = World(tmp_path / "runs")
    entity = Entity("run-1")
    world.run_dir(entity).mkdir(parents=True)
    world.add(entity, SourceC(csv_path=str(source), snapshot_hash=file_sha256(source)))
    ctx = make_ctx(tmp_path, StateStore(tmp_path / "state.db"))
    return world, entity, ctx, source


def test_scheduler_runs_systems_in_order_and_saves_after_each(harness):
    world, entity, ctx, _ = harness
    calls: list[str] = []
    systems = [
        Step("extract", calls, produces=(SnapshotC,),
             component=SnapshotC(parquet="p", rows_read=1, rows_in_file=1)),
        Step("validate", calls, requires=(SnapshotC,), produces=(ValidationC,),
             component=ValidationC(period="2026-W41", n_periods=0, cold_start=True, findings=[])),
    ]
    Scheduler(systems).drive(world, entity, ctx)
    assert calls == ["extract", "validate"]
    progress = json.loads((world.run_dir(entity) / PROGRESS_FILE).read_text(encoding="utf-8"))
    assert set(progress["components"]) == {"SourceC", "SnapshotC", "ValidationC"}
    assert ctx.store.load_run("run-1")["state"] == "validate"


def test_scheduler_skips_a_system_whose_outputs_already_exist(harness):
    world, entity, ctx, _ = harness
    world.add(entity, SnapshotC(parquet="p", rows_read=1, rows_in_file=1))
    calls: list[str] = []
    systems = [
        Step("extract", calls, produces=(SnapshotC,)),
        Step("validate", calls, requires=(SnapshotC,), produces=(ValidationC,),
             component=ValidationC(period="2026-W41", n_periods=0, cold_start=True, findings=[])),
    ]
    Scheduler(systems).drive(world, entity, ctx)
    assert calls == ["validate"]
    assert ctx.store.load_run("run-1")["state"] == "validate"  # the skipped state still checkpoints


def test_halt_records_the_reason_and_runs_no_later_system(harness):
    world, entity, ctx, _ = harness
    calls: list[str] = []
    systems = [
        Step("validate", calls, produces=(ValidationC,), halt="validate: schema broke"),
        Step("detect", calls, produces=(ItemsC,)),
    ]
    Scheduler(systems).drive(world, entity, ctx)
    assert calls == ["validate"]
    assert world.get(entity, HaltC) == HaltC(reason="validate: schema broke")
    assert ctx.store.load_run("run-1") is None  # a halted state writes no store row


def test_system_that_ends_the_run_stops_the_scheduler_after_its_checkpoint(harness):
    world, entity, ctx, _ = harness
    calls: list[str] = []
    systems = [
        Step("detect", calls, produces=(ItemsC,), ends=True,
             component=ItemsC(selected=[], material=False, alerts_before=0)),
        Step("plan", calls, produces=(PlanC,)),
    ]
    Scheduler(systems).drive(world, entity, ctx)
    assert calls == ["detect"]
    assert ctx.store.load_run("run-1")["state"] == "detect"


def test_source_changed_mid_run_raises_restart_before_the_next_state(harness):
    world, entity, ctx, source = harness
    calls: list[str] = []
    systems = [
        Step("extract", calls, produces=(SnapshotC,), action=lambda: source.write_bytes(b"new rows")),
        Step("validate", calls, produces=(ValidationC,)),
    ]
    with pytest.raises(Restart):
        Scheduler(systems).drive(world, entity, ctx)
    assert calls == ["extract"]


def test_wall_time_limit_halts_before_any_system_runs(tmp_path):
    source = tmp_path / "sales.csv"
    source.write_bytes(b"date,region\n")
    world = World(tmp_path / "runs")
    entity = Entity("run-1")
    world.run_dir(entity).mkdir(parents=True)
    world.add(entity, SourceC(csv_path=str(source), snapshot_hash=file_sha256(source)))
    ctx = make_ctx(tmp_path, StateStore(tmp_path / "state.db"), started=0.0, clock=lambda: 5.0,
                   settings=Settings(run_timeout_s=5))
    calls: list[str] = []
    Scheduler([Step("extract", calls, produces=(SnapshotC,))]).drive(world, entity, ctx)
    assert calls == []
    assert world.get(entity, HaltC).reason == "timeout: run stopped at 5 s"


def test_system_whose_required_component_is_missing_is_not_run(harness):
    world, entity, ctx, _ = harness
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="plan needs"):
        Scheduler([Step("plan", calls, requires=(ItemsC,), produces=(PlanC,))]).drive(world, entity, ctx)
    assert calls == []


def test_system_without_outputs_runs_every_time(harness):
    world, entity, ctx, _ = harness
    calls: list[str] = []
    Scheduler([Step("remember", calls, state_row=False)]).drive(world, entity, ctx)
    assert calls == ["remember"]


def test_state_row_false_writes_no_store_row(harness):
    world, entity, ctx, _ = harness
    calls: list[str] = []
    Scheduler([Step("outbox", calls, produces=(OutboxC,), state_row=False,
                    component=OutboxC(report="/x/r.pdf"))]).drive(world, entity, ctx)
    assert calls == ["outbox"]
    assert ctx.store.load_run("run-1") is None
