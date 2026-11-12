"""ECS core: entity, components, world, system and run context (architecture.md; ticket 041).

Each test names the production behavior it pins. The scheduler has its own file, test_ecs_scheduler.py.
"""

import json
from dataclasses import FrozenInstanceError

import pytest

from weekly_analyst.core.components import (
    COMPONENTS,
    CheckC,
    DraftC,
    EvidenceC,
    HaltC,
    ItemsC,
    OutboxC,
    PlanC,
    ReportC,
    SnapshotC,
    SourceC,
    ValidationC,
)
from weekly_analyst.core.config import Settings
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.items import Item
from weekly_analyst.core.store import StateStore
from weekly_analyst.core.system import Halt, System
from weekly_analyst.core.world import PROGRESS_FILE, World

ITEM = Item(
    metric="units",
    segment=(("region", "north"),),
    period="2026-W41",
    score=1.5,
    detail={"abs_change": 4.0},
)


# ---------------------------------------------------------------- entity and components


def test_entity_equality_is_by_run_id_and_it_is_frozen():
    assert Entity("run-1") == Entity("run-1")
    assert hash(Entity("run-1")) == hash(Entity("run-1"))
    with pytest.raises(FrozenInstanceError):
        Entity("run-1").run_id = "other"


def test_registry_names_every_component_the_architecture_lists():
    assert set(COMPONENTS) == {
        "SourceC", "SnapshotC", "ValidationC", "ItemsC", "PlanC", "EvidenceC",
        "DraftC", "CheckC", "ReportC", "OutboxC", "HaltC",
    }


SAMPLES = [
    SourceC(csv_path="/x/sales.csv", snapshot_hash="abc"),
    SnapshotC(parquet="/x/snap.parquet", rows_read=10, rows_in_file=10),
    ValidationC(
        period="2026-W41", n_periods=8, cold_start=False,
        findings=[{"kind": "schema_change", "metric": "units", "detail": "retyped"}],
    ),
    ItemsC(selected=[ITEM], material=True, alerts_before=1),
    PlanC(steps=[{"item_key": "k", "tool": "local_query", "args": {"sql": "SELECT 1"}}]),
    EvidenceC(results=[{"tool": "wilson", "item_key": "k", "result": {"low": 0.1, "high": 0.2}}]),
    DraftC(draft={"headline": "h", "recommended_action": "a", "findings": [], "caveats": []}),
    CheckC(round_no=1, failures=["number: 3 not in source"]),
    ReportC(pdf="/x/draft-0.pdf", pages=4),
    OutboxC(report="/x/out/run-1.pdf"),
    HaltC(reason="plan: x"),
]


@pytest.mark.parametrize("component", SAMPLES, ids=lambda c: type(c).__name__)
def test_component_round_trips_through_json(component):
    data = json.loads(json.dumps(component.to_json()))
    assert type(component).from_json(data) == component


def test_components_are_frozen():
    with pytest.raises(FrozenInstanceError):
        HaltC(reason="x").reason = "y"


# ---------------------------------------------------------------- world


def test_world_keeps_components_per_entity(tmp_path):
    world = World(tmp_path)
    a, b = Entity("run-a"), Entity("run-b")
    world.add(a, HaltC(reason="timeout"))
    assert world.has(a, HaltC)
    assert world.get(a, HaltC) == HaltC(reason="timeout")
    assert not world.has(b, HaltC)
    assert not world.has(a, OutboxC)
    with pytest.raises(KeyError):
        world.get(b, HaltC)


def test_world_save_writes_progress_file_and_load_restores_it(tmp_path):
    entity = Entity("run-1")
    world = World(tmp_path)
    world.run_dir(entity).mkdir()
    world.add(entity, SourceC(csv_path="/x/s.csv", snapshot_hash="h1"))
    world.add(entity, ItemsC(selected=[ITEM], material=True, alerts_before=0))
    world.save(entity)

    progress = json.loads((tmp_path / "run-1" / PROGRESS_FILE).read_text(encoding="utf-8"))
    assert progress["run_id"] == "run-1"
    assert set(progress["components"]) == {"SourceC", "ItemsC"}
    assert not (tmp_path / "run-1" / f"{PROGRESS_FILE}.tmp").exists()  # atomic write leaves no temp file

    fresh = World(tmp_path)
    loaded = fresh.load("run-1")
    assert loaded == entity
    assert fresh.get(loaded, SourceC) == SourceC(csv_path="/x/s.csv", snapshot_hash="h1")
    assert fresh.get(loaded, ItemsC) == ItemsC(selected=[ITEM], material=True, alerts_before=0)


def test_world_load_of_unknown_run_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        World(tmp_path).load("missing")


# ---------------------------------------------------------------- system and context


def test_system_is_abstract_and_ends_run_defaults_to_false():
    with pytest.raises(TypeError):
        System()  # run() is abstract

    class Noop(System):
        name = "noop"

        def run(self, world, entity, ctx):
            pass

    assert Noop().ends_run(None, None) is False


def make_ctx(tmp_path, store, *, started=0.0, clock=lambda: 0.0, settings=None):
    return RunContext(
        settings=settings or Settings(),
        metrics=None,
        client=None,
        store=store,
        out_dir=tmp_path / "out",
        paper="A4",
        recipients=[],
        started=started,
        clock=clock,
    )


def test_run_context_check_time_halts_once_the_limit_is_reached(tmp_path):
    now = [10.0]
    ctx = make_ctx(
        tmp_path, StateStore(tmp_path / "state.db"), started=0.0,
        clock=lambda: now[0], settings=Settings(run_timeout_s=10),
    )
    with pytest.raises(Halt, match="timeout: run stopped at 10 s"):
        ctx.check_time()
    now[0] = 9.9
    ctx.check_time()  # below the limit: no halt


def test_run_context_cache_is_not_shared_between_runs(tmp_path):
    store = StateStore(tmp_path / "state.db")
    first, second = make_ctx(tmp_path, store), make_ctx(tmp_path, store)
    first.cache["frame"] = object()
    assert "frame" not in second.cache


