"""named_query as an investigate tool (spec 6.2 step 6, 6.4): dispatch, sanitised labels, refusals run nothing."""

import json
from pathlib import Path

import pytest

from weekly_analyst.core.components import SnapshotC
from weekly_analyst.core.config import Settings
from weekly_analyst.core.context import RunContext
from weekly_analyst.core.entity import Entity
from weekly_analyst.core.store import StateStore
from weekly_analyst.core.world import World
from weekly_analyst.data import sanitize
from weekly_analyst.data.metrics_file import load_metrics_file
from weekly_analyst.data.named_checks import NamedQueryError
from weekly_analyst.data.query import LocalQuery
from weekly_analyst.systems.tools import ToolRunner

METRICS = load_metrics_file(Path(__file__).parent / "fixtures" / "metrics_sales.yaml")
PERIOD = "2026-W03"
INJECTION = "x'); DROP TABLE snapshot; --"


@pytest.fixture
def tools(named_snapshot, tmp_path):
    world = World(tmp_path / "runs")
    entity = Entity("run-named-tools")
    world.add(entity, SnapshotC(parquet=str(named_snapshot), rows_read=1, rows_in_file=1))
    ctx = RunContext(
        settings=Settings(),
        metrics=METRICS,
        client=None,
        store=StateStore(tmp_path / "state.db"),
        out_dir=tmp_path / "out",
        paper="A4",
        recipients=["ops@example.com"],
        started=0.0,
    )
    return ToolRunner(world, entity, ctx)


def test_named_query_returns_row_count_and_rows(tools):
    result = tools.call_tool(
        "named_query",
        {"name": "by_segment", "params": {"metric": "revenue", "segment_col": "region", "period": PERIOD}},
    )
    assert set(result) == {"row_count", "rows"}
    assert result["row_count"] == len(result["rows"]) and result["row_count"] > 0


def test_segment_labels_leave_the_tool_sanitised(tools):
    hostile_label = sanitize.label("west'); DROP TABLE snapshot; --")
    result = tools.call_tool(
        "named_query",
        {"name": "by_segment", "params": {"metric": "revenue", "segment_col": "region", "period": PERIOD}},
    )
    segments = [row["segment"] for row in result["rows"]]
    assert hostile_label in segments
    assert "DROP" not in json.dumps(result)
    assert all(sanitize.label(seg) == seg for seg in segments)


def test_refused_named_query_raises_named_query_error_and_runs_nothing(tools, monkeypatch):
    executed = []
    original = LocalQuery.run

    def spy(self, sql, timeout_s=10.0):
        executed.append(sql)
        return original(self, sql, timeout_s)

    monkeypatch.setattr(LocalQuery, "run", spy)
    params = {"metric": "revenue", "period": PERIOD, "filter_col": "region", "filter_value": INJECTION}
    with pytest.raises(NamedQueryError):
        tools.call_tool("named_query", {"name": "filtered_total", "params": params})
    assert executed == []


@pytest.mark.parametrize(
    "args",
    [{"name": "by_segment"}, {"params": {}}, {"name": "by_segment", "params": None}],
)
def test_malformed_named_query_args_are_named_query_errors_not_key_errors(tools, args):
    with pytest.raises(NamedQueryError):
        tools.call_tool("named_query", args)
