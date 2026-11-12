"""Runner: quiet snapshot, full run, halts, run lock, timeout, daily file, mid-run change (spec 6.2, 6.6, 6.7).

The model is a stand-in that answers from the data block of each prompt. The mechanics tests stub the
checker (checks_pass). test_full_run_with_scripted_client_produces_checked_pdf_in_outbox uses the real one.
"""

import hashlib
import json
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

import pypdf
import pytest

from weekly_analyst.core.config import Settings
from weekly_analyst.delivery.outbox import Outbox
from weekly_analyst.analysis.model import ScriptedClient
from weekly_analyst.runner import RunConfig, RunLocked, run
from weekly_analyst.systems import check as check_system
from weekly_analyst.core.store import StateStore

METRICS = Path(__file__).parent / "fixtures" / "metrics_sales.yaml"
START = date(2026, 8, 3)  # a Monday
WEEKS = 9  # eight prior weeks, so the cold start (fewer than 8 periods) does not apply
_LATEST = (START + timedelta(weeks=WEEKS - 1)).isocalendar()
PERIOD = f"{_LATEST.year}-W{_LATEST.week:02d}"
SEGMENTS = [(r, p, c) for r in ("north", "south") for p in ("basic", "pro") for c in ("web", "store")]


def write_csv(path: Path, *, bump: float = 1.0) -> Path:
    """Identical weeks, except the last week's units and revenue are scaled by bump."""
    rng = random.Random(7)
    base = {
        seg: (rng.randint(5, 25), round(rng.uniform(9.0, 90.0), 2), round(rng.uniform(0.0, 0.2), 2))
        for seg in SEGMENTS
    }
    lines = ["date,region,product,channel,units,revenue,discount,returned"]
    for week in range(WEEKS):
        scale = bump if week == WEEKS - 1 else 1.0
        for day in range(7):
            day_iso = (START + timedelta(weeks=week, days=day)).isoformat()
            for seg in SEGMENTS:
                units0, price, discount = base[seg]
                # Five rows per segment per day: 35 rows per segment-week, above the 30-row gate.
                for k in range(5):
                    units = round((units0 + (day + k) % 3) * scale)
                    revenue = round(units * price, 2)
                    returned = (day + k) % 2
                    lines.append(
                        f"{day_iso},{seg[0]},{seg[1]},{seg[2]},{units},{revenue:.2f},{discount:.2f},{returned}"
                    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class DataClient:
    """Stand-in model. Answers each request from the data block in its prompt; counts calls by kind."""

    def __init__(self, *, plan_sql="SELECT COUNT(*) AS n FROM snapshot", plan_reply=None, hook=None):
        self.kinds: list[str] = []
        self.plan_sql = plan_sql
        self.plan_reply = plan_reply
        self.hook = hook

    @property
    def calls(self) -> int:
        return len(self.kinds)

    def complete(self, prompt, json_schema, *, temperature, seed, grammar=None) -> str:
        if self.hook is not None:
            self.hook()
        if grammar is not None:  # the free-SQL call writes the SELECT that plan_sql stands for
            self.kinds.append("sql")
            return self.plan_sql
        body = json.loads(prompt.split("<data>\n", 1)[1].rsplit("\n</data>", 1)[0])
        if "steps" in json_schema["properties"]:
            self.kinds.append("plan")
            if self.plan_reply is not None:
                return self.plan_reply
            question = {"question": "How many rows are in the snapshot?", "required_columns": ["n"]}
            steps = [{"item_key": row["item_key"], "tool": "local_query", "args": question} for row in body]
            return json.dumps({"steps": steps})
        self.kinds.append("draft")
        findings = [
            {"item_key": item["item_key"], "so_what": f"{item['metric']} moved this week.", "metric_refs": [item["metric"]]}
            for item in body["items"]
        ]
        return json.dumps(
            {
                "headline": "Weekly sales moved",
                "recommended_action": "Review the flagged segments.",
                "findings": findings,
                "caveats": [],
            }
        )


@pytest.fixture
def checks_pass(monkeypatch):
    """Stub the checker for mechanics tests. Only the full-run test exercises the real checker."""
    monkeypatch.setattr(check_system, "check_report", lambda *args, **kwargs: [])


def make_cfg(tmp_path: Path, settings: Settings | None = None) -> RunConfig:
    return RunConfig(
        metrics_path=METRICS,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
        delivery_mode="approval",
        recipients=["ops@example.com"],
        paper="A4",
        settings=settings or Settings(),
    )


def outbox_pdfs(cfg: RunConfig) -> list[str]:
    return sorted(p.name for p in cfg.out_dir.glob("*.pdf")) if cfg.out_dir.exists() else []


def run_dirs(cfg: RunConfig) -> list[Path]:
    root = cfg.work_dir / "runs"
    return sorted(root.iterdir()) if root.exists() else []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(str(path)).pages)


def test_quiet_snapshot_ends_after_detect_with_no_model_call_no_outbox(tmp_path):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    client = ScriptedClient([])  # any model call raises IndexError
    outcome = run(write_csv(tmp_path / "sales.csv", bump=1.0), cfg, client, store)
    assert outcome.kind == "silent", outcome.reasons
    assert client.calls == 0
    assert outbox_pdfs(cfg) == []
    assert store.load_run(outcome.run_id)["state"] == "detect"


def test_full_run_with_scripted_client_produces_checked_pdf_in_outbox(tmp_path):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    outcome = run(write_csv(tmp_path / "sales.csv", bump=1.5), cfg, DataClient(), store)
    assert outcome.kind == "brief", outcome.reasons
    assert outbox_pdfs(cfg) == [f"{outcome.run_id}.pdf"]
    assert len(pypdf.PdfReader(str(outcome.report_path)).pages) <= 8
    assert run_dirs(cfg) == []  # run Parquet deleted at finish


def test_plan_failing_schema_twice_halts_with_report(tmp_path, checks_pass):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    client = DataClient(plan_reply="{not json")
    outcome = run(write_csv(tmp_path / "sales.csv", bump=1.5), cfg, client, store)
    assert outcome.kind == "halted"
    assert outcome.reasons and outcome.reasons[0].startswith("plan:")
    assert client.kinds == ["plan"] * 3  # first attempt and two re-prompts
    assert outbox_pdfs(cfg) == []


def test_tool_error_halts_with_report(tmp_path, checks_pass):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    client = DataClient(plan_sql="SELECT customer_email FROM snapshot")
    outcome = run(write_csv(tmp_path / "sales.csv", bump=1.5), cfg, client, store)
    assert outcome.kind == "halted"
    assert outcome.reasons and outcome.reasons[0].startswith("tool error: local_query")
    assert outbox_pdfs(cfg) == []
    assert run_dirs(cfg) == []


def test_two_concurrent_runs_second_exits(tmp_path, checks_pass):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    csv = write_csv(tmp_path / "sales.csv", bump=1.5)
    locked: list[bool] = []
    attempted: list[bool] = []

    def second_run_during_plan() -> None:
        if attempted:
            return
        attempted.append(True)
        try:
            run(csv, cfg, ScriptedClient([]), store)
        except RunLocked:
            locked.append(True)

    outcome = run(csv, cfg, DataClient(hook=second_run_during_plan), store)
    assert locked == [True]
    assert outcome.kind == "brief", outcome.reasons


def test_timeout_halts_cleanly(tmp_path, checks_pass):
    cfg = make_cfg(tmp_path, Settings(run_timeout_s=0))
    store = StateStore(tmp_path / "state.db")
    client = ScriptedClient([])
    outcome = run(write_csv(tmp_path / "sales.csv", bump=1.5), cfg, client, store)
    assert outcome.kind == "halted"
    assert outcome.reasons and outcome.reasons[0].startswith("timeout")
    assert client.calls == 0
    assert outbox_pdfs(cfg) == []
    assert run_dirs(cfg) == []


def test_daily_file_never_refused_after_finished_run(tmp_path, checks_pass):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    csv = write_csv(tmp_path / "sales.csv", bump=1.5)
    first = run(csv, cfg, DataClient(), store)
    second_client = DataClient()
    second = run(csv, cfg, second_client, store)  # same file again: not refused
    assert first.kind == "brief", first.reasons
    assert second.kind == "silent", second.reasons  # weekly budget already spent
    assert second.run_id != first.run_id
    assert second_client.calls == 0


def test_new_snapshot_mid_run_restarts_and_never_briefs_stale(tmp_path, checks_pass):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    csv = write_csv(tmp_path / "sales.csv", bump=1.5)
    old_hash = sha256(csv)
    replaced: list[bool] = []

    def replace_source_during_plan() -> None:
        if not replaced:
            replaced.append(True)
            write_csv(csv, bump=2.0)

    client = DataClient(hook=replace_source_during_plan)
    outcome = run(csv, cfg, client, store)
    new_hash = sha256(csv)
    assert outcome.kind == "brief", outcome.reasons
    assert outbox_pdfs(cfg) == [f"{outcome.run_id}.pdf"]
    text = pdf_text(outcome.report_path)
    assert new_hash[:16] in text  # the footer shows the first 16 characters of the snapshot hash
    assert old_hash[:16] not in text
    # The abandoned run planned once. The fresh run plans, then one free-SQL call per local_query step, then drafts.
    assert client.kinds == ["plan", "plan", "sql", "sql", "sql", "draft"]


def test_brief_waits_in_outbox_and_approve_accepts_it(tmp_path, checks_pass):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    csv = write_csv(tmp_path / "sales.csv", bump=1.5)
    outcome = run(csv, cfg, DataClient(), store)
    assert outcome.kind == "brief", outcome.reasons
    assert store.load_run(outcome.run_id)["state"] == "outbox"

    def stamp(draft_path: Path, approver: str) -> Path:
        stamped = draft_path.with_name("stamped.pdf")
        shutil.copyfile(draft_path, stamped)
        return stamped

    stamped = Outbox(cfg.out_dir, store, Settings()).approve(
        outcome.run_id, "ops-lead", sha256(csv), stamp, lambda path: []
    )
    assert stamped.is_file()


class MixedClient(DataClient):
    """First item gets a good local_query, every other item a named_query with n=99 (refused when it runs)."""

    def complete(self, prompt, json_schema, *, temperature, seed, grammar=None) -> str:
        if grammar is None and "steps" in json_schema.get("properties", {}):
            body = json.loads(prompt.split("<data>\n", 1)[1].rsplit("\n</data>", 1)[0])
            self.kinds.append("plan")
            question = {"question": "How many rows are in the snapshot?", "required_columns": ["n"]}
            steps = []
            for i, row in enumerate(body):
                if i == 0:
                    steps.append({"item_key": row["item_key"], "tool": "local_query", "args": question})
                else:
                    bad = {"name": "top_segments", "params": {"metric": "units", "segment_col": "region", "period": PERIOD, "n": 99, "direction": "desc"}}
                    steps.append({"item_key": row["item_key"], "tool": "named_query", "args": bad})
            return json.dumps({"steps": steps})
        return super().complete(prompt, json_schema, temperature=temperature, seed=seed, grammar=grammar)


def test_one_failed_follow_up_does_not_sink_the_report_and_is_stated(tmp_path, checks_pass):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    outcome = run(write_csv(tmp_path / "sales.csv", bump=1.5), cfg, MixedClient(), store)
    assert outcome.kind == "brief", outcome.reasons
    assert "could not be run" in pdf_text(outcome.report_path)
