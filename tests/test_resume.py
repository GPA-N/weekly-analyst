"""Resume after a kill (AC-008) and resume of an unfinished run (spec 6.2, 6.5; ticket 022).

A kill is raised as a BaseException at a state's save point, so no runner code catches it. Each
killed run must resume with the same run id, one outbox PDF, one alert per budget slot, and no repeated
model work. The checker is stubbed here, as in test_runner.py.
"""

import pytest

from test_runner import DataClient, PERIOD, make_cfg, outbox_pdfs, run_dirs, write_csv
from weekly_analyst.runner import STATES, resume, run
from weekly_analyst.core.store import StateStore
from weekly_analyst.systems import check as check_system, extract as extract_system

pytestmark = pytest.mark.usefixtures("checks_pass")


@pytest.fixture
def checks_pass(monkeypatch):
    """Same stub as test_runner.py: the mechanics under test do not depend on the checker's verdict."""
    monkeypatch.setattr(check_system, "check_report", lambda *args, **kwargs: [])


class Killed(BaseException):
    """Simulated process kill."""


@pytest.fixture
def kill_at(monkeypatch):
    """Arm one kill at a save point. Remember writes no state row (the outbox row must stay for
    approval), so its kill points sit on the alert writes instead."""
    real_save = StateStore.save_state
    real_record = StateStore.record_alert
    armed: dict[str, tuple[str, str]] = {}

    def fire(state: str, when: str) -> None:
        if armed.get("point") == (state, when):
            armed.clear()
            raise Killed()

    def save_state(self, run_id, state, payload):
        fire(state, "before")
        real_save(self, run_id, state, payload)
        fire(state, "after")

    def record_alert(self, item_key, iso_week, run_id):
        fire("remember", "before")
        real_record(self, item_key, iso_week, run_id)
        fire("remember", "after")

    monkeypatch.setattr(StateStore, "save_state", save_state)
    monkeypatch.setattr(StateStore, "record_alert", record_alert)

    def arm(state: str, when: str) -> None:
        armed["point"] = (state, when)

    return arm


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("state", STATES)
def test_kill_resumes_without_duplicate_or_partial_output(state, when, tmp_path, kill_at):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    client = DataClient()
    csv = write_csv(tmp_path / "sales.csv", bump=1.5)
    kill_at(state, when)
    with pytest.raises(Killed):
        run(csv, cfg, client, store)
    (killed,) = run_dirs(cfg)
    outcome = resume(killed.name, cfg, client, store)
    assert outcome.kind == "brief", outcome.reasons
    assert outcome.run_id == killed.name
    assert outbox_pdfs(cfg) == [f"{killed.name}.pdf"]
    assert store.load_run(killed.name)["state"] == "outbox"  # approve() needs this row
    assert store.alerts_sent_in_week(PERIOD) == 3  # one alert per budget slot, none repeated
    assert client.kinds == ["plan", "sql", "sql", "sql", "draft"]  # finished model work is never repeated
    assert run_dirs(cfg) == []


def test_kill_during_extract_resumes_from_extract(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    client = DataClient()
    csv = write_csv(tmp_path / "sales.csv", bump=1.5)
    real_extract = extract_system.extract

    def dying_extract(*args, **kwargs):
        monkeypatch.setattr(extract_system, "extract", real_extract)
        raise Killed()

    monkeypatch.setattr(extract_system, "extract", dying_extract)
    with pytest.raises(Killed):
        run(csv, cfg, client, store)
    (killed,) = run_dirs(cfg)
    outcome = resume(killed.name, cfg, client, store)
    assert outcome.kind == "brief", outcome.reasons
    assert outbox_pdfs(cfg) == [f"{killed.name}.pdf"]


def test_unfinished_same_hash_run_resumes_instead_of_refusing(tmp_path, kill_at):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    client = DataClient()
    csv = write_csv(tmp_path / "sales.csv", bump=1.5)
    kill_at("plan", "after")
    with pytest.raises(Killed):
        run(csv, cfg, client, store)
    (killed,) = run_dirs(cfg)
    outcome = run(csv, cfg, client, store)  # same file, same snapshot hash
    assert outcome.run_id == killed.name
    assert outcome.kind == "brief", outcome.reasons
    assert client.kinds == ["plan", "sql", "sql", "sql", "draft"]


def test_resume_rejects_unknown_or_unsafe_run_id(tmp_path):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    client = DataClient()
    with pytest.raises(ValueError):
        resume("missing-run", cfg, client, store)
    with pytest.raises(ValueError):
        resume("../outside", cfg, client, store)
