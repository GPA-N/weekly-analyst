from datetime import datetime

import pytest

from weekly_analyst.core.store import StateStore, iso_week_of


def test_alerts_counted_per_iso_week(tmp_path):
    store = StateStore(tmp_path / "state.sqlite")
    store.record_alert("sales:region=north:2026-11-02", "2026-W45", "run-1")
    store.record_alert("sales:region=south:2026-11-02", "2026-W45", "run-2")
    store.record_alert("sales:region=east:2026-11-09", "2026-W46", "run-3")

    assert store.alerts_sent_in_week("2026-W45") == 2
    assert store.alerts_sent_in_week("2026-W46") == 1
    assert store.alerts_sent_in_week("2026-W47") == 0


def test_reopen_keeps_alerts_and_run_state(tmp_path):
    path = tmp_path / "state.sqlite"
    store = StateStore(path)
    store.record_alert("sales:total:2026-11-09", "2026-W46", "run-1")
    store.save_state("run-1", "outbox", {"alerts": 1})

    reopened = StateStore(path)

    assert reopened.alerts_sent_in_week("2026-W46") == 1
    assert reopened.reported_keys() == {"sales:total:2026-11-09"}
    assert reopened.load_run("run-1") == {
        "run_id": "run-1",
        "state": "outbox",
        "payload": {"alerts": 1},
    }


def test_reported_keys_returns_every_recorded_item_key(tmp_path):
    store = StateStore(tmp_path / "state.sqlite")
    store.record_alert("item-a", "2026-W46", "run-1")
    store.record_alert("item-b", "2026-W47", "run-2")

    assert store.reported_keys() == {"item-a", "item-b"}


def test_save_and_load_run_round_trip(tmp_path):
    store = StateStore(tmp_path / "state.sqlite")
    payload = {"items": ["sales:total"], "week": "2026-W46", "note": "ok"}

    store.save_state("run-1", "plan", payload)

    assert store.load_run("run-1") == {
        "run_id": "run-1",
        "state": "plan",
        "payload": payload,
    }
    assert store.load_run("missing-run") is None


@pytest.mark.parametrize(
    "ts",
    [
        datetime(2026, 12, 27, 23, 59),
        datetime(2026, 12, 28, 0, 1),
        datetime(2020, 12, 31, 12, 0),
    ],
)
def test_iso_week_of_matches_isocalendar(ts):
    iso = ts.isocalendar()

    assert iso_week_of(ts) == f"{iso.year}-W{iso.week:02d}"


def test_sunday_night_and_monday_after_midnight_are_different_weeks():
    sunday_night = datetime(2026, 12, 27, 23, 59)
    monday_after = datetime(2026, 12, 28, 0, 1)

    assert iso_week_of(sunday_night) != iso_week_of(monday_after)
