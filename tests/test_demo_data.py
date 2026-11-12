"""Demo data generator (used by the hosted demo and the eval)."""

import pytest

from weekly_analyst.sim.demo import HEADER, SCENARIOS, WEEKS, write_demo_csv


def test_every_scenario_writes_nine_weeks_of_35_rows_per_segment(tmp_path):
    for scenario in SCENARIOS:
        lines = write_demo_csv(tmp_path / f"{scenario}.csv", scenario).read_text(encoding="utf-8").splitlines()
        assert lines[0] == HEADER
        assert len(lines) - 1 == WEEKS * 7 * 8 * 5


def test_same_scenario_is_byte_identical(tmp_path):
    a = write_demo_csv(tmp_path / "a.csv", "spike").read_bytes()
    b = write_demo_csv(tmp_path / "b.csv", "spike").read_bytes()
    assert a == b


def test_only_the_last_week_differs_between_scenarios(tmp_path):
    quiet = write_demo_csv(tmp_path / "q.csv", "quiet").read_text(encoding="utf-8").splitlines()
    spike = write_demo_csv(tmp_path / "s.csv", "spike").read_text(encoding="utf-8").splitlines()
    per_week = 7 * 8 * 5
    assert quiet[: 1 + (WEEKS - 1) * per_week] == spike[: 1 + (WEEKS - 1) * per_week]
    assert quiet[1 + (WEEKS - 1) * per_week :] != spike[1 + (WEEKS - 1) * per_week :]


def test_unknown_scenario_and_too_few_weeks_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_demo_csv(tmp_path / "x.csv", "nope")
    with pytest.raises(ValueError):
        write_demo_csv(tmp_path / "x.csv", "quiet", weeks=1)
