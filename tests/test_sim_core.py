import re
from datetime import date

import pytest

from weekly_analyst.sim.generate import DEV_SEEDS, HOLDOUT_SEEDS, SimResult, generate

HEADER = "date,region,product,channel,units,revenue,discount,returned"
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TWO_DECIMALS = re.compile(r"^\d+\.\d{2}$")


def _rows(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


def test_same_seed_is_byte_identical(tmp_path):
    a = generate(seed=7, weeks=3, events=[], out_dir=tmp_path / "a")
    b = generate(seed=7, weeks=3, events=[], out_dir=tmp_path / "b")
    assert [p.name for p in a.files] == [p.name for p in b.files]
    for pa, pb in zip(a.files, b.files):
        assert pa.read_bytes() == pb.read_bytes()


def test_different_seed_differs(tmp_path):
    a = generate(seed=1, weeks=1, events=[], out_dir=tmp_path / "s1")
    b = generate(seed=2, weeks=1, events=[], out_dir=tmp_path / "s2")
    assert a.files[0].read_bytes() != b.files[0].read_bytes()


def test_writes_one_csv_per_week_with_exact_header(tmp_path):
    result = generate(seed=3, weeks=4, events=[], out_dir=tmp_path)
    assert isinstance(result, SimResult)
    assert len(result.files) == 4
    for path in result.files:
        assert path.exists()
        assert path.read_text(encoding="utf-8").splitlines()[0] == HEADER


def test_dates_are_iso(tmp_path):
    result = generate(seed=3, weeks=2, events=[], out_dir=tmp_path)
    for path in result.files:
        for row in _rows(path):
            assert ISO_DATE.match(row["date"])
            assert date.fromisoformat(row["date"]).isoformat() == row["date"]


def test_dev_and_holdout_seed_ranges_do_not_overlap():
    assert DEV_SEEDS == range(0, 1000)
    assert HOLDOUT_SEEDS == range(10_000, 11_000)
    assert set(DEV_SEEDS).isdisjoint(HOLDOUT_SEEDS)


def test_truth_empty_without_events(tmp_path):
    result = generate(seed=5, weeks=2, events=[], out_dir=tmp_path)
    assert result.truth == []


def test_non_empty_events_rejected_until_ticket_004(tmp_path):
    with pytest.raises(ValueError):
        generate(seed=5, weeks=1, events=["quiet"], out_dir=tmp_path)


def test_odd_combo_discount_half_with_one_unit_is_present(tmp_path):
    result = generate(seed=0, weeks=4, events=[], out_dir=tmp_path)
    odd = [
        row
        for path in result.files
        for row in _rows(path)
        if row["discount"] == "0.50" and row["units"] == "1"
    ]
    assert odd


def test_revenue_and_discount_use_fixed_two_decimals(tmp_path):
    result = generate(seed=4, weeks=2, events=[], out_dir=tmp_path)
    for path in result.files:
        for row in _rows(path):
            assert TWO_DECIMALS.match(row["revenue"])
            assert TWO_DECIMALS.match(row["discount"])
