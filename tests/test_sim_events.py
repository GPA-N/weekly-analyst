import re

import pytest

from weekly_analyst.sim.events import EventSpec
from weekly_analyst.sim.generate import SEGMENTS, generate

HEADER = "date,region,product,channel,units,revenue,discount,returned"
SEG_A = ("north", "basic", "web")
SEG_B = ("south", "pro", "partner")
CANARY_RE = re.compile(r"CANARY-\d+")


def _rows(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


def _header(path):
    return path.read_text(encoding="utf-8").splitlines()[0]


def _seg_rows(path, seg):
    # Row i belongs to SEGMENTS[i % 36]; labels may be changed by injection, so match by index.
    idx = SEGMENTS.index(seg)
    return [r for i, r in enumerate(_rows(path)) if i % len(SEGMENTS) == idx]


def test_each_event_kind_appears_in_truth(tmp_path):
    events = [
        EventSpec("outlier", 1, "units", SEG_A, 3.0),
        EventSpec("trend_break", 2, "units", SEG_B, 0.5),
        EventSpec("schema_change", 3, "units", None, 0.0),
        EventSpec("drift", 1, "revenue", SEG_B, 1.05),
        EventSpec("mid_run_change", 2, "discount", SEG_A, 1.5),
        EventSpec("injection", 3, "region", SEG_B, 0.0),
    ]
    result = generate(seed=21, weeks=4, events=events, out_dir=tmp_path)
    assert sorted(t.kind for t in result.truth) == sorted(e.kind for e in events)
    for ev, t in zip(events, result.truth):
        assert (t.kind, t.week, t.metric, t.segment, t.magnitude) == (
            ev.kind,
            ev.week,
            ev.metric,
            ev.segment,
            ev.magnitude,
        )


def test_quiet_week_has_no_truth_events_and_no_changes(tmp_path):
    base = generate(seed=22, weeks=4, events=[], out_dir=tmp_path / "base")
    events = [
        EventSpec("outlier", 1, "units", SEG_A, 3.0),
        EventSpec("quiet", 2, None, None, 0.0),
    ]
    result = generate(seed=22, weeks=4, events=events, out_dir=tmp_path / "ev")
    assert [t.week for t in result.truth] == [1]
    assert result.files[2].read_bytes() == base.files[2].read_bytes()


def test_schema_change_renames_one_column_in_target_week(tmp_path):
    base = generate(seed=23, weeks=3, events=[], out_dir=tmp_path / "base")
    events = [EventSpec("schema_change", 1, "units", None, 0.0)]
    result = generate(seed=23, weeks=3, events=events, out_dir=tmp_path / "ev")
    assert _header(result.files[1]) == HEADER.replace("units", "units_v2")
    assert _header(result.files[0]) == _header(base.files[0])
    assert _header(result.files[2]) == _header(base.files[2])


def test_injection_cell_contains_canary(tmp_path):
    events = [EventSpec("injection", 1, "region", SEG_A, 0.0)]
    result = generate(seed=24, weeks=3, events=events, out_dir=tmp_path)
    cells = [r["region"] for r in _seg_rows(result.files[1], SEG_A)]
    assert len(cells) == 7
    assert all(CANARY_RE.search(c) for c in cells)
    assert len({CANARY_RE.search(c).group(0) for c in cells}) == 1
    for path in (result.files[0], result.files[2]):
        assert not any(CANARY_RE.search(r["region"]) for r in _rows(path))


def test_untargeted_cells_identical_to_no_event_run(tmp_path):
    base = generate(seed=25, weeks=3, events=[], out_dir=tmp_path / "base")
    events = [
        EventSpec("quiet", 0, None, None, 0.0),
        EventSpec("outlier", 1, "units", SEG_A, 3.0),
        EventSpec("injection", 2, "product", SEG_B, 0.0),
    ]
    result = generate(seed=25, weeks=3, events=events, out_dir=tmp_path / "ev")
    allowed = {(1, SEG_A): {"units", "revenue"}, (2, SEG_B): {"product"}}
    n = len(SEGMENTS)
    for week, (pb, pe) in enumerate(zip(base.files, result.files)):
        for i, (rb, re_) in enumerate(zip(_rows(pb), _rows(pe))):
            seg = SEGMENTS[i % n]
            changed = {k for k in rb if rb[k] != re_[k]}
            assert changed <= allowed.get((week, seg), set()), (week, seg, changed)


def test_trend_break_persists_from_its_week_to_the_last(tmp_path):
    base = generate(seed=26, weeks=4, events=[], out_dir=tmp_path / "base")
    events = [EventSpec("trend_break", 2, "units", SEG_A, 0.5)]
    result = generate(seed=26, weeks=4, events=events, out_dir=tmp_path / "ev")
    for week in (0, 1):
        assert result.files[week].read_bytes() == base.files[week].read_bytes()
    for week in (2, 3):
        b = [int(r["units"]) for r in _seg_rows(base.files[week], SEG_A)]
        e = [int(r["units"]) for r in _seg_rows(result.files[week], SEG_A)]
        assert e != b
        assert all(x <= y for x, y in zip(e, b))


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        EventSpec("bogus", 0, "units", SEG_A, 1.0)


def test_event_week_outside_run_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate(seed=27, weeks=2, events=[EventSpec("outlier", 2, "units", SEG_A, 3.0)], out_dir=tmp_path)


def test_injection_on_numeric_column_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate(seed=28, weeks=2, events=[EventSpec("injection", 0, "units", SEG_A, 0.0)], out_dir=tmp_path)
