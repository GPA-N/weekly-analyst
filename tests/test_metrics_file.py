import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from weekly_analyst.data.metrics_file import load_metrics_file

FIXTURE = Path(__file__).parent / "fixtures" / "metrics_sales.yaml"

BASE_METRIC = {
    "name": "revenue",
    "definition": "Sum of order revenue in the period.",
    "owner": "Finance analytics",
    "inclusions": "All completed orders.",
    "exclusions": "Cancelled orders.",
    "column": "revenue",
    "agg": "sum",
    "segment_columns": ["region", "product", "channel"],
    "min_change": 0.05,
    "version": 1,
}


def _doc(metric):
    return {"date_column": "date", "metrics": [metric]}


def _write_yaml(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def _write_json(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_checksum_changes_when_definition_changes(tmp_path):
    original = _write_yaml(tmp_path, "a.yaml", _doc(BASE_METRIC))
    changed = _write_yaml(
        tmp_path,
        "b.yaml",
        _doc({**BASE_METRIC, "definition": "Sum of order revenue after returns."}),
    )
    assert load_metrics_file(original).checksum != load_metrics_file(changed).checksum


def test_checksum_stable_across_key_order(tmp_path):
    reordered = dict(reversed(list(BASE_METRIC.items())))
    original = _write_yaml(tmp_path, "a.yaml", _doc(BASE_METRIC))
    shuffled = _write_yaml(tmp_path, "b.yaml", _doc(reordered))
    a = load_metrics_file(original)
    b = load_metrics_file(shuffled)
    assert a.checksum == b.checksum
    assert re.fullmatch(r"[0-9a-f]{64}", a.checksum)


@pytest.mark.parametrize(
    "bad_metric",
    [
        {k: v for k, v in BASE_METRIC.items() if k != "definition"},
        {**BASE_METRIC, "definition": ""},
        {**BASE_METRIC, "definition": "   "},
        {k: v for k, v in BASE_METRIC.items() if k != "min_change"},
    ],
)
def test_load_rejects_metric_without_definition_or_min_change(tmp_path, bad_metric):
    path = _write_yaml(tmp_path, "bad.yaml", _doc(bad_metric))
    with pytest.raises(ValidationError):
        load_metrics_file(path)


@pytest.mark.parametrize(
    "bad_metric",
    [
        {**BASE_METRIC, "agg": "median"},
        {**BASE_METRIC, "unexpected": "field"},
    ],
)
def test_load_rejects_unknown_agg_and_extra_keys(tmp_path, bad_metric):
    path = _write_yaml(tmp_path, "bad.yaml", _doc(bad_metric))
    with pytest.raises(ValidationError):
        load_metrics_file(path)


def test_load_json_gives_same_file_as_yaml(tmp_path):
    doc = _doc(BASE_METRIC)
    from_yaml = load_metrics_file(_write_yaml(tmp_path, "m.yaml", doc))
    from_json = load_metrics_file(_write_json(tmp_path, "m.json", doc))
    assert from_yaml == from_json


def test_columns_returns_expected_set_for_fixture():
    metrics_file = load_metrics_file(FIXTURE)
    assert metrics_file.columns() == {
        "revenue",
        "units",
        "returned",
        "discount",
        "region",
        "product",
        "channel",
        "date",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "rev|enue"),
        ("name", "rev,enue"),
        ("name", "rev=enue"),
        ("name", 'rev"enue'),
        ("name", "revenue\nx"),
        ("name", "1revenue"),
        ("name", ""),
        ("name", "a" * 65),
        ("column", "col umn"),
        ("column", "col;drop"),
        ("segment_columns", ["region", "re'gion"]),
        ("segment_columns", ["region", "region"]),
        ("min_change", -0.1),
        ("min_change", float("nan")),
        ("min_change", float("inf")),
        ("version", 0),
    ],
)
def test_load_rejects_unsafe_or_nonsensical_values(tmp_path, field, value):
    path = _write_json(tmp_path, "m.json", _doc({**BASE_METRIC, field: value}))
    with pytest.raises((ValidationError, ValueError)):
        load_metrics_file(path)


def test_load_rejects_unsafe_date_column(tmp_path):
    doc = {**_doc(BASE_METRIC), "date_column": "da te"}
    with pytest.raises((ValidationError, ValueError)):
        load_metrics_file(_write_json(tmp_path, "m.json", doc))


def test_load_rejects_duplicate_metric_names_and_empty_metric_list(tmp_path):
    with pytest.raises((ValidationError, ValueError)):
        load_metrics_file(_write_json(tmp_path, "dup.json", {"date_column": "date", "metrics": [BASE_METRIC, BASE_METRIC]}))
    with pytest.raises((ValidationError, ValueError)):
        load_metrics_file(_write_json(tmp_path, "none.json", {"date_column": "date", "metrics": []}))


def test_load_rejects_unknown_top_level_key(tmp_path):
    doc = {**_doc(BASE_METRIC), "plausiblity": []}
    with pytest.raises((ValidationError, ValueError)):
        load_metrics_file(_write_json(tmp_path, "m.json", doc))


def test_load_rejects_oversized_file(tmp_path):
    path = tmp_path / "big.yaml"
    path.write_text("# " + "x" * 1_100_000 + "\ndate_column: date\nmetrics: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="too large"):
        load_metrics_file(path)
