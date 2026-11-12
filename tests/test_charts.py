import re
import os
import shutil
import xml.etree.ElementTree as ET

import pytest
from pydantic import ValidationError

from weekly_analyst.report.charts import CATEGORICAL_LIGHT, ChartSpec, make_chart, validate_palette

NODE_MISSING = "node or DATAVIZ_VALIDATOR is not set; the palette validator (scripts/validate_palette.js) cannot run"


def _line_spec(**overrides) -> ChartSpec:
    fields = dict(
        kind="line",
        title="Weekly revenue",
        x=["W40", "W41", "W42"],
        series={"North": [10.0, 12.5, 11.0], "South": [8.0, 7.5, 9.0]},
        unit="EUR thousands",
        alt_text="Line chart of weekly revenue for North and South over three weeks.",
    )
    fields.update(overrides)
    return ChartSpec(**fields)


def _svg(spec: ChartSpec) -> str:
    return make_chart(spec).decode("utf-8")


def test_output_is_well_formed_svg_without_raster_image():
    svg = make_chart(_line_spec())
    assert b"<svg" in svg
    ET.fromstring(svg)
    assert b"<image" not in svg
    assert b"data:image" not in svg
    assert b"image/png" not in svg


def test_two_series_show_legend_as_svg_text():
    svg = _svg(_line_spec())
    assert ">North</text>" in svg
    assert ">South</text>" in svg


def test_single_series_has_no_legend():
    svg = _svg(_line_spec(series={"Solo": [1.0, 2.0, 3.0]}))
    assert "Solo" not in svg


def test_no_dual_axis_can_be_built():
    assert set(ChartSpec.model_fields) == {"kind", "title", "x", "series", "unit", "alt_text"}
    with pytest.raises(ValidationError):
        _line_spec(y2_series={"Margin %": [0.1, 0.2, 0.3]})


@pytest.mark.parametrize("alt", ["", "   "])
def test_empty_alt_text_raises(alt):
    with pytest.raises(ValueError, match="alt_text"):
        make_chart(_line_spec(alt_text=alt))


def test_alt_text_is_embedded_in_svg():
    spec = _line_spec()
    assert f"<desc>{spec.alt_text}</desc>" in _svg(spec)


def test_no_remote_references_or_web_fonts():
    svg = _svg(_line_spec())
    refs = re.findall(r'(?:href|src)="([^"]*)"', svg)
    assert all(ref.startswith("#") for ref in refs)
    assert "@font-face" not in svg
    assert "url(http" not in svg


def test_series_use_categorical_colors_in_fixed_order():
    svg = _svg(_line_spec())
    assert f"stroke: {CATEGORICAL_LIGHT[0]}" in svg
    assert f"stroke: {CATEGORICAL_LIGHT[1]}" in svg


def test_more_than_eight_series_raises():
    series = {f"S{i}": [1.0, 2.0, 3.0] for i in range(9)}
    with pytest.raises(ValueError, match="8"):
        make_chart(_line_spec(series=series))


def test_bar_chart_renders_well_formed_svg():
    svg = make_chart(_line_spec(kind="bar"))
    ET.fromstring(svg)
    assert b"<image" not in svg


def test_validator_reports_false_when_node_missing(monkeypatch):
    monkeypatch.setattr("weekly_analyst.report.charts.shutil.which", lambda name: None)
    assert validate_palette(list(CATEGORICAL_LIGHT)) is False


def test_reference_palette_passes_validator():
    if shutil.which("node") is None or not os.environ.get("DATAVIZ_VALIDATOR"):
        pytest.skip(NODE_MISSING)
    assert validate_palette(list(CATEGORICAL_LIGHT)) is True


def test_near_identical_colors_fail_validator():
    if shutil.which("node") is None or not os.environ.get("DATAVIZ_VALIDATOR"):
        pytest.skip(NODE_MISSING)
    assert validate_palette(["#2a78d6", "#2a79d6", "#2a78d7"]) is False
