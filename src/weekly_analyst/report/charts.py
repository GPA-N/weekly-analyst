"""Charts: vector SVG line and bar charts with fixed categorical colors (dataviz rules)."""

import io
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape

from matplotlib import rc_context
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

# Validated light-mode categorical palette, fixed order (dataviz references/palette.md).
CATEGORICAL_LIGHT: tuple[str, ...] = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
SURFACE_LIGHT = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# The palette validator is a node script from the dataviz skill. Its location is the caller's choice:
# set DATAVIZ_VALIDATOR to the path of validate_palette.js. Without it the check reports False.
VALIDATOR_ENV = "DATAVIZ_VALIDATOR"

_SVG_OPEN = re.compile(r"<svg\b[^>]*>")


class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["line", "bar"]
    title: str
    x: list
    series: dict[str, list[float]]
    unit: str
    alt_text: str


def validate_palette(colors: list[str]) -> bool:
    node = shutil.which("node")
    script = os.environ.get(VALIDATOR_ENV)
    if node is None or not script or not Path(script).is_file():
        return False
    result = subprocess.run(
        [node, script, ",".join(colors), "--mode", "light"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.returncode == 0


def _with_alt_text(svg: str, title: str, alt_text: str) -> str:
    alt = f"<title>{escape(title)}</title><desc>{escape(alt_text)}</desc>"
    return _SVG_OPEN.sub(lambda m: m.group(0) + alt, svg, count=1)


def make_chart(spec: ChartSpec) -> bytes:
    if not spec.alt_text.strip():
        raise ValueError("alt_text must not be empty")
    if len(spec.series) > len(CATEGORICAL_LIGHT):
        raise ValueError(
            f"at most {len(CATEGORICAL_LIGHT)} series: the palette has {len(CATEGORICAL_LIGHT)} slots; "
            "fold extra series into Other or facet"
        )

    fig = Figure(figsize=(6.4, 3.6), facecolor=SURFACE_LIGHT)
    ax = fig.add_subplot()
    n = len(spec.series)
    for i, (name, values) in enumerate(spec.series.items()):
        color = CATEGORICAL_LIGHT[i]
        if spec.kind == "line":
            ax.plot(spec.x, values, color=color, linewidth=2, marker="o", markersize=8, label=name)
        else:
            width = 0.8 / n
            offset = (i - (n - 1) / 2) * width
            positions = [j + offset for j in range(len(spec.x))]
            ax.bar(
                positions, values, width=width, color=color,
                edgecolor=SURFACE_LIGHT, linewidth=2, label=name,
            )
    if spec.kind == "bar":
        ax.set_xticks(range(len(spec.x)), labels=[str(v) for v in spec.x])

    ax.set_title(spec.title, loc="left", color=TEXT_PRIMARY)
    ax.set_ylabel(spec.unit, color=TEXT_SECONDARY)
    ax.set_facecolor(SURFACE_LIGHT)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    if n >= 2:
        ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()

    buf = io.BytesIO()
    with rc_context({"svg.fonttype": "none"}):
        fig.savefig(buf, format="svg", facecolor=SURFACE_LIGHT)
    return _with_alt_text(buf.getvalue().decode("utf-8"), spec.title, spec.alt_text).encode("utf-8")
