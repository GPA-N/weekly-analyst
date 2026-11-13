"""Deterministic report checks on the rendered PDF text (spec 6.3)."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from weekly_analyst.analysis.analyst import wording_check
from weekly_analyst.core.text import numerals
from weekly_analyst.data.metrics_file import MetricDef, MetricsFile
from weekly_analyst.data.recompute import matches_displayed, recompute


@dataclass
class ShownNumber:
    text: str
    metric: str
    segment: dict[str, str] | None
    period: str
    prev_period: str | None = None


@dataclass
class Failure:
    kind: Literal["wrong_number", "no_metric", "wording", "unlisted_number"]
    detail: str


def _recomputed(snapshot_path: Path, metric: MetricDef, shown: ShownNumber, date_column: str) -> float | None:
    """The value a shown number must match: its level, or the signed change from prev_period to period."""
    now = recompute(snapshot_path, metric, shown.segment, shown.period, date_column)
    if shown.prev_period is None:
        return now
    before = recompute(snapshot_path, metric, shown.segment, shown.prev_period, date_column)
    if now is None or before is None:
        return None
    return now - before


def check_report(
    pdf_text: str,
    shown_numbers: list[ShownNumber],
    metrics: MetricsFile,
    snapshot_path: Path,
    exempt: Sequence[str] = (),
    model_text: str = "",
) -> list[Failure]:
    """Check each shown number, the wording, and every numeral on the page.

    exempt holds the numerals the renderer writes itself (dates, page numbers, indices). model_text is
    the model-written text: every numeral in it must be a shown number, with no exemption.
    """
    failures: list[Failure] = []
    by_name = {m.name: m for m in metrics.metrics}

    for shown in shown_numbers:
        metric = by_name.get(shown.metric)
        if metric is None:
            failures.append(Failure("no_metric", f"{shown.text!r} names no known metric {shown.metric!r}"))
            continue
        try:
            value = _recomputed(snapshot_path, metric, shown, metrics.date_column)
            matched = value is not None and matches_displayed(value, shown.text)
        except ValueError as err:
            failures.append(Failure("wrong_number", f"{shown.text!r} cannot be verified: {err}"))
            continue
        if not matched:
            failures.append(
                Failure("wrong_number", f"{shown.text!r} not confirmed by recompute (value {value!r})")
            )

    for flag in wording_check(pdf_text):
        failures.append(Failure("wording", flag))

    listed = {shown.text for shown in shown_numbers}
    model_tokens = set(numerals(model_text))
    exempt_tokens = set(exempt)
    for token in dict.fromkeys(numerals(pdf_text)):
        # A numeral from model text is judged by the stricter model_text rule below, so it is not reported twice.
        if token not in listed and token not in exempt_tokens and token not in model_tokens:
            failures.append(Failure("unlisted_number", token))

    for token in dict.fromkeys(numerals(model_text)):
        if token not in listed:
            failures.append(Failure("unlisted_number", token))

    return failures
