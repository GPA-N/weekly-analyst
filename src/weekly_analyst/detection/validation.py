"""Validation detector: count of validation findings per metric, and the metrics validation halts (spec 6.6)."""

import polars as pl

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import MetricsFile
from weekly_analyst.data.validate import validate
from weekly_analyst.detection.base import Detector, Key


def _latest_period_rows(history: pl.DataFrame, date_column: str) -> int | None:
    """Row count of the most recent history week, used as the prior snapshot's row count."""
    if history.height == 0:
        return None
    periods = history.select(
        pl.col(date_column).cast(pl.Date).dt.strftime("%G-W%V").alias("period")
    ).get_column("period")
    return int((periods == periods.max()).sum())


def _validation_scores(
    history: pl.DataFrame, current: pl.DataFrame, metrics: MetricsFile
) -> tuple[dict[str, float], set[str]]:
    """Metric-level count of validation findings (dataset-level ones count for every metric)."""
    prior_schema = {name: str(dtype) for name, dtype in history.schema.items()}
    result = validate(
        current, metrics, prior_schema, _latest_period_rows(history, metrics.date_column)
    )
    dataset_level = sum(1 for f in result.findings if f.metric is None)
    scores = dict.fromkeys((m.name for m in metrics.metrics), float(dataset_level))
    for finding in result.findings:
        if finding.metric is not None:
            scores[finding.metric] += 1.0
    return scores, result.halted_metrics


class ValidationDetector(Detector):
    """Owns the validation-finding count per metric. A metric validation halts is not scored by the blend."""

    name = "validation"

    def score(
        self, history: pl.DataFrame, current: pl.DataFrame, metrics: MetricsFile, settings: Settings
    ) -> dict[Key, float]:
        return self.scan(history, current, metrics)[0]

    def scan(
        self, history: pl.DataFrame, current: pl.DataFrame, metrics: MetricsFile
    ) -> tuple[dict[str, float], set[str]]:
        """Scores plus the set of metric names validation halts."""
        return _validation_scores(history, current, metrics)
